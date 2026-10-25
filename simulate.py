"""Прогон шасси через препятствие в MuJoCo и поиск предельного размера.

Ровер разгоняется до заданной скорости (свой регулятор скорости у каждого
колеса или одинаковый момент всех колёс); момент каждого мотор-колеса
ограничен пиковым. Прохождение считается успешным, когда все
колёса оказались за препятствием, а рычаги и стойки ни разу не коснулись
грунта. Иначе возвращается причина неудачи.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import asin, atan2, degrees
from typing import Literal

import mujoco
import numpy as np

from model import FAR, SIDES, WHEELS, WHEEL_LABELS, SIDE_LABELS, Box, Design, Platform, build_mjcf, neutral_points

ObstacleKind = Literal["step_up", "step_down", "ditch"]

KIND_LABELS: dict[str, str] = {
    "step_up": "подъём на ступень",
    "step_down": "съезд со ступени",
    "ditch": "ров",
}
DIRECTION_LABELS = {1: "одиночное колесо вперёд", -1: "bogie вперёд"}

APPROACH_GAP_MM = 120.0
FINISH_MARGIN_MM = 60.0
HANG_MARGIN_MM = 800.0
SETTLE_S = 0.6
RAMP_S = 0.8
STUCK_WINDOW_S = 3.0
STUCK_PROGRESS_MM = 15.0
TIP_PITCH_DEG = 70.0
TIP_ROLL_DEG = 40.0
FRAME_DT = 0.05
CHECK_STEPS = 10

POINT_NAMES = (
    "main", "bogie", "top_front", "top_middle", "top_rear",
    "wheel_front", "wheel_middle", "wheel_rear",
)


@dataclass(frozen=True)
class Obstacle:
    kind: ObstacleKind
    size_mm: float
    direction: int = 1
    one_side: bool = False
    ditch_depth_mm: float = 500.0

    @property
    def label(self) -> str:
        unit = "ширина" if self.kind == "ditch" else "высота"
        side = ", под левым бортом" if self.one_side else ""
        return (
            f"{KIND_LABELS[self.kind]}, {unit} {self.size_mm:.0f} мм, "
            f"{DIRECTION_LABELS[self.direction]}{side}"
        )


@dataclass
class Frame:
    t: float
    points: dict[str, dict[str, tuple[float, float, float]]]
    com: tuple[float, float, float]
    pitch_deg: float
    roll_deg: float
    rocker_deg: dict[str, float]
    bogie_deg: dict[str, float]
    normal_n: dict[str, float]
    torque_nm: dict[str, float]
    forces: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]


@dataclass
class SimResult:
    obstacle: Obstacle
    success: bool
    reason: str
    terrain: list[Box]
    floor_z_mm: float
    duration_s: float
    max_torque_ratio: float = 0.0
    max_pitch_deg: float = 0.0
    max_roll_deg: float = 0.0
    max_rocker_deg: float = 0.0
    max_bogie_deg: float = 0.0
    notes: list[str] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)


@dataclass(frozen=True)
class LimitResult:
    obstacle: Obstacle
    value_mm: float
    reason: str
    upper_bound_mm: float
    runs: int


def _wheel_s(design: Design, platform: Platform, direction: int) -> list[float]:
    points = neutral_points(design, platform)
    return [direction * x for x, _ in points.wheels.values()]


def build_terrain(design: Design, platform: Platform, obstacle: Obstacle) -> tuple[list[Box], float, float, float]:
    """Возвращает блоки, высоту пола, высоту старта и координату финиша (в s)."""
    direction = obstacle.direction
    radius = platform.wheel_radius_mm
    wheel_s = _wheel_s(design, platform, direction)
    start = max(wheel_s) + radius + APPROACH_GAP_MM
    size = obstacle.size_mm
    full = (-FAR, FAR)
    left = (0.0, FAR)
    right = (-FAR, 0.0)

    def box(s0: float, s1: float, ys: tuple[float, float], z_top: float) -> Box:
        x0, x1 = sorted((direction * s0, direction * s1))
        return Box(x0, x1, ys[0], ys[1], z_top)

    boxes: list[Box] = []
    floor = 0.0
    start_z = 0.0
    finish = start + radius + FINISH_MARGIN_MM
    if obstacle.kind == "step_up":
        boxes.append(box(start, FAR, left if obstacle.one_side else full, size))
    elif obstacle.kind == "step_down":
        start_z = size
        boxes.append(box(-FAR, start, full, size))
        if obstacle.one_side:
            boxes.append(box(start, FAR, right, size))
    else:
        floor = -obstacle.ditch_depth_mm
        boxes.append(box(-FAR, start, full, 0.0))
        boxes.append(box(start + size, FAR, full, 0.0))
        if obstacle.one_side:
            boxes.append(box(start, start + size, right, 0.0))
        finish = start + size + radius + FINISH_MARGIN_MM
    return boxes, floor, start_z, finish


def make_model(design: Design, platform: Platform, obstacle: Obstacle, timestep: float = 0.001):
    boxes, floor, start_z, finish = build_terrain(design, platform, obstacle)
    xml = build_mjcf(design, platform, boxes, start_z_mm=start_z, floor_z_mm=floor, timestep=timestep)
    model = mujoco.MjModel.from_xml_string(xml)
    return model, boxes, floor, finish


class _Ids:
    def __init__(self, model: mujoco.MjModel):
        def body(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

        def geom(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

        def joint_qpos(name):
            return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]

        def joint_dof(name):
            return model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]

        self.chassis = body("chassis")
        self.ground = {
            index for index in range(model.ngeom)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith(("ground", "floor"))
        }
        self.tire = {geom(f"tire_{w}_{s}"): (s, w) for s in SIDES for w in WHEELS}
        self.arm = {}
        for index in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
            if name.startswith(("arm_", "strut_")):
                self.arm[index] = name
        self.rocker = {s: joint_qpos(f"rocker_{s}") for s in SIDES}
        self.bogie = {s: joint_qpos(f"bogie_{s}") for s in SIDES}
        self.spin_dof = {(s, w): joint_dof(f"spin_{w}_{s}") for s in SIDES for w in WHEELS}
        self.actuator = {
            (s, w): mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{w}_{s}")
            for s in SIDES for w in WHEELS
        }
        self.body = {
            (s, "main"): body(f"rocker_{s}") for s in SIDES
        } | {
            (s, "bogie"): body(f"bogie_{s}") for s in SIDES
        } | {
            (s, f"wheel_{w}"): body(f"wheel_{w}_{s}") for s in SIDES for w in WHEELS
        }
        self.site = {
            (s, f"top_{w}"): mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"top_{w}_{s}")
            for s in SIDES for w in WHEELS
        }


ARM_LABELS = {
    "arm_rocker_front": "труба rocker к переднему колесу",
    "arm_rocker_bogie": "труба rocker к bogie",
    "arm_bogie_middle": "труба bogie к среднему колесу",
    "arm_bogie_rear": "труба bogie к заднему колесу",
    "strut_front": "стойка переднего колеса",
    "strut_middle": "стойка среднего колеса",
    "strut_rear": "стойка заднего колеса",
}


def _arm_label(name: str) -> str:
    base, side = name.rsplit("_", 1)
    return f"{ARM_LABELS.get(base, base)} ({SIDE_LABELS[side]} борт)"


def _pitch_roll(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat
    # Тангаж положителен при подъёме передней части (нос вверх).
    pitch = -degrees(asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
    roll = degrees(atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    return pitch, roll


def _mm(vector: np.ndarray) -> tuple[float, float, float]:
    return (float(vector[0] * 1000.0), float(vector[1] * 1000.0), float(vector[2] * 1000.0))


def _record(model, data, ids: _Ids, wheel_contacts) -> Frame:
    points: dict[str, dict[str, tuple[float, float, float]]] = {}
    for side in SIDES:
        side_points = {}
        for name in POINT_NAMES:
            if name.startswith("top_"):
                side_points[name] = _mm(data.site_xpos[ids.site[(side, name)]])
            else:
                side_points[name] = _mm(data.xpos[ids.body[(side, name)]])
        points[side] = side_points
    pitch, roll = _pitch_roll(data.xquat[ids.chassis])
    normal = {f"{w}_{s}": 0.0 for s in SIDES for w in WHEELS}
    forces = []
    for side, wheel, position, force, normal_force in wheel_contacts:
        normal[f"{wheel}_{side}"] += normal_force
        forces.append((f"{wheel}_{side}", _mm(position), tuple(float(v) for v in force)))
    return Frame(
        t=float(data.time),
        points=points,
        com=_mm(data.xipos[ids.chassis]),
        pitch_deg=pitch,
        roll_deg=roll,
        rocker_deg={s: degrees(float(data.qpos[ids.rocker[s]])) for s in SIDES},
        bogie_deg={s: degrees(float(data.qpos[ids.bogie[s]])) for s in SIDES},
        normal_n=normal,
        torque_nm={
            f"{w}_{s}": float(data.actuator_force[ids.actuator[(s, w)]]) for s in SIDES for w in WHEELS
        },
        forces=forces,
    )


def _wheel_contacts(model, data, ids: _Ids):
    result = []
    force6 = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = contact.geom1, contact.geom2
        tire = ids.tire.get(g1) or ids.tire.get(g2)
        if tire is None or not (g1 in ids.ground or g2 in ids.ground):
            continue
        mujoco.mj_contactForce(model, data, index, force6)
        frame = contact.frame.reshape(3, 3)
        world = frame.T @ force6[:3]
        # Сила, действующая на колесо: нормаль контакта направлена от geom1 к geom2.
        if g1 in ids.tire:
            world = -world
        result.append((tire[0], tire[1], contact.pos.copy(), world, float(force6[0])))
    return result


def run(
    design: Design,
    platform: Platform,
    obstacle: Obstacle,
    record: bool = True,
    timestep: float = 0.001,
) -> SimResult:
    model, boxes, floor, finish_s = make_model(design, platform, obstacle, timestep)
    data = mujoco.MjData(model)
    ids = _Ids(model)
    direction = obstacle.direction
    omega = platform.speed_m_s / (platform.wheel_radius_mm / 1000.0)
    torque_limit = platform.wheel_torque_nm

    def result(success: bool, reason: str) -> SimResult:
        return SimResult(
            obstacle=obstacle,
            success=success,
            reason=reason,
            terrain=boxes,
            floor_z_mm=floor,
            duration_s=float(data.time),
            max_torque_ratio=max_torque / torque_limit,
            max_pitch_deg=max_pitch,
            max_roll_deg=max_roll,
            max_rocker_deg=max_rocker,
            max_bogie_deg=max_bogie,
            notes=notes,
            frames=frames,
        )

    frames: list[Frame] = []
    notes: list[str] = []
    max_torque = max_pitch = max_roll = max_rocker = max_bogie = 0.0
    progress_history: list[tuple[float, float]] = []
    wheel_body = [ids.body[(s, f"wheel_{w}")] for s in SIDES for w in WHEELS]
    distance = finish_s - min(_wheel_s(design, platform, direction))
    time_limit = SETTLE_S + 2.5 * distance / 1000.0 / platform.speed_m_s + 6.0
    next_frame = 0.0
    limit_hit = set()
    controller = None if platform.control == "speed" else _TorqueController(model, ids, platform, direction)

    mujoco.mj_forward(model, data)
    while data.time < time_limit:
        driving = data.time >= SETTLE_S
        ramp = min(1.0, max(0.0, (data.time - SETTLE_S) / RAMP_S))
        if controller is None:
            data.ctrl[:] = direction * omega * ramp
            mujoco.mj_step(model, data, nstep=CHECK_STEPS)
        else:
            for _ in range(CHECK_STEPS):
                controller.update(data, ramp)
                mujoco.mj_step(model, data)

        if not np.all(np.isfinite(data.qpos)):
            return result(False, "численная неустойчивость расчёта")

        touching = set()
        for index in range(data.ncon):
            contact = data.contact[index]
            g1, g2 = contact.geom1, contact.geom2
            if g1 in ids.tire and g2 in ids.ground:
                touching.add(g1)
            elif g2 in ids.tire and g1 in ids.ground:
                touching.add(g2)
            if g1 in ids.arm or g2 in ids.arm:
                arm = ids.arm.get(g1) or ids.arm.get(g2)
                return result(False, f"задевает препятствие: {_arm_label(arm)}")
            if g1 in ids.tire and g2 in ids.tire:
                return result(False, "колёса сталкиваются друг с другом")

        pitch, roll = _pitch_roll(data.xquat[ids.chassis])
        if data.time >= SETTLE_S + RAMP_S + 0.3:
            max_torque = max(max_torque, float(np.max(np.abs(data.actuator_force))))
            max_pitch = max(max_pitch, abs(pitch))
            max_roll = max(max_roll, abs(roll))
            for side in SIDES:
                rocker = abs(degrees(float(data.qpos[ids.rocker[side]])))
                bogie = abs(degrees(float(data.qpos[ids.bogie[side]])))
                max_rocker = max(max_rocker, rocker)
                max_bogie = max(max_bogie, bogie)
                if rocker > design.rocker_limit_deg - 0.5:
                    limit_hit.add("rocker")
                if bogie > design.bogie_limit_deg - 0.5:
                    limit_hit.add("bogie")

        if record and data.time >= next_frame:
            frames.append(_record(model, data, ids, _wheel_contacts(model, data, ids)))
            next_frame += FRAME_DT

        if abs(pitch) > TIP_PITCH_DEG or abs(roll) > TIP_ROLL_DEG:
            return result(False, "опрокидывание")

        wheel_s = [direction * data.xpos[body][0] * 1000.0 for body in wheel_body]
        if min(wheel_s) >= finish_s and len(touching) == len(ids.tire):
            for name in sorted(limit_hit):
                notes.append(f"шарнир {name} доходил до упора")
            if record:
                frames.append(_record(model, data, ids, _wheel_contacts(model, data, ids)))
            return result(True, "")
        if min(wheel_s) >= finish_s + HANG_MARGIN_MM:
            return result(False, "колёса висят в воздухе: подвеска не достаёт до опоры")

        if driving:
            chassis_s = direction * data.xpos[ids.chassis][0] * 1000.0
            progress_history.append((float(data.time), chassis_s))
            cutoff = data.time - STUCK_WINDOW_S
            while len(progress_history) > 1 and progress_history[1][0] <= cutoff:
                progress_history.pop(0)
            if progress_history[0][0] <= cutoff + 1e-9 and data.time > SETTLE_S + STUCK_WINDOW_S:
                if chassis_s - progress_history[0][1] < STUCK_PROGRESS_MM:
                    return result(False, _stuck_reason(model, data, ids, omega, torque_limit))

    return result(False, "не успел пройти за отведённое время")


class _TorqueController:
    """Одинаковый момент всех колёс от ПИ-регулятора скорости корпуса.

    Обновляется на каждом шаге расчёта. Колесо, потерявшее опору, плавно
    теряет момент между 1,2 и 2 заданными скоростями, чтобы не
    раскручивалось без нагрузки.
    """

    def __init__(self, model, ids: _Ids, platform: Platform, direction: int):
        radius = platform.wheel_radius_mm / 1000.0
        self.direction = direction
        self.torque = platform.wheel_torque_nm
        self.speed = platform.speed_m_s
        self.omega = self.speed / radius
        self.dt = model.opt.timestep
        self.kp = platform.total_mass_kg * radius / (6.0 * 0.1)
        self.ki = self.kp / 0.3
        self.integral = 0.0
        keys = list(ids.actuator)
        self.actuators = np.asarray([ids.actuator[key] for key in keys])
        self.dofs = np.asarray([ids.spin_dof[key] for key in keys])

    def update(self, data, ramp: float) -> None:
        torque = self.torque
        error = self.speed * ramp - self.direction * float(data.qvel[0])
        self.integral = min(torque, max(-torque, self.integral + self.ki * error * self.dt))
        common = min(torque, max(-torque, self.kp * error + self.integral))
        spin = self.direction * data.qvel[self.dofs]
        fade = np.clip((2.0 * self.omega - np.abs(spin)) / (0.8 * self.omega), 0.0, 1.0)
        # Ограничение оборотов снимает только разгоняющий момент, тормозной
        # момент колесо получает всегда.
        driving = np.sign(spin) == np.sign(common)
        data.ctrl[self.actuators] = self.direction * common * np.where(driving, fade, 1.0)


def _stuck_reason(model, data, ids: _Ids, omega: float, torque_limit: float) -> str:
    """Классифицирует застревание по нагруженным колёсам."""
    loads: dict[tuple[str, str], float] = {}
    for side, wheel, _, _, normal_force in _wheel_contacts(model, data, ids):
        loads[(side, wheel)] = loads.get((side, wheel), 0.0) + normal_force
    total = sum(loads.values()) or 1.0
    stalled, slipping = [], []
    for key, load in loads.items():
        if load < 0.03 * total:
            continue
        spin = abs(data.qvel[ids.spin_dof[key]]) / omega
        torque = abs(data.actuator_force[ids.actuator[key]]) / torque_limit
        label = f"{WHEEL_LABELS[key[1]]} ({SIDE_LABELS[key[0]]})"
        if spin > 0.3:
            slipping.append(label)
        elif torque > 0.95:
            stalled.append(label)
    if stalled and len(slipping) <= len(stalled):
        return "не хватает момента колёс: " + ", ".join(stalled)
    if slipping:
        return "пробуксовка, не хватает сцепления: " + ", ".join(slipping)
    return "застревание"


def _boundary(attempt, good: float, bad: float, tolerance: float, guess: float | None = None):
    """Сужает отрезок между проходимым (good) и непроходимым (bad) значением.

    Концы отрезка заранее не проверяются. С подсказкой guess поиск сначала
    проверяет её и шагает от неё с удвоением шага, пока не найдёт границу, —
    при точной подсказке это 3–4 прогона вместо 6–8 при делении пополам.
    Возвращает (good, bad, причина последней неудачи, прогоны, проверен ли good).
    """
    toward_bad = 1.0 if bad > good else -1.0
    runs = 0
    reason = ""
    good_tested = False

    def inside(value: float) -> bool:
        return min(good, bad) < value < max(good, bad)

    if guess is not None and inside(guess):
        step = 4.0 * tolerance
        value = guess
        runs += 1
        trial = attempt(value)
        moving_to_bad = trial.success
        if trial.success:
            good, good_tested = value, True
        else:
            bad, reason = value, trial.reason
        while abs(bad - good) > tolerance:
            value = value + (toward_bad if moving_to_bad else -toward_bad) * step
            if not inside(value):
                break
            runs += 1
            trial = attempt(value)
            if trial.success:
                good, good_tested = value, True
                if not moving_to_bad:
                    break
            else:
                bad, reason = value, trial.reason
                if moving_to_bad:
                    break
            step *= 2.0
    while abs(bad - good) > tolerance:
        middle = (good + bad) / 2.0
        runs += 1
        trial = attempt(middle)
        if trial.success:
            good, good_tested = middle, True
        else:
            bad, reason = middle, trial.reason
    return good, bad, reason, runs, good_tested


def find_limit(
    design: Design,
    platform: Platform,
    kind: ObstacleKind,
    direction: int,
    one_side: bool = False,
    upper_mm: float = 600.0,
    tolerance_mm: float = 10.0,
    ditch_depth_mm: float = 500.0,
    timestep: float = 0.001,
    guess_mm: float | None = None,
) -> LimitResult:
    """Наибольшее препятствие, которое шасси проходит (поиск границы по размеру)."""

    def attempt(size: float) -> SimResult:
        obstacle = Obstacle(kind, size, direction, one_side, ditch_depth_mm)
        return run(design, platform, obstacle, record=False, timestep=timestep)

    def limit(value: float, reason: str, runs: int) -> LimitResult:
        return LimitResult(Obstacle(kind, value, direction, one_side, ditch_depth_mm), value, reason, upper_mm, runs)

    top = None
    if guess_mm is None:
        top = attempt(upper_mm)
        if top.success:
            return limit(upper_mm, "граница поиска", 1)
    low, high, reason, runs, _ = _boundary(attempt, 0.0, upper_mm, tolerance_mm, guess_mm)
    if top is not None:
        return limit(low, reason or top.reason, runs + 1)
    if high >= upper_mm:
        # С подсказкой верхняя граница ещё не проверялась.
        top = attempt(upper_mm)
        if top.success:
            return limit(upper_mm, "граница поиска", runs + 1)
        runs += 1
        reason = reason or top.reason
    return limit(low, reason, runs)


@dataclass(frozen=True)
class TorqueResult:
    """Наименьший пиковый момент колеса, при котором препятствие проходится."""

    obstacle: Obstacle
    torque_nm: float
    reason: str
    upper_nm: float
    runs: int

    @property
    def passable(self) -> bool:
        return self.torque_nm <= self.upper_nm


def required_torque(
    design: Design,
    platform: Platform,
    obstacle: Obstacle,
    upper_nm: float = 60.0,
    tolerance_nm: float = 0.5,
    timestep: float = 0.001,
    guess_nm: float | None = None,
) -> TorqueResult:
    """Поиск границы по моменту: ниже неё не хватает момента, выше — проходит.

    Если препятствие не проходится и при upper_nm (задевает рычагом,
    опрокидывается), момент равен бесконечности, а reason объясняет почему.
    ``reason`` у проходимого препятствия — что мешает при чуть меньшем моменте.
    """

    def attempt(torque: float) -> SimResult:
        return run(design, replace(platform, wheel_torque_nm=torque), obstacle, record=False, timestep=timestep)

    good, _, reason, runs, tested = _boundary(attempt, upper_nm, 0.0, tolerance_nm, guess_nm)
    if not tested:
        runs += 1
        top = attempt(upper_nm)
        if not top.success:
            return TorqueResult(obstacle, float("inf"), top.reason, upper_nm, runs)
    return TorqueResult(obstacle, good, reason, upper_nm, runs)
