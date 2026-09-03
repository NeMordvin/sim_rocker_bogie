"""Квазистатический расчёт шестиколёсного rocker-bogie.

Две стороны подвески собираются из рассчитанных 2D-поз. Для каждого кадра
линейным программированием находятся нормальные и касательные реакции во всех
активных контактах. Ограничения описывают пассивный шарнир bogie, классический
дифференциал, сцепление и предельный момент шести одинаковых мотор-колёс.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import acos, atan2, cos, degrees, hypot, radians, sin
from typing import Literal

import numpy as np
from scipy.optimize import linprog

from geometry import (
    ChassisDesign,
    ChassisFrame,
    DitchTerrain,
    Point,
    Pose,
    StepTerrain,
    Terrain,
    TraversalResult,
    WHEEL_NAMES,
    calculate_frame,
    neutral_pose,
)


Scenario = Literal["full_width", "one_side"]

SCENARIO_LABELS: dict[Scenario, str] = {
    "full_width": "препятствие по всей ширине",
    "one_side": "препятствие только под левой стороной",
}


@dataclass(frozen=True)
class PhysicalParameters:
    """Изменяемые исходные данные силового расчёта."""

    mass_kg: float = 120.0
    wheel_torque_nm: float = 20.0
    track_width_mm: float = 600.0
    com_lateral_mm: float = 0.0
    friction_coefficient: float = 0.70
    rolling_resistance_coefficient: float = 0.03
    dynamic_factor: float = 1.20
    gravity_m_s2: float = 9.81
    minimum_support_margin_mm: float = 10.0


@dataclass(frozen=True)
class ContactLoad:
    side: str
    wheel: str
    mode: str
    point_mm: tuple[float, float, float]
    normal: tuple[float, float, float]
    tangent: tuple[float, float, float]
    normal_force_n: float
    net_tangent_force_n: float
    lateral_force_n: float
    motor_force_n: float
    motor_torque_nm: float
    friction_utilization: float
    torque_utilization: float


@dataclass(frozen=True)
class StaticFrameResult:
    feasible: bool
    scenario: Scenario
    pose_index: int
    reason: str
    body_pitch_deg: float
    body_roll_deg: float
    center_of_mass_mm: tuple[float, float, float]
    design_weight_n: float
    longitudinal_margin_mm: float
    lateral_margin_mm: float
    support_polygon_margin_mm: float
    tipover_angle_deg: float
    maximum_torque_utilization: float
    maximum_friction_utilization: float
    force_residual_n: float
    moment_residual_nm: float
    contacts: tuple[ContactLoad, ...]
    equations: tuple[str, ...]
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class TraversalStatics:
    success: bool
    scenario: Scenario
    frames: tuple[StaticFrameResult, ...]
    failure_reason: str = ""

    @property
    def maximum_torque_utilization(self) -> float:
        return max(
            (frame.maximum_torque_utilization for frame in self.frames),
            default=float("inf"),
        )

    @property
    def minimum_support_margin_mm(self) -> float:
        return min(
            (frame.support_polygon_margin_mm for frame in self.frames),
            default=float("-inf"),
        )


@dataclass(frozen=True)
class _ContactGeometry:
    side: str
    wheel: str
    mode: str
    point_m: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray


def validate_physical(parameters: PhysicalParameters) -> tuple[str, ...]:
    errors: list[str] = []
    positive = {
        "масса": parameters.mass_kg,
        "момент одного колеса": parameters.wheel_torque_nm,
        "колея": parameters.track_width_mm,
        "коэффициент динамики": parameters.dynamic_factor,
        "ускорение свободного падения": parameters.gravity_m_s2,
    }
    for name, value in positive.items():
        if value <= 0.0:
            errors.append(f"Параметр «{name}» должен быть больше нуля.")
    if parameters.friction_coefficient <= 0.0:
        errors.append("Коэффициент сцепления должен быть больше нуля.")
    if parameters.rolling_resistance_coefficient < 0.0:
        errors.append("Коэффициент сопротивления качению не может быть отрицательным.")
    if parameters.minimum_support_margin_mm < 0.0:
        errors.append("Минимальный запас устойчивости не может быть отрицательным.")
    if abs(parameters.com_lateral_mm) >= parameters.track_width_mm / 2.0:
        errors.append("Поперечная координата ЦМ должна находиться внутри колеи.")
    return tuple(errors)


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("Невозможно нормировать нулевой вектор.")
    return vector / length


def _flat_pose_at_x(design: ChassisDesign, main_x_mm: float) -> Pose:
    return replace(neutral_pose(design), main_pivot_x_mm=main_x_mm)


def _wheel_contact_2d(
    design: ChassisDesign,
    frame: ChassisFrame,
    pose: Pose,
    terrain: Terrain,
    wheel_index: int,
) -> tuple[str, Point, np.ndarray, np.ndarray] | None:
    name = WHEEL_NAMES[wheel_index]
    wheel = frame.wheels[wheel_index]
    radius = design.wheel_radius_mm

    if isinstance(terrain, StepTerrain):
        if name in pose.wall_contacts:
            return (
                "стенка",
                Point(0.0, wheel.y),
                np.asarray([-1.0, 0.0, 0.0]),
                np.asarray([0.0, 0.0, 1.0]),
            )
        if name not in pose.contacts:
            return None

        # В момент первого касания кромки колесо ещё опирается на нижнюю
        # площадку. Оставляем нижний контакт до начала подъёма центра; иначе
        # замена двух одновременных контактов одним почти горизонтальным
        # нормальным вектором создаёт искусственный скачок требуемых сил.
        if wheel.x < 0.0 and wheel.y <= radius + 1.0:
            return (
                "нижняя площадка",
                Point(wheel.x, 0.0),
                np.asarray([0.0, 0.0, 1.0]),
                np.asarray([1.0, 0.0, 0.0]),
            )

        corner_distance = hypot(wheel.x, wheel.y - terrain.height_mm)
        on_corner = (
            terrain.height_mm > 1e-9
            and -radius - 1.0 <= wheel.x <= 1.0
            and abs(corner_distance - radius) <= 1.5
            and wheel.y >= terrain.height_mm - 1.0
        )
        if on_corner:
            normal_2d = _normalize(
                np.asarray([wheel.x, 0.0, wheel.y - terrain.height_mm])
            )
            tangent = _normalize(
                np.asarray([normal_2d[2], 0.0, -normal_2d[0]])
            )
            return (
                "верхняя кромка",
                Point(0.0, terrain.height_mm),
                normal_2d,
                tangent,
            )

        support_x = float(
            terrain.support_x(np.asarray([wheel.x]), radius)[0]
        )
        support_z = terrain.height_mm if support_x >= -1e-8 else 0.0
        mode = "верхняя площадка" if support_z > 0.0 else "нижняя площадка"
        return (
            mode,
            Point(support_x, support_z),
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([1.0, 0.0, 0.0]),
        )

    if name not in pose.contacts:
        return None
    assert isinstance(terrain, DitchTerrain)
    support_x = float(terrain.support_x(np.asarray([wheel.x]), radius)[0])
    at_edge = abs(support_x) <= 1e-8 or abs(support_x - terrain.width_mm) <= 1e-8
    if at_edge and 0.0 < wheel.x < terrain.width_mm:
        vector = np.asarray([wheel.x - support_x, 0.0, wheel.y])
        normal = _normalize(vector)
        tangent = _normalize(np.asarray([normal[2], 0.0, -normal[0]]))
        mode = "левая кромка рва" if support_x <= 0.0 else "правая кромка рва"
        return mode, Point(support_x, 0.0), normal, tangent
    return (
        "берег",
        Point(support_x, 0.0),
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([1.0, 0.0, 0.0]),
    )


def _body_state(
    design: ChassisDesign,
    physical: PhysicalParameters,
    left_pose: Pose,
    right_pose: Pose,
    direction: int,
) -> tuple[float, float, np.ndarray]:
    pitch_deg = (left_pose.body_pitch_deg + right_pose.body_pitch_deg) / 2.0
    height_difference = right_pose.main_pivot_y_mm - left_pose.main_pivot_y_mm
    roll_deg = degrees(atan2(height_difference, physical.track_width_mm))

    main_center = np.asarray(
        [
            (left_pose.main_pivot_x_mm + right_pose.main_pivot_x_mm) / 2000.0,
            0.0,
            (left_pose.main_pivot_y_mm + right_pose.main_pivot_y_mm) / 2000.0,
        ]
    )
    local = np.asarray(
        [
            direction * design.com_x_from_main_pivot_mm / 1000.0,
            physical.com_lateral_mm / 1000.0,
            design.com_z_from_main_pivot_mm / 1000.0,
        ]
    )
    pitch = radians(pitch_deg)
    roll = radians(roll_deg)
    pitch_rotation = np.asarray(
        [
            [cos(pitch), 0.0, -sin(pitch)],
            [0.0, 1.0, 0.0],
            [sin(pitch), 0.0, cos(pitch)],
        ]
    )
    roll_rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(roll), -sin(roll)],
            [0.0, sin(roll), cos(roll)],
        ]
    )
    center_of_mass = main_center + roll_rotation @ pitch_rotation @ local
    return pitch_deg, roll_deg, center_of_mass


def _contact_geometry(
    design: ChassisDesign,
    physical: PhysicalParameters,
    pose: Pose,
    terrain: Terrain,
    direction: int,
    scenario: Scenario,
) -> tuple[
    tuple[_ContactGeometry, ...],
    dict[str, ChassisFrame],
    dict[str, np.ndarray],
    float,
    float,
    np.ndarray,
]:
    left_pose = pose
    left_terrain = terrain
    if scenario == "full_width":
        right_pose = pose
        right_terrain = terrain
    else:
        right_pose = _flat_pose_at_x(design, pose.main_pivot_x_mm)
        right_terrain = StepTerrain(0.0)

    frames = {
        "левая": calculate_frame(design, left_pose, direction),
        "правая": calculate_frame(design, right_pose, direction),
    }
    poses = {"левая": left_pose, "правая": right_pose}
    terrains = {"левая": left_terrain, "правая": right_terrain}
    side_y = {
        "левая": -physical.track_width_mm / 2000.0,
        "правая": physical.track_width_mm / 2000.0,
    }

    contacts: list[_ContactGeometry] = []
    for side in ("левая", "правая"):
        for wheel_index in range(3):
            contact = _wheel_contact_2d(
                design,
                frames[side],
                poses[side],
                terrains[side],
                wheel_index,
            )
            if contact is None:
                continue
            mode, point, normal, tangent = contact
            contacts.append(
                _ContactGeometry(
                    side=side,
                    wheel=WHEEL_NAMES[wheel_index],
                    mode=mode,
                    point_m=np.asarray([point.x / 1000.0, side_y[side], point.y / 1000.0]),
                    normal=normal,
                    tangent=tangent,
                )
            )

    pitch, roll, center_of_mass = _body_state(
        design,
        physical,
        left_pose,
        right_pose,
        direction,
    )
    pivots = {
        "main_левая": np.asarray(
            [frames["левая"].main_pivot.x / 1000.0, side_y["левая"], frames["левая"].main_pivot.y / 1000.0]
        ),
        "main_правая": np.asarray(
            [frames["правая"].main_pivot.x / 1000.0, side_y["правая"], frames["правая"].main_pivot.y / 1000.0]
        ),
        "bogie_левая": np.asarray(
            [frames["левая"].bogie_pivot.x / 1000.0, side_y["левая"], frames["левая"].bogie_pivot.y / 1000.0]
        ),
        "bogie_правая": np.asarray(
            [frames["правая"].bogie_pivot.x / 1000.0, side_y["правая"], frames["правая"].bogie_pivot.y / 1000.0]
        ),
    }
    return tuple(contacts), frames, pivots, pitch, roll, center_of_mass


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _support_margins(
    contacts: tuple[_ContactGeometry, ...],
    center_of_mass: np.ndarray,
) -> tuple[float, float, float, list[tuple[float, float]]]:
    points_mm = [
        (contact.point_m[0] * 1000.0, contact.point_m[1] * 1000.0)
        for contact in contacts
    ]
    hull = _convex_hull(points_mm)
    if len(hull) < 3:
        return float("-inf"), float("-inf"), float("-inf"), hull

    com_x = center_of_mass[0] * 1000.0
    com_y = center_of_mass[1] * 1000.0
    longitudinal = min(com_x - min(x for x, _ in hull), max(x for x, _ in hull) - com_x)
    lateral = min(com_y - min(y for _, y in hull), max(y for _, y in hull) - com_y)

    edge_margins: list[float] = []
    for index, first in enumerate(hull):
        second = hull[(index + 1) % len(hull)]
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = hypot(dx, dy)
        if length <= 1e-9:
            continue
        signed = (dx * (com_y - first[1]) - dy * (com_x - first[0])) / length
        edge_margins.append(signed)
    polygon_margin = min(edge_margins, default=float("-inf"))
    return longitudinal, lateral, polygon_margin, hull


def _tipover_angle(
    contacts: tuple[_ContactGeometry, ...],
    center_of_mass: np.ndarray,
    hull: list[tuple[float, float]],
    polygon_margin_mm: float,
) -> float:
    if len(hull) < 3:
        return float("-inf")
    by_xy: dict[tuple[float, float], np.ndarray] = {}
    for contact in contacts:
        key = (contact.point_m[0] * 1000.0, contact.point_m[1] * 1000.0)
        by_xy.setdefault(key, contact.point_m)

    gravity = np.asarray([0.0, 0.0, -1.0])
    angles: list[float] = []
    for index, first_xy in enumerate(hull):
        second_xy = hull[(index + 1) % len(hull)]
        first = by_xy[first_xy]
        second = by_xy[second_xy]
        axis = second - first
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= 1e-12:
            continue
        axis /= axis_length
        to_axis = first - center_of_mass
        closest = to_axis - axis * float(np.dot(to_axis, axis))
        force = gravity - axis * float(np.dot(gravity, axis))
        if np.linalg.norm(closest) <= 1e-12 or np.linalg.norm(force) <= 1e-12:
            continue
        cosine = float(np.dot(_normalize(closest), _normalize(force)))
        angles.append(degrees(acos(min(1.0, max(-1.0, cosine)))))
    angle = min(angles, default=float("-inf"))
    return angle if polygon_margin_mm >= 0.0 else -angle


def _equilibrium_system(
    contacts: tuple[_ContactGeometry, ...],
    pivots: dict[str, np.ndarray],
    center_of_mass: np.ndarray,
    physical: PhysicalParameters,
    friction_coefficient: float | None = None,
    motor_force_limit_n: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[float | None, float | None]], float]:
    count = len(contacts)
    # Для каждого контакта: N (нормаль), T (касательная в плоскости
    # движения), Q (поперечная касательная), затем общая загрузка приводов λ.
    variable_count = 3 * count + 1
    weight = physical.mass_kg * physical.gravity_m_s2 * physical.dynamic_factor
    friction = physical.friction_coefficient if friction_coefficient is None else friction_coefficient
    force_limit = 1e12 if motor_force_limit_n is None else motor_force_limit_n

    normal_columns: list[np.ndarray] = []
    tangent_columns: list[np.ndarray] = []
    lateral_columns: list[np.ndarray] = []
    for contact in contacts:
        normal_columns.append(contact.normal)
        tangent_columns.append(contact.tangent)
        lateral_columns.append(np.asarray([0.0, 1.0, 0.0]))

    rows: list[np.ndarray] = []
    targets: list[float] = []

    def add_force_equation(component: int, target: float) -> None:
        row = np.zeros(variable_count)
        for index in range(count):
            row[index] = normal_columns[index][component]
            row[count + index] = tangent_columns[index][component]
            row[2 * count + index] = lateral_columns[index][component]
        rows.append(row)
        targets.append(target)

    add_force_equation(0, 0.0)
    add_force_equation(1, 0.0)
    add_force_equation(2, weight)

    for component in (0, 1, 2):
        row = np.zeros(variable_count)
        for index, contact in enumerate(contacts):
            arm = contact.point_m - center_of_mass
            row[index] = np.cross(arm, normal_columns[index])[component]
            row[count + index] = np.cross(arm, tangent_columns[index])[component]
            row[2 * count + index] = np.cross(
                arm,
                lateral_columns[index],
            )[component]
        rows.append(row)
        targets.append(0.0)

    # Свободный шарнир bogie: сумма моментов двух его колёс равна нулю.
    for side in ("левая", "правая"):
        row = np.zeros(variable_count)
        pivot = pivots[f"bogie_{side}"]
        for index, contact in enumerate(contacts):
            if contact.side != side or contact.wheel not in ("middle", "rear"):
                continue
            arm = contact.point_m - pivot
            row[index] = np.cross(arm, normal_columns[index])[1]
            row[count + index] = np.cross(arm, tangent_columns[index])[1]
            row[2 * count + index] = np.cross(
                arm,
                lateral_columns[index],
            )[1]
        rows.append(row)
        targets.append(0.0)

    # Идеальный симметричный дифференциал создаёт одинаковые моменты на двух
    # выходных валах; реакция корпуса равна их сумме. Поэтому требуются равные
    # моменты от грунта относительно левого и правого главных шарниров.
    differential_row = np.zeros(variable_count)
    for index, contact in enumerate(contacts):
        pivot = pivots[f"main_{contact.side}"]
        arm = contact.point_m - pivot
        sign = 1.0 if contact.side == "левая" else -1.0
        differential_row[index] = sign * np.cross(arm, normal_columns[index])[1]
        differential_row[count + index] = sign * np.cross(arm, tangent_columns[index])[1]
        differential_row[2 * count + index] = sign * np.cross(
            arm,
            lateral_columns[index],
        )[1]
    rows.append(differential_row)
    targets.append(0.0)

    # На одинаковом профиле решение должно сохранять левую/правую симметрию.
    # Без этих равенств недоопределённая система может выбрать математически
    # допустимое, но произвольно несимметричное распределение реакций.
    for wheel in WHEEL_NAMES:
        left_index = next(
            (
                index
                for index, contact in enumerate(contacts)
                if contact.side == "левая" and contact.wheel == wheel
            ),
            None,
        )
        right_index = next(
            (
                index
                for index, contact in enumerate(contacts)
                if contact.side == "правая" and contact.wheel == wheel
            ),
            None,
        )
        if left_index is None or right_index is None:
            continue
        left = contacts[left_index]
        right = contacts[right_index]
        same_profile = (
            left.mode == right.mode
            and np.linalg.norm(left.point_m[[0, 2]] - right.point_m[[0, 2]]) < 1e-9
        )
        if not same_profile:
            continue
        normal_row = np.zeros(variable_count)
        normal_row[left_index] = 1.0
        normal_row[right_index] = -1.0
        rows.append(normal_row)
        targets.append(0.0)
        tangent_row = np.zeros(variable_count)
        tangent_row[count + left_index] = 1.0
        tangent_row[count + right_index] = -1.0
        rows.append(tangent_row)
        targets.append(0.0)
        lateral_row = np.zeros(variable_count)
        lateral_row[2 * count + left_index] = 1.0
        lateral_row[2 * count + right_index] = 1.0
        rows.append(lateral_row)
        targets.append(0.0)

    inequalities: list[np.ndarray] = []
    limits: list[float] = []
    rolling = physical.rolling_resistance_coefficient
    for index in range(count):
        # D_i = T_i + c_rr*N_i — продольная сила привода. Четыре
        # неравенства |D_i|+|Q_i|≤μN_i образуют консервативную линейную
        # аппроксимацию круга сцепления и учитывают совместную работу сил.
        for drive_sign in (-1.0, 1.0):
            for lateral_sign in (-1.0, 1.0):
                friction_row = np.zeros(variable_count)
                friction_row[index] = drive_sign * rolling - friction
                friction_row[count + index] = drive_sign
                friction_row[2 * count + index] = lateral_sign
                inequalities.append(friction_row)
                limits.append(0.0)

        motor_positive = np.zeros(variable_count)
        motor_positive[index] = rolling
        motor_positive[count + index] = 1.0
        motor_positive[-1] = -force_limit
        inequalities.append(motor_positive)
        limits.append(0.0)

        motor_negative = np.zeros(variable_count)
        motor_negative[index] = -rolling
        motor_negative[count + index] = -1.0
        motor_negative[-1] = -force_limit
        inequalities.append(motor_negative)
        limits.append(0.0)

    bounds: list[tuple[float | None, float | None]] = (
        [(0.0, None)] * count
        + [(None, None)] * count
        + [(None, None)] * count
        + [(0.0, 1.0)]
    )
    return (
        np.asarray(rows),
        np.asarray(targets),
        np.asarray(inequalities),
        np.asarray(limits),
        bounds,
        weight,
    )


def _solve_loads(
    design: ChassisDesign,
    contacts: tuple[_ContactGeometry, ...],
    pivots: dict[str, np.ndarray],
    center_of_mass: np.ndarray,
    physical: PhysicalParameters,
    friction: float | None = None,
    force_limit_multiplier: float = 1.0,
) -> tuple[object, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[float | None, float | None]], float]]:
    force_limit = (
        physical.wheel_torque_nm / (design.wheel_radius_mm / 1000.0)
        * force_limit_multiplier
    )
    system = _equilibrium_system(
        contacts,
        pivots,
        center_of_mass,
        physical,
        friction_coefficient=friction,
        motor_force_limit_n=force_limit,
    )
    a_eq, b_eq, a_ub, b_ub, bounds, _ = system
    objective = np.zeros(a_eq.shape[1])
    objective[-1] = 1.0
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    return result, system


def analyze_frame(
    design: ChassisDesign,
    physical: PhysicalParameters,
    pose: Pose,
    terrain: Terrain,
    direction: int,
    scenario: Scenario,
    pose_index: int = 0,
) -> StaticFrameResult:
    contacts, frames, pivots, pitch, roll, center_of_mass = _contact_geometry(
        design,
        physical,
        pose,
        terrain,
        direction,
        scenario,
    )
    weight = physical.mass_kg * physical.gravity_m_s2 * physical.dynamic_factor
    com_mm = tuple(float(value * 1000.0) for value in center_of_mass)
    longitudinal, lateral, polygon_margin, hull = _support_margins(contacts, center_of_mass)
    tipover = _tipover_angle(contacts, center_of_mass, hull, polygon_margin)

    left_frame = frames["левая"]
    right_frame = frames["правая"]
    right_rocker_deg = pose.body_pitch_deg if scenario == "full_width" else 0.0
    right_bogie_deg = pose.bogie_angle_deg if scenario == "full_width" else 0.0
    base_equations = (
        f"Rк={design.wheel_radius_mm:.3f} мм; dрыч={design.arm_diameter_mm:.3f} мм; B={physical.track_width_mm:.3f} мм",
        (
            "Длины: "
            f"Lrocker,F={hypot(left_frame.front_wheel.x - left_frame.main_pivot.x, left_frame.front_wheel.y - left_frame.main_pivot.y):.3f}; "
            f"Lrocker,B={hypot(left_frame.bogie_pivot.x - left_frame.main_pivot.x, left_frame.bogie_pivot.y - left_frame.main_pivot.y):.3f}; "
            f"Lbogie,M={hypot(left_frame.middle_wheel.x - left_frame.bogie_pivot.x, left_frame.middle_wheel.y - left_frame.bogie_pivot.y):.3f}; "
            f"Lbogie,R={hypot(left_frame.rear_wheel.x - left_frame.bogie_pivot.x, left_frame.rear_wheel.y - left_frame.bogie_pivot.y):.3f} мм"
        ),
        f"θrocker,L={pose.body_pitch_deg:.3f}°; θbogie,L={pose.bogie_angle_deg:.3f}°; θbogie,abs,L={pose.body_pitch_deg + pose.bogie_angle_deg:.3f}°",
        f"θrocker,R={right_rocker_deg:.3f}°; θbogie,R={right_bogie_deg:.3f}°; θbogie,abs,R={right_rocker_deg + right_bogie_deg:.3f}°",
        f"OL=({left_frame.main_pivot.x:.3f}; {left_frame.main_pivot.y:.3f}) мм; OR=({right_frame.main_pivot.x:.3f}; {right_frame.main_pivot.y:.3f}) мм",
        (
            "Центры колёс L: "
            f"F=({left_frame.front_wheel.x:.3f}; {left_frame.front_wheel.y:.3f}), "
            f"M=({left_frame.middle_wheel.x:.3f}; {left_frame.middle_wheel.y:.3f}), "
            f"R=({left_frame.rear_wheel.x:.3f}; {left_frame.rear_wheel.y:.3f}) мм"
        ),
        (
            "Центры колёс R: "
            f"F=({right_frame.front_wheel.x:.3f}; {right_frame.front_wheel.y:.3f}), "
            f"M=({right_frame.middle_wheel.x:.3f}; {right_frame.middle_wheel.y:.3f}), "
            f"R=({right_frame.rear_wheel.x:.3f}; {right_frame.rear_wheel.y:.3f}) мм"
        ),
        f"ЦМ=({com_mm[0]:.3f}; {com_mm[1]:.3f}; {com_mm[2]:.3f}) мм относительно мировой системы",
        f"Wр = m·g·Kд = {physical.mass_kg:.3f}·{physical.gravity_m_s2:.3f}·{physical.dynamic_factor:.3f} = {weight:.3f} Н",
        f"φплатформы = (θL + θR)/2 = {pitch:.3f}° — выход идеального дифференциала",
        f"κ = atan((zR − zL)/B) = {roll:.3f}° — поперечный крен",
        "ΣFx = 0; ΣFz = Wр; ΣMx = ΣMy = ΣMz = 0",
        "Для каждого bogie: ΣMy,B = 0; для дифференциала: My,L − My,R = 0",
        "Dᵢ = Tᵢ + crr·Nᵢ; |Dᵢ| + |Qᵢ| ≤ μNᵢ; |Dᵢ|·R ≤ τmax; Nᵢ ≥ 0",
    )

    if len(contacts) < 3 or len(hull) < 3:
        return StaticFrameResult(
            False, scenario, pose_index, "Недостаточно независимых точек опоры.",
            pitch, roll, com_mm, weight, longitudinal, lateral, polygon_margin,
            tipover, float("inf"), float("inf"), float("inf"), float("inf"),
            (), base_equations, ("HACOT_1998", "PAPADOPOULOS_2000"),
        )
    if polygon_margin < physical.minimum_support_margin_mm - 1e-8:
        return StaticFrameResult(
            False,
            scenario,
            pose_index,
            (
                f"Проекция ЦМ имеет запас {polygon_margin:.2f} мм, "
                f"требуется не менее {physical.minimum_support_margin_mm:.2f} мм."
            ),
            pitch, roll, com_mm, weight, longitudinal, lateral, polygon_margin,
            tipover, float("inf"), float("inf"), 0.0, 0.0, (), base_equations,
            ("PAPADOPOULOS_2000",),
        )

    result, system = _solve_loads(
        design, contacts, pivots, center_of_mass, physical
    )
    a_eq, b_eq, _, _, _, _ = system
    if not result.success:
        unlimited_motor, _ = _solve_loads(
            design,
            contacts,
            pivots,
            center_of_mass,
            physical,
            force_limit_multiplier=1e6,
        )
        unlimited_friction, _ = _solve_loads(
            design,
            contacts,
            pivots,
            center_of_mass,
            physical,
            friction=1e6,
            force_limit_multiplier=1e6,
        )
        if unlimited_motor.success:
            reason = "Недостаточен момент мотор-колёс."
        elif unlimited_friction.success:
            reason = "Недостаточно сцепления колёс с поверхностью."
        else:
            reason = "Уравнения равновесия пассивной подвески не имеют допустимого решения."
        return StaticFrameResult(
            False, scenario, pose_index, reason, pitch, roll, com_mm, weight,
            longitudinal, lateral, polygon_margin, tipover, float("inf"),
            float("inf"), float("inf"), float("inf"), (), base_equations,
            ("HACOT_1998", "LAMON_2004", "PAPADOPOULOS_2000"),
        )

    solution = np.asarray(result.x, dtype=float)
    count = len(contacts)
    normals = solution[:count]
    tangents = solution[count:2 * count]
    laterals = solution[2 * count:3 * count]
    wheel_radius_m = design.wheel_radius_mm / 1000.0
    loads: list[ContactLoad] = []
    for index, contact in enumerate(contacts):
        normal_force = max(0.0, float(normals[index]))
        tangent_force = float(tangents[index])
        lateral_force = float(laterals[index])
        motor_force = (
            tangent_force
            + physical.rolling_resistance_coefficient * normal_force
        )
        torque = motor_force * wheel_radius_m
        friction_capacity = physical.friction_coefficient * normal_force
        loads.append(
            ContactLoad(
                side=contact.side,
                wheel=contact.wheel,
                mode=contact.mode,
                point_mm=tuple(float(value * 1000.0) for value in contact.point_m),
                normal=tuple(float(value) for value in contact.normal),
                tangent=tuple(float(value) for value in contact.tangent),
                normal_force_n=normal_force,
                net_tangent_force_n=tangent_force,
                lateral_force_n=lateral_force,
                motor_force_n=motor_force,
                motor_torque_nm=torque,
                friction_utilization=(
                    hypot(motor_force, lateral_force) / friction_capacity
                    if friction_capacity > 1e-9
                    else (0.0 if abs(motor_force) <= 1e-9 else float("inf"))
                ),
                torque_utilization=abs(torque) / physical.wheel_torque_nm,
            )
        )

    # Геометрическое касание с N≈0 не является фактической опорой. После
    # решения равновесия устойчивость пересчитывается только по нагруженным
    # контактам; это особенно важно, когда колесо висит над рвом.
    active_threshold_n = max(1e-6, weight * 1e-8)
    active_contacts = tuple(
        contact
        for contact, normal_force in zip(contacts, normals)
        if normal_force > active_threshold_n
    )
    longitudinal, lateral, polygon_margin, hull = _support_margins(
        active_contacts,
        center_of_mass,
    )
    tipover = _tipover_angle(
        active_contacts,
        center_of_mass,
        hull,
        polygon_margin,
    )

    residual = a_eq @ solution - b_eq
    force_residual = float(np.max(np.abs(residual[:3])))
    moment_residual = float(np.max(np.abs(residual[3:6])))
    max_torque = max(load.torque_utilization for load in loads)
    max_friction = max(load.friction_utilization for load in loads)
    support_feasible = (
        len(hull) >= 3
        and polygon_margin >= physical.minimum_support_margin_mm - 1e-8
    )
    support_reason = ""
    if len(hull) < 3:
        support_reason = "Нагруженные контакты не образуют опорный многоугольник."
    elif not support_feasible:
        support_reason = (
            f"Запас по нагруженным контактам {polygon_margin:.2f} мм, "
            f"требуется не менее {physical.minimum_support_margin_mm:.2f} мм."
        )
    equations = base_equations + tuple(
        (
            f"{load.side}, {load.wheel}: N={load.normal_force_n:.3f} Н; "
            f"T={load.net_tangent_force_n:.3f} Н; "
            f"Q={load.lateral_force_n:.3f} Н; "
            f"D=T+crr·N={load.motor_force_n:.3f} Н; "
            f"τ=D·R={load.motor_torque_nm:.3f} Н·м"
        )
        for load in loads
    ) + (
        f"Нагруженных контактов (N > {active_threshold_n:.3e} Н): {len(active_contacts)} из {len(contacts)}",
        f"Запас опорного многоугольника = {polygon_margin:.3f} мм; продольный = {longitudinal:.3f} мм; поперечный = {lateral:.3f} мм",
        f"Максимальная загрузка момента = {100.0 * max_torque:.2f}%; сцепления = {100.0 * max_friction:.2f}%",
        f"Невязка равновесия: силы {force_residual:.3e} Н; моменты {moment_residual:.3e} Н·м",
    )
    return StaticFrameResult(
        support_feasible, scenario, pose_index, support_reason, pitch, roll, com_mm, weight,
        longitudinal, lateral, polygon_margin, tipover, max_torque,
        max_friction, force_residual, moment_residual, tuple(loads), equations,
        ("HACOT_1998", "LAMON_2004", "PAPADOPOULOS_2000"),
    )


def analyze_traversal(
    design: ChassisDesign,
    physical: PhysicalParameters,
    traversal: TraversalResult,
    scenario: Scenario,
    sample_stride: int = 1,
) -> TraversalStatics:
    if not traversal.success or not traversal.poses:
        return TraversalStatics(
            False,
            scenario,
            (),
            traversal.failure_reason or "Геометрический путь не найден.",
        )
    terrain: Terrain
    if traversal.obstacle_kind == "step":
        terrain = StepTerrain(traversal.obstacle_value_mm)
    else:
        terrain = DitchTerrain(traversal.obstacle_value_mm)

    stride = max(1, int(sample_stride))
    indices = list(range(0, len(traversal.poses), stride))
    if indices[-1] != len(traversal.poses) - 1:
        indices.append(len(traversal.poses) - 1)
    frames: list[StaticFrameResult] = []
    for index in indices:
        frame = analyze_frame(
            design,
            physical,
            traversal.poses[index],
            terrain,
            traversal.direction,
            scenario,
            index,
        )
        frames.append(frame)
        if not frame.feasible:
            return TraversalStatics(
                False,
                scenario,
                tuple(frames),
                f"Кадр {index + 1}: {frame.reason}",
            )
    return TraversalStatics(True, scenario, tuple(frames))
