"""Расчёты для отчёта: параметрические развёртки и прогон CMA-ES.

    python report/compute.py torque radius geometry cmaes_torque cmaes_limits compare
    python report/compute.py heights dip runs

Результаты — JSON в report/data, графики строит report/plots.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from model import Design, Platform, design_from_coordinates, validate  # noqa: E402
from optimize import Criterion, Goal, optimize  # noqa: E402
from simulate import Obstacle, find_limit, run  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
VARIANT = json.loads((ROOT / "Подвески" / "rocker_bogie_variant (3).json").read_text(encoding="utf-8"))
# Платформа и итоговая геометрия — из последнего варианта пользователя.
PLATFORM = Platform(**{**VARIANT["platform"], "friction": 0.8})
OPTIMIZED = Design(**VARIANT["design"])
TOLERANCE_MM = 10.0
WORKERS = os.cpu_count() or 1


def start_design() -> Design:
    """Геометрия, введённая вручную по изображениям марсоходов (коммит c1ca990)."""
    base = replace(Design(), outer_strut_angle_deg=0.0, middle_strut_angle_deg=0.0)
    return design_from_coordinates(base, PLATFORM, 380.0, 940.0, 600.0, 470.0, 190.0, 390.0)


def scaled(design: Design, k: float) -> Design:
    """Геометрическое подобие: все длины × k, углы без изменений."""
    return replace(design, **{name: value * k for name, value in asdict(design).items() if name.endswith("_mm")})


def step_limit(task: tuple) -> dict:
    key, design, platform, direction, upper = task
    started = time.time()
    result = find_limit(design, platform, "step_up", direction, upper_mm=upper, tolerance_mm=TOLERANCE_MM)
    return {**key, "direction": direction, "step_mm": result.value_mm, "reason": result.reason,
            "unstable": result.unstable, "seconds": round(time.time() - started, 1)}


def run_tasks(name: str, tasks: list[tuple], meta: dict) -> None:
    started = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(step_limit, tasks))
    DATA.mkdir(exist_ok=True)
    (DATA / f"{name}.json").write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"{name}: {len(tasks)} поисков за {time.time() - started:.0f} с", flush=True)


def single_run(task: tuple) -> dict:
    key, design, platform, obstacle = task
    result = run(design, platform, obstacle, record=False)
    return {**key, "success": result.success, "reason": result.reason}


def run_grid(name: str, tasks: list[tuple], meta: dict) -> None:
    started = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(single_run, tasks, chunksize=2))
    DATA.mkdir(exist_ok=True)
    (DATA / f"{name}.json").write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"{name}: {len(tasks)} прогонов за {time.time() - started:.0f} с", flush=True)


HEIGHT_KINDS = {"step_up": (560.0, 0.0), "step_down": (560.0, 0.0), "ditch": (400.0, 0.0),
                "bump_120": (300.0, 120.0), "bump_250": (300.0, 250.0)}


def designs_for_heights() -> dict[str, Design]:
    designs = {"start": start_design(), "user": OPTIMIZED}
    path = DATA / "cmaes_limits.json"
    if path.exists():
        designs["cmaes"] = Design(**json.loads(path.read_text(encoding="utf-8"))["best"])
    return designs


def sweep_heights(names: tuple[str, ...] = (), out: str = "heights") -> None:
    """Все размеры препятствий с шагом 10 мм: есть ли непроходимые размеры ниже предела (посадка на мель)."""
    if names:
        designs = {name: Design(**json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))["best"])
                   for name in names}
    else:
        designs = designs_for_heights()
    tasks = []
    for label, design in designs.items():
        for key, (upper, width) in HEIGHT_KINDS.items():
            kind = "bump" if key.startswith("bump") else key
            for size in np.arange(10.0, upper + 1.0, 10.0):
                for direction in (1, -1):
                    obstacle = Obstacle(kind, float(size), direction, False, 500.0, width or 120.0)
                    tasks.append(({"design": label, "kind": key, "size_mm": float(size), "direction": direction},
                                  design, PLATFORM, obstacle))
    run_grid(out, tasks, {"designs": {k: asdict(v) for k, v in designs.items()}})


def sweep_dip() -> None:
    """Подъём варианта (3) передним ходом: сетка момент × высота около провала при 13 Н·м."""
    tasks = []
    for torque in np.arange(11.0, 16.01, 0.5):
        for size in np.arange(300.0, 541.0, 20.0):
            for control in ("speed", "torque"):
                platform = replace(PLATFORM, wheel_torque_nm=float(torque), control=control)
                tasks.append(({"torque_nm": float(torque), "size_mm": float(size), "control": control},
                              OPTIMIZED, platform, Obstacle("step_up", float(size), 1, False, 500.0, 120.0)))
    run_grid("dip", tasks, {})


SPEEDS = (0.12, 0.15, 0.18)
CONTROLS = ("speed", "torque")


def sweep_torque() -> None:
    """Ступень от момента; каждая точка — по трём скоростям подъезда и двум законам управления."""
    designs = {"start": start_design(), "optimized": OPTIMIZED}
    torques = sorted({float(t) for t in range(4, 42, 3)} | {11.0, 12.0, 14.0})
    tasks = [({"design": label, "torque_nm": torque, "control": control, "speed": speed}, design,
              replace(PLATFORM, wheel_torque_nm=torque, control=control, speed_m_s=speed), direction, 900.0)
             for label, design in designs.items() for torque in torques for control in CONTROLS
             for speed in SPEEDS for direction in (1, -1)]
    run_tasks("torque", tasks, {"designs": {k: asdict(v) for k, v in designs.items()}})


def sweep_radius() -> None:
    """Ступень от радиуса колеса при подобном масштабировании варианта (3)."""
    base = OPTIMIZED
    radii = [70.0, 85.0, 100.0, 115.0, 130.0, 145.0, 160.0, 175.0]
    torques = [8.0, 16.0, 24.0]
    tasks = []
    for radius in radii:
        design = scaled(base, radius / base_radius())
        for torque in torques:
            for control in CONTROLS:
                for speed in SPEEDS:
                    # Габарит снят: при подобии длина растёт вместе с колесом.
                    platform = replace(PLATFORM, wheel_radius_mm=radius, wheel_torque_nm=torque,
                                       max_length_mm=1.0e5, control=control, speed_m_s=speed)
                    assert not validate(design, platform), validate(design, platform)
                    for direction in (1, -1):
                        tasks.append(({"radius_mm": radius, "torque_nm": torque, "control": control,
                                       "speed": speed}, design, platform, direction, 6.0 * radius))
    run_tasks("radius", tasks, {"base": asdict(base)})


def base_radius() -> float:
    return PLATFORM.wheel_radius_mm


def sweep_geometry() -> None:
    base = OPTIMIZED
    torque = PLATFORM.wheel_torque_nm
    platform = PLATFORM
    bogie_angles = [float(a) for a in range(150, 261, 10)]
    rocker_angles = [float(a) for a in range(150, 241, 10)]
    tasks, invalid = [], []
    for bogie in bogie_angles:
        for rocker in rocker_angles:
            design = replace(base, bogie_angle_deg=bogie, rocker_angle_deg=rocker)
            key = {"bogie_angle_deg": bogie, "rocker_angle_deg": rocker}
            errors = validate(design, platform)
            if errors:
                invalid.append({**key, "errors": errors})
                continue
            for direction in (1, -1):
                tasks.append((key, design, platform, direction, 900.0))
    run_tasks("geometry", tasks, {"base": asdict(base), "torque_nm": torque, "invalid": invalid,
                                  "bogie_angles": bogie_angles, "rocker_angles": rocker_angles})


FREE = ("bogie_rear_arm_mm", "bogie_front_arm_mm", "bogie_angle_deg", "rocker_rear_arm_mm",
        "rocker_front_arm_mm", "rocker_angle_deg", "outer_strut_angle_deg", "middle_strut_angle_deg")
BOUNDS = {"bogie_rear_arm_mm": (100, 450), "bogie_front_arm_mm": (100, 450), "bogie_angle_deg": (100, 280),
          "rocker_rear_arm_mm": (150, 600), "rocker_front_arm_mm": (200, 750), "rocker_angle_deg": (100, 280),
          "outer_strut_angle_deg": (-25, 25), "middle_strut_angle_deg": (-25, 25)}


WEIGHTS = {
    "base": {"step_up": 1.0, "step_down": 0.3, "ditch": 0.5, "bump_120": 0.5, "bump_250": 0.5},
    "step": {"step_up": 1.0, "step_down": 0.1, "ditch": 0.1, "bump_120": 0.1, "bump_250": 0.1},
    "bumps": {"step_up": 0.3, "step_down": 0.1, "ditch": 0.3, "bump_120": 1.0, "bump_250": 1.0},
    "equal": {"step_up": 1.0, "step_down": 1.0, "ditch": 1.0, "bump_120": 1.0, "bump_250": 1.0},
}
# Верхние границы поиска пределов в опытах с CMA-ES (ступень — выше, чем в интерфейсе).
UPPER = {"step_up": 600.0, "step_down": 600.0, "ditch": 500.0, "bump_120": 300.0, "bump_250": 300.0}


def limits_goal(weights: dict[str, float]) -> Goal:
    criteria = []
    for key, weight in weights.items():
        width = float(key.split("_")[1]) if key.startswith("bump") else 0.0
        kind = "bump" if key.startswith("bump") else key
        criteria.append(Criterion(kind, weight, UPPER[key], 0.0, width))
    return Goal(criteria=tuple(criteria), mode="limits")


def random_start(seed: int) -> Design:
    """Случайная допустимая геометрия в пределах BOUNDS."""
    rng = np.random.default_rng(seed)
    while True:
        values = {name: float(round(rng.uniform(*BOUNDS[name]), 1)) for name in FREE}
        design = replace(OPTIMIZED, **values)
        if not validate(design, PLATFORM):
            return design


STARTS = {"hand": start_design, "user": lambda: OPTIMIZED, "random1": lambda: random_start(11),
          "random2": lambda: random_start(12)}


def run_cmaes(mode: str = "torque", generations: int = 20, start_name: str = "hand", seed: int = 1,
              weights: str | None = None, name: str | None = None) -> None:
    start = STARTS[start_name]()
    platform = PLATFORM
    goal = limits_goal(WEIGHTS[weights]) if weights else Goal(mode=mode)
    name = name or f"cmaes_{mode}"
    free = {key: (float(BOUNDS[key][0]), float(BOUNDS[key][1])) for key in FREE}
    started = time.time()

    def progress(done: int, total: int, best) -> None:
        if done % 200 == 0:
            print(f"  {name} {done}/{total}, {time.time() - started:.0f} с, лучший "
                  f"{best.score if best else float('nan'):.1f}", flush=True)

    result = optimize(start, platform, goal, free, generations=generations, seed=seed, progress=progress)
    evaluations = [
        {"cost": e.cost, "score": e.score, "valid": e.valid, "refined": e.refined, "values": e.values}
        for e in result.evaluations
    ]
    data = {
        "start_name": start_name,
        "seed": seed,
        "weights": weights,
        "start": asdict(start),
        "best": asdict(result.best.design) if result.best else None,
        "best_values": result.best.values if result.best else None,
        "best_score": result.best.score if result.best else None,
        "history": [asdict(g) for g in result.history],
        "evaluations": evaluations,
        "goal": [asdict(c) for c in goal.active],
        "free": free,
        "seconds": time.time() - started,
    }
    DATA.mkdir(exist_ok=True)
    (DATA / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=float),
                                      encoding="utf-8")
    print(f"{name}: {time.time() - started:.0f} с, J = {data['best_score']}", flush=True)


# Опыты: влияние начальной точки и seed (п. 6) и весов (п. 8); все от одной цели, кроме весов.
RUNS = {
    "run_hand_s1": ("hand", 1, "base"),
    "run_hand_s2": ("hand", 2, "base"),
    "run_hand_s3": ("hand", 3, "base"),
    "run_user_s1": ("user", 1, "base"),
    "run_random1_s1": ("random1", 1, "base"),
    "run_random2_s1": ("random2", 1, "base"),
    "run_w_step": ("hand", 1, "step"),
    "run_w_bumps": ("hand", 1, "bumps"),
    "run_w_equal": ("hand", 1, "equal"),
}


def table() -> None:
    """Пределы для таблицы отчёта: начальная, лучшие из опытов и вариант (3) — с границами UPPER."""
    from optimize import evaluate_parallel

    designs = {"start": start_design(), "user": OPTIMIZED}
    for name in RUNS:
        path = DATA / f"{name}.json"
        if path.exists():
            designs[name] = Design(**json.loads(path.read_text(encoding="utf-8"))["best"])
    goal = limits_goal(WEIGHTS["base"])
    rows = {}
    for label, design in designs.items():
        evaluation = evaluate_parallel(design, PLATFORM, goal)
        rows[label] = {"values": evaluation.values, "J_base": evaluation.score}
        print(label, rows[label], flush=True)
    (DATA / "table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float),
                                     encoding="utf-8")


def start_vs_best() -> None:
    """Пределы ступени при 16 Н·м и требуемые моменты у старта и у найденного варианта."""
    from optimize import evaluate_parallel

    data = json.loads((DATA / "cmaes_torque.json").read_text(encoding="utf-8"))
    rows = {}
    for label, values in (("start", data["start"]), ("user", asdict(OPTIMIZED)), ("best", data["best"])):
        design = Design(**values)
        rows[label] = {mode: evaluate_parallel(design, PLATFORM, Goal(mode=mode)).values
                       for mode in ("limits", "torque")}
    (DATA / "compare.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float),
                                       encoding="utf-8")
    print("compare:", rows, flush=True)


if __name__ == "__main__":
    actions = {"torque": sweep_torque, "radius": sweep_radius, "geometry": sweep_geometry,
               "cmaes_torque": lambda: run_cmaes("torque"),
               "cmaes_limits": lambda: run_cmaes("limits"), "compare": start_vs_best,
               "heights": sweep_heights, "dip": sweep_dip, "table": table,
               "heights_weights": lambda: sweep_heights(("run_w_step", "run_w_bumps", "run_w_equal",
                                                         "run_hand_s2"), "heights_weights")}
    for run_name, (start_name, seed, weights) in RUNS.items():
        actions[run_name] = (lambda n=run_name, a=start_name, b=seed, c=weights:
                             run_cmaes("limits", start_name=a, seed=b, weights=c, name=n))
    actions["runs"] = lambda: [actions[n]() for n in RUNS]
    for name in sys.argv[1:]:
        actions[name]()
