"""Расчёты для отчёта: параметрические развёртки и прогон CMA-ES.

    python report/compute.py torque radius geometry cmaes_torque cmaes_limits compare

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
from optimize import Goal, optimize  # noqa: E402
from simulate import find_limit  # noqa: E402

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


def sweep_torque() -> None:
    designs = {"start": start_design(), "optimized": OPTIMIZED}
    torques = sorted({float(t) for t in range(4, 42, 3)} | {11.0, 12.0})
    tasks = [({"design": label, "torque_nm": torque}, design, replace(PLATFORM, wheel_torque_nm=torque), direction, 900.0)
             for label, design in designs.items() for torque in torques for direction in (1, -1)]
    run_tasks("torque", tasks, {"designs": {k: asdict(v) for k, v in designs.items()}})


def sweep_radius() -> None:
    base = OPTIMIZED
    radii = [70.0, 85.0, 100.0, 115.0, 130.0, 145.0, 160.0, 175.0]
    torques = [8.0, 16.0, 24.0]
    tasks = []
    for radius in radii:
        k = radius / base_radius()
        design = scaled(base, k)
        for torque in torques:
            # Габарит снят: при подобии длина растёт вместе с колесом.
            platform = replace(PLATFORM, wheel_radius_mm=radius, wheel_torque_nm=torque, max_length_mm=1.0e5)
            assert not validate(design, platform), validate(design, platform)
            for direction in (1, -1):
                tasks.append(({"radius_mm": radius, "torque_nm": torque}, design, platform, direction, 6.0 * radius))
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


def run_cmaes(mode: str = "torque", generations: int = 20) -> None:
    start = start_design()
    platform = PLATFORM
    goal = Goal(mode=mode)
    free = {name: (float(BOUNDS[name][0]), float(BOUNDS[name][1])) for name in FREE}
    started = time.time()

    def progress(done: int, total: int, best) -> None:
        if done % 50 == 0:
            print(f"  cmaes {mode} {done}/{total}, {time.time() - started:.0f} с, лучший "
                  f"{best.score if best else float('nan'):.1f} Н·м", flush=True)

    result = optimize(start, platform, goal, free, generations=generations, progress=progress)
    evaluations = [
        {"cost": e.cost, "score": e.score, "valid": e.valid, "refined": e.refined, "values": e.values}
        for e in result.evaluations
    ]
    data = {
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
    (DATA / f"cmaes_{mode}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=float),
                                     encoding="utf-8")
    print(f"cmaes {mode}: {time.time() - started:.0f} с", flush=True)


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
               "cmaes_limits": lambda: run_cmaes("limits"), "compare": start_vs_best}
    for name in sys.argv[1:]:
        actions[name]()
