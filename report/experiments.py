"""Расчёты для отчёта (вторая серия): 10 свободных параметров, выбор массы, ЛГК, CMA-ES.

    python report/experiments.py all          # всё по порядку, готовые этапы пропускаются
    python report/experiments.py mass lhs ... # отдельные этапы

Этапы:
    mass     — предел ступени от массы корпуса для нескольких разных геометрий; выбор массы;
    lhs      — латинский гиперкуб по 10 параметрам, грубая оценка, отбор стартов;
    starts   — CMA-ES из трёх лучших точек ЛГК;
    final    — дообучение лучшего результата из его же точки → итоговая геометрия;
    weights  — 4 набора весов × 3 seed из лучшей точки ЛГК;
    analysis — развёртки для итоговой геометрии: момент, радиус, масса, углы, все высоты.

Результаты — report/data2/*.json; прерванный расчёт можно перезапустить с того же места.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from compute import PLATFORM as BASE_PLATFORM, UPPER, WEIGHTS, limits_goal, scaled  # noqa: E402
from model import Design, Platform, validate  # noqa: E402
from optimize import evaluate_many, optimize  # noqa: E402
from simulate import Obstacle, find_limit, run  # noqa: E402

DATA = Path(__file__).resolve().parent / "data2"
OLD = Path(__file__).resolve().parent / "data"
WORKERS = os.cpu_count() or 1
SPEEDS = (0.12, 0.15, 0.18)
CONTROLS = ("speed", "torque")
GENERATIONS = 20

BOUNDS = {
    "bogie_rear_arm_mm": (100.0, 450.0), "bogie_front_arm_mm": (100.0, 450.0), "bogie_angle_deg": (100.0, 280.0),
    "rocker_rear_arm_mm": (150.0, 600.0), "rocker_front_arm_mm": (200.0, 750.0), "rocker_angle_deg": (100.0, 280.0),
    "outer_strut_height_mm": (150.0, 300.0), "outer_strut_angle_deg": (-25.0, 25.0),
    "middle_strut_height_mm": (150.0, 300.0), "middle_strut_angle_deg": (-25.0, 25.0),
}
FREE = tuple(BOUNDS)


def path(name: str) -> Path:
    return DATA / f"{name}.json"


def save(name: str, data) -> None:
    DATA.mkdir(exist_ok=True)
    path(name).write_text(json.dumps(data, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def load(name: str):
    return json.loads(path(name).read_text(encoding="utf-8")) if path(name).exists() else None


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def platform() -> Platform:
    """Платформа с массой, выбранной на этапе mass."""
    chosen = load("mass_choice")
    if chosen is None:
        raise SystemExit("Сначала этап mass: масса корпуса ещё не выбрана.")
    return replace(BASE_PLATFORM, body_mass_kg=float(chosen["body_mass_kg"]))


# ---------------------------------------------------------------------------
# Поиски пределов в пуле процессов
# ---------------------------------------------------------------------------

def _limit(task: tuple) -> dict:
    key, design, plat, kind, direction, upper, width = task
    result = find_limit(design, plat, kind, direction, upper_mm=upper, tolerance_mm=10.0, bump_width_mm=width)
    return {**key, "direction": direction, "value_mm": result.value_mm, "reason": result.reason}


def _single(task: tuple) -> dict:
    key, design, plat, obstacle = task
    result = run(design, plat, obstacle, record=False)
    return {**key, "success": result.success, "reason": result.reason}


def pool_map(function, tasks: list) -> list:
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(function, tasks, chunksize=1))


def step_tasks(key: dict, design: Design, plat: Platform, upper: float = 700.0) -> list[tuple]:
    """Подъём на ступень в обе стороны при трёх скоростях подъезда."""
    return [({**key, "speed": speed}, design, replace(plat, speed_m_s=speed), "step_up", direction, upper, 120.0)
            for speed in SPEEDS for direction in (1, -1)]


# ---------------------------------------------------------------------------
# Этап mass: выбор массы корпуса
# ---------------------------------------------------------------------------

MASSES = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0)


def reference_designs() -> dict[str, Design]:
    """Разные хорошие геометрии первой серии — чтобы выбор массы не зависел от одной геометрии."""
    designs = {"variant3": Design(**json.loads((ROOT / "Подвески" / "rocker_bogie_variant (3).json")
                                               .read_text(encoding="utf-8"))["design"])}
    for name in ("run_hand_s2", "run_random1_s1", "run_w_equal"):
        source = OLD / f"{name}.json"
        if source.exists():
            designs[name] = Design(**json.loads(source.read_text(encoding="utf-8"))["best"])
    return designs


def stage_mass() -> None:
    if load("mass_choice"):
        return log("mass: уже есть")
    designs = reference_designs()
    tasks = [task for label, design in designs.items() for mass in MASSES
             for task in step_tasks({"design": label, "mass_kg": mass}, design,
                                    replace(BASE_PLATFORM, body_mass_kg=mass))]
    log(f"mass: {len(tasks)} поисков")
    rows = pool_map(_limit, tasks)
    save("mass", {"designs": {k: asdict(v) for k, v in designs.items()}, "rows": rows})
    choice = choose_mass(rows)
    save("mass_choice", choice)
    log(f"mass: выбрано {choice['body_mass_kg']} кг ({choice['rule']})")


def median_worst(rows: list[dict], *keys: str) -> dict[tuple, float]:
    """Худшее направление при каждой скорости, затем медиана по скоростям."""
    worst: dict[tuple, float] = {}
    for row in rows:
        key = tuple(row[k] for k in keys) + (row["speed"],)
        worst[key] = min(worst.get(key, float("inf")), row["value_mm"])
    groups: dict[tuple, list[float]] = {}
    for key, value in worst.items():
        groups.setdefault(key[:-1], []).append(value)
    return {key: float(np.median(values)) for key, values in groups.items()}


def choose_mass(rows: list[dict], share: float = 0.9) -> dict:
    """Наибольшая масса, при которой у всех геометрий предел ≥ share × плато (плато — при 5 и 10 кг)."""
    limits = median_worst(rows, "design", "mass_kg")
    designs = sorted({d for d, _ in limits})
    plateau = {d: max(limits[(d, 5.0)], limits[(d, 10.0)]) for d in designs}
    chosen = MASSES[0]
    for mass in MASSES:
        if all(limits[(d, mass)] >= share * plateau[d] for d in designs):
            chosen = mass
        else:
            break
    return {"body_mass_kg": chosen, "plateau": plateau,
            "rule": f"наибольшая масса, при которой у всех геометрий предел ступени ≥ {share:.0%} плато"}


# ---------------------------------------------------------------------------
# Этап lhs: латинский гиперкуб
# ---------------------------------------------------------------------------

LHS_POINTS = 2048
LHS_KEEP = 128


def lhs_designs(seed: int = 7) -> tuple[list[Design], int]:
    sampler = qmc.LatinHypercube(d=len(FREE), seed=seed)
    unit = sampler.random(LHS_POINTS)
    lower = np.array([BOUNDS[n][0] for n in FREE])
    upper = np.array([BOUNDS[n][1] for n in FREE])
    designs, valid = [], 0
    plat = platform()
    for row in qmc.scale(unit, lower, upper):
        design = replace(Design(), **{n: float(round(v, 1)) for n, v in zip(FREE, row)})
        if not validate(design, plat):
            valid += 1
            if len(designs) < LHS_KEEP:
                designs.append(design)
    return designs, valid


def stage_lhs() -> None:
    if load("lhs"):
        return log("lhs: уже есть")
    plat = platform()
    designs, valid = lhs_designs()
    goal = replace(limits_goal(WEIGHTS["base"]), tolerance_mm=30.0)
    log(f"lhs: допустимых {valid} из {LHS_POINTS}, оцениваю {len(designs)} грубо (шаг 30 мм)")
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        evaluations = evaluate_many(pool, designs, plat, goal)
    rows = [{"design": asdict(e.design), "score": e.score, "values": e.values} for e in evaluations]
    rows.sort(key=lambda r: -r["score"])
    save("lhs", {"points": LHS_POINTS, "valid": valid, "rows": rows})
    log(f"lhs: лучшая J = {rows[0]['score']:.0f}, медиана {np.median([r['score'] for r in rows]):.0f}")


def lhs_start(rank: int) -> Design:
    return Design(**load("lhs")["rows"][rank]["design"])


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------

def cmaes(name: str, start: Design, seed: int, weights: str = "base") -> dict:
    if load(name):
        log(f"{name}: уже есть")
        return load(name)
    plat = platform()
    goal = limits_goal(WEIGHTS[weights])
    started = time.time()
    log(f"{name}: старт, seed {seed}, веса {weights}")
    result = optimize(start, plat, goal, dict(BOUNDS), generations=GENERATIONS, seed=seed)
    best = result.best
    data = {
        "seed": seed, "weights": weights, "start": asdict(start),
        "best": asdict(best.design), "best_values": best.values, "best_score": best.score,
        "best_continuous": best.continuous,
        "best_reasons": [r.reason for r in best.results],
        "history": [asdict(g) for g in result.history],
        "evaluations": [{"score": e.score, "valid": e.valid, "refined": e.refined, "continuous": e.continuous,
                         "values": e.values} for e in result.evaluations],
        "seconds": time.time() - started,
    }
    save(name, data)
    log(f"{name}: J = {best.score:.0f} за {data['seconds'] / 60:.0f} мин")
    return data


STARTS = ("lhs1", "lhs2", "lhs3")


def stage_starts() -> None:
    for rank, label in enumerate(STARTS):
        cmaes(f"run_{label}_s1", lhs_start(rank), 1)


def best_run() -> tuple[str, dict]:
    runs = [(n, load(n)) for n in [f"run_{label}_s1" for label in STARTS] + weight_runs()]
    runs = [(n, d) for n, d in runs if d and d["weights"] == "base"]
    return max(runs, key=lambda item: item[1]["best_score"])


def stage_final() -> None:
    name, data = best_run()
    log(f"final: дообучение из {name}")
    cmaes("run_final", Design(**data["best"]), 2)


def weight_runs() -> list[str]:
    return [f"run_w_{weights}_s{seed}" for weights in WEIGHTS for seed in (1, 2, 3)]


def stage_weights() -> None:
    start = lhs_start(0)
    for weights in WEIGHTS:
        for seed in (1, 2, 3):
            if weights == "base" and seed == 1 and load("run_lhs1_s1"):
                # Совпадает с запуском из лучшей точки ЛГК.
                if not load("run_w_base_s1"):
                    save("run_w_base_s1", load("run_lhs1_s1"))
                continue
            cmaes(f"run_w_{weights}_s{seed}", start, seed, weights)


# ---------------------------------------------------------------------------
# Этап analysis: итоговая геометрия
# ---------------------------------------------------------------------------

def final_design() -> Design:
    candidates = [("run_final", load("run_final")), best_run()]
    name, data = max(((n, d) for n, d in candidates if d), key=lambda item: item[1]["best_score"])
    return Design(**data["best"])


def stage_analysis() -> None:
    plat = platform()
    final = final_design()
    if not load("an_torque"):
        tasks = [task for torque in sorted({float(t) for t in range(4, 42, 3)} | {11.0, 12.0, 14.0})
                 for control in CONTROLS
                 for task in step_tasks({"torque_nm": torque, "control": control}, final,
                                        replace(plat, wheel_torque_nm=torque, control=control))]
        log(f"analysis: момент, {len(tasks)} поисков")
        save("an_torque", {"design": asdict(final), "rows": pool_map(_limit, tasks)})
    if not load("an_mass"):
        tasks = [task for mass in MASSES for task in step_tasks({"mass_kg": mass}, final,
                                                                replace(plat, body_mass_kg=mass))]
        log(f"analysis: масса, {len(tasks)} поисков")
        save("an_mass", {"design": asdict(final), "rows": pool_map(_limit, tasks)})
    if not load("an_radius"):
        tasks = []
        for radius in (70.0, 85.0, 100.0, 115.0, 130.0, 145.0, 160.0, 175.0):
            design = scaled(final, radius / plat.wheel_radius_mm)
            for torque in (8.0, 16.0, 24.0):
                for control in CONTROLS:
                    scaled_platform = replace(plat, wheel_radius_mm=radius, wheel_torque_nm=torque,
                                              max_length_mm=1.0e5, control=control)
                    tasks += step_tasks({"radius_mm": radius, "torque_nm": torque, "control": control}, design,
                                        scaled_platform, upper=6.0 * radius)
        log(f"analysis: радиус, {len(tasks)} поисков")
        save("an_radius", {"design": asdict(final), "rows": pool_map(_limit, tasks)})
    if not load("an_angles"):
        bogie = [float(a) for a in range(140, 271, 10)]
        rocker = [float(a) for a in range(120, 251, 10)]
        tasks, invalid = [], []
        for b in bogie:
            for r in rocker:
                design = replace(final, bogie_angle_deg=b, rocker_angle_deg=r)
                key = {"bogie_angle_deg": b, "rocker_angle_deg": r}
                if validate(design, plat):
                    invalid.append(key)
                    continue
                tasks += [(key, design, plat, "step_up", d, 700.0, 120.0) for d in (1, -1)]
        log(f"analysis: углы, {len(tasks)} поисков")
        save("an_angles", {"design": asdict(final), "bogie": bogie, "rocker": rocker, "invalid": invalid,
                           "rows": pool_map(_limit, tasks)})
    if not load("an_heights"):
        designs = {"final": final}
        for name in weight_runs():
            data = load(name)
            if data:
                designs[name] = Design(**data["best"])
        kinds = {"step_up": (600.0, 120.0), "step_down": (600.0, 120.0), "ditch": (400.0, 120.0),
                 "bump_120": (300.0, 120.0), "bump_250": (300.0, 250.0)}
        tasks = []
        for label, design in designs.items():
            for key, (upper, width) in kinds.items():
                kind = "bump" if key.startswith("bump") else key
                for size in np.arange(10.0, upper + 1.0, 10.0):
                    for direction in (1, -1):
                        tasks.append(({"design": label, "kind": key, "size_mm": float(size), "direction": direction},
                                      design, plat, Obstacle(kind, float(size), direction, False, 500.0, width)))
        log(f"analysis: все высоты, {len(tasks)} прогонов")
        save("an_heights", {"designs": {k: asdict(v) for k, v in designs.items()},
                            "rows": pool_map(_single, tasks)})
    log("analysis: готово")


STAGES = {"mass": stage_mass, "lhs": stage_lhs, "starts": stage_starts, "final": stage_final,
          "weights": stage_weights, "analysis": stage_analysis}

if __name__ == "__main__":
    names = list(STAGES) if sys.argv[1:] == ["all"] else sys.argv[1:]
    for stage in names:
        STAGES[stage]()
