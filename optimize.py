"""Подбор геометрии рычагов.

Два режима цели:

* ``limits`` — наибольшие препятствия при заданном моменте колёс. Оценка —
  Σ вес × предел (мм); для каждого вида препятствия берётся худшее из двух
  направлений (и из сценариев «по всей ширине» / «под одним бортом»).
  Кочек может быть несколько — с разной шириной; у кочки ищется высота.
* ``torque`` — наименьший пиковый момент колеса, при котором проходятся все
  заданные препятствия. Оценка — наибольший из требуемых моментов (Н·м).
  Момент — непрерывная величина, и геометрия влияет на неё плавно, поэтому
  у этой цели нет плато, как у пределов, которые меняются ступеньками.

Поиск — CMA-ES (библиотека ``cma``) в нормированных координатах [0, 1].
Недопустимая геометрия получает штраф, растущий с величиной нарушения, —
так поиск видит, в какую сторону возвращаться. Каждая граница (предел или
момент) ищется от подсказки — значения у лучшего на данный момент варианта,
это в 1,5–2 раза сокращает число прогонов. Все прогоны поколения (варианты ×
препятствия) считаются параллельно. В конце несколько лучших вариантов
пересчитываются с более мелким шагом поиска.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from typing import Callable, Literal

import cma
import numpy as np

from model import Design, Platform, validate, violation_amount
from simulate import KIND_LABELS, LimitResult, Obstacle, ObstacleKind, TorqueResult, find_limit, required_torque

Mode = Literal["limits", "torque"]
MODE_LABELS = {
    "limits": "наибольшие препятствия при заданном моменте",
    "torque": "наименьший момент для заданных препятствий",
}
JobResult = LimitResult | TorqueResult

# Стоимость недопустимой геометрии: больше любой допустимой.
INVALID_COST = 1.0e4
# Препятствие, не проходимое и при наибольшем моменте, считается как
# требующее FAILED_TORQUE_FACTOR × наибольший момент.
FAILED_TORQUE_FACTOR = 2.0


@dataclass(frozen=True)
class Criterion:
    kind: ObstacleKind
    # Режим limits: вес в оценке; режим torque: > 0 — препятствие учитывается.
    weight: float
    # Режим limits: верхняя граница поиска предела.
    upper_mm: float
    # Режим torque: размер препятствия, которое нужно пройти.
    target_mm: float = 0.0
    # Кочка: ширина вдоль хода, мм.
    width_mm: float = 0.0

    @property
    def key(self) -> str:
        return criterion_key(self.kind, self.width_mm)

    def matches(self, obstacle: Obstacle) -> bool:
        return criterion_key(obstacle.kind, obstacle.bump_width_mm) == self.key


@dataclass(frozen=True)
class Goal:
    criteria: tuple[Criterion, ...] = (
        Criterion("step_up", 1.0, 300.0, 80.0),
        Criterion("step_down", 0.3, 500.0, 300.0),
        Criterion("ditch", 0.5, 500.0, 180.0),
        # Кирпич плашмя поперёк хода и небольшое бревно.
        Criterion("bump", 0.5, 300.0, 65.0, 120.0),
        Criterion("bump", 0.5, 300.0, 150.0, 250.0),
    )
    mode: Mode = "limits"
    one_side: bool = False
    tolerance_mm: float = 5.0
    tolerance_nm: float = 0.5
    max_torque_nm: float = 60.0
    ditch_depth_mm: float = 500.0

    @property
    def active(self) -> tuple[Criterion, ...]:
        return tuple(item for item in self.criteria if item.weight > 0.0)

    def refined(self, factor: float = 0.4) -> Goal:
        return replace(self, tolerance_mm=self.tolerance_mm * factor, tolerance_nm=self.tolerance_nm * factor)


@dataclass(frozen=True)
class Job:
    kind: ObstacleKind
    direction: int
    one_side: bool
    # limits — верхняя граница поиска, torque — размер препятствия.
    size_mm: float
    width_mm: float = 0.0


@dataclass
class Evaluation:
    design: Design
    errors: list[str]
    mode: Mode = "limits"
    # limits: Σ вес × предел, мм (больше — лучше); torque: требуемый момент, Н·м (меньше — лучше).
    score: float = 0.0
    # Минимизируемая величина для поиска.
    cost: float = INVALID_COST
    values: dict[str, float] = field(default_factory=dict)
    results: list[JobResult] = field(default_factory=list)
    refined: bool = False

    @property
    def valid(self) -> bool:
        return not self.errors


def jobs_for(goal: Goal) -> list[Job]:
    sides = (False, True) if goal.one_side else (False,)
    return [
        Job(item.kind, direction, one_side, item.upper_mm if goal.mode == "limits" else item.target_mm,
            item.width_mm)
        for item in goal.active
        for one_side in sides
        for direction in (1, -1)
    ]


def run_job(design: Design, platform: Platform, goal: Goal, job: Job, guess: float | None = None) -> JobResult:
    if goal.mode == "limits":
        return find_limit(
            design,
            platform,
            job.kind,
            job.direction,
            one_side=job.one_side,
            upper_mm=job.size_mm,
            tolerance_mm=goal.tolerance_mm,
            ditch_depth_mm=goal.ditch_depth_mm,
            bump_width_mm=job.width_mm,
            guess_mm=guess,
        )
    obstacle = Obstacle(job.kind, job.size_mm, job.direction, job.one_side, goal.ditch_depth_mm, job.width_mm)
    return required_torque(design, platform, obstacle, upper_nm=goal.max_torque_nm,
                           tolerance_nm=goal.tolerance_nm, guess_nm=guess)


def result_value(result: JobResult) -> float:
    return result.value_mm if isinstance(result, LimitResult) else result.torque_nm


def summarize(design: Design, goal: Goal, results: list[JobResult]) -> Evaluation:
    values: dict[str, float] = {}
    if goal.mode == "limits":
        score = 0.0
        for item in goal.active:
            relevant = [result.value_mm for result in results if item.matches(result.obstacle)]
            values[item.key] = min(relevant) if relevant else 0.0
            score += item.weight * values[item.key]
        return Evaluation(design, [], goal.mode, score, -score, values, results)
    for item in goal.active:
        relevant = [result.torque_nm for result in results if item.matches(result.obstacle)]
        values[item.key] = max(relevant) if relevant else 0.0
    score = max((result.torque_nm for result in results), default=0.0)
    effective = [
        result.torque_nm if isfinite(result.torque_nm) else FAILED_TORQUE_FACTOR * goal.max_torque_nm
        for result in results
    ]
    # Худший момент плюс небольшая доля среднего: при равных худших
    # предпочтительнее вариант, которому легче даются остальные препятствия.
    cost = max(effective, default=0.0) + 0.1 * float(np.mean(effective)) if effective else 0.0
    return Evaluation(design, [], goal.mode, score, cost, values, results)


def _invalid(design: Design, platform: Platform, goal: Goal, errors: list[str]) -> Evaluation:
    return Evaluation(design, errors, goal.mode, cost=INVALID_COST + violation_amount(design, platform))


def evaluate(design: Design, platform: Platform, goal: Goal) -> Evaluation:
    """Последовательная оценка одного варианта."""
    errors = validate(design, platform)
    if errors:
        return _invalid(design, platform, goal, errors)
    return summarize(design, goal, [run_job(design, platform, goal, job) for job in jobs_for(goal)])


def evaluate_many(
    pool: ProcessPoolExecutor,
    designs: list[Design],
    platform: Platform,
    goal: Goal,
    guesses: dict[Job, float] | None = None,
    progress: Callable[[], None] | None = None,
) -> list[Evaluation]:
    """Оценивает варианты; все пары «вариант × препятствие» идут в пул сразу."""
    jobs = jobs_for(goal)
    guesses = guesses or {}
    evaluations: list[Evaluation | None] = [None] * len(designs)
    futures = {}
    for index, design in enumerate(designs):
        errors = validate(design, platform)
        if errors:
            evaluations[index] = _invalid(design, platform, goal, errors)
            continue
        for slot, job in enumerate(jobs):
            future = pool.submit(run_job, design, platform, goal, job, guesses.get(job))
            futures[future] = (index, slot)
    collected: dict[int, list[JobResult | None]] = {}
    for future in as_completed(futures):
        index, slot = futures[future]
        collected.setdefault(index, [None] * len(jobs))[slot] = future.result()
        if progress:
            progress()
    for index, results in collected.items():
        evaluations[index] = summarize(designs[index], goal, results)
    return evaluations


def evaluate_parallel(
    design: Design,
    platform: Platform,
    goal: Goal,
    progress: Callable[[int, int], None] | None = None,
) -> Evaluation:
    """Оценка одного варианта: каждая граница считается в своём процессе."""
    total = len(jobs_for(goal))
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if progress:
            progress(done, total)

    with ProcessPoolExecutor(max_workers=min(total, os.cpu_count() or 1)) as pool:
        return evaluate_many(pool, [design], platform, goal, progress=tick)[0]


def _design_from_vector(base: Design, names: list[str], vector: np.ndarray) -> Design:
    return replace(base, **{name: float(round(value, 1)) for name, value in zip(names, vector)})


def _guesses(goal: Goal, evaluation: Evaluation | None) -> dict[Job, float]:
    if evaluation is None or not evaluation.valid:
        return {}
    guesses = {}
    for job, result in zip(jobs_for(goal), evaluation.results):
        value = result_value(result)
        if isfinite(value) and 0.0 < value < (job.size_mm if goal.mode == "limits" else goal.max_torque_nm):
            guesses[job] = value
    return guesses


@dataclass
class Generation:
    index: int
    best_score: float
    median_score: float
    valid_share: float
    sigma: float


@dataclass
class OptimizationResult:
    best: Evaluation | None
    evaluations: list[Evaluation]
    goal: Goal
    history: list[Generation] = field(default_factory=list)

    def ranked(self, count: int = 10) -> list[Evaluation]:
        """Лучшие уникальные варианты; уточнённая оценка заменяет грубую."""
        latest: dict[tuple, Evaluation] = {}
        for item in self.evaluations:
            if item.valid:
                key = tuple(asdict(item.design).values())
                if item.refined or key not in latest or not latest[key].refined:
                    latest[key] = item
        ordered = sorted(latest.values(), key=lambda item: (not item.refined, item.cost))
        return ordered[:count]


def default_population(free_count: int) -> int:
    return 4 + int(3 * np.log(max(free_count, 1)))


def optimize(
    base: Design,
    platform: Platform,
    goal: Goal,
    free: dict[str, tuple[float, float]],
    population: int | None = None,
    generations: int = 25,
    seed: int = 1,
    refine: int = 3,
    progress: Callable[[int, int, Evaluation | None], None] | None = None,
) -> OptimizationResult:
    """CMA-ES по свободным параметрам; base — первая точка (центр поиска)."""
    names = list(free)
    if not names:
        raise ValueError("Не выбрано ни одного свободного параметра.")
    lower = np.asarray([free[name][0] for name in names], dtype=float)
    upper = np.asarray([free[name][1] for name in names], dtype=float)
    span = upper - lower
    start = np.asarray([getattr(base, name) for name in names], dtype=float)
    x0 = np.clip((start - lower) / span, 0.0, 1.0)
    population = max(4, population or default_population(len(names)))

    strategy = cma.CMAEvolutionStrategy(
        x0, 0.3, {"bounds": [0.0, 1.0], "popsize": population, "seed": seed + 1, "verbose": -9}
    )
    strategy.inject([x0], force=True)
    jobs = len(jobs_for(goal))
    total = (generations * population + refine) * jobs
    done = 0
    best: Evaluation | None = None
    evaluations: list[Evaluation] = []
    history: list[Generation] = []

    def tick() -> None:
        nonlocal done
        done += 1
        if progress:
            progress(min(done, total), total, best)

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        for generation in range(generations):
            vectors = strategy.ask()
            designs = [_design_from_vector(base, names, lower + span * np.asarray(x)) for x in vectors]
            batch = evaluate_many(pool, designs, platform, goal, _guesses(goal, best), tick)
            strategy.tell(vectors, [item.cost for item in batch])
            evaluations.extend(batch)
            valid = [item for item in batch if item.valid]
            for item in valid:
                if best is None or item.cost < best.cost:
                    best = item
            history.append(Generation(
                generation + 1,
                best.score if best else float("nan"),
                float(np.median([item.score for item in valid])) if valid else float("nan"),
                len(valid) / len(batch),
                float(strategy.sigma),
            ))
            # Пропуски недопустимых вариантов не считаются — сдвигаем счётчик.
            done = (generation + 1) * population * jobs
            if progress:
                progress(done, total, best)

        # Уточнение: несколько лучших вариантов с более мелким шагом поиска.
        top = OptimizationResult(None, evaluations, goal).ranked(refine)
        if top:
            fine_goal = goal.refined()
            refined = evaluate_many(pool, [item.design for item in top], platform, fine_goal,
                                    _guesses(goal, best), tick)
            for item in refined:
                item.refined = True
            evaluations.extend(refined)

    result = OptimizationResult(None, evaluations, goal, history)
    ranked = result.ranked(1)
    result.best = ranked[0] if ranked else None
    return result


def criterion_key(kind: str, width_mm: float) -> str:
    """Ключ цели в Evaluation.values: вид препятствия, у кочки — с шириной."""
    return f"bump_{width_mm:.0f}" if kind == "bump" else kind


def criterion_label(key: str) -> str:
    if key.startswith("bump_"):
        return f"{KIND_LABELS['bump']} {key.removeprefix('bump_')} мм"
    return KIND_LABELS[key]
