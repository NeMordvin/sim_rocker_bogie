"""Подбор геометрии рычагов по пределам препятствий.

Оценка варианта — взвешенная сумма пределов (мм) по выбранным видам
препятствий. Для каждого вида берётся худшее из двух направлений движения
(и худший из сценариев «по всей ширине» / «под одним бортом», если он
включён). Поиск — дифференциальная эволюция SciPy; варианты одного поколения
считаются параллельно в отдельных процессах.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from model import Design, Platform, validate
from simulate import KIND_LABELS, LimitResult, ObstacleKind, find_limit


@dataclass(frozen=True)
class Criterion:
    kind: ObstacleKind
    weight: float
    upper_mm: float


@dataclass(frozen=True)
class Goal:
    criteria: tuple[Criterion, ...] = (
        Criterion("step_up", 1.0, 300.0),
        Criterion("step_down", 0.3, 500.0),
        Criterion("ditch", 0.5, 500.0),
    )
    one_side: bool = False
    tolerance_mm: float = 10.0
    ditch_depth_mm: float = 500.0

    @property
    def active(self) -> tuple[Criterion, ...]:
        return tuple(item for item in self.criteria if item.weight > 0.0)


@dataclass(frozen=True)
class Job:
    kind: ObstacleKind
    direction: int
    one_side: bool
    upper_mm: float


@dataclass
class Evaluation:
    design: Design
    errors: list[str]
    score: float = 0.0
    values: dict[str, float] = field(default_factory=dict)
    limits: list[LimitResult] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def jobs_for(goal: Goal) -> list[Job]:
    sides = (False, True) if goal.one_side else (False,)
    return [
        Job(item.kind, direction, one_side, item.upper_mm)
        for item in goal.active
        for one_side in sides
        for direction in (1, -1)
    ]


def run_job(design: Design, platform: Platform, goal: Goal, job: Job) -> LimitResult:
    return find_limit(
        design,
        platform,
        job.kind,
        job.direction,
        one_side=job.one_side,
        upper_mm=job.upper_mm,
        tolerance_mm=goal.tolerance_mm,
        ditch_depth_mm=goal.ditch_depth_mm,
    )


def summarize(design: Design, goal: Goal, limits: list[LimitResult]) -> Evaluation:
    values: dict[str, float] = {}
    score = 0.0
    for item in goal.active:
        relevant = [limit.value_mm for limit in limits if limit.obstacle.kind == item.kind]
        worst = min(relevant) if relevant else 0.0
        values[item.kind] = worst
        score += item.weight * worst
    return Evaluation(design, [], score, values, limits)


def evaluate(design: Design, platform: Platform, goal: Goal) -> Evaluation:
    """Последовательная оценка одного варианта (используется в процессах)."""
    errors = validate(design, platform)
    if errors:
        return Evaluation(design, errors)
    limits = [run_job(design, platform, goal, job) for job in jobs_for(goal)]
    return summarize(design, goal, limits)


def evaluate_parallel(
    design: Design,
    platform: Platform,
    goal: Goal,
    progress: Callable[[int, int], None] | None = None,
) -> Evaluation:
    """Оценка одного варианта: каждый предел считается в своём процессе."""
    errors = validate(design, platform)
    if errors:
        return Evaluation(design, errors)
    jobs = jobs_for(goal)
    results: dict[int, LimitResult] = {}
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        futures = {pool.submit(run_job, design, platform, goal, job): index for index, job in enumerate(jobs)}
        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if progress:
                progress(done, len(jobs))
    return summarize(design, goal, [results[index] for index in range(len(jobs))])


def _design_from_vector(base: Design, names: list[str], vector: np.ndarray) -> Design:
    return replace(base, **{name: float(round(value, 1)) for name, value in zip(names, vector)})


@dataclass
class OptimizationResult:
    best: Evaluation | None
    evaluations: list[Evaluation]
    goal: Goal

    def ranked(self, count: int = 10) -> list[Evaluation]:
        valid = [item for item in self.evaluations if item.valid]
        valid.sort(key=lambda item: item.score, reverse=True)
        unique: list[Evaluation] = []
        seen = set()
        for item in valid:
            key = tuple(asdict(item.design).values())
            if key not in seen:
                seen.add(key)
                unique.append(item)
            if len(unique) >= count:
                break
        return unique


class _ParallelMap:
    """Заменяет map для differential_evolution и сохраняет все оценки."""

    def __init__(self, pool, base, names, platform, goal, total, progress):
        self.pool = pool
        self.base = base
        self.names = names
        self.platform = platform
        self.goal = goal
        self.total = total
        self.progress = progress
        self.evaluations: list[Evaluation] = []

    def __call__(self, _func, iterable):
        designs = [_design_from_vector(self.base, self.names, np.asarray(x)) for x in iterable]
        futures = {
            self.pool.submit(evaluate, design, self.platform, self.goal): index
            for index, design in enumerate(designs)
        }
        results: list[Evaluation | None] = [None] * len(designs)
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            self.evaluations.append(results[futures[future]])
            if self.progress:
                self.progress(min(len(self.evaluations), self.total), self.total)
        # Недопустимая геометрия хуже любой допустимой (у допустимой −score ≤ 0).
        return [(-item.score if item.valid else 1.0) for item in results]


def optimize(
    base: Design,
    platform: Platform,
    goal: Goal,
    free: dict[str, tuple[float, float]],
    population: int = 24,
    generations: int = 8,
    seed: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> OptimizationResult:
    names = list(free)
    if not names:
        raise ValueError("Не выбрано ни одного свободного параметра.")
    lower = np.asarray([free[name][0] for name in names])
    upper = np.asarray([free[name][1] for name in names])
    population = max(5, population)

    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    init = qmc.scale(sampler.random(population), lower, upper)
    start = np.asarray([getattr(base, name) for name in names], dtype=float)
    init[0] = np.clip(start, lower, upper)

    total = population * (generations + 1)
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        mapper = _ParallelMap(pool, base, names, platform, goal, total, progress)
        differential_evolution(
            lambda x: 0.0,
            bounds=list(zip(lower, upper)),
            init=init,
            maxiter=generations,
            seed=seed,
            workers=mapper,
            updating="deferred",
            polish=False,
            tol=0.0,
            mutation=(0.5, 1.0),
            recombination=0.7,
        )
    result = OptimizationResult(None, mapper.evaluations, goal)
    ranked = result.ranked(1)
    result.best = ranked[0] if ranked else None
    return result


def criterion_label(kind: str) -> str:
    return KIND_LABELS[kind]
