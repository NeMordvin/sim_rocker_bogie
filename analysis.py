"""Проверка препятствий и последовательная оптимизация подвески.

Для каждого препятствия определяются два предела:

* геометрический — существует непрерывный путь подвески без пересечений
  с препятствием и с соблюдением диапазонов шарниров;
* физический — вдобавок в каждом кадре выполняются уравнения равновесия
  с ограничениями сцепления, момента приводов и устойчивости.

Физический предел ищется только внутри геометрического, поэтому всегда
не превышает его.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable, Protocol, TypeVar

from geometry import (
    ChassisDesign,
    DitchTerrain,
    ObstacleKind,
    SolverSettings,
    StepTerrain,
    TraversalResult,
    latin_hypercube_designs,
    neutral_dimensions,
    simulate_traversal,
    validate_design,
)
from statics import (
    PhysicalParameters,
    Scenario,
    TraversalStatics,
    analyze_traversal,
)


DEFAULT_SCENARIOS: tuple[Scenario, ...] = ("full_width", "one_side")


@dataclass(frozen=True)
class CheckedTraversal:
    success: bool
    traversal: TraversalResult
    statics: tuple[TraversalStatics, ...]
    failure_reason: str = ""


@dataclass(frozen=True)
class GeometricLimitResult:
    value_mm: float
    traversal: TraversalResult
    reached_search_cap: bool
    failure_reason: str = ""


@dataclass(frozen=True)
class PhysicalCapabilityResult:
    value_mm: float
    checked: CheckedTraversal | None
    reached_search_cap: bool
    geometric: GeometricLimitResult | None = None
    # Причина отказа при ближайшем недопустимом значении (пусто, если
    # предел упёрся в границу поиска или совпал с геометрическим).
    failure_reason: str = ""

    @property
    def geometric_mm(self) -> float:
        return self.geometric.value_mm if self.geometric is not None else self.value_mm


@dataclass(frozen=True)
class StepDesignEvaluation:
    design: ChassisDesign
    single_first: PhysicalCapabilityResult
    bogie_first: PhysicalCapabilityResult
    refined: bool = False

    @property
    def bidirectional_step_mm(self) -> float:
        return min(self.single_first.value_mm, self.bogie_first.value_mm)

    @property
    def bidirectional_geometric_step_mm(self) -> float:
        return min(self.single_first.geometric_mm, self.bogie_first.geometric_mm)

    @property
    def tie_break_key(self) -> tuple[float, float, float]:
        dimensions = neutral_dimensions(self.design)
        return (
            self.bidirectional_step_mm,
            self.bidirectional_geometric_step_mm,
            -dimensions["total_length_mm"],
        )


@dataclass(frozen=True)
class SequentialOptimizationResult:
    ranked_step_designs: tuple[StepDesignEvaluation, ...]
    best_step_design: StepDesignEvaluation | None
    ditch_single_first: PhysicalCapabilityResult | None
    ditch_bogie_first: PhysicalCapabilityResult | None

    @property
    def bidirectional_ditch_mm(self) -> float:
        if self.ditch_single_first is None or self.ditch_bogie_first is None:
            return 0.0
        return min(
            self.ditch_single_first.value_mm,
            self.ditch_bogie_first.value_mm,
        )


class _Checked(Protocol):
    @property
    def success(self) -> bool: ...


CheckedT = TypeVar("CheckedT", bound=_Checked)


def _terrain(kind: ObstacleKind, value_mm: float) -> StepTerrain | DitchTerrain:
    return StepTerrain(value_mm) if kind == "step" else DitchTerrain(value_mm)


@lru_cache(maxsize=512)
def _cached_traversal(
    design: ChassisDesign,
    obstacle_kind: ObstacleKind,
    value_mm: float,
    direction: int,
    settings: SolverSettings,
) -> TraversalResult:
    # Геометрический и физический поиск проверяют одни и те же высоты;
    # кэш избавляет от повторного решения кинематики.
    return simulate_traversal(
        design,
        _terrain(obstacle_kind, value_mm),
        direction=direction,
        settings=settings,
    )


def _bisect(
    check: Callable[[float], CheckedT],
    maximum_value_mm: float,
    iterations: int,
) -> tuple[float, CheckedT, CheckedT | None]:
    """Наибольшее значение из [0, maximum], для которого check успешен.

    Возвращает значение, результат проверки при нём и результат при
    ближайшем недопустимом значении (None, если успешна верхняя граница).
    """
    zero = check(0.0)
    if not zero.success:
        return 0.0, zero, zero
    if maximum_value_mm <= 0.0:
        return 0.0, zero, None
    top = check(maximum_value_mm)
    if top.success:
        return maximum_value_mm, top, None

    lower, upper, best, failed = 0.0, float(maximum_value_mm), zero, top
    for _ in range(max(1, iterations)):
        middle = (lower + upper) / 2.0
        checked = check(middle)
        if checked.success:
            lower, best = middle, checked
        else:
            upper, failed = middle, checked
    return lower, best, failed


def check_obstacle(
    design: ChassisDesign,
    physical: PhysicalParameters,
    obstacle_kind: ObstacleKind,
    value_mm: float,
    direction: int,
    settings: SolverSettings,
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
    static_sample_stride: int = 1,
) -> CheckedTraversal:
    traversal = _cached_traversal(
        design, obstacle_kind, float(value_mm), direction, settings
    )
    if not traversal.success:
        return CheckedTraversal(
            False,
            traversal,
            (),
            f"Геометрия: {traversal.failure_reason}",
        )

    calculated: list[TraversalStatics] = []
    for scenario in scenarios:
        result = analyze_traversal(
            design,
            physical,
            traversal,
            scenario,
            sample_stride=static_sample_stride,
        )
        calculated.append(result)
        if not result.success:
            return CheckedTraversal(
                False,
                traversal,
                tuple(calculated),
                f"{scenario}: {result.failure_reason}",
            )
    return CheckedTraversal(True, traversal, tuple(calculated))


def find_geometric_limit(
    design: ChassisDesign,
    obstacle_kind: ObstacleKind,
    direction: int,
    maximum_value_mm: float,
    settings: SolverSettings,
    iterations: int = 7,
) -> GeometricLimitResult:
    if maximum_value_mm <= 0.0:
        raise ValueError("Верхняя граница поиска должна быть больше нуля.")
    value, traversal, failed = _bisect(
        lambda value_mm: _cached_traversal(
            design, obstacle_kind, float(value_mm), direction, settings
        ),
        maximum_value_mm,
        iterations,
    )
    return GeometricLimitResult(
        value,
        traversal,
        failed is None,
        failed.failure_reason if failed is not None else "",
    )


def find_physical_limit(
    design: ChassisDesign,
    physical: PhysicalParameters,
    obstacle_kind: ObstacleKind,
    direction: int,
    maximum_value_mm: float,
    settings: SolverSettings,
    iterations: int = 7,
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
    static_sample_stride: int = 1,
) -> PhysicalCapabilityResult:
    """Геометрический предел, затем физический внутри него."""
    geometric = find_geometric_limit(
        design, obstacle_kind, direction, maximum_value_mm, settings, iterations
    )
    value, checked, failed = _bisect(
        lambda value_mm: check_obstacle(
            design,
            physical,
            obstacle_kind,
            value_mm,
            direction,
            settings,
            scenarios,
            static_sample_stride,
        ),
        geometric.value_mm,
        iterations,
    )
    if failed is not None:
        reason = failed.failure_reason
    elif not geometric.reached_search_cap:
        reason = f"Геометрия: {geometric.failure_reason}"
    else:
        reason = ""
    return PhysicalCapabilityResult(
        value,
        checked,
        failed is None and geometric.reached_search_cap,
        geometric,
        reason,
    )


CapabilityJob = tuple[ObstacleKind, int, float]


def find_limits(
    design: ChassisDesign,
    physical: PhysicalParameters,
    jobs: dict[str, CapabilityJob],
    settings: SolverSettings,
    iterations: int = 7,
    parallel: bool = True,
) -> dict[str, PhysicalCapabilityResult]:
    """Считает несколько пределов (вид, направление, граница поиска) сразу.

    Пределы независимы, поэтому при parallel=True каждый считается в
    отдельном процессе.
    """
    arguments = {
        key: (design, physical, kind, direction, maximum, settings, iterations)
        for key, (kind, direction, maximum) in jobs.items()
    }
    if not parallel or len(jobs) < 2 or (os.cpu_count() or 1) < 2:
        return {key: find_physical_limit(*args) for key, args in arguments.items()}
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        futures = {key: pool.submit(find_physical_limit, *args) for key, args in arguments.items()}
        return {key: future.result() for key, future in futures.items()}


def evaluate_step_design(
    design: ChassisDesign,
    physical: PhysicalParameters,
    maximum_step_mm: float,
    settings: SolverSettings,
    iterations: int,
    static_sample_stride: int,
    refined: bool = False,
) -> StepDesignEvaluation:
    single, bogie = (
        find_physical_limit(
            design,
            physical,
            "step",
            direction,
            maximum_step_mm,
            settings,
            iterations,
            static_sample_stride=static_sample_stride,
        )
        for direction in (1, -1)
    )
    return StepDesignEvaluation(design, single, bogie, refined)


def _evaluate_many(
    designs: list[ChassisDesign],
    stage: str,
    progress_callback: Callable[[str, int, int], None] | None,
    parallel: bool,
    **evaluation: object,
) -> list[StepDesignEvaluation]:
    """Оценивает варианты; при parallel=True — в отдельных процессах.

    Варианты независимы, поэтому параллельный расчёт ускоряет перебор почти
    пропорционально числу ядер. Порядок результатов совпадает с designs.
    """
    total = len(designs)
    results: list[StepDesignEvaluation | None] = [None] * total
    workers = min(total, os.cpu_count() or 1)
    if parallel and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(evaluate_step_design, design, **evaluation): index
                for index, design in enumerate(designs)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if progress_callback is not None:
                    progress_callback(stage, done, total)
    else:
        for index, design in enumerate(designs):
            results[index] = evaluate_step_design(design, **evaluation)
            if progress_callback is not None:
                progress_callback(stage, index + 1, total)
    return [item for item in results if item is not None]


def _pattern_search(
    start: StepDesignEvaluation,
    free_ranges: dict[str, tuple[float, float]],
    rounds: int,
    progress_callback: Callable[[str, int, int], None] | None,
    parallel: bool,
    **evaluation: object,
) -> list[StepDesignEvaluation]:
    """Покоординатный поиск от лучшего варианта случайного перебора.

    На каждом шаге параллельно проверяются сдвиги каждой свободной
    координаты в обе стороны. Лучший сосед принимается, если он улучшает
    результат; иначе шаг уменьшается вдвое. Случайный перебор находит
    область оптимума, а этот этап доводит, например, положение главного
    шарнира до равномерной нагрузки колёс, от которой зависит сцепление.
    """
    current = start
    steps = {name: 0.2 * (high - low) for name, (low, high) in free_ranges.items()}
    evaluated: list[StepDesignEvaluation] = []
    for round_index in range(rounds):
        neighbours: list[ChassisDesign] = []
        for name, (low, high) in free_ranges.items():
            for sign in (1.0, -1.0):
                value = min(high, max(low, getattr(current.design, name) + sign * steps[name]))
                candidate = replace(current.design, **{name: value})
                if candidate != current.design and not validate_design(candidate):
                    neighbours.append(candidate)
        if not neighbours:
            break
        results = _evaluate_many(
            neighbours,
            f"Локальный поиск, шаг {round_index + 1} из {rounds}",
            progress_callback,
            parallel,
            **evaluation,
        )
        evaluated.extend(results)
        best = max(results, key=lambda item: item.tie_break_key)
        if best.tie_break_key > current.tie_break_key:
            current = best
        else:
            steps = {name: step / 2.0 for name, step in steps.items()}
    return evaluated


def optimize_step_then_calculate_ditch(
    base: ChassisDesign,
    physical: PhysicalParameters,
    free_ranges: dict[str, tuple[float, float]],
    count: int,
    seed: int,
    maximum_step_mm: float,
    maximum_ditch_mm: float,
    progress_callback: Callable[[str, int, int], None] | None = None,
    parallel: bool = True,
    local_search_rounds: int = 6,
) -> SequentialOptimizationResult:
    """Сначала максимизирует физическую ступень, потом считает ров лучшего варианта.

    Этапы: случайный перебор (латинский гиперкуб) → покоординатный поиск от
    лучшего варианта → точная проверка трёх лучших → ров победителя.
    """

    coarse_evaluation = dict(
        physical=physical,
        maximum_step_mm=maximum_step_mm,
        settings=SolverSettings.coarse(),
        iterations=7,
        static_sample_stride=3,
    )
    designs = latin_hypercube_designs(base, free_ranges, count, seed)
    quick = _evaluate_many(
        designs,
        "Случайный перебор по ступени",
        progress_callback,
        parallel,
        **coarse_evaluation,
    )
    if not quick:
        return SequentialOptimizationResult((), None, None, None)
    quick.extend(
        _pattern_search(
            max(quick, key=lambda item: item.tie_break_key),
            free_ranges,
            local_search_rounds,
            progress_callback,
            parallel,
            **coarse_evaluation,
        )
    )
    quick = list({item.design: item for item in quick}.values())
    quick.sort(key=lambda item: item.tie_break_key, reverse=True)

    refined = _evaluate_many(
        [evaluation.design for evaluation in quick[:3]],
        "Точная проверка лучших вариантов",
        progress_callback,
        parallel,
        physical=physical,
        maximum_step_mm=maximum_step_mm,
        settings=SolverSettings.fine(),
        iterations=7,
        static_sample_stride=1,
        refined=True,
    )
    refined.sort(key=lambda item: item.tie_break_key, reverse=True)
    best = refined[0]
    # Грубая оценка обычно завышена, поэтому точно проверенные варианты
    # идут первыми, а остальные — после них в порядке грубой оценки.
    refined_designs = {evaluation.design for evaluation in refined}
    ranked = refined + [item for item in quick if item.design not in refined_designs]

    ditches: list[PhysicalCapabilityResult] = []
    for done, direction in enumerate((1, -1)):
        if progress_callback is not None:
            progress_callback("Расчёт рва для лучшей геометрии", done, 2)
        ditches.append(
            find_physical_limit(
                best.design,
                physical,
                "ditch",
                direction,
                maximum_ditch_mm,
                SolverSettings.fine(),
                iterations=7,
                static_sample_stride=1,
            )
        )
    if progress_callback is not None:
        progress_callback("Расчёт рва для лучшей геометрии", 2, 2)

    return SequentialOptimizationResult(tuple(ranked), best, ditches[0], ditches[1])
