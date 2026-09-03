"""Проверка препятствий и последовательная оптимизация подвески."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
class PhysicalCapabilityResult:
    value_mm: float
    checked: CheckedTraversal | None
    reached_search_cap: bool


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
    def tie_break_key(self) -> tuple[float, float]:
        dimensions = neutral_dimensions(self.design)
        return self.bidirectional_step_mm, -dimensions["total_length_mm"]


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


def _terrain(kind: ObstacleKind, value_mm: float) -> StepTerrain | DitchTerrain:
    return StepTerrain(value_mm) if kind == "step" else DitchTerrain(value_mm)


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
    traversal = simulate_traversal(
        design,
        _terrain(obstacle_kind, value_mm),
        direction=direction,
        settings=settings,
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
    if maximum_value_mm <= 0.0:
        raise ValueError("Верхняя граница поиска должна быть больше нуля.")

    zero = check_obstacle(
        design,
        physical,
        obstacle_kind,
        0.0,
        direction,
        settings,
        scenarios,
        static_sample_stride,
    )
    if not zero.success:
        return PhysicalCapabilityResult(0.0, zero, False)

    cap = check_obstacle(
        design,
        physical,
        obstacle_kind,
        maximum_value_mm,
        direction,
        settings,
        scenarios,
        static_sample_stride,
    )
    if cap.success:
        return PhysicalCapabilityResult(maximum_value_mm, cap, True)

    lower = 0.0
    upper = float(maximum_value_mm)
    best = zero
    for _ in range(max(1, iterations)):
        middle = (lower + upper) / 2.0
        checked = check_obstacle(
            design,
            physical,
            obstacle_kind,
            middle,
            direction,
            settings,
            scenarios,
            static_sample_stride,
        )
        if checked.success:
            lower = middle
            best = checked
        else:
            upper = middle

    # Пересчёт найденного предела нужен, чтобы вернуть полный набор кадров.
    final = check_obstacle(
        design,
        physical,
        obstacle_kind,
        lower,
        direction,
        settings,
        scenarios,
        static_sample_stride,
    )
    if final.success:
        best = final
    return PhysicalCapabilityResult(lower, best, False)


def evaluate_step_design(
    design: ChassisDesign,
    physical: PhysicalParameters,
    maximum_step_mm: float,
    settings: SolverSettings,
    iterations: int,
    static_sample_stride: int,
    refined: bool = False,
) -> StepDesignEvaluation:
    return StepDesignEvaluation(
        design=design,
        single_first=find_physical_limit(
            design,
            physical,
            "step",
            1,
            maximum_step_mm,
            settings,
            iterations,
            static_sample_stride=static_sample_stride,
        ),
        bogie_first=find_physical_limit(
            design,
            physical,
            "step",
            -1,
            maximum_step_mm,
            settings,
            iterations,
            static_sample_stride=static_sample_stride,
        ),
        refined=refined,
    )


def optimize_step_then_calculate_ditch(
    base: ChassisDesign,
    physical: PhysicalParameters,
    free_ranges: dict[str, tuple[float, float]],
    count: int,
    seed: int,
    maximum_step_mm: float,
    maximum_ditch_mm: float,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> SequentialOptimizationResult:
    """Сначала максимизирует ступень, потом считает ров лучшего варианта."""

    designs = latin_hypercube_designs(base, free_ranges, count, seed)
    quick: list[StepDesignEvaluation] = []
    total = len(designs)
    for index, design in enumerate(designs, start=1):
        quick.append(
            evaluate_step_design(
                design,
                physical,
                maximum_step_mm,
                SolverSettings.coarse(),
                iterations=4,
                static_sample_stride=5,
            )
        )
        if progress_callback is not None:
            progress_callback("Быстрый поиск по ступени", index, total)

    quick.sort(key=lambda item: item.tie_break_key, reverse=True)
    if not quick:
        return SequentialOptimizationResult((), None, None, None)

    refine_count = min(3, len(quick))
    refined: list[StepDesignEvaluation] = []
    for index, evaluation in enumerate(quick[:refine_count], start=1):
        refined.append(
            evaluate_step_design(
                evaluation.design,
                physical,
                maximum_step_mm,
                SolverSettings.fine(),
                iterations=6,
                static_sample_stride=1,
                refined=True,
            )
        )
        if progress_callback is not None:
            progress_callback("Точная проверка лучших вариантов", index, refine_count)

    refined.sort(key=lambda item: item.tie_break_key, reverse=True)
    best = refined[0]
    refined_by_design = {evaluation.design: evaluation for evaluation in refined}
    ranked = [refined_by_design.get(item.design, item) for item in quick]
    ranked.sort(key=lambda item: item.tie_break_key, reverse=True)

    if progress_callback is not None:
        progress_callback("Расчёт рва для лучшей геометрии", 0, 2)
    ditch_single = find_physical_limit(
        best.design,
        physical,
        "ditch",
        1,
        maximum_ditch_mm,
        SolverSettings.fine(),
        iterations=7,
        static_sample_stride=1,
    )
    if progress_callback is not None:
        progress_callback("Расчёт рва для лучшей геометрии", 1, 2)
    ditch_bogie = find_physical_limit(
        best.design,
        physical,
        "ditch",
        -1,
        maximum_ditch_mm,
        SolverSettings.fine(),
        iterations=7,
        static_sample_stride=1,
    )
    if progress_callback is not None:
        progress_callback("Расчёт рва для лучшей геометрии", 2, 2)

    return SequentialOptimizationResult(
        tuple(ranked),
        best,
        ditch_single,
        ditch_bogie,
    )

