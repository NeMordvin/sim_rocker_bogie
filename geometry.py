"""Кинематическая модель подвески rocker-bogie.

Модуль строит непрерывный геометрический путь одной стороны подвески. Корпус
не входит в геометрические ограничения: учитываются только колёса, рычаги,
шарниры и препятствие. Квазистатические силы, устойчивость двух сторон и
классический дифференциал рассчитываются отдельно в :mod:`statics`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from math import asin, atan2, cos, degrees, hypot, pi, radians, sin, sqrt
from typing import Callable, Literal, Protocol

import numpy as np
from scipy.optimize import least_squares


ObstacleKind = Literal["step", "ditch"]

WHEEL_NAMES = ("front", "middle", "rear")
WHEEL_LABELS = {
    "front": "одиночное",
    "middle": "среднее bogie",
    "rear": "заднее bogie",
}


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class ChassisDesign:
    """Размеры шасси в исходном положении на горизонтальной поверхности.

    Координата X заднего колеса всегда равна нулю. Центры всех трёх колёс
    в исходном положении имеют одинаковую координату Y.
    """

    # Мотор-колесо Xiaomi M365 Pro: шина 8,5″, наружный диаметр ≈216 мм.
    wheel_radius_mm: float = 108.0
    middle_wheel_x_mm: float = 360.0
    front_wheel_x_mm: float = 900.0

    main_pivot_x_mm: float = 610.0
    main_pivot_y_mm: float = 310.0
    bogie_pivot_x_mm: float = 180.0
    bogie_pivot_y_mm: float = 230.0

    # Две верхние точки задают П-образную форму косвенно-углового bogie.
    bogie_rear_knee_x_mm: float = 95.0
    bogie_front_knee_x_mm: float = 285.0

    # Расчётный диаметр всех рычагов. Он используется как геометрическая
    # оболочка при проверке касания препятствия, а не как расчёт прочности.
    arm_diameter_mm: float = 30.0

    # Общий центр масс задаётся относительно середины оси главных шарниров:
    # X — вперёд к одиночному колесу, Z — вверх. Поперечная координата
    # хранится в PhysicalParameters, поскольку относится ко всему роверу.
    com_x_from_main_pivot_mm: float = 0.0
    com_z_from_main_pivot_mm: float = 130.0

    bogie_min_angle_deg: float = -80.0
    bogie_max_angle_deg: float = 80.0
    rocker_min_angle_deg: float = -65.0
    rocker_max_angle_deg: float = 65.0

    obstacle_clearance_mm: float = 3.0
    self_clearance_mm: float = 3.0
    max_total_length_mm: float = 1200.0


@dataclass(frozen=True)
class SolverSettings:
    """Точность и скорость дискретного поиска непрерывного пути."""

    progress_step_mm: float = 14.0
    angle_step_deg: float = 1.5
    pitch_window_deg: float = 9.0
    bogie_window_deg: float = 13.5
    contact_tolerance_mm: float = 4.0
    beam_width: int = 18
    structure_samples_per_segment: int = 7
    maximum_pitch_change_deg: float = 10.5
    maximum_bogie_change_deg: float = 15.0
    maximum_vertical_change_mm: float = 55.0
    minimum_progress_step_mm: float = 3.0
    validate_transition_midpoint: bool = False

    @classmethod
    def coarse(cls) -> "SolverSettings":
        """Быстрые настройки для перебора вариантов во время оптимизации."""
        return cls(
            progress_step_mm=25.0,
            angle_step_deg=3.0,
            pitch_window_deg=12.0,
            bogie_window_deg=18.0,
            contact_tolerance_mm=8.0,
            beam_width=10,
            structure_samples_per_segment=7,
            maximum_pitch_change_deg=15.0,
            maximum_bogie_change_deg=21.0,
            maximum_vertical_change_mm=90.0,
            minimum_progress_step_mm=6.25,
            validate_transition_midpoint=False,
        )

    @classmethod
    def fine(cls) -> "SolverSettings":
        """Более точные настройки для итогового расчёта и ключевых кадров."""
        return cls(
            progress_step_mm=16.0,
            angle_step_deg=2.0,
            pitch_window_deg=10.0,
            bogie_window_deg=14.0,
            contact_tolerance_mm=5.0,
            beam_width=14,
            structure_samples_per_segment=7,
            maximum_pitch_change_deg=7.0,
            maximum_bogie_change_deg=10.0,
            maximum_vertical_change_mm=45.0,
            minimum_progress_step_mm=2.0,
            validate_transition_midpoint=False,
        )


@dataclass(frozen=True)
class Pose:
    main_pivot_x_mm: float
    main_pivot_y_mm: float
    body_pitch_deg: float
    bogie_angle_deg: float
    support_margin_mm: float
    contacts: tuple[str, ...]
    wall_contacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChassisFrame:
    main_pivot: Point
    front_wheel: Point
    bogie_pivot: Point
    middle_wheel: Point
    rear_wheel: Point
    bogie_rear_knee: Point
    bogie_front_knee: Point
    body_corners: tuple[Point, ...]
    center_of_mass: Point

    @property
    def wheels(self) -> tuple[Point, Point, Point]:
        return self.front_wheel, self.middle_wheel, self.rear_wheel


@dataclass(frozen=True)
class TraversalResult:
    success: bool
    obstacle_kind: ObstacleKind
    obstacle_value_mm: float
    direction: int
    poses: tuple[Pose, ...]
    minimum_support_margin_mm: float
    failure_reason: str = ""


class Terrain(Protocol):
    kind: ObstacleKind
    value_mm: float

    def clearance_y(
        self,
        x: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """Минимальная Y центра окружности, не пересекающей грунт."""

    def support_x(
        self,
        x: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """X точки опоры для положения на верхней границе грунта."""


@dataclass(frozen=True)
class StepTerrain:
    height_mm: float
    kind: ObstacleKind = "step"

    @property
    def value_mm(self) -> float:
        return self.height_mm

    def clearance_y(self, x: np.ndarray, radius: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        height = self.height_mm

        if radius <= 1e-12:
            return np.where(x < 0.0, 0.0, height)

        result = np.full_like(x, height + radius)
        lower_mask = x <= -radius
        result[lower_mask] = radius

        corner_mask = (x > -radius) & (x < 0.0)
        if np.any(corner_mask):
            corner_x = x[corner_mask]
            arc = height + np.sqrt(
                np.maximum(0.0, radius * radius - corner_x * corner_x)
            )
            # Для невысокой ступени колесо сначала продолжает катиться по низу.
            result[corner_mask] = np.maximum(radius, arc)
        return result

    def support_x(self, x: np.ndarray, radius: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        support = x.copy()
        if radius <= 1e-12:
            return support

        corner_mask = (x > -radius) & (x < 0.0)
        if np.any(corner_mask):
            corner_x = x[corner_mask]
            arc = self.height_mm + np.sqrt(
                np.maximum(0.0, radius * radius - corner_x * corner_x)
            )
            # На дуге колесо опирается непосредственно на верхнюю кромку.
            use_corner = arc > radius + 1e-8
            values = corner_x.copy()
            values[use_corner] = 0.0
            support[corner_mask] = values
        return support


@dataclass(frozen=True)
class DitchTerrain:
    width_mm: float
    kind: ObstacleKind = "ditch"

    @property
    def value_mm(self) -> float:
        return self.width_mm

    def clearance_y(self, x: np.ndarray, radius: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        width = self.width_mm
        result = np.full_like(x, -np.inf)

        if radius <= 1e-12:
            result[x <= 0.0] = 0.0
            result[x >= width] = 0.0
            return result

        left_top = x <= 0.0
        result[left_top] = radius
        left_corner = (x > 0.0) & (x < radius)
        result[left_corner] = np.maximum(
            result[left_corner],
            np.sqrt(np.maximum(0.0, radius * radius - x[left_corner] ** 2)),
        )

        right_top = x >= width
        result[right_top] = np.maximum(result[right_top], radius)
        right_corner = (x > width - radius) & (x < width)
        result[right_corner] = np.maximum(
            result[right_corner],
            np.sqrt(
                np.maximum(
                    0.0,
                    radius * radius - (x[right_corner] - width) ** 2,
                )
            ),
        )
        return result

    def support_x(self, x: np.ndarray, radius: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        width = self.width_mm
        support = x.copy()

        if radius <= 1e-12:
            return support

        left_corner = (x > 0.0) & (x < radius)
        support[left_corner] = 0.0
        right_corner = (x > width - radius) & (x < width)
        support[right_corner] = width

        # При перекрытии двух дуг выбирается ближайшая кромка. Одно колесо
        # всё равно не считается двумя независимыми продольными опорами.
        overlap = left_corner & right_corner
        if np.any(overlap):
            support[overlap] = np.where(
                x[overlap] <= width / 2.0,
                0.0,
                width,
            )
        return support


def _rotate(point: Point, angle_deg: float) -> Point:
    angle = radians(angle_deg)
    cosine = cos(angle)
    sine = sin(angle)
    return Point(
        point.x * cosine - point.y * sine,
        point.x * sine + point.y * cosine,
    )


def _add(first: Point, second: Point) -> Point:
    return Point(first.x + second.x, first.y + second.y)


def _mirror(point: Point, direction: int) -> Point:
    return Point(point.x * direction, point.y)


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return hypot(point.x - start.x, point.y - start.y)
    t = (
        (point.x - start.x) * dx + (point.y - start.y) * dy
    ) / length_squared
    t = min(1.0, max(0.0, t))
    closest = Point(start.x + t * dx, start.y + t * dy)
    return hypot(point.x - closest.x, point.y - closest.y)


def _design_local_points(
    design: ChassisDesign,
    direction: int,
) -> dict[str, Point]:
    pivot_x = design.main_pivot_x_mm
    pivot_y = design.main_pivot_y_mm
    bogie_x = design.bogie_pivot_x_mm
    bogie_y = design.bogie_pivot_y_mm

    points = {
        "front": Point(design.front_wheel_x_mm - pivot_x, -pivot_y),
        "bogie_pivot": Point(bogie_x - pivot_x, bogie_y - pivot_y),
        "middle_from_bogie": Point(
            design.middle_wheel_x_mm - bogie_x,
            -bogie_y,
        ),
        "rear_from_bogie": Point(-bogie_x, -bogie_y),
        "rear_knee_from_bogie": Point(
            design.bogie_rear_knee_x_mm - bogie_x,
            0.0,
        ),
        "front_knee_from_bogie": Point(
            design.bogie_front_knee_x_mm - bogie_x,
            0.0,
        ),
        "com": Point(
            design.com_x_from_main_pivot_mm,
            design.com_z_from_main_pivot_mm,
        ),
    }
    return {name: _mirror(point, direction) for name, point in points.items()}


def calculate_frame(
    design: ChassisDesign,
    pose: Pose,
    direction: int = 1,
) -> ChassisFrame:
    """Строит точки одной стороны подвески для одного положения."""
    local = _design_local_points(design, direction)
    main = Point(pose.main_pivot_x_mm, pose.main_pivot_y_mm)

    front = _add(main, _rotate(local["front"], pose.body_pitch_deg))
    bogie_pivot = _add(
        main,
        _rotate(local["bogie_pivot"], pose.body_pitch_deg),
    )
    bogie_absolute_angle = pose.body_pitch_deg + pose.bogie_angle_deg

    middle = _add(
        bogie_pivot,
        _rotate(local["middle_from_bogie"], bogie_absolute_angle),
    )
    rear = _add(
        bogie_pivot,
        _rotate(local["rear_from_bogie"], bogie_absolute_angle),
    )
    rear_knee = _add(
        bogie_pivot,
        _rotate(local["rear_knee_from_bogie"], bogie_absolute_angle),
    )
    front_knee = _add(
        bogie_pivot,
        _rotate(local["front_knee_from_bogie"], bogie_absolute_angle),
    )

    center_of_mass = _add(
        main,
        _rotate(local["com"], pose.body_pitch_deg),
    )

    return ChassisFrame(
        main_pivot=main,
        front_wheel=front,
        bogie_pivot=bogie_pivot,
        middle_wheel=middle,
        rear_wheel=rear,
        bogie_rear_knee=rear_knee,
        bogie_front_knee=front_knee,
        body_corners=(),
        center_of_mass=center_of_mass,
    )


def neutral_pose(design: ChassisDesign) -> Pose:
    """Положение, в котором все центры колёс находятся на одной высоте."""
    return Pose(
        main_pivot_x_mm=design.main_pivot_x_mm,
        main_pivot_y_mm=design.wheel_radius_mm + design.main_pivot_y_mm,
        body_pitch_deg=0.0,
        bogie_angle_deg=0.0,
        support_margin_mm=float("inf"),
        contacts=WHEEL_NAMES,
        wall_contacts=(),
    )


def neutral_dimensions(design: ChassisDesign) -> dict[str, float]:
    """Возвращает габариты и углы, полученные из координат."""
    pose = neutral_pose(design)
    frame = calculate_frame(design, pose)
    radius = design.wheel_radius_mm
    x_values = [
        frame.front_wheel.x - radius,
        frame.front_wheel.x + radius,
        frame.middle_wheel.x - radius,
        frame.middle_wheel.x + radius,
        frame.rear_wheel.x - radius,
        frame.rear_wheel.x + radius,
        frame.main_pivot.x - design.arm_diameter_mm / 2.0,
        frame.main_pivot.x + design.arm_diameter_mm / 2.0,
        frame.bogie_pivot.x - design.arm_diameter_mm / 2.0,
        frame.bogie_pivot.x + design.arm_diameter_mm / 2.0,
        frame.bogie_rear_knee.x - design.arm_diameter_mm / 2.0,
        frame.bogie_rear_knee.x + design.arm_diameter_mm / 2.0,
        frame.bogie_front_knee.x - design.arm_diameter_mm / 2.0,
        frame.bogie_front_knee.x + design.arm_diameter_mm / 2.0,
    ]
    y_values = [
        0.0,
        frame.front_wheel.y + radius,
        frame.middle_wheel.y + radius,
        frame.rear_wheel.y + radius,
        frame.bogie_rear_knee.y + design.arm_diameter_mm / 2.0,
        frame.bogie_front_knee.y + design.arm_diameter_mm / 2.0,
        frame.bogie_pivot.y + design.arm_diameter_mm / 2.0,
        frame.main_pivot.y + design.arm_diameter_mm / 2.0,
    ]

    local = _design_local_points(design, 1)
    front = local["front"]
    bogie = local["bogie_pivot"]
    middle = local["middle_from_bogie"]
    rear = local["rear_from_bogie"]
    return {
        "total_length_mm": max(x_values) - min(x_values),
        "total_height_mm": max(y_values) - min(y_values),
        "wheelbase_mm": design.front_wheel_x_mm,
        "rocker_front_angle_deg": degrees(atan2(front.y, front.x)),
        "rocker_bogie_angle_deg": degrees(atan2(bogie.y, bogie.x)),
        "bogie_middle_angle_deg": degrees(atan2(middle.y, middle.x)),
        "bogie_rear_angle_deg": degrees(atan2(rear.y, rear.x)),
        "rocker_front_length_mm": hypot(front.x, front.y),
        "rocker_bogie_length_mm": hypot(bogie.x, bogie.y),
        "bogie_middle_length_mm": hypot(middle.x, middle.y),
        "bogie_rear_length_mm": hypot(rear.x, rear.y),
    }


def validate_design(design: ChassisDesign) -> tuple[str, ...]:
    """Возвращает список найденных ошибок геометрии."""
    errors: list[str] = []

    positive = {
        "радиус колеса": design.wheel_radius_mm,
        "X среднего колеса": design.middle_wheel_x_mm,
        "X переднего колеса": design.front_wheel_x_mm,
        "высота главного шарнира": design.main_pivot_y_mm,
        "высота шарнира bogie": design.bogie_pivot_y_mm,
        "диаметр рычагов": design.arm_diameter_mm,
        "максимальная длина": design.max_total_length_mm,
    }
    for name, value in positive.items():
        if value <= 0.0:
            errors.append(f"Параметр «{name}» должен быть больше нуля.")

    if not 0.0 < design.middle_wheel_x_mm < design.front_wheel_x_mm:
        errors.append(
            "Должно выполняться: 0 < X среднего колеса < X переднего колеса."
        )
    if not 0.0 < design.main_pivot_x_mm < design.front_wheel_x_mm:
        errors.append("Главный шарнир должен находиться между задним и передним колесом.")
    if not 0.0 < design.bogie_pivot_x_mm < design.middle_wheel_x_mm:
        errors.append("Шарнир bogie должен находиться между двумя колёсами bogie.")
    if not (
        0.0
        <= design.bogie_rear_knee_x_mm
        < design.bogie_front_knee_x_mm
        <= design.middle_wheel_x_mm
    ):
        errors.append("Верхние углы bogie должны находиться между его колёсами.")
    if not (
        design.bogie_rear_knee_x_mm
        <= design.bogie_pivot_x_mm
        <= design.bogie_front_knee_x_mm
    ):
        errors.append("Шарнир rocker должен лежать на верхней перемычке bogie.")
    if design.bogie_min_angle_deg >= design.bogie_max_angle_deg:
        errors.append("Минимальный угол bogie должен быть меньше максимального.")
    if design.rocker_min_angle_deg >= design.rocker_max_angle_deg:
        errors.append("Минимальный угол rocker должен быть меньше максимального.")
    if min(
        design.obstacle_clearance_mm,
        design.self_clearance_mm,
    ) < 0.0:
        errors.append("Геометрические запасы не могут быть отрицательными.")

    wheel_distances = (
        design.middle_wheel_x_mm,
        design.front_wheel_x_mm - design.middle_wheel_x_mm,
        design.front_wheel_x_mm,
    )
    required_wheel_distance = (
        2.0 * design.wheel_radius_mm + design.self_clearance_mm
    )
    if min(wheel_distances) < required_wheel_distance:
        errors.append("В исходном положении колёса пересекаются или не имеют зазора.")

    if not errors:
        dimensions = neutral_dimensions(design)
        if dimensions["total_length_mm"] > design.max_total_length_mm + 1e-6:
            errors.append(
                "Полная длина "
                f"{dimensions['total_length_mm']:.1f} мм превышает ограничение "
                f"{design.max_total_length_mm:.1f} мм."
            )

    return tuple(errors)


def _sample_segment(start: Point, end: Point, count: int) -> tuple[Point, ...]:
    count = max(2, count)
    return tuple(
        Point(
            start.x + (end.x - start.x) * index / (count - 1),
            start.y + (end.y - start.y) * index / (count - 1),
        )
        for index in range(count)
    )


@lru_cache(maxsize=128)
def _local_collision_samples(
    design: ChassisDesign,
    direction: int,
    sample_count: int,
) -> tuple[
    tuple[Point, ...],
    tuple[Point, ...],
    tuple[Point, ...],
]:
    """Точки rocker и bogie для проверки толщины и столкновений."""
    local = _design_local_points(design, direction)
    origin = Point(0.0, 0.0)

    rocker = (
        *_sample_segment(local["front"], origin, sample_count),
        *_sample_segment(origin, local["bogie_pivot"], sample_count),
    )

    rear = local["rear_from_bogie"]
    rear_knee = local["rear_knee_from_bogie"]
    front_knee = local["front_knee_from_bogie"]
    middle = local["middle_from_bogie"]
    bogie = (
        *_sample_segment(rear, rear_knee, sample_count),
        *_sample_segment(rear_knee, front_knee, sample_count),
        *_sample_segment(front_knee, middle, sample_count),
    )
    return tuple(rocker), (), tuple(bogie)


def _rotate_arrays(
    x: np.ndarray,
    y: np.ndarray,
    angles_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(angles_rad)
    sine = np.sin(angles_rad)
    return x * cosine - y * sine, x * sine + y * cosine


def _points_to_arrays(points: tuple[Point, ...]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([point.x for point in points], dtype=float)[:, None],
        np.asarray([point.y for point in points], dtype=float)[:, None],
    )


def _point_rectangle_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    dx = np.maximum(np.maximum(x_min - x, 0.0), x - x_max)
    dy = np.maximum(np.maximum(y_min - y, 0.0), y - y_max)
    return np.hypot(dx, dy)


def _evaluate_angle_candidates(
    design: ChassisDesign,
    terrain: Terrain,
    main_x: float,
    direction: int,
    angle_pairs: np.ndarray,
    settings: SolverSettings,
    preferred_main_y: tuple[float, ...] = (),
) -> list[Pose]:
    if angle_pairs.size == 0:
        return []

    pitches = angle_pairs[:, 0]
    bogie_angles = angle_pairs[:, 1]
    pitch_rad = np.radians(pitches)
    bogie_absolute_rad = np.radians(pitches + bogie_angles)
    bogie_relative_rad = np.radians(bogie_angles)
    local = _design_local_points(design, direction)
    count = len(pitches)

    def rotate_single(point: Point, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.full(count, point.x, dtype=float)
        y = np.full(count, point.y, dtype=float)
        return _rotate_arrays(x, y, angles)

    front_x, front_y = rotate_single(local["front"], pitch_rad)
    pivot_x, pivot_y = rotate_single(local["bogie_pivot"], pitch_rad)
    middle_dx, middle_dy = rotate_single(
        local["middle_from_bogie"],
        bogie_absolute_rad,
    )
    rear_dx, rear_dy = rotate_single(
        local["rear_from_bogie"],
        bogie_absolute_rad,
    )
    middle_x = pivot_x + middle_dx
    middle_y = pivot_y + middle_dy
    rear_x = pivot_x + rear_dx
    rear_y = pivot_y + rear_dy

    wheel_x = np.stack((front_x, middle_x, rear_x), axis=1)
    wheel_y = np.stack((front_y, middle_y, rear_y), axis=1)
    world_wheel_x = wheel_x + main_x

    wheel_required = np.empty_like(wheel_x)
    for wheel_index in range(3):
        wheel_required[:, wheel_index] = (
            terrain.clearance_y(
                world_wheel_x[:, wheel_index],
                design.wheel_radius_mm,
            )
            - wheel_y[:, wheel_index]
        )

    rocker_samples, _, bogie_samples = _local_collision_samples(
        design,
        direction,
        settings.structure_samples_per_segment,
    )

    structure_required = np.full(count, -np.inf)

    fixed_groups = ((
        rocker_samples,
        design.arm_diameter_mm / 2.0 + design.obstacle_clearance_mm,
    ),)
    for points, clearance_radius in fixed_groups:
        sample_x, sample_y = _points_to_arrays(points)
        rotated_x, rotated_y = _rotate_arrays(sample_x, sample_y, pitch_rad[None, :])
        world_x = rotated_x + main_x
        required = terrain.clearance_y(world_x, clearance_radius) - rotated_y
        structure_required = np.maximum(
            structure_required,
            np.max(required, axis=0),
        )

    bogie_sample_x, bogie_sample_y = _points_to_arrays(bogie_samples)
    rotated_bogie_x, rotated_bogie_y = _rotate_arrays(
        bogie_sample_x,
        bogie_sample_y,
        bogie_absolute_rad[None, :],
    )
    rotated_bogie_x += pivot_x[None, :]
    rotated_bogie_y += pivot_y[None, :]
    bogie_required = (
        terrain.clearance_y(
            rotated_bogie_x + main_x,
            design.arm_diameter_mm / 2.0
            + design.obstacle_clearance_mm,
        )
        - rotated_bogie_y
    )
    structure_required = np.maximum(
        structure_required,
        np.max(bogie_required, axis=0),
    )

    minimum_main_y = np.maximum(
        np.max(wheel_required, axis=1),
        structure_required,
    )

    source_indices = list(range(count))
    main_y_values = [float(value) for value in minimum_main_y]

    # На вертикальной грани координата Y не определяется минимальной высотой:
    # колесо может перемещаться вверх при неизменном X. Поэтому для таких
    # положений добавляются несколько высот, близких к предыдущим кадрам.
    if isinstance(terrain, StepTerrain) and preferred_main_y:
        wall_x_tolerance = max(
            0.25,
            settings.contact_tolerance_mm * 0.35,
        )
        near_wall = (
            np.abs(world_wheel_x + design.wheel_radius_mm)
            <= wall_x_tolerance
        )
        preferred = np.unique(
            np.asarray(preferred_main_y, dtype=float)
        )
        if len(preferred) > 5:
            preferred = preferred[
                np.linspace(0, len(preferred) - 1, 5).round().astype(int)
            ]
        vertical_increment = max(
            3.0,
            min(15.0, settings.maximum_vertical_change_mm / 3.0),
        )
        variants: set[tuple[int, float]] = {
            (index, round(float(minimum_main_y[index]), 5))
            for index in range(count)
        }
        for index in range(count):
            for wheel_index in np.flatnonzero(near_wall[index]):
                lower = max(
                    float(minimum_main_y[index]),
                    design.wheel_radius_mm - wheel_y[index, wheel_index],
                )
                upper = (
                    terrain.height_mm - wheel_y[index, wheel_index]
                )
                if upper < lower - 1e-8:
                    continue
                for previous_y in preferred:
                    for candidate_y in (
                        previous_y - vertical_increment,
                        previous_y,
                        previous_y + vertical_increment,
                        previous_y + 2.0 * vertical_increment,
                        upper,
                    ):
                        value = min(upper, max(lower, float(candidate_y)))
                        variants.add((index, round(value, 5)))
        source_indices = [index for index, _ in sorted(variants)]
        main_y_values = [value for _, value in sorted(variants)]

    source = np.asarray(source_indices, dtype=int)
    main_y = np.asarray(main_y_values, dtype=float)
    pitches = pitches[source]
    bogie_angles = bogie_angles[source]
    pitch_rad = pitch_rad[source]
    bogie_absolute_rad = bogie_absolute_rad[source]
    bogie_relative_rad = bogie_relative_rad[source]
    wheel_x = wheel_x[source]
    wheel_y = wheel_y[source]
    world_wheel_x = world_wheel_x[source]
    wheel_required = wheel_required[source]
    structure_required = structure_required[source]
    pivot_x = pivot_x[source]
    pivot_y = pivot_y[source]
    count = len(source)

    wheel_gaps = main_y[:, None] - wheel_required
    contacts = (
        np.isfinite(wheel_required)
        & (wheel_gaps <= settings.contact_tolerance_mm)
    )

    support_x = np.full_like(wheel_x, np.nan)
    for wheel_index in range(3):
        values = terrain.support_x(
            world_wheel_x[:, wheel_index],
            design.wheel_radius_mm,
        )
        support_x[:, wheel_index] = np.where(
            contacts[:, wheel_index],
            values,
            np.nan,
        )

    wall_contacts = np.zeros_like(contacts)
    if isinstance(terrain, StepTerrain) and terrain.height_mm > 0.0:
        wheel_center_y = main_y[:, None] + wheel_y
        wall_contacts = (
            (np.abs(world_wheel_x + design.wheel_radius_mm)
             <= max(0.25, settings.contact_tolerance_mm * 0.35))
            & (wheel_center_y >= design.wheel_radius_mm - settings.contact_tolerance_mm)
            & (wheel_center_y <= terrain.height_mm + settings.contact_tolerance_mm)
        )

    contact_count = np.sum(contacts, axis=1)
    kinematic_contact_count = np.sum(contacts | wall_contacts, axis=1)
    support_min = np.min(
        np.where(np.isfinite(support_x), support_x, np.inf),
        axis=1,
    )
    support_max = np.max(
        np.where(np.isfinite(support_x), support_x, -np.inf),
        axis=1,
    )

    com_x, _ = rotate_single(local["com"], pitch_rad)
    world_com_x = main_x + com_x
    support_margin = np.minimum(
        world_com_x - support_min,
        support_max - world_com_x,
    )

    # Колёса не должны менять свой продольный порядок или пересекаться.
    neutral_x = np.asarray(
        [
            local["front"].x,
            local["bogie_pivot"].x + local["middle_from_bogie"].x,
            local["bogie_pivot"].x + local["rear_from_bogie"].x,
        ]
    )
    order = np.argsort(neutral_x)
    ordered_x = wheel_x[:, order]
    order_valid = np.all(np.diff(ordered_x, axis=1) > 1e-3, axis=1)

    wheel_distance_valid = np.ones(count, dtype=bool)
    minimum_distance = (
        2.0 * design.wheel_radius_mm + design.self_clearance_mm
    )
    for first, second in ((0, 1), (0, 2), (1, 2)):
        distance = np.hypot(
            wheel_x[:, first] - wheel_x[:, second],
            wheel_y[:, first] - wheel_y[:, second],
        )
        wheel_distance_valid &= distance >= minimum_distance

    if isinstance(terrain, StepTerrain):
        # Ступень обрабатывается отдельным решателем
        # _simulate_step_contact_traversal(), где обязательны контакты всех
        # трёх колёс. Старый сеточный поиск для неё использовать нельзя.
        return []

    contact_valid = contact_count >= 2
    valid = (
        contact_valid
        & np.isfinite(main_y)
        & order_valid
        & wheel_distance_valid
    )

    poses: list[Pose] = []
    for index in np.flatnonzero(valid):
        contact_names = tuple(
            WHEEL_NAMES[wheel_index]
            for wheel_index in range(3)
            if contacts[index, wheel_index]
        )
        wall_contact_names = tuple(
            WHEEL_NAMES[wheel_index]
            for wheel_index in range(3)
            if wall_contacts[index, wheel_index]
        )
        poses.append(
            Pose(
                main_pivot_x_mm=float(main_x),
                main_pivot_y_mm=float(main_y[index]),
                body_pitch_deg=float(pitches[index]),
                bogie_angle_deg=float(bogie_angles[index]),
                support_margin_mm=float(support_margin[index]),
                contacts=contact_names,
                wall_contacts=wall_contact_names,
            )
        )
    return poses


@dataclass(frozen=True)
class _State:
    pose: Pose
    path: tuple[Pose, ...]
    cost: float


def _angle_candidates(
    states: list[_State],
    design: ChassisDesign,
    settings: SolverSettings,
) -> np.ndarray:
    pitch_offsets = np.arange(
        -settings.pitch_window_deg,
        settings.pitch_window_deg + settings.angle_step_deg * 0.5,
        settings.angle_step_deg,
    )
    bogie_offsets = np.arange(
        -settings.bogie_window_deg,
        settings.bogie_window_deg + settings.angle_step_deg * 0.5,
        settings.angle_step_deg,
    )

    offset_grid = np.stack(
        np.meshgrid(pitch_offsets, bogie_offsets, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    current = np.asarray(
        [[state.pose.body_pitch_deg, state.pose.bogie_angle_deg] for state in states]
    ).reshape(-1, 2)
    grid = (current[:, None, :] + offset_grid[None, :, :]).reshape(-1, 2)
    inside = (
        (grid[:, 0] >= design.rocker_min_angle_deg)
        & (grid[:, 0] <= design.rocker_max_angle_deg)
        & (grid[:, 1] >= design.bogie_min_angle_deg)
        & (grid[:, 1] <= design.bogie_max_angle_deg)
    )

    candidates: list[tuple[float, float]] = []
    for state in states:
        # Точные нулевые значения не всегда попадают в равномерную сетку,
        # поэтому отдельно добавляется один шаг возврата к исходной позе.
        neutral_pitch = (
            0.0
            if abs(state.pose.body_pitch_deg) <= settings.angle_step_deg
            else state.pose.body_pitch_deg
            - np.sign(state.pose.body_pitch_deg) * settings.angle_step_deg
        )
        neutral_bogie = (
            0.0
            if abs(state.pose.bogie_angle_deg) <= settings.angle_step_deg
            else state.pose.bogie_angle_deg
            - np.sign(state.pose.bogie_angle_deg) * settings.angle_step_deg
        )
        candidates.extend(
            (
                (float(neutral_pitch), state.pose.bogie_angle_deg),
                (state.pose.body_pitch_deg, float(neutral_bogie)),
                (float(neutral_pitch), float(neutral_bogie)),
                (0.0, 0.0),
            )
        )

    combined = np.vstack((grid[inside], np.asarray(candidates).reshape(-1, 2)))
    if combined.size == 0:
        return np.empty((0, 2), dtype=float)

    # Округление устраняет одинаковые пары, пришедшие из разных ветвей;
    # np.unique по строкам возвращает их отсортированными, как и раньше.
    return np.unique(np.round(combined, 6), axis=0)


def _angle_in_bounds(
    angle_rad: float,
    minimum_deg: float,
    maximum_deg: float,
) -> list[float]:
    """Возвращает эквивалентные углы, попавшие в заданный диапазон."""
    angle_deg = degrees(angle_rad)
    result: list[float] = []
    for turns in range(-2, 3):
        candidate = angle_deg + 360.0 * turns
        if minimum_deg - 1e-9 <= candidate <= maximum_deg + 1e-9:
            result.append(candidate)
    return result


def _enrich_step_wall_angles(
    design: ChassisDesign,
    terrain: Terrain,
    main_x: float,
    direction: int,
    base_pairs: np.ndarray,
    states: list[_State],
    settings: SolverSettings,
) -> np.ndarray:
    """Добавляет углы точного касания колеса вертикальной грани.

    Обычная угловая сетка почти никогда не попадает ровно в условие
    ``x центра = -R``. Без этих аналитических точек решатель ошибочно считал бы
    ступени выше радиуса непроходимыми.
    """
    if not isinstance(terrain, StepTerrain) or base_pairs.size == 0:
        return base_pairs

    local = _design_local_points(design, direction)
    # Малое смещение наружу устраняет неоднозначность машинного округления
    # непосредственно на разрыве функции у вертикальной грани.
    target_x = -design.wheel_radius_mm - 0.05
    pairs = {
        (round(float(pair[0]), 7), round(float(pair[1]), 7))
        for pair in base_pairs
    }

    def close_to_states(pitch: float, bogie: float) -> bool:
        return any(
            abs(pitch - state.pose.body_pitch_deg)
            <= settings.pitch_window_deg + 1e-9
            and abs(bogie - state.pose.bogie_angle_deg)
            <= settings.bogie_window_deg + 1e-9
            for state in states
        )

    # Одиночное колесо жёстко связано с rocker и корпусом.
    front = local["front"]
    front_length = hypot(front.x, front.y)
    if front_length > 1e-9:
        cosine_value = (target_x - main_x) / front_length
        if -1.0 <= cosine_value <= 1.0:
            alpha = atan2(front.y, front.x)
            root = float(np.arccos(cosine_value))
            pitch_roots: list[float] = []
            for angle in (root - alpha, -root - alpha):
                pitch_roots.extend(
                    _angle_in_bounds(
                        angle,
                        design.rocker_min_angle_deg,
                        design.rocker_max_angle_deg,
                    )
                )
            bogie_values = np.unique(base_pairs[:, 1])
            for pitch in pitch_roots:
                for bogie in bogie_values:
                    if close_to_states(pitch, float(bogie)):
                        pairs.add((round(pitch, 7), round(float(bogie), 7)))

    # Для каждого колеса bogie при фиксированном угле корпуса аналитически
    # решается относительный угол bogie.
    bogie_pivot = local["bogie_pivot"]
    for wheel_name in ("middle_from_bogie", "rear_from_bogie"):
        wheel = local[wheel_name]
        wheel_length = hypot(wheel.x, wheel.y)
        if wheel_length <= 1e-9:
            continue
        wheel_alpha = atan2(wheel.y, wheel.x)
        for pitch in np.unique(base_pairs[:, 0]):
            pitch_rad = radians(float(pitch))
            pivot_world_x = (
                main_x
                + bogie_pivot.x * cos(pitch_rad)
                - bogie_pivot.y * sin(pitch_rad)
            )
            cosine_value = (target_x - pivot_world_x) / wheel_length
            if not -1.0 <= cosine_value <= 1.0:
                continue
            root = float(np.arccos(cosine_value))
            for absolute_angle in (root - wheel_alpha, -root - wheel_alpha):
                relative_angle = absolute_angle - pitch_rad
                for bogie in _angle_in_bounds(
                    relative_angle,
                    design.bogie_min_angle_deg,
                    design.bogie_max_angle_deg,
                ):
                    if close_to_states(float(pitch), bogie):
                        pairs.add(
                            (round(float(pitch), 7), round(bogie, 7))
                        )

    return np.asarray(sorted(pairs), dtype=float)


def _pose_distance(
    previous: Pose,
    current: Pose,
    settings: SolverSettings,
) -> float:
    pitch_change = abs(current.body_pitch_deg - previous.body_pitch_deg)
    bogie_change = abs(current.bogie_angle_deg - previous.bogie_angle_deg)
    vertical_change = abs(current.main_pivot_y_mm - previous.main_pivot_y_mm)
    if (
        pitch_change > settings.maximum_pitch_change_deg + 1e-9
        or bogie_change > settings.maximum_bogie_change_deg + 1e-9
        or vertical_change > settings.maximum_vertical_change_mm + 1e-9
    ):
        return float("inf")
    return (
        (pitch_change / max(settings.angle_step_deg, 1e-6)) ** 2
        + (bogie_change / max(settings.angle_step_deg, 1e-6)) ** 2
        + (vertical_change / max(settings.progress_step_mm, 1e-6)) ** 2
    )


def _deduplicate_states(
    states: list[_State],
    beam_width: int,
) -> list[_State]:
    best: dict[tuple[int, int, int], _State] = {}
    for state in states:
        key = (
            round(state.pose.body_pitch_deg * 10),
            round(state.pose.bogie_angle_deg * 10),
            round(state.pose.main_pivot_y_mm / 5.0),
        )
        existing = best.get(key)
        if existing is None or state.cost < existing.cost:
            best[key] = state
    return sorted(best.values(), key=lambda state: state.cost)[:beam_width]


def _finish_reached(
    frame: ChassisFrame,
    pose: Pose,
    terrain: Terrain,
) -> bool:
    wheel_x = [wheel.x for wheel in frame.wheels]
    all_wheels_supported = set(pose.contacts) == set(WHEEL_NAMES)
    if terrain.kind == "step":
        return min(wheel_x) >= -1e-6 and all_wheels_supported
    assert isinstance(terrain, DitchTerrain)
    return (
        min(wheel_x) >= terrain.width_mm - 1e-6
        and all_wheels_supported
    )



StepContactMode = Literal["lower", "wall", "corner", "top"]


def _step_floor_contact_end(height_mm: float, radius_mm: float) -> float:
    """X центра колеса, где заканчивается контакт с нижней площадкой."""
    if height_mm >= radius_mm:
        return -radius_mm
    vertical_offset = radius_mm - height_mm
    return -sqrt(max(0.0, radius_mm * radius_mm - vertical_offset * vertical_offset))


def _step_signed_distance_to_solid(
    x_mm: float,
    y_mm: float,
    height_mm: float,
) -> float:
    """Знаковое расстояние от точки до твёрдого профиля ступени.

    Положительное значение соответствует свободному пространству. Нулевое
    значение находится на границе грунта. Профиль является объединением
    нижней полуплоскости ``y <= 0`` и блока ``x >= 0, y <= height``.
    """
    distance_to_ground = y_mm

    if x_mm < 0.0 and y_mm > height_mm:
        distance_to_block = hypot(x_mm, y_mm - height_mm)
    elif x_mm < 0.0 and y_mm <= height_mm:
        distance_to_block = -x_mm
    elif x_mm >= 0.0 and y_mm > height_mm:
        distance_to_block = y_mm - height_mm
    else:
        # Точка внутри блока. Минус означает проникновение в препятствие.
        distance_to_block = -min(x_mm, height_mm - y_mm)

    return min(distance_to_ground, distance_to_block)


def _step_contact_residual(
    wheel: Point,
    mode: StepContactMode,
    height_mm: float,
    radius_mm: float,
) -> float:
    if mode == "lower":
        return wheel.y - radius_mm
    if mode == "wall":
        return wheel.x + radius_mm
    if mode == "corner":
        return hypot(wheel.x, wheel.y - height_mm) - radius_mm
    return wheel.y - height_mm - radius_mm


def _step_contact_is_valid(
    wheel: Point,
    mode: StepContactMode,
    height_mm: float,
    radius_mm: float,
    tolerance_mm: float,
) -> bool:
    floor_end = _step_floor_contact_end(height_mm, radius_mm)
    clearance = (
        _step_signed_distance_to_solid(wheel.x, wheel.y, height_mm)
        - radius_mm
    )
    if clearance < -tolerance_mm:
        return False
    if abs(
        _step_contact_residual(
            wheel,
            mode,
            height_mm,
            radius_mm,
        )
    ) > tolerance_mm:
        return False

    if mode == "lower":
        return wheel.x <= floor_end + tolerance_mm
    if mode == "wall":
        return (
            height_mm >= radius_mm - tolerance_mm
            and radius_mm - tolerance_mm
            <= wheel.y
            <= height_mm + tolerance_mm
        )
    if mode == "corner":
        return (
            floor_end - tolerance_mm
            <= wheel.x
            <= tolerance_mm
            and wheel.y >= max(radius_mm, height_mm) - tolerance_mm
        )
    return wheel.x >= -tolerance_mm


def _step_climb_targets(
    height_mm: float,
    radius_mm: float,
    path_step_mm: float,
) -> tuple[
    tuple[float, float, StepContactMode],
    tuple[tuple[float, float, StepContactMode], ...],
]:
    """Возвращает непрерывный путь центра колеса вокруг кромки ступени."""
    floor_end = _step_floor_contact_end(height_mm, radius_mm)
    targets: list[tuple[float, float, StepContactMode]] = []

    if height_mm >= radius_mm:
        corner_start_angle = pi
        if height_mm > radius_mm + 1e-9:
            wall_count = max(
                1,
                int(np.ceil((height_mm - radius_mm) / path_step_mm)),
            )
            for y_mm in np.linspace(
                radius_mm,
                height_mm,
                wall_count + 1,
            )[1:]:
                targets.append(
                    (-radius_mm, float(y_mm), "wall")
                )
    else:
        corner_start_angle = pi - asin(1.0 - height_mm / radius_mm)

    corner_length = radius_mm * (corner_start_angle - pi / 2.0)
    corner_count = max(1, int(np.ceil(corner_length / path_step_mm)))
    for angle in np.linspace(
        corner_start_angle,
        pi / 2.0,
        corner_count + 1,
    )[1:]:
        targets.append(
            (
                radius_mm * cos(float(angle)),
                height_mm + radius_mm * sin(float(angle)),
                "corner",
            )
        )

    return (
        (floor_end, radius_mm, "lower"),
        tuple(targets),
    )


def _linear_contact_targets(
    start_x_mm: float,
    end_x_mm: float,
    y_mm: float,
    mode: StepContactMode,
    path_step_mm: float,
    include_end: bool = True,
) -> tuple[tuple[float, float, StepContactMode], ...]:
    distance = abs(end_x_mm - start_x_mm)
    if distance <= 1e-9:
        return ()
    count = max(1, int(np.ceil(distance / path_step_mm)))
    values = np.linspace(start_x_mm, end_x_mm, count + 1)[1:]
    if not include_end and len(values) > 0:
        values = values[:-1]
    return tuple((float(x_mm), y_mm, mode) for x_mm in values)


def _scalar_rectangle_distance(
    point: Point,
    bounds: tuple[float, float, float, float],
) -> float:
    x_min, x_max, y_min, y_max = bounds
    dx = max(x_min - point.x, 0.0, point.x - x_max)
    dy = max(y_min - point.y, 0.0, point.y - y_max)
    return hypot(dx, dy)


def _step_frame_has_clearance(
    design: ChassisDesign,
    terrain: StepTerrain,
    pose: Pose,
    direction: int,
    settings: SolverSettings,
) -> bool:
    """Проверяет проникновение колёс, корпуса и рычагов в ступень."""
    frame = calculate_frame(design, pose, direction)
    radius = design.wheel_radius_mm
    tolerance = max(0.05, settings.contact_tolerance_mm * 0.08)

    for wheel in frame.wheels:
        clearance = (
            _step_signed_distance_to_solid(
                wheel.x,
                wheel.y,
                terrain.height_mm,
            )
            - radius
        )
        if clearance < -tolerance:
            return False

    rocker_samples, _, bogie_samples = _local_collision_samples(
        design,
        direction,
        settings.structure_samples_per_segment,
    )
    fixed_groups = ((
        rocker_samples,
        design.arm_diameter_mm / 2.0 + design.obstacle_clearance_mm,
    ),)
    main = frame.main_pivot
    for points, clearance_radius in fixed_groups:
        for point in points:
            world = _add(main, _rotate(point, pose.body_pitch_deg))
            distance = _step_signed_distance_to_solid(
                world.x,
                world.y,
                terrain.height_mm,
            )
            if distance < clearance_radius - tolerance:
                return False

    bogie_angle = pose.body_pitch_deg + pose.bogie_angle_deg
    bogie_clearance = (
        design.arm_diameter_mm / 2.0
        + design.obstacle_clearance_mm
    )
    for point in bogie_samples:
        world = _add(frame.bogie_pivot, _rotate(point, bogie_angle))
        distance = _step_signed_distance_to_solid(
            world.x,
            world.y,
            terrain.height_mm,
        )
        if distance < bogie_clearance - tolerance:
            return False

    # Колёса не пересекаются и сохраняют исходный продольный порядок.
    minimum_wheel_distance = (
        2.0 * radius + design.self_clearance_mm
    )
    for first, second in ((0, 1), (0, 2), (1, 2)):
        if hypot(
            frame.wheels[first].x - frame.wheels[second].x,
            frame.wheels[first].y - frame.wheels[second].y,
        ) < minimum_wheel_distance - tolerance:
            return False

    local = _design_local_points(design, direction)
    neutral_x = np.asarray(
        [
            local["front"].x,
            local["bogie_pivot"].x + local["middle_from_bogie"].x,
            local["bogie_pivot"].x + local["rear_from_bogie"].x,
        ]
    )
    order = np.argsort(neutral_x)
    ordered_world_x = np.asarray(
        [frame.wheels[index].x for index in order]
    )
    if np.any(np.diff(ordered_world_x) <= 1e-3):
        return False

    return True


def _step_pose_from_modes(
    design: ChassisDesign,
    solution: np.ndarray,
    modes: dict[int, StepContactMode],
) -> Pose:
    contacts = tuple(
        WHEEL_NAMES[index]
        for index in range(3)
        if modes[index] != "wall"
    )
    wall_contacts = tuple(
        WHEEL_NAMES[index]
        for index in range(3)
        if modes[index] == "wall"
    )
    return Pose(
        main_pivot_x_mm=float(solution[0]),
        main_pivot_y_mm=float(solution[1]),
        body_pitch_deg=float(solution[2]),
        bogie_angle_deg=float(solution[3]),
        support_margin_mm=0.0,
        contacts=contacts,
        wall_contacts=wall_contacts,
    )


@dataclass(frozen=True)
class _NewtonResult:
    x: np.ndarray
    fun: np.ndarray


def _newton_bounded(
    residual: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float = 1e-7,
    iterations: int = 12,
) -> _NewtonResult | None:
    """Метод Ньютона с численным якобианом и проекцией на границы.

    Для квадратной системы контактных уравнений с тёплым стартом из
    соседнего кадра сходится за 2–3 итерации, что в десятки раз дешевле
    scipy.optimize.least_squares. При отсутствии сходимости возвращает
    None, и вызывающий код использует least_squares.
    """
    x = np.asarray(initial, dtype=float).copy()
    values = residual(x)
    for _ in range(iterations):
        if np.max(np.abs(values)) <= tolerance:
            return _NewtonResult(x, values)
        jacobian = np.empty((len(values), len(x)))
        for column in range(len(x)):
            step = 1e-6 * max(1.0, abs(x[column]))
            shifted = x.copy()
            shifted[column] += step
            jacobian[:, column] = (residual(shifted) - values) / step
        try:
            delta = np.linalg.solve(jacobian, -values)
        except np.linalg.LinAlgError:
            return None
        # Демпфирование: шаг уменьшается, пока невязка не начнёт убывать.
        norm = np.linalg.norm(values)
        for _ in range(6):
            candidate = np.clip(x + delta, lower, upper)
            candidate_values = residual(candidate)
            if np.linalg.norm(candidate_values) < norm:
                break
            delta *= 0.5
        else:
            return None
        x, values = candidate, candidate_values
    if np.max(np.abs(values)) <= tolerance:
        return _NewtonResult(x, values)
    return None


def _solve_step_contact_target(
    design: ChassisDesign,
    terrain: StepTerrain,
    direction: int,
    controlled_wheel: int,
    target: tuple[float, float, StepContactMode],
    passive_modes: dict[int, StepContactMode],
    initial_solution: np.ndarray,
    settings: SolverSettings,
) -> tuple[np.ndarray, Pose] | None:
    """Решает четыре уравнения контакта для одного положения колеса."""
    passive = tuple(
        index for index in range(3) if index != controlled_wheel
    )
    radius = design.wheel_radius_mm
    contact_tolerance = max(
        0.08,
        min(0.35, settings.contact_tolerance_mm * 0.06),
    )

    def residual(solution: np.ndarray) -> np.ndarray:
        temporary_pose = Pose(
            main_pivot_x_mm=float(solution[0]),
            main_pivot_y_mm=float(solution[1]),
            body_pitch_deg=float(solution[2]),
            bogie_angle_deg=float(solution[3]),
            support_margin_mm=0.0,
            contacts=(),
        )
        frame = calculate_frame(design, temporary_pose, direction)
        wheels = frame.wheels
        return np.asarray(
            [
                wheels[controlled_wheel].x - target[0],
                wheels[controlled_wheel].y - target[1],
                _step_contact_residual(
                    wheels[passive[0]],
                    passive_modes[passive[0]],
                    terrain.height_mm,
                    radius,
                ),
                _step_contact_residual(
                    wheels[passive[1]],
                    passive_modes[passive[1]],
                    terrain.height_mm,
                    radius,
                ),
            ],
            dtype=float,
        )

    vertical_limit = max(
        4000.0,
        terrain.height_mm
        + design.main_pivot_y_mm
        + 4.0 * design.front_wheel_x_mm,
    )
    lower_bounds = np.asarray(
        [
            -5000.0,
            0.0,
            design.rocker_min_angle_deg,
            design.bogie_min_angle_deg,
        ]
    )
    upper_bounds = np.asarray(
        [
            5000.0,
            vertical_limit,
            design.rocker_max_angle_deg,
            design.bogie_max_angle_deg,
        ]
    )

    seeds = [initial_solution]
    perturbations = (
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0, 2.0),
        (0.0, 0.0, 0.0, -2.0),
    )
    best: tuple[float, np.ndarray, Pose] | None = None

    for seed_index, seed in enumerate(seeds):
        result = _newton_bounded(
            residual,
            np.clip(seed, lower_bounds, upper_bounds),
            lower_bounds,
            upper_bounds,
        )
        if result is None:
            result = least_squares(
                residual,
                np.clip(seed, lower_bounds, upper_bounds),
                bounds=(lower_bounds, upper_bounds),
                x_scale=np.asarray([100.0, 100.0, 10.0, 10.0]),
                max_nfev=100 if seed_index == 0 else 180,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
        if np.max(np.abs(result.fun)) > contact_tolerance:
            if seed_index == 0:
                seeds.extend(
                    initial_solution + np.asarray(offset)
                    for offset in perturbations
                )
            continue

        frame = calculate_frame(
            design,
            Pose(
                main_pivot_x_mm=float(result.x[0]),
                main_pivot_y_mm=float(result.x[1]),
                body_pitch_deg=float(result.x[2]),
                bogie_angle_deg=float(result.x[3]),
                support_margin_mm=0.0,
                contacts=(),
            ),
            direction,
        )
        modes = dict(passive_modes)
        modes[controlled_wheel] = target[2]
        if not all(
            _step_contact_is_valid(
                frame.wheels[index],
                modes[index],
                terrain.height_mm,
                radius,
                contact_tolerance,
            )
            for index in range(3)
        ):
            if seed_index == 0:
                seeds.extend(
                    initial_solution + np.asarray(offset)
                    for offset in perturbations
                )
            continue

        pose = _step_pose_from_modes(design, result.x, modes)
        if not _step_frame_has_clearance(
            design,
            terrain,
            pose,
            direction,
            settings,
        ):
            if seed_index == 0:
                seeds.extend(
                    initial_solution + np.asarray(offset)
                    for offset in perturbations
                )
            continue

        normalized_jump = float(
            np.linalg.norm(
                (result.x - initial_solution)
                / np.asarray([20.0, 20.0, 2.0, 3.0])
            )
        )
        if best is None or normalized_jump < best[0]:
            best = (normalized_jump, result.x.copy(), pose)

        # На гладком участке решение из предыдущего кадра почти всегда
        # сходится к единственной непрерывной ветви. Дополнительные seeds
        # нужны только при переходе между площадкой, стенкой и кромкой.
        if seed_index == 0:
            break

    if best is None:
        return None
    return best[1], best[2]


def _simulate_step_contact_traversal(
    design: ChassisDesign,
    terrain: StepTerrain,
    direction: int,
    settings: SolverSettings,
) -> TraversalResult:
    """Квазистатическая кинематика ступени с контактом всех трёх колёс.

    Ведущее колесо каждой фазы задаётся точкой на точном эквидистантном
    контуре ступени. Положение главного шарнира, угол корпуса и угол bogie
    находятся из четырёх нелинейных уравнений. После подъёма одного колеса
    управление передаётся следующему, поэтому на вертикальной стенке
    главный шарнир не обязан монотонно двигаться по X.
    """
    if terrain.height_mm <= 1e-9:
        pose = neutral_pose(design)
        return TraversalResult(
            success=True,
            obstacle_kind="step",
            obstacle_value_mm=terrain.height_mm,
            direction=direction,
            poses=(pose,),
            minimum_support_margin_mm=0.0,
        )

    radius = design.wheel_radius_mm
    # Контактные переходы у вертикальной стенки и кромки нельзя
    # дискретизировать так же грубо, как поступательное движение по ровной
    # поверхности: иначе между кадрами можно пропустить столкновение.
    path_step = max(
        2.0,
        min(
            settings.progress_step_mm,
            max(settings.minimum_progress_step_mm, 8.0),
        ),
    )

    local = _design_local_points(design, direction)
    neutral_wheel_x = np.asarray(
        [
            local["front"].x,
            local["bogie_pivot"].x + local["middle_from_bogie"].x,
            local["bogie_pivot"].x + local["rear_from_bogie"].x,
        ]
    )
    traversal_order = tuple(
        int(index) for index in np.argsort(neutral_wheel_x)[::-1]
    )
    leading_x = float(np.max(neutral_wheel_x))
    start_buffer = max(25.0, 2.0 * path_step)
    solution = np.asarray(
        [
            -leading_x - radius - start_buffer,
            radius + design.main_pivot_y_mm,
            0.0,
            0.0,
        ],
        dtype=float,
    )
    modes: dict[int, StepContactMode] = {
        0: "lower",
        1: "lower",
        2: "lower",
    }
    initial_pose = _step_pose_from_modes(design, solution, modes)
    if not _step_frame_has_clearance(
        design,
        terrain,
        initial_pose,
        direction,
        settings,
    ):
        return TraversalResult(
            success=False,
            obstacle_kind="step",
            obstacle_value_mm=terrain.height_mm,
            direction=direction,
            poses=(),
            minimum_support_margin_mm=0.0,
            failure_reason="Исходное положение пересекается со ступенью.",
        )

    poses: list[Pose] = [initial_pose]
    start_target, climb_targets = _step_climb_targets(
        terrain.height_mm,
        radius,
        path_step,
    )

    def append_target(
        controlled_wheel: int,
        target: tuple[float, float, StepContactMode],
    ) -> bool:
        nonlocal solution
        solved = _solve_step_contact_target(
            design,
            terrain,
            direction,
            controlled_wheel,
            target,
            modes,
            solution,
            settings,
        )
        if solved is None:
            return False
        solution, pose = solved
        poses.append(pose)
        return True

    leading_wheel = traversal_order[0]
    initial_frame = calculate_frame(design, initial_pose, direction)
    for target in _linear_contact_targets(
        initial_frame.wheels[leading_wheel].x,
        start_target[0],
        radius,
        "lower",
        path_step,
    ):
        if not append_target(leading_wheel, target):
            return TraversalResult(
                success=False,
                obstacle_kind="step",
                obstacle_value_mm=terrain.height_mm,
                direction=direction,
                poses=tuple(poses),
                minimum_support_margin_mm=0.0,
                failure_reason="Не найден непрерывный подход к ступени.",
            )

    for order_index, wheel_index in enumerate(traversal_order):
        for target in climb_targets:
            if not append_target(wheel_index, target):
                return TraversalResult(
                    success=False,
                    obstacle_kind="step",
                    obstacle_value_mm=terrain.height_mm,
                    direction=direction,
                    poses=tuple(poses),
                    minimum_support_margin_mm=0.0,
                    failure_reason=(
                        "Колесо не может продолжить движение по стенке или "
                        "кромке без потери контакта либо столкновения."
                    ),
                )
        modes[wheel_index] = "top"

        if order_index + 1 < len(traversal_order):
            next_wheel = traversal_order[order_index + 1]
            switch_solution = _solve_step_contact_target(
                design,
                terrain,
                direction,
                next_wheel,
                start_target,
                modes,
                solution,
                settings,
            )
            if switch_solution is None:
                return TraversalResult(
                    success=False,
                    obstacle_kind="step",
                    obstacle_value_mm=terrain.height_mm,
                    direction=direction,
                    poses=tuple(poses),
                    minimum_support_margin_mm=0.0,
                    failure_reason=(
                        "После подъёма колеса не найдено положение, в котором "
                        "следующее колесо подходит к ступени."
                    ),
                )

            target_solution, target_pose = switch_solution
            current_frame = calculate_frame(
                design,
                poses[-1],
                direction,
            )
            target_frame = calculate_frame(
                design,
                target_pose,
                direction,
            )
            translation_targets = _linear_contact_targets(
                current_frame.wheels[wheel_index].x,
                target_frame.wheels[wheel_index].x,
                terrain.height_mm + radius,
                "top",
                path_step,
                include_end=False,
            )
            for target in translation_targets:
                if not append_target(wheel_index, target):
                    return TraversalResult(
                        success=False,
                        obstacle_kind="step",
                        obstacle_value_mm=terrain.height_mm,
                        direction=direction,
                        poses=tuple(poses),
                        minimum_support_margin_mm=0.0,
                        failure_reason=(
                            "Не найден непрерывный переход к следующему колесу."
                        ),
                    )

            solution = target_solution
            poses.append(target_pose)
        else:
            final_frame = calculate_frame(design, poses[-1], direction)
            final_x = final_frame.wheels[wheel_index].x + max(
                50.0,
                2.0 * path_step,
            )
            for target in _linear_contact_targets(
                final_frame.wheels[wheel_index].x,
                final_x,
                terrain.height_mm + radius,
                "top",
                path_step,
            ):
                if not append_target(wheel_index, target):
                    return TraversalResult(
                        success=False,
                        obstacle_kind="step",
                        obstacle_value_mm=terrain.height_mm,
                        direction=direction,
                        poses=tuple(poses),
                        minimum_support_margin_mm=0.0,
                        failure_reason="Не найден выход на верхнюю площадку.",
                    )

    # Последний кадр должен иметь три опоры на верхней площадке.
    final_pose = poses[-1]
    final_frame = calculate_frame(design, final_pose, direction)
    if not (
        set(final_pose.contacts) == set(WHEEL_NAMES)
        and not final_pose.wall_contacts
        and min(wheel.x for wheel in final_frame.wheels) >= -1e-4
    ):
        return TraversalResult(
            success=False,
            obstacle_kind="step",
            obstacle_value_mm=terrain.height_mm,
            direction=direction,
            poses=tuple(poses),
            minimum_support_margin_mm=0.0,
            failure_reason="Последнее колесо не вышло на верхнюю площадку.",
        )

    return TraversalResult(
        success=True,
        obstacle_kind="step",
        obstacle_value_mm=terrain.height_mm,
        direction=direction,
        poses=tuple(poses),
        minimum_support_margin_mm=0.0,
    )

def simulate_traversal(
    design: ChassisDesign,
    terrain: Terrain,
    direction: int = 1,
    settings: SolverSettings | None = None,
) -> TraversalResult:
    """Ищет непрерывную последовательность геометрически допустимых поз."""
    settings = settings or SolverSettings.fine()
    if direction not in (-1, 1):
        raise ValueError("Направление должно быть равно 1 или -1.")
    errors = validate_design(design)
    if errors:
        return TraversalResult(
            success=False,
            obstacle_kind=terrain.kind,
            obstacle_value_mm=terrain.value_mm,
            direction=direction,
            poses=(),
            minimum_support_margin_mm=0.0,
            failure_reason=errors[0],
        )

    if isinstance(terrain, StepTerrain):
        return _simulate_step_contact_traversal(
            design,
            terrain,
            direction,
            settings,
        )

    local = _design_local_points(design, direction)
    neutral_wheel_x = (
        local["front"].x,
        local["bogie_pivot"].x + local["middle_from_bogie"].x,
        local["bogie_pivot"].x + local["rear_from_bogie"].x,
    )
    leading_x = max(neutral_wheel_x)
    trailing_x = min(neutral_wheel_x)
    start_buffer_mm = max(25.0, 2.0 * settings.progress_step_mm)
    # После препятствия оставляется участок для возвращения пассивных звеньев
    # в горизонтальное положение. Иначе поиск мог завершиться раньше, чем
    # последнее колесо опустится на поверхность.
    end_buffer_mm = max(
        start_buffer_mm,
        0.60 * design.front_wheel_x_mm,
    )
    main_x_start = (
        -leading_x - design.wheel_radius_mm - start_buffer_mm
    )
    obstacle_end = (
        terrain.width_mm if isinstance(terrain, DitchTerrain) else 0.0
    )
    main_x_end = (
        obstacle_end
        - trailing_x
        + design.wheel_radius_mm
        + end_buffer_mm
    )

    initial_pairs = np.asarray([[0.0, 0.0]])
    initial_poses = _evaluate_angle_candidates(
        design,
        terrain,
        float(main_x_start),
        direction,
        initial_pairs,
        settings,
    )
    if not initial_poses:
        return TraversalResult(
            success=False,
            obstacle_kind=terrain.kind,
            obstacle_value_mm=terrain.value_mm,
            direction=direction,
            poses=(),
            minimum_support_margin_mm=0.0,
            failure_reason="Исходное горизонтальное положение недопустимо.",
        )

    initial = min(
        initial_poses,
        key=lambda pose: abs(pose.body_pitch_deg) + abs(pose.bogie_angle_deg),
    )
    states = [_State(initial, (initial,), 0.0)]
    current_main_x = float(main_x_start)
    current_step = settings.progress_step_mm
    accepted_steps = 0

    while current_main_x < main_x_end - 1e-9:
        main_x = min(current_main_x + current_step, main_x_end)
        angle_pairs = _angle_candidates(states, design, settings)
        angle_pairs = _enrich_step_wall_angles(
            design,
            terrain,
            float(main_x),
            direction,
            angle_pairs,
            states,
            settings,
        )
        poses = _evaluate_angle_candidates(
            design,
            terrain,
            float(main_x),
            direction,
            angle_pairs,
            settings,
            preferred_main_y=tuple(
                state.pose.main_pivot_y_mm for state in states
            ),
        )

        # Для каждой конечной позы сохраняются две ближайшие возможные ветви.
        connections: list[tuple[Pose, _State, float]] = []
        for pose in poses:
            parent_distances: list[tuple[float, _State]] = []
            for state in states:
                distance = _pose_distance(state.pose, pose, settings)
                if np.isfinite(distance):
                    parent_distances.append((distance, state))
            if not parent_distances:
                continue
            parent_distances.sort(key=lambda item: item[0])
            for distance, state in parent_distances[:6]:
                connections.append((pose, state, distance))
        connections.sort(key=lambda item: item[2])
        connections = connections[: max(60, settings.beam_width * 12)]

        midpoint_by_connection: dict[int, tuple[Pose, float]] = {}
        if settings.validate_transition_midpoint and connections:
            midpoint_x = (current_main_x + main_x) / 2.0
            midpoint_pairs = _angle_candidates(states, design, settings)
            midpoint_pairs = _enrich_step_wall_angles(
                design,
                terrain,
                midpoint_x,
                direction,
                midpoint_pairs,
                states,
                settings,
            )
            midpoint_poses = _evaluate_angle_candidates(
                design,
                terrain,
                midpoint_x,
                direction,
                midpoint_pairs,
                settings,
                preferred_main_y=tuple(
                    state.pose.main_pivot_y_mm for state in states
                ),
            )
            midpoint_poses.sort(
                key=lambda midpoint: min(
                    _pose_distance(state.pose, midpoint, settings)
                    for state in states
                )
            )
            midpoint_poses = midpoint_poses[
                : max(60, settings.beam_width * 12)
            ]
            for index, (pose, parent, _) in enumerate(connections):
                best_midpoint: Pose | None = None
                best_midpoint_cost = float("inf")
                for midpoint in midpoint_poses:
                    first_distance = _pose_distance(
                        parent.pose,
                        midpoint,
                        settings,
                    )
                    second_distance = _pose_distance(
                        midpoint,
                        pose,
                        settings,
                    )
                    total_distance = first_distance + second_distance
                    if total_distance < best_midpoint_cost:
                        best_midpoint = midpoint
                        best_midpoint_cost = total_distance
                if best_midpoint is not None and np.isfinite(best_midpoint_cost):
                    midpoint_by_connection[index] = (
                        best_midpoint,
                        best_midpoint_cost,
                    )

        next_states: list[_State] = []
        for connection_index, (pose, parent, distance) in enumerate(connections):
            midpoint_data = midpoint_by_connection.get(connection_index)
            if settings.validate_transition_midpoint and midpoint_data is None:
                continue
            midpoint = midpoint_data[0] if midpoint_data is not None else None
            transition_cost = (
                midpoint_data[1]
                if midpoint_data is not None
                else distance
            )
            # Небольшое предпочтение отдаётся позам с большим запасом опоры.
            margin_bonus = (
                min(max(pose.support_margin_mm, 0.0), 200.0) / 200.0
            )
            wall_progress_bonus = 0.0
            if (
                isinstance(terrain, StepTerrain)
                and terrain.height_mm > 1e-9
                and pose.wall_contacts
            ):
                frame = calculate_frame(design, pose, direction)
                wheel_by_name = dict(zip(WHEEL_NAMES, frame.wheels))
                wall_height = max(
                    wheel_by_name[name].y
                    for name in pose.wall_contacts
                )
                wall_progress_bonus = min(
                    1.0,
                    max(0.0, wall_height / terrain.height_mm),
                )
            contact_bonus = max(0, len(pose.contacts) - 1)
            neutral_angle_penalty = 1.0 * (
                abs(pose.body_pitch_deg)
                + 0.5 * abs(pose.bogie_angle_deg)
            )
            path_extension = (
                (midpoint, pose)
                if midpoint is not None
                else (pose,)
            )
            next_states.append(
                _State(
                    pose=pose,
                    path=parent.path + path_extension,
                    cost=(
                        0.94 * parent.cost
                        + transition_cost
                        + neutral_angle_penalty
                        - 0.08 * margin_bonus
                        - 5.0 * wall_progress_bonus
                        - 2.0 * contact_bonus
                    ),
                )
            )

        next_states = _deduplicate_states(next_states, settings.beam_width)
        if next_states:
            states = next_states
            current_main_x = float(main_x)
            current_step = min(
                settings.progress_step_mm,
                max(
                    settings.minimum_progress_step_mm,
                    current_step * 1.5,
                ),
            )
            accepted_steps += 1
            if accepted_steps > 5000:
                return TraversalResult(
                    success=False,
                    obstacle_kind=terrain.kind,
                    obstacle_value_mm=terrain.value_mm,
                    direction=direction,
                    poses=(),
                    minimum_support_margin_mm=0.0,
                    failure_reason="Превышено допустимое число шагов поиска.",
                )
            continue

        reduced_step = current_step / 2.0
        if reduced_step < settings.minimum_progress_step_mm - 1e-9:
            if current_step > settings.minimum_progress_step_mm + 1e-9:
                current_step = settings.minimum_progress_step_mm
                continue
            return TraversalResult(
                success=False,
                obstacle_kind=terrain.kind,
                obstacle_value_mm=terrain.value_mm,
                direction=direction,
                poses=states[0].path if states else (),
                minimum_support_margin_mm=0.0,
                failure_reason=(
                    "Не найден непрерывный переход без столкновения "
                    "и потери геометрической устойчивости "
                    f"около X={current_main_x:.1f} мм."
                ),
            )
        current_step = reduced_step

    def completed_states() -> list[_State]:
        return [
            state
            for state in states
            if _finish_reached(
                calculate_frame(design, state.pose, direction),
                state.pose,
                terrain,
            )
        ]

    # На ровном участке после препятствия пассивные звенья должны вернуться
    # к исходной позе. Этот отдельный этап выполняется при неизменном X, чтобы
    # результат не зависел от произвольно выбранной длины выходного участка.
    completed = completed_states()
    previous_settle_score = float("inf")
    stagnant_iterations = 0
    for _ in range(80):
        if completed:
            break
        settle_pairs = _angle_candidates(states, design, settings)
        settle_poses = _evaluate_angle_candidates(
            design,
            terrain,
            current_main_x,
            direction,
            settle_pairs,
            settings,
            preferred_main_y=tuple(
                state.pose.main_pivot_y_mm for state in states
            ),
        )
        settle_states: list[_State] = []
        for pose in settle_poses:
            best_parent: _State | None = None
            best_distance = float("inf")
            for state in states:
                distance = _pose_distance(state.pose, pose, settings)
                if distance < best_distance:
                    best_parent = state
                    best_distance = distance
            if best_parent is None or not np.isfinite(best_distance):
                continue
            unsupported = 3 - len(pose.contacts)
            settle_score = (
                abs(pose.body_pitch_deg)
                + 0.5 * abs(pose.bogie_angle_deg)
                + 20.0 * unsupported
            )
            settle_states.append(
                _State(
                    pose=pose,
                    path=best_parent.path + (pose,),
                    cost=settle_score,
                )
            )
        if not settle_states:
            break
        states = _deduplicate_states(settle_states, settings.beam_width)
        current_score = min(state.cost for state in states)
        if current_score >= previous_settle_score - 1e-8:
            stagnant_iterations += 1
        else:
            stagnant_iterations = 0
        previous_settle_score = current_score
        if stagnant_iterations >= 5:
            break
        completed = completed_states()

    if not completed:
        return TraversalResult(
            success=False,
            obstacle_kind=terrain.kind,
            obstacle_value_mm=terrain.value_mm,
            direction=direction,
            poses=states[0].path if states else (),
            minimum_support_margin_mm=0.0,
            failure_reason="Последнее колесо не достигло второго уровня или берега.",
        )

    best = min(completed, key=lambda state: state.cost)
    minimum_margin = (
        min(pose.support_margin_mm for pose in best.path)
        if terrain.kind == "ditch"
        else 0.0
    )
    return TraversalResult(
        success=True,
        obstacle_kind=terrain.kind,
        obstacle_value_mm=terrain.value_mm,
        direction=direction,
        poses=best.path,
        minimum_support_margin_mm=minimum_margin,
    )


def latin_hypercube_designs(
    base: ChassisDesign,
    free_ranges: dict[str, tuple[float, float]],
    count: int,
    seed: int,
) -> list[ChassisDesign]:
    """Создаёт равномерно распределённый набор вариантов без SciPy."""
    if count <= 0:
        return []
    if not free_ranges:
        return [base]

    target_count = max(1, count)
    rng = np.random.default_rng(seed)
    names = tuple(free_ranges)
    dimensions = len(names)
    designs: list[ChassisDesign] = [base]
    seen = {base}
    attempts = 0
    while len(designs) < target_count and attempts < 12:
        batch_count = max(target_count, 2 * (target_count - len(designs)))
        samples = np.empty((batch_count, dimensions), dtype=float)
        for dimension in range(dimensions):
            values = (
                np.arange(batch_count) + rng.random(batch_count)
            ) / batch_count
            rng.shuffle(values)
            samples[:, dimension] = values

        for row in samples:
            changes: dict[str, float] = {}
            for index, name in enumerate(names):
                minimum, maximum = free_ranges[name]
                changes[name] = (
                    minimum + row[index] * (maximum - minimum)
                )
            candidate = replace(base, **changes)
            if candidate in seen or validate_design(candidate):
                continue
            designs.append(candidate)
            seen.add(candidate)
            if len(designs) >= target_count:
                break
        attempts += 1
    return designs
