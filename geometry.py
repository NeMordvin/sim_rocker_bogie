from dataclasses import dataclass
from math import (
    asin,
    atan,
    atan2,
    ceil,
    cos,
    degrees,
    hypot,
    radians,
    sin,
    sqrt,
    tan,
)


GRAVITY = 9.80665
ANGLE_SEARCH_STEP_DEG = 0.25


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def point_from(start: Point, length: float, angle_deg: float) -> Point:
    """Строит точку на заданном расстоянии и под заданным углом."""
    angle_rad = radians(angle_deg)
    return Point(
        x=start.x + length * cos(angle_rad),
        y=start.y + length * sin(angle_rad),
    )


@dataclass(frozen=True)
class ChassisPoints:
    main_pivot: Point
    front_wheel: Point
    bogie_pivot: Point
    middle_wheel: Point
    rear_wheel: Point


def calculate_chassis_points(
    main_pivot: Point,
    front_length: float,
    front_angle: float,
    bogie_length: float,
    bogie_angle: float,
    middle_length: float,
    middle_angle: float,
    rear_length: float,
    rear_angle: float,
    rocker_rotation: float = 0.0,
    bogie_rotation: float = 0.0,
) -> ChassisPoints:
    """Рассчитывает положение шарниров и центров колёс."""
    front_wheel = point_from(
        main_pivot,
        front_length,
        front_angle + rocker_rotation,
    )
    bogie_pivot = point_from(
        main_pivot,
        bogie_length,
        bogie_angle + rocker_rotation,
    )
    middle_wheel = point_from(
        bogie_pivot,
        middle_length,
        middle_angle + bogie_rotation,
    )
    rear_wheel = point_from(
        bogie_pivot,
        rear_length,
        rear_angle + bogie_rotation,
    )

    return ChassisPoints(
        main_pivot=main_pivot,
        front_wheel=front_wheel,
        bogie_pivot=bogie_pivot,
        middle_wheel=middle_wheel,
        rear_wheel=rear_wheel,
    )


def calculate_wheel_clearances(
    points: ChassisPoints,
    wheel_radius: float,
) -> dict[str, float]:
    """Возвращает расстояние от нижней точки каждого колеса до земли."""
    return {
        "Переднее": points.front_wheel.y - wheel_radius,
        "Среднее": points.middle_wheel.y - wheel_radius,
        "Заднее": points.rear_wheel.y - wheel_radius,
    }


@dataclass(frozen=True)
class BodyPoints:
    bottom_left: Point
    bottom_right: Point
    top_right: Point
    top_left: Point


def calculate_body_points(
    main_pivot: Point,
    body_length: float,
    body_height: float,
    attachment_x: float,
    attachment_y: float,
) -> BodyPoints:
    """Строит прямоугольный корпус относительно главного шарнира."""
    bottom_left = Point(
        x=main_pivot.x - attachment_x,
        y=main_pivot.y - attachment_y,
    )
    bottom_right = Point(
        x=bottom_left.x + body_length,
        y=bottom_left.y,
    )
    top_right = Point(
        x=bottom_left.x + body_length,
        y=bottom_left.y + body_height,
    )
    top_left = Point(
        x=bottom_left.x,
        y=bottom_left.y + body_height,
    )

    return BodyPoints(
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        top_right=top_right,
        top_left=top_left,
    )


@dataclass(frozen=True)
class SuspensionParameters:
    wheel_radius_mm: float
    front_length_mm: float
    front_angle_deg: float
    bogie_length_mm: float
    bogie_angle_deg: float
    middle_length_mm: float
    middle_angle_deg: float
    rear_length_mm: float
    rear_angle_deg: float
    rocker_min_deg: float
    rocker_max_deg: float
    bogie_min_deg: float
    bogie_max_deg: float
    body_length_mm: float
    body_height_mm: float
    attachment_x_mm: float
    attachment_y_mm: float
    arm_radius_mm: float


@dataclass(frozen=True)
class VehicleParameters:
    total_mass_kg: float
    track_width_mm: float
    com_x_from_body_left_mm: float
    com_height_mm: float
    lateral_com_offset_mm: float
    wheel_torque_nm: float
    brake_torque_nm: float
    driven_wheels: int
    drivetrain_efficiency: float
    ground_friction: float
    obstacle_friction: float
    rolling_resistance: float
    support_margin_mm: float
    safety_factor: float


@dataclass(frozen=True)
class LimitResult:
    value: float
    unit: str
    limiting_factor: str
    theoretical_value: float | None = None


@dataclass(frozen=True)
class MobilityResults:
    step: LimitResult
    ditch: LimitResult
    uphill: LimitResult
    downhill: LimitResult
    cross_slope: LimitResult
    approach_angle: LimitResult
    departure_angle: LimitResult
    breakover_angle: LimitResult
    ground_clearance: LimitResult
    step_front_phase_mm: float
    step_rear_phase_mm: float
    step_force_limit_mm: float
    wall_climb_possible: bool
    flat_height_error_mm: float
    warnings: tuple[str, ...]


def _validate_parameters(
    suspension: SuspensionParameters,
    vehicle: VehicleParameters,
) -> None:
    positive_values = {
        "радиус колеса": suspension.wheel_radius_mm,
        "длина переднего плеча": suspension.front_length_mm,
        "длина плеча до bogie": suspension.bogie_length_mm,
        "длина плеча среднего колеса": suspension.middle_length_mm,
        "длина плеча заднего колеса": suspension.rear_length_mm,
        "длина корпуса": suspension.body_length_mm,
        "высота корпуса": suspension.body_height_mm,
        "масса": vehicle.total_mass_kg,
        "колея": vehicle.track_width_mm,
        "высота центра масс": vehicle.com_height_mm,
        "коэффициент запаса": vehicle.safety_factor,
    }
    for name, value in positive_values.items():
        if value <= 0.0:
            raise ValueError(f"Параметр «{name}» должен быть больше нуля.")

    if suspension.arm_radius_mm < 0.0:
        raise ValueError("Радиус сечения рычага не может быть отрицательным.")
    if suspension.rocker_min_deg >= suspension.rocker_max_deg:
        raise ValueError("Минимальный угол rocker должен быть меньше максимального.")
    if suspension.bogie_min_deg >= suspension.bogie_max_deg:
        raise ValueError("Минимальный угол bogie должен быть меньше максимального.")
    if not 0.0 <= suspension.attachment_x_mm <= suspension.body_length_mm:
        raise ValueError("Крепление по X находится за пределами корпуса.")
    if not 0.0 <= suspension.attachment_y_mm <= suspension.body_height_mm:
        raise ValueError("Крепление по Y находится за пределами корпуса.")
    if not 0.0 < vehicle.drivetrain_efficiency <= 1.0:
        raise ValueError("КПД привода должен находиться в диапазоне от 0 до 1.")
    if not 1 <= vehicle.driven_wheels <= 6:
        raise ValueError("Количество ведущих колёс должно быть от 1 до 6.")
    if vehicle.wheel_torque_nm < 0.0 or vehicle.brake_torque_nm < 0.0:
        raise ValueError("Момент двигателя и тормозной момент не могут быть отрицательными.")
    if vehicle.ground_friction < 0.0 or vehicle.obstacle_friction < 0.0:
        raise ValueError("Коэффициент трения не может быть отрицательным.")
    if vehicle.rolling_resistance < 0.0:
        raise ValueError("Коэффициент сопротивления качению не может быть отрицательным.")
    if vehicle.support_margin_mm < 0.0:
        raise ValueError("Запас до края опоры не может быть отрицательным.")


def _relative_points(
    suspension: SuspensionParameters,
    rocker_rotation_deg: float,
    bogie_rotation_deg: float,
) -> ChassisPoints:
    return calculate_chassis_points(
        main_pivot=Point(0.0, 0.0),
        front_length=suspension.front_length_mm,
        front_angle=suspension.front_angle_deg,
        bogie_length=suspension.bogie_length_mm,
        bogie_angle=suspension.bogie_angle_deg,
        middle_length=suspension.middle_length_mm,
        middle_angle=suspension.middle_angle_deg,
        rear_length=suspension.rear_length_mm,
        rear_angle=suspension.rear_angle_deg,
        rocker_rotation=rocker_rotation_deg,
        bogie_rotation=bogie_rotation_deg,
    )


def _rotation_values(minimum: float, maximum: float) -> list[float]:
    count = max(1, ceil((maximum - minimum) / ANGLE_SEARCH_STEP_DEG))
    return [
        minimum + (maximum - minimum) * index / count
        for index in range(count + 1)
    ]


def _unique_angles(values: list[float], tolerance: float = 1e-7) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def _level_bogie_roots(suspension: SuspensionParameters) -> list[float]:
    """Находит углы bogie, при которых среднее и заднее колёса на одной высоте."""
    middle_angle = radians(suspension.middle_angle_deg)
    rear_angle = radians(suspension.rear_angle_deg)

    coefficient_sin = (
        suspension.middle_length_mm * cos(middle_angle)
        - suspension.rear_length_mm * cos(rear_angle)
    )
    coefficient_cos = (
        suspension.middle_length_mm * sin(middle_angle)
        - suspension.rear_length_mm * sin(rear_angle)
    )

    if hypot(coefficient_sin, coefficient_cos) < 1e-9:
        return [0.0]

    phase_deg = degrees(atan2(coefficient_cos, coefficient_sin))
    roots = []
    for period in range(-4, 5):
        value = period * 180.0 - phase_deg
        if suspension.bogie_min_deg - 1e-9 <= value <= suspension.bogie_max_deg + 1e-9:
            roots.append(value)
    return _unique_angles(roots)


def _sine_rotation_roots(
    length: float,
    base_angle_deg: float,
    target: float,
    minimum_deg: float,
    maximum_deg: float,
) -> list[float]:
    if length <= 0.0 or abs(target) > length + 1e-8:
        return []

    ratio = max(-1.0, min(1.0, target / length))
    first = degrees(asin(ratio))
    total_angles = []

    for period in range(-3, 4):
        total_angles.append(first + period * 360.0)
        total_angles.append(180.0 - first + period * 360.0)

    rotations = [
        total_angle - base_angle_deg
        for total_angle in total_angles
        if minimum_deg - 1e-9
        <= total_angle - base_angle_deg
        <= maximum_deg + 1e-9
    ]
    return _unique_angles(rotations)


def _wheel_order_is_valid(points: ChassisPoints) -> bool:
    return (
        points.front_wheel.x > points.middle_wheel.x + 1e-6
        and points.middle_wheel.x > points.rear_wheel.x + 1e-6
    )


def _longitudinal_stability_is_valid(
    points: ChassisPoints,
    com_x_mm: float,
    margin_mm: float,
) -> bool:
    wheel_x = [
        points.front_wheel.x,
        points.middle_wheel.x,
        points.rear_wheel.x,
    ]
    return (
        min(wheel_x) + margin_mm
        <= com_x_mm
        <= max(wheel_x) - margin_mm
    )


def _shift_points_vertically(
    points: ChassisPoints,
    offset_y: float,
) -> ChassisPoints:
    def shifted(point: Point) -> Point:
        return Point(point.x, point.y + offset_y)

    return ChassisPoints(
        main_pivot=shifted(points.main_pivot),
        front_wheel=shifted(points.front_wheel),
        bogie_pivot=shifted(points.bogie_pivot),
        middle_wheel=shifted(points.middle_wheel),
        rear_wheel=shifted(points.rear_wheel),
    )


def _segment_above_step(
    start: Point,
    end: Point,
    edge_x: float,
    step_height: float,
    clearance: float,
) -> bool:
    """Проверяет, что рычаг не пересекает землю и вертикальную ступень."""
    candidates = [0.0, 1.0]
    delta_x = end.x - start.x

    if abs(delta_x) > 1e-12:
        crossing = (edge_x - start.x) / delta_x
        if 0.0 <= crossing <= 1.0:
            candidates.extend(
                [
                    max(0.0, crossing - 1e-8),
                    crossing,
                    min(1.0, crossing + 1e-8),
                ]
            )

    for position in candidates:
        x = start.x + (end.x - start.x) * position
        y = start.y + (end.y - start.y) * position
        terrain_height = step_height if x >= edge_x else 0.0
        if y - clearance < terrain_height - 1e-6:
            return False
    return True


def _step_configuration_is_feasible(
    relative_points: ChassisPoints,
    step_height: float,
    lower_wheels: tuple[str, ...],
    upper_wheels: tuple[str, ...],
    lower_reference_y: float,
    suspension: SuspensionParameters,
    vehicle: VehicleParameters,
    com_x_mm: float,
) -> bool:
    if step_height <= 0.0 or not _wheel_order_is_valid(relative_points):
        return False
    if not _longitudinal_stability_is_valid(
        relative_points,
        com_x_mm,
        vehicle.support_margin_mm,
    ):
        return False

    radius = suspension.wheel_radius_mm
    pivot_y = radius - lower_reference_y
    points = _shift_points_vertically(relative_points, pivot_y)

    wheel_points = {
        "front": points.front_wheel,
        "middle": points.middle_wheel,
        "rear": points.rear_wheel,
    }
    lower_points = [wheel_points[name] for name in lower_wheels]
    upper_points = [wheel_points[name] for name in upper_wheels]

    # Нижнее колесо должно пройти рядом с вертикальной гранью, не пересекая её.
    if step_height >= radius:
        edge_clearance = radius
    else:
        edge_clearance = sqrt(
            max(0.0, 2.0 * radius * step_height - step_height**2)
        )

    edge_min = max(point.x + edge_clearance for point in lower_points)
    edge_max = min(point.x for point in upper_points)
    if edge_min > edge_max + 1e-6:
        return False

    body_left = -suspension.attachment_x_mm
    body_right = body_left + suspension.body_length_mm
    body_bottom = pivot_y - suspension.attachment_y_mm
    if body_bottom < -1e-6:
        return False

    segments = [
        (points.front_wheel, points.main_pivot),
        (points.main_pivot, points.bogie_pivot),
        (points.middle_wheel, points.bogie_pivot),
        (points.bogie_pivot, points.rear_wheel),
    ]

    edge_samples = 48
    for index in range(edge_samples + 1):
        edge_x = edge_min + (edge_max - edge_min) * index / edge_samples

        # Если корпус нависает над верхней площадкой, его днище должно быть выше неё.
        if body_right >= edge_x and body_bottom < step_height - 1e-6:
            continue

        if all(
            _segment_above_step(
                start,
                end,
                edge_x,
                step_height,
                suspension.arm_radius_mm,
            )
            for start, end in segments
        ):
            return True

    return False


def _front_step_phase_limit(
    suspension: SuspensionParameters,
    vehicle: VehicleParameters,
    com_x_mm: float,
) -> float:
    """Переднее колесо наверху, среднее и заднее — на нижней площадке."""
    best_height = 0.0

    for bogie_rotation in _level_bogie_roots(suspension):
        for rocker_rotation in _rotation_values(
            suspension.rocker_min_deg,
            suspension.rocker_max_deg,
        ):
            points = _relative_points(
                suspension,
                rocker_rotation,
                bogie_rotation,
            )
            step_height = points.front_wheel.y - points.middle_wheel.y

            if step_height <= best_height:
                continue
            if _step_configuration_is_feasible(
                relative_points=points,
                step_height=step_height,
                lower_wheels=("middle", "rear"),
                upper_wheels=("front",),
                lower_reference_y=points.middle_wheel.y,
                suspension=suspension,
                vehicle=vehicle,
                com_x_mm=com_x_mm,
            ):
                best_height = step_height

    return best_height


def _rear_step_phase_limit(
    suspension: SuspensionParameters,
    vehicle: VehicleParameters,
    com_x_mm: float,
) -> float:
    """Переднее и среднее колёса наверху, заднее — на нижней площадке."""
    best_height = 0.0

    for rocker_rotation in _rotation_values(
        suspension.rocker_min_deg,
        suspension.rocker_max_deg,
    ):
        without_bogie = _relative_points(
            suspension,
            rocker_rotation,
            0.0,
        )
        required_middle_offset = (
            without_bogie.front_wheel.y
            - without_bogie.bogie_pivot.y
        )

        roots = _sine_rotation_roots(
            length=suspension.middle_length_mm,
            base_angle_deg=suspension.middle_angle_deg,
            target=required_middle_offset,
            minimum_deg=suspension.bogie_min_deg,
            maximum_deg=suspension.bogie_max_deg,
        )

        for bogie_rotation in roots:
            points = _relative_points(
                suspension,
                rocker_rotation,
                bogie_rotation,
            )
            step_height = points.front_wheel.y - points.rear_wheel.y

            if step_height <= best_height:
                continue
            if _step_configuration_is_feasible(
                relative_points=points,
                step_height=step_height,
                lower_wheels=("rear",),
                upper_wheels=("front", "middle"),
                lower_reference_y=points.rear_wheel.y,
                suspension=suspension,
                vehicle=vehicle,
                com_x_mm=com_x_mm,
            ):
                best_height = step_height

    return best_height


def _rolling_step_limit(
    wheel_radius_mm: float,
    available_push_n: float,
    required_vertical_load_n: float,
) -> float:
    """Предел перекатывания колеса через кромку без движения по вертикальной стенке."""
    if available_push_n <= 0.0 or required_vertical_load_n <= 0.0:
        return 0.0

    low = 0.0
    high = wheel_radius_mm * (1.0 - 1e-7)

    for _ in range(70):
        height = (low + high) / 2.0
        numerator = sqrt(
            max(0.0, 2.0 * wheel_radius_mm * height - height**2)
        )
        denominator = max(1e-9, wheel_radius_mm - height)
        required_push = required_vertical_load_n * numerator / denominator

        if required_push <= available_push_n:
            low = height
        else:
            high = height

    return low


def _calculate_step_limits(
    suspension: SuspensionParameters,
    vehicle: VehicleParameters,
    com_x_mm: float,
) -> tuple[LimitResult, float, float, float, bool]:
    front_phase = _front_step_phase_limit(suspension, vehicle, com_x_mm)
    rear_phase = _rear_step_phase_limit(suspension, vehicle, com_x_mm)
    geometric_limit = min(front_phase, rear_phase)

    if front_phase <= rear_phase:
        geometric_factor = "кинематика и зазоры при подъёме передней оси"
    else:
        geometric_factor = "кинематика и зазоры при подъёме задней оси"

    weight = vehicle.total_mass_kg * GRAVITY
    radius_m = suspension.wheel_radius_mm / 1000.0
    front_axle_load = weight / 3.0
    supporting_load = 2.0 * weight / 3.0

    front_driven = min(2, vehicle.driven_wheels)
    supporting_driven = max(0, vehicle.driven_wheels - front_driven)

    support_motor_force = (
        supporting_driven
        * vehicle.wheel_torque_nm
        * vehicle.drivetrain_efficiency
        / radius_m
    )
    support_traction_force = (
        vehicle.ground_friction
        * weight
        * supporting_driven
        / 6.0
    )
    normal_push = max(
        0.0,
        min(support_motor_force, support_traction_force)
        - vehicle.rolling_resistance * supporting_load,
    )

    front_motor_force = (
        front_driven
        * vehicle.wheel_torque_nm
        * vehicle.drivetrain_efficiency
        / radius_m
    )
    vertical_wall_force = min(
        front_motor_force,
        vehicle.obstacle_friction * normal_push,
    )
    wall_climb_possible = (
        vertical_wall_force
        >= front_axle_load * vehicle.safety_factor
    )

    if wall_climb_possible:
        force_limit = geometric_limit
    else:
        total_motor_force = (
            vehicle.driven_wheels
            * vehicle.wheel_torque_nm
            * vehicle.drivetrain_efficiency
            / radius_m
        )
        total_traction_force = (
            vehicle.ground_friction
            * weight
            * vehicle.driven_wheels
            / 6.0
        )
        available_push = max(
            0.0,
            min(total_motor_force, total_traction_force)
            - vehicle.rolling_resistance * weight,
        )
        force_limit = _rolling_step_limit(
            wheel_radius_mm=suspension.wheel_radius_mm,
            available_push_n=available_push,
            required_vertical_load_n=front_axle_load * vehicle.safety_factor,
        )

    geometric_with_reserve = geometric_limit / vehicle.safety_factor
    calculated_limit = min(geometric_with_reserve, force_limit)

    if force_limit < geometric_with_reserve - 0.5:
        limiting_factor = "момент и сцепление на кромке ступени"
    else:
        limiting_factor = geometric_factor

    return (
        LimitResult(
            value=max(0.0, calculated_limit),
            unit="мм",
            limiting_factor=limiting_factor,
            theoretical_value=max(0.0, geometric_limit),
        ),
        front_phase,
        rear_phase,
        force_limit,
        wall_climb_possible,
    )


def _ditch_is_crossable(
    width_mm: float,
    wheel_x: list[float],
    com_x_mm: float,
    edge_reach_mm: float,
    support_margin_mm: float,
) -> bool:
    """Проверяет все состояния опор при пересечении глубокого рва."""
    if not (
        min(wheel_x) + support_margin_mm
        <= com_x_mm
        <= max(wheel_x) - support_margin_mm
    ):
        return False
    if width_mm <= 2.0 * edge_reach_mm:
        return True

    events = []
    for x in wheel_x:
        events.append(edge_reach_mm - x)
        events.append(width_mm - edge_reach_mm - x)
    events = sorted(set(events))

    for left, right in zip(events, events[1:]):
        if right - left < 1e-9:
            continue
        translation = (left + right) / 2.0
        supported = [
            x
            for x in wheel_x
            if translation + x <= edge_reach_mm
            or translation + x >= width_mm - edge_reach_mm
        ]

        if len(supported) < 2:
            return False
        if not (
            min(supported) + support_margin_mm
            <= com_x_mm
            <= max(supported) - support_margin_mm
        ):
            return False

    return True


def _calculate_ditch_limit(
    suspension: SuspensionParameters,
    vehicle: VehicleParameters,
    flat_points: ChassisPoints,
    com_x_mm: float,
) -> LimitResult:
    wheel_x = [
        flat_points.front_wheel.x,
        flat_points.middle_wheel.x,
        flat_points.rear_wheel.x,
    ]
    if not (
        min(wheel_x) + vehicle.support_margin_mm
        <= com_x_mm
        <= max(wheel_x) - vehicle.support_margin_mm
    ):
        return LimitResult(
            value=0.0,
            unit="мм",
            limiting_factor="центр масс вне допустимой области опоры",
            theoretical_value=0.0,
        )

    wheel_span = max(wheel_x) - min(wheel_x)
    search_high = wheel_span + 2.0 * suspension.wheel_radius_mm

    low = 0.0
    high = search_high
    for _ in range(60):
        width = (low + high) / 2.0
        if _ditch_is_crossable(
            width_mm=width,
            wheel_x=wheel_x,
            com_x_mm=com_x_mm,
            edge_reach_mm=suspension.wheel_radius_mm,
            support_margin_mm=vehicle.support_margin_mm,
        ):
            low = width
        else:
            high = width

    return LimitResult(
        value=low / vehicle.safety_factor,
        unit="мм",
        limiting_factor="расстановка осей и положение центра масс",
        theoretical_value=low,
    )


def _force_limited_angle(
    force_capacity_n: float,
    weight_n: float,
    rolling_resistance: float,
    safety_factor: float,
    descending: bool,
) -> float:
    low = 0.0
    high = radians(89.9)

    for _ in range(70):
        angle = (low + high) / 2.0
        if descending:
            required = weight_n * max(
                0.0,
                sin(angle) - rolling_resistance * cos(angle),
            )
        else:
            required = weight_n * (
                sin(angle) + rolling_resistance * cos(angle)
            )

        if force_capacity_n >= required * safety_factor:
            low = angle
        else:
            high = angle

    return degrees(low)


def _calculate_slope_limits(
    suspension: SuspensionParameters,
    vehicle: VehicleParameters,
    flat_points: ChassisPoints,
    com_x_mm: float,
) -> tuple[LimitResult, LimitResult, LimitResult]:
    wheel_x = [
        flat_points.front_wheel.x,
        flat_points.middle_wheel.x,
        flat_points.rear_wheel.x,
    ]
    rear_margin = max(0.0, com_x_mm - min(wheel_x))
    front_margin = max(0.0, max(wheel_x) - com_x_mm)
    lateral_margin = max(
        0.0,
        vehicle.track_width_mm / 2.0 - abs(vehicle.lateral_com_offset_mm),
    )

    uphill_stability = degrees(
        atan(rear_margin / (vehicle.com_height_mm * vehicle.safety_factor))
    )
    downhill_stability = degrees(
        atan(front_margin / (vehicle.com_height_mm * vehicle.safety_factor))
    )
    cross_stability = degrees(
        atan(lateral_margin / (vehicle.com_height_mm * vehicle.safety_factor))
    )

    longitudinal_adhesion = (
        vehicle.ground_friction * vehicle.driven_wheels / 6.0
    )
    uphill_traction_tangent = max(
        0.0,
        longitudinal_adhesion / vehicle.safety_factor
        - vehicle.rolling_resistance,
    )
    downhill_traction_tangent = max(
        0.0,
        longitudinal_adhesion / vehicle.safety_factor
        + vehicle.rolling_resistance,
    )
    uphill_traction = degrees(atan(uphill_traction_tangent))
    downhill_traction = degrees(atan(downhill_traction_tangent))
    cross_traction = degrees(
        atan(vehicle.ground_friction / vehicle.safety_factor)
    )

    radius_m = suspension.wheel_radius_mm / 1000.0
    weight = vehicle.total_mass_kg * GRAVITY
    drive_force = (
        vehicle.driven_wheels
        * vehicle.wheel_torque_nm
        * vehicle.drivetrain_efficiency
        / radius_m
    )
    brake_force = (
        vehicle.driven_wheels
        * vehicle.brake_torque_nm
        * vehicle.drivetrain_efficiency
        / radius_m
    )

    uphill_motor = _force_limited_angle(
        drive_force,
        weight,
        vehicle.rolling_resistance,
        vehicle.safety_factor,
        descending=False,
    )
    downhill_brake = _force_limited_angle(
        brake_force,
        weight,
        vehicle.rolling_resistance,
        vehicle.safety_factor,
        descending=True,
    )

    uphill_candidates = {
        "продольная устойчивость": uphill_stability,
        "сцепление с грунтом": uphill_traction,
        "момент мотор-колёс": uphill_motor,
    }
    downhill_candidates = {
        "продольная устойчивость": downhill_stability,
        "сцепление с грунтом": downhill_traction,
        "тормозной момент": downhill_brake,
    }
    cross_candidates = {
        "поперечная устойчивость": cross_stability,
        "боковое сцепление": cross_traction,
    }

    uphill_factor = min(uphill_candidates, key=uphill_candidates.get)
    downhill_factor = min(downhill_candidates, key=downhill_candidates.get)
    cross_factor = min(cross_candidates, key=cross_candidates.get)

    return (
        LimitResult(
            value=uphill_candidates[uphill_factor],
            unit="°",
            limiting_factor=uphill_factor,
        ),
        LimitResult(
            value=downhill_candidates[downhill_factor],
            unit="°",
            limiting_factor=downhill_factor,
        ),
        LimitResult(
            value=cross_candidates[cross_factor],
            unit="°",
            limiting_factor=cross_factor,
        ),
    )


def _calculate_clearance_and_angles(
    suspension: SuspensionParameters,
    vehicle: VehicleParameters,
    flat_points: ChassisPoints,
) -> tuple[LimitResult, LimitResult, LimitResult, LimitResult]:
    wheel_y = [
        flat_points.front_wheel.y,
        flat_points.middle_wheel.y,
        flat_points.rear_wheel.y,
    ]
    pivot_y = suspension.wheel_radius_mm - sum(wheel_y) / len(wheel_y)
    absolute_points = _shift_points_vertically(flat_points, pivot_y)

    body_clearance = pivot_y - suspension.attachment_y_mm
    suspension_clearance = min(
        absolute_points.front_wheel.y,
        absolute_points.middle_wheel.y,
        absolute_points.rear_wheel.y,
        absolute_points.main_pivot.y,
        absolute_points.bogie_pivot.y,
    ) - suspension.arm_radius_mm
    theoretical_clearance = max(
        0.0,
        min(body_clearance, suspension_clearance),
    )
    working_clearance = theoretical_clearance / vehicle.safety_factor

    body_left = -suspension.attachment_x_mm
    body_right = body_left + suspension.body_length_mm
    wheel_x = sorted(
        [
            flat_points.front_wheel.x,
            flat_points.middle_wheel.x,
            flat_points.rear_wheel.x,
        ]
    )
    rear_wheel_x = wheel_x[0]
    front_wheel_x = wheel_x[-1]

    front_overhang = max(0.0, body_right - front_wheel_x)
    rear_overhang = max(0.0, rear_wheel_x - body_left)
    safe_body_clearance = max(0.0, body_clearance / vehicle.safety_factor)

    approach = (
        90.0
        if front_overhang <= 1e-9
        else degrees(atan2(safe_body_clearance, front_overhang))
    )
    departure = (
        90.0
        if rear_overhang <= 1e-9
        else degrees(atan2(safe_body_clearance, rear_overhang))
    )

    axle_gaps = [
        wheel_x[index + 1] - wheel_x[index]
        for index in range(len(wheel_x) - 1)
    ]
    largest_gap = max(axle_gaps)
    breakover = degrees(
        2.0 * atan2(2.0 * working_clearance, largest_gap)
    )

    return (
        LimitResult(
            value=working_clearance,
            unit="мм",
            limiting_factor="нижняя точка корпуса или рычагов",
            theoretical_value=theoretical_clearance,
        ),
        LimitResult(
            value=approach,
            unit="°",
            limiting_factor="передний свес корпуса",
        ),
        LimitResult(
            value=departure,
            unit="°",
            limiting_factor="задний свес корпуса",
        ),
        LimitResult(
            value=breakover,
            unit="°",
            limiting_factor="клиренс между соседними осями",
        ),
    )


def calculate_mobility_limits(
    suspension: SuspensionParameters,
    vehicle: VehicleParameters,
) -> MobilityResults:
    """Выполняет квазистатические проверки геометрии, тяги и устойчивости."""
    _validate_parameters(suspension, vehicle)

    flat_points = _relative_points(
        suspension,
        rocker_rotation_deg=0.0,
        bogie_rotation_deg=0.0,
    )
    if not _wheel_order_is_valid(flat_points):
        raise ValueError(
            "В исходном положении нарушен порядок колёс: "
            "переднее → среднее → заднее."
        )

    com_x_mm = (
        vehicle.com_x_from_body_left_mm
        - suspension.attachment_x_mm
    )
    wheel_y = [
        flat_points.front_wheel.y,
        flat_points.middle_wheel.y,
        flat_points.rear_wheel.y,
    ]
    flat_height_error = max(wheel_y) - min(wheel_y)

    warnings: list[str] = []
    if flat_height_error > 1.0:
        warnings.append(
            "В исходной геометрии центры колёс отличаются по высоте "
            f"на {flat_height_error:.1f} мм."
        )
    if not 0.0 <= vehicle.com_x_from_body_left_mm <= suspension.body_length_mm:
        warnings.append("Центр масс по X находится за пределами корпуса.")
    if abs(vehicle.lateral_com_offset_mm) >= vehicle.track_width_mm / 2.0:
        warnings.append("Поперечная проекция центра масс находится вне колеи.")
    if suspension.arm_radius_mm >= suspension.wheel_radius_mm:
        warnings.append("Рычаг толще радиуса колеса: дорожный просвет отсутствует.")

    wheel_x = [
        flat_points.front_wheel.x,
        flat_points.middle_wheel.x,
        flat_points.rear_wheel.x,
    ]
    if not (
        min(wheel_x) + vehicle.support_margin_mm
        <= com_x_mm
        <= max(wheel_x) - vehicle.support_margin_mm
    ):
        warnings.append(
            "Проекция центра масс не имеет заданного запаса "
            "внутри продольной области опоры."
        )

    (
        step,
        front_phase,
        rear_phase,
        force_limit,
        wall_climb_possible,
    ) = _calculate_step_limits(
        suspension,
        vehicle,
        com_x_mm,
    )
    ditch = _calculate_ditch_limit(
        suspension,
        vehicle,
        flat_points,
        com_x_mm,
    )
    uphill, downhill, cross_slope = _calculate_slope_limits(
        suspension,
        vehicle,
        flat_points,
        com_x_mm,
    )
    (
        ground_clearance,
        approach,
        departure,
        breakover,
    ) = _calculate_clearance_and_angles(
        suspension,
        vehicle,
        flat_points,
    )

    if step.theoretical_value is not None and step.theoretical_value <= 0.0:
        warnings.append(
            "Для заданных углов и зазоров не найден непрерывный "
            "геометрически допустимый подъём на ступень."
        )

    return MobilityResults(
        step=step,
        ditch=ditch,
        uphill=uphill,
        downhill=downhill,
        cross_slope=cross_slope,
        approach_angle=approach,
        departure_angle=departure,
        breakover_angle=breakover,
        ground_clearance=ground_clearance,
        step_front_phase_mm=front_phase,
        step_rear_phase_mm=rear_phase,
        step_force_limit_mm=force_limit,
        wall_climb_possible=wall_climb_possible,
        flat_height_error_mm=flat_height_error,
        warnings=tuple(warnings),
    )


def grade_percent(angle_deg: float) -> float:
    """Переводит уклон из градусов в проценты."""
    return tan(radians(angle_deg)) * 100.0


if __name__ == "__main__":
    example_suspension = SuspensionParameters(
        wheel_radius_mm=100.0,
        front_length_mm=500.0,
        front_angle_deg=-36.87,
        bogie_length_mm=335.41,
        bogie_angle_deg=-153.43,
        middle_length_mm=250.0,
        middle_angle_deg=-36.87,
        rear_length_mm=250.0,
        rear_angle_deg=-143.13,
        rocker_min_deg=-45.0,
        rocker_max_deg=45.0,
        bogie_min_deg=-90.0,
        bogie_max_deg=90.0,
        body_length_mm=600.0,
        body_height_mm=250.0,
        attachment_x_mm=300.0,
        attachment_y_mm=50.0,
        arm_radius_mm=15.0,
    )
    example_vehicle = VehicleParameters(
        total_mass_kg=120.0,
        track_width_mm=600.0,
        com_x_from_body_left_mm=300.0,
        com_height_mm=500.0,
        lateral_com_offset_mm=0.0,
        wheel_torque_nm=20.0,
        brake_torque_nm=20.0,
        driven_wheels=6,
        drivetrain_efficiency=0.8,
        ground_friction=0.7,
        obstacle_friction=0.7,
        rolling_resistance=0.03,
        support_margin_mm=20.0,
        safety_factor=1.2,
    )

    print(calculate_mobility_limits(example_suspension, example_vehicle))
