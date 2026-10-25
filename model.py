"""Параметры шасси rocker-bogie и построение модели MuJoCo.

Система координат: X — вперёд (к одиночному колесу), Y — влево, Z — вверх.
Все размеры в параметрах задаются в миллиметрах для исходного положения на
ровной площадке: центры всех колёс на высоте радиуса, X заднего колеса = 0.

Каждое колесо висит на стойке (вилка с рулевой колонкой или кронштейн).
Стойка идёт от оси колеса вверх на высоту h с наклоном α от вертикали;
верх стойки — точка крепления к рычагу. У крайних колёс наклон зеркальный:
при α > 0 верхи стоек наклонены к центру шасси.

Рычаги задаются длинами плеч и углом между плечами, измеренным под
шарниром: < 180° — шарнир выше концов плеч (Λ), 180° — прямая труба,
> 180° — шарнир ниже концов (V). Положение рычагов и колёс следует из
условия «все колёса стоят на ровной площадке»; координаты шарниров
вычисляются (``neutral_points``) и показываются для наглядности.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from math import acos, asin, atan2, cos, degrees, hypot, radians, sin, sqrt

# Мотор-колесо Xiaomi M365 Pro: шина 8,5″, наружный диаметр ≈216 мм,
# ширина шины ≈55 мм.
M365_WHEEL_RADIUS_MM = 108.0
M365_WHEEL_WIDTH_MM = 55.0

WHEELS = ("front", "middle", "rear")
WHEEL_LABELS = {"front": "переднее", "middle": "среднее", "rear": "заднее"}
SIDES = ("L", "R")
SIDE_LABELS = {"L": "левый", "R": "правый"}


@dataclass(frozen=True)
class Design:
    """Геометрия одного борта (второй борт — зеркальный).

    По умолчанию — лучший вариант прежней оптимизации при моменте 16 Н·м.
    """

    # Bogie: плечи от шарнира bogie к верху стоек заднего и среднего колёс.
    bogie_rear_arm_mm: float = 212.0
    bogie_front_arm_mm: float = 202.0
    bogie_angle_deg: float = 211.0
    # Rocker: плечи от главного шарнира к шарниру bogie и к верху стойки
    # переднего колеса.
    rocker_rear_arm_mm: float = 304.0
    rocker_front_arm_mm: float = 420.0
    rocker_angle_deg: float = 191.5

    # Стойки крайних (поворотных) колёс: вилка + рулевая колонка.
    outer_strut_height_mm: float = 250.0
    outer_strut_angle_deg: float = 6.4
    # Стойка среднего (неповоротного) колеса.
    middle_strut_height_mm: float = 150.0
    middle_strut_angle_deg: float = -12.8

    # Центр масс корпуса с грузом относительно середины оси главных шарниров.
    com_x_mm: float = 0.0
    com_z_mm: float = 130.0
    com_y_mm: float = 0.0

    rocker_limit_deg: float = 60.0
    bogie_limit_deg: float = 75.0


@dataclass(frozen=True)
class Platform:
    """Неизменяемые для оптимизации исходные данные платформы."""

    max_length_mm: float = 1200.0
    max_width_mm: float = 600.0
    wheel_radius_mm: float = M365_WHEEL_RADIUS_MM
    wheel_width_mm: float = M365_WHEEL_WIDTH_MM
    tube_diameter_mm: float = 40.0
    strut_diameter_mm: float = 35.0

    body_mass_kg: float = 96.0
    wheel_module_mass_kg: float = 4.0
    tube_mass_kg_per_m: float = 0.45

    wheel_torque_nm: float = 16.0
    friction: float = 0.8
    speed_m_s: float = 0.15
    # "torque" — все колёса получают одинаковый момент от регулятора скорости
    # корпуса; "speed" — у каждого колеса свой регулятор скорости.
    control: str = "speed"

    @property
    def track_mm(self) -> float:
        """Расстояние между средними плоскостями колёс левого и правого борта."""
        return self.max_width_mm - self.wheel_width_mm

    @property
    def total_mass_kg(self) -> float:
        return self.body_mass_kg + 6.0 * self.wheel_module_mass_kg


@dataclass(frozen=True)
class Points:
    """Точки одного борта в исходном положении, мм (x, z)."""

    wheels: dict[str, tuple[float, float]]
    strut_tops: dict[str, tuple[float, float]]
    main_pivot: tuple[float, float]
    bogie_pivot: tuple[float, float]
    com: tuple[float, float]


class GeometryError(ValueError):
    """Рычаг с заданными плечами и углом не ставит колёса на площадку."""

    def __init__(self, message: str, shortfall_mm: float):
        super().__init__(message)
        self.shortfall_mm = shortfall_mm


# Прежний формат вариантов: координаты колёс и шарниров.
COORDINATE_FIELDS = (
    "middle_wheel_x_mm", "front_wheel_x_mm",
    "main_pivot_x_mm", "main_pivot_z_mm", "bogie_pivot_x_mm", "bogie_pivot_z_mm",
)


def design_from_dict(values: dict, platform: Platform | None = None) -> Design:
    """Вариант из словаря; понимает и прежний формат с координатами."""
    names = {item.name for item in fields(Design)}
    known = {key: float(value) for key, value in values.items() if key in names}
    if all(key in values for key in COORDINATE_FIELDS) and "rocker_rear_arm_mm" not in values:
        coordinates = {key: float(values[key]) for key in COORDINATE_FIELDS}
        return design_from_coordinates(Design(**known), platform or Platform(), **coordinates)
    return Design(**known)


def design_to_dict(design: Design) -> dict[str, float]:
    return asdict(design)


def _strut_tops_z(design: Design, platform: Platform) -> tuple[float, float]:
    radius = platform.wheel_radius_mm
    outer = radius + design.outer_strut_height_mm * cos(radians(design.outer_strut_angle_deg))
    middle = radius + design.middle_strut_height_mm * cos(radians(design.middle_strut_angle_deg))
    return outer, middle


def _place_lever(
    name: str,
    rear_end: tuple[float, float],
    rear_arm: float,
    front_arm: float,
    angle_deg: float,
    front_z: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Ставит рычаг: задний конец известен, у переднего известна только высота.

    Возвращает (шарнир, передний конец). Угол между плечами отсчитывается
    под шарниром, поэтому при угле > 180° шарнир оказывается ниже хорды.
    """
    if rear_arm <= 0.0 or front_arm <= 0.0 or not 0.0 < angle_deg < 360.0:
        raise GeometryError(f"{name}: длины плеч должны быть положительными, угол — от 0 до 360°.", 100.0)
    chord = sqrt(rear_arm**2 + front_arm**2 - 2.0 * rear_arm * front_arm * cos(radians(angle_deg)))
    rise = front_z - rear_end[1]
    if chord <= abs(rise) + 1e-6:
        raise GeometryError(
            f"{name}: концы плеч расходятся на {chord:.0f} мм, а по высоте стойки требуют "
            f"{abs(rise):.0f} мм — колёса не встают на площадку.",
            abs(rise) - chord + 1.0,
        )
    chord_angle = asin(rise / chord)
    front_end = (rear_end[0] + chord * cos(chord_angle), front_z)
    cos_beta = (rear_arm**2 + chord**2 - front_arm**2) / (2.0 * rear_arm * chord)
    beta = acos(max(-1.0, min(1.0, cos_beta)))
    direction = chord_angle + (beta if angle_deg < 180.0 else -beta)
    pivot = (rear_end[0] + rear_arm * cos(direction), rear_end[1] + rear_arm * sin(direction))
    return pivot, front_end


def neutral_points(design: Design, platform: Platform) -> Points:
    """Исходное положение на ровной площадке; GeometryError, если рычаг не собирается."""
    radius = platform.wheel_radius_mm
    outer_h = design.outer_strut_height_mm
    outer_a = radians(design.outer_strut_angle_deg)
    middle_h = design.middle_strut_height_mm
    middle_a = radians(design.middle_strut_angle_deg)
    outer_z, middle_z = _strut_tops_z(design, platform)
    # Зеркальный наклон: верх переднего — назад, верх заднего — вперёд.
    rear_top = (outer_h * sin(outer_a), outer_z)
    bogie, middle_top = _place_lever("Bogie", rear_top, design.bogie_rear_arm_mm, design.bogie_front_arm_mm,
                                     design.bogie_angle_deg, middle_z)
    main, front_top = _place_lever("Rocker", bogie, design.rocker_rear_arm_mm, design.rocker_front_arm_mm,
                                   design.rocker_angle_deg, outer_z)
    wheels = {
        "front": (front_top[0] + outer_h * sin(outer_a), radius),
        "middle": (middle_top[0] - middle_h * sin(middle_a), radius),
        "rear": (0.0, radius),
    }
    return Points(
        wheels=wheels,
        strut_tops={"front": front_top, "middle": middle_top, "rear": rear_top},
        main_pivot=main,
        bogie_pivot=bogie,
        com=(main[0] + design.com_x_mm, main[1] + design.com_z_mm),
    )


def _lever_params(pivot, rear_end, front_end) -> tuple[float, float, float]:
    rear = atan2(rear_end[1] - pivot[1], rear_end[0] - pivot[0])
    front = atan2(front_end[1] - pivot[1], front_end[0] - pivot[0])
    return (
        hypot(rear_end[0] - pivot[0], rear_end[1] - pivot[1]),
        hypot(front_end[0] - pivot[0], front_end[1] - pivot[1]),
        degrees(front - rear) % 360.0,
    )


def design_from_coordinates(
    base: Design,
    platform: Platform,
    middle_wheel_x_mm: float,
    front_wheel_x_mm: float,
    main_pivot_x_mm: float,
    main_pivot_z_mm: float,
    bogie_pivot_x_mm: float,
    bogie_pivot_z_mm: float,
) -> Design:
    """Переводит координаты колёс и шарниров в длины плеч и углы (стойки — из base)."""
    outer_h = base.outer_strut_height_mm
    outer_a = radians(base.outer_strut_angle_deg)
    middle_h = base.middle_strut_height_mm
    middle_a = radians(base.middle_strut_angle_deg)
    outer_z, middle_z = _strut_tops_z(base, platform)
    rear_top = (outer_h * sin(outer_a), outer_z)
    middle_top = (middle_wheel_x_mm + middle_h * sin(middle_a), middle_z)
    front_top = (front_wheel_x_mm - outer_h * sin(outer_a), outer_z)
    main = (main_pivot_x_mm, main_pivot_z_mm)
    bogie = (bogie_pivot_x_mm, bogie_pivot_z_mm)
    bogie_rear, bogie_front, bogie_angle = _lever_params(bogie, rear_top, middle_top)
    rocker_rear, rocker_front, rocker_angle = _lever_params(main, bogie, front_top)
    return replace(
        base,
        bogie_rear_arm_mm=round(bogie_rear, 1),
        bogie_front_arm_mm=round(bogie_front, 1),
        bogie_angle_deg=round(bogie_angle, 1),
        rocker_rear_arm_mm=round(rocker_rear, 1),
        rocker_front_arm_mm=round(rocker_front, 1),
        rocker_angle_deg=round(rocker_angle, 1),
    )


def overall_length_mm(design: Design, platform: Platform) -> float:
    points = neutral_points(design, platform)
    radius = platform.wheel_radius_mm
    half_tube = platform.tube_diameter_mm / 2.0
    half_strut = platform.strut_diameter_mm / 2.0
    xs = [x - radius for x, _ in points.wheels.values()]
    xs += [x + radius for x, _ in points.wheels.values()]
    xs += [x - half_strut for x, _ in points.strut_tops.values()]
    xs += [x + half_strut for x, _ in points.strut_tops.values()]
    for x, _ in (points.main_pivot, points.bogie_pivot):
        xs += [x - half_tube, x + half_tube]
    return max(xs) - min(xs)


def link_dimensions(design: Design, platform: Platform) -> list[dict[str, object]]:
    """Длины и углы труб в исходном положении (угол от горизонтали)."""
    points = neutral_points(design, platform)
    links = (
        ("Rocker: главный шарнир → стойка переднего", points.main_pivot, points.strut_tops["front"]),
        ("Rocker: главный шарнир → шарнир bogie", points.main_pivot, points.bogie_pivot),
        ("Bogie: шарнир → стойка среднего", points.bogie_pivot, points.strut_tops["middle"]),
        ("Bogie: шарнир → стойка заднего", points.bogie_pivot, points.strut_tops["rear"]),
    )
    rows = []
    for label, start, end in links:
        dx, dz = end[0] - start[0], end[1] - start[1]
        rows.append(
            {
                "Звено": label,
                "Длина, мм": round(hypot(dx, dz), 1),
                "Угол к горизонтали, °": round(degrees(atan2(dz, dx)), 1),
            }
        )
    return rows


def coordinates(design: Design, platform: Platform) -> list[dict[str, object]]:
    """Координаты колёс и шарниров в исходном положении — для справки."""
    points = neutral_points(design, platform)
    rows = [
        ("X среднего колеса (от заднего)", points.wheels["middle"][0]),
        ("X переднего колеса (от заднего)", points.wheels["front"][0]),
        ("Главный шарнир: X", points.main_pivot[0]),
        ("Главный шарнир: высота над землёй", points.main_pivot[1]),
        ("Шарнир bogie: X", points.bogie_pivot[0]),
        ("Шарнир bogie: высота над землёй", points.bogie_pivot[1]),
        ("Габаритная длина", overall_length_mm(design, platform)),
    ]
    return [{"Величина": label, "мм": round(value, 1)} for label, value in rows]


def _violations(design: Design, platform: Platform) -> list[tuple[str, float]]:
    """Нарушения ограничений: (сообщение, величина нарушения в мм или градусах)."""
    found: list[tuple[str, float]] = []
    radius = platform.wheel_radius_mm
    if design.outer_strut_height_mm < radius or design.middle_strut_height_mm < radius:
        found.append(("Стойка должна выходить за обод колеса (высота ≥ радиуса).",
                      radius - min(design.outer_strut_height_mm, design.middle_strut_height_mm)))
    if design.rocker_limit_deg <= 0.0 or design.bogie_limit_deg <= 0.0:
        found.append(("Пределы углов шарниров должны быть положительными.", 1.0))
    if platform.track_mm <= 100.0:
        found.append(("Слишком малая ширина: колея меньше 100 мм.", 100.0 - platform.track_mm))
    try:
        points = neutral_points(design, platform)
    except GeometryError as error:
        # Без собранной геометрии остальное не проверить; штраф больше любого другого.
        found.append((str(error), 1000.0 + error.shortfall_mm))
        return found
    middle_x = points.wheels["middle"][0]
    front_x = points.wheels["front"][0]
    gap = 2.0 * radius + 5.0
    if middle_x < gap:
        found.append(("Среднее и заднее колёса пересекаются.", gap - middle_x))
    if front_x - middle_x < gap:
        found.append(("Переднее и среднее колёса пересекаются.", gap - (front_x - middle_x)))
    bogie_x, bogie_z = points.bogie_pivot
    main_x, main_z = points.main_pivot
    if not 0.0 < bogie_x < middle_x:
        found.append(("Шарнир bogie должен находиться между задним и средним колёсами.",
                      max(-bogie_x, bogie_x - middle_x) + 1.0))
    if not middle_x < main_x < front_x:
        found.append(("Главный шарнир должен находиться между средним и передним колёсами.",
                      max(middle_x - main_x, main_x - front_x) + 1.0))
    for name, (_, z) in points.strut_tops.items():
        if z < radius:
            found.append((f"Верх стойки ({name}) ниже оси колеса.", radius - z))
    lowest_pivot = min(main_z, bogie_z)
    if lowest_pivot < 2.0 * radius:
        found.append(("Шарниры должны быть выше колёс (высота ≥ диаметра колеса).", 2.0 * radius - lowest_pivot))
    length = overall_length_mm(design, platform)
    if length > platform.max_length_mm + 1e-6:
        found.append((f"Длина {length:.0f} мм превышает габарит {platform.max_length_mm:.0f} мм.",
                      length - platform.max_length_mm))
    return found


def validate(design: Design, platform: Platform) -> list[str]:
    return [message for message, _ in _violations(design, platform)]


def violation_amount(design: Design, platform: Platform) -> float:
    """Суммарная величина нарушений (0 — геометрия допустима); штраф для оптимизатора."""
    return sum(amount for _, amount in _violations(design, platform))


# --------------------------------------------------------------------------
# Препятствия
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Box:
    """Прямоугольный блок грунта, мм: X и Y диапазоны и высота верха."""

    x0: float
    x1: float
    y0: float
    y1: float
    z_top: float
    z_bottom: float = -1000.0


FAR = 6000.0


def _mjcf_boxes(boxes: list[Box]) -> str:
    lines = []
    for index, box in enumerate(boxes):
        cx = (box.x0 + box.x1) / 2000.0
        cy = (box.y0 + box.y1) / 2000.0
        cz = (box.z_top + box.z_bottom) / 2000.0
        sx = (box.x1 - box.x0) / 2000.0
        sy = (box.y1 - box.y0) / 2000.0
        sz = (box.z_top - box.z_bottom) / 2000.0
        lines.append(
            f'<geom name="ground{index}" class="ground" type="box" '
            f'pos="{cx:.5f} {cy:.5f} {cz:.5f}" size="{sx:.5f} {sy:.5f} {sz:.5f}"/>'
        )
    return "\n    ".join(lines)


def _m(value_mm: float) -> str:
    return f"{value_mm / 1000.0:.5f}"


def _capsule(name: str, start: tuple[float, float], end: tuple[float, float],
             origin: tuple[float, float], diameter_mm: float, cls: str,
             mass_kg: float) -> str:
    x0, z0 = start[0] - origin[0], start[1] - origin[1]
    x1, z1 = end[0] - origin[0], end[1] - origin[1]
    return (
        f'<geom name="{name}" class="{cls}" type="capsule" size="{_m(diameter_mm / 2.0)}" '
        f'fromto="{_m(x0)} 0 {_m(z0)} {_m(x1)} 0 {_m(z1)}" mass="{mass_kg:.4f}"/>'
    )


def _tube_mass(start, end, platform: Platform) -> float:
    return max(0.05, hypot(end[0] - start[0], end[1] - start[1]) / 1000.0 * platform.tube_mass_kg_per_m)


def _side_xml(side: str, design: Design, platform: Platform) -> str:
    points = neutral_points(design, platform)
    y = platform.track_mm / 2.0 * (1.0 if side == "L" else -1.0)
    main = points.main_pivot
    bogie = points.bogie_pivot
    radius = platform.wheel_radius_mm
    tube = platform.tube_diameter_mm
    strut = platform.strut_diameter_mm
    strut_mass = 0.05

    def site(name: str, point: tuple[float, float], origin: tuple[float, float]) -> str:
        return f'<site name="{name}_{side}" pos="{_m(point[0] - origin[0])} 0 {_m(point[1] - origin[1])}"/>'

    def wheel_body(name: str, origin: tuple[float, float]) -> str:
        wx, wz = points.wheels[name]
        return f"""
          <body name="wheel_{name}_{side}" pos="{_m(wx - origin[0])} 0 {_m(wz - origin[1])}">
            <joint name="spin_{name}_{side}" type="hinge" axis="0 1 0" damping="0.05" armature="0.02"/>
            <geom name="tire_{name}_{side}" class="wheel" type="cylinder"
                  size="{_m(radius)} {_m(platform.wheel_width_mm / 2.0)}" zaxis="0 1 0"
                  mass="{platform.wheel_module_mass_kg:.3f}"/>
          </body>"""

    rocker_geoms = "\n          ".join(
        (
            _capsule(f"arm_rocker_front_{side}", main, points.strut_tops["front"], main, tube, "arm",
                     _tube_mass(main, points.strut_tops["front"], platform)),
            _capsule(f"arm_rocker_bogie_{side}", main, bogie, main, tube, "arm",
                     _tube_mass(main, bogie, platform)),
            _capsule(f"strut_front_{side}", points.strut_tops["front"], points.wheels["front"], main,
                     strut, "arm", strut_mass),
        )
    )
    bogie_geoms = "\n            ".join(
        (
            _capsule(f"arm_bogie_middle_{side}", bogie, points.strut_tops["middle"], bogie, tube, "arm",
                     _tube_mass(bogie, points.strut_tops["middle"], platform)),
            _capsule(f"arm_bogie_rear_{side}", bogie, points.strut_tops["rear"], bogie, tube, "arm",
                     _tube_mass(bogie, points.strut_tops["rear"], platform)),
            _capsule(f"strut_middle_{side}", points.strut_tops["middle"], points.wheels["middle"], bogie,
                     strut, "arm", strut_mass),
            _capsule(f"strut_rear_{side}", points.strut_tops["rear"], points.wheels["rear"], bogie,
                     strut, "arm", strut_mass),
        )
    )
    return f"""
      <body name="rocker_{side}" pos="0 {_m(y)} 0">
        <joint name="rocker_{side}" type="hinge" axis="0 1 0" limited="true"
               range="{-design.rocker_limit_deg} {design.rocker_limit_deg}" damping="2"/>
          {rocker_geoms}
          {site("top_front", points.strut_tops["front"], main)}
          {wheel_body("front", main)}
        <body name="bogie_{side}" pos="{_m(bogie[0] - main[0])} 0 {_m(bogie[1] - main[1])}">
          <joint name="bogie_{side}" type="hinge" axis="0 1 0" limited="true"
                 range="{-design.bogie_limit_deg} {design.bogie_limit_deg}" damping="2"/>
            {bogie_geoms}
            {site("top_middle", points.strut_tops["middle"], bogie)}
            {site("top_rear", points.strut_tops["rear"], bogie)}
            {wheel_body("middle", bogie)}
            {wheel_body("rear", bogie)}
        </body>
      </body>"""


def build_mjcf(
    design: Design,
    platform: Platform,
    boxes: list[Box],
    start_x_mm: float = 0.0,
    start_z_mm: float = 0.0,
    floor_z_mm: float = 0.0,
    timestep: float = 0.001,
) -> str:
    """Собирает MJCF-описание ровера и препятствия.

    ``start_x_mm``/``start_z_mm`` — смещение исходного положения ровера
    (X заднего колеса и высота опорной площадки под ним).
    """
    points = neutral_points(design, platform)
    main = points.main_pivot
    body_pos = (start_x_mm + main[0], start_z_mm + main[1])
    mu = platform.friction
    torque = platform.wheel_torque_nm
    omega = platform.speed_m_s / (platform.wheel_radius_mm / 1000.0)
    # Регулятор скорости каждого мотор-колеса (скорость ротора относительно
    # вилки): полный момент при отставании на 50 % от заданной скорости.
    kv = torque / max(0.5 * omega, 1e-3)
    sides = "".join(_side_xml(side, design, platform) for side in SIDES)
    if platform.control == "speed":
        limits = f'kv="{kv:.3f}" forcelimited="true" forcerange="{-torque:.3f} {torque:.3f}"'
        kind = "velocity"
    else:
        limits = f'ctrllimited="true" ctrlrange="{-torque:.3f} {torque:.3f}"'
        kind = "motor"
    actuators = "\n    ".join(
        f'<{kind} name="motor_{wheel}_{side}" joint="spin_{wheel}_{side}" {limits}/>'
        for side in SIDES
        for wheel in WHEELS
    )
    body_half = (
        max(points.wheels["front"][0] - 2.0 * platform.wheel_radius_mm, 200.0) / 2000.0,
        platform.track_mm / 2000.0 * 0.7,
        0.05,
    )
    return f"""<mujoco model="rocker_bogie">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic" impratio="10">
    <flag multiccd="enable"/>
  </option>
  <default>
    <geom solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <default class="ground">
      <geom contype="1" conaffinity="3" friction="{mu} 0.005 0.0001" rgba="0.55 0.62 0.45 1"/>
    </default>
    <default class="wheel">
      <geom contype="1" conaffinity="1" condim="3" friction="{mu} 0.005 0.0001" rgba="0.2 0.2 0.22 1"/>
    </default>
    <default class="arm">
      <geom contype="2" conaffinity="0" condim="3" friction="0.3 0.005 0.0001" rgba="0.2 0.4 0.8 1"/>
    </default>
  </default>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1"/>
    <geom name="floor" class="ground" type="plane" pos="0 0 {_m(floor_z_mm)}" size="0 0 1"/>
    {_mjcf_boxes(boxes)}
    <body name="chassis" pos="{_m(body_pos[0])} 0 {_m(body_pos[1])}">
      <freejoint name="root"/>
      <inertial pos="{_m(design.com_x_mm)} {_m(design.com_y_mm)} {_m(design.com_z_mm)}"
                mass="{platform.body_mass_kg:.3f}"
                diaginertia="{platform.body_mass_kg * 0.03:.3f} {platform.body_mass_kg * 0.09:.3f} {platform.body_mass_kg * 0.1:.3f}"/>
      <geom name="body_visual" type="box" size="{body_half[0]:.4f} {body_half[1]:.4f} {body_half[2]:.4f}"
            pos="0 0 0.06" contype="0" conaffinity="0" mass="0" rgba="0.7 0.7 0.75 0.4"/>
      {sides}
    </body>
  </worldbody>
  <equality>
    <!-- Классический дифференциал: корпус на среднем угле двух rocker. -->
    <joint joint1="rocker_L" joint2="rocker_R" polycoef="0 -1 0 0 0" solref="0.004 1"/>
  </equality>
  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""
