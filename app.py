"""Streamlit-интерфейс: подбор геометрии rocker-bogie на физике MuJoCo."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace

import pandas as pd
import streamlit as st

from drawing import animation_figure, animation_gif, design_figure, timeseries_figure
from model import (
    M365_WHEEL_RADIUS_MM,
    M365_WHEEL_WIDTH_MM,
    Design,
    Platform,
    coordinates,
    design_from_dict,
    link_dimensions,
    overall_length_mm,
    validate,
)
from optimize import (
    MODE_LABELS,
    Criterion,
    Evaluation,
    Goal,
    Mode,
    OptimizationResult,
    criterion_key,
    criterion_label,
    default_population,
    evaluate_parallel,
    optimize,
)
from simulate import DIRECTION_LABELS, KIND_LABELS, LimitResult, Obstacle, SimResult, run



@dataclass(frozen=True)
class Spec:
    field: str
    label: str
    unit: str
    minimum: float
    maximum: float
    free: bool
    free_allowed: bool
    group: str


SPECS = (
    Spec("bogie_rear_arm_mm", "Плечо к заднему колесу (шарнир → верх стойки)", "мм", 100, 450, True, True,
         "Bogie (заднее и среднее колёса)"),
    Spec("bogie_front_arm_mm", "Плечо к среднему колесу (шарнир → верх стойки)", "мм", 100, 450, True, True,
         "Bogie (заднее и среднее колёса)"),
    Spec("bogie_angle_deg", "Угол между плечами (под шарниром; <180 — Λ, >180 — V)", "°", 100, 280, True, True,
         "Bogie (заднее и среднее колёса)"),
    Spec("rocker_rear_arm_mm", "Плечо к шарниру bogie", "мм", 150, 600, True, True,
         "Rocker (главный рычаг)"),
    Spec("rocker_front_arm_mm", "Плечо к переднему колесу (шарнир → верх стойки)", "мм", 200, 750, True, True,
         "Rocker (главный рычаг)"),
    Spec("rocker_angle_deg", "Угол между плечами (под шарниром; <180 — Λ, >180 — V)", "°", 100, 280, True, True,
         "Rocker (главный рычаг)"),
    Spec("outer_strut_height_mm", "Крайние колёса: высота стойки (ось → крепление)", "мм", 120, 400, False, True, "Стойки"),
    Spec("outer_strut_angle_deg", "Крайние колёса: наклон стойки (+ верх к центру)", "°", -25, 25, True, True, "Стойки"),
    Spec("middle_strut_height_mm", "Среднее колесо: высота стойки", "мм", 120, 400, False, True, "Стойки"),
    Spec("middle_strut_angle_deg", "Среднее колесо: наклон стойки (+ верх вперёд)", "°", -25, 25, True, True, "Стойки"),
    Spec("com_x_mm", "Центр масс корпуса: X от главного шарнира", "мм", -200, 200, False, False, "Центр масс и шарниры"),
    Spec("com_z_mm", "Центр масс корпуса: Z над главным шарниром", "мм", -100, 400, False, False, "Центр масс и шарниры"),
    Spec("com_y_mm", "Центр масс корпуса: смещение влево", "мм", -150, 150, False, False, "Центр масс и шарниры"),
    Spec("rocker_limit_deg", "Предел угла rocker, ±", "°", 10, 90, False, False, "Центр масс и шарниры"),
    Spec("bogie_limit_deg", "Предел угла bogie, ±", "°", 10, 90, False, False, "Центр масс и шарниры"),
)
SPEC_BY_FIELD = {spec.field: spec for spec in SPECS}
GROUPS = tuple(dict.fromkeys(spec.group for spec in SPECS))


def _short_label(name: str) -> str:
    """Подпись параметра без пояснений в скобках, с группой — для заголовков таблиц."""
    spec = SPEC_BY_FIELD[name]
    return f"{spec.group.split(' (')[0]}: {spec.label.split(' (')[0]}"

PLATFORM_WIDGETS = {
    "max_length_mm": "platform_length",
    "max_width_mm": "platform_width",
    "body_mass_kg": "platform_body_mass",
    "wheel_module_mass_kg": "platform_wheel_mass",
    "wheel_torque_nm": "platform_torque",
    "friction": "platform_friction",
    "speed_m_s": "platform_speed",
    "control": "platform_control",
    "tube_diameter_mm": "platform_tube",
    "strut_diameter_mm": "platform_strut",
    "wheel_radius_mm": "platform_wheel_radius",
    "wheel_width_mm": "platform_wheel_width",
    "tire_stiffness_n_mm": "platform_tire_stiffness",
    "tire_damping_ratio": "platform_tire_damping",
    "tire_max_deflection_mm": "platform_tire_travel",
    "tire_edge_wrap_mm": "platform_tire_wrap",
    "motor_no_load_speed_m_s": "platform_motor_no_load",
    "motor_knee_speed_m_s": "platform_motor_knee",
}
CONTROL_LABELS = {
    "speed": "свой регулятор скорости у каждого колеса",
    "torque": "одинаковый момент всех колёс",
}


# ---------------------------------------------------------------------------
# Таблицы параметров
# ---------------------------------------------------------------------------

def _default_table(group: str, values: dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "field": spec.field,
            "Параметр": spec.label,
            "Значение": float(values[spec.field]),
            "Свободный": spec.free,
            "Минимум": float(spec.minimum),
            "Максимум": float(spec.maximum),
            "Ед.": spec.unit,
        }
        for spec in SPECS
        if spec.group == group
    ]
    return pd.DataFrame(rows).set_index("field")


def _apply_design(design: Design) -> None:
    values = asdict(design)
    for index, group in enumerate(GROUPS):
        key = f"table_{index}"
        table = st.session_state.get(key, _default_table(group, values)).copy()
        for field in table.index:
            table.loc[field, "Значение"] = float(values[field])
        st.session_state[key] = table
        st.session_state.pop(f"editor_{index}", None)


def _render_tables() -> tuple[Design, dict[str, tuple[float, float]], list[str]]:
    defaults = asdict(Design())
    values: dict[str, float] = {}
    free: dict[str, tuple[float, float]] = {}
    errors: list[str] = []
    for index, group in enumerate(GROUPS):
        key = f"table_{index}"
        if key not in st.session_state:
            st.session_state[key] = _default_table(group, defaults)
        with st.expander(group, expanded=index < 3):
            edited = st.data_editor(
                st.session_state[key],
                key=f"editor_{index}",
                hide_index=True,
                width="stretch",
                num_rows="fixed",
                disabled=["Параметр", "Ед."],
                column_config={
                    "Параметр": st.column_config.TextColumn(width="large"),
                    "Значение": st.column_config.NumberColumn(format="%.1f"),
                    "Свободный": st.column_config.CheckboxColumn(
                        help="Оптимизатор меняет только отмеченные параметры в пределах [минимум, максимум]."
                    ),
                    "Минимум": st.column_config.NumberColumn(format="%.1f"),
                    "Максимум": st.column_config.NumberColumn(format="%.1f"),
                    "Ед.": st.column_config.TextColumn(width="small"),
                },
            )
        st.session_state[key] = edited
        for field, row in edited.iterrows():
            spec = SPEC_BY_FIELD[field]
            values[field] = float(row["Значение"])
            if bool(row["Свободный"]):
                if not spec.free_allowed:
                    errors.append(f"«{spec.label}» не оптимизируется — снимите отметку «Свободный».")
                elif float(row["Минимум"]) >= float(row["Максимум"]):
                    errors.append(f"«{spec.label}»: минимум должен быть меньше максимума.")
                else:
                    free[field] = (float(row["Минимум"]), float(row["Максимум"]))
    return design_from_dict(values), free, errors


# ---------------------------------------------------------------------------
# Кэшируемые расчёты
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=40)
def _cached_run(design: Design, platform: Platform, obstacle: Obstacle) -> SimResult:
    return run(design, platform, obstacle, record=True)


@st.cache_data(show_spinner=False, max_entries=10)
def _cached_gif(design: Design, platform: Platform, obstacle: Obstacle, side: str) -> bytes:
    return animation_gif(_cached_run(design, platform, obstacle), platform, side)


def _show_simulation(design: Design, platform: Platform, obstacle: Obstacle, key: str) -> None:
    with st.spinner("Моделирование…"):
        result = _cached_run(design, platform, obstacle)
    if result.success:
        st.success(f"Пройдено: {obstacle.label}.")
    else:
        st.error(f"Не пройдено: {result.reason}.")
    for note in result.notes:
        st.warning(note)
    metrics = st.columns(5)
    metrics[0].metric("Макс. момент колеса", f"{100 * result.max_torque_ratio:.0f} %")
    metrics[1].metric("Макс. тангаж", f"{result.max_pitch_deg:.1f}°")
    metrics[2].metric("Макс. крен", f"{result.max_roll_deg:.1f}°")
    metrics[3].metric("Макс. угол bogie", f"{result.max_bogie_deg:.1f}°")
    metrics[4].metric("Время прохождения", f"{result.duration_s:.1f} с")
    if not result.frames:
        return
    left, middle, right = st.columns([1, 1, 2])
    side = left.radio("Борт", ("L", "R"), format_func=lambda s: "левый" if s == "L" else "правый",
                      horizontal=True, key=f"{key}_side")
    ghost = middle.checkbox("Показать второй борт", value=obstacle.one_side, key=f"{key}_ghost")
    st.plotly_chart(animation_figure(result, platform, side, ghost), width="stretch",
                    config={"displaylogo": False}, key=f"{key}_animation")
    st.caption("Красные отрезки — реакции опоры на колёса (0,35 мм на 1 Н). "
               "Пунктир на графиках — правый борт.")
    if right.button("Подготовить GIF", key=f"{key}_gif_button"):
        with st.spinner("Рендер GIF…"):
            st.session_state[f"{key}_gif"] = _cached_gif(design, platform, obstacle, side)
    if f"{key}_gif" in st.session_state:
        right.download_button("Скачать GIF", st.session_state[f"{key}_gif"], file_name="rocker_bogie.gif",
                              mime="image/gif", key=f"{key}_gif_download")
    with st.expander("Графики по времени", expanded=False):
        st.plotly_chart(timeseries_figure(result, platform), width="stretch",
                        config={"displaylogo": False}, key=f"{key}_series")


def _torque_text(value: float, upper: float | None = None) -> str:
    if value == float("inf"):
        return f"не проходится и при {upper:.0f} Н·м" if upper else "не проходится"
    return f"{value:.1f} Н·м"


def _score_text(evaluation: Evaluation) -> str:
    if evaluation.mode == "limits":
        return f"{evaluation.score:.0f}"
    return _torque_text(evaluation.score)


def _value_text(evaluation: Evaluation, value: float) -> str:
    return f"{value:.0f} мм" if evaluation.mode == "limits" else _torque_text(value)


def _obstacle_kind_text(obstacle: Obstacle) -> str:
    return criterion_label(criterion_key(obstacle.kind, obstacle.bump_width_mm))


def _result_rows(evaluation: Evaluation) -> list[dict[str, object]]:
    rows = []
    for result in evaluation.results:
        row: dict[str, object] = {
            "Препятствие": _obstacle_kind_text(result.obstacle),
            "Направление": DIRECTION_LABELS[result.obstacle.direction],
            "Сценарий": "под левым бортом" if result.obstacle.one_side else "по всей ширине",
        }
        if isinstance(result, LimitResult):
            row["Предел, мм"] = round(result.value_mm)
        else:
            row["Размер, мм"] = round(result.obstacle.size_mm)
            row["Нужный момент"] = _torque_text(result.torque_nm, result.upper_nm)
        row["Что ограничило"] = result.reason
        row["Прогонов"] = result.runs
        rows.append(row)
    return rows


def _result_case(result, platform: Platform) -> tuple[str, Platform]:
    """Подпись для выбора прохождения и платформа, с которой его показывать."""
    text = (f"{_obstacle_kind_text(result.obstacle)}, {DIRECTION_LABELS[result.obstacle.direction]}"
            f"{', под левым бортом' if result.obstacle.one_side else ''}")
    if isinstance(result, LimitResult):
        return f"{text}: {result.value_mm:.0f} мм", platform
    torque = min(result.torque_nm, result.upper_nm)
    return (f"{text}, {result.obstacle.size_mm:.0f} мм: {_torque_text(result.torque_nm, result.upper_nm)}",
            replace(platform, wheel_torque_nm=torque))


def _criteria_inputs(mode: Mode, prefix: str, weights: bool) -> tuple[Criterion, ...]:
    """Поля целей по видам препятствий для выбранного режима."""
    defaults = Goal().criteria
    bumps = 0
    criteria = []
    for column, default in zip(st.columns(len(defaults)), defaults):
        kind = default.kind
        key = f"{kind}_{bumps}" if kind == "bump" else kind
        with column:
            if kind == "bump":
                bumps += 1
                st.markdown(f"**{KIND_LABELS[kind].capitalize()} {bumps}**")
                width = st.number_input("Ширина вдоль хода, мм", min_value=20.0, value=default.width_mm,
                                        step=10.0, key=f"{prefix}_width_{key}")
            else:
                st.markdown(f"**{KIND_LABELS[kind].capitalize()}**")
                width = default.width_mm
            if mode == "limits":
                weight = (st.number_input("Вес", min_value=0.0, value=default.weight, step=0.05,
                                          key=f"{prefix}_weight_{key}") if weights else 1.0)
                upper = st.number_input("Искать до, мм", min_value=20.0, value=default.upper_mm, step=25.0,
                                        key=f"{prefix}_upper_{key}")
                criteria.append(Criterion(kind, float(weight), float(upper), default.target_mm, float(width)))
            else:
                active = st.checkbox("Учитывать", value=True, key=f"{prefix}_use_{key}")
                target = st.number_input("Ширина рва, мм" if kind == "ditch" else "Высота, мм", min_value=10.0,
                                         value=default.target_mm, step=10.0, key=f"{prefix}_target_{key}")
                criteria.append(Criterion(kind, 1.0 if active else 0.0, default.upper_mm, float(target),
                                          float(width)))
    return tuple(criteria)


# ---------------------------------------------------------------------------
# Боковая панель: платформа
# ---------------------------------------------------------------------------

def _platform_sidebar() -> Platform:
    defaults = Platform()
    for field, key in PLATFORM_WIDGETS.items():
        st.session_state.setdefault(key, getattr(defaults, field))
    with st.sidebar:
        st.header("Платформа")
        length = st.number_input("Габарит по длине, мм", min_value=500.0, step=25.0, key="platform_length")
        width = st.number_input("Габарит по ширине, мм", min_value=200.0, step=25.0, key="platform_width")
        body = st.number_input("Масса корпуса с грузом, кг", min_value=1.0, step=5.0, key="platform_body_mass")
        wheel_mass = st.number_input("Масса колёсного модуля (мотор-колесо + вилка), кг", min_value=0.1,
                                     step=0.5, key="platform_wheel_mass")
        st.caption(f"Общая масса: {body + 6 * wheel_mass:.1f} кг")
        st.header("Привод и грунт")
        torque = st.number_input("Пиковый момент мотор-колеса, Н·м", min_value=0.5, step=1.0, key="platform_torque",
                                 help="Xiaomi M365 Pro: производитель момент не публикует, 16 Н·м — оценка. "
                                      "Это главный ограничитель высоты ступени — уточните замером.")
        friction = st.number_input("Коэффициент сцепления μ", min_value=0.05, step=0.05, key="platform_friction")
        speed = st.number_input("Скорость движения, м/с", min_value=0.02, max_value=1.0, step=0.05,
                                key="platform_speed")
        control = st.radio("Управление колёсами", tuple(CONTROL_LABELS), format_func=CONTROL_LABELS.get,
                           key="platform_control")
        with st.expander("Колесо и трубы"):
            radius = st.number_input("Радиус колеса, мм", min_value=50.0, step=1.0, key="platform_wheel_radius",
                                     help=f"M365 Pro, шина 8,5″: {M365_WHEEL_RADIUS_MM:.0f} мм")
            wheel_width = st.number_input("Ширина колеса, мм", min_value=20.0, step=5.0, key="platform_wheel_width",
                                          help=f"M365 Pro: ≈{M365_WHEEL_WIDTH_MM:.0f} мм")
            tube = st.number_input("Диаметр труб рычагов, мм", min_value=10.0, step=5.0, key="platform_tube")
            strut = st.number_input("Диаметр стоек (вилок), мм", min_value=10.0, step=5.0, key="platform_strut")
        with st.expander("Шина"):
            tire_stiffness = st.number_input(
                "Радиальная жёсткость шины, Н/мм", min_value=0.0, step=10.0, key="platform_tire_stiffness",
                help="Сила на 1 мм прогиба — одинаково под грунтом и под кромкой ступени. "
                     "0 — жёсткое колесо. Для надутой шины 8,5″ ориентир 80–200 Н/мм, уточните замером: "
                     "нагрузите колесо известным весом и измерьте просадку оси.")
            tire_damping = st.number_input(
                "Демпфирование шины, доля от критического", min_value=0.0, max_value=2.0, step=0.05,
                key="platform_tire_damping",
                help="Гасит подпрыгивание ровера на шинах. 0,2–0,4 — типично для пневматики.")
            tire_travel = st.number_input(
                "Наибольший прогиб до удара обода, мм", min_value=1.0, step=1.0, key="platform_tire_travel",
                help="Примерно высота профиля шины без запаса на обод.")
            tire_wrap = st.number_input(
                "Обминание о кромку (радиус скругления кромок), мм", min_value=0.0, step=1.0,
                key="platform_tire_wrap",
                help="Шина местно обминается вокруг острой кромки, и пятно контакта ложится на неё дугой. "
                     "В расчёте это заменено скруглением кромок препятствий этим радиусом (не больше высоты "
                     "стенки). 0 — острая кромка 90°. Ориентир — 5–15 мм для надутой шины 8,5″.")
        with st.expander("Мотор: момент на оборотах"):
            motor_no_load = st.number_input(
                "Скорость холостого хода (по ободу), м/с", min_value=0.5, step=0.1, key="platform_motor_no_load",
                help="При ней противо-ЭДС съедает всё напряжение и разгоняющий момент падает до нуля. "
                     "M365 Pro (36 В): ≈30 км/ч ≈ 8,3 м/с — оценка, уточните по самокату без нагрузки.")
            motor_knee = st.number_input(
                "До какой скорости доступен полный момент, м/с", min_value=0.0, step=0.1,
                key="platform_motor_knee",
                help="Ниже этой скорости момент ограничен током контроллера (пиковый момент), выше — линейно "
                     "падает до нуля к холостому ходу. Тормозной момент не ограничивается. При скорости "
                     "ровера 0,15 м/с характеристика заметна только у буксующих, раскрутившихся колёс.")
    return replace(
        defaults,
        max_length_mm=float(length),
        max_width_mm=float(width),
        body_mass_kg=float(body),
        wheel_module_mass_kg=float(wheel_mass),
        wheel_torque_nm=float(torque),
        friction=float(friction),
        speed_m_s=float(speed),
        control=str(control),
        wheel_radius_mm=float(radius),
        wheel_width_mm=float(wheel_width),
        tube_diameter_mm=float(tube),
        strut_diameter_mm=float(strut),
        tire_stiffness_n_mm=float(tire_stiffness),
        tire_damping_ratio=float(tire_damping),
        tire_max_deflection_mm=float(tire_travel),
        tire_edge_wrap_mm=float(tire_wrap),
        motor_no_load_speed_m_s=float(motor_no_load),
        motor_knee_speed_m_s=float(motor_knee),
    )


def _load_variant(raw: bytes) -> None:
    data = json.loads(raw.decode("utf-8"))
    known = {item.name for item in fields(Platform)}
    platform = Platform(**{k: v for k, v in data.get("platform", {}).items() if k in known})
    # Старые файлы с координатами шарниров переводятся в длины плеч и углы.
    _apply_design(design_from_dict(data.get("design", {}), platform))
    for field, value in data.get("platform", {}).items():
        if field in PLATFORM_WIDGETS:
            st.session_state[PLATFORM_WIDGETS[field]] = value


# ---------------------------------------------------------------------------
# Страница
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Rocker-Bogie", page_icon="⚙️", layout="wide")
    if "pending_variant" in st.session_state:
        _load_variant(st.session_state.pop("pending_variant"))
    platform = _platform_sidebar()

    st.title("Подбор геометрии Rocker-Bogie")
    st.caption(
        "Физика — MuJoCo: контакты с трением, шарниры, дифференциал, мотор-колёса с ограничением момента. "
        "Корпус не участвует в столкновениях."
    )

    geometry_tab, obstacle_tab, limits_tab, optimization_tab, method_tab = st.tabs(
        ["1. Геометрия", "2. Препятствие", "3. Пределы", "4. Оптимизация", "5. О модели"]
    )

    with geometry_tab:
        left, right = st.columns([5, 4])
        with left:
            design, free_ranges, table_errors = _render_tables()
        errors = table_errors + validate(design, platform)
        with right:
            if errors:
                for error in errors:
                    st.error(error)
            else:
                columns = st.columns(3)
                columns[0].metric("Длина", f"{overall_length_mm(design, platform):.0f} мм")
                columns[1].metric("Колея", f"{platform.track_mm:.0f} мм")
                columns[2].metric("Свободных параметров", len(free_ranges))
                st.plotly_chart(design_figure(design, platform), width="stretch", config={"displaylogo": False})
                st.caption("Числа у труб — длины плеч, мм; под шарнирами — углы между плечами.")
                with st.expander("Координаты и звенья (для справки)", expanded=True):
                    st.dataframe(coordinates(design, platform), width="stretch", hide_index=True)
                    st.dataframe(link_dimensions(design, platform), width="stretch", hide_index=True)
            st.download_button(
                "Сохранить вариант (JSON)",
                json.dumps({"design": asdict(design), "platform": asdict(platform)}, ensure_ascii=False, indent=2),
                file_name="rocker_bogie_variant.json",
                mime="application/json",
            )
            uploaded = st.file_uploader("Загрузить вариант", type="json")
            if uploaded is not None and st.button("Применить загруженный вариант"):
                st.session_state["pending_variant"] = uploaded.getvalue()
                st.rerun()

    with obstacle_tab:
        st.subheader("Проверка одного препятствия")
        columns = st.columns(5)
        kind = columns[0].selectbox("Препятствие", tuple(KIND_LABELS), format_func=KIND_LABELS.get)
        size_label = {"ditch": "Ширина рва, мм", "bump": "Высота кочки, мм"}.get(kind, "Высота ступени, мм")
        size = columns[1].number_input(size_label, min_value=0.0, value=40.0, step=10.0)
        direction = columns[2].selectbox("Направление", (1, -1), format_func=DIRECTION_LABELS.get)
        one_side = columns[3].checkbox("Только под левым бортом")
        depth, width = 500.0, 120.0
        if kind == "bump":
            width = columns[4].number_input("Ширина кочки вдоль хода, мм", min_value=20.0, value=120.0, step=10.0)
        else:
            depth = columns[4].number_input("Глубина рва, мм", min_value=50.0, value=500.0, step=50.0,
                                            disabled=kind != "ditch")
        obstacle = Obstacle(kind, float(size), int(direction), bool(one_side), float(depth), float(width))
        if st.button("Смоделировать", type="primary", disabled=bool(errors)):
            st.session_state["obstacle_run"] = (design, platform, obstacle)
        stored = st.session_state.get("obstacle_run")
        if stored is not None:
            if stored != (design, platform, obstacle):
                st.info("Параметры изменились — нажмите «Смоделировать», чтобы пересчитать.")
            else:
                _show_simulation(*stored, key="single")

    with limits_tab:
        st.subheader("Пределы текущей геометрии")
        limits_mode = st.radio("Что искать", tuple(MODE_LABELS), horizontal=True, key="limits_mode",
                               format_func=lambda m: {"limits": "наибольшие препятствия при моменте платформы",
                                                      "torque": "момент, нужный для заданных препятствий"}[m])
        if limits_mode == "limits":
            st.write("Поиск наибольшей ступени, высоты съезда, ширины рва и высоты кочек заданной ширины "
                     "в обоих направлениях с шагом 2,5 мм. Каждый предел считается в отдельном процессе.")
        else:
            st.write("Для каждого препятствия ищется наименьший пиковый момент колеса, при котором оно "
                     "проходится (шаг 0,25 Н·м). Сравните с моментом мотор-колеса на боковой панели.")
        limits_criteria = _criteria_inputs(limits_mode, "limits", weights=False)
        columns = st.columns(2)
        limits_one_side = columns[0].checkbox("Добавить «под одним бортом»", value=True)
        limits_max_torque = columns[1].number_input("Искать момент до, Н·м", min_value=5.0, value=60.0, step=5.0,
                                                    disabled=limits_mode != "torque", key="limits_max_torque")
        limits_goal = Goal(criteria=limits_criteria, mode=limits_mode, one_side=limits_one_side,
                           max_torque_nm=float(limits_max_torque)).refined(0.5)
        if st.button("Рассчитать", type="primary", disabled=bool(errors) or not limits_goal.active):
            bar = st.progress(0.0, text="Поиск границ")
            evaluation = evaluate_parallel(
                design, platform, limits_goal,
                progress=lambda done, total: bar.progress(done / total, text=f"Готово {done} из {total}"),
            )
            st.session_state["limits"] = (design, platform, limits_goal, evaluation)
        stored = st.session_state.get("limits")
        if stored is not None:
            limits_design, limits_platform, stored_goal, evaluation = stored
            if (limits_design, limits_platform) != (design, platform):
                st.info("Геометрия или платформа изменились — результаты относятся к прошлому варианту.")
            if stored_goal.mode == "torque":
                needed = evaluation.score
                if needed <= limits_platform.wheel_torque_nm:
                    st.success(f"Всем препятствиям хватает {needed:.1f} Н·м — "
                               f"у платформы {limits_platform.wheel_torque_nm:.0f} Н·м.")
                else:
                    st.warning(f"Нужно {_torque_text(needed)} — у платформы "
                               f"{limits_platform.wheel_torque_nm:.0f} Н·м.")
            st.dataframe(_result_rows(evaluation), width="stretch", hide_index=True)
            cases = {label: (result, case_platform) for result in evaluation.results
                     for label, case_platform in [_result_case(result, limits_platform)]}
            chosen = st.selectbox("Показать прохождение на границе", tuple(cases))
            case, case_platform = cases[chosen]
            _show_simulation(limits_design, case_platform, case.obstacle, key="limit")

    with optimization_tab:
        st.subheader("Подбор геометрии")
        st.write(
            "Меняются параметры, отмеченные «Свободный» на вкладке «Геометрия». Метод — CMA-ES "
            "(эволюционная стратегия с адаптацией ковариации): поиск начинается с текущей геометрии "
            "и сам подстраивает размер и направление шагов. Все прогоны поколения считаются параллельно."
        )
        mode = st.radio("Цель", tuple(MODE_LABELS), format_func=MODE_LABELS.get, horizontal=True,
                        key="optimization_mode")
        if mode == "limits":
            st.caption("Оценка — **Σ вес × предел (мм)**, больше — лучше. Для каждого препятствия берётся худшее "
                       "из направлений (и сценариев, если включён «под одним бортом»). Пределы меняются "
                       "ступеньками по шагу поиска, поэтому у оценки есть плато.")
        else:
            st.caption("Оценка — **наибольший из требуемых моментов колеса (Н·м)**, меньше — лучше; при равных "
                       "худших выигрывает вариант с меньшим средним. Момент меняется с геометрией плавно — "
                       "это лучший сигнал для поиска, чем предел в миллиметрах. Препятствие, не проходимое "
                       "и при наибольшем моменте (задевает, опрокидывается), считается как 2 × наибольший момент.")
        criteria = _criteria_inputs(mode, "opt", weights=True)
        columns = st.columns(5)
        opt_one_side = columns[0].checkbox("Учитывать «под одним бортом»", help="Вдвое дольше.")
        population = columns[1].number_input(
            "Вариантов в поколении", min_value=4, max_value=64, value=12, step=2,
            help=f"Для {len(free_ranges)} свободных параметров стандарт CMA-ES — "
                 f"{default_population(len(free_ranges))}; больше — надёжнее, но дольше.")
        generations = columns[2].number_input("Поколений", min_value=1, max_value=200, value=25, step=1)
        refine = columns[3].number_input("Уточнить лучших", min_value=0, max_value=10, value=3, step=1,
                                         help="Лучшие варианты пересчитываются с шагом поиска в 2,5 раза мельче.")
        seed = columns[4].number_input("Seed", min_value=0, value=1, step=1)
        max_torque = 60.0
        if mode == "torque":
            max_torque = st.number_input("Искать момент до, Н·м", min_value=5.0, value=60.0, step=5.0,
                                         key="opt_max_torque")
        goal = Goal(criteria=criteria, mode=mode, one_side=bool(opt_one_side), max_torque_nm=float(max_torque))
        st.caption(f"Будет оценено {int(population) * int(generations) + int(refine)} вариантов.")
        can_run = not errors and bool(free_ranges) and bool(goal.active)
        if st.button("Оптимизировать", type="primary", disabled=not can_run):
            bar = st.progress(0.0, text="Оптимизация")

            def report(done: int, total: int, best: Evaluation | None) -> None:
                text = f"Поисков границ {done} из {total}"
                if best is not None:
                    text += f"; лучшая оценка {_score_text(best)}"
                bar.progress(done / total, text=text)

            result = optimize(design, platform, goal, free_ranges, int(population), int(generations), int(seed),
                              int(refine), progress=report)
            st.session_state["optimization"] = (design, platform, result)
        stored = st.session_state.get("optimization")
        if stored is not None:
            source, opt_platform, result = stored
            result: OptimizationResult
            if opt_platform != platform:
                st.info("Параметры платформы изменились после оптимизации.")
            if result.best is None:
                st.error("Не найдено ни одного допустимого варианта.")
            else:
                start = next((e for e in result.evaluations if e.design == source and not e.refined), None)
                best = result.best
                metrics = st.columns(1 + len(best.values))
                delta = None
                if start is not None and start.valid and abs(start.score) != float("inf"):
                    unit = "" if best.mode == "limits" else " Н·м"
                    delta = f"{best.score - start.score:+.1f}{unit} к исходному"
                metrics[0].metric("Оценка лучшего", _score_text(best), delta,
                                  delta_color="normal" if best.mode == "limits" else "inverse")
                for column, (crit_key, value) in zip(metrics[1:], best.values.items()):
                    column.metric(criterion_label(crit_key), _value_text(best, value))
                if best.mode == "torque":
                    if best.score <= opt_platform.wheel_torque_nm:
                        st.success(f"Лучшему варианту хватает {best.score:.1f} Н·м — у платформы "
                                   f"{opt_platform.wheel_torque_nm:.0f} Н·м.")
                    else:
                        st.warning(f"Лучшему варианту нужно {_torque_text(best.score)} — у платформы "
                                   f"{opt_platform.wheel_torque_nm:.0f} Н·м.")
                if st.button("Применить лучший вариант"):
                    _apply_design(best.design)
                    st.rerun()
                history = pd.DataFrame(
                    {"лучшая": [g.best_score for g in result.history],
                     "медиана поколения": [g.median_score for g in result.history]},
                    index=pd.Index([g.index for g in result.history], name="поколение"),
                ).replace(float("inf"), float("nan"))
                left, right = st.columns([3, 2])
                left.markdown("**Сходимость** (оценка по поколениям)")
                left.line_chart(history, x_label="поколение",
                                y_label="Σ вес × мм" if best.mode == "limits" else "Н·м")
                right.markdown("**Ход поиска**")
                right.dataframe(
                    [{"Поколение": g.index, "Допустимых": f"{100 * g.valid_share:.0f} %", "Шаг σ": round(g.sigma, 3)}
                     for g in result.history],
                    width="stretch", hide_index=True, height=300,
                )
                ranked = result.ranked(10)
                st.dataframe(
                    [
                        {"Место": index, "Оценка": _score_text(item), "Уточнён": item.refined,
                         **{criterion_label(k): _value_text(item, v) for k, v in item.values.items()},
                         **{_short_label(name): getattr(item.design, name) for name in free_ranges}}
                        for index, item in enumerate(ranked, start=1)
                    ],
                    width="stretch",
                    hide_index=True,
                )
                left, right = st.columns(2)
                left.markdown("**Лучший вариант**")
                left.plotly_chart(design_figure(best.design, opt_platform), width="stretch",
                                  config={"displaylogo": False}, key="best_design")
                right.markdown("**Результаты лучшего варианта**")
                right.dataframe(_result_rows(best), width="stretch", hide_index=True)

    with method_tab:
        st.subheader("Как устроен расчёт")
        st.markdown(
            """
    **Модель.** По параметрам собирается модель MuJoCo: корпус со свободным движением, два rocker на
    главных шарнирах, связанные классическим дифференциалом (угол левого = −угол правого относительно
    корпуса), на каждом rocker — bogie на свободном шарнире. Колесо висит на стойке высотой *h* с наклоном
    *α*; трубы идут от шарниров к верху стоек. Колёса — цилиндры M365 Pro, у каждого свой мотор с
    ограничением пикового момента. Препятствия — блоки с острой кромкой 90°: ступень вверх и вниз, ров
    и кочка (блок заданной ширины вдоль хода — лежащий кирпич, упавшее дерево), у которой ищется высота.

    **Шина.** Обод колеса подвешен к оси на двух пружинах с демпферами — вдоль и поперёк рычага, так что
    шина прогибается одинаково под грунтом и под кромкой ступени (радиальная жёсткость в Н/мм).
    Наибольший прогиб ограничен — дальше обод бьёт в упор. Упругость гасит подпрыгивание жёстких колёс на
    жёстком грунте, из-за которого реакции опоры скакали от нуля до сотен ньютонов. Местное обминание шины
    вокруг кромки заменено скруглением кромок препятствий (радиус задаётся в разделе «Шина»).

    **Мотор.** До заданной скорости обода доступен полный пиковый момент (ограничение тока), дальше
    разгоняющий момент линейно падает до нуля на скорости холостого хода. Тормозной момент ограничен
    только пиковым.

    **Графики** показывают реакции и моменты, усреднённые за 50 мс между кадрами анимации, — мгновенные
    значения выхватывали случайную фазу быстрых колебаний контактов.

    **Параметры рычагов.** Каждый рычаг (bogie и rocker) задаётся двумя плечами — от шарнира до верха
    стоек (у rocker заднее плечо идёт к шарниру bogie) — и углом между плечами. Угол отсчитывается
    *под шарниром*: меньше 180° — рычаг «домиком» Λ (шарнир выше концов), 180° — прямая труба, больше
    180° — «галочкой» V (шарнир ниже концов). Сборка идёт от заднего колеса: верх его стойки → bogie →
    верх стойки среднего колеса; от шарнира bogie → rocker → переднее плечо дотягивается до высоты верха
    передней стойки (все колёса стоят на ровном грунте). Если плечо не дотягивается, геометрия
    недопустима. Координаты колёс и шарниров получаются из этих параметров и показаны для справки.

    **Прохождение.** Ровер плавно разгоняется до заданной скорости и едет через препятствие. Успех — все
    шесть колёс оказались за препятствием и опираются на грунт. Ровер считается застрявшим, если 6 с
    ни корпус, ни одно колесо не продвинулись вперёд и ни одно колесо не поднялось выше прежнего хотя бы
    на 5 мм (колесо, медленно ползущее по стенке, пока корпус стоит, — это продвижение). Явный упор —
    все нагруженные колёса 3 с стоят на предельном моменте и не крутятся — прерывается, не дожидаясь 6 с. Неудача
    и её причина:
    - *не хватает момента колёс* — колесо упёрлось и стоит на пиковом моменте;
    - *пробуксовка* — колёса крутятся, ровер стоит (не хватает μ);
    - *задевает препятствие* — касание трубы или стойки (считается недопустимым);
    - *опрокидывание* — тангаж > 70° или крен > 40°;
    - *колёса висят в воздухе* — подвеска не достаёт до опоры (например, bogie перекинулся).

    **Предел** — наибольший размер препятствия при заданном моменте колёс. **Требуемый момент** —
    наименьший пиковый момент колеса, при котором препятствие заданного размера проходится. Обе границы
    ищутся одинаково: если известна подсказка (например, значение у соседнего варианта), поиск идёт от неё
    шагами, удваивающимися до смены результата, затем — делением пополам до заданной точности. Деление
    пополам предполагает, что ниже границы всё проходится; поэтому найденная граница проверяется ещё
    двумя прогонами (в двух шагах ниже и на 40 % пути к нулю; для момента — выше). Если там не
    проходится, граница ищется заново ниже провала, а в причине пишется «результат неустойчив».

    **Оптимизация** — CMA-ES в нормированных координатах свободных параметров, первая точка — текущая
    геометрия. Каждое поколение: выборка вариантов из многомерного нормального распределения → оценка
    (параллельно) → среднее смещается к лучшим, а ковариация вытягивается вдоль удачных направлений.
    Недопустимая геометрия получает штраф, растущий с величиной нарушения, — поиск видит, куда
    возвращаться. Несколько лучших вариантов в конце пересчитываются с более мелким шагом.

    **Управление колёсами.** «Свой регулятор скорости» — каждое колесо держит заданную скорость
    (как у большинства роверов); при повороте bogie колёса могут «спорить» друг с другом.
    «Одинаковый момент» — общий момент от регулятора скорости корпуса, как у самокатных контроллеров
    в режиме тока.
            """
        )
        st.warning(
            "Главный вывод первых расчётов: при массе ~120 кг и моменте 16 Н·м на колесо высота ступени "
            "ограничена моментом, а не геометрией. Чтобы колесо поднялось по стенке выше своей оси, трение "
            "о стенку должно нести его долю веса: τ ≥ F·R. При R = 108 мм и F ≈ 200–300 Н нужно 22–32 Н·м. "
            "Геометрия влияет на то, насколько разгружается упирающееся колесо и как сильно его подталкивают остальные."
        )
        st.markdown(
            """
    **Ограничения.** Обминание шины о кромку приближено скруглением кромки; грунт жёсткий, прочность, нагрев моторов и люфты не учитываются, рулевые приводы не поворачивают колёса.
    Результат слегка зависит от скорости и способа управления — это динамический расчёт.
            """
        )


# Дочерние процессы пула (spawn в Windows) импортируют этот файл как
# __mp_main__ — интерфейс им строить не нужно.
if __name__ == "__main__":
    main()
