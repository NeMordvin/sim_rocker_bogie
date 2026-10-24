"""Streamlit-интерфейс расчёта и оптимизации rocker-bogie."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import streamlit as st

from analysis import (
    PhysicalCapabilityResult,
    SequentialOptimizationResult,
    find_limits,
    optimize_step_then_calculate_ditch,
)
from drawing import create_design_figure, create_key_frame_figure
from geometry import (
    ChassisDesign,
    DitchTerrain,
    Pose,
    SolverSettings,
    StepTerrain,
    neutral_dimensions,
    validate_design,
)
from reports import select_key_frames
from sources import SOURCES
from statics import (
    PhysicalParameters,
    SCENARIO_LABELS,
    required_friction_coefficient,
    validate_physical,
)


st.set_page_config(
    page_title="Расчёт Rocker-Bogie",
    page_icon="⚙️",
    layout="wide",
)


@dataclass(frozen=True)
class ParameterSpec:
    field: str
    label: str
    unit: str
    minimum: float
    maximum: float
    free_by_default: bool
    free_allowed: bool
    group: str
    display_factor: float = 1.0


PARAMETERS = (
    ParameterSpec(
        "wheel_radius_mm", "Диаметр колёс", "мм", 140.0, 300.0,
        False, False, "Колёса и рычаги", 2.0,
    ),
    ParameterSpec(
        "arm_diameter_mm", "Расчётный диаметр рычагов", "мм", 15.0, 70.0,
        False, False, "Колёса и рычаги",
    ),
    ParameterSpec(
        "middle_wheel_x_mm", "X среднего колеса", "мм", 220.0, 500.0,
        True, True, "Независимая геометрия",
    ),
    ParameterSpec(
        "front_wheel_x_mm", "X одиночного колеса", "мм", 650.0, 1000.0,
        True, True, "Независимая геометрия",
    ),
    ParameterSpec(
        "main_pivot_x_mm", "X главного шарнира", "мм", 430.0, 760.0,
        True, True, "Независимая геометрия",
    ),
    ParameterSpec(
        "main_pivot_y_mm", "Главный шарнир над линией центров", "мм", 180.0, 420.0,
        True, True, "Независимая геометрия",
    ),
    ParameterSpec(
        "bogie_pivot_x_mm", "X шарнира bogie", "мм", 100.0, 300.0,
        True, True, "Независимая геометрия",
    ),
    ParameterSpec(
        "bogie_pivot_y_mm", "Шарнир bogie над линией центров", "мм", 120.0, 330.0,
        True, True, "Независимая геометрия",
    ),
    ParameterSpec(
        "bogie_rear_knee_x_mm", "X заднего верхнего угла bogie", "мм", 30.0, 160.0,
        True, True, "Форма bogie",
    ),
    ParameterSpec(
        "bogie_front_knee_x_mm", "X переднего верхнего угла bogie", "мм", 210.0, 390.0,
        True, True, "Форма bogie",
    ),
    ParameterSpec(
        "com_x_from_main_pivot_mm", "X центра масс от главного шарнира", "мм", -250.0, 250.0,
        False, False, "Центр масс",
    ),
    ParameterSpec(
        "com_z_from_main_pivot_mm", "Z центра масс над главным шарниром", "мм", 20.0, 400.0,
        False, False, "Центр масс",
    ),
)

GROUPS = tuple(dict.fromkeys(spec.group for spec in PARAMETERS))
SPEC_BY_FIELD = {spec.field: spec for spec in PARAMETERS}


def _default_parameter_table(group: str) -> pd.DataFrame:
    defaults = asdict(ChassisDesign())
    rows: list[dict[str, object]] = []
    for spec in PARAMETERS:
        if spec.group != group:
            continue
        rows.append(
            {
                "field": spec.field,
                "Параметр": spec.label,
                "Значение": float(defaults[spec.field]) * spec.display_factor,
                "Свободный": spec.free_by_default,
                "Минимум": spec.minimum,
                "Максимум": spec.maximum,
                "Ед.": spec.unit,
            }
        )
    return pd.DataFrame(rows).set_index("field")


def _render_parameter_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for group_index, group in enumerate(GROUPS):
        storage_key = f"parameter_table_{group_index}"
        editor_key = f"parameter_editor_{group_index}"
        if storage_key not in st.session_state:
            st.session_state[storage_key] = _default_parameter_table(group)
        with st.expander(group, expanded=group_index < 2):
            edited = st.data_editor(
                st.session_state[storage_key],
                key=editor_key,
                hide_index=True,
                width="stretch",
                num_rows="fixed",
                disabled=["Параметр", "Ед."],
                column_config={
                    "Параметр": st.column_config.TextColumn(width="large"),
                    "Значение": st.column_config.NumberColumn(format="%.2f"),
                    "Свободный": st.column_config.CheckboxColumn(
                        help=(
                            "Оптимизатор меняет только независимые координаты. "
                            "Диаметры и центр масс остаются заданными."
                        )
                    ),
                    "Минимум": st.column_config.NumberColumn(format="%.2f"),
                    "Максимум": st.column_config.NumberColumn(format="%.2f"),
                    "Ед.": st.column_config.TextColumn(width="small"),
                },
            )
            st.session_state[storage_key] = edited
            tables[group] = edited
    return tables


def _table_values(
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, float], dict[str, tuple[float, float]], list[str]]:
    values: dict[str, float] = {}
    free_ranges: dict[str, tuple[float, float]] = {}
    errors: list[str] = []
    for table in tables.values():
        for field, row in table.iterrows():
            spec = SPEC_BY_FIELD[field]
            factor = spec.display_factor
            value = float(row["Значение"]) / factor
            minimum = float(row["Минимум"]) / factor
            maximum = float(row["Максимум"]) / factor
            values[field] = value
            requested_free = bool(row["Свободный"])
            if requested_free and not spec.free_allowed:
                errors.append(
                    f"«{spec.label}» является жёстко заданным параметром и не оптимизируется."
                )
            elif requested_free:
                if minimum >= maximum:
                    errors.append(
                        f"Для «{spec.label}» минимум должен быть меньше максимума."
                    )
                else:
                    free_ranges[field] = (minimum, maximum)
    return values, free_ranges, errors


def _apply_design_to_tables(design: ChassisDesign) -> None:
    values = asdict(design)
    for group_index, group in enumerate(GROUPS):
        storage_key = f"parameter_table_{group_index}"
        editor_key = f"parameter_editor_{group_index}"
        table = st.session_state.get(storage_key, _default_parameter_table(group)).copy()
        for field in table.index:
            spec = SPEC_BY_FIELD[field]
            table.loc[field, "Значение"] = float(values[field]) * spec.display_factor
        st.session_state[storage_key] = table
        st.session_state.pop(editor_key, None)


def _derived_rows(design: ChassisDesign) -> list[dict[str, object]]:
    dimensions = neutral_dimensions(design)
    mapping = (
        ("Rocker: главный шарнир → одиночное колесо", "rocker_front_length_mm", "rocker_front_angle_deg"),
        ("Rocker: главный шарнир → bogie", "rocker_bogie_length_mm", "rocker_bogie_angle_deg"),
        ("Bogie: шарнир → среднее колесо", "bogie_middle_length_mm", "bogie_middle_angle_deg"),
        ("Bogie: шарнир → заднее колесо", "bogie_rear_length_mm", "bogie_rear_angle_deg"),
    )
    return [
        {
            "Звено": label,
            "Длина, мм": round(dimensions[length_key], 2),
            "Исходный угол, °": round(dimensions[angle_key], 2),
        }
        for label, length_key, angle_key in mapping
    ]


CAPABILITY_DEFINITIONS = (
    ("step_single", "Ступень", "одиночное колесо вперёд"),
    ("step_bogie", "Ступень", "bogie вперёд"),
    ("ditch_single", "Ров", "одиночное колесо вперёд"),
    ("ditch_bogie", "Ров", "bogie вперёд"),
)


def _capability_rows(results: dict[str, PhysicalCapabilityResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, obstacle, direction in CAPABILITY_DEFINITIONS:
        result = results[key]
        rows.append(
            {
                "Препятствие": obstacle,
                "Направление": direction,
                "Геометрический предел, мм": round(result.geometric_mm, 2),
                "Физический предел, мм": round(result.value_mm, 2),
                "Что ограничивает": result.failure_reason or "граница поиска",
            }
        )
    return rows


def _show_capability_metrics(results: dict[str, PhysicalCapabilityResult]) -> None:
    columns = st.columns(4)
    for column, (key, obstacle, direction) in zip(columns, CAPABILITY_DEFINITIONS):
        result = results[key]
        column.metric(
            f"{obstacle}: {direction}",
            f"{result.value_mm:.1f} мм",
            f"геометрия {result.geometric_mm:.1f} мм",
            delta_color="off",
        )
    st.dataframe(_capability_rows(results), width="stretch", hide_index=True)


def _contact_rows(frame) -> list[dict[str, object]]:
    return [
        {
            "Сторона": load.side,
            "Колесо": load.wheel,
            "Контакт": load.mode,
            "N, Н": round(load.normal_force_n, 3),
            "T, Н": round(load.net_tangent_force_n, 3),
            "Q поперёк, Н": round(load.lateral_force_n, 3),
            "Сила привода, Н": round(load.motor_force_n, 3),
            "Момент, Н·м": round(load.motor_torque_nm, 3),
            "Момент, %": round(100.0 * load.torque_utilization, 2),
            "Сцепление, %": round(100.0 * load.friction_utilization, 2),
        }
        for load in frame.contacts
    ]


@st.cache_data(show_spinner=False)
def _cached_limits(
    design: ChassisDesign,
    physical: PhysicalParameters,
    maximum_step_mm: float,
    maximum_ditch_mm: float,
) -> dict[str, PhysicalCapabilityResult]:
    return find_limits(
        design,
        physical,
        {
            "step_single": ("step", 1, maximum_step_mm),
            "step_bogie": ("step", -1, maximum_step_mm),
            "ditch_single": ("ditch", 1, maximum_ditch_mm),
            "ditch_bogie": ("ditch", -1, maximum_ditch_mm),
        },
        SolverSettings.fine(),
    )


@st.cache_data(show_spinner=False)
def _cached_required_friction(
    design: ChassisDesign,
    physical: PhysicalParameters,
    pose: Pose,
    obstacle_kind: str,
    obstacle_value_mm: float,
    direction: int,
    scenario: str,
) -> float:
    terrain = (
        StepTerrain(obstacle_value_mm)
        if obstacle_kind == "step"
        else DitchTerrain(obstacle_value_mm)
    )
    return required_friction_coefficient(
        design, physical, pose, terrain, direction, scenario  # type: ignore[arg-type]
    )


st.title("Расчёт и оптимизация Rocker-Bogie")
st.caption(
    "Геометрия подвески, квазистатические силы и моменты, продольная и "
    "поперечная устойчивость. Корпус и его габариты не учитываются."
)

with st.sidebar:
    st.header("Физическая модель")
    mass_kg = st.number_input("Общая масса, кг", min_value=0.1, value=120.0, step=5.0)
    st.caption(
        "Мотор-колесо Xiaomi M365 Pro: шина 8,5″ (Ø≈216 мм), 300 Вт номинально, "
        "600 Вт пик. Пиковый момент производитель не публикует — 16 Н·м является "
        "оценкой, уточните её замером."
    )
    torque_nm = st.number_input(
        "Пиковый момент одного мотор-колеса, Н·м",
        min_value=0.01,
        value=PhysicalParameters().wheel_torque_nm,
        step=1.0,
    )
    track_mm = st.number_input("Колея по точкам контакта, мм", min_value=100.0, value=600.0, step=25.0)
    com_lateral_mm = st.number_input("Поперечное смещение ЦМ, мм", value=0.0, step=10.0)
    friction = st.number_input("Коэффициент сцепления μ", min_value=0.01, value=0.70, step=0.05)
    rolling = st.number_input("Коэффициент сопротивления качению", min_value=0.0, value=0.03, step=0.01)
    dynamic_factor = st.number_input("Коэффициент запаса на динамику", min_value=1.0, value=1.20, step=0.05)
    gravity = st.number_input("Ускорение свободного падения, м/с²", min_value=0.1, value=9.81, step=0.1)

    st.header("Жёсткие ограничения")
    max_total_length = st.number_input("Максимальная длина подвески, мм", min_value=300.0, value=1200.0, step=25.0)
    support_margin = st.number_input("Минимальный запас опорного многоугольника, мм", min_value=0.0, value=10.0, step=2.0)
    obstacle_clearance = st.number_input("Зазор рычагов до препятствия, мм", min_value=0.0, value=3.0, step=1.0)
    self_clearance = st.number_input("Зазор между колёсами, мм", min_value=0.0, value=3.0, step=1.0)
    rocker_min = st.number_input("Rocker: минимальный угол, °", min_value=-170.0, max_value=0.0, value=-65.0, step=5.0)
    rocker_max = st.number_input("Rocker: максимальный угол, °", min_value=0.0, max_value=170.0, value=65.0, step=5.0)
    bogie_min = st.number_input("Bogie: минимальный угол, °", min_value=-170.0, max_value=0.0, value=-80.0, step=5.0)
    bogie_max = st.number_input("Bogie: максимальный угол, °", min_value=0.0, max_value=170.0, value=80.0, step=5.0)

    st.header("Границы поиска")
    maximum_step = st.number_input("Искать ступень до, мм", min_value=10.0, value=500.0, step=25.0)
    maximum_ditch = st.number_input("Искать ров до, мм", min_value=10.0, value=600.0, step=25.0)

physical = PhysicalParameters(
    mass_kg=float(mass_kg),
    wheel_torque_nm=float(torque_nm),
    track_width_mm=float(track_mm),
    com_lateral_mm=float(com_lateral_mm),
    friction_coefficient=float(friction),
    rolling_resistance_coefficient=float(rolling),
    dynamic_factor=float(dynamic_factor),
    gravity_m_s2=float(gravity),
    minimum_support_margin_mm=float(support_margin),
)

geometry_tab, calculation_tab, optimization_tab, frames_tab, method_tab = st.tabs(
    [
        "1. Геометрия",
        "2. Расчёт варианта",
        "3. Оптимизация",
        "4. Ключевые кадры",
        "5. Методика и статьи",
    ]
)

with geometry_tab:
    st.subheader("Параметры подвески")
    st.caption(
        "Свободными могут быть только независимые координаты. Длины и углы "
        "звеньев пересчитываются автоматически, поэтому не дублируются как "
        "независимые переменные."
    )
    parameter_tables = _render_parameter_tables()

values, free_ranges, table_errors = _table_values(parameter_tables)
design = ChassisDesign(
    **values,
    bogie_min_angle_deg=float(bogie_min),
    bogie_max_angle_deg=float(bogie_max),
    rocker_min_angle_deg=float(rocker_min),
    rocker_max_angle_deg=float(rocker_max),
    obstacle_clearance_mm=float(obstacle_clearance),
    self_clearance_mm=float(self_clearance),
    max_total_length_mm=float(max_total_length),
)
errors = [*table_errors, *validate_design(design), *validate_physical(physical)]

with geometry_tab:
    if errors:
        for error in errors:
            st.error(error)
    else:
        dimensions = neutral_dimensions(design)
        columns = st.columns(4)
        columns[0].metric("Полная длина", f"{dimensions['total_length_mm']:.1f} мм")
        columns[1].metric("Полная высота", f"{dimensions['total_height_mm']:.1f} мм")
        columns[2].metric("Колёсная база", f"{dimensions['wheelbase_mm']:.1f} мм")
        columns[3].metric("Свободных координат", str(len(free_ranges)))
        st.plotly_chart(
            create_design_figure(design),
            width="stretch",
            config={"displaylogo": False},
        )
        st.subheader("Полученные длины и углы")
        st.dataframe(_derived_rows(design), width="stretch", hide_index=True)

with calculation_tab:
    st.subheader("Точный расчёт текущей геометрии")
    st.write(
        "Для ступени и рва проверяются оба направления движения и два сценария: "
        "препятствие по всей ширине и препятствие только под одной стороной. "
        "**Геометрический предел** — подвеска проходит препятствие без "
        "пересечений и выхода шарниров за диапазон. **Физический предел** — "
        "вдобавок хватает сцепления и момента колёс, а ровер не опрокидывается."
    )
    if st.button("Рассчитать текущий вариант", type="primary", disabled=bool(errors)):
        with st.spinner("Расчёт четырёх пределов параллельно…"):
            results = _cached_limits(
                design, physical, float(maximum_step), float(maximum_ditch)
            )
        st.session_state["current_results"] = {
            "design": design,
            "physical": physical,
            "values": results,
        }

    stored_current = st.session_state.get("current_results")
    if stored_current is not None:
        if stored_current["design"] != design or stored_current["physical"] != physical:
            st.warning("Параметры изменены — расчёт нужно запустить заново.")
        else:
            _show_capability_metrics(stored_current["values"])

with optimization_tab:
    st.subheader("Последовательная оптимизация")
    st.write(
        "Этап 1: максимизируется **физический** предел ступени в обоих "
        "направлениях и обоих сценариях; при равенстве выше ставится вариант с "
        "большим геометрическим пределом, затем более короткий. Этап 2: для "
        "выбранной геометрии вычисляется ров; ров не влияет на выбор размеров. "
        "Варианты считаются параллельно на всех ядрах процессора."
    )
    left, middle = st.columns(2)
    candidate_count = left.number_input(
        "Количество вариантов",
        min_value=2,
        max_value=40,
        value=6,
        step=2,
    )
    random_seed = middle.number_input("Seed", min_value=0, value=42, step=1)
    if st.button(
        "Оптимизировать по ступени",
        type="primary",
        disabled=bool(errors) or not free_ranges,
    ):
        progress = st.progress(0.0, text="Подготовка вариантов")

        def update_progress(stage: str, done: int, total: int) -> None:
            progress.progress(done / max(total, 1), text=f"{stage}: {done} из {total}")

        result = optimize_step_then_calculate_ditch(
            base=design,
            physical=physical,
            free_ranges=free_ranges,
            count=int(candidate_count),
            seed=int(random_seed),
            maximum_step_mm=float(maximum_step),
            maximum_ditch_mm=float(maximum_ditch),
            progress_callback=update_progress,
        )
        st.session_state["optimization_result"] = {
            "source_design": design,
            "physical": physical,
            "result": result,
        }
        progress.progress(1.0, text="Оптимизация завершена")

    stored_optimization = st.session_state.get("optimization_result")
    if stored_optimization is not None:
        result: SequentialOptimizationResult = stored_optimization["result"]
        if stored_optimization["physical"] != physical:
            st.warning("Физические параметры изменены — оптимизацию нужно повторить.")
        elif result.best_step_design is None:
            st.error("Не найдено допустимых вариантов.")
        else:
            best = result.best_step_design
            metrics = st.columns(3)
            metrics[0].metric("Ступень в обе стороны: физически", f"{best.bidirectional_step_mm:.1f} мм")
            metrics[1].metric("Ступень в обе стороны: геометрически", f"{best.bidirectional_geometric_step_mm:.1f} мм")
            metrics[2].metric("Ров этой геометрии в обе стороны", f"{result.bidirectional_ditch_mm:.1f} мм")
            for label, item in (
                ("одиночное вперёд", best.single_first),
                ("bogie вперёд", best.bogie_first),
            ):
                if item.failure_reason:
                    st.caption(f"Ступень, {label} — ограничение: {item.failure_reason}")
            if st.button("Применить найденную геометрию"):
                _apply_design_to_tables(best.design)
                st.rerun()
            st.dataframe(
                [
                    {
                        "Место": index,
                        "Физически в обе стороны, мм": round(item.bidirectional_step_mm, 2),
                        "Геометрически в обе стороны, мм": round(item.bidirectional_geometric_step_mm, 2),
                        "Одиночное вперёд, мм": round(item.single_first.value_mm, 2),
                        "Bogie вперёд, мм": round(item.bogie_first.value_mm, 2),
                        "Точная проверка": "да" if item.refined else "нет",
                    }
                    for index, item in enumerate(result.ranked_step_designs, start=1)
                ],
                width="stretch",
                hide_index=True,
            )
            st.subheader("Параметры лучшего варианта")
            st.dataframe(_derived_rows(best.design), width="stretch", hide_index=True)

with frames_tab:
    st.subheader("Ключевые положения и полный расчёт")
    available: dict[str, tuple[ChassisDesign, PhysicalCapabilityResult]] = {}
    stored_current = st.session_state.get("current_results")
    if stored_current is not None:
        labels = {
            "step_single": "Текущий: ступень, одиночное вперёд",
            "step_bogie": "Текущий: ступень, bogie вперёд",
            "ditch_single": "Текущий: ров, одиночное вперёд",
            "ditch_bogie": "Текущий: ров, bogie вперёд",
        }
        for key, label in labels.items():
            available[label] = (stored_current["design"], stored_current["values"][key])

    stored_optimization = st.session_state.get("optimization_result")
    if stored_optimization is not None:
        optimized: SequentialOptimizationResult = stored_optimization["result"]
        best = optimized.best_step_design
        if best is not None:
            available["Оптимум: ступень, одиночное вперёд"] = (best.design, best.single_first)
            available["Оптимум: ступень, bogie вперёд"] = (best.design, best.bogie_first)
            if optimized.ditch_single_first is not None:
                available["Оптимум: ров, одиночное вперёд"] = (best.design, optimized.ditch_single_first)
            if optimized.ditch_bogie_first is not None:
                available["Оптимум: ров, bogie вперёд"] = (best.design, optimized.ditch_bogie_first)

    if not available:
        st.info("Сначала выполните расчёт текущего варианта или оптимизацию.")
    else:
        selected_label = st.selectbox("Набор кадров", tuple(available))
        selected_design, selected_capability = available[selected_label]
        checked = selected_capability.checked
        if checked is None or not checked.success:
            st.error("Для найденного предела нет полного допустимого пути.")
        else:
            scenarios = tuple(item.scenario for item in checked.statics)
            scenario = st.selectbox(
                "Сценарий",
                scenarios,
                format_func=lambda item: SCENARIO_LABELS[item],
            )
            key_frames = select_key_frames(checked, scenario)
            st.caption(
                f"Предел препятствия: {selected_capability.value_mm:.2f} мм. "
                f"Показано ключевых кадров: {len(key_frames)}."
            )
            for frame in key_frames:
                with st.expander(frame.title, expanded=True):
                    st.plotly_chart(
                        create_key_frame_figure(
                            selected_design,
                            checked.traversal,
                            frame.pose,
                            frame.static,
                            frame.title,
                        ),
                        width="stretch",
                        config={"displaylogo": False},
                    )
                    required_mu = _cached_required_friction(
                        selected_design,
                        physical,
                        frame.pose,
                        checked.traversal.obstacle_kind,
                        checked.traversal.obstacle_value_mm,
                        checked.traversal.direction,
                        scenario,
                    )
                    metrics = st.columns(6)
                    metrics[0].metric("Тангаж платформы", f"{frame.static.body_pitch_deg:.2f}°")
                    metrics[1].metric("Крен платформы", f"{frame.static.body_roll_deg:.2f}°")
                    metrics[2].metric("Продольный запас", f"{frame.static.longitudinal_margin_mm:.2f} мм")
                    metrics[3].metric("Поперечный запас", f"{frame.static.lateral_margin_mm:.2f} мм")
                    metrics[4].metric("Макс. момент", f"{100.0 * frame.static.maximum_torque_utilization:.1f}%")
                    metrics[5].metric(
                        "Требуемое μ",
                        f"{required_mu:.2f}" if required_mu != float("inf") else "> 3",
                        help="Минимальный коэффициент сцепления, при котором кадр "
                        "уравновешен при неограниченном моменте колёс.",
                    )
                    st.dataframe(_contact_rows(frame.static), width="stretch", hide_index=True)
                    st.markdown("**Полный расчёт кадра**")
                    for equation in frame.static.equations:
                        st.markdown(f"- `{equation}`")
                    st.markdown("**Статьи, использованные в этом кадре**")
                    for source_id in frame.static.source_ids:
                        source = SOURCES[source_id]
                        st.markdown(
                            f"- [{source['short']} — {source['title']}]({source['url']}) — "
                            f"{source['used_for']}."
                        )

with method_tab:
    st.subheader("Что именно рассчитывает программа")
    st.markdown(
        """
1. Для заданной высоты ступени или ширины рва ищется непрерывная последовательность положений подвески без пересечения колёс и рычагов с препятствием. Наибольшее такое препятствие — **геометрический предел**.
2. Для каждого положения собираются левая и правая стороны. На одностороннем препятствии угол корпуса равен среднему углу rocker, а разница высот главных шарниров задаёт крен.
3. Решаются уравнения квазистатического равновесия всего ровера, свободного шарнира каждого bogie и дифференциала. Нормальные реакции не могут быть отрицательными. Сила, передаваемая грунтом, ограничена сцеплением; момент каждого привода — пиковым моментом мотор-колеса с учётом сопротивления качению.
4. Распределение тяги между колёсами выбирается так, чтобы минимизировать максимальную загрузку приводов (идеальное управление тягой). Именно так задние колёса прижимают переднее к стенке: их тяга создаёт нормальную реакцию стенки, а сила трения на ней поднимает колесо.
5. Проверяются продольный, поперечный и общий запас опорного многоугольника. Для рва колесо может потерять контакт, если оставшаяся система опор сохраняет равновесие.
6. Бинарным поиском внутри геометрического предела определяется **физический предел**. В оптимизации сначала выбирается максимальная физическая ступень, после чего для этой же геометрии отдельно рассчитывается ров.

**Ориентир для вертикальной стенки.** Пусть колесо на стенке несёт долю веса F из полного W, а остальные колёса упираются с тем же μ. Подъём возможен при μ²·(W − F) ≥ F. При F = W/3 нужно μ ≥ 0,71, при F = 0,6·W — уже μ ≥ 1,22. Поэтому положение главного шарнира относительно центра масс сильнее всего влияет на физический предел.
        """
    )
    st.warning(
        "Модель считает рычаги и колёса абсолютно жёсткими, не моделирует "
        "деформацию грунта, удар о кромку, прочность рычагов и нагрев двигателей. "
        "Коэффициент динамики увеличивает расчётный вес, но не заменяет динамическую модель."
    )
    st.subheader("Основные формулы")
    st.code(
        """Wр = m·g·Kд
φплатформы = (θлевого rocker + θправого rocker) / 2
ΣF = 0; ΣM = 0
Nᵢ ≥ 0
|Tᵢ| + |Qᵢ| ≤ μ·Nᵢ        — сцепление ограничивает силу от грунта
Dᵢ = Tᵢ + crr·Nᵢ          — сила привода с учётом потерь качения
|Dᵢ|·R ≤ τmax"""
    )
    st.subheader("Использованные публикации")
    for source in SOURCES.values():
        st.markdown(
            f"- [{source['short']} — {source['title']}]({source['url']}) — "
            f"{source['used_for']}."
        )
