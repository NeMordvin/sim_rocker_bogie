"""Интерфейс программы геометрической оптимизации rocker-bogie."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from drawing import create_design_figure, create_traversal_animation
from geometry import (
    CapabilityResult,
    ChassisDesign,
    DesignEvaluation,
    SolverSettings,
    find_obstacle_limit,
    neutral_dimensions,
    optimize_designs,
    pareto_front,
    validate_design,
)


st.set_page_config(
    page_title="Оптимизация Rocker-Bogie",
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
    group: str


PARAMETERS = (
    ParameterSpec(
        "wheel_radius_mm",
        "Радиус колеса",
        "мм",
        70.0,
        150.0,
        False,
        "Колёса и оси",
    ),
    ParameterSpec(
        "middle_wheel_x_mm",
        "X среднего колеса",
        "мм",
        260.0,
        470.0,
        True,
        "Колёса и оси",
    ),
    ParameterSpec(
        "front_wheel_x_mm",
        "X одиночного переднего колеса",
        "мм",
        760.0,
        1000.0,
        True,
        "Колёса и оси",
    ),
    ParameterSpec(
        "main_pivot_x_mm",
        "X главного шарнира",
        "мм",
        480.0,
        730.0,
        True,
        "Главный шарнир и bogie",
    ),
    ParameterSpec(
        "main_pivot_y_mm",
        "Y главного шарнира над центрами колёс",
        "мм",
        240.0,
        390.0,
        True,
        "Главный шарнир и bogie",
    ),
    ParameterSpec(
        "bogie_pivot_x_mm",
        "X шарнира bogie",
        "мм",
        130.0,
        230.0,
        True,
        "Главный шарнир и bogie",
    ),
    ParameterSpec(
        "bogie_pivot_y_mm",
        "Высота верхней перемычки bogie",
        "мм",
        180.0,
        300.0,
        True,
        "Главный шарнир и bogie",
    ),
    ParameterSpec(
        "bogie_rear_knee_x_mm",
        "X заднего верхнего угла bogie",
        "мм",
        55.0,
        125.0,
        True,
        "Главный шарнир и bogie",
    ),
    ParameterSpec(
        "bogie_front_knee_x_mm",
        "X переднего верхнего угла bogie",
        "мм",
        245.0,
        335.0,
        True,
        "Главный шарнир и bogie",
    ),
    ParameterSpec(
        "body_length_mm",
        "Длина корпуса",
        "мм",
        450.0,
        680.0,
        False,
        "Корпус",
    ),
    ParameterSpec(
        "body_height_mm",
        "Высота корпуса",
        "мм",
        170.0,
        310.0,
        False,
        "Корпус",
    ),
    ParameterSpec(
        "attachment_x_mm",
        "Крепление от левого края корпуса",
        "мм",
        220.0,
        400.0,
        True,
        "Корпус",
    ),
    ParameterSpec(
        "attachment_y_mm",
        "Крепление от нижней грани корпуса",
        "мм",
        45.0,
        130.0,
        True,
        "Корпус",
    ),
    ParameterSpec(
        "rocker_thickness_mm",
        "Толщина rocker",
        "мм",
        20.0,
        55.0,
        False,
        "Толщина и центр масс",
    ),
    ParameterSpec(
        "bogie_thickness_mm",
        "Толщина bogie",
        "мм",
        20.0,
        55.0,
        False,
        "Толщина и центр масс",
    ),
    ParameterSpec(
        "com_x_from_body_left_mm",
        "X центра масс от левого края корпуса",
        "мм",
        170.0,
        390.0,
        False,
        "Толщина и центр масс",
    ),
    ParameterSpec(
        "com_y_from_body_bottom_mm",
        "Y центра масс от днища корпуса",
        "мм",
        70.0,
        210.0,
        False,
        "Толщина и центр масс",
    ),
)

GROUPS = tuple(dict.fromkeys(spec.group for spec in PARAMETERS))


def _default_parameter_table(group: str) -> pd.DataFrame:
    defaults = asdict(ChassisDesign())
    rows: list[dict[str, object]] = []
    index: list[str] = []
    for spec in PARAMETERS:
        if spec.group != group:
            continue
        index.append(spec.field)
        rows.append(
            {
                "Параметр": spec.label,
                "Значение": float(defaults[spec.field]),
                "Свободный": spec.free_by_default,
                "Минимум": spec.minimum,
                "Максимум": spec.maximum,
                "Ед.": spec.unit,
            }
        )
    return pd.DataFrame(rows, index=index)


def _render_parameter_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for group_index, group in enumerate(GROUPS):
        storage_key = f"parameter_table_{group_index}"
        widget_key = f"parameter_editor_{group_index}"
        if storage_key not in st.session_state:
            st.session_state[storage_key] = _default_parameter_table(group)
        with st.expander(group, expanded=group_index < 2):
            st.caption(
                "Снимите флажок «Свободный», чтобы оптимизатор не менял "
                "этот параметр."
            )
            edited = st.data_editor(
                st.session_state[storage_key],
                key=widget_key,
                hide_index=True,
                width="stretch",
                num_rows="fixed",
                disabled=["Параметр", "Ед."],
                column_config={
                    "Параметр": st.column_config.TextColumn(width="large"),
                    "Значение": st.column_config.NumberColumn(format="%.1f"),
                    "Свободный": st.column_config.CheckboxColumn(),
                    "Минимум": st.column_config.NumberColumn(format="%.1f"),
                    "Максимум": st.column_config.NumberColumn(format="%.1f"),
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
            value = float(row["Значение"])
            minimum = float(row["Минимум"])
            maximum = float(row["Максимум"])
            values[field] = value
            if bool(row["Свободный"]):
                if minimum >= maximum:
                    errors.append(
                        f"Для «{row['Параметр']}» минимум должен быть меньше максимума."
                    )
                else:
                    free_ranges[field] = (minimum, maximum)
    return values, free_ranges, errors


def _apply_design_to_tables(design: ChassisDesign) -> None:
    values = asdict(design)
    for group_index, group in enumerate(GROUPS):
        storage_key = f"parameter_table_{group_index}"
        widget_key = f"parameter_editor_{group_index}"
        table = st.session_state.get(
            storage_key,
            _default_parameter_table(group),
        ).copy()
        for field in table.index:
            table.loc[field, "Значение"] = float(values[field])
        st.session_state[storage_key] = table
        st.session_state.pop(widget_key, None)
    st.session_state.pop("limit_results", None)


def _design_rows(design: ChassisDesign) -> list[dict[str, object]]:
    values = asdict(design)
    return [
        {
            "Параметр": spec.label,
            "Значение": round(float(values[spec.field]), 2),
            "Ед.": spec.unit,
        }
        for spec in PARAMETERS
    ]


@st.cache_data(show_spinner=False)
def _cached_limit(
    design: ChassisDesign,
    kind: str,
    direction: int,
    maximum: float,
    settings: SolverSettings,
    iterations: int,
) -> CapabilityResult:
    return find_obstacle_limit(
        design=design,
        obstacle_kind=kind,  # type: ignore[arg-type]
        direction=direction,
        maximum_value_mm=maximum,
        settings=settings,
        iterations=iterations,
    )


def _show_limit_results(
    results: dict[str, CapabilityResult],
) -> None:
    columns = st.columns(4)
    columns[0].metric(
        "Ступень: одиночное колесо вперёд",
        f"{results['step_single'].value_mm:.0f} мм",
    )
    columns[1].metric(
        "Ступень: bogie вперёд",
        f"{results['step_bogie'].value_mm:.0f} мм",
    )
    columns[2].metric(
        "Ров: одиночное колесо вперёд",
        f"{results['ditch_single'].value_mm:.0f} мм",
    )
    columns[3].metric(
        "Ров: bogie вперёд",
        f"{results['ditch_bogie'].value_mm:.0f} мм",
    )

    rows = [
        {
            "Проверка": "Ступень",
            "Направление": "одиночное колесо вперёд",
            "Предел, мм": round(results["step_single"].value_mm, 1),
            "Мин. фактический запас опоры, мм": "не проверяется",
        },
        {
            "Проверка": "Ступень",
            "Направление": "bogie вперёд",
            "Предел, мм": round(results["step_bogie"].value_mm, 1),
            "Мин. фактический запас опоры, мм": "не проверяется",
        },
        {
            "Проверка": "Глубокий ров",
            "Направление": "одиночное колесо вперёд",
            "Предел, мм": round(results["ditch_single"].value_mm, 1),
            "Мин. фактический запас опоры, мм": (
                round(
                    results["ditch_single"].traversal.minimum_support_margin_mm,
                    1,
                )
                if results["ditch_single"].traversal
                else "—"
            ),
        },
        {
            "Проверка": "Глубокий ров",
            "Направление": "bogie вперёд",
            "Предел, мм": round(results["ditch_bogie"].value_mm, 1),
            "Мин. фактический запас опоры, мм": (
                round(
                    results["ditch_bogie"].traversal.minimum_support_margin_mm,
                    1,
                )
                if results["ditch_bogie"].traversal
                else "—"
            ),
        },
    ]
    st.dataframe(rows, width="stretch", hide_index=True)

    capped = [
        name
        for name, result in results.items()
        if result.reached_search_cap
    ]
    if capped:
        st.info(
            "Для части проверок достигнута заданная верхняя граница поиска. "
            "Чтобы найти реальный предел, увеличьте её в боковой панели."
        )


def _optimization_rows(
    evaluations: list[DesignEvaluation],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for index, evaluation in enumerate(evaluations[:20], start=1):
        rows.append(
            {
                "Место": index,
                "Ступень в обе стороны, мм": round(
                    evaluation.bidirectional_step_mm,
                    1,
                ),
                "Ров в обе стороны, мм": round(
                    evaluation.bidirectional_ditch_mm,
                    1,
                ),
                "Ступень одиночным, мм": round(
                    evaluation.step_single_first_mm,
                    1,
                ),
                "Ступень bogie, мм": round(
                    evaluation.step_bogie_first_mm,
                    1,
                ),
                "Ров одиночным, мм": round(
                    evaluation.ditch_single_first_mm,
                    1,
                ),
                "Ров bogie, мм": round(
                    evaluation.ditch_bogie_first_mm,
                    1,
                ),
            }
        )
    return rows


st.title("Геометрическая оптимизация Rocker-Bogie")
st.caption(
    "Косвенно-угловой bogie, одиночное колесо спереди. "
    "Трение, мощность, деформация колёс и динамика не учитываются."
)

with st.sidebar:
    st.header("Жёсткие ограничения")
    max_total_length = st.number_input(
        "Максимальная полная длина, мм",
        min_value=300.0,
        value=1200.0,
        step=25.0,
    )
    support_margin = st.number_input(
        "Запас проекции ЦМ до края опоры, мм",
        min_value=0.0,
        value=10.0,
        step=2.0,
    )
    obstacle_clearance = st.number_input(
        "Минимальный зазор до препятствия, мм",
        min_value=0.0,
        value=3.0,
        step=1.0,
    )
    self_clearance = st.number_input(
        "Зазор между деталями, мм",
        min_value=0.0,
        value=3.0,
        step=1.0,
    )

    st.subheader("Пределы поворота")
    bogie_min = st.number_input(
        "Bogie: минимум, °",
        min_value=-170.0,
        max_value=0.0,
        value=-80.0,
        step=5.0,
    )
    bogie_max = st.number_input(
        "Bogie: максимум, °",
        min_value=0.0,
        max_value=170.0,
        value=80.0,
        step=5.0,
    )
    pitch_min = st.number_input(
        "Корпус: минимум, °",
        min_value=-89.0,
        max_value=0.0,
        value=-65.0,
        step=5.0,
    )
    pitch_max = st.number_input(
        "Корпус: максимум, °",
        min_value=0.0,
        max_value=89.0,
        value=65.0,
        step=5.0,
    )

    st.subheader("Область поиска препятствий")
    maximum_step = st.number_input(
        "Искать ступень до, мм",
        min_value=20.0,
        value=700.0,
        step=25.0,
    )
    maximum_ditch = st.number_input(
        "Искать ров до, мм",
        min_value=20.0,
        value=600.0,
        step=25.0,
    )


geometry_tab, limits_tab, optimization_tab, animation_tab = st.tabs(
    ["1. Геометрия", "2. Пределы", "3. Оптимизация", "4. Анимация"]
)

with geometry_tab:
    st.subheader("Параметры конструкции")
    parameter_tables = _render_parameter_tables()

values, free_ranges, table_errors = _table_values(parameter_tables)
design = ChassisDesign(
    **values,
    bogie_min_angle_deg=float(bogie_min),
    bogie_max_angle_deg=float(bogie_max),
    body_pitch_min_deg=float(pitch_min),
    body_pitch_max_deg=float(pitch_max),
    support_margin_mm=float(support_margin),
    obstacle_clearance_mm=float(obstacle_clearance),
    self_clearance_mm=float(self_clearance),
    max_total_length_mm=float(max_total_length),
)
design_errors = [*table_errors, *validate_design(design)]

with geometry_tab:
    if design_errors:
        for error in design_errors:
            st.error(error)
    else:
        dimensions = neutral_dimensions(design)
        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Полная длина",
            f"{dimensions['total_length_mm']:.0f} мм",
        )
        metric_columns[1].metric(
            "Полная высота",
            f"{dimensions['total_height_mm']:.0f} мм",
        )
        metric_columns[2].metric(
            "Колёсная база",
            f"{dimensions['wheelbase_mm']:.0f} мм",
        )
        metric_columns[3].metric(
            "Клиренс корпуса",
            f"{dimensions['body_ground_clearance_mm']:.0f} мм",
        )
        st.plotly_chart(
            create_design_figure(design),
            width="stretch",
            config={"displaylogo": False},
        )

        with st.expander("Углы, автоматически полученные из координат"):
            st.dataframe(
                [
                    {
                        "Звено": "Главный шарнир → одиночное колесо",
                        "Угол, °": round(
                            dimensions["rocker_front_angle_deg"],
                            2,
                        ),
                    },
                    {
                        "Звено": "Главный шарнир → шарнир bogie",
                        "Угол, °": round(
                            dimensions["rocker_bogie_angle_deg"],
                            2,
                        ),
                    },
                    {
                        "Звено": "Шарнир bogie → среднее колесо",
                        "Угол, °": round(
                            dimensions["bogie_middle_angle_deg"],
                            2,
                        ),
                    },
                    {
                        "Звено": "Шарнир bogie → заднее колесо",
                        "Угол, °": round(
                            dimensions["bogie_rear_angle_deg"],
                            2,
                        ),
                    },
                ],
                width="stretch",
                hide_index=True,
            )

with limits_tab:
    st.subheader("Геометрические пределы текущего варианта")
    accuracy = st.radio(
        "Режим расчёта",
        ["Быстрый", "Точный"],
        horizontal=True,
        help=(
            "Быстрый режим удобен для проверки параметров. "
            "Точный использует более мелкий шаг и лучше подходит для итоговой анимации."
        ),
    )
    if st.button(
        "Рассчитать ступень и ров",
        type="primary",
        disabled=bool(design_errors),
    ):
        settings = (
            SolverSettings.coarse()
            if accuracy == "Быстрый"
            else SolverSettings.fine()
        )
        iterations = 6 if accuracy == "Быстрый" else 7
        jobs = (
            ("step_single", "step", 1, maximum_step),
            ("step_bogie", "step", -1, maximum_step),
            ("ditch_single", "ditch", 1, maximum_ditch),
            ("ditch_bogie", "ditch", -1, maximum_ditch),
        )
        progress = st.progress(0.0, text="Подготовка расчёта")
        calculated: dict[str, CapabilityResult] = {}
        for index, (key, kind, direction, maximum) in enumerate(jobs, start=1):
            direction_name = (
                "одиночное колесо"
                if direction == 1
                else "bogie"
            )
            progress.progress(
                (index - 1) / len(jobs),
                text=f"Расчёт: {kind}, впереди {direction_name}",
            )
            calculated[key] = _cached_limit(
                design,
                kind,
                direction,
                float(maximum),
                settings,
                iterations,
            )
        progress.progress(1.0, text="Расчёт завершён")
        st.session_state["limit_results"] = {
            "design": design,
            "values": calculated,
            "accuracy": accuracy,
        }

    stored_limits = st.session_state.get("limit_results")
    if stored_limits is not None:
        if stored_limits["design"] != design:
            st.warning(
                "Параметры изменены. Нажмите кнопку расчёта ещё раз, "
                "чтобы обновить результаты."
            )
        else:
            _show_limit_results(stored_limits["values"])
            st.caption(
                "Это геометрические пределы. Успех означает наличие "
                "непрерывного пути без столкновения. Для рва дополнительно "
                "проверяется проекция центра масс относительно опор."
            )

with optimization_tab:
    st.subheader("Поиск одного шасси для ступени и рва")
    free_count = len(free_ranges)
    st.write(f"Свободных параметров: **{free_count}**")

    control_left, control_middle, control_right = st.columns(3)
    candidate_count = control_left.number_input(
        "Количество вариантов",
        min_value=2,
        max_value=100,
        value=6,
        step=2,
        help="Увеличение повышает шанс найти лучший вариант, но удлиняет расчёт.",
    )
    step_priority = control_middle.slider(
        "Приоритет ступени",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0 — приоритет рва, 1 — приоритет ступени, 0.5 — баланс.",
    )
    random_seed = control_right.number_input(
        "Seed",
        min_value=0,
        value=42,
        step=1,
    )

    if st.button(
        "Запустить оптимизацию",
        type="primary",
        disabled=bool(design_errors) or free_count == 0,
    ):
        progress = st.progress(0.0, text="Создание вариантов")

        def update_progress(done: int, total: int) -> None:
            progress.progress(
                done / max(total, 1),
                text=f"Проверено вариантов: {done} из {total}",
            )

        ranked, best = optimize_designs(
            base=design,
            free_ranges=free_ranges,
            count=int(candidate_count),
            seed=int(random_seed),
            maximum_step_mm=float(maximum_step),
            maximum_ditch_mm=float(maximum_ditch),
            step_weight=float(step_priority),
            progress_callback=update_progress,
        )
        st.session_state["optimization_results"] = {
            "source_design": design,
            "ranked": ranked,
            "best": best,
            "free_ranges": free_ranges,
        }
        progress.progress(1.0, text="Оптимизация завершена")

    stored_optimization = st.session_state.get("optimization_results")
    if stored_optimization is not None:
        ranked = stored_optimization["ranked"]
        best = stored_optimization["best"]
        if best is None:
            st.error("Не найдено ни одного допустимого варианта.")
        else:
            best_columns = st.columns(2)
            best_columns[0].metric(
                "Ступень в обоих направлениях",
                f"{best.bidirectional_step_mm:.0f} мм",
            )
            best_columns[1].metric(
                "Ров в обоих направлениях",
                f"{best.bidirectional_ditch_mm:.0f} мм",
            )

            if st.button("Применить лучший вариант к параметрам"):
                _apply_design_to_tables(best.design)
                st.rerun()

            with st.expander("Параметры лучшего варианта", expanded=True):
                st.dataframe(
                    _design_rows(best.design),
                    width="stretch",
                    hide_index=True,
                )

            st.dataframe(
                _optimization_rows(ranked),
                width="stretch",
                hide_index=True,
            )

            frontier = pareto_front(ranked)
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=[
                        item.bidirectional_ditch_mm
                        for item in ranked
                    ],
                    y=[
                        item.bidirectional_step_mm
                        for item in ranked
                    ],
                    mode="markers",
                    name="Проверенные варианты",
                    marker={"size": 8, "color": "lightgray"},
                    text=[f"Вариант {index + 1}" for index in range(len(ranked))],
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=[
                        item.bidirectional_ditch_mm
                        for item in frontier
                    ],
                    y=[
                        item.bidirectional_step_mm
                        for item in frontier
                    ],
                    mode="lines+markers",
                    name="Компромиссная граница",
                    marker={"size": 10, "color": "royalblue"},
                    line={"color": "royalblue"},
                )
            )
            figure.update_layout(
                title="Компромисс между ступенью и рвом",
                xaxis_title="Ров в обоих направлениях, мм",
                yaxis_title="Ступень в обоих направлениях, мм",
                height=500,
                margin={"l": 20, "r": 20, "t": 55, "b": 20},
            )
            st.plotly_chart(
                figure,
                width="stretch",
                config={"displaylogo": False},
            )

with animation_tab:
    st.subheader("Анимация найденного предельного препятствия")
    stored_limits = st.session_state.get("limit_results")
    if stored_limits is None or stored_limits["design"] != design:
        st.info(
            "Сначала рассчитайте текущий вариант на вкладке «Пределы»."
        )
    else:
        animation_options = {
            "Ступень — одиночное колесо вперёд": "step_single",
            "Ступень — bogie вперёд": "step_bogie",
            "Ров — одиночное колесо вперёд": "ditch_single",
            "Ров — bogie вперёд": "ditch_bogie",
        }
        selected_label = st.selectbox(
            "Что показать",
            list(animation_options),
        )
        selected = stored_limits["values"][
            animation_options[selected_label]
        ]
        if selected.traversal is None:
            st.warning("Для этой проверки не найден успешный путь.")
        else:
            st.plotly_chart(
                create_traversal_animation(
                    design,
                    selected.traversal,
                ),
                width="stretch",
                config={"displaylogo": False},
            )
            st.write(
                (
                    "Минимальный фактический запас проекции центра масс: "
                    f"**{selected.traversal.minimum_support_margin_mm:.1f} мм**."
                    if selected.traversal.obstacle_kind == "ditch"
                    else (
                        "Для ступени все три колеса борта сохраняют контакт "
                        "с нижней площадкой, стенкой, кромкой или верхней "
                        "площадкой. Силы, сцепление и моменты двигателей пока "
                        "не рассчитываются."
                    )
                )
            )
