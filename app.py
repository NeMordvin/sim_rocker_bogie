"""Streamlit-интерфейс: подбор геометрии rocker-bogie на физике MuJoCo."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace

import pandas as pd
import streamlit as st

from drawing import animation_figure, animation_gif, design_figure, timeseries_figure
from model import (
    M365_WHEEL_RADIUS_MM,
    M365_WHEEL_WIDTH_MM,
    Design,
    Platform,
    design_from_dict,
    link_dimensions,
    overall_length_mm,
    validate,
)
from optimize import Criterion, Evaluation, Goal, OptimizationResult, evaluate_parallel, optimize
from simulate import DIRECTION_LABELS, KIND_LABELS, Obstacle, SimResult, run



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
    Spec("middle_wheel_x_mm", "X среднего колеса (от заднего)", "мм", 230, 500, True, True, "Колёса"),
    Spec("front_wheel_x_mm", "X переднего колеса (от заднего)", "мм", 700, 984, True, True, "Колёса"),
    Spec("main_pivot_x_mm", "X главного шарнира", "мм", 450, 850, True, True, "Шарниры"),
    Spec("main_pivot_z_mm", "Высота главного шарнира над землёй", "мм", 230, 550, True, True, "Шарниры"),
    Spec("bogie_pivot_x_mm", "X шарнира bogie", "мм", 60, 330, True, True, "Шарниры"),
    Spec("bogie_pivot_z_mm", "Высота шарнира bogie над землёй", "мм", 220, 500, True, True, "Шарниры"),
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


def _limits_rows(evaluation: Evaluation) -> list[dict[str, object]]:
    return [
        {
            "Препятствие": KIND_LABELS[limit.obstacle.kind],
            "Направление": DIRECTION_LABELS[limit.obstacle.direction],
            "Сценарий": "под левым бортом" if limit.obstacle.one_side else "по всей ширине",
            "Предел, мм": round(limit.value_mm),
            "Что ограничило": limit.reason,
        }
        for limit in evaluation.limits
    ]


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
    )


def _load_variant(raw: bytes) -> None:
    data = json.loads(raw.decode("utf-8"))
    _apply_design(design_from_dict(data.get("design", {})))
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
        size = columns[1].number_input("Ширина рва, мм" if kind == "ditch" else "Высота ступени, мм",
                                       min_value=0.0, value=40.0, step=10.0)
        direction = columns[2].selectbox("Направление", (1, -1), format_func=DIRECTION_LABELS.get)
        one_side = columns[3].checkbox("Только под левым бортом")
        depth = columns[4].number_input("Глубина рва, мм", min_value=50.0, value=500.0, step=50.0,
                                        disabled=kind != "ditch")
        obstacle = Obstacle(kind, float(size), int(direction), bool(one_side), float(depth))
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
        st.write("Бинарный поиск наибольшей ступени, высоты съезда и ширины рва в обоих направлениях. "
                 "Каждый предел считается в отдельном процессе.")
        columns = st.columns(4)
        up_max = columns[0].number_input("Ступень: искать до, мм", min_value=20.0, value=300.0, step=25.0)
        down_max = columns[1].number_input("Съезд: искать до, мм", min_value=20.0, value=500.0, step=25.0)
        ditch_max = columns[2].number_input("Ров: искать до, мм", min_value=20.0, value=500.0, step=25.0)
        limits_one_side = columns[3].checkbox("Добавить «под одним бортом»", value=True)
        limits_goal = Goal(
            criteria=(Criterion("step_up", 1.0, up_max), Criterion("step_down", 1.0, down_max),
                      Criterion("ditch", 1.0, ditch_max)),
            one_side=limits_one_side,
        )
        if st.button("Найти пределы", type="primary", disabled=bool(errors)):
            bar = st.progress(0.0, text="Поиск пределов")
            evaluation = evaluate_parallel(
                design, platform, limits_goal,
                progress=lambda done, total: bar.progress(done / total, text=f"Пределы: {done} из {total}"),
            )
            st.session_state["limits"] = (design, platform, limits_goal, evaluation)
        stored = st.session_state.get("limits")
        if stored is not None:
            limits_design, limits_platform, stored_goal, evaluation = stored
            if (limits_design, limits_platform) != (design, platform):
                st.info("Геометрия или платформа изменились — пределы относятся к прошлому варианту.")
            st.dataframe(_limits_rows(evaluation), width="stretch", hide_index=True)
            options = {
                f"{KIND_LABELS[l.obstacle.kind]}, {DIRECTION_LABELS[l.obstacle.direction]}"
                f"{', под левым бортом' if l.obstacle.one_side else ''}: {l.value_mm:.0f} мм": l
                for l in evaluation.limits
            }
            chosen = st.selectbox("Показать прохождение на пределе", tuple(options))
            limit = options[chosen]
            _show_simulation(limits_design, limits_platform, limit.obstacle, key="limit")

    with optimization_tab:
        st.subheader("Подбор геометрии")
        st.write(
            "Меняются параметры, отмеченные «Свободный» на вкладке «Геометрия». Оценка варианта — "
            "**Σ вес × предел (мм)**; для каждого препятствия берётся худшее из двух направлений "
            "(и из сценариев, если включён «под одним бортом»). Метод — дифференциальная эволюция; "
            "варианты считаются параллельно на всех ядрах."
        )
        weights = st.columns(3)
        criteria = []
        for column, (crit_kind, weight, upper) in zip(
            weights, (("step_up", 1.0, 300.0), ("step_down", 0.25, 500.0), ("ditch", 0.5, 500.0))
        ):
            with column:
                st.markdown(f"**{KIND_LABELS[crit_kind].capitalize()}**")
                w = st.number_input("Вес", min_value=0.0, value=weight, step=0.05, key=f"weight_{crit_kind}")
                u = st.number_input("Искать до, мм", min_value=20.0, value=upper, step=25.0, key=f"upper_{crit_kind}")
                criteria.append(Criterion(crit_kind, float(w), float(u)))
        columns = st.columns(4)
        opt_one_side = columns[0].checkbox("Учитывать «под одним бортом»", help="Вдвое дольше.")
        population = columns[1].number_input("Вариантов в поколении", min_value=6, max_value=128, value=32, step=4)
        generations = columns[2].number_input("Поколений", min_value=1, max_value=60, value=8, step=1)
        seed = columns[3].number_input("Seed", min_value=0, value=1, step=1)
        goal = Goal(criteria=tuple(criteria), one_side=bool(opt_one_side))
        st.caption(f"Будет оценено {int(population) * (int(generations) + 1)} вариантов.")
        can_run = not errors and bool(free_ranges) and bool(goal.active)
        if st.button("Оптимизировать", type="primary", disabled=not can_run):
            bar = st.progress(0.0, text="Оптимизация")
            result = optimize(
                design, platform, goal, free_ranges, int(population), int(generations), int(seed),
                progress=lambda done, total: bar.progress(done / total, text=f"Оценено {done} из {total}"),
            )
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
                start = next((e for e in result.evaluations if e.design == source), None)
                best = result.best
                metrics = st.columns(4)
                metrics[0].metric("Оценка лучшего", f"{best.score:.0f}",
                                  f"{best.score - start.score:+.0f} к исходному" if start and start.valid else None)
                for column, (crit_kind, value) in zip(metrics[1:], best.values.items()):
                    column.metric(KIND_LABELS[crit_kind], f"{value:.0f} мм")
                if st.button("Применить лучший вариант"):
                    _apply_design(best.design)
                    st.rerun()
                ranked = result.ranked(10)
                st.dataframe(
                    [
                        {"Место": index, "Оценка": round(item.score),
                         **{KIND_LABELS[k]: round(v) for k, v in item.values.items()},
                         **{SPEC_BY_FIELD[name].label: getattr(item.design, name) for name in free_ranges}}
                        for index, item in enumerate(ranked, start=1)
                    ],
                    width="stretch",
                    hide_index=True,
                )
                left, right = st.columns(2)
                left.markdown("**Лучший вариант**")
                left.plotly_chart(design_figure(best.design, opt_platform), width="stretch",
                                  config={"displaylogo": False}, key="best_design")
                right.markdown("**Пределы лучшего варианта**")
                right.dataframe(_limits_rows(best), width="stretch", hide_index=True)

    with method_tab:
        st.subheader("Как устроен расчёт")
        st.markdown(
            """
    **Модель.** По параметрам собирается модель MuJoCo: корпус со свободным движением, два rocker на
    главных шарнирах, связанные классическим дифференциалом (угол левого = −угол правого относительно
    корпуса), на каждом rocker — bogie на свободном шарнире. Колесо висит на стойке высотой *h* с наклоном
    *α*; трубы идут от шарниров к верху стоек. Колёса — цилиндры M365 Pro, у каждого свой мотор с
    ограничением пикового момента. Препятствия — блоки с острой кромкой 90°.

    **Прохождение.** Ровер плавно разгоняется до заданной скорости и едет через препятствие. Успех — все
    шесть колёс оказались за препятствием и опираются на грунт. Неудача и её причина:
    - *не хватает момента колёс* — колесо упёрлось и стоит на пиковом моменте;
    - *пробуксовка* — колёса крутятся, ровер стоит (не хватает μ);
    - *задевает препятствие* — касание трубы или стойки (считается недопустимым);
    - *опрокидывание* — тангаж > 70° или крен > 40°;
    - *колёса висят в воздухе* — подвеска не достаёт до опоры (например, bogie перекинулся).

    **Предел** — наибольший размер препятствия, найденный бинарным поиском с шагом 10 мм.

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
    **Ограничения.** Колёса и грунт жёсткие (шина не обминается о кромку — реальное сцепление на кромке
    выше), прочность, нагрев моторов и люфты не учитываются, рулевые приводы не поворачивают колёса.
    Результат слегка зависит от скорости и способа управления — это динамический расчёт.
            """
        )


# Дочерние процессы пула (spawn в Windows) импортируют этот файл как
# __mp_main__ — интерфейс им строить не нужно.
if __name__ == "__main__":
    main()
