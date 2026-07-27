from math import isfinite

import streamlit as st

from drawing import create_chassis_figure
from geometry import (
    Point,
    SuspensionParameters,
    VehicleParameters,
    calculate_body_points,
    calculate_chassis_points,
    calculate_mobility_limits,
    calculate_wheel_clearances,
    grade_percent,
)


st.set_page_config(
    page_title="Rocker-Bogie Calculator",
    layout="wide",
)

st.title("Расчёт проходимости Rocker-Bogie")


with st.sidebar:
    st.header("Исходные параметры")

    with st.expander("Колёса", expanded=True):
        wheel_radius = st.number_input(
            "Радиус колеса, мм",
            min_value=10.0,
            value=100.0,
            step=5.0,
        )
        arm_diameter = st.number_input(
            "Расчётный диаметр рычагов, мм",
            min_value=0.0,
            value=30.0,
            step=2.0,
        )

    with st.expander("Главный шарнир"):
        main_pivot_x = st.number_input(
            "Координата главного шарнира X, мм",
            value=0.0,
            step=10.0,
        )
        main_pivot_y = st.number_input(
            "Координата главного шарнира Y, мм",
            value=400.0,
            step=10.0,
        )

    with st.expander("Корпус", expanded=True):
        body_length = st.number_input(
            "Длина корпуса, мм",
            min_value=1.0,
            value=600.0,
            step=10.0,
        )
        body_height = st.number_input(
            "Высота корпуса, мм",
            min_value=1.0,
            value=250.0,
            step=10.0,
        )
        attachment_x = st.number_input(
            "Крепление от левого края, мм",
            min_value=0.0,
            value=300.0,
            step=10.0,
        )
        attachment_y = st.number_input(
            "Крепление от нижней грани, мм",
            min_value=0.0,
            value=50.0,
            step=10.0,
        )

    with st.expander("Rocker", expanded=True):
        front_length = st.number_input(
            "До переднего колеса, мм",
            min_value=1.0,
            value=500.0,
            step=10.0,
        )
        front_angle = st.number_input(
            "Угол переднего плеча, °",
            min_value=-180.0,
            max_value=180.0,
            value=-36.87,
            step=1.0,
        )
        bogie_length = st.number_input(
            "До шарнира bogie, мм",
            min_value=1.0,
            value=335.41,
            step=10.0,
        )
        bogie_angle = st.number_input(
            "Угол заднего плеча, °",
            min_value=-180.0,
            max_value=180.0,
            value=-153.43,
            step=1.0,
        )

    with st.expander("Bogie", expanded=True):
        middle_length = st.number_input(
            "До среднего колеса, мм",
            min_value=1.0,
            value=250.0,
            step=10.0,
        )
        middle_angle = st.number_input(
            "Угол к среднему колесу, °",
            min_value=-180.0,
            max_value=180.0,
            value=-36.87,
            step=1.0,
        )
        rear_length = st.number_input(
            "До заднего колеса, мм",
            min_value=1.0,
            value=250.0,
            step=10.0,
        )
        rear_angle = st.number_input(
            "Угол к заднему колесу, °",
            min_value=-180.0,
            max_value=180.0,
            value=-143.13,
            step=1.0,
        )

    with st.expander("Пределы подвески", expanded=True):
        rocker_min = st.number_input(
            "Минимальный поворот rocker, °",
            min_value=-180.0,
            max_value=179.0,
            value=-45.0,
            step=1.0,
        )
        rocker_max = st.number_input(
            "Максимальный поворот rocker, °",
            min_value=-179.0,
            max_value=180.0,
            value=45.0,
            step=1.0,
        )
        bogie_min = st.number_input(
            "Минимальный поворот bogie, °",
            min_value=-180.0,
            max_value=179.0,
            value=-90.0,
            step=1.0,
        )
        bogie_max = st.number_input(
            "Максимальный поворот bogie, °",
            min_value=-179.0,
            max_value=180.0,
            value=90.0,
            step=1.0,
        )

    with st.expander("Масса и центр масс", expanded=True):
        total_mass = st.number_input(
            "Полная масса с нагрузкой, кг",
            min_value=0.1,
            value=120.0,
            step=5.0,
        )
        track_width = st.number_input(
            "Колея, мм",
            min_value=1.0,
            value=600.0,
            step=10.0,
        )
        com_x_from_left = st.number_input(
            "Центр масс от левого края корпуса, мм",
            value=300.0,
            step=10.0,
        )
        com_height = st.number_input(
            "Высота центра масс над землёй, мм",
            min_value=1.0,
            value=500.0,
            step=10.0,
        )
        lateral_com_offset = st.number_input(
            "Поперечное смещение центра масс, мм",
            value=0.0,
            step=10.0,
        )

    with st.expander("Привод и сцепление", expanded=True):
        wheel_torque = st.number_input(
            "Момент одного мотор-колеса, Н·м",
            min_value=0.0,
            value=20.0,
            step=1.0,
        )
        brake_torque = st.number_input(
            "Тормозной момент одного колеса, Н·м",
            min_value=0.0,
            value=20.0,
            step=1.0,
        )
        driven_wheels = st.number_input(
            "Количество ведущих колёс",
            min_value=1,
            max_value=6,
            value=6,
            step=1,
        )
        drivetrain_efficiency = st.number_input(
            "КПД привода",
            min_value=0.01,
            max_value=1.0,
            value=0.80,
            step=0.05,
        )
        ground_friction = st.number_input(
            "Сцепление колесо–грунт",
            min_value=0.0,
            value=0.70,
            step=0.05,
        )
        obstacle_friction = st.number_input(
            "Сцепление колесо–ступень",
            min_value=0.0,
            value=0.70,
            step=0.05,
        )
        rolling_resistance = st.number_input(
            "Коэффициент сопротивления качению",
            min_value=0.0,
            value=0.03,
            step=0.01,
            format="%.3f",
        )

    with st.expander("Запасы расчёта", expanded=True):
        support_margin = st.number_input(
            "Запас до края опоры, мм",
            min_value=0.0,
            value=20.0,
            step=5.0,
        )
        safety_factor = st.number_input(
            "Коэффициент запаса",
            min_value=1.0,
            value=1.20,
            step=0.05,
        )

    with st.expander("Положение на схеме"):
        rocker_rotation = st.slider(
            "Поворот rocker на схеме, °",
            min_value=-90.0,
            max_value=90.0,
            value=0.0,
            step=1.0,
        )
        bogie_rotation = st.slider(
            "Поворот bogie на схеме, °",
            min_value=-120.0,
            max_value=120.0,
            value=0.0,
            step=1.0,
        )


main_pivot = Point(
    x=main_pivot_x,
    y=main_pivot_y,
)
body = calculate_body_points(
    main_pivot=main_pivot,
    body_length=body_length,
    body_height=body_height,
    attachment_x=attachment_x,
    attachment_y=attachment_y,
)
points = calculate_chassis_points(
    main_pivot=main_pivot,
    front_length=front_length,
    front_angle=front_angle,
    bogie_length=bogie_length,
    bogie_angle=bogie_angle,
    middle_length=middle_length,
    middle_angle=middle_angle,
    rear_length=rear_length,
    rear_angle=rear_angle,
    rocker_rotation=rocker_rotation,
    bogie_rotation=bogie_rotation,
)

suspension = SuspensionParameters(
    wheel_radius_mm=wheel_radius,
    front_length_mm=front_length,
    front_angle_deg=front_angle,
    bogie_length_mm=bogie_length,
    bogie_angle_deg=bogie_angle,
    middle_length_mm=middle_length,
    middle_angle_deg=middle_angle,
    rear_length_mm=rear_length,
    rear_angle_deg=rear_angle,
    rocker_min_deg=rocker_min,
    rocker_max_deg=rocker_max,
    bogie_min_deg=bogie_min,
    bogie_max_deg=bogie_max,
    body_length_mm=body_length,
    body_height_mm=body_height,
    attachment_x_mm=attachment_x,
    attachment_y_mm=attachment_y,
    arm_radius_mm=arm_diameter / 2.0,
)
vehicle = VehicleParameters(
    total_mass_kg=total_mass,
    track_width_mm=track_width,
    com_x_from_body_left_mm=com_x_from_left,
    com_height_mm=com_height,
    lateral_com_offset_mm=lateral_com_offset,
    wheel_torque_nm=wheel_torque,
    brake_torque_nm=brake_torque,
    driven_wheels=int(driven_wheels),
    drivetrain_efficiency=drivetrain_efficiency,
    ground_friction=ground_friction,
    obstacle_friction=obstacle_friction,
    rolling_resistance=rolling_resistance,
    support_margin_mm=support_margin,
    safety_factor=safety_factor,
)

center_of_mass = Point(
    x=body.bottom_left.x + com_x_from_left,
    y=com_height,
)
figure = create_chassis_figure(
    points=points,
    body=body,
    wheel_radius=wheel_radius,
    center_of_mass=center_of_mass,
)


def angle_text(value: float) -> str:
    if not isfinite(value):
        return "—"
    return f"{value:.1f}° ({grade_percent(value):.0f} %)"


result_tab, geometry_tab = st.tabs(
    ["Проходимость", "Геометрия и текущая поза"]
)

with result_tab:
    try:
        results = calculate_mobility_limits(
            suspension=suspension,
            vehicle=vehicle,
        )
    except ValueError as error:
        st.error(str(error))
    else:
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric(
            "Вертикальная ступень",
            f"{results.step.value:.0f} мм",
        )
        metric_2.metric(
            "Глубокий ров",
            f"{results.ditch.value:.0f} мм",
        )
        metric_3.metric(
            "Продольный подъём",
            angle_text(results.uphill.value),
        )
        metric_4.metric(
            "Поперечный уклон",
            angle_text(results.cross_slope.value),
        )

        rows = [
            {
                "Проверка": "Вертикальная ступень",
                "Расчётный предел": f"{results.step.value:.1f} мм",
                "Геометрический предел": (
                    f"{results.step.theoretical_value:.1f} мм"
                ),
                "Ограничивающий фактор": results.step.limiting_factor,
            },
            {
                "Проверка": "Ширина глубокого рва",
                "Расчётный предел": f"{results.ditch.value:.1f} мм",
                "Геометрический предел": (
                    f"{results.ditch.theoretical_value:.1f} мм"
                ),
                "Ограничивающий фактор": results.ditch.limiting_factor,
            },
            {
                "Проверка": "Продольный подъём",
                "Расчётный предел": angle_text(results.uphill.value),
                "Геометрический предел": "—",
                "Ограничивающий фактор": results.uphill.limiting_factor,
            },
            {
                "Проверка": "Продольный спуск",
                "Расчётный предел": angle_text(results.downhill.value),
                "Геометрический предел": "—",
                "Ограничивающий фактор": results.downhill.limiting_factor,
            },
            {
                "Проверка": "Поперечный уклон",
                "Расчётный предел": angle_text(results.cross_slope.value),
                "Геометрический предел": "—",
                "Ограничивающий фактор": results.cross_slope.limiting_factor,
            },
            {
                "Проверка": "Угол въезда",
                "Расчётный предел": f"{results.approach_angle.value:.1f}°",
                "Геометрический предел": "—",
                "Ограничивающий фактор": (
                    results.approach_angle.limiting_factor
                ),
            },
            {
                "Проверка": "Угол съезда",
                "Расчётный предел": f"{results.departure_angle.value:.1f}°",
                "Геометрический предел": "—",
                "Ограничивающий фактор": (
                    results.departure_angle.limiting_factor
                ),
            },
            {
                "Проверка": "Перелом выпуклой рампы",
                "Расчётный предел": f"{results.breakover_angle.value:.1f}°",
                "Геометрический предел": "—",
                "Ограничивающий фактор": (
                    results.breakover_angle.limiting_factor
                ),
            },
            {
                "Проверка": "Минимальный дорожный просвет",
                "Расчётный предел": (
                    f"{results.ground_clearance.value:.1f} мм"
                ),
                "Геометрический предел": (
                    f"{results.ground_clearance.theoretical_value:.1f} мм"
                ),
                "Ограничивающий фактор": (
                    results.ground_clearance.limiting_factor
                ),
            },
        ]
        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )

        with st.expander("Промежуточные пределы ступени"):
            st.write(
                f"Передняя фаза: **{results.step_front_phase_mm:.1f} мм**"
            )
            st.write(
                f"Задняя фаза: **{results.step_rear_phase_mm:.1f} мм**"
            )
            st.write(
                "Предел по тяге и сцеплению: "
                f"**{results.step_force_limit_mm:.1f} мм**"
            )
            if results.wall_climb_possible:
                st.success(
                    "Тяги и сцепления хватает для движения колёс "
                    "по вертикальной грани."
                )
            else:
                st.warning(
                    "Движение по вертикальной грани невозможно; "
                    "ступень ограничена перекатыванием через кромку."
                )

        for warning in results.warnings:
            st.warning(warning)

        st.caption(
            "Расчёт квазистатический, двухмерный и выполнен с заданным "
            "коэффициентом запаса. Итоговые пределы требуют проверки "
            "в многотельной модели и на прототипе."
        )

with geometry_tab:
    attachment_is_valid = (
        0.0 <= attachment_x <= body_length
        and 0.0 <= attachment_y <= body_height
    )
    if not attachment_is_valid:
        st.error(
            "Точка крепления находится за пределами корпуса. "
            "Проверьте её координаты."
        )

    st.plotly_chart(
        figure,
        width="stretch",
    )

    wheel_data = [
        {
            "Колесо": "Переднее",
            "X, мм": round(points.front_wheel.x, 2),
            "Y, мм": round(points.front_wheel.y, 2),
        },
        {
            "Колесо": "Среднее",
            "X, мм": round(points.middle_wheel.x, 2),
            "Y, мм": round(points.middle_wheel.y, 2),
        },
        {
            "Колесо": "Заднее",
            "X, мм": round(points.rear_wheel.x, 2),
            "Y, мм": round(points.rear_wheel.y, 2),
        },
    ]
    st.dataframe(
        wheel_data,
        width="stretch",
        hide_index=True,
    )

    clearances = calculate_wheel_clearances(
        points=points,
        wheel_radius=wheel_radius,
    )
    clearance_values = list(clearances.values())
    height_difference = max(clearance_values) - min(clearance_values)
    ground_error = max(abs(value) for value in clearance_values)

    result_left, result_right = st.columns(2)
    result_left.metric(
        "Разница высот колёс",
        f"{height_difference:.2f} мм",
    )
    result_right.metric(
        "Максимальное отклонение от земли",
        f"{ground_error:.2f} мм",
    )

    tolerance = 0.1
    if height_difference <= tolerance and ground_error <= tolerance:
        st.success("Все три колеса находятся на горизонтальной поверхности.")
    elif height_difference <= tolerance:
        st.warning(
            "Колёса находятся на одной высоте, "
            "но конструкция смещена относительно земли."
        )
    else:
        st.info(
            "На схеме задана артикулированная поза: "
            "колёса находятся на разных высотах."
        )
