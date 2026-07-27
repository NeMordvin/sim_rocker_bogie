import streamlit as st

from drawing import create_chassis_figure
from geometry import (
    Point,
    calculate_body_points,
    calculate_chassis_points,
    calculate_wheel_clearances,
)


st.set_page_config(
    page_title="Rocker-Bogie Calculator",
    layout="wide",
)

st.title("Расчёт геометрии Rocker-Bogie")


# Левая панель с исходными параметрами
with st.sidebar:
    st.header("Исходные параметры")

    with st.expander("Колёса", expanded=True):
        wheel_radius = st.number_input(
            "Радиус колеса, мм",
            min_value=10.0,
            value=100.0,
            step=5.0,
        )

    with st.expander("Главный шарнир", expanded=True):
        main_pivot_x = st.number_input(
            "Координата X, мм",
            value=0.0,
            step=10.0,
        )
        main_pivot_y = st.number_input(
            "Координата Y, мм",
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

    with st.expander("Положение подвески", expanded=True):
        rocker_rotation = st.slider(
            "Поворот rocker, °",
            min_value=-45.0,
            max_value=45.0,
            value=0.0,
            step=1.0,
        )

        bogie_rotation = st.slider(
            "Поворот bogie, °",
            min_value=-90.0,
            max_value=90.0,
            value=0.0,
            step=1.0,
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


# Расчёт координат
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


# Построение схемы
figure = create_chassis_figure(
    points=points,
    body=body,
    wheel_radius=wheel_radius,
)

attachment_is_valid = ( 0.0 <= attachment_x <= body_length
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

st.subheader("Проверка геометрии")

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

tolerance = 0.1

result_left, result_right = st.columns(2)

result_left.metric(
    "Разница высот колёс",
    f"{height_difference:.2f} мм",
)

result_right.metric(
    "Максимальное отклонение от земли",
    f"{ground_error:.2f} мм",
)

if height_difference <= tolerance and ground_error <= tolerance:
    st.success("Все три колеса находятся на горизонтальной поверхности.")

elif height_difference <= tolerance:
    st.warning(
        "Колёса находятся на одной высоте, "
        "но вся конструкция смещена относительно земли."
    )

else:
    st.error(
        "Колёса находятся на разной высоте и не могут одновременно "
        "касаться горизонтальной поверхности."
    )