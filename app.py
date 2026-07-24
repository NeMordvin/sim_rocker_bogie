import streamlit as st

st.set_page_config(
    page_title="Rocker-Bogie Simulator",
    layout="wide",
)


st.title("Расчёт шасси Rocker-Bogie")

st.sidebar.header("Параметры шасси")

wheel_radius = st.sidebar.number_input(
    "Радиус колеса, мм",
    min_value=10.0,
    max_value=1000.0,
    value=127.0,
    step=1.0,
)

rocker_length = st.sidebar.number_input(
    "Длина rocker, мм",
    min_value=50.0,
    max_value=3000.0,
    value=500.0,
    step=10.0,
)

bogie_length = st.sidebar.number_input(
    "Длина bogie, мм",
    min_value=50.0,
    max_value=2000.0,
    value=300.0,
    step=10.0,
)

st.subheader("Исходные параметры")

col1, col2, col3 = st.columns(3)

col1.metric("Радиус колеса", f"{wheel_radius:.0f} мм")
col2.metric("Длина rocker", f"{rocker_length:.0f} мм")
col3.metric("Длина bogie", f"{bogie_length:.0f} мм")

st.subheader("Предварительные результаты")

wheel_diameter = wheel_radius * 2
total_length = rocker_length + bogie_length + wheel_diameter

st.write(f"Диаметр колеса: **{wheel_diameter:.0f} мм**")
st.write(f"Ориентировочная длина одного борта: **{total_length:.0f} мм**")