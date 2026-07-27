import plotly.graph_objects as go

from geometry import ChassisPoints, BodyPoints


def create_chassis_figure(
    points: ChassisPoints,
    body: BodyPoints,
    wheel_radius: float,
) -> go.Figure:
    figure = go.Figure()
    # Корпус
    figure.add_trace(
        go.Scatter(
            x=[
                body.bottom_left.x,
                body.bottom_right.x,
                body.top_right.x,
                body.top_left.x,
                body.bottom_left.x,
            ],
            y=[
                body.bottom_left.y,
                body.bottom_right.y,
                body.top_right.y,
                body.top_left.y,
                body.bottom_left.y,
            ],
            mode="lines",
            name="Корпус",
            line=dict(
                width=4,
                color="firebrick",
            ),
            fill="toself",
            fillcolor="rgba(178, 34, 34, 0.25)",
        )
    )

    # Rocker: переднее колесо → главный шарнир → шарнир bogie
    figure.add_trace(
        go.Scatter(
            x=[
                points.front_wheel.x,
                points.main_pivot.x,
                points.bogie_pivot.x,
            ],
            y=[
                points.front_wheel.y,
                points.main_pivot.y,
                points.bogie_pivot.y,
            ],
            mode="lines+markers",
            name="Rocker",
            line=dict(width=5, color="royalblue"),
        )
    )

    # Bogie: среднее колесо → шарнир bogie → заднее колесо
    figure.add_trace(
        go.Scatter(
            x=[
                points.middle_wheel.x,
                points.bogie_pivot.x,
                points.rear_wheel.x,
            ],
            y=[
                points.middle_wheel.y,
                points.bogie_pivot.y,
                points.rear_wheel.y,
            ],
            mode="lines+markers",
            name="Bogie",
            line=dict(width=5, color="darkorange"),
        )
    )

    wheel_centers = [
        points.front_wheel,
        points.middle_wheel,
        points.rear_wheel,
    ]

    # Колёса
    for wheel in wheel_centers:
        figure.add_shape(
            type="circle",
            x0=wheel.x - wheel_radius,
            y0=wheel.y - wheel_radius,
            x1=wheel.x + wheel_radius,
            y1=wheel.y + wheel_radius,
            line=dict(color="black", width=4),
            fillcolor="lightgray",
            layer="below",
        )

    # Поверхность земли
    figure.add_hline(
        y=0,
        line_width=3,
        line_color="green",
    )

    figure.update_layout(
        title="Геометрия шасси — вид сбоку",
        height=650,
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    figure.update_xaxes(
        title="X, мм",
        scaleanchor="y",
        scaleratio=1,
    )

    figure.update_yaxes(
        title="Y, мм",
    )

    return figure