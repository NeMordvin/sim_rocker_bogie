"""Отрисовка геометрии подвески и отдельных ключевых кадров."""

from __future__ import annotations

from math import atan2, cos, hypot, pi, sin

import numpy as np
import plotly.graph_objects as go

from geometry import (
    ChassisDesign,
    ChassisFrame,
    DitchTerrain,
    Point,
    Pose,
    StepTerrain,
    Terrain,
    TraversalResult,
    WHEEL_LABELS,
    calculate_frame,
    neutral_pose,
)
from statics import StaticFrameResult


ROCKER_COLOR = "rgba(48, 96, 190, 0.88)"
BOGIE_COLOR = "rgba(225, 132, 25, 0.90)"
WHEEL_COLOR = "rgba(72, 76, 86, 0.92)"
GROUND_COLOR = "rgba(111, 139, 90, 0.55)"


def _closed_polygon_trace(
    points: list[Point],
    name: str,
    fillcolor: str,
    linecolor: str,
    showlegend: bool = True,
) -> go.Scatter:
    closed = [*points, points[0]]
    return go.Scatter(
        x=[point.x for point in closed],
        y=[point.y for point in closed],
        mode="lines",
        name=name,
        line={"color": linecolor, "width": 1.5},
        fill="toself",
        fillcolor=fillcolor,
        hoverinfo="name",
        showlegend=showlegend,
    )


def _circle_points(center: Point, radius: float, count: int = 48) -> list[Point]:
    return [
        Point(
            center.x + radius * cos(2.0 * pi * index / count),
            center.y + radius * sin(2.0 * pi * index / count),
        )
        for index in range(count)
    ]


def _capsule_points(
    start: Point,
    end: Point,
    radius: float,
    arc_count: int = 10,
) -> list[Point]:
    """Строит контур толстого прямолинейного звена в координатах графика."""
    angle = atan2(end.y - start.y, end.x - start.x)
    if abs(end.x - start.x) + abs(end.y - start.y) < 1e-9:
        return _circle_points(start, radius, max(16, 2 * arc_count))

    points: list[Point] = []
    for value in np.linspace(angle + pi / 2.0, angle - pi / 2.0, arc_count):
        points.append(
            Point(
                end.x + radius * cos(float(value)),
                end.y + radius * sin(float(value)),
            )
        )
    for value in np.linspace(angle - pi / 2.0, angle - 3.0 * pi / 2.0, arc_count):
        points.append(
            Point(
                start.x + radius * cos(float(value)),
                start.y + radius * sin(float(value)),
            )
        )
    return points


def _terrain_traces(
    terrain: Terrain,
    x_min: float,
    x_max: float,
    y_bottom: float,
) -> list[go.Scatter]:
    if isinstance(terrain, StepTerrain):
        height = terrain.height_mm
        points = [
            Point(x_min, y_bottom),
            Point(x_max, y_bottom),
            Point(x_max, height),
            Point(0.0, height),
            Point(0.0, 0.0),
            Point(x_min, 0.0),
        ]
        return [
            _closed_polygon_trace(
                points,
                "Грунт и ступень",
                GROUND_COLOR,
                "darkgreen",
            )
        ]

    assert isinstance(terrain, DitchTerrain)
    left = [
        Point(x_min, y_bottom),
        Point(0.0, y_bottom),
        Point(0.0, 0.0),
        Point(x_min, 0.0),
    ]
    right = [
        Point(terrain.width_mm, y_bottom),
        Point(x_max, y_bottom),
        Point(x_max, 0.0),
        Point(terrain.width_mm, 0.0),
    ]
    return [
        _closed_polygon_trace(
            left,
            "Левый берег",
            GROUND_COLOR,
            "darkgreen",
        ),
        _closed_polygon_trace(
            right,
            "Правый берег",
            GROUND_COLOR,
            "darkgreen",
            showlegend=False,
        ),
    ]


def _contact_points(
    design: ChassisDesign,
    frame: ChassisFrame,
    pose: Pose,
    terrain: Terrain,
) -> list[Point]:
    result: list[Point] = []
    for name, wheel in zip(("front", "middle", "rear"), frame.wheels):
        if name in pose.contacts:
            support_x = terrain.support_x(
                np.asarray([wheel.x]),
                design.wheel_radius_mm,
            )[0]
            if isinstance(terrain, StepTerrain):
                support_y = terrain.height_mm if support_x >= 0.0 else 0.0
            else:
                support_y = 0.0
            result.append(Point(float(support_x), float(support_y)))
        if name in pose.wall_contacts and isinstance(terrain, StepTerrain):
            result.append(Point(0.0, wheel.y))
    return result


def _rover_traces(
    design: ChassisDesign,
    pose: Pose,
    direction: int,
    terrain: Terrain,
    showlegend: bool,
    center_of_mass: Point | None = None,
) -> list[go.Scatter]:
    frame = calculate_frame(design, pose, direction)
    rocker_radius = design.arm_diameter_mm / 2.0
    bogie_radius = design.arm_diameter_mm / 2.0
    rocker_segments = (
        (frame.front_wheel, frame.main_pivot),
        (frame.main_pivot, frame.bogie_pivot),
    )
    bogie_segments = (
        (frame.rear_wheel, frame.bogie_rear_knee),
        (frame.bogie_rear_knee, frame.bogie_front_knee),
        (frame.bogie_front_knee, frame.middle_wheel),
    )

    traces: list[go.Scatter] = []
    for index, (start, end) in enumerate(rocker_segments):
        traces.append(
            _closed_polygon_trace(
                _capsule_points(start, end, rocker_radius),
                "Rocker",
                ROCKER_COLOR,
                "royalblue",
                showlegend and index == 0,
            )
        )
    for index, (start, end) in enumerate(bogie_segments):
        traces.append(
            _closed_polygon_trace(
                _capsule_points(start, end, bogie_radius),
                "Косвенно-угловой bogie",
                BOGIE_COLOR,
                "darkorange",
                showlegend and index == 0,
            )
        )

    for index, (name, wheel) in enumerate(
        zip(("front", "middle", "rear"), frame.wheels)
    ):
        traces.append(
            _closed_polygon_trace(
                _circle_points(wheel, design.wheel_radius_mm),
                f"Колесо: {WHEEL_LABELS[name]}",
                WHEEL_COLOR,
                "black",
                showlegend and index == 0,
            )
        )

    traces.append(
        go.Scatter(
            x=[frame.main_pivot.x, frame.bogie_pivot.x],
            y=[frame.main_pivot.y, frame.bogie_pivot.y],
            mode="markers",
            name="Шарниры",
            marker={"size": 8, "color": "white", "line": {"color": "black", "width": 2}},
            showlegend=showlegend,
        )
    )
    displayed_com = center_of_mass or frame.center_of_mass
    traces.append(
        go.Scatter(
            x=[displayed_com.x],
            y=[displayed_com.y],
            mode="markers+text",
            text=["ЦМ"],
            textposition="top center",
            name="Центр масс",
            marker={"size": 12, "color": "purple", "symbol": "x"},
            showlegend=showlegend,
        )
    )
    traces.append(
        go.Scatter(
            x=[displayed_com.x, displayed_com.x],
            y=[displayed_com.y, 0.0],
            mode="lines",
            name="Проекция ЦМ",
            line={"color": "purple", "width": 2, "dash": "dot"},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    contacts = _contact_points(design, frame, pose, terrain)
    traces.append(
        go.Scatter(
            x=[point.x for point in contacts],
            y=[point.y for point in contacts],
            mode="markers",
            name="Точки опоры",
            marker={"size": 9, "color": "limegreen", "line": {"color": "black", "width": 1}},
            showlegend=showlegend,
        )
    )
    return traces


def _figure_ranges(
    design: ChassisDesign,
    poses: tuple[Pose, ...],
    direction: int,
    terrain: Terrain,
) -> tuple[float, float, float, float]:
    x_values: list[float] = []
    y_values: list[float] = [0.0]
    for pose in poses:
        frame = calculate_frame(design, pose, direction)
        x_values.extend(point.x for point in frame.wheels)
        y_values.extend(point.y for point in frame.wheels)

    if isinstance(terrain, DitchTerrain):
        x_values.extend((0.0, terrain.width_mm))
    else:
        x_values.append(0.0)
        y_values.append(terrain.height_mm)

    radius = design.wheel_radius_mm
    x_margin = max(50.0, 1.3 * radius)
    y_margin = max(50.0, 0.8 * radius)
    x_min = min(x_values) - radius - x_margin
    x_max = max(x_values) + radius + x_margin
    y_min = min(0.0, min(y_values) - radius) - y_margin
    y_max = max(y_values) + radius + y_margin
    return x_min, x_max, y_min, y_max


def _apply_layout(
    figure: go.Figure,
    title: str,
    ranges: tuple[float, float, float, float],
) -> None:
    x_min, x_max, y_min, y_max = ranges
    figure.update_layout(
        title=title,
        height=680,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        hovermode="closest",
    )
    figure.update_xaxes(
        title="X, мм",
        range=[x_min, x_max],
        scaleanchor="y",
        scaleratio=1,
        constrain="domain",
        gridcolor="rgba(0,0,0,0.08)",
    )
    figure.update_yaxes(
        title="Y, мм",
        range=[y_min, y_max],
        gridcolor="rgba(0,0,0,0.08)",
    )


def create_design_figure(
    design: ChassisDesign,
    pose: Pose | None = None,
    direction: int = 1,
) -> go.Figure:
    """Показывает исходную форму с геометрически верной толщиной рычагов."""
    pose = pose or neutral_pose(design)
    terrain = StepTerrain(0.0)
    ranges = _figure_ranges(design, (pose,), direction, terrain)
    terrain_data = _terrain_traces(terrain, ranges[0], ranges[1], ranges[2])
    rover_data = _rover_traces(
        design,
        pose,
        direction,
        terrain,
        showlegend=True,
    )
    figure = go.Figure(data=[*terrain_data, *rover_data])
    _apply_layout(figure, "Исходная геометрия шасси", ranges)
    return figure


def create_key_frame_figure(
    design: ChassisDesign,
    traversal: TraversalResult,
    pose: Pose,
    static_result: StaticFrameResult,
    title: str,
) -> go.Figure:
    """Статический кадр с точками контакта, ЦМ и реакциями левой стороны."""
    terrain: Terrain
    if traversal.obstacle_kind == "step":
        terrain = StepTerrain(traversal.obstacle_value_mm)
    else:
        terrain = DitchTerrain(traversal.obstacle_value_mm)

    ranges = _figure_ranges(
        design,
        (pose,),
        traversal.direction,
        terrain,
    )
    center_of_mass = Point(
        static_result.center_of_mass_mm[0],
        static_result.center_of_mass_mm[2],
    )
    terrain_data = _terrain_traces(terrain, ranges[0], ranges[1], ranges[2])
    rover_data = _rover_traces(
        design,
        pose,
        traversal.direction,
        terrain,
        showlegend=True,
        center_of_mass=center_of_mass,
    )
    figure = go.Figure(data=[*terrain_data, *rover_data])

    left_loads = [
        load for load in static_result.contacts if load.side == "левая"
    ]
    maximum_force = max(
        (
            hypot(
                load.normal_force_n * load.normal[0]
                + load.net_tangent_force_n * load.tangent[0],
                load.normal_force_n * load.normal[2]
                + load.net_tangent_force_n * load.tangent[2],
            )
            for load in left_loads
        ),
        default=1.0,
    )
    force_scale = min(0.35, 170.0 / max(maximum_force, 1.0))
    for index, load in enumerate(left_loads):
        force_x = (
            load.normal_force_n * load.normal[0]
            + load.net_tangent_force_n * load.tangent[0]
        )
        force_z = (
            load.normal_force_n * load.normal[2]
            + load.net_tangent_force_n * load.tangent[2]
        )
        start_x = load.point_mm[0]
        start_z = load.point_mm[2]
        end_x = start_x + force_scale * force_x
        end_z = start_z + force_scale * force_z
        figure.add_trace(
            go.Scatter(
                x=[start_x, end_x],
                y=[start_z, end_z],
                mode="lines+markers",
                name="Реакция опоры" if index == 0 else "Реакция опоры",
                legendgroup="forces",
                showlegend=index == 0,
                line={"color": "crimson", "width": 3},
                marker={"size": [5, 9], "symbol": ["circle", "triangle-up"]},
                customdata=[[load.normal_force_n, load.net_tangent_force_n]] * 2,
                hovertemplate=(
                    f"{WHEEL_LABELS[load.wheel]}<br>"
                    "N=%{customdata[0]:.2f} Н<br>"
                    "T=%{customdata[1]:.2f} Н<extra></extra>"
                ),
            )
        )

    _apply_layout(figure, title, ranges)
    figure.update_layout(height=620)
    return figure


