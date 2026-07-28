"""Отрисовка шасси и анимации прохождения препятствий в Plotly."""

from __future__ import annotations

from math import atan2, cos, pi, sin

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
    downsample_poses,
    neutral_pose,
)


BODY_COLOR = "rgba(178, 34, 34, 0.55)"
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
) -> list[go.Scatter]:
    frame = calculate_frame(design, pose, direction)
    body = _closed_polygon_trace(
        list(frame.body_corners),
        "Корпус",
        BODY_COLOR,
        "firebrick",
        showlegend,
    )

    rocker_radius = design.rocker_thickness_mm / 2.0
    bogie_radius = design.bogie_thickness_mm / 2.0
    rocker_segments = (
        (frame.front_wheel, frame.main_pivot),
        (frame.main_pivot, frame.bogie_pivot),
    )
    bogie_segments = (
        (frame.rear_wheel, frame.bogie_rear_knee),
        (frame.bogie_rear_knee, frame.bogie_front_knee),
        (frame.bogie_front_knee, frame.middle_wheel),
    )

    traces: list[go.Scatter] = [body]
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
    traces.append(
        go.Scatter(
            x=[frame.center_of_mass.x],
            y=[frame.center_of_mass.y],
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
            x=[frame.center_of_mass.x, frame.center_of_mass.x],
            y=[frame.center_of_mass.y, 0.0],
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
        x_values.extend(point.x for point in frame.body_corners)
        y_values.extend(point.y for point in frame.wheels)
        y_values.extend(point.y for point in frame.body_corners)

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


def create_traversal_animation(
    design: ChassisDesign,
    traversal: TraversalResult,
    maximum_frames: int = 80,
) -> go.Figure:
    """Создаёт анимацию с кнопками запуска, паузы и ползунком кадров."""
    if not traversal.success or not traversal.poses:
        raise ValueError("Для анимации требуется успешно найденный путь.")

    poses = downsample_poses(traversal.poses, maximum_frames)
    terrain: Terrain
    if traversal.obstacle_kind == "step":
        terrain = StepTerrain(traversal.obstacle_value_mm)
        obstacle_name = f"ступень {traversal.obstacle_value_mm:.1f} мм"
    else:
        terrain = DitchTerrain(traversal.obstacle_value_mm)
        obstacle_name = f"ров {traversal.obstacle_value_mm:.1f} мм"

    direction_name = (
        "одиночным колесом вперёд"
        if traversal.direction == 1
        else "bogie вперёд"
    )
    ranges = _figure_ranges(design, poses, traversal.direction, terrain)
    terrain_data = _terrain_traces(terrain, ranges[0], ranges[1], ranges[2])
    first_rover = _rover_traces(
        design,
        poses[0],
        traversal.direction,
        terrain,
        showlegend=True,
    )
    figure = go.Figure(data=[*terrain_data, *first_rover])

    rover_trace_indices = list(
        range(len(terrain_data), len(terrain_data) + len(first_rover))
    )
    frames: list[go.Frame] = []
    for index, pose in enumerate(poses):
        frames.append(
            go.Frame(
                name=str(index),
                data=_rover_traces(
                    design,
                    pose,
                    traversal.direction,
                    terrain,
                    showlegend=False,
                ),
                traces=rover_trace_indices,
            )
        )
    figure.frames = frames

    slider_steps = [
        {
            "args": [
                [str(index)],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": str(index + 1),
            "method": "animate",
        }
        for index in range(len(frames))
    ]
    figure.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.08,
                "buttons": [
                    {
                        "label": "▶ Запуск",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 90, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸ Пауза",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Кадр: "},
                "pad": {"t": 55},
                "steps": slider_steps,
            }
        ],
    )
    _apply_layout(
        figure,
        f"Прохождение: {obstacle_name}, {direction_name}",
        ranges,
    )
    return figure
