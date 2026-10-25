"""Схема шасси, покадровая анимация прохождения и графики по времени."""

from __future__ import annotations

import os
import tempfile
from math import cos, pi, sin

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model import WHEELS, WHEEL_LABELS, Box, Design, Platform, neutral_points
from simulate import Frame, SimResult

ROCKER_COLOR = "#2a78d6"
BOGIE_COLOR = "#eb6834"
STRUT_COLOR = "#52514e"
WHEEL_LINE = "#3a3a38"
GROUND_FILL = "rgba(111, 139, 90, 0.45)"
GROUND_LINE = "#5f7a4c"
FORCE_COLOR = "#e34948"
COM_COLOR = "#6250d6"
WHEEL_SERIES = {"front": "#2a78d6", "middle": "#eb6834", "rear": "#1baf7a"}
SIDE_DASH = {"L": "solid", "R": "dash"}
SIDE_NAME = {"L": "левый", "R": "правый"}

FORCE_SCALE_MM_PER_N = 0.35


def _circle(cx: float, cz: float, radius: float, count: int = 40) -> tuple[list[float], list[float]]:
    xs = [cx + radius * cos(2 * pi * i / count) for i in range(count + 1)]
    zs = [cz + radius * sin(2 * pi * i / count) for i in range(count + 1)]
    return xs, zs


def _segments(pairs: list[tuple[tuple[float, float], tuple[float, float]]]) -> tuple[list, list]:
    xs: list = []
    zs: list = []
    for start, end in pairs:
        xs += [start[0], end[0], None]
        zs += [start[1], end[1], None]
    return xs, zs


def _wheels_xy(centers: list[tuple[float, float]], radius: float) -> tuple[list, list]:
    xs: list = []
    zs: list = []
    for cx, cz in centers:
        cxs, czs = _circle(cx, cz, radius)
        xs += cxs + [None]
        zs += czs + [None]
    return xs, zs


def _side_traces(
    points: dict[str, tuple[float, float]],
    platform: Platform,
    opacity: float = 1.0,
    show_legend: bool = True,
) -> list[go.Scatter]:
    """Трассы одного борта; points — плоские (x, z) координаты в мм."""
    radius = platform.wheel_radius_mm
    tube_width = 6 if opacity == 1.0 else 3
    rocker = _segments([(points["main"], points["top_front"]), (points["main"], points["bogie"])])
    bogie = _segments([(points["bogie"], points["top_middle"]), (points["bogie"], points["top_rear"])])
    struts = _segments([(points[f"top_{w}"], points[f"wheel_{w}"]) for w in WHEELS])
    wheels = _wheels_xy([points[f"wheel_{w}"] for w in WHEELS], radius)
    pivots = [points["main"], points["bogie"]]
    return [
        go.Scatter(x=wheels[0], y=wheels[1], mode="lines", line=dict(color=WHEEL_LINE, width=2),
                   fill="toself", fillcolor="rgba(60,60,58,0.18)", name="колёса", opacity=opacity,
                   showlegend=show_legend, hoverinfo="skip"),
        go.Scatter(x=struts[0], y=struts[1], mode="lines", line=dict(color=STRUT_COLOR, width=tube_width - 1),
                   name="стойки (вилки)", opacity=opacity, showlegend=show_legend, hoverinfo="skip"),
        go.Scatter(x=rocker[0], y=rocker[1], mode="lines", line=dict(color=ROCKER_COLOR, width=tube_width),
                   name="rocker", opacity=opacity, showlegend=show_legend, hoverinfo="skip"),
        go.Scatter(x=bogie[0], y=bogie[1], mode="lines", line=dict(color=BOGIE_COLOR, width=tube_width),
                   name="bogie", opacity=opacity, showlegend=show_legend, hoverinfo="skip"),
        go.Scatter(x=[p[0] for p in pivots], y=[p[1] for p in pivots], mode="markers",
                   marker=dict(size=10, color="white", line=dict(color="#0b0b0b", width=2)),
                   name="шарниры", opacity=opacity, showlegend=show_legend,
                   hovertemplate="x=%{x:.0f} мм<br>z=%{y:.0f} мм<extra>шарнир</extra>"),
    ]


def _layout(fig: go.Figure, x_range, z_range, height: int) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(range=list(x_range), title="X, мм", zeroline=False, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(range=list(z_range), title="Z, мм", scaleanchor="x", scaleratio=1, zeroline=False,
                     gridcolor="rgba(128,128,128,0.15)")


def design_figure(design: Design, platform: Platform) -> go.Figure:
    points = neutral_points(design, platform)
    flat = {
        "main": points.main_pivot,
        "bogie": points.bogie_pivot,
        **{f"top_{w}": points.strut_tops[w] for w in WHEELS},
        **{f"wheel_{w}": points.wheels[w] for w in WHEELS},
    }
    fig = go.Figure(_side_traces(flat, platform))
    radius = platform.wheel_radius_mm
    left = min(x for x, _ in points.wheels.values()) - radius
    fig.add_shape(type="line", x0=left - 200, x1=left + platform.max_length_mm + 200, y0=0, y1=0,
                  line=dict(color=GROUND_LINE, width=2))
    fig.add_shape(type="rect", x0=left, x1=left + platform.max_length_mm, y0=-15, y1=0,
                  line=dict(color="rgba(0,0,0,0)"), fillcolor="rgba(128,128,128,0.25)")
    fig.add_annotation(x=left + platform.max_length_mm / 2, y=-35, showarrow=False,
                       text=f"габарит {platform.max_length_mm:.0f} мм")
    fig.add_trace(go.Scatter(x=[points.com[0]], y=[points.com[1]], mode="markers",
                             marker=dict(size=12, color=COM_COLOR, symbol="x"), name="центр масс корпуса",
                             hovertemplate="x=%{x:.0f} мм<br>z=%{y:.0f} мм<extra>центр масс</extra>"))
    # Подписи параметров рычагов: длины плеч у середин труб, углы под шарнирами.
    levers = (
        (points.bogie_pivot, points.strut_tops["rear"], points.strut_tops["middle"],
         design.bogie_rear_arm_mm, design.bogie_front_arm_mm, design.bogie_angle_deg),
        (points.main_pivot, points.bogie_pivot, points.strut_tops["front"],
         design.rocker_rear_arm_mm, design.rocker_front_arm_mm, design.rocker_angle_deg),
    )
    for pivot, rear, front, rear_arm, front_arm, angle in levers:
        for end, length in ((rear, rear_arm), (front, front_arm)):
            fig.add_annotation(x=(pivot[0] + end[0]) / 2, y=(pivot[1] + end[1]) / 2 + 22, showarrow=False,
                               text=f"{length:.0f}", font=dict(size=11))
        fig.add_annotation(x=pivot[0], y=pivot[1] - 32, showarrow=False, text=f"{angle:.0f}°",
                           font=dict(size=11, color=COM_COLOR))
    top = max(points.main_pivot[1], points.bogie_pivot[1], points.com[1]) + 80
    _layout(fig, (left - 100, left + platform.max_length_mm + 100), (-80, top), 420)
    return fig


def _terrain_polygons(boxes: list[Box], floor_z: float, y_mm: float, x_range) -> list[go.Scatter]:
    traces = []
    for box in boxes:
        if not box.y0 <= y_mm <= box.y1:
            continue
        x0 = max(box.x0, x_range[0] - 50)
        x1 = min(box.x1, x_range[1] + 50)
        bottom = min(floor_z, box.z_top) - 400
        traces.append(go.Scatter(x=[x0, x1, x1, x0, x0], y=[bottom, bottom, box.z_top, box.z_top, bottom],
                                 mode="lines", fill="toself", fillcolor=GROUND_FILL,
                                 line=dict(color=GROUND_LINE, width=1.5), showlegend=False, hoverinfo="skip"))
    left, right = x_range[0] - 50, x_range[1] + 50
    traces.append(go.Scatter(x=[left, right, right, left, left],
                             y=[floor_z, floor_z, floor_z - 400, floor_z - 400, floor_z],
                             mode="lines", fill="toself", fillcolor=GROUND_FILL,
                             line=dict(color=GROUND_LINE, width=1.5), showlegend=False, hoverinfo="skip"))
    return traces


def _flat(frame: Frame, side: str) -> dict[str, tuple[float, float]]:
    return {name: (p[0], p[2]) for name, p in frame.points[side].items()}


def _force_trace(frame: Frame, side: str) -> go.Scatter:
    xs: list = []
    zs: list = []
    for key, position, force in frame.forces:
        if not key.endswith(side):
            continue
        xs += [position[0], position[0] + force[0] * FORCE_SCALE_MM_PER_N, None]
        zs += [position[2], position[2] + force[2] * FORCE_SCALE_MM_PER_N, None]
    return go.Scatter(x=xs, y=zs, mode="lines", line=dict(color=FORCE_COLOR, width=3),
                      name="реакции опоры", hoverinfo="skip")


def _frame_traces(frame: Frame, platform: Platform, side: str, ghost: str | None) -> list[go.Scatter]:
    traces = _side_traces(_flat(frame, side), platform)
    if ghost:
        traces += _side_traces(_flat(frame, ghost), platform, opacity=0.3, show_legend=False)
    traces.append(_force_trace(frame, side))
    traces.append(go.Scatter(x=[frame.com[0]], y=[frame.com[2]], mode="markers",
                             marker=dict(size=12, color=COM_COLOR, symbol="x"), name="центр масс",
                             hoverinfo="skip"))
    return traces


def _frame_title(frame: Frame) -> str:
    return (f"t = {frame.t:.2f} с · тангаж {frame.pitch_deg:+.1f}° · крен {frame.roll_deg:+.1f}° · "
            f"bogie {frame.bogie_deg['L']:+.1f}°/{frame.bogie_deg['R']:+.1f}°")


def sample_frames(result: SimResult, limit: int = 160) -> list[Frame]:
    frames = result.frames
    if len(frames) <= limit:
        return frames
    step = len(frames) / limit
    picked = [frames[int(i * step)] for i in range(limit)]
    if picked[-1] is not frames[-1]:
        picked.append(frames[-1])
    return picked


def _motion_range(result: SimResult, platform: Platform) -> tuple[tuple[float, float], tuple[float, float]]:
    xs, zs = [], []
    for frame in result.frames:
        for side_points in frame.points.values():
            for p in side_points.values():
                xs.append(p[0])
                zs.append(p[2])
    radius = platform.wheel_radius_mm
    x_range = (min(xs) - radius - 80, max(xs) + radius + 80)
    z_low = min(min(zs) - radius, result.floor_z_mm, 0.0) - 60
    z_high = max(max(zs), max(frame.com[2] for frame in result.frames)) + 120
    return x_range, (z_low, z_high)


def animation_figure(result: SimResult, platform: Platform, side: str = "L", show_other_side: bool = False) -> go.Figure:
    frames = sample_frames(result)
    x_range, z_range = _motion_range(result, platform)
    y_mm = platform.track_mm / 2.0 * (1 if side == "L" else -1)
    terrain = _terrain_polygons(result.terrain, result.floor_z_mm, y_mm, x_range)
    ghost = ("R" if side == "L" else "L") if show_other_side else None
    first = _frame_traces(frames[0], platform, side, ghost)
    fig = go.Figure(terrain + first)
    dynamic = list(range(len(terrain), len(terrain) + len(first)))
    fig.frames = [
        go.Frame(data=_frame_traces(frame, platform, side, ghost), traces=dynamic, name=f"{frame.t:.2f}",
                 layout=go.Layout(title_text=_frame_title(frame)))
        for frame in frames
    ]
    play = dict(frame=dict(duration=50, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")
    fig.update_layout(
        title=dict(text=_frame_title(frames[0]), font=dict(size=14)),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.08, xanchor="left", yanchor="top",
                          showactive=False,
                          buttons=[dict(label="▶ Пуск", method="animate", args=[None, play]),
                                   dict(label="❚❚ Пауза", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(x=0.18, len=0.82, y=-0.04, yanchor="top", currentvalue=dict(prefix="t = ", suffix=" с"),
                      steps=[dict(method="animate", label=f"{frame.t:.1f}",
                                  args=[[f"{frame.t:.2f}"], dict(frame=dict(duration=0, redraw=True),
                                                                   mode="immediate", transition=dict(duration=0))])
                             for frame in frames])],
    )
    _layout(fig, x_range, z_range, 520)
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=90))
    return fig


def timeseries_figure(result: SimResult, platform: Platform) -> go.Figure:
    frames = result.frames
    t = [frame.t for frame in frames]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Нормальная реакция колеса, Н", "Момент мотор-колеса, Н·м",
                                        "Тангаж и крен корпуса, °", "Углы bogie, °"))
    for wheel in WHEELS:
        for side in ("L", "R"):
            key = f"{wheel}_{side}"
            label = f"{WHEEL_LABELS[wheel]}, {SIDE_NAME[side]}"
            style = dict(color=WHEEL_SERIES[wheel], width=2, dash=SIDE_DASH[side])
            fig.add_trace(go.Scatter(x=t, y=[f.normal_n[key] for f in frames], name=label, legendgroup=key,
                                     line=style, hovertemplate="%{y:.0f} Н"), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=[f.torque_nm[key] for f in frames], name=label, legendgroup=key,
                                     showlegend=False, line=style, hovertemplate="%{y:.1f} Н·м"), row=2, col=1)
    for sign in (1, -1):
        fig.add_hline(y=sign * platform.wheel_torque_nm, row=2, col=1, line=dict(color="#898781", dash="dot", width=1))
    fig.add_trace(go.Scatter(x=t, y=[f.pitch_deg for f in frames], name="тангаж",
                             line=dict(color="#6250d6", width=2), hovertemplate="%{y:.1f}°"), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=[f.roll_deg for f in frames], name="крен",
                             line=dict(color="#e87ba4", width=2), hovertemplate="%{y:.1f}°"), row=3, col=1)
    for side in ("L", "R"):
        fig.add_trace(go.Scatter(x=t, y=[f.bogie_deg[side] for f in frames], name=f"bogie, {SIDE_NAME[side]}",
                                 line=dict(color=BOGIE_COLOR, width=2, dash=SIDE_DASH[side]),
                                 hovertemplate="%{y:.1f}°"), row=4, col=1)
    fig.update_layout(height=820, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0))
    fig.update_xaxes(title="время, с", row=4, col=1)
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


def animation_gif(result: SimResult, platform: Platform, side: str = "L", limit: int = 120) -> bytes:
    """GIF вида сбоку для отчёта (matplotlib + Pillow)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = sample_frames(result, limit)
    x_range, z_range = _motion_range(result, platform)
    y_mm = platform.track_mm / 2.0 * (1 if side == "L" else -1)
    radius = platform.wheel_radius_mm
    width = 9.0
    height = max(3.0, width * (z_range[1] - z_range[0]) / (x_range[1] - x_range[0]))
    fig, ax = plt.subplots(figsize=(width, height), dpi=90)
    ax.set_xlim(*x_range)
    ax.set_ylim(*z_range)
    ax.set_aspect("equal")
    ax.set_xlabel("X, мм")
    ax.set_ylabel("Z, мм")
    for box in result.terrain:
        if box.y0 <= y_mm <= box.y1:
            bottom = min(result.floor_z_mm, box.z_top) - 400
            ax.fill([box.x0, box.x1, box.x1, box.x0], [bottom, bottom, box.z_top, box.z_top],
                    color="#8aa57a", alpha=0.6, lw=0)
    ax.fill([x_range[0], x_range[1], x_range[1], x_range[0]],
            [result.floor_z_mm, result.floor_z_mm, result.floor_z_mm - 400, result.floor_z_mm - 400],
            color="#8aa57a", alpha=0.6, lw=0)
    ax.axhline(result.floor_z_mm, color=GROUND_LINE, lw=1.5)
    rocker_line, = ax.plot([], [], color=ROCKER_COLOR, lw=4)
    bogie_line, = ax.plot([], [], color=BOGIE_COLOR, lw=4)
    strut_line, = ax.plot([], [], color=STRUT_COLOR, lw=3)
    wheel_line, = ax.plot([], [], color=WHEEL_LINE, lw=1.5)
    force_line, = ax.plot([], [], color=FORCE_COLOR, lw=2.5)
    com_point, = ax.plot([], [], "x", color=COM_COLOR, ms=9, mew=2)
    title = ax.set_title("")

    def draw(index: int):
        frame = frames[index]
        p = _flat(frame, side)
        xs, zs = _segments([(p["main"], p["top_front"]), (p["main"], p["bogie"])])
        rocker_line.set_data(xs, zs)
        xs, zs = _segments([(p["bogie"], p["top_middle"]), (p["bogie"], p["top_rear"])])
        bogie_line.set_data(xs, zs)
        xs, zs = _segments([(p[f"top_{w}"], p[f"wheel_{w}"]) for w in WHEELS])
        strut_line.set_data(xs, zs)
        xs, zs = _wheels_xy([p[f"wheel_{w}"] for w in WHEELS], radius)
        wheel_line.set_data(xs, zs)
        force = _force_trace(frame, side)
        force_line.set_data(list(force.x or []), list(force.y or []))
        com_point.set_data([frame.com[0]], [frame.com[2]])
        title.set_text(f"t = {frame.t:.2f} с, тангаж {frame.pitch_deg:+.1f}°")
        return rocker_line, bogie_line, strut_line, wheel_line, force_line, com_point, title

    fig.tight_layout()
    animation = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "animation.gif")
        animation.save(path, writer=PillowWriter(fps=15))
        with open(path, "rb") as handle:
            data = handle.read()
    plt.close(fig)
    return data
