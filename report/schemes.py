"""Схемы для отчёта: параметризация борта и сравнение геометрий → report/fig."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from math import atan2, cos, degrees, radians, sin
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Circle  # noqa: E402

from compute import PLATFORM, start_design  # noqa: E402
from model import Design, Platform, neutral_points  # noqa: E402
from plots import FIG, MUTED, TEXT  # noqa: E402

ROCKER = "#2a78d6"
BOGIE = "#eb6834"
STRUT = "#6b6a66"
GROUND = "#b8b4a8"


def _under_arc(pivot, rear, front) -> tuple[float, float]:
    """Дуга угла между плечами, проходящая под шарниром: (theta1, theta2) против часовой."""
    a_rear = degrees(atan2(rear[1] - pivot[1], rear[0] - pivot[0])) % 360.0
    a_front = degrees(atan2(front[1] - pivot[1], front[0] - pivot[0])) % 360.0
    span = (a_rear - a_front) % 360.0
    if (270.0 - a_front) % 360.0 < span:
        return a_front, a_front + span
    return a_rear, a_rear + (a_front - a_rear) % 360.0


def draw_side(ax, design: Design, platform: Platform, annotate: bool = False, lw: float = 3.0) -> None:
    points = neutral_points(design, platform)
    radius = platform.wheel_radius_mm
    wheels, tops = points.wheels, points.strut_tops
    main, bogie = points.main_pivot, points.bogie_pivot
    ax.axhline(0.0, color=GROUND, linewidth=1.5, zorder=0)
    for name in ("rear", "middle", "front"):
        ax.add_patch(Circle(wheels[name], radius, facecolor="#e9e8e4", edgecolor="#3a3a38", linewidth=1.2,
                            zorder=1))
        ax.plot(*zip(wheels[name], tops[name]), color=STRUT, linewidth=lw - 0.8, solid_capstyle="round", zorder=2)
        ax.plot(*wheels[name], "o", color="#3a3a38", markersize=3, zorder=3)
    ax.plot(*zip(tops["rear"], bogie, tops["middle"]), color=BOGIE, linewidth=lw, solid_capstyle="round",
            zorder=3)
    ax.plot(*zip(bogie, main, tops["front"]), color=ROCKER, linewidth=lw, solid_capstyle="round", zorder=3)
    for pivot in (main, bogie):
        ax.plot(*pivot, "o", markersize=8, markerfacecolor="white", markeredgecolor=TEXT, markeredgewidth=1.5,
                zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    if not annotate:
        return

    def label(p, q, text, color, offset):
        mid = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
        ax.annotate(text, mid, xytext=offset, textcoords="offset points", ha="center", va="center",
                    fontsize=10, color=color)

    label(tops["rear"], bogie, "$l_1$", BOGIE, (-8, 10))
    label(bogie, tops["middle"], "$l_2$", BOGIE, (6, 10))
    label(bogie, main, "$l_3$", ROCKER, (0, 12))
    label(main, tops["front"], "$l_4$", ROCKER, (8, 10))
    for pivot, rear, front, color, text in ((bogie, tops["rear"], tops["middle"], BOGIE, "α"),
                                            (main, bogie, tops["front"], ROCKER, "β")):
        theta1, theta2 = _under_arc(pivot, rear, front)
        ax.add_patch(Arc(pivot, 110, 110, theta1=theta1, theta2=theta2, color=color, linewidth=1.2, zorder=5))
        mid = radians((theta1 + theta2) / 2.0)
        ax.annotate(text, (pivot[0] + 80 * cos(mid), pivot[1] + 80 * sin(mid)), ha="center", va="center",
                    fontsize=12, color=color)
    # Наклоны стоек: от вертикали через центр колеса.
    for name, text in (("front", "γ₁"), ("rear", "γ₁"), ("middle", "γ₂")):
        center, top = wheels[name], tops[name]
        height = top[1] - center[1]
        ax.plot([center[0], center[0]], [center[1], center[1] + height], color=MUTED, linewidth=0.8,
                linestyle="--", zorder=2)
        tilt = degrees(atan2(top[1] - center[1], top[0] - center[0]))
        theta1, theta2 = sorted((90.0, tilt))
        ax.add_patch(Arc(center, 2 * 0.75 * height, 2 * 0.75 * height, theta1=theta1, theta2=theta2,
                         color=MUTED, linewidth=1.0, zorder=5))
        ax.annotate(text, (center[0], center[1] + 0.8 * height), xytext=(10 if tilt < 90 else -10, 0),
                    textcoords="offset points", ha="center", fontsize=10, color=TEXT)
    for name, text, offset, align in (("front", "$h_1$", (-12, 0), "right"), ("rear", "$h_1$", (12, 0), "left"),
                                      ("middle", "$h_2$", (12, 0), "left")):
        center, top = wheels[name], tops[name]
        ax.annotate(text, ((center[0] + top[0]) / 2.0, (center[1] + top[1]) / 2.0), xytext=offset,
                    textcoords="offset points", ha=align, va="center", fontsize=10, color=TEXT)
    ax.annotate(f"R = {radius:.0f} мм", (wheels["rear"][0], 0.0), xytext=(0, -12), textcoords="offset points",
                ha="center", fontsize=8, color=MUTED)


def shape_inset(ax, angle_deg: float, title: str, color: str) -> None:
    """Рычаг с углом между плечами под шарниром: Λ при < 180°, V при > 180°."""
    half = radians(angle_deg / 2.0)
    pivot = (0.0, 0.0)
    rear = (-sin(half), -cos(half))
    front = (sin(half), -cos(half))
    ax.plot(*zip(rear, pivot, front), color=color, linewidth=3, solid_capstyle="round")
    ax.plot(*pivot, "o", markersize=8, markerfacecolor="white", markeredgecolor=TEXT, markeredgewidth=1.5)
    theta1, theta2 = _under_arc(pivot, rear, front)
    ax.add_patch(Arc(pivot, 0.7, 0.7, theta1=theta1, theta2=theta2, color=color, linewidth=1.2))
    ax.set_title(title, fontsize=9, color=TEXT)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")


def params_figure() -> None:
    fig = plt.figure(figsize=(9.6, 3.6))
    main_ax = fig.add_axes([0.0, 0.0, 0.74, 1.0])
    # Иллюстрация (не старт поиска): наглядная Λ-геометрия, стойки наклонены, чтобы углы были различимы.
    draw_side(main_ax, replace(start_design(), outer_strut_angle_deg=10.0, middle_strut_angle_deg=10.0),
              PLATFORM, annotate=True)
    lambda_ax = fig.add_axes([0.76, 0.52, 0.22, 0.42])
    shape_inset(lambda_ax, 130.0, "угол < 180°: форма Λ", BOGIE)
    v_ax = fig.add_axes([0.76, 0.04, 0.22, 0.42])
    shape_inset(v_ax, 230.0, "угол > 180°: форма V", BOGIE)
    fig.savefig(FIG / "fig_params.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def compare_figure(designs: list[tuple[str, Design]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(designs), figsize=(3.3 * len(designs), 2.2), squeeze=False)
    axes = axes[0]
    lows, highs = [], []
    for ax, (title, design) in zip(axes, designs):
        draw_side(ax, design, PLATFORM, lw=2.4)
        ax.set_title(title, fontsize=9, color=TEXT)
        lows.append(ax.get_ylim()[0])
        highs.append(ax.get_ylim()[1])
    for ax in axes:
        ax.set_xlim(-150, 1150)
        ax.set_ylim(-20, 620)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


DATA = ROOT / "report" / "data2"


def load(name: str) -> dict | None:
    path = DATA / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def lhs_best() -> Design:
    return Design(**load("lhs")["rows"][0]["design"])


def final_run() -> dict | None:
    """Итоговая геометрия — как в experiments.final_design: лучшее из дообучения и запусков из точек ЛГК."""
    runs = [load(name) for name in ("run_final", "run_lhs1_s1", "run_lhs2_s1", "run_lhs3_s1")]
    runs = [data for data in runs if data]
    return max(runs, key=lambda data: data["best_score"]) if runs else None


def designs_figure() -> None:
    lhs = load("lhs")
    designs = [(f"Лучшая точка ЛГК (J = {lhs['rows'][0]['score']:.0f})", lhs_best())]
    found = final_run()
    if found:
        designs.append((f"Итоговая геометрия (J = {found['best_score']:.0f})", Design(**found["best"])))
    compare_figure(designs, FIG / "fig_designs.png")


def weights_figure() -> None:
    """Лучший из трёх seed для каждого набора весов."""
    titles = {"base": "базовые веса", "step": "упор на ступень", "bumps": "упор на кочки", "equal": "равные веса"}
    designs = []
    for weights, title in titles.items():
        runs = [load("run_lhs1_s1" if weights == "base" and seed == 1 else f"run_w_{weights}_s{seed}")
                for seed in (1, 2, 3)]
        runs = [data for data in runs if data]
        if runs:
            designs.append((title, Design(**max(runs, key=lambda data: data["best_score"])["best"])))
    if len(designs) > 1:
        compare_figure(designs, FIG / "fig_weights_designs.png")


if __name__ == "__main__":
    params_figure()
    designs_figure()
    weights_figure()
