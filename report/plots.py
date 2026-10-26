"""Графики для отчёта из report/data/*.json → report/fig/*.png."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "fig"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load(name: str) -> dict | None:
    path = DATA / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def worst(rows: list[dict], *keys: str) -> dict[tuple, float]:
    """Худшее из двух направлений движения для каждого сочетания ключей."""
    found: dict[tuple, float] = defaultdict(lambda: float("inf"))
    for row in rows:
        key = tuple(row[name] for name in keys)
        found[key] = min(found[key], row["step_mm"])
    return dict(found)


def label_end(ax, x, y, text, color) -> None:
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=TEXT, fontsize=9)
    ax.plot([x], [y], "o", color=color, markeredgecolor="white", markeredgewidth=1.5)


def plot_torque() -> None:
    data = load("torque")
    if not data:
        return
    limits = worst(data["rows"], "design", "torque_nm")
    labels = {"start": "Начальная (ручная) геометрия", "optimized": "После оптимизации"}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for color, (design, label) in zip(SERIES, labels.items()):
        points = sorted((t, v) for (d, t), v in limits.items() if d == design)
        x, y = zip(*points)
        ax.plot(x, y, color=color, label=label, marker="o", markeredgecolor="white", markeredgewidth=1.2)
    ax.axvline(16.0, color=MUTED, linewidth=1.0, linestyle="--")
    ax.annotate("пиковый момент\nM365 Pro ≈ 16 Н·м", (16.0, 130), xytext=(4, 0),
                textcoords="offset points", va="bottom", fontsize=8, color=MUTED)
    ax.set_xlabel("Пиковый момент мотор-колеса, Н·м")
    ax.set_ylabel("Наибольшая ступень, мм")
    ax.set_ylim(0, 560)
    ax.legend(loc="center right")
    fig.savefig(FIG / "fig_torque.png")
    plt.close(fig)


def plot_radius() -> None:
    data = load("radius")
    if not data:
        return
    limits = worst(data["rows"], "torque_nm", "radius_mm")
    torques = sorted({t for t, _ in limits})
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for color, torque in zip(SERIES, torques):
        points = sorted((r, v) for (t, r), v in limits.items() if t == torque)
        x, y = zip(*points)
        ax.plot(x, y, color=color, label=f"{torque:.0f} Н·м", marker="o", markeredgecolor="white",
                markeredgewidth=1.2)
    radii = np.array(sorted({r for _, r in limits}))
    ax.plot(radii, 2.0 * radii, color=MUTED, linewidth=1.0, linestyle=":")
    ax.annotate("h = 2R", (radii[-1], 2.0 * radii[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=MUTED)
    ax.axvline(108.0, color=MUTED, linewidth=1.0, linestyle="--")
    ax.annotate("M365 Pro", (108.0, 0), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel("Радиус колеса R, мм (рычаги масштабированы ∝ R)")
    ax.set_ylabel("Наибольшая ступень, мм")
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=radii[-1] + 20)
    ax.legend(title="Момент колеса", loc="upper right")
    fig.savefig(FIG / "fig_radius.png")
    plt.close(fig)


def plot_geometry() -> None:
    data = load("geometry")
    if not data:
        return
    meta = data["meta"]
    limits = worst(data["rows"], "bogie_angle_deg", "rocker_angle_deg")
    bogie, rocker = meta["bogie_angles"], meta["rocker_angles"]
    grid = np.full((len(rocker), len(bogie)), np.nan)
    for (b, r), value in limits.items():
        grid[rocker.index(r), bogie.index(b)] = value
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.grid(False)
    step_b, step_r = bogie[1] - bogie[0], rocker[1] - rocker[0]
    extent = [bogie[0] - step_b / 2, bogie[-1] + step_b / 2, rocker[0] - step_r / 2, rocker[-1] + step_r / 2]
    ax.set_facecolor("#f0efec")
    image = ax.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap=BLUES, vmin=0)
    for item in meta["invalid"]:
        ax.add_patch(plt.Rectangle((item["bogie_angle_deg"] - step_b / 2, item["rocker_angle_deg"] - step_r / 2),
                                   step_b, step_r, fill=False, hatch="///", edgecolor="#c3c2b7", linewidth=0))
    for i, r in enumerate(rocker):
        for j, b in enumerate(bogie):
            if np.isfinite(grid[i, j]):
                value = grid[i, j]
                color = "white" if value > 0.55 * np.nanmax(grid) else TEXT
                ax.text(b, r, f"{value:.0f}", ha="center", va="center", fontsize=6.5, color=color)
    base = meta["base"]
    ax.plot(base["bogie_angle_deg"], base["rocker_angle_deg"], marker="o", markersize=14, fillstyle="none",
            color=SERIES[1], markeredgewidth=2)
    ax.set_xlabel("Угол bogie между плечами, ° (<180 — Λ, >180 — V)")
    ax.set_ylabel("Угол rocker между плечами, °")
    bar = fig.colorbar(image, ax=ax, pad=0.02)
    bar.set_label("Наибольшая ступень, мм")
    bar.outline.set_visible(False)
    fig.savefig(FIG / "fig_geometry.png")
    plt.close(fig)


def plot_cmaes() -> None:
    panels = [(mode, load(f"cmaes_{mode}")) for mode in ("limits",)]
    panels = [(mode, data) for mode, data in panels if data]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 if len(panels) == 1 else 9.6, 3.4), squeeze=False)
    titles = {"torque": ("", "Требуемый момент, Н·м"), "limits": ("", "Оценка J = Σ wᵢ·hᵢ, мм")}
    for ax, (mode, data) in zip(axes[0], panels):
        history = data["history"]
        gen = [h["index"] for h in history]
        best = [h["best_score"] for h in history]
        median = [h["median_score"] for h in history]
        # Отдельные варианты поколения — бледными точками.
        per_gen = len(data["evaluations"]) // max(len(history), 1)
        for g in range(len(history)):
            scores = [e["score"] for e in data["evaluations"][g * per_gen:(g + 1) * per_gen]
                      if e["valid"] and np.isfinite(e["score"])]
            ax.scatter([g + 1] * len(scores), scores, s=10, color=SERIES[0], alpha=0.25, linewidths=0)
        ax.plot(gen, median, color=SERIES[0], linewidth=1.2, linestyle="--", label="медиана поколения")
        ax.plot(gen, best, color=SERIES[1], label="лучший найденный")
        compare = load("compare")
        if mode == "limits" and compare:
            weights = {c["kind"] if c["kind"] != "bump" else f"bump_{c['width_mm']:.0f}": c["weight"]
                       for c in data["goal"]}
            for label, key in (("начальная геометрия", "start"), ("вариант (3)", "user")):
                value = sum(weights[k] * v for k, v in compare[key]["limits"].items())
                ax.axhline(value, color=MUTED, linewidth=1.0, linestyle=":")
                ax.annotate(f"{label}: {value:.0f}", (gen[-1], value), xytext=(0, 3), textcoords="offset points",
                            ha="right", va="bottom", fontsize=8, color=MUTED)
        ax.set_title(titles[mode][0], fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("Поколение CMA-ES")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_ylabel(titles[mode][1])
        ax.legend(loc="best", fontsize=8)
        if mode == "torque":
            top = np.nanpercentile([e["score"] for e in data["evaluations"]
                                    if e["valid"] and np.isfinite(e["score"])], 95)
            ax.set_ylim(0, top * 1.1)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cmaes.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for plot in (plot_torque, plot_radius, plot_geometry, plot_cmaes):
        plot()
