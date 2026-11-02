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


def spread(rows: list[dict], *keys: str) -> dict[tuple, tuple[float, float, float]]:
    """Худшее направление при каждой скорости подъезда, затем медиана, min и max по скоростям."""
    per_speed = worst(rows, *keys, "speed")
    groups: dict[tuple, list[float]] = defaultdict(list)
    for key, value in per_speed.items():
        groups[key[:-1]].append(value)
    return {key: (float(np.median(v)), min(v), max(v)) for key, v in groups.items()}


def line_with_band(ax, x, stats, color, label, linestyle="-") -> None:
    median, low, high = zip(*stats)
    ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
    marker = "o" if linestyle == "-" else None
    ax.plot(x, median, color=color, label=label, linestyle=linestyle, marker=marker, markeredgecolor="white",
            markeredgewidth=1.2)


def plot_torque() -> None:
    data = load("torque")
    if not data:
        return
    stats = spread(data["rows"], "design", "control", "torque_nm")
    series = [("start", "speed", SERIES[0], "Начальная геометрия", "-"),
              ("optimized", "speed", SERIES[1], "Вариант (3)", "-"),
              ("optimized", "torque", SERIES[1], "Вариант (3), одинаковый момент колёс", "--")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for design, control, color, label, style in series:
        torques = sorted(t for d, c, t in stats if d == design and c == control)
        line_with_band(ax, torques, [stats[(design, control, t)] for t in torques], color, label, style)
    ax.axvline(16.0, color=MUTED, linewidth=1.0, linestyle=":")
    ax.annotate("M365 Pro ≈ 16 Н·м", (16.0, 15), xytext=(4, 0), textcoords="offset points", fontsize=8,
                color=MUTED)
    ax.set_xlabel("Пиковый момент мотор-колеса, Н·м")
    ax.set_ylabel("Наибольшая ступень, мм")
    ax.set_ylim(0, 560)
    ax.legend(loc="center right", fontsize=8)
    fig.savefig(FIG / "fig_torque.png")
    plt.close(fig)


def plot_radius() -> None:
    data = load("radius")
    if not data:
        return
    stats = spread(data["rows"], "control", "torque_nm", "radius_mm")
    torques = sorted({t for _, t, _ in stats})
    radii = np.array(sorted({r for _, _, r in stats}))
    titles = {"speed": "а) регулятор скорости у каждого колеса", "torque": "б) одинаковый момент всех колёс"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (control, title) in zip(axes, titles.items()):
        for color, torque in zip(SERIES, torques):
            line_with_band(ax, radii, [stats[(control, torque, r)] for r in radii], color, f"{torque:.0f} Н·м")
        ax.plot(radii, 2.0 * radii, color=MUTED, linewidth=1.0, linestyle=":")
        ax.annotate("h = 2R", (radii[-1], 2.0 * radii[-1]), xytext=(4, -10), textcoords="offset points",
                    fontsize=8, color=MUTED)
        ax.axvline(108.0, color=MUTED, linewidth=1.0, linestyle="--")
        ax.annotate("M365 Pro", (108.0, 0), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("Радиус колеса R, мм (рычаги ∝ R)")
        ax.set_ylim(0, 880)
    axes[0].set_ylabel("Наибольшая ступень, мм")
    axes[1].legend(title="Момент колеса", loc="upper left", fontsize=8)
    fig.tight_layout()
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
    panels = [("limits", load("run_hand_s1") or load("cmaes_limits"))]
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
        table = load("table")
        if mode == "limits" and table:
            weights = {c["kind"] if c["kind"] != "bump" else f"bump_{c['width_mm']:.0f}": c["weight"]
                       for c in data["goal"]}
            for label, key in (("начальная геометрия", "start"), ("вариант (3)", "user")):
                value = table[key]["J_base"]
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


KIND_TITLES = {"step_up": "ступень", "step_down": "съезд", "ditch": "ров (ширина)", "bump_120": "кочка 120 мм",
               "bump_250": "кочка 250 мм"}


def plot_heights() -> None:
    """Полосы «прошёл / не прошёл» по всем размерам препятствий с шагом 10 мм."""
    data = load("heights")
    if not data:
        return
    designs = {"user": "Вариант (3)", "cmaes": "CMA-ES, базовые веса"}
    rows = list(data["rows"])
    extra = load("heights_weights")
    if extra:
        designs["run_w_bumps"] = "CMA-ES, упор на кочки"
        rows += extra["rows"]
    passed = {(r["design"], r["kind"], r["direction"], r["size_mm"]): r["success"] for r in rows}
    fig, axes = plt.subplots(1, len(designs), figsize=(3.6 * len(designs) + 1.2, 3.4), sharey=True)
    for ax, (design, title) in zip(axes, designs.items()):
        ax.grid(False)
        labels = []
        for row, (kind, direction) in enumerate((k, d) for k in KIND_TITLES for d in (1, -1)):
            sizes = sorted(size for (de, ki, di, size) in passed if de == design and ki == kind and di == direction)
            for size in sizes:
                ok = passed[(design, kind, direction, size)]
                ax.add_patch(plt.Rectangle((size - 5, row - 0.38), 10, 0.76, linewidth=0,
                                           color=SERIES[2] if ok else "#d6d4cd"))
            labels.append(f"{KIND_TITLES[kind]}, {'вперёд' if direction == 1 else 'назад'}")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xlim(0, 570)
        ax.set_xlabel("Размер препятствия, мм")
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[2]), plt.Rectangle((0, 0), 1, 1, color="#d6d4cd")]
    axes[-1].legend(handles, ["прошёл", "не прошёл"], loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_heights.png")
    plt.close(fig)


STARTS = [("run_hand_s1", "ручная, seed 1", SERIES[0], "-"), ("run_hand_s2", "ручная, seed 2", SERIES[0], "--"),
          ("run_hand_s3", "ручная, seed 3", SERIES[0], ":"), ("run_user_s1", "вариант (3)", SERIES[1], "-"),
          ("run_random1_s1", "случайная 1", SERIES[2], "-"), ("run_random2_s1", "случайная 2", "#eda100", "-")]


def plot_starts() -> None:
    """Лучшая оценка по поколениям из разных начальных точек и с разными seed."""
    runs = [(load(name), label, color, style) for name, label, color, style in STARTS]
    runs = [run for run in runs if run[0]]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for data, label, color, style in runs:
        history = data["history"]
        gen = [0] + [h["index"] for h in history]
        start = data["evaluations"][0]["score"] if data["evaluations"][0]["valid"] else float("nan")
        best = [start] + [h["best_score"] for h in history]
        ax.plot(gen, best, color=color, linestyle=style, label=f"{label}: {best[-1]:.0f}")
        ax.plot([0], [start], "o", color=color, markeredgecolor="white", markeredgewidth=1.2)
    ax.set_xlabel("Поколение CMA-ES (0 — начальная точка)")
    ax.set_ylabel("Лучшая оценка J, мм")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(title="Старт: лучшая J", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.savefig(FIG / "fig_starts.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for plot in (plot_torque, plot_radius, plot_geometry, plot_cmaes, plot_heights, plot_starts):
        plot()
