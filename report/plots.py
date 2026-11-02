"""Графики для отчёта из report/data2/*.json → report/fig/*.png (графики без данных пропускаются)."""

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
DATA = HERE / "data2"
FIG = HERE / "fig"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
STYLES = ["-", "--", "-.", ":"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
FAIL = "#d6d4cd"
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

# Подписи наборов весов (ступень / съезд / ров / кочки).
WEIGHT_LABELS = {"base": "базовые 1/0,3/0,5/0,5", "step": "упор на ступень", "bumps": "упор на кочки",
                 "equal": "равные"}
KIND_TITLES = {"step_up": "ступень", "step_down": "съезд", "ditch": "ров (ширина)", "bump_120": "кочка 120 мм",
               "bump_250": "кочка 250 мм"}
# Эталонные геометрии первой серии в развёртке по массе.
REFERENCE_LABELS = {"variant3": "геометрия A", "run_hand_s2": "геометрия B", "run_random1_s1": "геометрия C",
                    "run_w_equal": "геометрия D"}
WHEELS_KG = 6 * 4.0
BUMP_CEILING_MM = 300.0  # UPPER["bump_250"] в compute.py


def load(name: str) -> dict | None:
    path = DATA / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def spread(rows: list[dict], *keys: str) -> dict[tuple, tuple[float, float, float]]:
    """Худшее направление при каждой скорости подъезда, затем медиана, min и max по скоростям."""
    per_speed: dict[tuple, float] = defaultdict(lambda: float("inf"))
    for row in rows:
        key = tuple(row[name] for name in keys) + (row["speed"],)
        per_speed[key] = min(per_speed[key], row["value_mm"])
    groups: dict[tuple, list[float]] = defaultdict(list)
    for key, value in per_speed.items():
        groups[key[:-1]].append(value)
    return {key: (float(np.median(v)), min(v), max(v)) for key, v in groups.items()}


def line_with_band(ax, x, stats, index: int, label: str, linestyle: str = "-") -> None:
    median, low, high = zip(*stats)
    color = SERIES[index]
    ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, median, color=color, label=label, linestyle=linestyle, marker=MARKERS[index],
            markeredgecolor="white", markeredgewidth=1.2)


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def plot_mass() -> None:
    data = load("mass")
    choice = load("mass_choice")
    if not data or not choice:
        return
    stats = spread(data["rows"], "design", "mass_kg")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for index, (design, label) in enumerate(REFERENCE_LABELS.items()):
        masses = sorted(m for d, m in stats if d == design)
        if masses:
            line_with_band(ax, masses, [stats[(design, m)] for m in masses], index, label)
    chosen = choice["body_mass_kg"]
    ax.axvline(chosen, color=MUTED, linewidth=1.0, linestyle="--")
    ax.annotate(f"выбрано {chosen:.0f} кг\n(всего {chosen + WHEELS_KG:.0f} кг)", (chosen, 560), xytext=(5, 0),
                textcoords="offset points", fontsize=8, color=MUTED, va="top")
    ax.set_xlabel("Подрессоренная масса (без мотор-колёс), кг")
    ax.set_ylabel("Наибольшая ступень, мм")
    ax.set_ylim(0, 580)
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "fig_mass")


STARTS = (("run_lhs1_s1", "из точки ЛГК 1"), ("run_lhs2_s1", "из точки ЛГК 2"), ("run_lhs3_s1", "из точки ЛГК 3"),
          ("run_final", "повторный запуск из лучшего"))


def plot_search() -> None:
    """Слева — оценки точек ЛГК, справа — лучшая J по поколениям мультистарта CMA-ES."""
    lhs = load("lhs")
    if not lhs:
        return
    runs = [(load(name), label) for name, label in STARTS]
    runs = [(data, label, index) for index, (data, label) in enumerate(runs) if data]
    scores = np.array([row["score"] for row in lhs["rows"]])
    median = float(np.median(scores))
    fig, axes = plt.subplots(1, 2 if runs else 1, figsize=(9.6 if runs else 5.0, 3.4), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.4]} if runs else None)
    ax = axes[0][0]
    ax.hist(scores, bins=np.arange(0, scores.max() + 50, 50), color=SERIES[0], alpha=0.55,
            edgecolor="white", linewidth=2)
    ax.axvline(median, color=MUTED, linewidth=1.0, linestyle="--")
    ax.annotate(f"медиана {median:.0f}", (median, ax.get_ylim()[1]), xytext=(4, -4), textcoords="offset points",
                fontsize=8, color=MUTED, va="top")
    for rank in range(3):
        ax.annotate(f"{rank + 1}", (scores[rank], 0), xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=8, color=TEXT, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1})
    ax.set_xlabel("Оценка J, мм (грубый поиск, шаг 30 мм)")
    ax.set_ylabel("Число геометрий")
    ax.set_title(f"а) {len(scores)} допустимых точек ЛГК", fontsize=10, color=TEXT, loc="left")
    if runs:
        ax = axes[0][1]
        offset = 0
        for data, label, index in runs:
            history = data["history"]
            gen = [offset + h["index"] for h in history]
            best = [h["best_score"] for h in history]
            ax.plot(gen, best, color=SERIES[index], linestyle=STYLES[index], label=f"{label}: {best[-1]:.0f}")
            if label.startswith("повторный"):
                ax.axvline(gen[0] - 0.5, color=GRID, linewidth=1.0)
            else:
                offset = max(offset, gen[-1])
        ax.axhline(median, color=MUTED, linewidth=1.0, linestyle="--")
        ax.annotate("медиана ЛГК", (0, median), xytext=(2, 3), textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_xlabel("Поколение CMA-ES")
        ax.set_ylabel("Лучшая оценка J, мм")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title("б) мультистарт CMA-ES", fontsize=10, color=TEXT, loc="left")
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_search")


def weight_run(weights: str, seed: int) -> str:
    return "run_lhs1_s1" if weights == "base" and seed == 1 else f"run_w_{weights}_s{seed}"


def plot_weights() -> None:
    """Каждый запуск — точка (ступень, кочка 250 мм); форма и цвет — набор весов."""
    points = {w: [load(weight_run(w, s)) for s in (1, 2, 3)] for w in WEIGHT_LABELS}
    if not any(d for runs in points.values() for d in runs):
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for index, (weights, runs) in enumerate(points.items()):
        values = [d["best_values"] for d in runs if d]
        if values:
            ax.scatter([v["step_up"] for v in values], [v["bump_250"] for v in values], s=60, color=SERIES[index],
                       marker=MARKERS[index], edgecolors="white", linewidths=1.5, zorder=3,
                       label=WEIGHT_LABELS[weights])
    # Кочка ищется не выше 300 мм: точки на этой линии упёрлись в границу поиска, а не в предел.
    ax.axhline(BUMP_CEILING_MM, color=MUTED, linewidth=1.0, linestyle="--")
    ax.annotate("граница поиска", (0, BUMP_CEILING_MM), xytext=(4, -10), textcoords="offset points",
                fontsize=8, color=MUTED)
    ax.set_xlabel("Наибольшая ступень, мм")
    ax.set_ylabel("Наибольшая кочка 250 мм, мм")
    ax.set_xlim(left=0)
    ax.set_ylim(0, BUMP_CEILING_MM * 1.08)
    ax.legend(title="Веса (3 seed каждый)", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    save(fig, "fig_weights")


def plot_torque() -> None:
    data = load("an_torque")
    if not data:
        return
    stats = spread(data["rows"], "control", "torque_nm")
    labels = {"speed": "регулятор скорости у каждого колеса", "torque": "одинаковый момент всех колёс"}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for index, (control, label) in enumerate(labels.items()):
        torques = sorted(t for c, t in stats if c == control)
        line_with_band(ax, torques, [stats[(control, t)] for t in torques], index, label, STYLES[index])
    ax.axvline(16.0, color=MUTED, linewidth=1.0, linestyle=":")
    ax.annotate("M365 Pro ≈ 16 Н·м", (16.0, 10), xytext=(4, 0), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel("Пиковый момент мотор-колеса, Н·м")
    ax.set_ylabel("Наибольшая ступень, мм")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig_torque")


def plot_radius() -> None:
    data = load("an_radius")
    if not data:
        return
    stats = spread(data["rows"], "control", "torque_nm", "radius_mm")
    torques = sorted({t for _, t, _ in stats})
    radii = np.array(sorted({r for _, _, r in stats}))
    titles = {"speed": "а) регулятор скорости у каждого колеса", "torque": "б) одинаковый момент всех колёс"}
    top = max(high for _, _, high in stats.values())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (control, title) in zip(axes, titles.items()):
        for index, torque in enumerate(torques):
            line_with_band(ax, radii, [stats[(control, torque, r)] for r in radii], index, f"{torque:.0f} Н·м")
        ax.plot(radii, 2.0 * radii, color=MUTED, linewidth=1.0, linestyle=":")
        ax.annotate("h = 2R", (radii[-1], 2.0 * radii[-1]), xytext=(4, -10), textcoords="offset points",
                    fontsize=8, color=MUTED)
        ax.axvline(108.0, color=MUTED, linewidth=1.0, linestyle="--")
        ax.annotate("M365 Pro", (108.0, 0), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("Радиус колеса R, мм (рычаги ∝ R)")
        ax.set_ylim(0, top * 1.08)
    axes[0].set_ylabel("Наибольшая ступень, мм")
    axes[1].legend(title="Момент колеса", loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_radius")


def plot_geometry() -> None:
    data = load("an_angles")
    if not data:
        return
    limits: dict[tuple, float] = defaultdict(lambda: float("inf"))
    for row in data["rows"]:
        key = (row["bogie_angle_deg"], row["rocker_angle_deg"])
        limits[key] = min(limits[key], row["value_mm"])
    bogie, rocker = data["bogie"], data["rocker"]
    grid = np.full((len(rocker), len(bogie)), np.nan)
    for (b, r), value in limits.items():
        grid[rocker.index(r), bogie.index(b)] = value
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.grid(False)
    step_b, step_r = bogie[1] - bogie[0], rocker[1] - rocker[0]
    extent = [bogie[0] - step_b / 2, bogie[-1] + step_b / 2, rocker[0] - step_r / 2, rocker[-1] + step_r / 2]
    ax.set_facecolor("#f0efec")
    image = ax.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap=BLUES, vmin=0)
    for item in data["invalid"]:
        ax.add_patch(plt.Rectangle((item["bogie_angle_deg"] - step_b / 2, item["rocker_angle_deg"] - step_r / 2),
                                   step_b, step_r, fill=False, hatch="///", edgecolor="#c3c2b7", linewidth=0))
    for i, r in enumerate(rocker):
        for j, b in enumerate(bogie):
            if np.isfinite(grid[i, j]):
                color = "white" if grid[i, j] > 0.55 * np.nanmax(grid) else TEXT
                ax.text(b, r, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color=color)
    base = data["design"]
    ax.plot(base["bogie_angle_deg"], base["rocker_angle_deg"], marker="o", markersize=14, fillstyle="none",
            color=SERIES[1], markeredgewidth=2)
    ax.set_xlabel("Угол bogie между плечами, ° (<180 — Λ, >180 — V)")
    ax.set_ylabel("Угол rocker между плечами, °")
    bar = fig.colorbar(image, ax=ax, pad=0.02)
    bar.set_label("Наибольшая ступень, мм")
    bar.outline.set_visible(False)
    save(fig, "fig_geometry")


def plot_heights() -> None:
    """Полосы «прошёл / не прошёл» по всем размерам препятствий с шагом 10 мм."""
    data = load("an_heights")
    if not data:
        return
    designs = {"final": "Итоговая геометрия"}
    rows = list(data["rows"])
    extra = load("an_heights_weights")
    if extra:
        # Запуск с «мелью» (если есть) — справа: на нём видно, зачем нужен сплошной предел.
        holes = [r["design"] for r in extra["rows"] if not r["success"]]
        if holes:
            designs[holes[0]] = holes[0]
        rows += extra["rows"]
    passed = {(r["design"], r["kind"], r["direction"], r["size_mm"]): r["success"] for r in rows}
    fig, axes = plt.subplots(1, len(designs), figsize=(3.6 * len(designs) + 1.2, 3.4), sharey=True, squeeze=False)
    for ax, (design, title) in zip(axes[0], designs.items()):
        ax.grid(False)
        labels = []
        for row, (kind, direction) in enumerate((k, d) for k in KIND_TITLES for d in (1, -1)):
            sizes = sorted(size for (de, ki, di, size) in passed if de == design and ki == kind and di == direction)
            for size in sizes:
                ok = passed[(design, kind, direction, size)]
                ax.add_patch(plt.Rectangle((size - 5, row - 0.38), 10, 0.76, linewidth=0,
                                           color=SERIES[2] if ok else FAIL))
            labels.append(f"{KIND_TITLES[kind]}, {'вперёд' if direction == 1 else 'назад'}")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xlim(0, 610)
        ax.set_xlabel("Размер препятствия, мм")
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[2]), plt.Rectangle((0, 0), 1, 1, color=FAIL)]
    axes[0][-1].legend(handles, ["прошёл", "не прошёл"], loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_heights")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for plot in (plot_mass, plot_search, plot_weights, plot_torque, plot_radius, plot_geometry, plot_heights):
        plot()
