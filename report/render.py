"""Рендеры MuJoCo для отчёта: препятствия и GIF прохождения.

    python report/render.py obstacles gif  # gif — с раскадровкой fig_climb.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from dataclasses import replace  # noqa: E402

from compute import PLATFORM as BASE_PLATFORM  # noqa: E402
from model import Design  # noqa: E402
from schemes import final_run, lhs_best, load  # noqa: E402
from simulate import CHECK_STEPS, RAMP_S, SETTLE_S, Obstacle, _Ids, _Motor, make_model, run  # noqa: E402

FIG = Path(__file__).resolve().parent / "fig"
PLATFORM = replace(BASE_PLATFORM, body_mass_kg=float(load("mass_choice")["body_mass_kg"]))
# Итоговая геометрия (пока её нет — лучшая точка ЛГК) и её наибольшая ступень, округлённая вниз до 10 мм.
_FINAL = final_run()
OPTIMIZED = Design(**_FINAL["best"]) if _FINAL else lhs_best()
STEP_MM = 10.0 * int(_FINAL["best_values"]["step_up"] // 10) if _FINAL else 300.0
CLIMB_GIF = FIG / "step_final.gif"


def _rgba(hex_color: str, alpha: float = 1.0) -> list[float]:
    value = hex_color.lstrip("#")
    return [int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)] + [alpha]


def style(model) -> None:
    """Цвета как на схемах: rocker — синий, bogie — оранжевый, грунт — светлый, препятствие — песочное."""
    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1440
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.55, 0.55, 0.55]
    model.vis.quality.shadowsize = 4096
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name == "floor":
            color = _rgba("#d9d6cc")
        elif name.startswith("ground"):
            color = _rgba("#b98f62")
        elif name.startswith("arm_rocker"):
            color = _rgba("#2a78d6")
        elif name.startswith("arm_bogie"):
            color = _rgba("#eb6834")
        elif name.startswith("strut"):
            color = _rgba("#6b6a66")
        elif name.startswith("tire"):
            color = _rgba("#2b2b2d")
        elif name == "body_visual":
            # Корпус в расчёте не участвует — не показываем.
            color = _rgba("#9aa3ad", 0.0)
        else:
            continue
        model.geom_rgba[i] = color


def camera(lookat, distance: float, azimuth: float = 115.0, elevation: float = -22.0):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


OBSTACLES = [
    ("Подъём на ступень", Obstacle("step_up", 300.0, 1)),
    ("Съезд со ступени", Obstacle("step_down", 300.0, 1)),
    ("Ров (глубина 500 мм)", Obstacle("ditch", 200.0, 1, ditch_depth_mm=500.0)),
    ("Кочка шириной 120 мм", Obstacle("bump", 100.0, 1, bump_width_mm=120.0)),
    ("Кочка шириной 250 мм", Obstacle("bump", 150.0, 1, bump_width_mm=250.0)),
]


def obstacle_images(width: int = 900, height: int = 560) -> list[tuple[str, np.ndarray]]:
    images = []
    for title, obstacle in OBSTACLES:
        model, boxes, floor, _ = make_model(OPTIMIZED, PLATFORM, obstacle)
        style(model)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        edges = [x for box in boxes for x in (box.x0, box.x1) if abs(x) < 5000.0]
        edge = min(edges) / 1000.0
        x = (data.qpos[0] + edge) / 2.0 + 0.2
        z = data.qpos[2] - 0.1
        # Ров виден только сверху.
        elevation = -32.0 if obstacle.kind == "ditch" else -24.0
        with mujoco.Renderer(model, height, width) as renderer:
            renderer.update_scene(data, camera=camera([x, 0.0, z], 2.5, azimuth=62.0, elevation=elevation))
            images.append((title, crop_sky(renderer.render())))
    return images


def crop_sky(image: np.ndarray) -> np.ndarray:
    """Срезает сверху чёрные строки фона (неба нет)."""
    dark = (image.max(axis=2) < 8).mean(axis=1) > 0.05
    top = int(np.argmin(dark)) if dark[0] else 0
    return image[top:]


def obstacles_figure() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = obstacle_images()
    rows = min(image.shape[0] for _, image in images)
    images = [(title, image[image.shape[0] - rows:]) for title, image in images]
    fig, axes = plt.subplots(1, len(images), figsize=(3.2 * len(images), 2.4))
    for ax, (title, image) in zip(axes, images):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_obstacles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def gif(obstacle: Obstacle, path: Path, fps: int = 20, width: int = 720, height: int = 450) -> None:
    """Повторяет прогон run() в режиме регуляторов скорости и снимает кадры с камеры, едущей за ровером."""
    platform = PLATFORM
    assert platform.control == "speed"
    result = run(OPTIMIZED, platform, obstacle, record=False)
    print(f"{path.name}: {'прошёл' if result.success else 'не прошёл'} за {result.duration_s:.1f} с {result.reason}")
    model, boxes, floor, _ = make_model(OPTIMIZED, platform, obstacle)
    style(model)
    data = mujoco.MjData(model)
    ids = _Ids(model)
    motor = _Motor(ids, platform)
    omega = platform.speed_m_s / (platform.wheel_radius_mm / 1000.0)
    mujoco.mj_forward(model, data)
    frames = []
    # Видео ускорено: кадр каждые 0,2 с модели при 20 к/с — в 4 раза быстрее.
    every = 0.2
    next_frame = 0.0
    with mujoco.Renderer(model, height, width) as renderer:
        while data.time < result.duration_s:
            ramp = min(1.0, max(0.0, (data.time - SETTLE_S) / RAMP_S))
            data.ctrl[:] = obstacle.direction * omega * ramp
            motor.apply_forcerange(model, data)
            mujoco.mj_step(model, data, nstep=CHECK_STEPS)
            if data.time >= next_frame:
                next_frame += every
                lookat = [data.qpos[0], 0.0, max(data.qpos[2] - 0.05, 0.2)]
                renderer.update_scene(data, camera=camera(lookat, 2.8, azimuth=62.0, elevation=-24.0))
                frames.append(Image.fromarray(renderer.render()))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    frames[len(frames) // 2].save(path.with_suffix(".png"))
    print(f"{path.name}: {len(frames)} кадров")


def climb_frames(path: Path = CLIMB_GIF, fractions=(0.12, 0.32, 0.5, 0.68, 0.98)) -> None:
    """Раскадровка GIF для печатного отчёта."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    animation = Image.open(path)
    picks = [int(animation.n_frames * f) for f in fractions]
    fig, axes = plt.subplots(1, len(picks), figsize=(16, 2.4))
    for ax, index in zip(axes, picks):
        animation.seek(index)
        width, height = animation.size
        ax.imshow(animation.convert("RGB").crop((width // 6, 0, width - width // 20, height)))
        ax.set_title(f"t = {index * 0.2:.0f} с", fontsize=10)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_climb.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for name in sys.argv[1:]:
        if name == "obstacles":
            obstacles_figure()
        elif name == "gif":
            gif(Obstacle("step_up", STEP_MM, 1), CLIMB_GIF)
            climb_frames()
