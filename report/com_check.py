"""Итоговая геометрия: пределы при ЦМ на 130 мм над главным шарниром и в самом шарнире (раздел «Высота центра масс»).

    python report/com_check.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))

from experiments import UPPER, final_name, load, platform  # noqa: E402
from model import Design  # noqa: E402
from simulate import find_limit  # noqa: E402

KINDS = [("step_up", 0.0), ("step_down", 0.0), ("ditch", 0.0), ("bump", 120.0), ("bump", 250.0)]


def task(args):
    com_z, kind, width, direction = args
    design = replace(Design(**load(final_name())["best"]), com_z_mm=com_z)
    key = f"bump_{width:.0f}" if kind == "bump" else kind
    r = find_limit(design, platform(), kind, direction, upper_mm=UPPER[key], tolerance_mm=5.0,
                   bump_width_mm=width or 120.0)
    return com_z, key, direction, r.value_mm, r.reason


if __name__ == "__main__":
    tasks = [(z, k, w, d) for z in (130.0, 0.0) for k, w in KINDS for d in (1, -1)]
    with ProcessPoolExecutor() as pool:
        for z, key, d, value, reason in pool.map(task, tasks):
            print(f"ЦМ +{z:3.0f}  {key:10s} {'вперёд' if d == 1 else 'назад '} {value:5.0f}  {reason[:70]}")
