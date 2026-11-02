"""Бенчмарк скорости расчёта: одинаковый набор поисков пределов на всех ядрах.

    python report/bench.py

Сравнивайте «всё на всех ядрах» между машинами: это то, сколько займёт оптимизация.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import Design, Platform  # noqa: E402
from simulate import find_limit  # noqa: E402

DESIGN = Design(bogie_rear_arm_mm=238.0, bogie_front_arm_mm=270.8, bogie_angle_deg=202.8, rocker_rear_arm_mm=248.0,
                rocker_front_arm_mm=356.3, rocker_angle_deg=192.4, outer_strut_angle_deg=18.0,
                middle_strut_angle_deg=18.5)
PLATFORM = Platform(body_mass_kg=25.0, friction=0.8)
# 48 поисков: 4 препятствия × 2 направления × 6 повторов — как одно поколение CMA-ES.
TASKS = [(kind, direction, upper) for kind, upper in (("step_up", 600.0), ("step_down", 600.0), ("ditch", 500.0),
                                                      ("bump", 300.0))
         for direction in (1, -1) for _ in range(6)]


def search(task: tuple) -> float:
    kind, direction, upper = task
    return find_limit(DESIGN, PLATFORM, kind, direction, upper_mm=upper, tolerance_mm=5.0).value_mm


def main() -> None:
    workers = os.cpu_count() or 1
    print(f"{platform.processor() or platform.machine()}, {workers} логических ядер, Python {platform.python_version()}")
    started = time.perf_counter()
    search(TASKS[0])
    single = time.perf_counter() - started
    print(f"один поиск на одном ядре: {single:.1f} с")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(search, TASKS))
    total = time.perf_counter() - started
    print(f"{len(TASKS)} поисков на всех ядрах: {total:.1f} с  ← сравнивайте это число")


if __name__ == "__main__":
    main()
