"""Выбор ключевых кадров вместо анимации."""

from __future__ import annotations

from dataclasses import dataclass

from analysis import CheckedTraversal
from geometry import Pose
from statics import Scenario, StaticFrameResult


@dataclass(frozen=True)
class KeyFrame:
    title: str
    pose: Pose
    static: StaticFrameResult


def select_key_frames(
    checked: CheckedTraversal,
    scenario: Scenario,
    maximum_frames: int = 12,
) -> tuple[KeyFrame, ...]:
    """Возвращает начало, смены контактов и наиболее нагруженные кадры."""
    if not checked.success or not checked.traversal.poses:
        return ()
    traversal_statics = next(
        (item for item in checked.statics if item.scenario == scenario),
        None,
    )
    if traversal_statics is None or not traversal_statics.frames:
        return ()

    frames = traversal_statics.frames
    reasons: dict[int, list[str]] = {
        frames[0].pose_index: ["Исходное положение"],
        frames[-1].pose_index: ["Завершение прохождения"],
    }

    previous_signature: tuple[tuple[str, str, str], ...] | None = None
    seen_special: set[tuple[str, str]] = set()
    for frame in frames:
        signature = tuple(
            (load.side, load.wheel, load.mode) for load in frame.contacts
        )
        if previous_signature is not None and signature != previous_signature:
            reasons.setdefault(frame.pose_index, []).append("Смена схемы контактов")
        previous_signature = signature
        for load in frame.contacts:
            if load.side != "левая":
                continue
            if "кромка" not in load.mode and "стенка" not in load.mode:
                continue
            key = load.wheel, load.mode
            if key not in seen_special:
                reasons.setdefault(frame.pose_index, []).append(
                    f"{load.wheel}: {load.mode}"
                )
                seen_special.add(key)

    extrema = (
        (
            max(frames, key=lambda item: item.maximum_torque_utilization),
            "Максимальная загрузка приводов",
        ),
        (
            max(frames, key=lambda item: item.maximum_friction_utilization),
            "Максимальное использование сцепления",
        ),
        (
            min(frames, key=lambda item: item.support_polygon_margin_mm),
            "Минимальный запас устойчивости",
        ),
        (
            max(frames, key=lambda item: abs(item.body_pitch_deg)),
            "Максимальный наклон корпуса",
        ),
        (
            max(frames, key=lambda item: abs(item.body_roll_deg)),
            "Максимальный поперечный крен",
        ),
    )
    for frame, label in extrema:
        reasons.setdefault(frame.pose_index, []).append(label)

    static_by_index = {frame.pose_index: frame for frame in frames}
    prioritized = sorted(
        reasons,
        key=lambda index: (
            index not in (frames[0].pose_index, frames[-1].pose_index),
            index,
        ),
    )
    if len(prioritized) > maximum_frames:
        required = {frames[0].pose_index, frames[-1].pose_index}
        extrema_indices = {frame.pose_index for frame, _ in extrema}
        keep = sorted(required | extrema_indices)
        for index in prioritized:
            if len(keep) >= maximum_frames:
                break
            if index not in keep:
                keep.append(index)
        prioritized = sorted(keep)

    result: list[KeyFrame] = []
    for ordinal, index in enumerate(sorted(prioritized), start=1):
        static = static_by_index.get(index)
        if static is None:
            continue
        labels = "; ".join(dict.fromkeys(reasons[index]))
        result.append(
            KeyFrame(
                title=f"Ключевой кадр {ordinal}: {labels}",
                pose=checked.traversal.poses[index],
                static=static,
            )
        )
    return tuple(result)

