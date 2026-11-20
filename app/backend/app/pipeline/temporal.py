"""Temporal grouping of retrieved evidence frames."""

from __future__ import annotations

from typing import Any


def group_temporally(
    frames: list[dict[str, Any]],
    gap_seconds: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Group frames whose timestamps are within gap_seconds of the previous frame
    in the sorted sequence.
    """
    if not frames:
        info = {
            "stage": "temporal_grouping",
            "status": "completed",
            "input_count": 0,
            "output_count": 0,
            "gap_seconds": gap_seconds,
            "groups": [],
        }
        return [], info

    ordered = sorted(frames, key=lambda f: float(f["timestamp"]))
    groups: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = [ordered[0]]

    for frame in ordered[1:]:
        prev_t = float(current[-1]["timestamp"])
        cur_t = float(frame["timestamp"])
        if cur_t - prev_t <= gap_seconds:
            current.append(frame)
        else:
            groups.append(_finalize_group(len(groups) + 1, current))
            current = [frame]
    groups.append(_finalize_group(len(groups) + 1, current))

    info = {
        "stage": "temporal_grouping",
        "status": "completed",
        "input_count": len(frames),
        "output_count": len(groups),
        "gap_seconds": gap_seconds,
        "groups": [
            {
                "group_id": g["group_id"],
                "start_timestamp": g["start_timestamp"],
                "end_timestamp": g["end_timestamp"],
                "frame_count": len(g["frames"]),
                "timestamps": [f["timestamp"] for f in g["frames"]],
            }
            for g in groups
        ],
    }
    return groups, info


def _finalize_group(group_id: int, frames: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps = [float(f["timestamp"]) for f in frames]
    return {
        "group_id": group_id,
        "start_timestamp": min(timestamps),
        "end_timestamp": max(timestamps),
        "frames": frames,
    }


def merge_context_windows(
    timestamps: list[float],
    window_seconds: float = 5.0,
) -> list[tuple[float, float]]:
    """Merge overlapping ±window regions around timestamps."""
    if not timestamps:
        return []
    intervals = sorted(
        (max(0.0, t - window_seconds), t + window_seconds) for t in timestamps
    )
    merged: list[list[float]] = [[intervals[0][0], intervals[0][1]]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(round(a, 4), round(b, 4)) for a, b in merged]
