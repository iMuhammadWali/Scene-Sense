"""Diversity / representative evidence selection from temporal groups."""

from __future__ import annotations

from typing import Any, Literal


def select_diverse_evidence(
    groups: list[dict[str, Any]],
    query_type: Literal["SIMPLE", "COMPLEX"] = "SIMPLE",
    max_simple: int = 3,
    max_complex_per_group: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Select representative frames:
    - Prefer highest similarity within each group
    - Preserve temporal coverage across groups
    - For SIMPLE: small evidence set (1-3 when sufficient)
    - For COMPLEX: chronological coverage across groups
    """
    if not groups:
        info = {
            "stage": "diversity_selection",
            "status": "completed",
            "input_groups": 0,
            "selected_count": 0,
            "query_type": query_type,
            "selected": [],
        }
        return [], info

    selected: list[dict[str, Any]] = []

    if query_type == "SIMPLE":
        # Flatten and pick top unique by similarity, prefer covering different groups
        group_reps: list[dict[str, Any]] = []
        for g in groups:
            best = max(g["frames"], key=lambda f: float(f.get("similarity", 0.0)))
            item = dict(best)
            item["source"] = f"group_{g['group_id']}_best"
            group_reps.append(item)
        group_reps.sort(key=lambda f: float(f.get("similarity", 0.0)), reverse=True)
        selected = group_reps[: max(1, min(max_simple, len(group_reps)))]
        selected.sort(key=lambda f: float(f["timestamp"]))
    else:
        for g in groups:
            frames = sorted(
                g["frames"],
                key=lambda f: float(f.get("similarity", 0.0)),
                reverse=True,
            )
            take = frames[: max(1, min(max_complex_per_group, len(frames)))]
            # Also include earliest/latest if group is wide and we only took middle-ish
            if len(g["frames"]) >= 3 and len(take) < 3:
                ordered = sorted(g["frames"], key=lambda f: float(f["timestamp"]))
                for edge in (ordered[0], ordered[-1]):
                    if edge["frame_id"] not in {x["frame_id"] for x in take}:
                        take.append(edge)
            for frame in take:
                item = dict(frame)
                item["source"] = f"group_{g['group_id']}"
                selected.append(item)
        # Deduplicate by frame_id, preserve chronological order
        dedup: dict[int, dict[str, Any]] = {}
        for item in selected:
            fid = int(item["frame_id"])
            if fid not in dedup or float(item.get("similarity", 0)) > float(
                dedup[fid].get("similarity", 0)
            ):
                dedup[fid] = item
        selected = sorted(dedup.values(), key=lambda f: float(f["timestamp"]))

    info = {
        "stage": "diversity_selection",
        "status": "completed",
        "input_groups": len(groups),
        "selected_count": len(selected),
        "query_type": query_type,
        "selected": [
            {
                "frame_id": s.get("frame_id"),
                "timestamp": s.get("timestamp"),
                "similarity": s.get("similarity"),
                "source": s.get("source"),
            }
            for s in selected
        ],
    }
    return selected, info
