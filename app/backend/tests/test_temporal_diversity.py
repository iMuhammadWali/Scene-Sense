"""Tests for temporal grouping and diversity selection."""

from app.pipeline.diversity import select_diverse_evidence
from app.pipeline.temporal import group_temporally, merge_context_windows


def test_temporal_grouping_gap():
    frames = [
        {"frame_id": 1, "timestamp": 3.0, "similarity": 0.9},
        {"frame_id": 2, "timestamp": 3.4, "similarity": 0.8},
        {"frame_id": 3, "timestamp": 3.8, "similarity": 0.7},
        {"frame_id": 4, "timestamp": 4.2, "similarity": 0.6},
        {"frame_id": 5, "timestamp": 9.5, "similarity": 0.85},
    ]
    groups, info = group_temporally(frames, gap_seconds=2.0)
    assert info["output_count"] == 2
    assert groups[0]["start_timestamp"] == 3.0
    assert groups[0]["end_timestamp"] == 4.2
    assert groups[1]["start_timestamp"] == 9.5


def test_merge_context_windows():
    windows = merge_context_windows([10.0, 10.4, 10.8, 12.0], window_seconds=5.0)
    assert len(windows) == 1
    assert windows[0][0] <= 5.0
    assert windows[0][1] >= 17.0


def test_diversity_simple_small_set():
    groups = [
        {
            "group_id": 1,
            "start_timestamp": 1.0,
            "end_timestamp": 1.4,
            "frames": [
                {"frame_id": 1, "timestamp": 1.0, "similarity": 0.9},
                {"frame_id": 2, "timestamp": 1.4, "similarity": 0.8},
            ],
        },
        {
            "group_id": 2,
            "start_timestamp": 8.0,
            "end_timestamp": 8.0,
            "frames": [{"frame_id": 3, "timestamp": 8.0, "similarity": 0.7}],
        },
    ]
    selected, info = select_diverse_evidence(groups, query_type="SIMPLE", max_simple=3)
    assert 1 <= info["selected_count"] <= 3
    assert selected[0]["timestamp"] <= selected[-1]["timestamp"]


def test_diversity_complex_preserves_chronology():
    groups = [
        {
            "group_id": 1,
            "start_timestamp": 12.0,
            "end_timestamp": 13.0,
            "frames": [
                {"frame_id": 10, "timestamp": 12.0, "similarity": 0.8},
                {"frame_id": 11, "timestamp": 13.0, "similarity": 0.9},
            ],
        },
        {
            "group_id": 2,
            "start_timestamp": 17.0,
            "end_timestamp": 17.0,
            "frames": [{"frame_id": 12, "timestamp": 17.0, "similarity": 0.7}],
        },
    ]
    selected, _ = select_diverse_evidence(groups, query_type="COMPLEX")
    ts = [s["timestamp"] for s in selected]
    assert ts == sorted(ts)
