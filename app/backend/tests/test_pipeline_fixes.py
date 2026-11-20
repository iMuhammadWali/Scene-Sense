"""Tests for pipeline fixes: classifier labels, similarity honesty, temporal metadata."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from app.pipeline.classifier import LABEL_COMPLEX, LABEL_SIMPLE, QueryClassifier
from app.pipeline.query_pipeline import QueryPipeline, _timestamp_to_source_frame_id
from app.pipeline.retrieval import filter_by_similarity


def test_classifier_label_definitions():
    assert "visual retrieval" in LABEL_SIMPLE.lower() or "evidence" in LABEL_SIMPLE.lower()
    assert "natural-language" in LABEL_COMPLEX.lower() or "natural language" in LABEL_COMPLEX.lower()
    assert "aggregation" in LABEL_COMPLEX.lower() or "counting" in LABEL_COMPLEX.lower()


def test_similarity_fallback_is_honest():
    hits = [
        {"rank": 1, "frame_id": 1, "timestamp": 1.0, "similarity": 0.0698},
        {"rank": 2, "frame_id": 2, "timestamp": 2.0, "similarity": 0.0545},
        {"rank": 3, "frame_id": 3, "timestamp": 3.0, "similarity": 0.0540},
    ]
    filtered, info = filter_by_similarity(hits, threshold=0.20)
    assert info["threshold"] == 0.20
    assert info["candidate_count"] == 3
    assert info["passed_count"] == 0
    assert info["fallback_used"] is True
    assert info["best_similarity"] == 0.0698
    assert len(filtered) == 1
    assert filtered[0]["fallback"] is True
    assert filtered[0]["passed_threshold"] is False
    assert "did NOT pass" in info["note"] or "NOT" in info["note"]


def test_similarity_passed_no_fallback():
    hits = [
        {"rank": 1, "frame_id": 1, "timestamp": 1.0, "similarity": 0.31},
        {"rank": 2, "frame_id": 2, "timestamp": 2.0, "similarity": 0.25},
        {"rank": 3, "frame_id": 3, "timestamp": 3.0, "similarity": 0.10},
    ]
    filtered, info = filter_by_similarity(hits, threshold=0.20)
    assert info["passed_count"] == 2
    assert info["fallback_used"] is False
    assert info["best_similarity"] == 0.31
    assert all(f["passed_threshold"] for f in filtered)
    assert all(not f["fallback"] for f in filtered)


def test_timestamp_to_source_frame_id():
    assert _timestamp_to_source_frame_id(15.8, fps=5.0, frame_count=200) == 79
    assert _timestamp_to_source_frame_id(16.8, fps=5.0, frame_count=200) == 84
    assert _timestamp_to_source_frame_id(20.8, fps=5.0, frame_count=200) == 104
    # Clamp to last frame
    assert _timestamp_to_source_frame_id(999.0, fps=5.0, frame_count=100) == 99


def test_temporal_evidence_unique_source_frame_ids(tmp_path):
    """Temporal samples must not all reuse the FAISS retrieved_frame_id."""
    pipe = QueryPipeline.__new__(QueryPipeline)
    from app.config import Settings

    pipe.settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        complex_context_window_seconds=5.0,
        complex_sampling_fps=1.0,
    )

    # Fake video metadata + frame extraction
    selected = [
        {"frame_id": 104, "timestamp": 20.8, "similarity": 0.07},
    ]
    fake_samples = [
        (15.8, np.zeros((4, 4, 3), dtype=np.uint8)),
        (16.8, np.ones((4, 4, 3), dtype=np.uint8)),
        (17.8, np.full((4, 4, 3), 2, dtype=np.uint8)),
        (18.8, np.full((4, 4, 3), 3, dtype=np.uint8)),
        (19.8, np.full((4, 4, 3), 4, dtype=np.uint8)),
        (20.8, np.full((4, 4, 3), 5, dtype=np.uint8)),
        (21.8, np.full((4, 4, 3), 6, dtype=np.uint8)),
        (22.8, np.full((4, 4, 3), 7, dtype=np.uint8)),
        (23.8, np.full((4, 4, 3), 8, dtype=np.uint8)),
    ]
    with (
        patch(
            "app.pipeline.query_pipeline.read_video_metadata",
            return_value={"fps": 5.0, "frame_count": 200, "duration_seconds": 40.0},
        ),
        patch(
            "app.pipeline.query_pipeline.extract_frames_in_range",
            return_value=fake_samples,
        ),
    ):
        images, evidence, debug = pipe._build_complex_vlm_inputs("fake.mp4", selected)

    assert len(evidence) == len(images) >= 5
    source_ids = [e["source_frame_id"] for e in evidence]
    assert len(set(source_ids)) == len(source_ids), "source_frame_ids must be unique"
    # Must NOT all equal retrieved FAISS id 104
    assert not all(sid == 104 for sid in source_ids)
    assert evidence[0]["source_frame_id"] == 79
    assert evidence[1]["source_frame_id"] == 84
    # Retrieved trigger preserved separately
    assert all(e["retrieved_frame_id"] == 104 for e in evidence)
    # Chronological
    ts = [e["timestamp"] for e in evidence]
    assert ts == sorted(ts)
    assert debug["chronological"] is True


@pytest.mark.parametrize(
    "question,expected",
    [
        ("How many total vehicles can be seen in this video?", "COMPLEX"),
        ("How many vehicles are visible in this frame?", "COMPLEX"),
        ("What happens after the vehicle appears?", "COMPLEX"),
        ("Show me the vehicle", "SIMPLE"),
    ],
)
def test_bart_classifier_semantics(question, expected):
    from app.utils.model_cache import model_cached
    from app.config import get_settings

    if not model_cached(get_settings().classifier_model):
        pytest.skip("BART classifier weights not cached")

    QueryClassifier._instance = None
    clf = QueryClassifier.get()
    result, info = clf.classify(question)
    assert result == expected, (
        f"Expected {expected} for {question!r}, got {result} "
        f"(simple={info.get('simple_confidence')}, complex={info.get('complex_confidence')})"
    )
    assert info["simple_confidence"] is not None
    assert info["complex_confidence"] is not None
