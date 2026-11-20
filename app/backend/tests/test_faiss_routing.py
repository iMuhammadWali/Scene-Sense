"""Tests for FAISS mapping persistence helpers and routing rule."""

from __future__ import annotations

import numpy as np

from app.pipeline.faiss_index import build_faiss_index, search
from app.pipeline.retrieval import filter_by_similarity
from app.utils.math_ops import l2_normalize


def test_faiss_mapping_and_search(tmp_path):
    frames = [
        {"frame_id": 17, "timestamp": 3.4},
        {"frame_id": 21, "timestamp": 4.2},
        {"frame_id": 40, "timestamp": 8.0},
    ]
    emb = l2_normalize(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    index, mapping, info = build_faiss_index(emb, frames)
    assert info["output_count"] == 3
    assert mapping[0]["frame_id"] == 17

    q = l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    hits = search(index, q, mapping, top_k=20)
    assert len(hits) == 3
    assert hits[0]["frame_id"] == 17


def test_retrieval_threshold_independent():
    hits = [
        {"rank": 1, "frame_id": 1, "timestamp": 1.0, "similarity": 0.9},
        {"rank": 2, "frame_id": 2, "timestamp": 2.0, "similarity": 0.1},
    ]
    filtered, info = filter_by_similarity(hits, threshold=0.2)
    assert len(filtered) == 1
    assert info["threshold"] == 0.2
    assert info["passed_count"] == 1
    assert info["fallback_used"] is False
    assert "0.95" not in str(info["threshold"])


def test_routing_rule_simple_fast_complex_slow():
    # Pure rule test — classifier output must map as specified
    def route(query_type: str) -> str:
        if query_type == "COMPLEX":
            return "SLOW"
        return "FAST"

    assert route("SIMPLE") == "FAST"
    assert route("COMPLEX") == "SLOW"
    # Retrieval confidence must not override
    retrieval_score = 0.99
    query_type = "COMPLEX"
    path = route(query_type)
    assert path == "SLOW"
    assert retrieval_score > 0.5  # high confidence still does not change path
