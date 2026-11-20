"""Tests for last-retained-frame pruning behavior."""

from __future__ import annotations

import numpy as np

from app.pipeline.frame_extraction import CandidateFrame
from app.pipeline.pruning import prune_frames
from app.utils.math_ops import cosine_similarity, l2_normalize


def _frame(i: int) -> CandidateFrame:
    return CandidateFrame(frame_id=i, timestamp=float(i) * 0.2, source_frame_index=i)


def test_cosine_normalization_unit_norm():
    v = np.array([3.0, 4.0], dtype=np.float32)
    n = l2_normalize(v)
    assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-5
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-5


def test_pruning_compares_against_last_retained():
    """
    Synthetic case from the research spec:

    F1 KEEP
    F2 similar to F1 -> REJECT
    F3 similar to F1 -> REJECT
    F4 different from F1 -> KEEP
    F5 similar to F4 -> REJECT
    F6 different from F4 -> KEEP
    """
    # Construct embeddings in 2D for clarity
    e1 = l2_normalize(np.array([1.0, 0.0], dtype=np.float32))
    e2 = l2_normalize(np.array([0.999, 0.01], dtype=np.float32))  # ~1 vs e1
    e3 = l2_normalize(np.array([0.998, 0.02], dtype=np.float32))  # ~1 vs e1
    e4 = l2_normalize(np.array([0.0, 1.0], dtype=np.float32))  # orthogonal to e1
    e5 = l2_normalize(np.array([0.01, 0.999], dtype=np.float32))  # ~1 vs e4
    e6 = l2_normalize(np.array([-1.0, 0.0], dtype=np.float32))  # different from e4

    # Ensure similarities match intended decisions at threshold 0.95
    assert cosine_similarity(e2, e1) > 0.95
    assert cosine_similarity(e3, e1) > 0.95
    assert cosine_similarity(e4, e1) <= 0.95
    assert cosine_similarity(e5, e4) > 0.95
    assert cosine_similarity(e6, e4) <= 0.95

    candidates = [_frame(i) for i in range(1, 7)]
    emb = np.vstack([e1, e2, e3, e4, e5, e6])
    retained, retained_emb, info = prune_frames(candidates, emb, threshold=0.95)

    retained_ids = [f.frame_id for f in retained]
    assert retained_ids == [1, 4, 6]
    assert info["rejected_count"] == 3
    assert info["output_count"] == 3
    assert retained_emb.shape[0] == 3


def test_pruning_does_not_compare_only_to_previous_candidate():
    """
    If implementation wrongly compared to previous extracted frame,
    F3 vs F2 might be kept/rejected incorrectly.

    Setup: F2 similar to F1 (reject). F3 similar to F2 but still similar to F1.
    Correct: reject F3 vs last retained F1.
    """
    e1 = l2_normalize(np.array([1.0, 0.0], dtype=np.float32))
    e2 = l2_normalize(np.array([0.999, 0.02], dtype=np.float32))
    e3 = l2_normalize(np.array([0.997, 0.04], dtype=np.float32))
    candidates = [_frame(1), _frame(2), _frame(3)]
    emb = np.vstack([e1, e2, e3])
    retained, _, _ = prune_frames(candidates, emb, threshold=0.95)
    assert [f.frame_id for f in retained] == [1]
