"""CLIP cosine-similarity frame pruning against last RETAINED frame."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.pipeline.frame_extraction import CandidateFrame
from app.utils.logging_setup import get_logger
from app.utils.math_ops import cosine_similarity
from app.utils.timing import timed

logger = get_logger("scenesense.pruning")


def prune_frames(
    candidates: list[CandidateFrame],
    embeddings: np.ndarray,
    threshold: float = 0.95,
) -> tuple[list[CandidateFrame], np.ndarray, dict[str, Any]]:
    """
    Keep a frame if cosine similarity with the LAST RETAINED frame <= threshold.

    Important research rule:
    Compare each candidate against the most recently retained embedding,
    NOT merely against the previous extracted candidate.
    """
    if len(candidates) != len(embeddings):
        raise ValueError("candidates and embeddings length mismatch")

    with timed() as t:
        if not candidates:
            info = {
                "stage": "frame_pruning",
                "status": "completed",
                "input_count": 0,
                "output_count": 0,
                "rejected_count": 0,
                "threshold": threshold,
                "pruning_percentage": 0.0,
                "processing_time_seconds": t[0],
            }
            return [], np.zeros((0, embeddings.shape[1] if embeddings.ndim == 2 else 0), dtype=np.float32), info

        retained_frames: list[CandidateFrame] = []
        retained_embs: list[np.ndarray] = []
        decisions: list[dict[str, Any]] = []

        # First frame always kept
        retained_frames.append(candidates[0])
        retained_embs.append(embeddings[0])
        decisions.append(
            {
                "frame_id": candidates[0].frame_id,
                "timestamp": candidates[0].timestamp,
                "decision": "KEEP",
                "similarity_to_last_retained": None,
                "reference_frame_id": None,
            }
        )

        for i in range(1, len(candidates)):
            ref_emb = retained_embs[-1]
            ref_frame = retained_frames[-1]
            sim = cosine_similarity(embeddings[i], ref_emb)
            if sim > threshold:
                decisions.append(
                    {
                        "frame_id": candidates[i].frame_id,
                        "timestamp": candidates[i].timestamp,
                        "decision": "REJECT",
                        "similarity_to_last_retained": round(sim, 6),
                        "reference_frame_id": ref_frame.frame_id,
                    }
                )
            else:
                retained_frames.append(candidates[i])
                retained_embs.append(embeddings[i])
                decisions.append(
                    {
                        "frame_id": candidates[i].frame_id,
                        "timestamp": candidates[i].timestamp,
                        "decision": "KEEP",
                        "similarity_to_last_retained": round(sim, 6),
                        "reference_frame_id": ref_frame.frame_id,
                    }
                )

        retained_matrix = np.vstack(retained_embs).astype(np.float32)
        rejected = len(candidates) - len(retained_frames)
        pct = (rejected / len(candidates)) * 100.0 if candidates else 0.0
        info = {
            "stage": "frame_pruning",
            "status": "completed",
            "input_count": len(candidates),
            "output_count": len(retained_frames),
            "rejected_count": rejected,
            "threshold": threshold,
            "pruning_percentage": round(pct, 2),
            "processing_time_seconds": t[0],
            "decisions_preview": decisions[:50],
            "decisions_count": len(decisions),
        }

    logger.info(
        "[PRUNING] input=%s retained=%s rejected=%s threshold=%s duration=%.2fs",
        len(candidates),
        len(retained_frames),
        rejected,
        threshold,
        t[0],
    )
    return retained_frames, retained_matrix, info
