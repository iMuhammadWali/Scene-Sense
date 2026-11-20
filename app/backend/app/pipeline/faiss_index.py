"""FAISS index over retained CLIP embeddings (cosine via IndexFlatIP)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.utils.logging_setup import get_logger
from app.utils.math_ops import l2_normalize
from app.utils.timing import timed

logger = get_logger("scenesense.faiss")


def build_faiss_index(
    embeddings: np.ndarray,
    frames: list[dict[str, Any]],
) -> tuple[faiss.Index, list[dict[str, Any]], dict[str, Any]]:
    """Build IndexFlatIP on L2-normalized embeddings for cosine similarity."""
    emb = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    if emb.ndim != 2 or emb.shape[0] == 0:
        raise ValueError("No embeddings available to build FAISS index")
    if len(frames) != emb.shape[0]:
        raise ValueError("frames/embeddings length mismatch for FAISS mapping")

    with timed() as t:
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        mapping = []
        for i, frame in enumerate(frames):
            mapping.append(
                {
                    "faiss_index": i,
                    "frame_id": int(frame["frame_id"]),
                    "timestamp": float(frame["timestamp"]),
                }
            )
        info = {
            "stage": "faiss_index",
            "status": "completed",
            "input_count": emb.shape[0],
            "output_count": int(index.ntotal),
            "dimension": dim,
            "metric": "inner_product_on_normalized_vectors",
            "processing_time_seconds": t[0],
        }
    logger.info(
        "[FAISS] vectors=%s dimension=%s duration=%.2fs",
        emb.shape[0],
        dim,
        t[0],
    )
    return index, mapping, info


def save_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    if not path.exists():
        raise FileNotFoundError(f"FAISS index missing: {path}")
    return faiss.read_index(str(path))


def search(
    index: faiss.Index,
    query_embedding: np.ndarray,
    mapping: list[dict[str, Any]],
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Return Top-K hits with frame_id/timestamp/similarity."""
    q = l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
    k = min(int(top_k), int(index.ntotal))
    if k <= 0:
        return []
    scores, indices = index.search(q, k)
    results: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0:
            continue
        meta = mapping[int(idx)]
        results.append(
            {
                "rank": rank,
                "faiss_index": int(idx),
                "frame_id": int(meta["frame_id"]),
                "timestamp": float(meta["timestamp"]),
                "similarity": float(score),
            }
        )
    return results
