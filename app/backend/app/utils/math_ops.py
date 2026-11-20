"""Embedding math helpers used by pruning and FAISS."""

from __future__ import annotations

import numpy as np


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize vectors along the last axis."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm < eps:
            return arr.copy()
        return (arr / norm).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (arr / norms).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return float(np.dot(a_n, b_n))
