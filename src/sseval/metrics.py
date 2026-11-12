"""Temporal IoU, recall/hit metrics, bootstrap CIs, rank-based AUROC."""
from __future__ import annotations
import numpy as np, pandas as pd

BOOT_B = 1000
BOOT_SEED = 12345

def temporal_iou(a, b) -> float:
    """IoU of two [start, end] intervals. Degenerate (zero-union) -> 0.0."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0

def iou_arrays(ps, pe, gs, ge) -> np.ndarray:
    ps, pe, gs, ge = map(lambda x: np.asarray(x, dtype=np.float64), (ps, pe, gs, ge))
    inter = np.maximum(0.0, np.minimum(pe, ge) - np.maximum(ps, gs))
    union = (pe - ps) + (ge - gs) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)

def bootstrap_indices(n: int, B: int = BOOT_B, seed: int = BOOT_SEED) -> np.ndarray:
    """[B, n] resample indices over queries (shared across methods => paired comparisons)."""
    return np.random.default_rng(seed).integers(0, n, size=(B, n))

def bootstrap_ci(x, idx: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of the mean of x under resample indices idx [B, n]."""
    x = np.asarray(x, dtype=np.float64)
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))

def summarize(pq: pd.DataFrame, idx: np.ndarray | None = None) -> dict:
    """pq columns: hit1, hit5, iou. Returns the standard metric dict (+ CIs for R@1@0.5 and Hit@1 if idx given)."""
    n = len(pq)
    r5 = (pq["iou"].values >= 0.5)
    out = dict(n_queries=n, hit1=pq["hit1"].mean(), hit5=pq["hit5"].mean(),
               r1_iou03=(pq["iou"] >= 0.3).mean(), r1_iou05=r5.mean(), r1_iou07=(pq["iou"] >= 0.7).mean(),
               miou=pq["iou"].mean())
    if idx is not None and n > 0:
        out["r1_iou05_lo"], out["r1_iou05_hi"] = bootstrap_ci(r5.astype(float), idx)
        out["hit1_lo"], out["hit1_hi"] = bootstrap_ci(pq["hit1"].values.astype(float), idx)
    return out

def auroc(pos, neg) -> float:
    """Rank-based AUROC (Mann-Whitney U with average ranks for ties). P(score_pos > score_neg) + 0.5 P(tie)."""
    pos = np.asarray(pos, dtype=np.float64); neg = np.asarray(neg, dtype=np.float64)
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    ranks = np.empty(len(allv), dtype=np.float64)
    # average ranks over ties
    i = 0
    n = len(sv)
    uniq_start = np.r_[0, np.flatnonzero(np.diff(sv)) + 1, n]
    for a, b in zip(uniq_start[:-1], uniq_start[1:]):
        ranks[order[a:b]] = (a + 1 + b) / 2.0
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))
