"""E10 utilities: re-ranking fast-path candidates with a per-candidate score (BLIP-ITM), fusion, query text variants, AUROC bootstrap.

Pure numpy; no model code here (the model lives in experiments/e10_blip_rerank/run.py).
Matrices are [n_queries, n_cand]; NaN marks a missing candidate (queries with fewer candidates than the matrix width).
"""
from __future__ import annotations
import re
import numpy as np
from .localize import fixed_span
from .metrics import iou_arrays, auroc

_PERSON_RE = re.compile(r"^\s*(?:a\s+|the\s+|one\s+)?(?:person|someone)\s+", re.I)

def strip_person(q: str) -> str:
    """Drop a leading 'person' / 'a person' (Charades captions: 'person closing the door.' -> 'closing the door.').
    Returns the original if nothing would be left or there is no such prefix."""
    s = _PERSON_RE.sub("", q, count=1).strip()
    return s if s else q

def frame_times(t: float, duration: float, offsets=(0.0,)) -> list[float]:
    """Timestamps t+offset, clipped into [0, duration-0.1] (last decodable frame), rounded to ms. Duplicates are kept (same order as offsets)."""
    hi = max(0.0, duration - 0.1)
    return [round(float(min(max(t + o, 0.0), hi)), 3) for o in offsets]

def zscore_rows(x: np.ndarray) -> np.ndarray:
    """z-normalise each row over its non-NaN entries; a constant row (or a single candidate) -> zeros; NaN stays NaN."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(all="ignore"):
        mu = np.nanmean(x, axis=1, keepdims=True); sd = np.nanstd(x, axis=1, keepdims=True)
    ok = np.isfinite(sd) & (sd > 1e-12)
    z = np.where(ok, (x - mu) / np.where(ok, sd, 1.0), 0.0)
    return np.where(np.isnan(x), np.nan, z)

def row_ranks(x: np.ndarray) -> np.ndarray:
    """Rank within each row, 1 = highest; ties broken by lower column index (= better CLIP rank); NaN -> NaN."""
    x = np.asarray(x, dtype=np.float64)
    key = np.where(np.isnan(x), -np.inf, x)
    order = np.argsort(-key, axis=1, kind="stable")
    r = np.empty_like(order, dtype=np.float64)
    rows = np.arange(len(x))[:, None]
    r[rows, order] = np.arange(1, x.shape[1] + 1)[None, :]
    return np.where(np.isnan(x), np.nan, r)

def fuse_z(clip: np.ndarray, itm: np.ndarray, w: float) -> np.ndarray:
    """z(clip) + w * z(itm) per query. w=0 -> CLIP order; w large -> ITM order."""
    return zscore_rows(clip) + w * zscore_rows(itm)

def fuse_rank(clip: np.ndarray, itm: np.ndarray, w: float, k: float = 2.0) -> np.ndarray:
    """Weighted reciprocal-rank fusion: 1/(k+rank_clip) + w/(k+rank_itm)."""
    return 1.0 / (k + row_ranks(clip)) + w / (k + row_ranks(itm))

def argmax_rows(x: np.ndarray) -> np.ndarray:
    """Row argmax, NaN = -inf, ties -> lowest column (best CLIP rank, since candidates are stored in CLIP order)."""
    return np.argmax(np.where(np.isnan(x), -np.inf, x), axis=1)

def choose_w(clip, itm, hit, ws, fuse=fuse_z, idx=None):
    """w maximising Hit@1 on rows `idx` (default all); ties -> smaller w. -> (w, {w: hit1})."""
    idx = np.arange(len(clip)) if idx is None else np.asarray(idx)
    n = np.arange(len(clip))
    scores = {}
    for w in ws:
        ch = argmax_rows(fuse(clip, itm, w))
        scores[w] = float(hit[n, ch][idx].mean())
    best = max(sorted(ws), key=lambda w: (scores[w], -w))
    return best, scores

def candidate_score(frame_scores: np.ndarray, how: str = "max") -> float:
    """Aggregate the ITM scores of one candidate's frames (max or mean), ignoring NaN."""
    a = np.asarray(frame_scores, dtype=np.float64)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    return float(a.max() if how == "max" else a.mean())

def window_iou(t: np.ndarray, gs, ge, dur, L: float = 8.0) -> np.ndarray:
    """IoU of the fixed-length (L s) window centred on each candidate time (E7 M2 rule) with the GT. t: [n, c]; gs/ge/dur: [n]. NaN stays NaN."""
    t = np.asarray(t, dtype=np.float64)
    out = np.full(t.shape, np.nan)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            if np.isnan(t[i, j]):
                continue
            a, b = fixed_span(float(t[i, j]), L, float(dur[i]))
            out[i, j] = iou_arrays([a], [b], [gs[i]], [ge[i]])[0]
    return out

def auroc_ci(pos, neg, B: int = 1000, seed: int = 7, alpha: float = 0.05):
    """AUROC (positives should score higher) with a percentile bootstrap resampling positives and negatives separately."""
    pos = np.asarray(pos, dtype=np.float64); neg = np.asarray(neg, dtype=np.float64)
    rng = np.random.default_rng(seed)
    vals = [auroc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(B)]
    return auroc(pos, neg), float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
