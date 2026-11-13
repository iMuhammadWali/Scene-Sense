"""E7 span utilities: score curves on a regular grid, smoothing, fixed-length / threshold / contrast spans, NMS.

All spans are (start, end) in seconds. Curves live on a regular grid of DT seconds: s[j] covers [j*DT, (j+1)*DT).
"""
from __future__ import annotations
import numpy as np

DT = 0.2

def make_grid(duration: float, dt: float = DT) -> np.ndarray:
    """Grid-cell centres covering [0, duration]."""
    n = max(1, int(np.ceil(duration / dt - 1e-9)))
    return (np.arange(n) + 0.5) * dt

def item_ids(span_end: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the kept item whose coverage span contains each grid centre (spans tile [0, duration])."""
    return np.clip(np.searchsorted(span_end, centers, side="right"), 0, len(span_end) - 1)

def curve(item_emb: np.ndarray, ids: np.ndarray, qe: np.ndarray) -> np.ndarray:
    """Piecewise-constant score curve: cosine(query, item) held over each item's span, sampled on the grid."""
    return (item_emb @ qe)[ids].astype(np.float64)

def smooth(s: np.ndarray, w_s: float, dt: float = DT) -> np.ndarray:
    """Centered moving average of width w_s seconds (edge-replicated); w_s<=dt -> unchanged."""
    k = int(round(w_s / dt))
    if k <= 1:
        return s.copy()
    pad_l, pad_r = k // 2, k - 1 - k // 2
    sp = np.pad(s, (pad_l, pad_r), mode="edge")
    c = np.concatenate([[0.0], np.cumsum(sp)])
    return (c[k:] - c[:-k]) / k

def zscore(s: np.ndarray) -> np.ndarray:
    sd = s.std()
    return (s - s.mean()) / (sd if sd > 1e-12 else 1.0)

def fixed_span(center: float, L: float, duration: float) -> tuple[float, float]:
    """Length-L span centred on `center`; shifted (not shrunk) to stay inside [0, duration] when possible."""
    L = min(L, duration)
    a = min(max(center - L / 2.0, 0.0), duration - L)
    return a, a + L

def grow_span(s: np.ndarray, peak: int, alpha: float, dt: float = DT, duration: float | None = None) -> tuple[float, float]:
    """Contiguous span around `peak` where s >= s[peak] - alpha*(s[peak]-median(s))."""
    thr = s[peak] - alpha * (s[peak] - np.median(s))
    below = s < thr
    l = peak
    while l > 0 and not below[l - 1]:
        l -= 1
    r = peak
    while r < len(s) - 1 and not below[r + 1]:
        r += 1
    end = (r + 1) * dt
    return l * dt, min(end, duration) if duration is not None else end

def iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0

def contrast_spans(s: np.ndarray, lam: float, duration: float, dt: float = DT, step: float = 0.5,
                   min_len: float = 2.0, max_len: float = 25.0):
    """TFVTG-style proposal scoring. Candidates on a `step` grid; score = mean(s in) - mean(s out) - lam*len/duration.
    Integrals of the piecewise-constant curve are exact (interpolated cumulative sum). Returns (starts, ends, scores)."""
    T = len(s) * dt
    edges = np.arange(len(s) + 1) * dt
    C = np.concatenate([[0.0], np.cumsum(s) * dt])
    D = min(duration, T)
    hi = min(max_len, D - step)                     # keep a non-empty outside
    lens = np.arange(min_len, hi + 1e-9, step)
    if len(lens) == 0:
        lens = np.array([min(D, max(step, hi))]) if hi > 0 else np.array([D])
    starts = np.arange(0.0, D + 1e-9, step)
    a = starts[:, None]; b = a + lens[None, :]
    valid = b <= D + 1e-9
    Cf = lambda t: np.interp(t, edges, C)
    inside = Cf(np.minimum(b, T)) - Cf(a)
    ln = lens[None, :] + 0 * a
    mean_in = inside / ln
    out_len = np.maximum(T - ln, 1e-9)
    mean_out = (C[-1] - inside) / out_len
    sc = mean_in - mean_out - lam * ln / D
    sc = np.where(valid, sc, -np.inf)
    A = np.broadcast_to(a, sc.shape); B = np.broadcast_to(b, sc.shape)
    m = np.isfinite(sc)
    return A[m], B[m], sc[m]

def nms_topk(starts, ends, scores, k: int = 5, thr: float = 0.5):
    """Greedy temporal NMS. Returns list of (start, end) (at most k)."""
    order = np.argsort(-scores, kind="stable")
    keep: list[tuple[float, float]] = []
    for i in order:
        c = (float(starts[i]), float(ends[i]))
        if all(iou(c, kk) < thr for kk in keep):
            keep.append(c)
            if len(keep) == k:
                break
    return keep
