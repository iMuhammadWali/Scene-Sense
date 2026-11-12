"""Selection: the output of every frame-selection method, plus helpers to build spans.

An item = (timestamp, embedding, span_start, span_end). The span is the time interval of the video the item "covers";
a retrieved group's predicted moment is the union of its items' spans.
  * drop/sample methods (uniform, cos_*, maxinfo_*): midpoint rule - an item covers from the midpoint with the
    previous kept item to the midpoint with the next one; first/last extend to 0 / duration.
  * merge methods (streammem, hybrid): an item covers its merged run. Runs tile the video, so a run covers
    [first frame ts, next run's first frame ts) and the last run ends at the video duration. (With ideal 5 FPS this equals
    [first ts, last ts + one frame period].)  Item timestamp = run midpoint; item embedding = L2-renormalised mean.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Selection:
    ts: np.ndarray            # [n] item timestamps
    emb: np.ndarray          # [n, d] float32, L2-normalised
    span_start: np.ndarray   # [n]
    span_end: np.ndarray     # [n]
    n_input: int = 0         # frames the method was given (5 FPS count)
    select_s: float = 0.0    # wall time of the selection call (filled by pruning.select)
    meta: dict = field(default_factory=dict)

    @property
    def n_kept(self) -> int:
        return len(self.ts)

def l2n(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)).astype(np.float32)

def midpoint_spans(ts_kept: np.ndarray, duration: float):
    """Coverage spans for kept timestamps (sorted ascending). Tiles [0, duration] exactly."""
    t = np.asarray(ts_kept, dtype=np.float64)
    mids = (t[:-1] + t[1:]) / 2.0
    return np.concatenate([[0.0], mids]), np.concatenate([mids, [float(duration)]])

def from_indices(emb: np.ndarray, ts: np.ndarray, duration: float, idx) -> Selection:
    idx = np.unique(np.asarray(idx, dtype=int))           # sorted, unique
    if len(idx) == 0:
        idx = np.array([0])
    s, e = midpoint_spans(ts[idx], duration)
    return Selection(ts=ts[idx].astype(np.float64), emb=emb[idx], span_start=s, span_end=e, n_input=len(emb),
                     meta={"idx": idx})

def runs_from_breaks(is_new_run: np.ndarray) -> np.ndarray:
    """is_new_run[i] True if frame i starts a new run (is_new_run[0] must be True). Returns run start indices."""
    return np.flatnonzero(is_new_run)

def from_runs(emb: np.ndarray, ts: np.ndarray, duration: float, starts: np.ndarray) -> Selection:
    """Merge contiguous runs beginning at `starts` (sorted, starts[0]==0)."""
    n = len(emb)
    starts = np.asarray(starts, dtype=int)
    ends = np.r_[starts[1:], n]                            # exclusive
    counts = (ends - starts).astype(np.float32)
    mean = np.add.reduceat(emb, starts, axis=0) / counts[:, None]
    run_t0 = ts[starts].astype(np.float64)
    run_t1 = ts[ends - 1].astype(np.float64)
    span_s = run_t0.copy(); span_s[0] = 0.0
    span_e = np.r_[run_t0[1:], float(duration)]
    return Selection(ts=(run_t0 + run_t1) / 2.0, emb=l2n(mean), span_start=span_s, span_end=span_e, n_input=n,
                     meta={"starts": starts, "ends": ends, "run_t0": run_t0, "run_t1": run_t1, "counts": counts})
