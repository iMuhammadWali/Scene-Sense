"""Hybrid: StreamMem run-merging, then MaxInfo-chunk over the merged representatives."""
import numpy as np
from ..selection import Selection, from_runs
from .maxinfo import maxinfo_chunk_indices

def hybrid(emb, ts, duration, delta=0.95, chunk_s=4.0, R=2, tol=0.3):
    sm = from_runs(emb, ts, duration, _starts(emb, delta))
    keep = np.unique(maxinfo_chunk_indices(sm.emb, sm.ts, chunk_s, R, tol))   # chunking on run-midpoint timestamps
    # a kept representative keeps its run span; dropped neighbours' spans are absorbed by the midpoint of the gap
    run_start = sm.span_start.copy(); run_start[0] = ts[0]
    run_end = sm.span_end.copy()
    k_start, k_end = run_start[keep], run_end[keep]
    gap_mid_prev = (k_end[:-1] + k_start[1:]) / 2.0
    span_s = np.r_[0.0, gap_mid_prev]
    span_e = np.r_[gap_mid_prev, float(duration)]
    # extend each kept span to include its own run (it always does, since gap midpoints lie outside the run)
    return Selection(ts=sm.ts[keep], emb=sm.emb[keep], span_start=span_s, span_end=span_e, n_input=len(emb),
                     meta={"n_runs": sm.n_kept})

def _starts(emb, delta):
    sims = np.einsum("ij,ij->i", emb[1:], emb[:-1])
    return np.flatnonzero(np.r_[True, sims <= delta])
