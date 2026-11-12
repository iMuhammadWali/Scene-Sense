"""StreamMem input-frame filter, re-implemented for retrieval indexing.

StreamMem (query-agnostic KV-cache memory for streaming video) merges near-duplicate consecutive input frames:
consecutive-frame cosine > delta (paper: 0.95) joins the current run; runs are merged by simple averaging.
Here each run becomes ONE index entry (L2-renormalised mean embedding, covering the run's time extent).
Adapted from a VLM KV-memory method; this is our re-implementation for retrieval indexing.
"""
import numpy as np
from ..selection import from_runs

def streammem_merge(emb, ts, duration, delta=0.95):
    sims = np.einsum("ij,ij->i", emb[1:], emb[:-1])
    new_run = np.r_[True, sims <= delta]        # sim > delta  => joins current run
    return from_runs(emb, ts, duration, np.flatnonzero(new_run))
