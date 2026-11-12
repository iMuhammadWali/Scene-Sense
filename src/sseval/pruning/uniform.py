"""Uniform sampling baseline."""
import numpy as np
from ..selection import from_indices

def full(emb, ts, duration):
    """Keep every 5 FPS frame (upper-bound reference)."""
    return from_indices(emb, ts, duration, np.arange(len(emb)))

def uniform(emb, ts, duration, fraction=None, step=None):
    """Keep `fraction` of frames evenly spaced (linspace incl. both ends), or every `step`-th frame."""
    n = len(emb)
    if step is not None:
        idx = np.arange(0, n, int(step))
    else:
        k = int(max(1, min(n, round(float(fraction) * n))))
        idx = np.round(np.linspace(0, n - 1, k)).astype(int)
    return from_indices(emb, ts, duration, idx)
