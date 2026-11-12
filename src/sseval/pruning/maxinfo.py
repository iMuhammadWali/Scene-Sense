"""MaxInfo (Li et al., WACV 2026, arXiv 2502.03183) - our re-implementation for retrieval indexing.

Original code (github.com/FusionBrainLab/MaxInfo, notebooks): frame embeddings Q (n x d) -> SVD, keep the first R left singular
vectors Us = U[:, :R] (n x R) -> rectangular MaxVol (Mikhalev & Oseledets, maxvolpy.rect_maxvol) -> selected row indices.
NOTE (deviation, see docs): the released notebooks call `rect_maxvol(M, tol=0.1)` and IGNORE their own `tol` argument; we honour `tol`.
maxvolpy does not build in this environment, so `rect_maxvol` is implemented here in numpy (tested in tests/test_maxvol.py).

rect_maxvol semantics (as in maxvolpy): start from a square maxvol (R rows), then greedily add the row whose coefficient-row norm
||C_i|| = sqrt(a_i^T (A_S^T A_S)^-1 a_i) is largest, until every unselected row has ||C_i|| <= tol.  For orthonormal U_R this
makes K ~ R / tol^2 independent of video length (so on 30 s clips small tol saturates at "keep everything").
"""
from __future__ import annotations
import numpy as np
from ..selection import Selection, from_indices, midpoint_spans

# ------------------------------------------------------------------ MaxVol primitives
def _greedy_init_rows(A: np.ndarray) -> np.ndarray:
    """Pick r well-conditioned rows by greedy pivoted Gram-Schmidt (== complete-pivot LU on rows). A: [n, r], n >= r."""
    n, r = A.shape
    W = A.astype(np.float64).copy()
    piv = np.empty(r, dtype=int)
    avail = np.ones(n, dtype=bool)
    for k in range(r):
        norms = np.einsum("ij,ij->i", W, W)
        norms[~avail] = -1.0
        i = int(np.argmax(norms))
        if norms[i] <= 1e-24:                               # rank-deficient: fill with any unused rows
            rest = np.flatnonzero(avail)[: r - k]
            piv[k:] = rest
            avail[rest] = False
            break
        piv[k] = i; avail[i] = False
        q = W[i] / np.sqrt(norms[i])
        W -= np.outer(W @ q, q)                             # remove the q-component from every row
    return piv

def maxvol_square(A: np.ndarray, tol: float = 1.0, max_iters: int = 200) -> np.ndarray:
    """Square maxvol: rows `idx` (r of them) with |A @ inv(A[idx])|_ij <= 1 + tol (Goreinov et al.)."""
    n, r = A.shape
    A = A.astype(np.float64)
    idx = _greedy_init_rows(A) if n > r else np.arange(n)
    if n <= r:
        return idx
    for _ in range(max_iters):
        B = A[idx]
        try:
            C = A @ np.linalg.inv(B)
        except np.linalg.LinAlgError:
            break
        # fresh C each pass is O(n r^2); rank-1 updates inside the pass keep it cheap
        for _ in range(r * 4):
            i, j = np.unravel_index(np.argmax(np.abs(C)), C.shape)
            if abs(C[i, j]) <= 1.0 + tol:
                return idx
            # swap row j of the selection for row i: update C by a rank-1 formula
            cj = C[:, j].copy(); ri = C[i].copy(); ri[j] -= 1.0
            C -= np.outer(cj, ri) / C[i, j]
            idx[j] = i
        # periodically recompute to limit drift
    return idx

def rect_maxvol(A: np.ndarray, tol: float = 0.3, min_k: int | None = None, max_k: int | None = None) -> np.ndarray:
    """Rectangular maxvol. A: [n, r]. Returns selected row indices (unsorted, unique), r <= K <= max_k."""
    A = np.asarray(A, dtype=np.float64)
    n, r = A.shape
    if n <= r:
        return np.arange(n)
    max_k = n if max_k is None else int(min(max(max_k, r), n))
    min_k = r if min_k is None else int(min(max(min_k, r), max_k))
    idx = list(maxvol_square(A, tol=1.0))
    chosen = np.zeros(n, dtype=bool); chosen[idx] = True
    G = A[idx].T @ A[idx]
    Ginv = np.linalg.pinv(G)
    # l_i = ||C_i||^2 = a_i^T G^-1 a_i
    W = A @ Ginv                                            # [n, r]
    l = np.einsum("ij,ij->i", W, A)
    l[chosen] = -np.inf
    while len(idx) < max_k:
        i = int(np.argmax(l))
        if len(idx) >= min_k and l[i] <= tol * tol:
            break
        if not np.isfinite(l[i]):
            break
        # Sherman-Morrison update of Ginv after adding row a_i
        a = A[i]
        v = W[i]                                            # = G^-1 a_i  (Ginv symmetric)
        denom = 1.0 + float(a @ v)
        Av = A @ v                                          # a_j^T G^-1 a_i for all j
        l = l - Av * Av / denom
        Ginv = Ginv - np.outer(v, v) / denom
        W = W - np.outer(Av, v) / denom                     # W = A @ Ginv
        idx.append(i); chosen[i] = True; l[i] = -np.inf
        l[chosen] = -np.inf
    return np.array(idx, dtype=int)

# ------------------------------------------------------------------ MaxInfo selection
def svd_reduce(Q: np.ndarray, R: int) -> np.ndarray:
    """Left singular vectors U[:, :R] of Q [n, d], R clamped to min(n, d, numerical rank)."""
    n, d = Q.shape
    R = int(max(1, min(R, n, d)))
    if n <= d or n <= 1500:
        U, s, _ = np.linalg.svd(Q.astype(np.float64), full_matrices=False)
    else:                                                    # long pools: eigh of the d x d Gram matrix (same U up to sign)
        Qd = Q.astype(np.float64)
        w, V = np.linalg.eigh(Qd.T @ Qd)
        o = np.argsort(w)[::-1]
        w, V = w[o], V[:, o]
        s = np.sqrt(np.maximum(w, 0))
        keep = max(1, min(R, int((s > 1e-9 * max(s[0], 1e-30)).sum())))
        U = (Qd @ V[:, :keep]) / s[:keep]
        s = s[:keep]
    rank = int((s > 1e-9 * max(s[0], 1e-30)).sum())
    return U[:, : max(1, min(R, rank))]

def maxinfo_indices(Q: np.ndarray, R: int, tol: float, max_k: int | None = None) -> np.ndarray:
    """Sorted row indices of Q selected by SVD(R) + rect_maxvol(tol)."""
    n = len(Q)
    if n <= 1:
        return np.arange(n)
    U = svd_reduce(Q, min(R, n))
    if n <= U.shape[1]:
        return np.arange(n)
    return np.sort(rect_maxvol(U, tol=tol, max_k=max_k))

def maxinfo_fast(emb, ts, duration, R=8, tol=0.3):
    """MaxInfo-Fast: pool = 1 FPS subsample (every 5th 5-FPS frame)."""
    pool = np.arange(0, len(emb), 5)
    sel = pool[maxinfo_indices(emb[pool], R, tol)]
    return from_indices(emb, ts, duration, sel)

def maxinfo_slow(emb, ts, duration, R=8, tol=0.3, cap_frac=0.10):
    """MaxInfo-Slow: pool = all 5 FPS frames; if more than cap = ceil(cap_frac * n) are selected, downsample uniformly to cap."""
    n = len(emb)
    sel = maxinfo_indices(emb, R, tol)
    cap = int(max(1, np.ceil(cap_frac * n)))
    if len(sel) > cap:
        sel = sel[np.round(np.linspace(0, len(sel) - 1, cap)).astype(int)]
    return from_indices(emb, ts, duration, sel)

def chunk_ids(ts: np.ndarray, chunk_s: float) -> np.ndarray:
    return np.floor(np.asarray(ts) / float(chunk_s)).astype(int)

def maxinfo_chunk_indices(emb, ts, chunk_s, R, tol) -> np.ndarray:
    """Per-chunk MaxInfo (chunks of chunk_s seconds), indices concatenated in time order."""
    cid = chunk_ids(ts, chunk_s)
    bounds = np.flatnonzero(np.r_[True, np.diff(cid) != 0, True])
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        out.append(a + maxinfo_indices(emb[a:b], min(R, b - a), tol))
    return np.concatenate(out)

def maxinfo_chunk(emb, ts, duration, chunk_s=4.0, R=2, tol=0.3):
    return from_indices(emb, ts, duration, maxinfo_chunk_indices(emb, ts, chunk_s, R, tol))
