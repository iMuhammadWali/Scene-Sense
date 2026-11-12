import numpy as np
from sseval.pruning.maxinfo import maxvol_square, rect_maxvol, svd_reduce, maxinfo_indices

def logvol(B): return np.linalg.slogdet(B)[1]

def test_square_maxvol_dominant_submatrix():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(200, 8))
    idx = maxvol_square(A, tol=0.05)
    assert len(set(idx)) == 8
    C = A @ np.linalg.inv(A[idx])
    assert np.abs(C).max() <= 1.05 + 1e-9                  # defining property of a (near-)maximum-volume submatrix
    # volume beats random 8-row submatrices
    rv = max(logvol(A[rng.choice(200, 8, replace=False)]) for _ in range(200))
    assert logvol(A[idx]) > rv

def test_rect_maxvol_stopping_rule_and_monotonic_tol():
    rng = np.random.default_rng(1)
    A = np.linalg.qr(rng.normal(size=(300, 6)))[0]
    prev = 0
    for tol in [1.0, 0.6, 0.3, 0.15]:
        idx = rect_maxvol(A, tol=tol)
        assert len(idx) == len(set(idx)) and len(idx) >= 6
        S = A[idx]; Ginv = np.linalg.inv(S.T @ S)
        rest = np.setdiff1d(np.arange(300), idx)
        l = np.einsum("ij,jk,ik->i", A[rest], Ginv, A[rest])      # squared coefficient-row norms
        assert l.max() <= tol * tol + 1e-9                         # every unselected row is covered within tol
        assert len(idx) >= prev; prev = len(idx)                   # smaller tol => more rows
    # K ~ R / tol^2 for orthonormal columns
    assert 0.4 * 6 / 0.09 < len(rect_maxvol(A, tol=0.3)) < 2.5 * 6 / 0.09

def test_rect_maxvol_equals_C_row_norm_definition():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(60, 5))
    idx = rect_maxvol(A, tol=0.5)
    C = A @ np.linalg.pinv(A[idx])                                  # A = C A_S, minimal-norm C
    assert np.allclose(C @ A[idx], A, atol=1e-8)
    rest = np.setdiff1d(np.arange(60), idx)
    assert np.linalg.norm(C[rest], axis=1).max() <= 0.5 + 1e-9

def test_maxk_and_small_inputs():
    A = np.random.default_rng(3).normal(size=(50, 4))
    assert len(rect_maxvol(A, tol=0.01, max_k=10)) == 10
    assert list(rect_maxvol(A[:3], tol=0.3)) == [0, 1, 2]
    assert len(maxinfo_indices(np.ones((1, 512), dtype=np.float32), 8, 0.3)) == 1

def test_svd_reduce_orthonormal_and_gram_path_agree():
    rng = np.random.default_rng(4)
    Q = rng.normal(size=(2000, 64)).astype(np.float32)
    U = svd_reduce(Q, 5)
    assert np.allclose(U.T @ U, np.eye(5), atol=1e-6)
    P1 = U @ U.T
    U2 = np.linalg.svd(Q.astype(np.float64), full_matrices=False)[0][:, :5]
    assert np.allclose(P1, U2 @ U2.T, atol=1e-6)                    # same subspace
