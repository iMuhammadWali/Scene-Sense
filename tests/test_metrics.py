import numpy as np
from sseval.metrics import temporal_iou, iou_arrays, auroc, bootstrap_indices, bootstrap_ci

def test_iou_basic():
    assert temporal_iou((0, 10), (0, 10)) == 1.0
    assert temporal_iou((0, 10), (10, 20)) == 0.0
    assert abs(temporal_iou((0, 10), (5, 15)) - 5 / 15) < 1e-12
    assert abs(temporal_iou((2, 4), (0, 10)) - 0.2) < 1e-12
    assert temporal_iou((0, 5), (6, 9)) == 0.0
    assert temporal_iou((3, 3), (3, 3)) == 0.0

def test_iou_vectorised_matches_scalar():
    rng = np.random.default_rng(0)
    a = np.sort(rng.uniform(0, 30, (200, 2)), axis=1); b = np.sort(rng.uniform(0, 30, (200, 2)), axis=1)
    v = iou_arrays(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    s = np.array([temporal_iou(x, y) for x, y in zip(a, b)])
    assert np.allclose(v, s)

def test_auroc():
    assert auroc([3, 4], [1, 2]) == 1.0
    assert auroc([1, 2], [3, 4]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    rng = np.random.default_rng(1); p = rng.normal(1, 1, 500); n = rng.normal(0, 1, 700)
    brute = np.mean([(x > y) + 0.5 * (x == y) for x in p for y in n])
    assert abs(auroc(p, n) - brute) < 1e-9

def test_bootstrap_ci_contains_mean():
    x = (np.random.default_rng(0).random(500) < 0.4).astype(float)
    lo, hi = bootstrap_ci(x, bootstrap_indices(500))
    assert lo < x.mean() < hi and hi - lo < 0.15
