import numpy as np
from sseval import localize as L

def test_smooth_constant_and_width():
    s = np.ones(20); assert np.allclose(L.smooth(s, 1.0), 1.0)
    x = np.zeros(21); x[10] = 5.0
    y = L.smooth(x, 1.0)                       # k=5 -> peak 1.0 spread over 5 cells
    assert np.isclose(y.sum(), 5.0) and np.isclose(y[10], 1.0) and y[7] == 0
    assert np.array_equal(L.smooth(x, 0.2), x)

def test_fixed_span_shift_and_clip():
    assert L.fixed_span(1.0, 8.0, 30.0) == (0.0, 8.0)
    assert L.fixed_span(29.0, 8.0, 30.0) == (22.0, 30.0)
    assert L.fixed_span(5.0, 8.0, 6.0) == (0.0, 6.0)
    assert L.fixed_span(15.0, 8.0, 30.0) == (11.0, 19.0)

def test_grow_span():
    s = np.zeros(50); s[20:30] = 1.0; s[24] = 1.2
    a, b = L.grow_span(s, 24, 0.5, duration=10.0)
    assert (a, b) == (4.0, 6.0)

def test_pruned_curve_piecewise_constant():
    span_end = np.array([1.0, 2.0, 3.0]); cen = L.make_grid(3.0)
    ids = L.item_ids(span_end, cen); assert ids[0] == 0 and ids[-1] == 2 and len(cen) == 15
    emb = np.eye(3, dtype=np.float32); q = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    s = L.curve(emb, ids, q); assert s[:5].sum() == 0 and s[5:10].min() == 1 and s[10:].sum() == 0

def test_contrast_finds_planted_moment():
    s = np.zeros(150); s[50:90] = 1.0          # 10..18 s of a 30 s video
    st, en, sc = L.contrast_spans(s, 0.0, 30.0)
    i = np.argmax(sc); assert abs(st[i] - 10) <= 0.5 and abs(en[i] - 18) <= 0.5
    top = L.nms_topk(st, en, sc, 5, 0.5)
    assert len(top) >= 2 and all(L.iou(top[0], t) < 0.5 for t in top[1:])
    assert (en - st).max() <= 25.0 + 1e-9 and (en - st).min() >= 2.0 - 1e-9

def test_lambda_prefers_shorter():
    s = np.zeros(150); s[50:90] = 1.0; s[40:100] += 0.05
    st, en, sc = L.contrast_spans(s, 0.5, 30.0)
    i = np.argmax(sc); assert en[i] - st[i] <= 8.6
