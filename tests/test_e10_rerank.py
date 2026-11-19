"""Tests for E10 re-ranking utilities."""
import numpy as np, pytest
from sseval import rerank as R

def test_strip_person():
    assert R.strip_person("person closing the door.") == "closing the door."
    assert R.strip_person("a person sits in a chair.") == "sits in a chair."
    assert R.strip_person("A Person is cooking.") == "is cooking."
    assert R.strip_person("person") == "person"                       # nothing left -> unchanged
    assert R.strip_person("a persona runs") == "a persona runs"       # whole-word only
    assert R.strip_person("the person opens a laptop") == "opens a laptop"
    assert R.strip_person("someone is sneezing") == "is sneezing"
    assert R.strip_person("a dog and a person") == "a dog and a person"   # only leading

def test_frame_times_clip():
    assert R.frame_times(10.0, 30.0, (-1, 0, 1)) == [9.0, 10.0, 11.0]
    assert R.frame_times(0.4, 30.0, (-1, 0, 1)) == [0.0, 0.4, 1.4]
    assert R.frame_times(29.95, 30.0, (0, 1)) == [29.9, 29.9]

def test_zscore_rows_nan_and_constant():
    x = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [1.0, 3.0, np.nan]])
    z = R.zscore_rows(x)
    assert np.allclose(z[0], [-1.2247, 0, 1.2247], atol=1e-3)
    assert np.allclose(z[1], 0)
    assert z[2, 0] == pytest.approx(-1) and z[2, 1] == pytest.approx(1) and np.isnan(z[2, 2])

def test_row_ranks_ties_prefer_low_index_and_nan():
    r = R.row_ranks(np.array([[0.1, 0.9, 0.9, np.nan]]))
    assert r[0, 1] == 1 and r[0, 2] == 2 and r[0, 0] == 3 and np.isnan(r[0, 3])

def test_fuse_extremes_give_pure_orders():
    clip = np.array([[0.9, 0.8, 0.7], [0.5, 0.6, 0.4]])
    itm = np.array([[0.1, 0.9, 0.2], [0.3, 0.2, 0.8]])
    assert list(R.argmax_rows(R.fuse_z(clip, itm, 0.0))) == [0, 1]
    assert list(R.argmax_rows(R.fuse_z(clip, itm, 1e3))) == [1, 2]
    assert list(R.argmax_rows(R.fuse_rank(clip, itm, 0.0))) == [0, 1]
    assert list(R.argmax_rows(R.fuse_rank(clip, itm, 1e3))) == [1, 2]

def test_argmax_rows_nan_never_wins_ties_lowest():
    assert list(R.argmax_rows(np.array([[np.nan, 1.0, 1.0], [2.0, 2.0, np.nan]]))) == [1, 0]

def test_choose_w_picks_best_and_ties_smaller():
    clip = np.array([[0.9, 0.5, 0.1]] * 4)
    itm = np.array([[0.0, 1.0, 0.5]] * 4)
    hit = np.array([[0, 1, 0]] * 4, dtype=float)               # ITM's pick (col 1) is correct
    w, sc = R.choose_w(clip, itm, hit, [0.0, 0.1, 1.0, 5.0])
    assert sc[0.0] == 0.0 and sc[5.0] == 1.0 and w == 1.0 or w == 5.0
    assert sc[w] == 1.0 and w == min(k for k, v in sc.items() if v == 1.0)
    w2, _ = R.choose_w(clip, itm, hit, [0.0, 5.0], idx=[0, 1])
    assert w2 == 5.0

def test_candidate_score():
    assert R.candidate_score([1.0, 3.0, np.nan], "max") == 3.0
    assert R.candidate_score([1.0, 3.0, np.nan], "mean") == 2.0
    assert np.isnan(R.candidate_score([np.nan]))

def test_window_iou_matches_hand_calc_and_edges():
    t = np.array([[10.0, 1.0, np.nan]])
    iou = R.window_iou(t, [8.0], [16.0], [30.0], 8.0)            # window [6,14] vs GT [8,16]: inter 6 union 10
    assert iou[0, 0] == pytest.approx(0.6)
    assert iou[0, 1] == pytest.approx(0.0)                       # window shifted to [0,8], GT starts at 8
    assert np.isnan(iou[0, 2])

def test_auroc_ci_perfect_and_chance():
    a, lo, hi = R.auroc_ci([2, 3, 4, 5], [0, 1, 0.5, 1.5], B=200)
    assert a == 1.0 and lo == 1.0 and hi == 1.0
    rng = np.random.default_rng(0)
    a, lo, hi = R.auroc_ci(rng.normal(size=300), rng.normal(size=300), B=200)
    assert lo < 0.5 < hi
