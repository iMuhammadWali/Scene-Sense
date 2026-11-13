"""Tests for E9: yes/no probability extraction, token-id variants, z-fusion, CV weight choice, NONE threshold."""
import importlib.util
from pathlib import Path
import numpy as np, pytest
spec = importlib.util.spec_from_file_location("e9_run", Path(__file__).resolve().parents[1] / "experiments" / "e9_vlm_scoring" / "run.py")
e9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e9)

def test_yes_no_prob_is_two_way_softmax():
    lg = np.full(20, -5.0); lg[3], lg[7] = 2.0, 1.0           # yes=3, no=7
    p, d = e9.yes_no_prob(lg, [3], [7])
    assert d == pytest.approx(1.0) and p == pytest.approx(np.exp(2) / (np.exp(2) + np.exp(1)))
    assert 1 / (1 + np.exp(-d)) == pytest.approx(p)

def test_yes_no_prob_ignores_other_tokens_and_takes_max_over_variants():
    lg = np.full(20, 50.0)                                     # huge logits elsewhere must not matter
    lg[[1, 2]] = [0.0, 3.0]; lg[[5, 6]] = [4.0, -9.0]          # yes variants 1,2 -> 3 ; no variants 5,6 -> 4
    p, d = e9.yes_no_prob(lg, [1, 2], [5, 6])
    assert d == pytest.approx(-1.0) and p == pytest.approx(1 / (1 + np.e))

def test_yes_no_prob_symmetry_and_extremes():
    lg = np.zeros(10); lg[0], lg[1] = 2.5, -1.0
    p1, _ = e9.yes_no_prob(lg, [0], [1]); p2, _ = e9.yes_no_prob(lg, [1], [0])
    assert p1 + p2 == pytest.approx(1.0)
    lg2 = np.zeros(10); lg2[0] = 1e4; lg2[1] = -1e4            # no overflow
    assert e9.yes_no_prob(lg2, [0], [1])[0] == 1.0 and e9.yes_no_prob(lg2, [1], [0])[0] == 0.0
    assert e9.yes_no_prob(np.zeros(4), [0], [1])[0] == pytest.approx(0.5)

def test_single_token_ids_keeps_one_token_variants_only():
    class Tok:
        table = {"Yes": [10], " Yes": [11], "yes": [12], " yes": [12], "YES": [1, 2], " YES": [13]}
        def encode(self, w, add_special_tokens=False): return self.table[w]
    assert e9.single_token_ids(Tok(), e9.YES_WORDS) == [10, 11, 12, 13]     # dup id and multi-token variant dropped

def test_zscore_rows_and_constant_row():
    z = e9.zscore_rows(np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]))
    assert z[0].mean() == pytest.approx(0) and z[0].std() == pytest.approx(1) and (z[1] == 0).all()

def test_fuse_choice_limits():
    clip = np.array([[0.30, 0.25, 0.20], [0.20, 0.30, 0.10]]); vlm = np.array([[0.0, 5.0, 1.0], [2.0, 1.0, 0.0]])
    assert e9.fuse_choice(clip, vlm, 0.0).tolist() == [0, 1]                  # w=0 -> CLIP argmax
    assert e9.fuse_choice(clip, vlm, 1e6).tolist() == [1, 0]                  # w huge -> VLM argmax

def test_pick_w_cv_pooled_is_heldout_and_deterministic():
    rng = np.random.default_rng(0); n = 40
    clip = rng.normal(size=(n, 5)); vlm = rng.normal(size=(n, 5)); hit = np.zeros((n, 5), bool); hit[np.arange(n), np.argmax(vlm, 1)] = True
    h1, w1 = e9.pick_w_cv(clip, vlm, hit, e9.W_GRID, seed=3); h2, w2 = e9.pick_w_cv(clip, vlm, hit, e9.W_GRID, seed=3)
    assert (h1 == h2).all() and w1 == w2 and len(h1) == n and set(w1) <= set(e9.W_GRID)
    assert h1.mean() > 0.6 and w1 == [2.0, 2.0]                               # VLM is the oracle here -> largest w wins in both folds

def test_none_threshold_gives_requested_rate():
    s = np.arange(100, dtype=float)
    th = e9.none_threshold(s, 0.05)
    assert (s <= th).mean() == pytest.approx(0.05)

def test_missing_candidate_nan_never_chosen():
    clip = np.array([[0.1, 0.2, 0.3, np.nan, np.nan]]); vlm = np.array([[1.0, 0.0, -1.0, np.nan, np.nan]])
    z = e9.zscore_rows(clip); assert np.isnan(z[0, 3:]).all() and z[0, :3].mean() == pytest.approx(0)
    for w in (0.0, 1.0, 1e6): assert e9.fuse_choice(clip, vlm, w)[0] in (0, 1, 2)
    assert e9.argmax_nan(np.array([[np.nan, 1.0, 2.0]]))[0] == 2
