import numpy as np, pytest
from sseval import data
from sseval.selection import midpoint_spans, from_runs
from sseval.pruning import select, METHODS

def synth(n=150, seed=0, d=512, nseg=6):
    """Piecewise-constant scene embeddings + noise, ~5 FPS irregular timestamps."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(nseg, d))
    seg = np.sort(rng.integers(0, nseg, n)) if False else np.repeat(np.arange(nseg), int(np.ceil(n / nseg)))[:n]
    e = centers[seg] + 0.15 * rng.normal(size=(n, d))
    e = (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float32)
    ts = np.cumsum(rng.uniform(0.17, 0.23, n)) - 0.2
    ts[0] = 0.0
    return e, ts, float(ts[-1] + 0.2)

CONFIGS = [("full_5fps", {}), ("uniform", {"fraction": 0.1}), ("uniform", {"fraction": 1.0}), ("uniform", {"step": 5}),
           ("cos_last_drop", {"tau": 0.9}), ("cos_consec_drop", {"tau": 0.9}), ("streammem_merge", {"delta": 0.9}),
           ("maxinfo_fast", {"R": 4, "tol": 0.3}), ("maxinfo_slow", {"R": 8, "tol": 0.15, "cap_frac": 0.1}),
           ("maxinfo_chunk", {"chunk_s": 4, "R": 2, "tol": 0.6}), ("hybrid", {"delta": 0.9, "chunk_s": 4, "R": 2, "tol": 0.6})]

@pytest.mark.parametrize("method,params", CONFIGS)
def test_selection_invariants(method, params):
    e, ts, dur = synth()
    s = select(method, e, ts, dur, **params)
    n = s.n_kept
    assert 1 <= n <= len(e)
    assert np.all(np.diff(s.ts) > 0), "item timestamps strictly increasing"
    assert s.ts.min() >= 0 and s.ts.max() <= dur
    assert s.emb.shape == (n, 512) and np.allclose(np.linalg.norm(s.emb, axis=1), 1, atol=1e-4)
    # spans tile [0, duration] with no gaps/overlaps and each item lies inside its own span
    assert s.span_start[0] == 0 and s.span_end[-1] == pytest.approx(dur)
    assert np.allclose(s.span_start[1:], s.span_end[:-1])
    assert np.all(s.span_end > s.span_start)
    assert np.all((s.ts >= s.span_start - 1e-9) & (s.ts <= s.span_end + 1e-9))
    assert s.select_s >= 0
    if "idx" in s.meta:
        i = s.meta["idx"]; assert np.all(np.diff(i) > 0) and i.min() >= 0 and i.max() < len(e)

def test_uniform_one_equals_full():
    e, ts, dur = synth()
    a, b = select("full_5fps", e, ts, dur), select("uniform", e, ts, dur, fraction=1.0)
    assert np.array_equal(a.meta["idx"], b.meta["idx"]) and np.array_equal(a.span_start, b.span_start)

def test_cos_last_drop_is_teams_prune_frames():
    from sseval import config as C; C.setup_team()
    from app.pipeline.frame_extraction import CandidateFrame
    from app.pipeline.pruning import prune_frames
    e, ts, dur = synth(seed=3)
    for tau in [0.8, 0.9, 0.95, 0.98]:
        cands = [CandidateFrame(i, float(ts[i]), i) for i in range(len(e))]
        kept, kemb, _ = prune_frames(cands, e, tau)
        s = select("cos_last_drop", e, ts, dur, tau=tau)
        assert [c.frame_id for c in kept] == list(s.meta["idx"])
        assert np.array_equal(kemb, s.emb)

def test_cos_tau_extremes_synthetic():
    e, ts, dur = synth(seed=5)
    for m in ["cos_last_drop", "cos_consec_drop"]:
        assert select(m, e, ts, dur, tau=1.0 + 1e-5).n_kept == len(e)     # nothing can exceed cos 1 -> nothing dropped
        assert select(m, e, ts, dur, tau=-1.0).n_kept == 1                # everything is "similar" -> only the first kept
    k = [select("cos_last_drop", e, ts, dur, tau=t).n_kept for t in (0.8, 0.9, 0.95, 0.98)]
    assert k == sorted(k)                                                  # higher tau keeps more

def test_streammem_run_semantics():
    e, ts, dur = synth(seed=2)
    s = select("streammem_merge", e, ts, dur, delta=0.9)
    starts, ends, cnt = s.meta["starts"], s.meta["ends"], s.meta["counts"]
    assert cnt.sum() == len(e) and starts[0] == 0 and ends[-1] == len(e)        # runs partition the frames
    sims = np.einsum("ij,ij->i", e[1:], e[:-1])
    for a, b in zip(starts, ends):
        assert np.all(sims[a:b - 1] > 0.9)                                      # inside a run: consecutive cos > delta
    assert np.all(sims[starts[1:] - 1] <= 0.9)                                  # run boundary: cos <= delta
    mean = e[starts[0]:ends[0]].mean(0); mean /= np.linalg.norm(mean)
    assert np.allclose(s.emb[0], mean, atol=1e-5)

def test_merge_spans_match_run_extent():
    e, ts, dur = synth(seed=7)
    s = select("streammem_merge", e, ts, dur, delta=0.85)
    for k, (a, b) in enumerate(zip(s.meta["starts"], s.meta["ends"])):
        if k > 0: assert s.span_start[k] == pytest.approx(ts[a])
        assert s.span_end[k] == pytest.approx(ts[b] if b < len(ts) else dur)

def test_maxinfo_slow_cap_and_fast_pool():
    e, ts, dur = synth()
    s = select("maxinfo_slow", e, ts, dur, R=8, tol=0.05, cap_frac=0.1)
    assert s.n_kept <= int(np.ceil(0.1 * len(e)))
    f = select("maxinfo_fast", e, ts, dur, R=8, tol=0.05)
    assert set(f.meta["idx"]) <= set(range(0, len(e), 5))                       # only 1 FPS pool frames

def test_maxinfo_chunk_covers_every_chunk():
    e, ts, dur = synth()
    s = select("maxinfo_chunk", e, ts, dur, chunk_s=4, R=1, tol=1.0)
    chunks = np.floor(ts[s.meta["idx"]] / 4).astype(int)
    assert set(chunks) == set(np.floor(ts / 4).astype(int))

def test_midpoint_spans_rule():
    s, e = midpoint_spans(np.array([1.0, 3.0, 7.0]), 10.0)
    assert list(s) == [0.0, 2.0, 5.0] and list(e) == [2.0, 5.0, 10.0]

def test_hybrid_kept_items_cover_their_runs():
    e, ts, dur = synth(seed=4)
    h = select("hybrid", e, ts, dur, delta=0.9, chunk_s=4, R=1, tol=1.0)
    sm = select("streammem_merge", e, ts, dur, delta=0.9)
    assert h.n_kept <= sm.n_kept
    for t, a, b in zip(h.ts, h.span_start, h.span_end):
        k = np.argmin(np.abs(sm.ts - t))
        assert a <= sm.meta["run_t0"][k] + 1e-9 and b >= sm.meta["run_t1"][k] - 1e-9

def test_real_video_all_methods_nonempty():
    df = data.load_annotations(); vids = data.select_videos(df, 2)
    if not vids: pytest.skip("no cached embeddings")
    e, ts, dur = data.load_video(vids[0])
    for m, p in CONFIGS:
        s = select(m, e, ts, dur, **p); assert s.n_kept >= 1 and np.isfinite(s.emb).all()

def test_cos_tau_one_keeps_all_on_real_video():
    df = data.load_annotations(); vids = data.select_videos(df, 5)
    if not vids: pytest.skip("no cached embeddings")
    tot = kept = 0
    for v in vids:
        e, ts, dur = data.load_video(v)
        # tau slightly above 1 removes float32 rounding artefacts (identical frames can give cos = 1.0000001)
        assert select("cos_last_drop", e, ts, dur, tau=1.0 + 1e-5).n_kept == len(e)
        assert select("cos_consec_drop", e, ts, dur, tau=1.0 + 1e-5).n_kept == len(e)
