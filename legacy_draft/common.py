"""Shared helpers for the Scene Sense evaluation harness.

Reuses the team's implementation (prune_frames, build_faiss_index, search,
group_temporally, CLIPEncoder) from /home/claude/fyp/backend (not modified).
"""
import os, sys, time, glob, logging
os.environ.setdefault("SCENESENSE_DATA_DIR", "/home/claude/work/data")
os.environ.setdefault("SCENESENSE_MODELS_DIR", "/home/claude/work/models")
os.environ.setdefault("SCENESENSE_DEVICE", "cpu")
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/fyp/backend")

import numpy as np, pandas as pd
import faiss
faiss.omp_set_num_threads(1)   # single-thread: stable per-query latency, leaves CPU to encode_all.py

from app.pipeline.frame_extraction import CandidateFrame
from app.pipeline.pruning import prune_frames
from app.pipeline.faiss_index import build_faiss_index, search
from app.pipeline.temporal import group_temporally
logging.disable(logging.INFO)   # silence the team's per-call INFO logs

ANN = "/home/claude/datasets/charades_sta/test.parquet"
EMB_DIR = "/home/claude/work/emb"
EXP = "/home/claude/experiments"
RES = f"{EXP}/results"
QCACHE = f"{EXP}/cache/query_emb.npz"
os.makedirs(RES, exist_ok=True)

DIVERSITY_LABEL = "Diversity (k-center, MaxInfo-style)"  # NOT the real MaxInfo algorithm
THRESHOLDS_ALL = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97]

# ---------------------------------------------------------------- metrics
def temporal_iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0

def selftest_iou():
    assert temporal_iou((0, 10), (0, 10)) == 1.0
    assert temporal_iou((0, 10), (10, 20)) == 0.0
    assert abs(temporal_iou((0, 10), (5, 15)) - 5 / 15) < 1e-9
    assert abs(temporal_iou((2, 4), (0, 10)) - 0.2) < 1e-9
    assert temporal_iou((0, 5), (6, 9)) == 0.0
    assert temporal_iou((3, 3), (3, 3)) == 0.0   # degenerate -> 0, no div-by-zero

# ---------------------------------------------------------------- data
def load_queries():
    df = pd.read_parquet(ANN).reset_index(drop=True)
    df["qid"] = np.arange(len(df))
    df["gs"] = df["timestamp"].map(lambda x: float(x[0]))
    df["ge"] = df["timestamp"].map(lambda x: float(x[1]))
    df["vid"] = df["video"].str.replace(".mp4", "", regex=False)
    return df

def load_query_emb():
    if not os.path.exists(QCACHE):
        raise SystemExit("Run encode_queries.py first")
    return np.load(QCACHE)["emb"]

def load_videos(max_videos=None):
    """Return {vid: (emb, ts, duration)} for every complete npz that exists."""
    out = {}
    files = sorted(f for f in glob.glob(f"{EMB_DIR}/*.npz") if not f.endswith(".tmp.npz"))
    for f in files:
        try:
            d = np.load(f)
            emb, ts, dur = d["emb"].astype(np.float32), d["ts"].astype(np.float64), float(d["duration"])
        except Exception as e:
            print("skip unreadable", f, e); continue
        if len(emb) == 0: continue
        out[os.path.basename(f)[:-4]] = (emb, ts, dur)
        if max_videos and len(out) >= max_videos: break
    return out

# ---------------------------------------------------------------- selection methods
def sel_full(n): return np.arange(n)
def sel_fixed(n, step): return np.arange(0, n, step)

def sel_threshold(emb, ts, T):
    cands = [CandidateFrame(i, float(ts[i]), i) for i in range(len(emb))]
    kept, _, _ = prune_frames(cands, emb, T)      # THE TEAM'S IMPLEMENTATION
    return np.array([c.frame_id for c in kept], dtype=int)

def sel_uniform(n, k):
    k = max(1, min(k, n))
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))

def sel_kcenter(emb, k):
    """Greedy farthest-point (k-center) on cosine distance, start at frame 0.
    Running min-distance array => O(N*k*d). Diversity baseline, NOT real MaxInfo."""
    n = len(emb); k = max(1, min(k, n))
    if k >= n: return np.arange(n)
    e = emb.astype(np.float32)
    sel = [0]
    mind = 1.0 - e @ e[0]
    mind[0] = -np.inf
    for _ in range(k - 1):
        j = int(np.argmax(mind))
        sel.append(j)
        mind = np.minimum(mind, 1.0 - e @ e[j])
        mind[j] = -np.inf
    return np.array(sorted(sel), dtype=int)

def build_selections(emb, ts, fixed=(("fixed_1fps", 5), ("fixed_0.5fps", 10)),
                     thresholds=THRESHOLDS_ALL, matched=THRESHOLDS_ALL):
    """-> ordered dict name -> (indices, selection_seconds)."""
    n = len(emb); out = {}
    t = time.perf_counter(); s = sel_full(n); out["full_5fps"] = (s, time.perf_counter() - t)
    for name, step in fixed:
        t = time.perf_counter(); s = sel_fixed(n, step); out[name] = (s, time.perf_counter() - t)
    thr = {}
    for T in thresholds:
        t = time.perf_counter(); s = sel_threshold(emb, ts, T); dt = time.perf_counter() - t
        out[f"threshold_{T:.2f}"] = (s, dt); thr[T] = len(s)
    for T in matched:
        k = thr[T] if T in thr else len(sel_threshold(emb, ts, T))
        t = time.perf_counter(); s = sel_uniform(n, k); out[f"uniform_matched_{T:.2f}"] = (s, time.perf_counter() - t)
    for T in matched:
        k = thr[T] if T in thr else len(sel_threshold(emb, ts, T))
        t = time.perf_counter(); s = sel_kcenter(emb, k); out[f"diversity_matched_{T:.2f}"] = (s, time.perf_counter() - t)
    return out

def method_family(name):
    if name.startswith("threshold"): return "threshold"
    if name.startswith("uniform"): return "uniform"
    if name.startswith("diversity"): return "diversity"
    if name.startswith("fixed"): return "fixed"
    return name
def method_T(name):
    try: return float(name.rsplit("_", 1)[1])
    except Exception: return np.nan

# ---------------------------------------------------------------- retrieval + moment eval
def eval_selection(emb, ts, duration, sel, qs, top_k=5, gap=2.0):
    """qs: list of dicts(qid, qe, gs, ge). Searches ONLY this index (video or stream).
    Returns (per_query_rows, index_bytes)."""
    sel = np.asarray(sel, dtype=int)
    e = emb[sel]; t = ts[sel]
    frames = [{"frame_id": int(i), "timestamp": float(ts[i])} for i in sel]
    index, mapping, _ = build_faiss_index(e, frames)       # TEAM'S IMPLEMENTATION
    mids = (t[:-1] + t[1:]) / 2.0
    cov_s = np.concatenate([[0.0], mids]); cov_e = np.concatenate([mids, [duration]])
    rows = []
    for q in qs:
        t0 = time.perf_counter()
        hits = search(index, q["qe"], mapping, top_k)       # TEAM'S IMPLEMENTATION
        lat = (time.perf_counter() - t0) * 1000.0
        gs, ge = q["gs"], q["ge"]
        hit1 = gs <= hits[0]["timestamp"] <= ge
        hit5 = any(gs <= h["timestamp"] <= ge for h in hits)
        groups, _ = group_temporally(hits, gap)              # TEAM'S IMPLEMENTATION
        g = next(g for g in groups if any(f["rank"] == 1 for f in g["frames"]))
        idxs = [f["faiss_index"] for f in g["frames"]]
        pred = (float(cov_s[idxs].min()), float(cov_e[idxs].max()))
        rows.append(dict(qid=q["qid"], hit1=int(hit1), hit5=int(hit5),
                         iou=temporal_iou(pred, (gs, ge)), pred_s=pred[0], pred_e=pred[1], lat_ms=lat))
    return rows, int(len(sel) * 512 * 4)

def random_baseline(gs, ge, dur, seed=0, top_k=5):
    """Chance reference. Interval of the mean GT length placed uniformly in the video/stream;
    hit@1/hit@5 from 1 / 5 uniformly random timestamps. gs,ge,dur: arrays per query."""
    rng = np.random.default_rng(seed)
    gs, ge, dur = map(np.asarray, (gs, ge, dur))
    L = float((ge - gs).mean())
    rows = []
    for s, e, d in zip(gs, ge, dur):
        l = min(L, d); st = rng.uniform(0, d - l); pred = (st, st + l)
        r = rng.uniform(0, d, size=top_k)
        rows.append(dict(hit1=int(s <= r[0] <= e), hit5=int(((r >= s) & (r <= e)).any()),
                         iou=temporal_iou(pred, (s, e)), pred_s=pred[0], pred_e=pred[1], lat_ms=np.nan))
    return pd.DataFrame(rows), L

def metric_row(pq):
    n = len(pq)
    r = lambda t: float((pq["iou"] >= t).mean())
    se = lambda p: float(np.sqrt(p * (1 - p) / n))
    return dict(n_queries=n, hit1=pq["hit1"].mean(), hit5=pq["hit5"].mean(),
                R1_iou0_3=r(0.3), R1_iou0_5=r(0.5), R1_iou0_7=r(0.7), mIoU=pq["iou"].mean(),
                se_hit1=se(pq["hit1"].mean()), se_R1_iou0_5=se(r(0.5)),
                search_ms_per_query=pq["lat_ms"].mean() if pq["lat_ms"].notna().any() else np.nan)

def fmt_progress(i, n, t0, what="videos"):
    el = time.time() - t0
    print(f"  [{i}/{n} {what}] {el:.0f}s elapsed, ETA {el / max(i, 1) * (n - i):.0f}s", flush=True)
