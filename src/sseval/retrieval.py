"""The retrieval protocol shared by every method (E1/E1b/E3).

team's build_faiss_index -> team's search (top-5) -> team's group_temporally (gap 2 s)
-> predicted moment = union of the coverage spans of the group that contains the top-1 hit.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np, pandas as pd
from . import config as C
from .metrics import iou_arrays
from .selection import Selection

C.setup_team()
from app.pipeline.faiss_index import build_faiss_index, search          # noqa: E402  (team's code, unmodified)
from app.pipeline.temporal import group_temporally                      # noqa: E402

@dataclass
class Query:
    qid: int
    qe: np.ndarray
    gs: float
    ge: float

def build_index(sel: Selection):
    frames = [{"frame_id": i, "timestamp": float(t)} for i, t in enumerate(sel.ts)]
    index, mapping, _ = build_faiss_index(sel.emb, frames)
    return index, mapping

def index_bytes(sel: Selection) -> int:
    """Size of the stored vectors (float32, IndexFlatIP)."""
    return int(sel.n_kept * sel.emb.shape[1] * 4)

def predict(sel: Selection, index, mapping, qe: np.ndarray, top_k: int = C.TOP_K, gap: float = C.GROUP_GAP_S):
    """-> dict(hit_ts [top_k], pred_s, pred_e, top1_sim)."""
    hits = search(index, qe, mapping, top_k)                                # team's
    groups, _ = group_temporally(hits, gap)                                 # team's
    g = next(g for g in groups if any(f["rank"] == 1 for f in g["frames"]))
    ids = [f["faiss_index"] for f in g["frames"]]
    return dict(hit_ts=[h["timestamp"] for h in hits], pred_s=float(sel.span_start[ids].min()),
                pred_e=float(sel.span_end[ids].max()), top1_sim=hits[0]["similarity"])

def evaluate(sel: Selection, queries: list[Query], top_k: int = C.TOP_K, gap: float = C.GROUP_GAP_S,
             time_search: bool = False) -> pd.DataFrame:
    """Per-query results against ONE index. Columns: qid, hit1, hit5, iou, pred_s, pred_e[, lat_ms]."""
    index, mapping = build_index(sel)
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        p = predict(sel, index, mapping, q.qe, top_k, gap)
        lat = (time.perf_counter() - t0) * 1000.0 if time_search else np.nan
        h = np.asarray(p["hit_ts"])
        inside = (h >= q.gs) & (h <= q.ge)
        rows.append((q.qid, bool(inside[0]), bool(inside.any()), p["pred_s"], p["pred_e"], lat))
    df = pd.DataFrame(rows, columns=["qid", "hit1", "hit5", "pred_s", "pred_e", "lat_ms"])
    gs = np.array([q.gs for q in queries]); ge = np.array([q.ge for q in queries])
    df["iou"] = iou_arrays(df["pred_s"], df["pred_e"], gs, ge)
    return df

def random_moment(gs, ge, dur, seed: int = C.SEED, top_k: int = C.TOP_K) -> pd.DataFrame:
    """Chance reference. Per query: a random interval of length = mean GT length (clipped to the video), uniformly placed;
    hit@1 / hit@5 from 1 / 5 uniformly random timestamps. gs, ge, dur are per-query arrays."""
    rng = np.random.default_rng(seed)
    gs, ge, dur = (np.asarray(x, dtype=np.float64) for x in (gs, ge, dur))
    L = float((ge - gs).mean())
    rows = []
    for s, e, d in zip(gs, ge, dur):
        l = min(L, d); st = rng.uniform(0, d - l)
        r = rng.uniform(0, d, size=top_k)
        inside = (r >= s) & (r <= e)
        rows.append((bool(inside[0]), bool(inside.any()), st, st + l))
    df = pd.DataFrame(rows, columns=["hit1", "hit5", "pred_s", "pred_e"])
    df["iou"] = iou_arrays(df["pred_s"], df["pred_e"], gs, ge)
    df["lat_ms"] = np.nan
    return df
