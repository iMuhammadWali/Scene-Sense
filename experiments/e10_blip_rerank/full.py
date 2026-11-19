"""E10 stage s5: the FULL Charades-STA test set (all 3,720 queries, 1,334 videos), real groups only, K=1 peak frame, as-is query text.

Protocol: pruned index (cos_last_drop tau 0.95), fixed single-projection text embeddings, top-20 FAISS hits -> team's group_temporally (gap 2 s)
-> groups ranked by best-hit similarity -> at most 5 groups (E8 `build_candidates`; padding candidates are DROPPED, so 1-5 real groups per query,
as the app forms them). Candidate time = the group's peak CLIP frame. One BLIP-ITM score per group from that frame.
Scores are cached per query in <out>/s5_full_<res>/raw/<qid>.json  ({"vid", "t": [...], "logit": [...]}), so the run is resumable in chunks.
Frames are decoded once per video (all needed timestamps of all its queries, ascending).
"""
from __future__ import annotations
import sys, json, time, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd
from sseval import config as C, data, rerank as R
from sseval.metrics import bootstrap_indices, bootstrap_ci, iou_arrays

METHOD, PARAMS = "cos_last_drop", {"tau": 0.95}
HERE = Path(__file__).resolve().parent

def _e8():
    spec = importlib.util.spec_from_file_location("e8_run", HERE.parents[1] / "experiments" / "e8_slowpath" / "run.py")
    m = importlib.util.module_from_spec(spec); sys.modules["e8_run"] = m; spec.loader.exec_module(m)
    return m

def real_group_candidates(sel, hits, sims, dur, e8):
    """<=5 real groups (cand_src == 'group'), CLIP order. Padding candidates are removed."""
    return [c for c in e8.build_candidates(sel, hits, sims, dur) if c["src"] == "group"]

def build_full_plans(limit: int | None = None):
    """-> list of dict(qid, vid, caption, gs, ge, dur, cands) for every test query (index and selection built once per video)."""
    from sseval.pruning import select
    from sseval.retrieval import build_index, search
    e8 = _e8()
    df = data.load_annotations(); QE = data.load_query_emb(df)
    vids = data.load_videos(data.select_videos(df))
    out = []
    for v, g in df.groupby("vid"):
        if v not in vids:
            continue
        emb, ts, dur = vids[v]
        sel = select(METHOD, emb, ts, dur, **PARAMS); index, mapping = build_index(sel)
        for i in g.index:
            qe = QE[i] / max(np.linalg.norm(QE[i]), 1e-12)
            hits = search(index, qe, mapping, e8.TOP_HITS)
            cands = real_group_candidates(sel, hits, sel.emb @ qe, dur, e8)
            out.append(dict(qid=int(df.qid[i]), vid=v, caption=df.caption[i], gs=float(df["gs"][i]), ge=float(df["ge"][i]), dur=float(dur), cands=cands))
    out.sort(key=lambda r: r["qid"])
    return out[:limit] if limit else out

def run_s5(blip, out: Path, res: int, limit, decode):
    d = out / f"s5_full_{res}" / "raw"; d.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); plans = build_full_plans(limit)
    print(f"[s5] {len(plans)} queries, {sum(len(p['cands']) for p in plans)} real-group candidates ({time.time() - t0:.0f}s to build); res={res}", flush=True)
    by_vid = {}
    for p in plans:
        by_vid.setdefault(p["vid"], []).append(p)
    t0, done_q, scored = time.time(), 0, 0
    for k, (v, ps) in enumerate(by_vid.items()):
        todo = [p for p in ps if not (d / f"{p['qid']}.json").exists()]
        done_q += len(ps) - len(todo)
        if not todo:
            continue
        times = sorted({t for p in todo for t in [R.frame_times(c["t"], p["dur"])[0] for c in p["cands"]]})
        frames = dict(zip(times, decode(C.VIDEOS_DIR / f"{v}.mp4", times)))
        for p in todo:
            ts = [R.frame_times(c["t"], p["dur"])[0] for c in p["cands"]]
            lg, _, _, _ = blip.score([frames[t] for t in ts], [p["caption"]], res)
            tmp = d / f"{p['qid']}.tmp"; tmp.write_text(json.dumps(dict(vid=v, t=ts, logit=[float(x) for x in lg[0]])))
            tmp.rename(d / f"{p['qid']}.json"); scored += 1; done_q += 1
        if (k + 1) % 20 == 0:
            el = time.time() - t0
            print(f"[s5] {k + 1}/{len(by_vid)} videos, {done_q}/{len(plans)} queries ({scored} scored this run), {el:.0f}s", flush=True)
    print(f"[s5] finished, {scored} scored in {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------------------------------------------------------- aggregation
def fixed_weights(out: Path, res: int):
    """Fusion weights fixed from the 300-query sample at the SAME resolution (first-half-chosen w, real-group config)."""
    sm = pd.read_csv(out / "summary.csv")
    cfg = "K1_asis_real" if res == 384 else "K1_288_asis_real"
    w = {}
    for m in ("fuse_z_devw", "fuse_rank_devw"):
        r = sm[(sm.subset == cfg) & (sm.method == m) & (sm.metric == "chosen_w")]
        w[m] = float(r.value.iloc[0])
    return w, cfg

def aggregate_s5(out: Path, res: int):
    plans = build_full_plans()
    d = out / f"s5_full_{res}" / "raw"
    done = [p for p in plans if (d / f"{p['qid']}.json").exists()]
    print(f"[s5] aggregating {len(done)}/{len(plans)} queries", flush=True)
    W, cfg = fixed_weights(out, res)
    n, NC = len(done), 5
    clip = np.full((n, NC), np.nan); itm = np.full((n, NC), np.nan); t = np.full((n, NC), np.nan)
    sp_s = np.full((n, NC), np.nan); sp_e = np.full((n, NC), np.nan)
    for i, p in enumerate(done):
        r = json.loads((d / f"{p['qid']}.json").read_text())
        for j, c in enumerate(p["cands"]):
            clip[i, j], t[i, j], sp_s[i, j], sp_e[i, j] = c["sim"], c["t"], c["span_s"], c["span_e"]
            itm[i, j] = r["logit"][j]
    gs = np.array([p["gs"] for p in done]); ge = np.array([p["ge"] for p in done]); dur = np.array([p["dur"] for p in done])
    hit = ((t >= gs[:, None]) & (t <= ge[:, None])).astype(float); hit[np.isnan(t)] = 0.0
    iou8 = R.window_iou(t, gs, ge, dur, 8.0)
    ious = iou_arrays(sp_s, sp_e, gs[:, None], ge[:, None]) ; ious = np.where(np.isnan(t), np.nan, ious)
    methods = {"clip": clip, "blip": itm, "fuse_z": R.fuse_z(clip, itm, W["fuse_z_devw"]), "fuse_rank": R.fuse_rank(clip, itm, W["fuse_rank_devw"], 2.0)}
    idx = bootstrap_indices(n); rows = []; vec = {}
    ar = np.arange(n)
    for m, S in methods.items():
        ch = R.argmax_rows(S)
        v = dict(hit1=hit[ar, ch], iou8=iou8[ar, ch], iouspan=ious[ar, ch])
        sc = np.where(np.isnan(S), -np.inf, S); order = np.argsort(-sc, axis=1, kind="stable")
        v["hit3"] = np.take_along_axis(hit, order[:, :3], 1).max(axis=1)
        vec[m] = v
    oracle = hit.max(axis=1)
    def add(m, metric, x, ref=None):
        x = np.asarray(x, float); lo, hi = bootstrap_ci(x, idx)
        rows.append(dict(method=m, metric=metric, n=n, value=float(x.mean()), lo=lo, hi=hi, res=res, w=W.get({"fuse_z": "fuse_z_devw", "fuse_rank": "fuse_rank_devw"}.get(m, ""), np.nan)))
    for m, v in vec.items():
        add(m, "hit1", v["hit1"]); add(m, "hit3", v["hit3"]); add(m, "hit5_oracle_real_groups", oracle)
        for th in (0.3, 0.5, 0.7):
            add(m, f"r1_iou{th}_window8s", v["iou8"] >= th); add(m, f"r1_iou{th}_groupspan", v["iouspan"] >= th)
        add(m, "miou_window8s", v["iou8"]); add(m, "miou_groupspan", v["iouspan"])
        if m != "clip":
            for met, key in (("hit1", "hit1"), ("r1_iou0.5_window8s", None), ("r1_iou0.5_groupspan", None), ("miou_window8s", "iou8")):
                a = {"hit1": v["hit1"], "r1_iou0.5_window8s": (v["iou8"] >= .5) * 1.0, "r1_iou0.5_groupspan": (v["iouspan"] >= .5) * 1.0, "miou_window8s": v["iou8"]}[met]
                b = {"hit1": vec["clip"]["hit1"], "r1_iou0.5_window8s": (vec["clip"]["iou8"] >= .5) * 1.0,
                     "r1_iou0.5_groupspan": (vec["clip"]["iouspan"] >= .5) * 1.0, "miou_window8s": vec["clip"]["iou8"]}[met]
                add(m, f"delta_{met}_vs_clip", a - b)
    ng = np.sum(~np.isnan(t), axis=1)
    for k in range(1, 6):
        rows.append(dict(method="n_groups", metric=f"share_{k}", n=n, value=float((ng == k).mean()), res=res))
        sel = ng == k
        if sel.sum():
            rows.append(dict(method="oracle_by_groups", metric=f"hit5_oracle_ngroups{k}", n=int(sel.sum()), value=float(oracle[sel].mean()), res=res))
    sm = pd.DataFrame(rows); sm.to_csv(out / f"summary_s5_full_{res}.csv", index=False)
    pq = pd.DataFrame(dict(qid=[p["qid"] for p in done], vid=[p["vid"] for p in done], n_groups=ng, clip_hit1=vec["clip"]["hit1"], blip_hit1=vec["blip"]["hit1"],
                           fuse_z_hit1=vec["fuse_z"]["hit1"], oracle=oracle, clip_iou8=vec["clip"]["iou8"], fuse_z_iou8=vec["fuse_z"]["iou8"]))
    pq.to_csv(out / f"per_query_s5_full_{res}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        print(f"weights (fixed from the 300-query sample, config {cfg}): {W}")
        print(sm[sm.method.isin(["clip", "blip", "fuse_z", "fuse_rank"]) & ~sm.metric.str.startswith("miou")].to_string(index=False))
