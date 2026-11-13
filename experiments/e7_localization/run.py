"""E7 - moment localization upgrades on Charades-STA (single-video search).
Usage: python -W ignore experiments/e7_localization/run.py [--limit N] [--force]
Per-query results for ALL configs/videos are computed once (shards of 100 videos, resumable); the DEV/HELD-OUT protocol is
applied at aggregation: configs are chosen on DEV (R1@0.5) only, then reported on HELD-OUT and ALL.
Indexes: full_5fps and cos_last_drop tau=0.95. Pruned index curves are piecewise-constant over each kept item's coverage span."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd
from sseval import config as C, data, runutil, localize as L
from sseval.metrics import bootstrap_indices, summarize
from sseval.pruning import select
from sseval.retrieval import Query, evaluate

INDEXES = {"full_5fps": ("full_5fps", {}), "cos095": ("cos_last_drop", {"tau": 0.95})}
WS, LS, ALPHAS, LAMS, LAMS_Z = [1, 2, 3], [6, 8, 10], [0.3, 0.5, 0.7], [0.0, 0.1, 0.2], [0.0, 0.25, 0.5, 1.0]
LAMS_X = [-0.1, -0.2, -0.4]
SHARD_N = 100
METHOD_LABEL = {"M0": "M0 baseline (E1 protocol)", "M1": "M1 smoothed peak", "M2": "M2 fixed length", "M3": "M3 threshold growth",
                "M4": "M4 contrast proposals", "M5": "M5 z-score + best of M3/M4",
                "M3x": "M3x threshold growth (extended grid)", "P0": "P0 centre prior (no query)"}

def configs():
    c = [("M0", "M0")]
    c += [("M1", f"M1[w={w}]") for w in WS]
    c += [("M2", f"M2[w={w},L={l}]") for w in WS for l in LS]
    c += [("M3", f"M3[w={w},a={a}]") for w in WS for a in ALPHAS]
    c += [("M4", f"M4[lam={l}]") for l in LAMS]
    c += [("M5", f"M5/M4[lam={l}]") for l in LAMS_Z]
    c += [("M5", f"M5/M3[w={w},a={a}]") for w in WS for a in ALPHAS]
    # extensions OUTSIDE the prescribed grid (the DEV optimum sat on the grid edge); reported as separate rows M3x / M4x
    c += [("EXT3", f"M3x[w={w},a={a}]") for w in (3, 5) for a in (0.8, 0.9, 1.0)]
    c += [("EXT4", f"M4x[lam={l}]") for l in LAMS_X]
    # query-agnostic prior (no model): fixed window at the video centre
    c += [("P0", f"P0[L={l}]") for l in LS]
    return c

def make_split(all_vids, seed=C.SEED):
    v = sorted(all_vids); rng = np.random.default_rng(seed); perm = rng.permutation(len(v))
    half = len(v) // 2
    return {"seed": seed, "dev": sorted(v[i] for i in perm[:half]), "heldout": sorted(v[i] for i in perm[half:])}

def per_video(sel, qs, dur, m0):
    """-> list of rows (qid, config, hit1, hit5, r5_iou05, pred_s, pred_e)."""
    cen = L.make_grid(dur); ids = L.item_ids(sel.span_end, cen); rows = []
    for k, q in enumerate(qs):
        gt = (q.gs, q.ge); s = L.curve(sel.emb, ids, q.qe)
        hit = lambda j: bool(q.gs <= sel.ts[ids[j]] <= q.ge)
        item_span = lambda j: (float(sel.span_start[ids[j]]), float(sel.span_end[ids[j]]))
        r = m0.iloc[k]
        rows.append((q.qid, "M0", bool(r.hit1), bool(r.hit5), np.nan, float(r.pred_s), float(r.pred_e)))
        sm = {w: L.smooth(s, w) for w in WS}; pk = {w: int(np.argmax(sm[w])) for w in WS}
        for w in WS:
            p = pk[w]; a, b = item_span(p)
            rows.append((q.qid, f"M1[w={w}]", hit(p), np.nan, np.nan, a, b))
            for l in LS:
                a, b = L.fixed_span(float(cen[p]), l, dur)
                rows.append((q.qid, f"M2[w={w},L={l}]", hit(p), np.nan, np.nan, a, b))
        for pre, curv in (("M3", sm), ("M5/M3", {w: L.zscore(sm[w]) for w in WS})):   # z-score is affine -> M3 invariant (sanity)
            for w in WS:
                for al in ALPHAS:
                    a, b = L.grow_span(curv[w], pk[w], al, duration=dur)
                    rows.append((q.qid, f"{pre}[w={w},a={al}]", hit(pk[w]), np.nan, np.nan, a, b))
        for w in (3, 5):
            smx = sm[w] if w in sm else L.smooth(s, w); px = int(np.argmax(smx))
            for al in (0.8, 0.9, 1.0):
                a, b = L.grow_span(smx, px, al, duration=dur)
                rows.append((q.qid, f"M3x[w={w},a={al}]", hit(px), np.nan, np.nan, a, b))
        for l in LS:
            a, b = L.fixed_span(dur / 2.0, l, dur)
            rows.append((q.qid, f"P0[L={l}]", np.nan, np.nan, np.nan, a, b))
        for pre, ss, lams in (("M4", s, LAMS), ("M5/M4", L.zscore(s), LAMS_Z), ("M4x", s, LAMS_X)):
            for lam in lams:
                st, en, sc = L.contrast_spans(ss, lam, dur)
                top = L.nms_topk(st, en, sc, 5, 0.5)
                a, b = top[0]
                inside = (cen >= a) & (cen <= b)
                j = int(np.flatnonzero(inside)[np.argmax(s[inside])]) if inside.any() else int(np.argmax(s))
                r5 = float(any(L.iou(t, gt) >= 0.5 for t in top))
                rows.append((q.qid, f"{pre}[lam={lam}]", hit(j), np.nan, r5, a, b))
    return rows

def run_shards(vids, by_vid, QE, out, force):
    (out / "shards").mkdir(exist_ok=True)
    for si in range(0, len(vids), SHARD_N):
        f = out / "shards" / f"s{si // SHARD_N:03d}.parquet"
        if f.exists() and not force:
            print(f"[e7] {f.name} exists, skip", flush=True); continue
        parts, t0 = [], time.time()
        for v in vids[si:si + SHARD_N]:
            emb, ts, dur = data.load_video(v)
            g = by_vid[v]; qs = [Query(int(r.qid), QE[r.qid], r.gs, r.ge) for r in g.itertuples()]
            for iname, (m, p) in INDEXES.items():
                sel = select(m, emb, ts, dur, **p)
                m0 = evaluate(sel, qs)
                df = pd.DataFrame(per_video(sel, qs, dur, m0), columns=["qid", "config", "hit1", "hit5", "r5_iou05", "pred_s", "pred_e"])
                df["index"] = iname; df["vid"] = v; df["n_kept"] = sel.n_kept; df["n_frames"] = len(emb)
                parts.append(df)
        pq = pd.concat(parts, ignore_index=True)
        pq.to_parquet(f.with_suffix(".tmp"), index=False); f.with_suffix(".tmp").rename(f)
        print(f"[e7] {f.name}: {min(si + SHARD_N, len(vids))}/{len(vids)} videos, {time.time() - t0:.0f}s", flush=True)

def metrics_for(pq, q, idx):
    g = pq.set_index("qid").loc[q.qid].reset_index()          # identical order -> paired CIs
    g["iou"] = L_iou(g, q)
    g["hit1"] = g.hit1.astype(float); g["hit5"] = g.hit5.astype(float)
    r = summarize(g, idx)
    r["r5_iou05"] = float(g.r5_iou05.mean()) if g.r5_iou05.notna().any() else np.nan
    r["pred_len_mean_s"] = float((g.pred_e - g.pred_s).mean())
    return r

def L_iou(g, q):
    from sseval.metrics import iou_arrays
    return iou_arrays(g.pred_s.values, g.pred_e.values, q["gs"].values, q["ge"].values)

def aggregate(out, df, split, vids):
    pq = pd.concat([pd.read_parquet(p) for p in sorted((out / "shards").glob("s*.parquet"))], ignore_index=True)
    qall = df[df.vid.isin(vids)].sort_values("qid").reset_index(drop=True)
    sets = {"DEV": qall[qall.vid.isin(split["dev"])], "HELD-OUT": qall[qall.vid.isin(split["heldout"])], "ALL": qall}
    boots = {k: bootstrap_indices(len(v)) for k, v in sets.items()}
    cfg_method = {c: {"EXT3": "M3x", "EXT4": "M4x"}.get(m, m) for m, c in configs()}
    cache = {}
    def get(index, cfg, split_name):
        k = (index, cfg, split_name)
        if k not in cache:
            q = sets[split_name]; g = pq[(pq["index"] == index) & (pq.config == cfg) & pq.vid.isin(q.vid.unique())]
            r = metrics_for(g, q, boots[split_name]); r.update(index=index, config=cfg, method=cfg_method[cfg], split=split_name)
            cache[k] = r
        return cache[k]
    dev_rows = [get(i, c, "DEV") for i in INDEXES for m, c in configs()]
    dev = pd.DataFrame(dev_rows)
    cols = ["index", "method", "config", "split", "n_queries", "hit1", "hit1_lo", "hit1_hi", "hit5", "r1_iou03", "r1_iou05", "r1_iou05_lo", "r1_iou05_hi",
            "r1_iou07", "miou", "r5_iou05", "pred_len_mean_s"]
    dev[cols].to_csv(out / "dev_grid.csv", index=False)
    best, rows = {}, []
    for i in INDEXES:
        d = dev[dev["index"] == i]
        # M5 is reported only for whichever of M3/M4 is better on DEV (its z-scored variant)
        win = "M3" if d[d.method == "M3"].r1_iou05.max() >= d[d.method == "M4"].r1_iou05.max() else "M4"
        for m in METHOD_LABEL:
            if m == "M5": dm = d[d.config.str.startswith(f"M5/{win}")]
            elif m == "M3x": dm = d[d.method.isin(["M3x", "M3"])]      # spec grid + extension (M4x lambda<0 never beat M4 on DEV: dev_grid only)
            else: dm = d[d.method == m]
            b = dm.sort_values(["r1_iou05", "hit1"], ascending=False).iloc[0]
            best[f"{i}|{m}"] = dict(index=i, method=m, config=b.config, dev_r1_iou05=float(b.r1_iou05), dev_hit1=float(b.hit1))
            for sp in ("HELD-OUT", "ALL"):
                rows.append(get(i, b.config, sp))
        best[f"{i}|M5"]["z_scored_variant_of"] = win
    held = pd.DataFrame(rows); held["label"] = held.method.map(METHOD_LABEL)
    held[["index", "method", "label", "config", "split"] + cols[4:]].to_csv(out / "heldout.csv", index=False)
    (out / "best_configs.json").write_text(json.dumps(best, indent=2))
    # sanity: z-scored M3 must equal M3; M0 vs SUMMARY.md
    chk = []
    for i in INDEXES:
        a = get(i, "M3[w=2,a=0.5]", "ALL"); b = get(i, "M5/M3[w=2,a=0.5]", "ALL")
        chk.append(f"{i}: M3 vs z-scored M3 (w=2,a=0.5) R1@0.5 {a['r1_iou05']:.4f} vs {b['r1_iou05']:.4f}")
        m0 = get(i, "M0", "ALL"); chk.append(f"{i}: M0 ALL Hit@1={m0['hit1']:.3f} R1@0.5={m0['r1_iou05']:.3f} mIoU={m0['miou']:.3f} (SUMMARY: full 0.486/0.024/0.097, cos 0.482/0.078/0.150)")
    (out / "sanity.txt").write_text("\n".join(chk) + "\n"); print("SANITY\n  " + "\n  ".join(chk))
    return dev, held, best, sets

def main():
    ap = runutil.base_parser("E7 moment localization"); a = ap.parse_args()
    out = C.results_dir("e7_localization" if not a.limit else "e7_localization_validate")
    df = data.load_annotations(); QE = data.load_query_emb(df)
    allv = data.list_available_videos(df)
    split = make_split(allv)
    (out / "split.json").write_text(json.dumps(split))
    vids = data.select_videos(df, a.limit)
    by_vid = {v: g for v, g in df.groupby("vid")}
    print(f"[e7] {len(vids)} videos ({len(allv)} available; dev {len(split['dev'])}, held-out {len(split['heldout'])})", flush=True)
    if a.force:
        for p in (out / "shards").glob("*"): p.unlink()
    t0 = time.time()
    run_shards(vids, by_vid, QE, out, a.force)
    dev, held, best, sets = aggregate(out, df, split, vids)
    runutil.write_meta(out, limit=a.limit, n_videos=len(vids), wall_s=time.time() - t0)
    print(held[["index", "method", "config", "split", "hit1", "r1_iou03", "r1_iou05", "r1_iou07", "miou"]].round(3).to_string(index=False))

if __name__ == "__main__":
    main()
