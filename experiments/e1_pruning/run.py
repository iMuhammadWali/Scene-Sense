"""E1 - query-agnostic frame selection at indexing time (single-video search).
Usage: python -W ignore experiments/e1_pruning/run.py [--limit N] [--force]
Resumable: one shard per method family in results/e1_pruning/shards/."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd
from sseval import config as C, data, runutil
from sseval.grids import e1_grid, cfg_name, FAMILY_OF
from sseval.metrics import bootstrap_indices, summarize
from sseval.pruning import select
from sseval.retrieval import Query, evaluate, random_moment, index_bytes

def run_shard(shard, cfgs, vids, by_vid, QE, out):
    f = out / "shards" / f"{shard}.parquet"
    if f.exists():
        print(f"[e1] shard {shard}: exists, skipping", flush=True); return
    pq_parts, vid_rows = [], []
    t0 = time.time()
    for ci, (method, params) in enumerate(cfgs, 1):
        name = cfg_name(method, params)
        for v, (emb, ts, dur) in vids.items():
            sel = select(method, emb, ts, dur, **params)
            g = by_vid[v]
            qs = [Query(int(r.qid), QE[r.qid], r.gs, r.ge) for r in g.itertuples()]
            res = evaluate(sel, qs)
            res["config"] = name; res["vid"] = v
            pq_parts.append(res[["config", "vid", "qid", "hit1", "hit5", "iou", "pred_s", "pred_e"]])
            vid_rows.append((name, v, len(emb), sel.n_kept, index_bytes(sel), sel.select_s))
        print(f"  [{shard}] {ci}/{len(cfgs)} {name}  ({time.time() - t0:.0f}s)", flush=True)
    pq = pd.concat(pq_parts, ignore_index=True)
    pq["iou"] = pq["iou"].astype("float32"); pq["pred_s"] = pq["pred_s"].astype("float32"); pq["pred_e"] = pq["pred_e"].astype("float32")
    vs = pd.DataFrame(vid_rows, columns=["config", "vid", "n_frames", "n_kept", "index_bytes", "select_s"])
    (out / "shards").mkdir(exist_ok=True)
    vs.to_parquet(out / "shards" / f"{shard}.videos.parquet")
    pq.to_parquet(f.with_suffix(".tmp"), index=False)
    f.with_suffix(".tmp").rename(f)
    print(f"[e1] shard {shard} done in {time.time() - t0:.0f}s", flush=True)

def aggregate(out, vids, df, grid):
    sh = out / "shards"
    pq = pd.concat([pd.read_parquet(p) for p in sorted(sh.glob("*.parquet")) if not p.name.endswith(".videos.parquet")], ignore_index=True)
    vs = pd.concat([pd.read_parquet(p) for p in sorted(sh.glob("*.videos.parquet"))], ignore_index=True)
    q = df[df.vid.isin(vids)].sort_values("qid").reset_index(drop=True)
    idx = bootstrap_indices(len(q))
    cfg_info = {cfg_name(m, p): (m, p) for cfgs in grid.values() for m, p in cfgs}
    rows, pq_out = [], []
    for name, g in pq.groupby("config", sort=False):
        g = g.set_index("qid").loc[q.qid].reset_index()                 # identical query order for every config -> paired CIs
        m, p = cfg_info[name]
        v = vs[vs.config == name]
        r = dict(method=m, params=";".join(f"{k}={x}" for k, x in p.items()), config=name, family=FAMILY_OF[m],
                 n_videos=len(vids), pct_kept=100 * (v.n_kept / v.n_frames).mean(), frames_per_video=v.n_kept.mean(),
                 index_MB_per_video=v.index_bytes.mean() / 1e6, select_ms_per_video=1000 * v.select_s.mean())
        r.update(summarize(g, idx)); r["pred_len_mean_s"] = float((g.pred_e - g.pred_s).mean()); rows.append(r)
        pq_out.append(g)
    # chance reference (mean GT length, fixed seed)
    rm = random_moment(q["gs"], q["ge"], q["vid"].map(lambda x: vids[x][2]))
    rm["qid"] = q.qid.values; rm["config"] = "random_moment"; rm["vid"] = q.vid.values
    r = dict(method="random_moment", params="", config="random_moment", family="random", n_videos=len(vids),
             pct_kept=float("nan"), frames_per_video=float("nan"), index_MB_per_video=float("nan"), select_ms_per_video=float("nan"))
    r.update(summarize(rm, idx)); r["pred_len_mean_s"] = float((rm.pred_e - rm.pred_s).mean()); rows.append(r)
    pq_out.append(rm[["config", "vid", "qid", "hit1", "hit5", "iou", "pred_s", "pred_e"]])
    op = pd.DataFrame(rows)
    cols = ["method", "params", "config", "family", "n_videos", "n_queries", "pct_kept", "frames_per_video", "index_MB_per_video",
            "select_ms_per_video", "hit1", "hit5", "r1_iou03", "r1_iou05", "r1_iou07", "miou", "r1_iou05_lo", "r1_iou05_hi", "hit1_lo", "hit1_hi", "pred_len_mean_s"]
    op = op[cols]
    op["gt_len_mean_s"] = float((q["ge"] - q["gs"]).mean())
    op.to_csv(out / "operating_points.csv", index=False)
    allpq = pd.concat(pq_out, ignore_index=True)
    allpq["config"] = allpq["config"].astype("category")
    allpq.to_parquet(out / "per_query.parquet", index=False)
    # budget table: best operating point per method at <=10/20/30 % frames kept, by R@1@0.5 (plan) and, as a protocol-robust
    # companion, by Hit@1 (R@1 IoU rewards long coverage spans of sparse indexes - see pred_len_mean_s in operating_points.csv)
    b = []
    for crit in ("r1_iou05", "hit1"):
        for method, g in op[~op.method.isin(["random_moment"])].groupby("method"):
            for bud in (10, 20, 30):
                c = g[g.pct_kept <= bud]
                if len(c):
                    best = c.sort_values([crit, "pct_kept"], ascending=[False, True]).iloc[0]
                    b.append(dict(criterion=crit, method=method, budget_pct=bud, params=best.params, pct_kept=best.pct_kept,
                                  r1_iou05=best.r1_iou05, r1_iou05_lo=best.r1_iou05_lo, r1_iou05_hi=best.r1_iou05_hi,
                                  hit1=best.hit1, hit1_lo=best.hit1_lo, hit1_hi=best.hit1_hi, miou=best.miou))
                else:
                    b.append(dict(criterion=crit, method=method, budget_pct=bud, params="(no operating point within budget)"))
    pd.DataFrame(b).to_csv(out / "budget_table.csv", index=False)
    return op

def sanity(op, out):
    msgs = []
    full = op[op.config == "full_5fps"].iloc[0]; rnd = op[op.config == "random_moment"].iloc[0]
    msgs.append(f"full_5fps Hit@1={full.hit1:.3f} (95% CI {full.hit1_lo:.3f}-{full.hit1_hi:.3f}) vs random {rnd.hit1:.3f} -> {'OK (CI above chance)' if full.hit1_lo > rnd.hit1 else 'CONCERN'}")
    u1 = op[op.config == "uniform[fraction=1.0]"].iloc[0]
    same = all(abs(u1[c] - full[c]) < 1e-12 for c in ["hit1", "hit5", "r1_iou03", "r1_iou05", "r1_iou07", "miou"])
    msgs.append(f"uniform fraction=1.0 identical to full_5fps -> {'OK' if same else 'FAIL'}")
    msgs.append(f"NaNs in non-random rows -> {int(op[op.method != 'random_moment'].select_dtypes('number').isna().sum().sum())}")
    (out / "sanity.txt").write_text("\n".join(msgs) + "\n"); print("SANITY\n  " + "\n  ".join(msgs))

def main():
    ap = runutil.base_parser("E1 frame selection sweep"); a = ap.parse_args()
    out = C.results_dir("e1_pruning")
    if runutil.is_done(out, a.force): print("[e1] DONE exists, skipping"); return
    if a.force:
        for p in (out / "shards").glob("*"): p.unlink()
    df = data.load_annotations(); QE = data.load_query_emb(df)
    vids = data.load_videos(data.select_videos(df, a.limit))
    by_vid = {v: g for v, g in df.groupby("vid")}
    print(f"[e1] {len(vids)} videos, {sum(len(by_vid[v]) for v in vids)} queries", flush=True)
    grid = e1_grid(); t0 = time.time()
    for shard, cfgs in grid.items():
        run_shard(shard, cfgs, vids, by_vid, QE, out)
    op = aggregate(out, vids, df, grid); sanity(op, out)
    runutil.write_meta(out, limit=a.limit, n_videos=len(vids), n_queries=int(op.n_queries.iloc[0]), wall_s=time.time() - t0)
    runutil.mark_done(out)
    print(op[["config", "pct_kept", "hit1", "r1_iou05", "miou"]].round(3).to_string(index=False))

if __name__ == "__main__":
    main()
