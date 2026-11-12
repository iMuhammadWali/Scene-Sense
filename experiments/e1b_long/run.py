"""E1b - long videos: BOLT-style multi-source streams (5/10/20/30 min), one index per stream, whole-stream selection.
Usage: python -W ignore experiments/e1b_long/run.py [--limit N] [--lengths 5 10 20 30] [--force]
Resumable: one shard per stream length in results/e1b_long/shards/.
Limitations (state in the report): hard cuts between clips; generic actions repeat across clips (distractors)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd
from sseval import config as C, data, runutil
from sseval.grids import e1b_methods, cfg_name, FAMILY_OF
from sseval.metrics import bootstrap_indices, summarize
from sseval.pruning import select
from sseval.retrieval import Query, evaluate, random_moment, index_bytes

def make_streams(order, vids, L_min):
    """Consecutive clips (in shuffled order) until the summed duration >= L minutes; leftover clips dropped."""
    streams, cur, tot = [], [], 0.0
    for v in order:
        cur.append(v); tot += vids[v][2]
        if tot >= L_min * 60:
            streams.append(cur); cur, tot = [], 0.0
    return streams

def run_length(L, order, vids, by_vid, QE, out):
    f = out / "shards" / f"L{L}.parquet"
    if f.exists():
        print(f"[e1b] L={L}: shard exists, skipping", flush=True); return
    streams = make_streams(order, vids, L)
    print(f"[e1b] L={L} min: {len(streams)} streams", flush=True)
    if not streams:
        return
    pq_parts, st_rows, t0 = [], [], time.time()
    for si, members in enumerate(streams):
        off, E, T, qs = 0.0, [], [], []
        for v in members:
            emb, ts, dur = vids[v]
            E.append(emb); T.append(ts + off)
            qs += [Query(int(r.qid), QE[r.qid], r.gs + off, r.ge + off) for r in by_vid[v].itertuples()]
            off += dur
        E = np.concatenate(E); T = np.concatenate(T); dur_tot = off
        kept_by_name = {}
        for method, params in e1b_methods():
            p = dict(params)
            name = cfg_name(method, params)
            if "match" in p:                                    # uniform at the same budget as another method on THIS stream
                p = {"fraction": kept_by_name[p["match"]] / len(E)}
            sel = select(method, E, T, dur_tot, **p)
            kept_by_name[name] = sel.n_kept
            res = evaluate(sel, qs, time_search=True)
            res["config"] = name; res["L"] = L; res["stream"] = si
            pq_parts.append(res[["L", "stream", "config", "qid", "hit1", "hit5", "iou", "pred_s", "pred_e", "lat_ms"]])
            st_rows.append((L, si, name, len(members), dur_tot, len(E), sel.n_kept, index_bytes(sel), sel.select_s, len(qs)))
            print(f"   L={L} stream {si+1}/{len(streams)} {name}: kept {sel.n_kept}/{len(E)} select {sel.select_s:.2f}s ({time.time()-t0:.0f}s)", flush=True)
    pq = pd.concat(pq_parts, ignore_index=True)
    st = pd.DataFrame(st_rows, columns=["L", "stream", "config", "n_clips", "dur_s", "n_frames", "n_kept", "index_bytes", "select_s", "n_queries"])
    st.to_parquet(out / "shards" / f"L{L}.streams.parquet")
    pq.to_parquet(f.with_suffix(".tmp"), index=False); f.with_suffix(".tmp").rename(f)
    print(f"[e1b] L={L} done in {time.time()-t0:.0f}s", flush=True)

def aggregate(out, df, vids, order, lengths):
    sh = out / "shards"; rows = []
    for L in lengths:
        f = sh / f"L{L}.parquet"
        if not f.exists():
            continue
        pq = pd.read_parquet(f); st = pd.read_parquet(sh / f"L{L}.streams.parquet")
        q = df.set_index("qid")
        idx = bootstrap_indices(pq[pq.config == pq.config.iloc[0]].shape[0])
        for name, g in pq.groupby("config", sort=False):
            g = g.sort_values("qid").reset_index(drop=True)
            s = st[st.config == name]
            m = name.split("[")[0]
            r = dict(L_min=L, method=m, params=name[len(m) + 1:-1] if "[" in name else "", config=name, family=FAMILY_OF[m],
                     n_streams=len(s), mean_stream_min=s.dur_s.mean() / 60, mean_frames_per_stream=s.n_frames.mean(),
                     pct_kept=100 * (s.n_kept / s.n_frames).mean(), kept_per_stream=s.n_kept.mean(),
                     index_MB_per_stream=s.index_bytes.mean() / 1e6, select_s_per_stream=s.select_s.mean(),
                     search_ms_per_query=g.lat_ms.mean(), search_ms_p95=g.lat_ms.quantile(0.95))
            r.update(summarize(g, idx)); rows.append(r)
        # chance reference: interval placed uniformly in the stream
        g0 = pq[pq.config == pq.config.iloc[0]].sort_values("qid").reset_index(drop=True)
        sd = st[st.config == pq.config.iloc[0]].set_index("stream").dur_s
        gt = q.loc[g0.qid]
        dur_q = g0["stream"].map(sd).values
        rm = random_moment(gt["gs"].values, gt["ge"].values, dur_q)
        r = dict(L_min=L, method="random_moment", params="", config="random_moment", family="random", n_streams=len(sd),
                 mean_stream_min=sd.mean() / 60)
        r.update(summarize(rm, idx)); rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / "long_results.csv", index=False)
    return res

def main():
    ap = runutil.base_parser("E1b long streams"); ap.add_argument("--lengths", type=int, nargs="+", default=[5, 10, 20, 30])
    ap.add_argument("--seed", type=int, default=C.SEED); a = ap.parse_args()
    out = C.results_dir("e1b_long")
    if runutil.is_done(out, a.force): print("[e1b] DONE exists, skipping"); return
    (out / "shards").mkdir(exist_ok=True)
    if a.force:
        for p in (out / "shards").glob("*"): p.unlink()
    df = data.load_annotations(); QE = data.load_query_emb(df)
    vids = data.load_videos(data.select_videos(df, a.limit))
    by_vid = {v: g for v, g in df.groupby("vid")}
    names = sorted(vids)
    order = [names[i] for i in np.random.default_rng(a.seed).permutation(len(names))]
    print(f"[e1b] {len(names)} videos, {sum(vids[v][2] for v in names)/60:.1f} min total, seed {a.seed}", flush=True)
    t0 = time.time()
    for L in a.lengths:
        run_length(L, order, vids, by_vid, QE, out)
    res = aggregate(out, df, vids, order, a.lengths)
    runutil.write_meta(out, limit=a.limit, n_videos=len(names), lengths=a.lengths, seed=a.seed, wall_s=time.time() - t0,
                       lengths_with_streams=sorted(res.L_min.unique().tolist()))
    if len(res) == 0:
        print("[e1b] no streams could be formed (not enough video)"); return
    runutil.mark_done(out)
    print(res[res.method != "random_moment"][["L_min", "config", "n_streams", "n_queries", "pct_kept", "hit1", "r1_iou05", "search_ms_per_query", "index_MB_per_stream", "select_s_per_stream"]].round(3).to_string(index=False))

if __name__ == "__main__":
    main()
