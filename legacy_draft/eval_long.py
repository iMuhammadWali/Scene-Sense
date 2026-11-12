"""Experiment 1b: stitched long streams (BOLT-style multi-source). Whole-stream selection + one FAISS index per stream.
Usage: python -W ignore eval_long.py [--lengths 5 10 20 30] [--seed 0]"""
import argparse
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("--lengths", type=int, nargs="+", default=[5, 10, 20, 30]); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
selftest_iou()
df = load_queries(); QE = load_query_emb()
vids = load_videos(); by_vid = {v: g for v, g in df.groupby("vid")}
names = sorted(v for v in vids if v in by_vid)
rng = np.random.default_rng(a.seed); order = list(rng.permutation(names))
print(f"Experiment 1b: {len(order)} videos available, total {sum(vids[v][2] for v in order)/60:.1f} min", flush=True)

TH = [0.85, 0.90, 0.95]
out_rows = []
for L in a.lengths:
    # consecutive clips until summed duration >= L minutes; leftover dropped
    streams, cur, tot = [], [], 0.0
    for v in order:
        cur.append(v); tot += vids[v][2]
        if tot >= L * 60: streams.append(cur); cur, tot = [], 0.0
    print(f"\nL={L} min: {len(streams)} streams", flush=True)
    if not streams: continue
    pq_all, un_all, gt = [], [], []; t0 = time.time()
    for si, members in enumerate(streams):
        off, E, T_, qs = 0.0, [], [], []
        for v in members:
            emb, ts, dur = vids[v]
            E.append(emb); T_.append(ts + off)
            for r in by_vid[v].itertuples():
                qs.append(dict(qid=int(r.qid), qe=QE[r.qid], gs=r.gs + off, ge=r.ge + off))
            off += dur
        E = np.concatenate(E); T_ = np.concatenate(T_); dur_tot = off
        sels = build_selections(E, T_, fixed=(("fixed_1fps", 5),), thresholds=TH, matched=TH)
        for name, (sel, sec) in sels.items():
            rows, nbytes = eval_selection(E, T_, dur_tot, sel, qs)
            for r in rows: r["method"] = name
            pq_all += rows
            un_all.append(dict(method=name, n_frames=len(E), n_sel=len(sel), index_bytes=nbytes, sel_s=sec, dur=dur_tot))
        gt += [(q["gs"], q["ge"], dur_tot) for q in qs]
        print(f"   stream {si+1}/{len(streams)}: {len(members)} clips, {dur_tot/60:.1f} min, {len(E)} frames, {len(qs)} queries, "
              f"thr0.95 kept {100*len(sels['threshold_0.95'][0])/len(E):.1f}% ({time.time()-t0:.0f}s)", flush=True)
    pq = pd.DataFrame(pq_all); un = pd.DataFrame(un_all)
    g = np.array(gt); rb, _ = random_baseline(g[:, 0], g[:, 1], g[:, 2], seed=a.seed); rb["method"] = "random_moment"
    pq = pd.concat([pq, rb], ignore_index=True)
    for m, gq in pq.groupby("method", sort=False):
        row = dict(L_min=L, method=m, family=method_family(m), T=method_T(m), n_streams=len(streams),
                   mean_stream_min=un.dur.mean() / 60)
        row.update(metric_row(gq))
        u = un[un.method == m]
        if len(u):
            row.update(pct_frames_kept=100 * (u.n_sel / u.n_frames).mean(), frames_per_stream=u.n_sel.mean(),
                       index_MB_per_stream=u.index_bytes.mean() / 1e6, sel_s_per_stream=u.sel_s.mean())
        row["label"] = DIVERSITY_LABEL if row["family"] == "diversity" else m
        out_rows.append(row)

res = pd.DataFrame(out_rows)
if len(res) == 0: raise SystemExit("no streams could be formed (not enough video yet)")
res.to_csv(f"{RES}/exp1b_long.csv", index=False)
num = res[res.method != "random_moment"].select_dtypes("number")
print("NaNs (non-random rows):", int(num.isna().sum().sum()))
print(res[["L_min", "method", "n_streams", "n_queries", "pct_frames_kept", "hit1", "R1_iou0_5", "mIoU", "search_ms_per_query", "index_MB_per_stream", "sel_s_per_stream"]].round(3).to_string(index=False))
print(f"wrote {RES}/exp1b_long.csv")
