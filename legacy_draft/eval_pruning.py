"""Experiment 1: per-video (short-video) retrieval + moment localisation, per frame-selection method.
Usage: python -W ignore eval_pruning.py [--max-videos N]"""
import argparse
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--max-videos", type=int, default=None)
a = ap.parse_args()
selftest_iou()
df = load_queries(); QE = load_query_emb()
vids = load_videos(a.max_videos)
by_vid = {v: g for v, g in df.groupby("vid")}
vids = {v: x for v, x in vids.items() if v in by_vid}
print(f"Experiment 1: {len(vids)} videos, {sum(len(by_vid[v]) for v in vids)} queries", flush=True)

pq_rows, unit_rows = [], []
t0 = time.time()
for i, (v, (emb, ts, dur)) in enumerate(vids.items(), 1):
    g = by_vid[v]
    qs = [dict(qid=int(r.qid), qe=QE[r.qid], gs=r.gs, ge=r.ge) for r in g.itertuples()]
    for name, (sel, sec) in build_selections(emb, ts).items():
        rows, nbytes = eval_selection(emb, ts, dur, sel, qs)
        for r in rows: r.update(method=name, vid=v, dur=dur)
        pq_rows += rows
        unit_rows.append(dict(method=name, vid=v, n_frames=len(emb), n_sel=len(sel), index_bytes=nbytes, sel_s=sec, dur=dur))
    if i % 10 == 0 or i == len(vids): fmt_progress(i, len(vids), t0)

pq = pd.DataFrame(pq_rows); un = pd.DataFrame(unit_rows)
# chance-level baseline (per query, once)
q_all = df[df.vid.isin(vids)]
rb, L = random_baseline(q_all["gs"].values, q_all["ge"].values, q_all.vid.map(lambda v: vids[v][2]).values, seed=0)
rb["method"] = "random_moment"; rb["vid"] = q_all.vid.values; rb["qid"] = q_all.qid.values; rb["dur"] = q_all.vid.map(lambda v: vids[v][2]).values
pq = pd.concat([pq, rb], ignore_index=True)

# summary: one row per method
out = []
full_total = un[un.method == "full_5fps"].n_frames.sum()
for m, g in pq.groupby("method", sort=False):
    row = dict(method=m, family=method_family(m), T=method_T(m), n_videos=len(vids))
    row.update(metric_row(g))
    u = un[un.method == m]
    if len(u):
        row.update(pct_frames_kept=100 * (u.n_sel / u.n_frames).mean(), frames_per_video=u.n_sel.mean(),
                   index_MB_total=u.index_bytes.sum() / 1e6, sel_ms_per_video=1000 * u.sel_s.mean())
    else:
        row.update(pct_frames_kept=np.nan, frames_per_video=np.nan, index_MB_total=np.nan, sel_ms_per_video=np.nan)
    row["label"] = DIVERSITY_LABEL if row["family"] == "diversity" else m
    out.append(row)
summ = pd.DataFrame(out)

# sanity checks
rnd = summ[summ.method == "random_moment"].iloc[0]; full = summ[summ.method == "full_5fps"].iloc[0]
t95 = summ[summ.method == "threshold_0.95"].iloc[0]
chk = summ.drop(columns=["T", "label"]).select_dtypes("number").drop(columns=["search_ms_per_query"]).copy()
print("\nSANITY")
print(f"  full_5fps hit@1={full.hit1:.3f} vs random hit@1={rnd.hit1:.3f}   -> {'OK' if full.hit1 > 2 * rnd.hit1 else 'CONCERN'}")
print(f"  threshold_0.95 frames kept={t95.pct_frames_kept:.1f}%  -> {'OK' if t95.pct_frames_kept < 100 else 'CONCERN'}")
print(f"  NaNs in numeric summary (excluding random's n/a efficiency cols): "
      f"{int(chk[summ.method != 'random_moment'].isna().sum().sum())}")
print(f"  random mean GT length used: {L:.2f}s")

pq.to_csv(f"{RES}/exp1_per_query.csv", index=False)
un.to_csv(f"{RES}/exp1_per_video.csv", index=False)
summ.to_csv(f"{RES}/exp1_short.csv", index=False)
print("\n", summ[["method", "pct_frames_kept", "hit1", "hit5", "R1_iou0_3", "R1_iou0_5", "R1_iou0_7", "mIoU"]].round(3).to_string(index=False))
print(f"\nwrote {RES}/exp1_short.csv, exp1_per_query.csv, exp1_per_video.csv")
