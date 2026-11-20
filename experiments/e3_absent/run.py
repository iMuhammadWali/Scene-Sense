"""E3 - absent-event queries (Moment of Untruth protocol).
Score = max query-frame cosine similarity in the video's index (team's FAISS search, top-1 similarity).
Positives: a video's own queries. Negatives: (a) in-domain captions from OTHER videos sharing no content word with ANY of the
target video's captions (up to 3/video); (b) out-of-domain hand-written events impossible in Charades homes (3/video).
Index variants: full_5fps and cos_last_drop tau=0.95 (current system).
Usage: python -W ignore experiments/e3_absent/run.py [--limit N] [--force]"""
import sys, time, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd
from sseval import config as C, data, runutil
from sseval.metrics import auroc, bootstrap_indices
from sseval.pruning import select
from sseval.retrieval import build_index, search

HERE = Path(__file__).parent
METHODS = [("full_5fps", {}), ("cos_last_drop", {"tau": 0.95})]
N_NEG = 3

def vrng(seed, vid):
    return np.random.default_rng([seed, zlib.crc32(vid.encode())])

def main():
    ap = runutil.base_parser("E3 absent-event queries"); ap.add_argument("--seed", type=int, default=C.SEED); a = ap.parse_args()
    out = C.results_dir("e3_absent")
    if runutil.is_done(out, a.force): print("[e3] DONE exists, skipping"); return
    df = data.load_annotations(); QE = data.load_query_emb(df)
    vids = data.load_videos(data.select_videos(df, a.limit))
    ood = [l.strip() for l in (HERE / "ood_queries.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    OE = data.encode_texts(ood)
    # content words of every caption (negatives are drawn from the whole annotation file, other videos only)
    cw = [data.content_words(c) for c in df["caption"]]
    by_vid = {v: g.index.values for v, g in df.groupby("vid")}
    vid_arr = df["vid"].values
    items = []   # (vid, kind, text, embedding)
    n_indomain = []
    for v in vids:
        rows = by_vid[v]
        vwords = set().union(*[cw[i] for i in rows])
        rng = vrng(a.seed, v)
        cand = [i for i in range(len(df)) if vid_arr[i] != v and cw[i] and cw[i].isdisjoint(vwords)]
        take = rng.choice(len(cand), size=min(N_NEG, len(cand)), replace=False) if cand else []
        n_indomain.append(len(take))
        for t in take:
            items.append((v, "indomain", df.caption.iloc[cand[t]], QE[cand[t]]))
        for i in rows:
            items.append((v, "positive", df.caption.iloc[i], QE[i]))
        for k in rng.choice(len(ood), size=N_NEG, replace=False):
            items.append((v, "ood", ood[k], OE[k]))
    print(f"[e3] {len(vids)} videos; mean in-domain negatives/video {np.mean(n_indomain):.2f}", flush=True)
    by_v = {}
    for it in items:
        by_v.setdefault(it[0], []).append(it)
    recs, t0 = [], time.time()
    for n, (v, (emb, ts, dur)) in enumerate(vids.items(), 1):
        for method, params in METHODS:
            sel = select(method, emb, ts, dur, **params)
            index, mapping = build_index(sel)
            name = method + ("" if method == "full_5fps" else "_tau0.95")
            for (_, kind, text, qe) in by_v[v]:
                hit = search(index, qe, mapping, 1)[0]
                recs.append((v, kind, text, name, hit["similarity"], hit["timestamp"], sel.n_kept))
        if n % 200 == 0: runutil.progress(n, len(vids), t0)
    sc = pd.DataFrame(recs, columns=["vid", "kind", "text", "method", "score", "top1_ts", "n_kept"])
    sc["score"] = sc["score"].astype("float32")
    sc.to_parquet(out / "scores.parquet", index=False)

    rows, hist = [], []
    bins = np.arange(0.0, 0.5001, 0.005)
    for m, g in sc.groupby("method", sort=False):
        pos = g[g.kind == "positive"].score.values
        neg = {"indomain": g[g.kind == "indomain"].score.values, "ood": g[g.kind == "ood"].score.values}
        neg["all"] = np.concatenate([neg["indomain"], neg["ood"]])
        for nk, nv in neg.items():
            thr_fpr10 = float(np.quantile(nv, 0.90))              # threshold leaving 10% of negatives above it
            rows.append(dict(method=m, negatives=nk, n_pos=len(pos), n_neg=len(nv), auroc=auroc(pos, nv),
                             threshold_sys=C.RETRIEVAL_THRESHOLD, tpr_at_sys=(pos >= C.RETRIEVAL_THRESHOLD).mean(),
                             fpr_at_sys=(nv >= C.RETRIEVAL_THRESHOLD).mean(), thr_at_fpr10=thr_fpr10,
                             tpr_at_fpr10=(pos > thr_fpr10).mean(), fpr_check=(nv > thr_fpr10).mean(),
                             mean_pos=pos.mean(), mean_neg=nv.mean()))
        for kind, gg in g.groupby("kind"):
            h, _ = np.histogram(np.clip(gg.score.values, bins[0], bins[-1] - 1e-9), bins=bins)
            hist += [dict(method=m, kind=kind, bin_left=bins[i], bin_right=bins[i + 1], count=int(h[i])) for i in range(len(h))]
    res = pd.DataFrame(rows)
    # bootstrap CI for pooled AUROC (resample queries within each class)
    rng = np.random.default_rng(a.seed); lo, hi = [], []
    for r in res.itertuples():
        g = sc[sc.method == r.method]
        pos = g[g.kind == "positive"].score.values
        nv = g[g.kind.isin(["indomain", "ood"] if r.negatives == "all" else [r.negatives])].score.values
        b = [auroc(rng.choice(pos, len(pos)), rng.choice(nv, len(nv))) for _ in range(200)]
        lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
    res["auroc_lo"], res["auroc_hi"] = lo, hi
    res.to_csv(out / "absent_results.csv", index=False, encoding="utf-8")
    pd.DataFrame(hist).to_csv(out / "score_hist.csv", index=False, encoding="utf-8")
    runutil.write_meta(out, limit=a.limit, n_videos=len(vids), n_positive=int((sc.kind == "positive").sum() // 2),
                       n_indomain=int((sc.kind == "indomain").sum() // 2), n_ood=int((sc.kind == "ood").sum() // 2),
                       text_encoder="team CLIP, single projection" if not C._TEAM_TEXT else "team encode_text (double projection)")
    runutil.mark_done(out)
    print(res.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
