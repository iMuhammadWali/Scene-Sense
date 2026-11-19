"""E10 aggregation: metrics (Hit@1 / Hit@3 / Hit@5 / R@1 IoU>=0.5 with bootstrap CIs), fusion with dev-chosen w, negative rejection AUROC,
cost table, figure. Reads <out>/raw/*.json written by run.py. Importable (aggregate(out, items, plans)) or runnable on its own."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd
from sseval import config as C, rerank as R
from sseval.metrics import bootstrap_indices, bootstrap_ci, auroc

WS = (0.25, 0.5, 1.0, 2.0, 4.0)
RRF_K = 2.0
NC = 10

def logit_of(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

class Data:
    """Per-kind matrices. rec[kind] = list of raw records in sample order."""
    def __init__(self, out: Path, items, plans):
        self.out = out
        self.q = {k: [x for x in items if x["kind"] == k] for k in ("pos", "neg")}
        self.pl = {k: [plans[(k, x["qid"])] for x in self.q[k]] for k in self.q}
        self.raw = {k: [json.loads((out / "raw" / f"{k}_{x['qid']}.json").read_text()) if (out / "raw" / f"{k}_{x['qid']}.json").exists()
                        else {"scores": {}} for x in self.q[k]] for k in self.q}
        for k in self.q:
            n = len(self.q[k])
            t = np.full((n, NC), np.nan); sim = np.full((n, NC), np.nan)
            for i, p in enumerate(self.pl[k]):
                for j, c in enumerate(p["cands"][:NC]):
                    t[i, j], sim[i, j] = c["t"], c["sim"]
            setattr(self, f"t_{k}", t); setattr(self, f"clip_{k}", sim)
        P = self.q["pos"]
        self.gs = np.array([x["gs"] for x in P]); self.ge = np.array([x["ge"] for x in P]); self.dur = np.array([p["dur"] for p in self.pl["pos"]])
        self.real = np.zeros((len(P), NC), bool)                 # True where the candidate is a real temporal group (cand_src == 'group')
        for i, p in enumerate(self.pl["pos"]):
            for j, c in enumerate(p["cands"][:NC]):
                self.real[i, j] = c.get("src") == "group"
        self.hit = ((self.t_pos >= self.gs[:, None]) & (self.t_pos <= self.ge[:, None])).astype(float)
        self.hit[np.isnan(self.t_pos)] = np.nan
        self.iou = R.window_iou(self.t_pos, self.gs, self.ge, self.dur, 8.0)

    def itm(self, kind, res, variant, crange, offsets, how="max"):
        """[n, len(crange)] aggregated ITM logit per candidate (NaN: candidate absent or a needed frame not scored)."""
        n = len(self.q[kind]); out = np.full((n, len(crange)), np.nan)
        for i, rec in enumerate(self.raw[kind]):
            sc = rec["scores"]
            for jj, c in enumerate(crange):
                if c >= len(self.pl[kind][i]["cands"]):
                    continue
                vals = [sc.get(f"{res}|{variant}|{c}|{o:+.0f}") for o in offsets]
                if any(v is None for v in vals):
                    continue
                lg = np.array([v[0] for v in vals])
                out[i, jj] = lg.max() if how == "max" else float(logit_of(np.mean(1 / (1 + np.exp(-lg)))))
        return out

def complete_rows(clip, S):
    """Rows where S is scored for every existing candidate."""
    return np.flatnonzero((np.isnan(S) == np.isnan(clip[:, :S.shape[1]])).all(axis=1) & ~np.isnan(S).all(axis=1))

def pad(S, width):
    out = np.full((S.shape[0], width), np.nan); out[:, :S.shape[1]] = S; return out

# ------------------------------------------------------------------------------------------------------------ metrics on a row set
def method_stats(D, ch, rows):
    """ch: chosen column per row (indexes into the full arrays via rows). -> dict of per-query vectors."""
    r = np.asarray(rows); c = np.asarray(ch)
    h1 = D.hit[r, c]; i1 = D.iou[r, c]
    return dict(hit1=h1, r1=(i1 >= 0.5).astype(float), iou=i1)

def top3_hit(D, score, rows):
    r = np.asarray(rows); s = np.where(np.isnan(score[r]), -np.inf, score[r])
    order = np.argsort(-s, axis=1, kind="stable")[:, :3]
    return np.nan_to_num(np.take_along_axis(D.hit[r], order, 1), nan=0).max(axis=1)

def ci(x, idx):
    return bootstrap_ci(np.asarray(x, float), idx)

def evaluate(D, name_scores: dict, rows, tag, add, ref="clip"):
    """name_scores: method -> score matrix [n, NC] over ALL positives (NaN where unavailable). rows: evaluated query rows.
    Writes Hit@1, Hit@3, Hit@5 (candidate recall = identical for every order), R@1@0.5, mIoU, and paired deltas vs `ref`."""
    rows = np.asarray(rows); n = len(rows)
    idx = bootstrap_indices(n)
    vecs = {}
    for m, S in name_scores.items():
        ch = np.argmax(np.where(np.isnan(S[rows]), -np.inf, S[rows]), axis=1)
        v = method_stats(D, ch, rows); v["hit3"] = top3_hit(D, S, rows); vecs[m] = v
    avail_c = ~np.isnan(name_scores[ref][rows]) if ref in name_scores else np.ones((n, D.hit.shape[1]), bool)
    hit5 = np.nan_to_num(np.where(avail_c, D.hit[rows], 0.0), nan=0).max(axis=1)      # oracle over the candidates actually present
    for m, v in vecs.items():
        for met in ("hit1", "hit3", "r1", "iou"):
            lo, hi = ci(v[met], idx)
            add(tag, m, met, n, float(np.mean(v[met])), lo, hi)
        lo, hi = ci(hit5, idx); add(tag, m, "hit5_cand_recall", n, float(hit5.mean()), lo, hi)
        if m != ref and ref in vecs:
            for met in ("hit1", "r1"):
                d = v[met] - vecs[ref][met]; lo, hi = ci(d, idx)
                add(tag, m, f"delta_{met}_vs_{ref}", n, float(d.mean()), lo, hi)
    return vecs

# ------------------------------------------------------------------------------------------------------------ main aggregation
def aggregate(out: Path, items, plans):
    D = Data(out, items, plans)
    rows_summary = []
    def add(tag, method, metric, n, v, lo=np.nan, hi=np.nan):
        rows_summary.append(dict(subset=tag, method=method, metric=metric, n=n, value=v, lo=lo, hi=hi))
    clip = D.clip_pos
    cr5 = range(5)
    # --- score matrices (positives)
    mats = {}
    mats["itm_K1_asis"] = pad(D.itm("pos", 384, "asis", cr5, (0.0,)), NC)
    mats["itm_K1_strip"] = pad(D.itm("pos", 384, "strip", cr5, (0.0,)), NC)
    for how in ("max", "mean"):
        mats[f"itm_K3{how}_asis"] = pad(D.itm("pos", 384, "asis", cr5, (-1.0, 0.0, 1.0), how), NC)
        mats[f"itm_K3{how}_strip"] = pad(D.itm("pos", 384, "strip", cr5, (-1.0, 0.0, 1.0), how), NC)
    mats["itm_K1_288_asis"] = pad(D.itm("pos", 288, "asis", cr5, (0.0,)), NC)
    mats["itm_K1_top10_asis"] = D.itm("pos", 384, "asis", range(10), (0.0,))          # needs cands 0-9; s1 gives 0-4, s4 gives 5-9
    top5_clip = np.where(np.arange(NC)[None, :] < 5, clip, np.nan)
    clip10 = clip
    # hit matrix restricted to 5 candidates for top-5 sets
    def mask5(M):
        return np.where(np.arange(NC)[None, :] < 5, M, np.nan)
    avail = {k: complete_rows(top5_clip, mask5(v)) for k, v in mats.items() if k != "itm_K1_top10_asis"}
    avail["itm_K1_top10_asis"] = complete_rows(clip10, mats["itm_K1_top10_asis"])
    D_hit_full = D.hit
    info = {k: len(v) for k, v in avail.items()}
    print("[e10] rows with complete scores:", info, flush=True)

    def run_config(cfg, S, rows, cwidth, real=False):
        """Evaluate CLIP, BLIP-only, fused (z, rank) on `rows` for matrices truncated to cwidth candidates."""
        if len(rows) < 8:
            return None
        if real:
            cfg = cfg + "_real"

        keep = (np.arange(NC)[None, :] < cwidth) & (D.real if real else True)
        c = np.where(keep, clip, np.nan); s = np.where(keep, S, np.nan)
        ms = {"clip": c, "blip": s}
        for w in WS:
            ms[f"fuse_z_w{w:g}"] = R.fuse_z(c, s, w)
            ms[f"fuse_rank_w{w:g}"] = R.fuse_rank(c, s, w, RRF_K)
        hit = np.nan_to_num(D.hit, nan=0.0)
        rows = np.asarray(rows)
        h = len(rows) // 2
        dev, test = rows[:h], rows[h:]
        chosen = {}
        for fn, nm in ((R.fuse_z, "fuse_z"), (lambda a, b, w: R.fuse_rank(a, b, w, RRF_K), "fuse_rank")):
            w, scores = R.choose_w(c[rows], s[rows], hit[rows], WS, fn, idx=np.arange(h))
            chosen[nm] = w
            ms[f"{nm}_devw"] = fn(c, s, w)
            add(cfg, f"{nm}_devw", "chosen_w", h, w)
            for ww, sc in scores.items():
                add(cfg, f"{nm}_w{ww:g}", "dev_hit1", h, sc)
        add_tag = lambda tag, *a: add(f"{cfg}|{tag}", *a)
        evaluate(D, ms, rows, f"{cfg}|all", lambda t, *a: add(t, *a))
        evaluate(D, {k: ms[k] for k in ("clip", "blip", "fuse_z_devw", "fuse_rank_devw")}, test, f"{cfg}|test_half", lambda t, *a: add(t, *a))
        evaluate(D, {k: ms[k] for k in ("clip", "blip", "fuse_z_devw", "fuse_rank_devw")}, dev, f"{cfg}|dev_half", lambda t, *a: add(t, *a))
        return ms, chosen

    results = {}
    main_rows = avail["itm_K1_asis"]
    results["K1_asis"] = run_config("K1_asis", mats["itm_K1_asis"], main_rows, 5)
    results["K1_asis_real"] = run_config("K1_asis", mats["itm_K1_asis"], main_rows, 5, real=True)
    results["K1_strip"] = run_config("K1_strip", mats["itm_K1_strip"], avail["itm_K1_strip"], 5)
    results["K1_strip_real"] = run_config("K1_strip", mats["itm_K1_strip"], avail["itm_K1_strip"], 5, real=True)
    for nm in ("K3max_asis", "K3mean_asis", "K3max_strip"):
        results[nm] = run_config(nm, mats[f"itm_{nm}"], avail[f"itm_{nm}"], 5)
        results[nm + "_real"] = run_config(nm, mats[f"itm_{nm}"], avail[f"itm_{nm}"], 5, real=True)
    # same-subset references for the K=3 / 288 / top-10 comparisons: K=1 on the same rows
    sub = avail["itm_K3max_asis"]
    results["K1_asis_on_K3rows"] = run_config("K1_asis_on_K3rows", mats["itm_K1_asis"], sub, 5)
    results["K1_asis_on_K3rows_real"] = run_config("K1_asis_on_K3rows", mats["itm_K1_asis"], sub, 5, real=True)
    results["K1_288_asis"] = run_config("K1_288_asis", mats["itm_K1_288_asis"], avail["itm_K1_288_asis"], 5)
    results["K1_288_asis_real"] = run_config("K1_288_asis", mats["itm_K1_288_asis"], avail["itm_K1_288_asis"], 5, real=True)
    results["K1_asis_on_288rows"] = run_config("K1_asis_on_288rows", mats["itm_K1_asis"], avail["itm_K1_288_asis"], 5)
    results["K1_asis_on_288rows_real"] = run_config("K1_asis_on_288rows", mats["itm_K1_asis"], avail["itm_K1_288_asis"], 5, real=True)
    results["top10_asis"] = run_config("top10_asis", mats["itm_K1_top10_asis"], avail["itm_K1_top10_asis"], 10)
    results["top10_asis_real"] = run_config("top10_asis", mats["itm_K1_top10_asis"], avail["itm_K1_top10_asis"], 10, real=True)
    results["K1_asis_top5_on_top10rows"] = run_config("K1_asis_top5_on_top10rows", mats["itm_K1_asis"], avail["itm_K1_top10_asis"], 5)
    results["K1_asis_top5_on_top10rows_real"] = run_config("K1_asis_top5_on_top10rows", mats["itm_K1_asis"], avail["itm_K1_top10_asis"], 5, real=True)

    # DeepSeek (E8 C2K3) on the same queries, paired
    e8 = pd.read_csv(C.RESULTS_DIR / "e8_slowpath" / "per_query.csv"); e8 = e8[(e8.cond == "C2K3") & (e8.kind == "pos")].set_index("qid")
    qids = np.array([x["qid"] for x in D.q["pos"]])
    ds = np.array([float(e8.loc[q, "slow_hit1"]) if q in e8.index else np.nan for q in qids])
    ds_iou_w = np.array([float(e8.loc[q, "slow_iou_wide"]) if q in e8.index else np.nan for q in qids])
    r = np.asarray(main_rows); idx = bootstrap_indices(len(r))
    add("K1_asis|all", "deepseek_C2K3", "hit1", len(r), float(np.nanmean(ds[r])), *ci(ds[r], idx))
    ms = results["K1_asis"][0]
    for m in ("blip", "fuse_z_devw", "fuse_rank_devw"):
        ch = np.argmax(np.where(np.isnan(ms[m][r]), -np.inf, ms[m][r]), axis=1)
        d = D.hit[r, ch] - ds[r]
        add("K1_asis|all", m, "delta_hit1_vs_deepseek", len(r), float(d.mean()), *ci(d, idx))
    oracle = np.nan_to_num(D.hit[r][:, :5], nan=0).max(axis=1)
    add("K1_asis|all", "oracle_best_of_5", "hit1", len(r), float(oracle.mean()), *ci(oracle, idx))

    # --- absent events: positives vs negatives
    negs = {}
    for nm, args in {"itm_K1_asis": (384, "asis", (0.0,), "max"), "itm_K1_strip": (384, "strip", (0.0,), "max"),
                     "itm_K3max_asis": (384, "asis", (-1.0, 0.0, 1.0), "max"), "itm_K1_288_asis": (288, "asis", (0.0,), "max")}.items():
        negs[nm] = pad(D.itm("neg", args[0], args[1], cr5, args[2], args[3]), NC)
    tn = D.clip_neg; tn5 = mask5(tn)
    def rowmax(M):
        return np.where(np.isnan(M).all(axis=1), np.nan, np.nanmax(np.where(np.isnan(M), -np.inf, M), axis=1))
    def rowmean_p(M):                     # mean ITM prob over candidates (as logit)
        p = 1 / (1 + np.exp(-M)); return np.where(np.isnan(M).all(axis=1), np.nan, logit_of(np.nanmean(p, axis=1)))
    def at_top1(M):
        return M[:, 0]
    pos_clip_max = rowmax(top5_clip); neg_clip_max = rowmax(tn5)
    def au(tag, ps, ns, name):
        ok_p, ok_n = ~np.isnan(ps), ~np.isnan(ns)
        if ok_p.sum() < 5 or ok_n.sum() < 5:
            return ok_p, ok_n
        a, lo, hi = R.auroc_ci(ps[ok_p], ns[ok_n])
        add(tag, name, "auroc_pos_vs_neg", f"{ok_p.sum()}/{ok_n.sum()}", a, lo, hi)
        thr = np.sort(ps[ok_p])[max(0, int(np.ceil(0.05 * ok_p.sum())) - 1)]
        add(tag, name, "true_none_neg_at_5pct_none_pos", f"{ok_p.sum()}/{ok_n.sum()}", float((ns[ok_n] <= thr).mean()))
        return ok_p, ok_n
    for nm in negs:
        Sp = mats[nm]
        okp = ~np.isnan(rowmax(mask5(Sp))); okn = ~np.isnan(rowmax(negs[nm]))
        # restrict CLIP to the same query subsets for a paired comparison
        tag = f"absent|{nm}"
        au(tag, np.where(okp, rowmax(mask5(Sp)), np.nan), np.where(okn, rowmax(negs[nm]), np.nan), "itm_max")
        au(tag, np.where(okp, rowmean_p(mask5(Sp)), np.nan), np.where(okn, rowmean_p(negs[nm]), np.nan), "itm_mean_prob")
        au(tag, np.where(okp, at_top1(mask5(Sp)), np.nan), np.where(okn, at_top1(negs[nm]), np.nan), "itm_at_clip_top1")
        au(tag, np.where(okp, pos_clip_max, np.nan), np.where(okn, neg_clip_max, np.nan), "clip_max")
        # combined: z over all (pos+neg) of each score, summed
        pm, nmx = rowmax(mask5(Sp)), rowmax(negs[nm])
        allv = np.concatenate([pm[okp], nmx[okn]]); allc = np.concatenate([pos_clip_max[okp], neg_clip_max[okn]])
        zi = lambda x: (x - allv.mean()) / allv.std(); zc = lambda x: (x - allc.mean()) / allc.std()
        cmb_p = np.where(okp, zi(pm) + zc(pos_clip_max), np.nan); cmb_n = np.where(okn, zi(nmx) + zc(neg_clip_max), np.nan)
        au(tag, cmb_p, cmb_n, "z(itm_max)+z(clip_max)")
    sm = pd.DataFrame(rows_summary)
    sm.to_csv(out / "summary.csv", index=False)
    per_query(out, D, mats, negs, results)
    cost_table(out, sm)
    make_figure(out, sm)
    with pd.option_context("display.width", 220, "display.max_rows", 400, "display.float_format", "{:.3f}".format):
        show = sm[(sm.metric.isin(["hit1", "hit3", "r1", "auroc_pos_vs_neg"])) & (~sm.method.str.contains("_w[0-9]", regex=True))]
        print(show.to_string(index=False))

def per_query(out, D, mats, negs, results):
    n = len(D.q["pos"]); rows = []
    for i, x in enumerate(D.q["pos"]):
        r = dict(kind="pos", qid=x["qid"], vid=x["vid"], caption=x["caption"], gt_s=x["gs"], gt_e=x["ge"], n_cands=int(np.sum(~np.isnan(D.t_pos[i]))),
                 cand_t=";".join(f"{t:.2f}" for t in D.t_pos[i] if not np.isnan(t)), clip_sim=";".join(f"{s:.4f}" for s in D.clip_pos[i] if not np.isnan(s)),
                 cand_src=";".join("group" if D.real[i, j] else "pad" for j in range(NC) if not np.isnan(D.t_pos[i, j])),
                 cand_hit=";".join(str(int(h)) for h in D.hit[i] if not np.isnan(h)), cand_iou8s=";".join(f"{u:.3f}" for u in D.iou[i] if not np.isnan(u)))
        for nm, M in mats.items():
            if not np.isnan(M[i]).all():
                r[nm + "_logit"] = ";".join(f"{v:.3f}" for v in M[i] if not np.isnan(v))
        for cfg in ("K1_asis", "K3max_asis", "K1_asis_real", "K3max_asis_real"):
            if results.get(cfg):
                ms = results[cfg][0]
                for m in ("blip", "fuse_z_devw", "fuse_rank_devw"):
                    if not np.isnan(ms[m][i]).all():
                        ch = int(np.argmax(np.where(np.isnan(ms[m][i]), -np.inf, ms[m][i])))
                        r[f"{cfg}_{m}_choice"] = ch; r[f"{cfg}_{m}_hit1"] = D.hit[i, ch]; r[f"{cfg}_{m}_iou"] = D.iou[i, ch]
        r["clip_hit1"] = D.hit[i, 0]; r["clip_iou"] = D.iou[i, 0]
        rows.append(r)
    for i, x in enumerate(D.q["neg"]):
        r = dict(kind="neg", qid=x["qid"], vid=x["vid"], caption=x["caption"], n_cands=int(np.sum(~np.isnan(D.t_neg[i]))),
                 cand_t=";".join(f"{t:.2f}" for t in D.t_neg[i] if not np.isnan(t)), clip_sim=";".join(f"{s:.4f}" for s in D.clip_neg[i] if not np.isnan(s)))
        for nm, M in negs.items():
            if not np.isnan(M[i]).all():
                r[nm + "_logit"] = ";".join(f"{v:.3f}" for v in M[i] if not np.isnan(v))
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "per_query.csv", index=False)

# ------------------------------------------------------------------------------------------------------------ cost table
def cost_table(out: Path, sm: pd.DataFrame):
    rows = []
    e6 = pd.read_csv(C.RESULTS_DIR / "e6_latency" / "stage_summary.csv")
    q6 = e6[(e6.phase == "query")].set_index("stage")["mean_s"]
    retrieval = float(q6["clip_text_embedding"] + q6["faiss_search"])
    wall = float(q6["TOTAL (wall)"])
    tm = pd.read_csv(out / "timing.csv") if (out / "timing.csv").exists() else None
    def hit(cfg, m, tag="all"):
        r = sm[(sm.subset == f"{cfg}|{tag}") & (sm.method == m) & (sm.metric == "hit1")]
        return (float(r.value.iloc[0]), float(r.lo.iloc[0]), float(r.hi.iloc[0])) if len(r) else (np.nan,) * 3
    e8s = pd.read_csv(C.RESULTS_DIR / "e8_slowpath" / "summary.csv"); e8s = e8s[(e8s.condition == "C2K3") & (e8s.system == "slow_vlm")].iloc[0]
    e9s = pd.read_csv(C.RESULTS_DIR / "e9_vlm_scoring" / "summary_C2.csv")
    def blip_s(K, res=384):
        if tm is None: return np.nan, np.nan
        g = tm[(tm.K == K) & (tm.res == res)]
        return float(g.itm_s.mean()), float(g.decode_s.mean())
    for suffix, tagname in (("_real", "[real groups]"), ("", "[padded to 5]")):
        for name, (cfg, m, K) in {"fast path (CLIP only)": ("K1_asis", "clip", 0), "fast + BLIP-ITM K=1, fused z": ("K1_asis", "fuse_z_devw", 1),
                                  "fast + BLIP-ITM K=1, BLIP order": ("K1_asis", "blip", 1), "fast + BLIP-ITM K=3 (max), fused z": ("K3max_asis", "fuse_z_devw", 3),
                                  "fast + BLIP-ITM K=3 (max), BLIP order": ("K3max_asis", "blip", 3)}.items():
            cfg = cfg + suffix
            h = hit(cfg, m)
            itm, dec = blip_s(K) if K else (0.0, 0.0)
            rows.append(dict(system=f"{name} {tagname}", n=int(sm[(sm.subset == f"{cfg}|all") & (sm.method == m) & (sm.metric == "hit1")].n.iloc[0]) if not np.isnan(h[0]) else 0,
                             hit1=h[0], lo=h[1], hi=h[2], added_itm_s=itm, added_decode_s=dec, added_total_s=(itm + dec) if K else 0.0,
                             base_retrieval_s=retrieval, base_pipeline_wall_s=wall))
    rows.append(dict(system="DeepSeek slow path (E8 C2K3, API) [padded to 5]", n=300, hit1=float(e8s.hit1), lo=float(e8s.hit1_lo), hi=float(e8s.hit1_hi), added_itm_s=np.nan,
                     added_decode_s=np.nan, added_total_s=float(e8s.latency_mean_s), base_retrieval_s=retrieval, base_pipeline_wall_s=wall))
    h9 = e9s[(e9s.set == "full") & (e9s.section == "ranking") & (e9s.name == "vlm_only") & (e9s.metric == "hit1")]
    if len(h9):
        rows.append(dict(system="SmolVLM2-500M yes/no re-rank (E9 C2, local) [padded to 5]", n=300, hit1=float(h9.value.iloc[0]), lo=float(h9.lo.iloc[0]), hi=float(h9.hi.iloc[0]),
                         added_total_s=2.97 + 0.10 + 0.29, base_retrieval_s=retrieval, base_pipeline_wall_s=wall))
    pd.DataFrame(rows).to_csv(out / "cost_table.csv", index=False)

# ------------------------------------------------------------------------------------------------------------ figure
def make_figure(out: Path, sm: pd.DataFrame):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ct = pd.read_csv(out / "cost_table.csv")
    def g(cfg, m, tag="all", metric="hit1"):
        r = sm[(sm.subset == f"{cfg}|{tag}") & (sm.method == m) & (sm.metric == metric)]
        return (100 * float(r.value.iloc[0]), 100 * float(r.lo.iloc[0]), 100 * float(r.hi.iloc[0])) if len(r) else None
    R_ = "K1_asis_real"
    spec = [("Fast path\n(CLIP)", g(R_, "clip"), "#9aa0a6"), ("BLIP-ITM\nonly (K=1)", g(R_, "blip"), "#E69F00"),
            ("CLIP + BLIP\nz-fusion (K=1)", g(R_, "fuse_z_devw"), "#0072B2"), ("CLIP + BLIP\nrank-fusion (K=1)", g(R_, "fuse_rank_devw"), "#56B4E9"),
            ("Oracle\n(real groups)", g(R_, "clip", metric="hit5_cand_recall"), "#d8d8d8"),
            ("DeepSeek (E8,\npadded, ref.)", g("K1_asis", "deepseek_C2K3"), "#009E73")]
    spec = [s for s in spec if s[1]]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    for k, (lab, (v, lo, hi), c) in enumerate(spec):
        ax.bar(k, v, 0.62, color=c, yerr=[[v - lo], [hi - v]], capsize=4, error_kw=dict(lw=1, color="#333")); ax.text(k, hi + 1.5, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(spec))); ax.set_xticklabels([s[0] for s in spec], fontsize=8); ax.set_ylim(0, 105); ax.set_ylabel("Hit@1 (%), 95% bootstrap CI")
    n = int(sm[(sm.subset == "K1_asis|all") & (sm.method == "clip") & (sm.metric == "hit1")].n.iloc[0])
    ax.set_title(f"Hit@1 by method, REAL groups only (no padding), n={n} positives\n(fusion w chosen on the first half; DeepSeek bar used padded candidates)", fontsize=10)
    ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    col = {"fast path (CLIP only)": "#9aa0a6", "DeepSeek": "#009E73", "SmolVLM2": "#CC79A7", "K=1": "#0072B2", "K=3": "#56B4E9"}
    for _, r in ct.iterrows():
        if np.isnan(r.hit1) or "[padded" in r.system and "DeepSeek" not in r.system and "Smol" not in r.system: continue
        x = r.base_retrieval_s + (0 if np.isnan(r.added_total_s) else r.added_total_s)
        c = next((v for k, v in col.items() if k in r.system), "#444")
        ax.errorbar(x, 100 * r.hit1, yerr=[[100 * (r.hit1 - r.lo)], [100 * (r.hi - r.hit1)]], fmt="o", color=c, capsize=3)
        ax.annotate(r.system.replace(" (E8 C2K3, API)", "").replace(" (E9 C2, local)", "").replace(" [real groups]", ""), (x, 100 * r.hit1), textcoords="offset points", xytext=(6, -10), fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlabel("query latency incl. added re-ranker/VLM (s, log), 2 CPU threads"); ax.set_ylabel("Hit@1 (%)")
    ax.set_title("Hit@1 vs latency (retrieval core 0.016 s + added stage)", fontsize=10); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / "figures" / "e10_blip_rerank.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    import run
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else C.results_dir("e10_blip_rerank")
    e8, items, plans = run.build_items(None)
    aggregate(out, items, plans)
