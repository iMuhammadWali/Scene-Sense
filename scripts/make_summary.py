"""results/SUMMARY.md from the result CSVs. Observations are generated from the numbers only. Usage: python -W ignore scripts/make_summary.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from sseval import config as C

R = C.RESULTS_DIR
def rd(p):
    f = R / p
    return pd.read_csv(f) if f.exists() else None
def meta(e):
    f = R / e / "meta.json"
    return json.loads(f.read_text()) if f.exists() else {}
def pct(x, d=1): return "n/a" if pd.isna(x) else f"{100 * x:.{d}f}"
def ci(v, lo, hi): return f"{pct(v)} [{pct(lo)}, {pct(hi)}]"
def overlap(a_lo, a_hi, b_lo, b_hi): return not (a_hi < b_lo or b_hi < a_lo)
def md(df): 
    if df is None or len(df) == 0: return "_(no data)_\n"
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in df.itertuples(index=False): out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"
DAG = "†"
def nice(method, params=""):
    lab = {"uniform": "Uniform", "cos_last_drop": "Cosine vs last kept", "cos_consec_drop": "Cosine vs previous", "streammem_merge": f"StreamMem merge {DAG}",
           "maxinfo_fast": f"MaxInfo-fast {DAG}", "maxinfo_slow": f"MaxInfo-slow {DAG}", "maxinfo_chunk": f"MaxInfo-chunk {DAG}",
           "hybrid": f"Hybrid {DAG}", "full_5fps": "Full 5 FPS", "random_moment": "Chance (random moment)"}[method]
    return lab + (f" ({params.replace(';', ', ')})" if isinstance(params, str) and params else "")

L = []; obs = []
e1m, e1bm, e3m, e6m = meta("e1_pruning"), meta("e1b_long"), meta("e3_absent"), meta("e6_latency")
hw = (e1m or e1bm or e3m or e6m).get("hardware", {})
L += ["# Scene Sense - Tier-1 evaluation summary", ""]
sub = [f"{n}: limit={m.get('limit')}" for n, m in (("E1", e1m), ("E1b", e1bm), ("E3", e3m)) if m.get("limit")]
if sub: L += [f"> **PARTIAL / SUBSET RUN** ({'; '.join(sub)}). Do not quote these numbers as final.", ""]
L += ["## Setup", "",
      "- Dataset: Charades-STA test split (full: 1,334 videos, 3,720 queries). Frames extracted at 5 FPS and CLIP-encoded once with the team's `extract_candidates` + `CLIPEncoder` (openai/clip-vit-base-patch32).",
      f"- Hardware: {hw.get('cpu_model', 'CPU')} x{hw.get('cpu_count', '?')} cores, {hw.get('mem_gb', '?')} GB RAM, no GPU; torch {hw.get('torch', '?')}, transformers {hw.get('transformers', '?')}, faiss {hw.get('faiss', '?')}.",
      f"- Videos / queries actually used: E1 {e1m.get('n_videos', 'n/a')} / {e1m.get('n_queries', 'n/a')}; E1b {e1bm.get('n_videos', 'n/a')} videos; E3 {e3m.get('n_videos', 'n/a')} videos; E6 {e6m.get('n_videos', 'n/a')} videos / {e6m.get('n_queries', 'n/a')} queries.",
      "- Retrieval protocol (all methods): team's FAISS flat index -> top-5 -> team's `group_temporally` (gap 2 s) -> predicted moment = union of coverage spans of the group containing the top-1 hit. 95% CIs: 1,000 bootstrap resamples over queries (fixed seed).",
      f"- {DAG} MaxInfo and StreamMem are **our re-implementation for retrieval indexing, adapted from methods designed for VLM frame selection**; they are not the authors' pipelines.",
      "- Text embeddings use a single-projection CLIP text encoder (see `docs`/report: the team's `encode_text` double-projects under transformers 4.57). Image embeddings come from the team's encoder unchanged.", ""]

op = rd("e1_pruning/operating_points.csv")
if op is not None:
    full = op[op.config == "full_5fps"].iloc[0]; rnd = op[op.method == "random_moment"].iloc[0]
    L += ["## E1 - frame selection at indexing time", "", "Reference and system-default rows (mean over videos; R@1 = recall at top-1 with IoU threshold):", ""]
    sysd = op[op.config == "cos_last_drop[tau=0.95]"]
    ref = pd.concat([op[op.config == "full_5fps"], sysd, op[op.method == "random_moment"]])
    sysnote = "(system default = cosine vs last kept, tau=0.95)"
    t = pd.DataFrame({"method": [nice(r.method, r.params) for r in ref.itertuples()], "% kept": ref.pct_kept.round(1),
                      "Hit@1 [95% CI]": [ci(r.hit1, r.hit1_lo, r.hit1_hi) for r in ref.itertuples()], "Hit@5": (100 * ref.hit5).round(1),
                      "R@1 IoU0.3": (100 * ref.r1_iou03).round(1), "R@1 IoU0.5 [95% CI]": [ci(r.r1_iou05, r.r1_iou05_lo, r.r1_iou05_hi) for r in ref.itertuples()],
                      "R@1 IoU0.7": (100 * ref.r1_iou07).round(1), "mIoU": (100 * ref.miou).round(1), "mean pred. span (s)": ref.pred_len_mean_s.round(1)})
    L += [md(t), sysnote, "", f"Mean ground-truth moment length: {op.gt_len_mean_s.iloc[0]:.1f} s.", ""]
    bt = rd("e1_pruning/budget_table.csv")
    if bt is not None:
        for crit, ttl in (("r1_iou05", "best operating point per method by R@1 IoU>=0.5 (plan)"), ("hit1", "best operating point per method by Hit@1 (protocol-robust companion)")):
            b = bt[(bt.criterion == crit) & ~bt.params.str.startswith("(no")]
            L += [f"### Budget table - {ttl}", ""]
            tb = pd.DataFrame({"budget (% frames)": b.budget_pct, "method": [nice(m) for m in b.method], "params": b.params, "% kept": b.pct_kept.round(1),
                               "R@1 IoU0.5 [95% CI]": [ci(r.r1_iou05, r.r1_iou05_lo, r.r1_iou05_hi) for r in b.itertuples()],
                               "Hit@1 [95% CI]": [ci(r.hit1, r.hit1_lo, r.hit1_hi) for r in b.itertuples()]})
            L += [md(tb.sort_values(["budget (% frames)", "method"]))]
    L += ["Selection cost (mean ms per video, per method family, over all swept configs): ", ""]
    sel = op[~op.method.isin(["random_moment"])].groupby("method").select_ms_per_video.agg(["mean", "max"]).round(2).reset_index()
    sel["method"] = [nice(m) for m in sel.method]; L += [md(sel.rename(columns={"mean": "mean ms/video", "max": "max ms/video (any config)"}))]
    # observations
    o = obs.append
    o(f"Full 5 FPS indexing reaches Hit@1 = {pct(full.hit1)}% (95% CI {pct(full.hit1_lo)}-{pct(full.hit1_hi)}) against chance {pct(rnd.hit1)}% - "
      + ("the CI lies above chance." if full.hit1_lo > rnd.hit1 else "the CI does NOT exclude chance."))
    if len(sysd):
        s = sysd.iloc[0]
        o(f"The current system default (cosine vs last kept, tau=0.95) keeps {s.pct_kept:.1f}% of frames; its Hit@1 {pct(s.hit1)}% vs {pct(full.hit1)}% for full 5 FPS - "
          + ("within overlapping CIs (not a significant difference)." if overlap(s.hit1_lo, s.hit1_hi, full.hit1_lo, full.hit1_hi) else "CIs do not overlap."))
    cand = op[(op.pct_kept <= 20) & ~op.method.isin(["random_moment", "full_5fps"])]
    if len(cand):
        top = cand.sort_values("hit1", ascending=False).iloc[0]
        uni = cand[cand.method == "uniform"].sort_values("hit1", ascending=False)
        txt = f"Among operating points keeping <=20% of frames, the highest Hit@1 is {nice(top.method, top.params)} at {top.pct_kept:.1f}% kept: {pct(top.hit1)}% [{pct(top.hit1_lo)}, {pct(top.hit1_hi)}]"
        if len(uni):
            u = uni.iloc[0]
            txt += f"; best uniform within the same budget: {pct(u.hit1)}% [{pct(u.hit1_lo)}, {pct(u.hit1_hi)}] - " + ("overlapping CIs, not significant." if overlap(top.hit1_lo, top.hit1_hi, u.hit1_lo, u.hit1_hi) else "non-overlapping CIs.")
        o(txt)
    sp = op[~op.method.isin(["random_moment", "full_5fps"])].sort_values("r1_iou05", ascending=False).iloc[0]
    o(f"R@1 IoU>=0.5 is strongly span-dependent under this protocol: full 5 FPS scores {pct(full.r1_iou05)}% with a mean predicted span of {full.pred_len_mean_s:.1f} s "
      f"(ground truth {op.gt_len_mean_s.iloc[0]:.1f} s), whereas the best sparse point ({nice(sp.method, sp.params)}, {sp.pct_kept:.1f}% kept) scores {pct(sp.r1_iou05)}% with span {sp.pred_len_mean_s:.1f} s, "
      f"and the random-interval chance reference scores {pct(rnd.r1_iou05)}%. Sparser indexes inherit longer coverage spans, so R@1 gains need not reflect better matching; Hit@k is the cleaner comparison.")
    lastk = op[op.method == "cos_last_drop"]; cons = op[op.method == "cos_consec_drop"]
    if len(lastk) and len(cons):
        a, b = lastk[lastk.config == "cos_last_drop[tau=0.95]"], cons[cons.config == "cos_consec_drop[tau=0.95]"]
        if len(a) and len(b):
            o(f"At tau=0.95, comparing against the last kept frame keeps {a.pct_kept.iloc[0]:.1f}% of frames vs {b.pct_kept.iloc[0]:.1f}% when comparing consecutive frames (Hit@1 {pct(a.hit1.iloc[0])}% vs {pct(b.hit1.iloc[0])}%).")
    fa = op[op.method == "maxinfo_fast"]
    if len(fa):
        sat = int((fa.pct_kept >= 0.98 * fa.pct_kept.max()).sum())
        o(f"MaxInfo (rect-maxvol) selects roughly K ~ R/tol^2 rows independent of clip length, so on ~30 s clips small tol saturates: {sat} of {len(fa)} MaxInfo-fast configs select the whole 1 FPS pool ({fa.pct_kept.max():.1f}% of 5 FPS frames). "
          f"Mean selection cost over all MaxInfo configs {op[op.method.str.startswith('maxinfo')].select_ms_per_video.mean():.1f} ms/video vs {op[op.method=='cos_last_drop'].select_ms_per_video.mean():.1f} ms/video for the team's cosine pruning.")

lg = rd("e1b_long/long_results.csv")
if lg is not None and len(lg):
    L += ["## E1b - long streams (concatenated test videos)", "",
          "Limitations: hard cuts between clips; generic actions repeat across clips (many plausible distractors); one index per stream, every member query searched against the whole stream.", ""]
    d = lg.copy(); d["method_label"] = [nice(m, p if m != "uniform" else p) for m, p in zip(d.method, d.params.fillna(""))]
    t = pd.DataFrame({"L (min)": d.L_min, "streams": d.n_streams, "method": d.method_label, "% kept": d.pct_kept.round(1),
                      "Hit@1 [95% CI]": [ci(r.hit1, r.hit1_lo, r.hit1_hi) for r in d.itertuples()],
                      "R@1 IoU0.5 [95% CI]": [ci(r.r1_iou05, r.r1_iou05_lo, r.r1_iou05_hi) for r in d.itertuples()],
                      "search ms/query": d.search_ms_per_query.round(3), "index MB/stream": d.index_MB_per_stream.round(2), "selection s/stream": d.select_s_per_stream.round(3)})
    L += [md(t)]
    fl = lg[lg.config == "full_5fps"].sort_values("L_min"); 
    if len(fl) >= 2:
        a, b = fl.iloc[0], fl.iloc[-1]
        o(f"Full 5 FPS Hit@1 falls from {pct(a.hit1)}% at {int(a.L_min)} min to {pct(b.hit1)}% at {int(b.L_min)} min ({b.n_queries:.0f} queries, {int(b.n_streams)} stream(s)); search latency grows {a.search_ms_per_query:.2f} -> {b.search_ms_per_query:.2f} ms/query and the index {a.index_MB_per_stream:.1f} -> {b.index_MB_per_stream:.1f} MB.")
        sy = lg[lg.config == "cos_last_drop[tau=0.95]"].sort_values("L_min")
        if len(sy):
            s = sy.iloc[-1]
            o(f"At {int(b.L_min)} min the system default (tau=0.95) keeps {s.pct_kept:.1f}% of frames: Hit@1 {pct(s.hit1)}% vs {pct(b.hit1)}% for full 5 FPS "
              + ("(overlapping CIs)." if overlap(s.hit1_lo, s.hit1_hi, b.hit1_lo, b.hit1_hi) else "(non-overlapping CIs).") + f" Latency {s.search_ms_per_query:.2f} ms/query.")

ab = rd("e3_absent/absent_results.csv")
if ab is not None:
    L += ["## E3 - absent-event queries", "",
          "AUROC is pooled across videos (positives = a video's own queries vs negatives scored against that video's index). 'in-domain' negatives are captions of other videos with no content-word overlap with any caption of the target video - lexical, not verified, absence. OOD queries are hand-written (`experiments/e3_absent/ood_queries.txt`) and also differ linguistically from Charades captions.", ""]
    t = pd.DataFrame({"index": ab.method, "negatives": ab.negatives, "n pos/neg": [f"{p}/{n}" for p, n in zip(ab.n_pos, ab.n_neg)],
                      "AUROC [95% CI]": [f"{r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}]" for r in ab.itertuples()],
                      "TPR @0.20": (100 * ab.tpr_at_sys).round(1), "FPR @0.20": (100 * ab.fpr_at_sys).round(1),
                      "thr @FPR10%": ab.thr_at_fpr10.round(3), "TPR @FPR10%": (100 * ab.tpr_at_fpr10).round(1)})
    L += [md(t)]
    f_ = ab[ab.method == "full_5fps"].set_index("negatives")
    if len(f_):
        o(f"E3 (full index): AUROC {f_.loc['indomain'].auroc:.3f} for present vs in-domain absent, {f_.loc['ood'].auroc:.3f} vs out-of-domain; at the system's 0.20 threshold TPR {pct(f_.loc['all'].tpr_at_sys)}% but FPR {pct(f_.loc['all'].fpr_at_sys)}% (all negatives; in-domain {pct(f_.loc['indomain'].fpr_at_sys)}%), so the 0.20 threshold does little to reject absent in-domain events.")
        c_ = ab[ab.method != "full_5fps"].set_index("negatives")
        if len(c_):
            o(f"Pruning (tau=0.95) changes pooled AUROC (all negatives) from {f_.loc['all'].auroc:.3f} to {c_.loc['all'].auroc:.3f}" + (" - within overlapping CIs." if overlap(f_.loc['all'].auroc_lo, f_.loc['all'].auroc_hi, c_.loc['all'].auroc_lo, c_.loc['all'].auroc_hi) else " - non-overlapping CIs."))

ss = rd("e6_latency/stage_summary.csv"); rt = rd("e6_latency/routing_summary.csv"); rs = rd("e6_latency/resources.csv")
if ss is not None:
    L += ["## E6 - where the time goes (team's real pipeline)", "",
          "Wall-clock on a shared 2-core CPU (a background encoding job may be running; see `results/e6_latency/meta.json` loadavg). First video and first query are warm-ups and excluded from means. COMPLEX-routed queries stop at the VLM stage (no OpenRouter key), so no end-to-end slow-path time is reported.", ""]
    s = ss[ss.phase.isin(["preprocess (per video)", "query", "model load (cold, once)"])]
    L += [md(s.round(3))]
    if rt is not None: L += ["Routing decisions:", "", md(rt.round(1))]
    if rs is not None: L += ["Memory / resources:", "", md(rs.round(1))]
    pre = ss[(ss.phase == "preprocess (per video)") & (ss.stage != "TOTAL (wall)")]
    if len(pre):
        tot = pre.mean_s.sum(); top = pre.sort_values("mean_s", ascending=False).iloc[0]
        o(f"E6: per-video preprocessing is dominated by {top.stage.replace('_', ' ')} ({top.mean_s:.1f} s, {100 * top.mean_s / tot:.0f}% of staged time); pruning + FAISS build together take {pre[pre.stage.isin(['frame_pruning', 'faiss_index'])].mean_s.sum() * 1000:.0f} ms.")
    qq = ss[(ss.phase == "query") & (ss.stage != "TOTAL (wall)")]
    if len(qq):
        top = qq.sort_values("mean_s", ascending=False).iloc[0]
        o(f"E6: per query, the slowest completed stage is {top.stage.replace('_', ' ')} ({top.mean_s:.2f} s); FAISS search itself is {qq[qq.stage == 'faiss_search'].mean_s.sum() * 1000:.1f} ms.")
    if rt is not None and len(rt):
        c = rt[rt.route == "COMPLEX"]
        o(f"E6: {c.pct.sum():.0f}% of the {int(rt.n.sum())} sampled Charades queries were routed COMPLEX (VLM path) by the zero-shot classifier, {100 - c.pct.sum():.0f}% SIMPLE.")

L += ["## Observations (derived only from the numbers above)", ""] + [f"{i}. {x}" for i, x in enumerate(obs[:8], 1)] + [""]
(R / "SUMMARY.md").write_text("\n".join(L))
print("wrote", R / "SUMMARY.md")
