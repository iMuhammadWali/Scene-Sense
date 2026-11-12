"""Figures + summary.md from results/*.csv.  Usage: python -W ignore make_figures.py"""
import os, platform
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = "/home/claude/experiments/results"
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10.5, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 200})
COL = {"threshold": "#1f5fa8", "uniform": "#e08a1e", "diversity": "#2f9e63", "fixed": "#7a7a7a", "full": "#222222", "random": "#b33a3a"}
DL = "Diversity (k-center, MaxInfo-style)"
LAB = {"threshold": "Threshold pruning (team)", "uniform": "Uniform, budget-matched", "diversity": DL}
S = pd.read_csv(f"{RES}/exp1_short.csv")
full = S[S.method == "full_5fps"].iloc[0]

# ---- Fig 1
fig, axs = plt.subplots(1, 2, figsize=(10, 4.1), sharex=True)
for ax, (col, yl) in zip(axs, [("R1_iou0_5", "R@1, IoU >= 0.5"), ("hit1", "Hit@1")]):
    for fam in ["threshold", "uniform", "diversity"]:
        d = S[S.family == fam].sort_values("pct_frames_kept")
        ax.plot(d.pct_frames_kept, d[col], "o-", color=COL[fam], label=LAB[fam], ms=4.5, lw=1.6)
    for _, r in S[S.family == "fixed"].iterrows():
        ax.scatter(r.pct_frames_kept, r[col], marker="s", color=COL["fixed"], zorder=3, s=30,
                   label="Fixed rate (1 / 0.5 FPS)" if r.method == "fixed_1fps" else None)
    ax.axhline(full[col], color=COL["full"], ls="--", lw=1, label="Full 5 FPS (100% frames)")
    rr = S[S.method == "random_moment"].iloc[0][col]
    ax.axhline(rr, color=COL["random"], ls=":", lw=1, label="Random baseline")
    ax.set_xscale("log"); ax.set_xlabel("Frames kept (% of 5 FPS, log scale)"); ax.set_ylabel(yl)
    ax.set_xticks([5, 10, 20, 50, 100]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    t95 = S[S.method == "threshold_0.95"].iloc[0]
    ax.annotate("T=0.95\n(default)", (t95.pct_frames_kept, t95[col]), textcoords="offset points", xytext=(8, -22), fontsize=8.5, color=COL["threshold"])
axs[0].set_title(f"Moment retrieval (n={int(S.n_queries.iloc[0])} queries, {int(S.n_videos.iloc[0])} videos)")
axs[1].set_title("Frame hit")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(f"{RES}/fig_acc_vs_frames.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 3
V = pd.read_csv(f"{RES}/exp1_per_video.csv"); v = V[V.method == "threshold_0.95"]
pct = 100 * v.n_sel / v.n_frames
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.hist(pct, bins=25, color=COL["threshold"], edgecolor="white")
ax.axvline(pct.mean(), color="k", ls="--", lw=1, label=f"mean {pct.mean():.1f}%")
ax.axvline(pct.median(), color="k", ls=":", lw=1, label=f"median {pct.median():.1f}%")
ax.set_xlabel("Frames kept per video at T = 0.95 (%)"); ax.set_ylabel("Videos"); ax.set_title(f"Pruning ratio distribution (n={len(v)} videos)")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{RES}/fig_frames_kept_hist.png"); plt.close(fig)

# ---- Fig 2 + long table
Lp = f"{RES}/exp1b_long.csv"; have_long = os.path.exists(Lp)
if have_long:
    Lg = pd.read_csv(Lp)
    show = ["full_5fps", "fixed_1fps", "threshold_0.95", "uniform_matched_0.95", "diversity_matched_0.95"]
    sty = {"full_5fps": ("full", "-"), "fixed_1fps": ("fixed", "-"), "threshold_0.95": ("threshold", "-"),
           "uniform_matched_0.95": ("uniform", "--"), "diversity_matched_0.95": ("diversity", "--")}
    nm = {"full_5fps": "Full 5 FPS", "fixed_1fps": "Fixed 1 FPS", "threshold_0.95": "Threshold 0.95 (team)",
          "uniform_matched_0.95": "Uniform, matched to T=0.95", "diversity_matched_0.95": DL + ", matched to T=0.95"}
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for m in show:
        d = Lg[Lg.method == m].sort_values("L_min"); c, ls = sty[m]
        for ax, col in zip(axs, ["R1_iou0_5", "search_ms_per_query", "index_MB_per_stream"]):
            ax.plot(d.L_min, d[col], "o" + ls, color=COL[c], label=nm[m], ms=4.5, lw=1.6)
    rr = Lg[Lg.method == "random_moment"].sort_values("L_min")
    axs[0].plot(rr.L_min, rr.R1_iou0_5, ":", color=COL["random"], label="Random baseline")
    for ax, yl, t in zip(axs, ["R@1, IoU >= 0.5", "Search latency per query (ms)", "Index size per stream (MB)"],
                         ["Accuracy vs stream length", "Latency vs stream length", "Index size vs stream length"]):
        ax.set_xlabel("Stream length (min)"); ax.set_ylabel(yl); ax.set_title(t)
        ax.set_xticks(sorted(Lg.L_min.unique()))
    axs[1].set_yscale("log"); axs[2].set_yscale("log")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(f"{RES}/fig_long_scaling.png", bbox_inches="tight"); plt.close(fig)

# ---- summary.md
def md(df):
    cols = list(df.columns); s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows(): s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s
def pc(x): return f"{100*x:.1f}"
T1 = S.copy(); T1["Method"] = T1.apply(lambda r: DL + f" T={r['T']:.2f}" if r.family == "diversity" else r.method, axis=1)
T1 = pd.DataFrame({"Method": T1.Method, "Frames kept %": T1.pct_frames_kept.round(1), "Frames/video": T1.frames_per_video.round(1),
                   "Hit@1 %": T1.hit1.map(pc), "Hit@5 %": T1.hit5.map(pc), "R@1 IoU0.3 %": T1.R1_iou0_3.map(pc),
                   "R@1 IoU0.5 %": T1.R1_iou0_5.map(pc), "R@1 IoU0.7 %": T1.R1_iou0_7.map(pc), "mIoU %": T1.mIoU.map(pc)}).fillna("-")
T1b = pd.DataFrame({"Method": S.method, "Index MB (total)": S.index_MB_total.round(1), "Select ms/video": S.sel_ms_per_video.round(2),
                    "Search ms/query": S.search_ms_per_query.round(3), "SE Hit@1 (pp)": (100 * S.se_hit1).round(1),
                    "SE R@1 IoU0.5 (pp)": (100 * S.se_R1_iou0_5).round(1)}).fillna("-")
get = lambda m: S[S.method == m].iloc[0]
t95, u95, d95, f1, rnd = get("threshold_0.95"), get("uniform_matched_0.95"), get("diversity_matched_0.95"), get("fixed_1fps"), get("random_moment")
nv, nq = int(S.n_videos.iloc[0]), int(S.n_queries.iloc[0])
B = [f"Full 5 FPS reaches Hit@1 {pc(full.hit1)}% and R@1 IoU>=0.5 {pc(full.R1_iou0_5)}%, versus {pc(rnd.hit1)}% and {pc(rnd.R1_iou0_5)}% for the random baseline.",
     f"Threshold 0.95 (team default) keeps {t95.pct_frames_kept:.1f}% of frames (mean {t95.frames_per_video:.1f} per video) with R@1 IoU>=0.5 {pc(t95.R1_iou0_5)}% (full: {pc(full.R1_iou0_5)}%) and Hit@1 {pc(t95.hit1)}% (full: {pc(full.hit1)}%).",
     f"At the same frame budget as T=0.95, uniform sampling gives R@1 IoU>=0.5 {pc(u95.R1_iou0_5)}% / Hit@1 {pc(u95.hit1)}%, and the diversity k-center baseline gives {pc(d95.R1_iou0_5)}% / {pc(d95.hit1)}%. Standard errors are about {100*t95.se_R1_iou0_5:.1f} pp (binomial, queries treated as independent; clustering within videos makes this optimistic), so differences of this size or smaller should not be read as real.",
     f"Fixed 1 FPS keeps {f1.pct_frames_kept:.0f}% of frames with R@1 IoU>=0.5 {pc(f1.R1_iou0_5)}% and Hit@1 {pc(f1.hit1)}%; compare against threshold methods at similar % kept.",
     "Charades-STA clips are short (~30 s) and fairly static, so frame budgets differ little in absolute terms; efficiency effects (index size, search latency) are more visible in the stitched-stream experiment."]
if have_long:
    Lg["Method"] = Lg.method
    mx = Lg.L_min.max(); g = lambda m: Lg[(Lg.method == m) & (Lg.L_min == mx)]
    if len(g("full_5fps")) and len(g("threshold_0.95")):
        a, b = g("full_5fps").iloc[0], g("threshold_0.95").iloc[0]
        B.append(f"At L={mx} min ({int(b.n_streams)} streams, {int(b.n_queries)} queries) full 5 FPS gives R@1 IoU>=0.5 {pc(a.R1_iou0_5)}% with {a.search_ms_per_query:.3f} ms/query and {a.index_MB_per_stream:.1f} MB/stream; threshold 0.95 gives {pc(b.R1_iou0_5)}% with {b.search_ms_per_query:.3f} ms/query and {b.index_MB_per_stream:.1f} MB/stream. Few streams means wide uncertainty.")
    TL = Lg[Lg.method.isin(show + ["threshold_0.85", "threshold_0.90", "random_moment"])]
    TL = pd.DataFrame({"L (min)": TL.L_min, "Method": TL.method, "Streams": TL.n_streams, "Queries": TL.n_queries,
                       "Frames kept %": TL.pct_frames_kept.round(1), "Hit@1 %": TL.hit1.map(pc), "R@1 IoU0.5 %": TL.R1_iou0_5.map(pc),
                       "mIoU %": TL.mIoU.map(pc), "Search ms/q": TL.search_ms_per_query.round(3),
                       "Index MB/stream": TL.index_MB_per_stream.round(1)}).fillna("-")
hw = f"CPU only, {os.cpu_count()} cores, {platform.platform()}; FAISS IndexFlatIP, 1 thread; CLIP ViT-B/32 embeddings cached at 5 FPS. Latencies measured while a background encoding job shared the CPU, so treat them as indicative."
with open(f"{RES}/summary.md", "w") as f:
    f.write(f"# Scene Sense evaluation summary (Charades-STA test, partial data)\n\n")
    f.write(f"**Experiment 1 used {nv} videos / {nq} queries** (of 1334 videos / 3720 queries in the split; whatever embeddings existed at run time).\n\n")
    f.write(f"Hardware: {hw}\n\n")
    f.write("Method labels: `threshold_T` = team's prune_frames; `uniform_matched_T` / `diversity_matched_T` use the same per-video frame count as `threshold_T`. "
            f"`diversity_matched` is **{DL}**, a greedy farthest-point baseline; it is NOT the MaxInfo algorithm. `random_moment` is a chance-level reference (random interval of the mean GT length; random timestamps for Hit@k).\n\n")
    f.write("## Experiment 1: single-video search\n\n" + md(T1) + "\n### Efficiency and uncertainty\n\n" + md(T1b) + "\n")
    if have_long:
        f.write("## Experiment 1b: stitched streams\n\n" + md(TL) + "\n")
    f.write("## Observations\n\n" + "\n".join(f"- {x}" for x in B) + "\n\n")
    f.write("## Limitations\n\n- Stitched streams have hard cuts between unrelated clips (which helps pruning, since a cut always triggers a keep) and captions are often duplicated across clips (e.g. similar actions in different videos), so a retrieved moment in another clip counts as a miss even if semantically valid.\n"
            "- Results are on a partial subset of the test split; selection of the subset is by file availability, not random.\n"
            "- Moment prediction is derived from frame coverage spans, not a trained localiser; GT end times beyond the video duration are left unclipped.\n"
            "- Charades-STA videos are short, so frame-budget effects are small compared with genuinely long video.\n")
print("figures + summary written")
