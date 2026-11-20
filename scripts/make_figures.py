"""Figures from results/*. Usage: python -W ignore scripts/make_figures.py   (honours SSEVAL_RESULTS_DIR)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sseval import config as C

R = C.RESULTS_DIR; FIG = R / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10.5, "legend.fontsize": 9, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 200})
# one colour per method family; marker/linestyle distinguish members of a family
STYLE = {
    "uniform":          ("#e08a1e", "o", "-",  "Uniform"),
    "cos_last_drop":    ("#1f5fa8", "s", "-",  "Cosine vs last kept (current system)"),
    "cos_consec_drop":  ("#7fa8d8", "s", "--", "Cosine vs previous frame"),
    "streammem_merge":  ("#2f9e63", "^", "-",  "StreamMem merge †"),
    "maxinfo_fast":     ("#8e44ad", "v", "-",  "MaxInfo-fast †"),
    "maxinfo_slow":     ("#b57edc", "D", "--", "MaxInfo-slow †"),
    "maxinfo_chunk":    ("#5b2c83", "P", ":",  "MaxInfo-chunk †"),
    "hybrid":           ("#c0392b", "*", "-",  "Hybrid (StreamMem + MaxInfo-chunk) †"),
}
DAGGER = "† our re-implementation for retrieval indexing, adapted from methods designed for VLM frame selection"

def pareto(d, ycol):
    """Frontier: sorted by % kept, keep points that improve the metric."""
    d = d.sort_values(["pct_kept", ycol], ascending=[True, False]); best = -1; keep = []
    for r in d.itertuples():
        v = getattr(r, ycol)
        if v > best: keep.append(r.Index); best = v
    return d.loc[keep]

def fig1():
    f = R / "e1_pruning" / "operating_points.csv"
    if not f.exists(): print("[fig] skip e1 (no results)"); return
    op = pd.read_csv(f, encoding="utf-8"); full = op[op.config == "full_5fps"].iloc[0]; rnd = op[op.method == "random_moment"].iloc[0]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    for ax, (col, yl) in zip(axs, [("r1_iou05", "R@1, IoU ≥ 0.5"), ("hit1", "Hit@1")]):
        for m, (c, mk, ls, lab) in STYLE.items():
            d = op[op.method == m]
            if len(d) == 0: continue
            p = pareto(d, col)
            ax.plot(p.pct_kept, p[col], marker=mk, ls=ls, color=c, ms=5, lw=1.4, label=lab)
        ax.axhline(full[col], color="#222222", ls="--", lw=1, label="Full 5 FPS (100% of frames)")
        ax.axhline(rnd[col], color="#999999", ls=":", lw=1.2, label="Chance (random moment)")
        sysd = op[op.config == "cos_last_drop[tau=0.95]"]
        if len(sysd):
            ax.scatter(sysd.pct_kept, sysd[col], s=120, facecolors="none", edgecolors="#1f5fa8", linewidths=1.8, zorder=5)
        ax.set_xscale("log"); ax.set_xlabel("Frames kept (% of 5 FPS, log scale)"); ax.set_ylabel(yl)
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.xaxis.set_major_formatter(ScalarFormatter())
    n = int(op.n_queries.iloc[0]); nv = int(op.n_videos.iloc[0])
    axs[0].set_title(f"Moment localisation ({nv} videos, {n} queries)"); axs[1].set_title("Top-1 frame inside ground-truth moment")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, -0.005, DAGGER + ".  Ringed point = system default (τ = 0.95).  Lines show each method's Pareto frontier.", ha="center", va="top", fontsize=8)
    fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.savefig(FIG / "e1_quality_vs_kept.png", bbox_inches="tight"); plt.close(fig)

def fig2():
    f = R / "e1b_long" / "long_results.csv"
    if not f.exists(): print("[fig] skip e1b"); return
    d = pd.read_csv(f, encoding="utf-8"); d = d[d.method != "random_moment"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
    for cfg, g in d.groupby("config", sort=False):
        m = g.method.iloc[0]; c, mk, ls, lab = STYLE.get(m, ("#222222", "o", "-", "Full 5 FPS"))
        if m == "full_5fps": c, mk, ls = "#222222", "o", "-"
        name = "Full 5 FPS" if m == "full_5fps" else cfg.replace("uniform[match=cos_last_drop[tau=0.95]]", "Uniform, matched to τ=0.95") \
            .replace("uniform[match=cos_last_drop[tau=0.9]]", "Uniform, matched to τ=0.9").replace("uniform[fraction=0.1]", "Uniform 10%")
        if m == "cos_last_drop": name = f"Cosine τ={cfg.split('=')[1].rstrip(']')}" + (" (system)" if "0.95" in cfg else "")
        elif m not in ("full_5fps", "uniform"): name = lab.replace(" †", "") + " †"
        if m == "uniform" and "match" in cfg: ls = "--" if "0.9]" in cfg else "-"
        if m == "uniform" and "fraction" in cfg: ls = ":"
        if m == "cos_last_drop" and "0.9]" in cfg: ls = "--"
        for ax, col in zip(axs, ["r1_iou05", "search_ms_per_query", "index_MB_per_stream"]):
            ax.plot(g.L_min, g[col], marker=mk, ls=ls, color=c, ms=4.5, lw=1.3, label=name)
    for ax, yl in zip(axs, ["R@1, IoU ≥ 0.5", "Search latency (ms / query)", "Index size (MB / stream)"]):
        ax.set_xlabel("Stream length (min)"); ax.set_ylabel(yl); ax.set_xticks(sorted(d.L_min.unique()))
    axs[1].set_yscale("log"); axs[2].set_yscale("log")
    axs[0].set_title("Moment localisation"); axs[1].set_title("Search latency (flat FAISS, 1 thread)"); axs[2].set_title("Index size")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, -0.005, DAGGER + ".", ha="center", va="top", fontsize=8)
    fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(FIG / "e1b_scaling.png", bbox_inches="tight"); plt.close(fig)

def fig3():
    f = R / "e3_absent" / "scores.parquet"
    if not f.exists(): print("[fig] skip e3"); return
    sc = pd.read_parquet(f); fig, axs = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
    cols = {"positive": ("#1f5fa8", "Present (own query)"), "indomain": ("#e08a1e", "Absent, in-domain"), "ood": ("#b33a3a", "Absent, out-of-domain")}
    bins = np.arange(0.05, 0.40, 0.005)
    for ax, (m, title) in zip(axs, [("full_5fps", "Full 5 FPS index"), ("cos_last_drop_tau0.95", "Pruned index (cosine τ = 0.95, current system)")]):
        for k, (c, lab) in cols.items():
            ax.hist(sc[(sc.method == m) & (sc.kind == k)].score, bins=bins, histtype="stepfilled", alpha=0.35, color=c, density=True)
            ax.hist(sc[(sc.method == m) & (sc.kind == k)].score, bins=bins, histtype="step", color=c, lw=1.4, density=True, label=lab)
        ax.axvline(C.RETRIEVAL_THRESHOLD, color="k", ls="--", lw=1.2, label="System threshold 0.20")
        ax.set_title(title); ax.set_xlabel("Max query–frame cosine similarity in index")
    axs[0].set_ylabel("Density"); axs[0].legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "e3_score_hist.png", bbox_inches="tight"); plt.close(fig)

def fig4():
    f = R / "e6_latency" / "stage_summary.csv"
    if not f.exists(): print("[fig] skip e6"); return
    s = pd.read_csv(f, encoding="utf-8"); s = s[s.phase.isin(["preprocess (per video)", "query"]) & (s.stage != "TOTAL (wall)")]
    s = s.assign(label=s.stage.str.replace("_", " "))
    order = list(s[s.phase.str.startswith("pre")].label) + list(s[s.phase == "query"].label)
    s = s.set_index(["phase", "label"]).sort_index()
    pre = s.loc["preprocess (per video)"] if "preprocess (per video)" in s.index.get_level_values(0) else pd.DataFrame(columns=s.columns)
    q = s.loc["query"] if "query" in s.index.get_level_values(0) else pd.DataFrame(columns=s.columns)
    rows = [("Upload/index (once per video)", n, v) for n, v in pre.mean_s.items()] + [("Per query", n, v) for n, v in q.mean_s.items()]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    colors = ["#1f5fa8" if r[0].startswith("Upload") else "#e08a1e" for r in rows]
    ax.barh(y, [r[2] for r in rows], color=colors, height=0.65)
    for yi, r in zip(y, rows): ax.text(r[2] + max(x[2] for x in rows) * 0.01, yi, f"{r[2]:.3f} s", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([r[1] for r in rows]); ax.set_xlabel("Mean seconds per stage (warm-up excluded; 2-core CPU)")
    ax.set_xlim(0, max(r[2] for x in [rows] for r in x) * 1.15); ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f5fa8", label="Video processing (once per video)"), Patch(color="#e08a1e", label="Query pipeline (per query)")], frameon=False, loc="center right")
    fig.tight_layout(); fig.savefig(FIG / "e6_stage_breakdown.png", bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4): fn()
    print("figures ->", FIG)
