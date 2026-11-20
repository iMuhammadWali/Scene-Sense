"""E7 figure from results/e7_localization/heldout.csv (HELD-OUT rows). Usage: python experiments/e7_localization/figure.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sseval import config as C
out = C.results_dir("e7_localization"); h = pd.read_csv(out / "heldout.csv", encoding="utf-8"); h = h[h.split == "HELD-OUT"]
order = ["P0", "M0", "M1", "M2", "M3", "M3x", "M4", "M5"]
# the M4x extension row duplicated M4 (lambda<0 never won on DEV) -> not drawn
cols = {"full_5fps": "#2a6fbb", "cos095": "#e08a1e"}; names = {"full_5fps": "Full 5 FPS", "cos095": "cos_last_drop tau=0.95"}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
for ax, (m, lo, hi, title) in zip(axs, [("r1_iou05", "r1_iou05_lo", "r1_iou05_hi", "R@1 IoU>=0.5"), ("hit1", "hit1_lo", "hit1_hi", "Hit@1")]):
    for k, (i, c) in enumerate(cols.items()):
        d = h[h["index"] == i].set_index("method").reindex(order)
        x = np.arange(len(order)) + (k - 0.5) * 0.38
        v = d[m].values * 100
        ax.bar(x, v, 0.36, color=c, label=names[i], yerr=[v - d[lo].values * 100, d[hi].values * 100 - v], capsize=2, ecolor="#333", error_kw=dict(lw=0.9))
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order); ax.set_title(f"HELD-OUT {title} (%, 95% CI)"); ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
axs[0].axhline(20.1, ls="--", color="gray", lw=1); axs[0].text(7.4, 21, "random moment 20.1", ha="right", fontsize=8, color="gray")
axs[1].text(0, 2, "no query\n(n/a)", ha="center", fontsize=8, color="gray")
axs[0].legend(loc="upper left", fontsize=8)
fig.tight_layout(); (C.RESULTS_DIR / "figures").mkdir(exist_ok=True); fig.savefig(C.RESULTS_DIR / "figures" / "e7_methods.png", dpi=150)
