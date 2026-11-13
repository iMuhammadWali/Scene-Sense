# E7 - moment localization upgrades (Charades-STA test, single-video search)

Problem: the E1 protocol predicts the coverage span of the top-1 frame group (1.3 s on the full index, 2.5 s on the pruned one) while
ground-truth moments average 8.0 s, so R@1 IoU>=0.5 is 2-8% although Hit@1 is ~48%. E7 keeps the same CLIP frame scores and changes only how
a span is read off the score curve s(t) = cos(query, frame).

## Protocol
- 1,334 videos split 50/50 with a fixed seed (`split.json`, seed 0): 667 DEV, 667 HELD-OUT. All hyper-parameters are chosen on DEV by R1@0.5
  (ties: Hit@1); the chosen config per (method, index) is then reported on HELD-OUT (and ALL). `dev_grid.csv` holds every config on DEV.
- Indexes: full 5 FPS, and `cos_last_drop` tau=0.95 (the system default). On the pruned index the curve is piecewise-constant over each kept
  frame's coverage span, so every method runs on both. Curves are sampled on a 0.2 s grid. Text embeddings: the fixed single-projection cache.
- Methods: M0 = existing E1 protocol (team FAISS + grouping; reproduces SUMMARY.md exactly on ALL: full 48.6/2.4/9.7, cos 48.2/7.8/15.0 Hit@1/R1@0.5/mIoU);
  M1 = moving-average smoothing (w in 1,2,3 s), argmax, span = coverage span of the kept item at the peak; M2 = smoothed peak +/- L/2 (L in 6,8,10 s,
  window shifted to stay inside the video); M3 = grow from the smoothed peak while s_smooth >= peak - alpha*(peak - median(s)) (alpha in .3,.5,.7);
  M4 = TFVTG-style contrast proposals (0.5 s grid, length 2..min(25 s, duration), score = mean_in - mean_out - lambda*len/duration, lambda in 0,.1,.2,
  top-1 + NMS(0.5) top-5 for R@5); M5 = per-video (per query curve) z-score before M3/M4.
- Extras (clearly outside the prescribed grids): M3x (w in 3,5 s; alpha in .8,.9,1.0; chosen on DEV over the union with M3, because the M3 optimum sat on the
  grid edge), M4x (lambda in -.1,-.2,-.4; never beat lambda=0 on DEV, only in `dev_grid.csv`), P0 = query-agnostic centre window (L in 6,8,10 s), a
  no-model prior that shows how much of M2 is "just the 8 s length prior".
- Hit@1 for M1-M3 = kept-item timestamp at the smoothed peak lies inside GT; for M4/M5 = peak of the raw curve inside the predicted span.
  Hit@5 is only defined for M0. CIs: 1,000 query-level bootstrap resamples (fixed seed, paired across methods).

## HELD-OUT results (667 videos, 1835 queries; %)
| method | index | chosen config | Hit@1 [95% CI] | R1@0.3 | R1@0.5 [95% CI] | R1@0.7 | mIoU | R5@0.5 | mean pred. len (s) |
|---|---|---|---|---|---|---|---|---|---|
| M0 | full_5fps | `M0` | 50.1 [47.7, 52.3] | 10.4 | 2.3 [1.6, 3.1] | 0.5 | 10.1 |  | 1.4 |
| M1 | full_5fps | `M1[w=1]` | 48.3 [46.0, 50.5] | 0.0 | 0.0 [0.0, 0.0] | 0.0 | 1.3 |  | 0.2 |
| M2 | full_5fps | `M2[w=1,L=8]` | 48.3 [46.0, 50.5] | 55.1 | 43.2 [40.8, 45.4] | 19.2 | 37.0 |  | 8.0 |
| M3 | full_5fps | `M3[w=3,a=0.7]` | 45.8 [43.3, 48.1] | 44.2 | 29.1 [27.0, 31.2] | 14.2 | 29.4 |  | 5.7 |
| M4 | full_5fps | `M4[lam=0.0]` | 49.6 [47.2, 51.8] | 30.8 | 11.2 [9.9, 12.7] | 3.7 | 21.3 | 42.0 | 6.1 |
| M5 | full_5fps | `M5/M3[w=3,a=0.7]` | 45.8 [43.3, 48.1] | 44.2 | 29.1 [27.0, 31.2] | 14.2 | 29.4 |  | 5.7 |
| M3x | full_5fps | `M3x[w=5,a=0.9]` | 42.2 [40.1, 44.4] | 49.3 | 36.7 [34.4, 39.0] | 19.8 | 34.0 |  | 8.2 |
| P0 | full_5fps | `P0[L=10]` | n/a | 27.7 | 15.4 [13.8, 17.0] | 4.1 | 19.6 |  | 10.0 |
| M0 | cos095 | `M0` | 48.7 [46.3, 51.0] | 20.8 | 8.2 [7.0, 9.4] | 2.5 | 15.2 |  | 2.5 |
| M1 | cos095 | `M1[w=3]` | 45.0 [42.6, 47.1] | 5.7 | 1.2 [0.8, 1.7] | 0.2 | 6.9 |  | 1.6 |
| M2 | cos095 | `M2[w=2,L=8]` | 45.9 [43.7, 48.1] | 53.6 | 40.3 [38.0, 42.4] | 18.1 | 36.0 |  | 8.0 |
| M3 | cos095 | `M3[w=3,a=0.7]` | 45.0 [42.6, 47.1] | 42.5 | 27.6 [25.6, 29.6] | 11.8 | 27.8 |  | 5.7 |
| M4 | cos095 | `M4[lam=0.0]` | 48.1 [45.7, 50.4] | 30.1 | 10.0 [8.7, 11.4] | 2.7 | 20.4 | 42.1 | 5.9 |
| M5 | cos095 | `M5/M3[w=3,a=0.7]` | 45.0 [42.6, 47.1] | 42.5 | 27.6 [25.6, 29.6] | 11.8 | 27.8 |  | 5.7 |
| M3x | cos095 | `M3x[w=5,a=0.9]` | 43.0 [40.6, 45.3] | 48.4 | 35.0 [32.8, 37.1] | 18.2 | 33.3 |  | 8.1 |
| P0 | cos095 | `P0[L=10]` | n/a | 27.7 | 15.4 [13.8, 17.0] | 4.1 | 19.6 |  | 10.0 |

Figure: `../figures/e7_methods.png`. Same table on ALL queries: `heldout.csv` (split = ALL). Chosen configs: `best_configs.json`.

## Findings
- Reading a span of realistic length off the same scores lifts R1@0.5 from 2.3% / 8.2% (M0, full / pruned) to ~43% / ~40% (M2, L = 8 s) on HELD-OUT; the
  pruned index loses only 1-3 points of R1@0.5 and 2-4 of Hit@1 relative to the full index.
- Hit@1 is unchanged or slightly lower (smoothing moves the peak): localization, not retrieval, was the bottleneck.
- The query-agnostic centre prior P0 gets 15.4% and a random 8 s moment ~20%, so M2's gain is real but the largest part of it comes from knowing the mean
  length (8 s) - a prior taken from the annotations' length distribution; this should be stated when comparing with the literature (zero-shot 32-38, supervised 48.2).
- M3 (adaptive growth) is below fixed length unless it is allowed to grow to ~8 s (M3x ~36%/35%); M4 with raw CLIP curves prefers short, peaky spans (mean 6 s at
  lambda=0, 2 s at lambda>0) and is poor at R@1 (10-11%) but decent at R@5 (~42% on DEV). M5: z-scoring is an affine map, so M3 is exactly invariant (verified, `sanity.txt`);
  for M4 it only rescales lambda and did not help.

## Caveats
- DEV and HELD-OUT are both Charades-STA test videos (the 3,720 test queries are not used for training anything here), but M2/P0 lengths come from the data's length statistics.
- Query-level bootstrap ignores the grouping of ~2.8 queries per video, so CIs are slightly optimistic. Best M3 on the prescribed grid is on the grid edge (see M3x).
- Single CLIP ViT-B/32 text prompt (no ensembling); long streams (E1b) and prompt ensembling not done.
- Not a comparison with TimeChat etc.: different model, zero training, length prior tuned on DEV.

Reproduce: `python -W ignore experiments/e7_localization/run.py [--limit N] [--force]`, then `python experiments/e7_localization/figure.py`. Full run: ~1 min on 2 CPU cores.
