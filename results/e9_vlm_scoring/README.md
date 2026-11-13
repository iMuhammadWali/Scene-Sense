# E9 - small local VLM (SmolVLM2-500M) as a per-clip scorer

Runner: `experiments/e9_vlm_scoring/run.py` (`--limit N`, `--skip-posbias`, `--aggregate-only`, `--out DIR`; resumes from `raw/` and `posbias/raw/`).
Outputs: `summary.csv` (long format: section, name, metric, n, value, lo, hi), `per_query.csv` (5 CLIP scores, 5 p_yes, per-method hits, DeepSeek choice),
`posbias_per_query.csv`, `raw/` (one JSON per query x candidate forward pass), `posbias/raw/`, figure `../figures/e9_vlm_scoring.png`. Unit test: `tests/test_e9_vlm_scoring.py`.
E8 files/results untouched; E9 imports E8's `run.py` (candidates, frame decoding, parser, prompt) and `local_vlm.py` (model loading, `run_local`).

## Design
- Queries: extended to the SAME sample DeepSeek saw in E8: all 300 positives and all 100 negatives (seed 8). The first pass (first 100 pos + 50 neg, the E8 local-run subset) is kept as a separate
  reference set (`set=first100` in summary.csv; `in_first100` in per_query.csv). C1 = full 5 FPS index candidates (E8 C1K1 on positives, C1K3 on negatives); C2 = pruned index (cos_last_drop tau 0.95; E8 C2K3), 300 positives only.
  Candidates are E8's (5 per query; 13/300 C1 and 40/300 C2 positives have fewer than 5, NaN-padded); candidate times were verified identical to E8's `per_query.csv` for every query. Cached 150 C1 results were reused; C2 reuses C1 scores where the (query, time) pair coincides.
- Frame: K=1, the 1 FPS grid frame nearest the candidate time (E8's `clip_times`), 448 px JPEG q80, processor longest edge 384, no splitting. bf16, 2 threads.
- Yes/no scoring: 5 separate forward passes per query (one image each), prompt `Does this image show: '<query>'? Answer Yes or No.` via the chat template, generation prompt appended, no generation.
  Next-token logits; Yes ids = {"Yes", " Yes", "yes", " yes", "YES", " YES"} (single-token variants only), No likewise; each side = max logit over variants; p_yes = 2-way softmax.
  The model's top token was always " Yes" or " No" (leading-space variants), so the max-over-variants matters. ~94 input tokens per pass.
- Fusion: per query, z-normalise the 5 CLIP scores and the 5 logit(p_yes) (= yes-no logit difference); score = z_clip + w z_vlm, w in {0.25, 0.5, 1, 2}.
  w chosen by 2-fold CV over the positives (seed 0 split; fit = best Hit@1 on one half, ties -> smaller w; evaluated on the other half; pooled). All w are also reported (in-sample).
- Position-bias check: E8's A-E multiple-choice prompt (K=1, lenient bare-letter parser, same as the E8 local run), candidate order shuffled with a per-query seeded permutation.

## Results, full sample (300 positives; Hit@1 %, bootstrap 95% CI; differences are paired)
**C1 (full 5 FPS index)**
| system | Hit@1 | minus fast | minus DeepSeek |
|---|---|---|---|
| fast path top-1 (CLIP) | 49.7 [43.7, 55.3] | - | -13.3 |
| VLM only (argmax p_yes) | 52.7 [47.0, 58.3] | +3.0 [-3.0, 9.0] | -10.3 [-16.0, -4.7] |
| fusion w=0.25 / 0.5 / 1 / 2 (all w, in-sample) | 51.3 / 50.7 / 54.0 / 53.3 | +1.7 [-1.0, 4.0] / +1.0 [-2.7, 4.3] / +4.3 [0.0, 8.7] / +3.7 [-2.0, 9.0] | -11.7 / -12.3 / -9.0 / -9.7 (all CIs exclude 0) |
| fusion, w by 2-fold CV | 49.3 [43.3, 54.7] | -0.3 [-4.7, 4.0] | -13.7 [-19.0, -7.7] |
| DeepSeek (E8 C1K1) | 63.0 [57.7, 68.3] | +13.3 [6.3, 20.0] | - |
| oracle best-of-5 | 93.0 [90.0, 95.7] | +43.3 | |

**C2 (pruned index, 300 positives; DeepSeek = C2K3)**
| system | Hit@1 | minus fast | minus DeepSeek |
|---|---|---|---|
| fast path top-1 | 49.3 [43.7, 55.0] | - | -12.7 |
| VLM only | 51.3 [45.7, 57.0] | +2.0 [-4.0, 8.3] | -10.7 [-18.0, -3.7] |
| fusion w=0.25 / 0.5 / 1 / 2 | 50.7 / 51.7 / 52.7 / 52.3 | +1.3 / +2.3 / +3.3 [-1.0, 8.0] / +3.0 | -11.3 / -10.3 / -9.3 / -9.7 (all CIs exclude 0) |
| fusion, w by 2-fold CV | 51.3 [45.7, 57.0] | +2.0 [-2.0, 6.0] | -10.7 [-17.0, -4.0] |
| DeepSeek (E8 C2K3) | 62.0 [56.0, 67.3] | +12.7 [6.3, 19.0] | - |
| oracle best-of-5 | 91.3 [88.0, 94.3] | +42.0 | |

- Chosen w (seed-0 split): C1 0.25 on fold 1, 2 on fold 2 (the choice is unstable: over 20 random splits w=1 was picked in 27 of 40 folds, w=2 in 9, 0.25 in 4; mean pooled CV Hit@1 52.8);
  C2 0.5 / 1 (w=1 in 17, w=2 in 15 of 40). Best in-sample w is 1 for both (C1 54.0, C2 52.7). The 2-fold-CV estimate is the honest one; the in-sample best is optimistic.
- Within-query pairwise AUC (correct vs wrong candidate): p_yes 0.67 (C1) / 0.68 (C2), CLIP 0.65 / 0.67. Mean p_yes on correct candidates 0.54, on wrong ones 0.42. Max-logit ties in about 10% of queries (resolved to the better CLIP rank).
- Conclusion: on the full sample the 500M yes/no scorer is at most marginally better than CLIP (+2 to +4 pts, only the in-sample w=1 fusion on C1 touches significance) and 10-14 pts below DeepSeek with every CI excluding 0.
  The +6 / +8 pts seen on the first 100 queries was largely sampling noise (see the reference table below).

Absent events (C1: 300 positives vs 100 negatives):
| score | AUROC pos vs neg | NONE threshold for 5% NONE on positives (in-sample) | true-NONE on negatives |
|---|---|---|---|
| max_c p_yes | 0.789 [0.734, 0.841] | p_yes <= 0.182 | 37% [27, 47] |
| max CLIP | 0.733 [0.673, 0.781] | sim <= 0.231 | 9% [4, 15] |
| DeepSeek (E8 C1K3 NONE; C1K1 NONE on positives 0.7%) | - | - | 46% [36, 56] |
The p_yes AUROC is higher than CLIP's by 0.056 (CIs overlap, paired difference not tested). The NONE threshold is chosen on the same positives it is evaluated on (in-sample). C2 negatives were not run.

Reference: the first 100 positives / 50 negatives only (the earlier pass; C1; `set=first100`):
fast 49 [40, 58], VLM-only 55 [45, 65], fusion CV 53 [43, 63] (w 1 / 2), fusion w=2 57, DeepSeek 63 [53, 72], oracle 95; AUROC 0.784 [0.697, 0.856] (p_yes) vs 0.749 [0.668, 0.829] (CLIP), true-NONE on 50 neg 30% vs 2% (CLIP).
On the other 200 positives: fast 50.0, VLM-only 51.5 (+1.5), fusion CV 49.5, DeepSeek 63.0.

Position bias of E8's multiple-choice prompt (500M, K=1, same 100 positives, order shuffled):
- picks the first-shown letter (A) 99% [97, 100] of the time (E8 unshuffled: 98%); letters B-E: 0%; NONE 1%.
- Hit@1 = 43% [33, 53], exactly the rate at which the first-shown clip happens to be correct (43%), i.e. the answer is the first-shown clip, regardless of content
  (Hit@1 minus "always pick first shown" = 0.0). It picks the fast top-1 clip only 22% of the time once that clip is no longer shown first (shuffled Hit@1 minus fast top-1: -6 [-19, 7]).
  Unshuffled the 48% was the fast path's top-1 echoed back; the multiple-choice prompt carries no usable signal for this model, while the same model scored per clip (above) is at least as good as the fast path.

Timing (CPU, 2 threads, no GPU, C1 full set of 400 queries): 3.05 s per query for the 5 forward passes (0.61 s each; 5th-95th percentile 2.6-3.7 s), +0.10 s processor preprocessing, +0.29 s frame decoding (not in the forward figure);
C2: 2.97 s (some records reused from C1). Peak RSS 3.52 GB (3.55 GB after the C2 run) for the whole process. E8's 500M multiple-choice call was 2.3 s/query at K=1, so per-clip scoring is ~30% slower than one-shot choice.

## Caveats
- Full sample n=300: Hit@1 CIs are about +-5-6 pts; paired differences vs fast are within about +-5. n=100 was too small: the earlier +6 pts (VLM-only) shrank to +3 on 300.
- w is poorly identified and the seed-0 CV split picked a different w on each fold; report fusion-CV (and all w), not the in-sample best.
- NONE thresholds are in-sample; absent-event analysis (100 negatives) exists for C1 only. Negatives are the E8 C1 in-domain construction.
- One prompt, one frame per candidate (K=1), no prompt tuning, bf16 logits (coarse). Hit@1 only. DeepSeek numbers are E8's (K=1 for C1, K=3 for C2 and for negatives, per E8's conditions).
- The position-bias rerun covers only the first 100 positives (not extended); it is a separate sample of model calls (order permuted per query, seed 9), not a paired rerun.
- Timing was measured with no E7 process visible, other jobs not otherwise controlled.
