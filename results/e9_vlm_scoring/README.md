# E9 - small local VLM (SmolVLM2-500M) as a per-clip scorer

Runner: `experiments/e9_vlm_scoring/run.py` (`--limit N`, `--skip-posbias`, `--aggregate-only`, `--out DIR`; resumes from `raw/` and `posbias/raw/`).
Outputs: `summary.csv` (long format: section, name, metric, n, value, lo, hi), `per_query.csv` (5 CLIP scores, 5 p_yes, per-method hits, DeepSeek choice),
`posbias_per_query.csv`, `raw/` (one JSON per query x candidate forward pass), `posbias/raw/`, figure `../figures/e9_vlm_scoring.png`. Unit test: `tests/test_e9_vlm_scoring.py`.
E8 files/results untouched; E9 imports E8's `run.py` (candidates, frame decoding, parser, prompt) and `local_vlm.py` (model loading, `run_local`).

## Design
- Queries: the same 100 positives + 50 negatives as the E8 local runs (first of the seed-8 C1 sample); candidates = E8 C1K1 (5 per query; 5 of the 150 queries, 3 positives, have only 4,
  handled with NaN padding; candidate times verified identical to E8's `per_query.csv`).
- Frame: K=1, the 1 FPS grid frame nearest the candidate time (E8's `clip_times`), 448 px JPEG q80, processor longest edge 384, no splitting. bf16, 2 threads.
- Yes/no scoring: 5 separate forward passes per query (one image each), prompt `Does this image show: '<query>'? Answer Yes or No.` via the chat template, generation prompt appended, no generation.
  Next-token logits; Yes ids = {"Yes", " Yes", "yes", " yes", "YES", " YES"} (single-token variants only), No likewise; each side = max logit over variants; p_yes = 2-way softmax.
  The model's top token was always " Yes" or " No" (leading-space variants), so the max-over-variants matters. ~94 input tokens per pass.
- Fusion: per query, z-normalise the 5 CLIP scores and the 5 logit(p_yes) (= yes-no logit difference); score = z_clip + w z_vlm, w in {0.25, 0.5, 1, 2}.
  w chosen by 2-fold CV over the 100 positives (seed 0 split; fit = best Hit@1 on one half, ties -> smaller w; evaluated on the other half; pooled). All w are also reported (in-sample).
- Position-bias check: E8's A-E multiple-choice prompt (K=1, lenient bare-letter parser, same as the E8 local run), candidate order shuffled with a per-query seeded permutation.

## Results (100 positives, Hit@1 %, bootstrap 95% CI, paired differences vs fast path)
| system | Hit@1 | minus fast |
|---|---|---|
| fast path top-1 (CLIP) | 49 [40, 58] | - |
| VLM only: argmax p_yes | 55 [45, 65] | +6 [-7, +19] |
| fusion w=0.25 / 0.5 / 1 / 2 (in-sample) | 52 / 53 / 54 / 57 | +3 [-2, 8] / +4 [-2, 10] / +5 [-3, 13] / +8 [-3, 20] |
| fusion, w by 2-fold CV (chosen w: 1 on fold 1, 2 on fold 2) | 53 [43, 63] | +4 [-6, +15] |
| DeepSeek (E8 C1K1, same queries) | 63 [53, 72] | +14 [3, 25] |
| oracle best-of-5 | 95 [90, 99] | +46 |
Fusion-CV minus VLM-only -2 [-10, 6]; fusion-CV minus DeepSeek -10 [-20, 1]; VLM-only minus DeepSeek -8 [-18, 2].
Over 20 random 2-fold splits the pooled CV Hit@1 averages 53.05; w=2 was chosen in 24 of 40 folds, w=1 in 7, 0.25 in 5, 0.5 in 4 (w is poorly identified at n=100; larger w is monotonically better here).
Within-query pairwise AUC (correct vs wrong candidate, queries having both): p_yes 0.69, CLIP 0.63. Mean p_yes: 0.56 on correct candidates (n=161), 0.41 on wrong ones (n=336).
In 9% of queries the max VLM logit is tied between candidates (bf16 logits are coarse; ties resolve to the better CLIP rank).

Absent events (100 pos vs 50 neg; AUROC, bootstrap CI over both sets):
| score | AUROC | NONE threshold for 5% NONE on positives (in-sample) | true-NONE on negatives |
|---|---|---|---|
| max_c p_yes | 0.78 [0.70, 0.86] | p_yes <= 0.202 | 30% [18, 42] (positives NONE: 5%) |
| max CLIP | 0.75 [0.67, 0.83] | sim <= 0.227 | 2% [0, 6] |
The threshold is chosen on the same positives it is evaluated on (in-sample), so the 5% positive rate is by construction. For reference E8's DeepSeek had 46% true-NONE on 100 negatives (C1K3, 1.3% false rejects).

Position bias of E8's multiple-choice prompt (500M, K=1, same 100 positives, order shuffled):
- picks the first-shown letter (A) 99% [97, 100] of the time (E8 unshuffled: 98%); letters B-E: 0%; NONE 1%.
- Hit@1 = 43% [33, 53], exactly the rate at which the first-shown clip happens to be correct (43%), i.e. the answer is the first-shown clip, regardless of content
  (Hit@1 minus "always pick first shown" = 0.0). It picks the fast top-1 clip only 22% of the time once that clip is no longer shown first (shuffled Hit@1 minus fast top-1: -6 [-19, 7]).
  Unshuffled the 48% was the fast path's top-1 echoed back; the multiple-choice prompt carries no usable signal for this model, while the same model scored per clip (above) is at least as good as the fast path.

Timing (CPU, 2 threads, no GPU): 3.24 s per query for the 5 forward passes (0.65 s each; 5th-95th percentile of per-query 2.8-3.9 s), +0.10 s processor preprocessing, +0.30 s frame decoding
(not in the forward figure). Peak RSS 3.51 GB for the whole process (the E8 500M MC run reported 3.9 GB). For comparison, E8's 500M multiple-choice call was 2.3 s/query at K=1.

## Caveats
- n=100 positives: every CI on Hit@1 is about +-10 pts; none of the VLM/fusion gains over the fast path has a CI excluding 0 (DeepSeek's +14 does). Do not read the VLM-only 55 vs fusion 53 ordering as meaningful.
- w by 2-fold CV with 4 grid values on 50 queries per fold is noisy; the pooled CV number (53) is below the in-sample best (57, w=2) as expected.
- NONE rule thresholds are in-sample; absent-event analysis uses 50 negatives (E3-style in-domain construction from the DeepSeek C1 sample).
- One prompt, one frame per candidate (K=1), no prompt tuning, bf16 logits (coarse; 9% max-logit ties). Hit@1 only (no IoU / widened-span metrics were requested).
- Candidate time = fast-path peak time; the single frame is the 1 FPS grid frame nearest it, so the VLM sees the same moment CLIP scored.
- Timing was measured with E7 not running (no `e7_localization` process or results folder existed during the run), but other agents' jobs were not otherwise controlled.
- The position-bias rerun is a separate sample of model calls (order permuted per query, seed 9), not a paired rerun of the same prompts.
