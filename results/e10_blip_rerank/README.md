# E10 - extend the fast path with a BLIP-ITM re-ranker (no VLM, no LLM, no API)

Runner `experiments/e10_blip_rerank/run.py` (stages `s1..s5`, `--limit`, `--timing`, `--aggregate-only`; resumable, scores cached per query in `raw/` and `s5_full_288/raw/`),
aggregation `aggregate.py` (sample) and `full.py` (full test set). Outputs: `summary.csv`, `per_query.csv`, `cost_table.csv`, `timing.csv`, `summary_s5_full_288.csv`,
`per_query_s5_full_288.csv`; figure `../figures/e10_blip_rerank.png`; tests `tests/test_e10_rerank.py` (utilities in `src/sseval/rerank.py`).

**Idea.** The fast path (CLIP ViT-B/32 + FAISS, pruned index cos_last_drop tau 0.95, fixed single-projection text encoder) has the right moment among its candidate groups
most of the time (oracle 83% on real groups) but Hit@1 is ~49%. `Salesforce/blip-itm-base-coco` (`BlipForImageTextRetrieval`, ITM head) scores (frame, query) -> match logit/probability
and is used ONLY to re-order the groups the fast path already formed. Score = logit(match) - logit(no match) of the peak CLIP frame of each group (K=1) or max over peak and +-1 s (K=3).

## Protocol
- Candidates: E8/E9 protocol on the pruned index (E8 "C2"): top-20 hits -> team's `group_temporally` (gap 2 s) -> groups ranked by best hit -> <=5. Candidate time = the group's peak CLIP frame;
  verified identical to E8's `cand_times` for all 400 queries. **Headline = REAL groups only (`cand_src == group`, no padding)**, which is what the app does. The "padded" variant (E8's extra next-best peaks >=3 s apart) is reported second.
- Sample: E8/E9 seed-8 sample, 300 positives + 100 negatives. Frames decoded from the raw video with OpenCV (full resolution, no JPEG), BLIP processor resize to 384 (or 288, see below).
- Hit@1 = centre (peak time) of the top-ranked candidate inside GT. Oracle/Hit@5 = any candidate inside GT (identical for every ordering; Hit@5 and "oracle" are the same number here). R@1 IoU>=0.5 = fixed 8 s window around the winning peak (E7 M2).
  Fusion: per-query z(CLIP) + w z(ITM logit) (and weighted reciprocal-rank fusion), w in {0.25,0.5,1,2,4} picked by Hit@1 on the FIRST HALF of the sample (first 150 positives), reported on the second half and on all 300 (all-300 includes the tuning half = optimistic).
  95% CIs: 1,000 query-level bootstrap resamples, paired across methods.

## Headline A - full Charades-STA test set (s5: 3,720 queries, 1,334 videos, REAL groups, K=1, 288 px, w FIXED from the 300-query sample, not tuned on the full set)
Same index/queries as E1. Groups per query: 1:182, 2:546, 3:685, 4:793, 5:1,514 (14,071 groups). Weight: z-fusion w=1, rank-fusion w=1 (both chosen on the first half of the 300 sample at 288 px).
| method | Hit@1 | Hit@3 | Hit@5 = oracle | R@1 IoU0.5 (8 s window) | R@1 IoU0.5 (group span) | mIoU (8 s) |
|---|---|---|---|---|---|---|
| CLIP order | 48.2 [46.5, 49.8] | 76.7 | 83.1 [82.0, 84.3] | 41.7 [40.2, 43.5] | 25.9 [24.4, 27.3] | 36.7 |
| BLIP-ITM only | 49.0 [47.4, 50.6] | 76.5 | 83.1 | 41.9 [40.4, 43.4] | 22.6 [21.3, 23.9] | 36.4 |
| CLIP+BLIP z-fusion | 51.2 [49.6, 52.8] | 77.4 | 83.1 | 43.6 [42.1, 45.3] | 26.2 [24.8, 27.6] | 37.9 |
| CLIP+BLIP rank-fusion | 50.0 [48.4, 51.6] | 77.4 | 83.1 | 42.7 [41.2, 44.4] | 26.0 [24.6, 27.4] | |
Paired deltas vs CLIP order: BLIP-only Hit@1 +0.9 [-0.8, +2.6] (R@1 8 s window +0.2 [-1.5, +1.6]; group-span R@1 -3.3 [-4.4, -2.1]); z-fusion Hit@1 **+3.0 [+1.9, +4.2]**, R@1 (8 s) +1.9 [+0.8, +3.0], mIoU +1.2 [+0.5, +1.9], group-span R@1 +0.4 [-0.5, +1.2]; rank-fusion Hit@1 +1.8 [+1.0, +2.7].
IoU 0.3 / 0.7 and everything else: `summary_s5_full_288.csv`. Oracle by number of groups: 1: 63.7, 2: 78.4, 3: 82.3, 4: 83.5, 5: 87.4 (queries with few groups cannot be repaired by re-ranking).
Oracle caps what re-ranking can add: fusion recovers ~9% of the 35-point gap between CLIP order (48.2) and the oracle (83.1).

**Comparison with E1 (48.2% Hit@1).** CLIP-order Hit@1 here is 48.2% [46.5, 49.8] - the same number as E1's for tau=0.95 (E1 reports Hit@1 = centre of the top-1 hit in GT, which is the first group's best hit, the same item). Protocol differences, none affecting Hit@1:
E1 groups the TOP-5 hits (so it sees <=5 groups from 5 hits, often fewer), this run takes the top-20 hits -> groups -> <=5 groups (E8 `build_candidates`), so more distinct groups survive (oracle 83.1% vs E1's Hit@5 81.6%, which counts hits, not groups); Hit@1 only depends on the best hit, hence equal. E1's R@1 IoU0.5 (7.8%) uses the one group's span around the top-1 hit with a ~1 s span from the pruned index;
the group-span R@1 here (25.9%) uses the span of the group chosen from the larger 20-hit pool (larger spans), so it is NOT comparable to E1's 7.8 - use the 8 s fixed-window column (41.7%) for localization (E7 M2).

## Headline B - the 300+100 sample (REAL groups, K=1, 384 px)
| method (n=300) | Hit@1 | Hit@3 | oracle | R@1 IoU0.5 (8 s) | paired dHit@1 vs CLIP |
|---|---|---|---|---|---|
| CLIP order | 49.3 [43.7, 55.0] | 81.3 | 83.7 [79.3, 87.7] | 44.3 [39.0, 50.3] | - |
| BLIP only | 52.3 [46.7, 57.7] | 79.3 | 83.7 | 45.7 [40.3, 51.3] | +3.0 [-2.7, 8.7] |
| z-fusion (w=1, first half) | 53.3 [47.7, 58.7] | 79.3 | 83.7 | 47.3 [42.0, 53.0] | +4.0 [0.0, 8.3] (in-sample) |
| rank-fusion (w=4) | 52.3 [46.7, 57.7] | 79.7 | 83.7 | 45.7 | +3.0 [-2.7, 8.7] |
Held-out half (150 positives, w from the other half): CLIP 49.3 [41.3, 57.3], BLIP 49.3, z-fusion 48.7, rank-fusion 49.3 -> **no gain on the held-out half at K=1** (CIs +-8). The full-set run (A) is the better-powered estimate (+3.0 [1.9, 4.2]).
Padded variant (5 candidates incl. pads; oracle 91.3%): CLIP 49.3, BLIP 53.7 [48.0, 59.0], z-fusion 55.7 [50.0, 61.0] (+6.3 [2.0, 10.7], in-sample), rank-fusion 54.0; held-out half 49.3 / 50.7 / 49.3 / 50.7. DeepSeek (E8 C2K3, padded candidates) = 62.0 [56.7, 67.0]; fused BLIP is -6.3 [-12.7, 0.0] below it (BLIP-only -9 [-15, -2]).

## Frames per candidate, resolution, query text, top-10 (first 150 positives unless noted, REAL groups; Hit@1 %)
| config | n | CLIP | BLIP only | z-fusion (w on first half of the subset) | dHit@1 fusion vs CLIP |
|---|---|---|---|---|---|
| K=1, 384 px (same 150 rows) | 150 | 49.3 | 55.3 [48.0, 63.3] | 57.3 | +8.0 [0.7, 16.0] |
| **K=3 peak+-1 s, max**, 384 px | 150 | 49.3 | 56.7 [48.0, 64.0] | 60.0 [52.0, 67.3] | +10.7 [3.3, 18.7] |
| K=3, mean prob | 150 | 49.3 | 55.3 | 55.3 | +6.0 [-1.3, 13.3] |
| K=1, **288 px** (300 rows) | 300 | 49.3 | 52.7 [47.0, 58.3] | 54.7 [48.7, 60.3] | +5.3 [1.3, 9.7] |
| K=1, 384 px (300 rows) | 300 | 49.3 | 52.3 | 53.3 | +4.0 [0.0, 8.3] |
| top-10 re-ranking (real groups, 150 rows; oracle 88.7 vs 86.7 at top-5) | 150 | 49.3 | 56.0 [48.0, 64.0] | 58.0 [50.0, 66.7] | +8.7 [0.7, 16.7] |
| same 150 rows, top-5 only | 150 | 49.3 | 55.3 | 57.3 | +8.0 [0.6, 16.0] |
- The first 150 positives happen to favour BLIP more than the other 150 (K=1 fused +8 vs +4 over 300); treat the K=3 vs K=1 comparison as paired on the same 150 rows: K=3 max adds ~+2.7 Hit@1 over K=1 (fused 60.0 vs 57.3), within noise (each CI +-8).
- **Resolution**: 288 px is no worse than 384 px on the 300 positives (BLIP only 52.7 vs 52.3, fused 54.7 vs 53.3; the 288 run is also 2x faster), so the full-set run uses **288 px**. Per-candidate quality of ITM seems saturated at the low resolution of Charades frames (270x480 source).
- **Stripping "person"/"a person"**: no difference (K=1, 300: BLIP only 51.3 vs 52.3, fused z 53.7 vs 53.3; rank-fusion 53.0 vs 52.3); within noise, the as-is query is kept.
- **Top-10**: more candidates raise the oracle by ~2 points (real groups, 86.7 -> 88.7 on these 150 rows) and re-ranking gains are the same as top-5 (+0.7 Hit@1 for BLIP-only, +0.7 fused); little reason to widen. Latency grows linearly with candidates.

## Absent events (can ITM reject the 100 negatives?) - AUROC positive vs negative (K=1, 384 px, 300 vs 100; max over the 5 candidates)
| score | AUROC [95% CI] | true NONE on negatives at 5% NONE on positives |
|---|---|---|
| **BLIP ITM max logit** | **0.818 [0.767, 0.864]** | 31% |
| BLIP ITM at the CLIP top-1 candidate | 0.795 [0.746, 0.848] | 32% |
| z(ITM max) + z(CLIP max) | 0.800 [0.748, 0.848] | 29% |
| CLIP max similarity | 0.727 [0.668, 0.781] | 5% |
| (E9: SmolVLM2-500M max p_yes) | 0.789 | 37% |
Paired CIs not computed; the ITM-CLIP gap is +0.09 and the CIs barely overlap. 288 px: ITM max 0.786 [0.710, 0.857] vs CLIP 0.745 on 300 vs 50 negatives. K=3 max (150 vs 50): 0.795 vs CLIP 0.746. The thresholds are in-sample.
BLIP separates absent from present queries better than CLIP (the E3 weakness) but a 31% rejection rate at 5% false rejects is still weak; DeepSeek's correct-NONE was 46-49%.

## Cost (CPU, 2 threads, no other load; `timing.csv`: 30 real queries, 5 candidates, one text variant)
| stage | seconds |
|---|---|
| CLIP retrieval core per query (text embed + FAISS; E6) | 0.016 |
| full E6 query pipeline wall (incl. query classification, evidence frame extraction) | 0.75 |
| BLIP-ITM per (image,text) pair, 384 px, batch 5 | 0.747 |
| BLIP-ITM per pair, 288 px | 0.35 |
| + K=1, 5 candidates: ITM / decode | 3.61 s / 0.24 s at 384 px; 1.70 s / 0.24 s at 288 px |
| + K=3, 5 candidates (15 images): ITM / decode | 10.8 s / 0.68 s at 384 px; 5.05 s / 0.68 s at 288 px |
| mean real groups per query (sample 4.0, full set 3.8) | real-group K=1 re-rank ~ 3.0 s (384) / ~1.4 s (288) |
Vision tower dominates (a second query text variant costs +15%, the text encoder cross-attends 577 image tokens). Memory ~2 GB. The full s5 run took ~1.7 h for 14,071 groups at 288 px (1.6 s/query incl. decode).

**Hit@1 vs added latency** (real groups unless noted; sample of 300, K=3 on 150):
| system | Hit@1 | added latency per query |
|---|---|---|
| fast path (CLIP) | 49.3 [43.7, 55.0] (full set 48.2) | 0 (0.016 s core, 0.75 s full pipeline) |
| fast + BLIP K=1, z-fusion, 384 px | 53.3 [47.7, 58.7] (in-sample w) | ~3.0-3.9 s |
| fast + BLIP K=1, z-fusion, 288 px | 54.7 [48.7, 60.3]; full set 51.2 [49.6, 52.8] | ~1.4-1.9 s |
| fast + BLIP K=3 max, z-fusion, 384 px (150 rows) | 60.0 [52.0, 67.3] | ~11.5 s (288 px: ~5.7 s, not measured for Hit@1 at K=3) |
| DeepSeek slow path (E8 C2K3, padded candidates, API) | 62.0 [56.7, 67.0] | 1.45 s API round trip (+0.3 s frames) |
| SmolVLM2-500M yes/no (E9 C2, padded, local) | 51.3 [45.7, 57.0] | ~3.4 s |
BLIP on CPU is NOT cheaper than the DeepSeek API call; its advantage is local/free/no data egress. Figure: `../figures/e10_blip_rerank.png`.

## Conclusions
1. BLIP-ITM re-ranking helps modestly: on the full test set the CLIP+BLIP fusion lifts Hit@1 from 48.2 to 51.2 (+3.0 [1.9, 4.2]) and R@1 (8 s window) by +1.9; BLIP alone is not distinguishable from CLIP order. It recovers ~9% of the 35-point oracle gap, far below DeepSeek (+13 on the sample).
2. Three frames (peak +-1 s) look better than one (+2.7 on the same 150 rows, within noise) at 3x the cost; stripping "person", a bigger image size and top-10 do not matter.
3. ITM is a better absent-event detector than CLIP (AUROC 0.82 vs 0.73) but not a reliable one.
4. Hit@1 stays well below the oracle: BLIP-ITM (COCO captions) understands objects/scenes, while Charades queries are indoor actions ("person closing the door") performed by the same actor in near-identical frames.

## Caveats
- BLIP is trained on COCO captions; Charades-STA is indoor, low-resolution (270x480, often dark), action-centred video; one frame cannot show an action.
- w is picked on 150 queries and its in-sample number is optimistic; the held-out half of the 300 sample shows no gain at K=1 (CI +-8). The full-set run (3,720 queries, w fixed) is the reliable estimate; it was scored at 288 px (the sample at 384 px; 288 px was shown equivalent on the sample).
- K=3, top-10 and 288 px comparisons use the first 150 positives (K=3, top-10) or 300 (288 px), not the full set; their CIs are +-8 points.
- The real-group counts on the pruned index differ from the full-5-FPS counts (this run: 1:5%, 2:15%, 3:18%, 4:21%, 5:41% of the full set; the 300-sample 1:13, 2:38, 3:57, 4:69, 5:123). DeepSeek and SmolVLM2 numbers come from E8/E9 with padded candidates, so they are not strictly comparable to the real-group rows.
- HF `BlipForImageTextRetrieval` feeds [CLS] where the original BLIP uses an [ENC] token; a check on 5 frames showed negligible difference (<0.01 in probability). ITM probabilities saturate near 0/1, so logits are used.
- Latency is CPU wall time, 2 threads, a shared VM; DeepSeek latency is the E8 API round trip.
