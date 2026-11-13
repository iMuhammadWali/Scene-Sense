# E8 - slow path: VLM verification (DeepSeek `deepseek-flash`)

Runner: `experiments/e8_slowpath/run.py` (`--dry-run`, `--list-models`, `--limit N`, `--aggregate-only`; resumes from `raw/`).
Outputs: `summary.csv` (metrics + bootstrap 95% CIs, cost/latency), `per_query.csv`, `raw/` (API responses, no key, no images),
`examples/` (3 prompt image grids), `../figures/e8_fast_vs_slow.png`. `raw_trial1_thinking/` = an abandoned 17-call trial (see Caveats).

## Design
- 300 positive queries (seed 8, <=1 query/video) + 100 in-domain negatives (E3 construction: caption of another video with no content-word
  overlap with ANY caption of the target video; target videos disjoint from the positive videos).
- Fast-path candidates: top-20 hits -> `group_temporally` (gap 2 s) -> groups ranked by best hit -> 5 groups, padded with next-best peaks
  >=3 s apart. Candidate time = group's best-hit time; span = group coverage span; widened span = time +-4 s. Fixed single-projection text embeddings.
- Frames decoded from the raw video: +-5 s window at 1 FPS (as the real slow path), K evenly spaced per candidate (K=3 main, K=1 cheap), long side 448 px, JPEG q80.
- One call per query (clips A-E, "Clip A (t=12.4 s):" + K images), JSON `{"choice","confidence"}`, temperature 0, parse failure -> one retry.
- Conditions: C1K3 = full 5 FPS index (300 pos + 100 neg); C2K3 = cos_last_drop tau 0.95 index (300 + 100); C1K1 = full index, K=1 (300 pos).
  All 1,100 planned calls completed (no budget stop). Slow = VLM's choice, NONE/invalid counted as a miss; "fallback" variant uses fast top-1 on NONE.

## Results (300 positives; Hit@1 and R@1 IoU0.5 in %, bootstrap 95% CI)
| cond | system | Hit@1 | R@1 IoU0.5 (candidate span) | R@1 IoU0.5 (widened +-4 s) |
|---|---|---|---|---|
| C1K3 | fast top-1 | 49.7 [43.7, 55.3] | 21.3 [16.7, 26.0] | 38.0 [32.3, 43.3] |
| C1K3 | slow VLM | 60.3 [54.7, 65.7] | 13.0 [9.3, 16.7] | 45.7 [39.7, 51.0] |
| C2K3 | fast top-1 | 49.3 [43.7, 54.7] | 24.7 [20.0, 29.7] | 36.0 [30.7, 42.0] |
| C2K3 | slow VLM | 62.0 [56.7, 67.0] | 23.7 [19.0, 28.0] | 45.3 [39.7, 51.0] |
| C1K1 | fast top-1 | 49.7 [43.7, 55.3] | 21.3 [16.7, 26.0] | 38.0 [32.3, 43.3] |
| C1K1 | slow VLM | 63.0 [57.3, 68.0] | 12.7 [9.0, 16.7] | 53.7 [48.0, 59.0] |
Paired slow-fast Hit@1: C1K3 +10.7 [3.7, 17.3] pts; C2K3 +12.7 [6.3, 19.3]; C1K1 +13.3 [7.0, 19.7] (see summary.csv for R@1 deltas).
Pruned vs full for the slow path (K=3, paired): Hit@1 +1.7 [-4.0, 6.7] pts, widened R@1 -0.3 [-6.0, 5.3] -> no measurable effect of tau=0.95 pruning on VLM verification.
Oracle best-of-5 Hit@1 (reranking upper bound): 93.0 (C1), 91.3 (C2). The VLM recovers about a quarter of the headroom.

Absent events (100 negatives): fast path never rejects (0%); VLM correct-NONE 46% [37, 56] (C1K3), 49% [39, 59] (C2K3). False rejects on positives: 1.3% (C1K3), 2.3% (C2K3), 0.7% (C1K1).
Parse failures: 0 invalid responses in all 1,100 calls (no retries needed).

Cost and latency (per query = one call): C1K3 1.48 s mean (p95 1.85 s), ~3,070 input / 14 output tokens, $0.00052 actual (off-peak; $0.00091 at peak price);
C2K3 1.45 s, ~2,970 in tokens; C1K1 1.28 s, ~1,130 in tokens, $0.00022 actual ($0.00039 peak). Images bill ~207-228 tokens each, far under the 1,024 cap.

## Caveats
- Spend: $0.441 actual (all 1,133 calls were off-peak, Fri ~05:20-05:50 UTC); $0.881 at worst-case peak price. Both include the 17-call abandoned trial. Cap was $1.00 (worst-case peak); never reached.
- `deepseek-flash` reasons by default: the first trial (max_tokens 256) returned empty content (all tokens spent on reasoning). The real run sends `"thinking": {"type": "disabled"}` (E8_THINKING env var). Results are for non-thinking mode; latency would be higher with thinking.
- Hit@1 improves but span-based R@1 IoU0.5 does not: the candidate span from a full 5 FPS index is ~1 s, so IoU against ~8 s ground truth is low whatever is chosen. Use the widened span (+-4 s) column for localization.
- Position/label bias, label time ("t=") in the prompt and a single prompt were not ablated. Retrieval metrics use one query per video, not all 3,720 queries.
- Peak window assumed Mon-Fri 01:00-04:00 and 06:00-10:00 UTC (end-exclusive). Cached input tokens (prompt_cache_hit) are billed at the miss price (conservative).

## Local VLM backend (SmolVLM2 on CPU) - `experiments/e8_slowpath/local_vlm.py`
Same candidates, prompt, parser (plus lenient bare-letter parsing, see below) and metrics; first 100 positives + 50 negatives of the DeepSeek C1 sample.
Outputs per model: `local_500M/`, `local_2.2B/` (`raw/`, `per_query.csv`, `summary.csv`, `compare.csv` = paired vs fast path and vs DeepSeek on the same queries).
Settings: bfloat16, 2 CPU threads, greedy, max 32 new tokens, frames = the same 448 px JPEGs, processor longest edge 384 px, no image splitting
(~64-81 visual tokens/image; 500M: ~1,180 input tokens at K=3, 2.2B: ~1,440). Figure: `../figures/e8_local_vs_deepseek.png`.

| model | cond | n pos / neg | Hit@1 [95% CI] | fast top-1 (same queries) | DeepSeek (same queries) | NONE on pos | correct NONE on neg | s/query | peak RAM |
|---|---|---|---|---|---|---|---|---|---|
| SmolVLM2-500M | C1K1 | 100 / 50 | 48 [38, 58] | 49 | 63 | 2% | 4% | 2.3 | 3.9 GB |
| SmolVLM2-500M | C1K3 | 100 / 50 | 49 [40, 59] | 49 | 62 | 0% | 0% | 6.8 | 3.9 GB |
| SmolVLM2-2.2B | C1K1 | 100 / 50 | 11 [6, 18] | 49 | 63 | 84% | 90% | 7.3 | 6.0 GB |
| SmolVLM2-2.2B | C1K3 | 26 / 12 (PARTIAL, stopped on request) | 31 [15, 50] | 50 | 65 | 42% | 92% | 20.3 | 6.0 GB |
Paired Hit@1 local - fast (95% CI): 500M K1 -1 pts [-3, 0]; 500M K3 0 [0, 0]; 2.2B K1 -38 [-48, -28]; 2.2B K3 (n=26) -19 [-35, -8].
Local - DeepSeek: 500M K1 -15 [-26, -3]; K3 -13 [-24, -1]; 2.2B K1 -52 [-62, -42]; K3 (n=26) -35 [-54, -12]. Neither small model beats the fast path.

Answer-distribution diagnosis (positives): 500M answers "A" 98% (K1) / 100% (K3) - pure position bias, i.e. it just echoes the fast path's top-1 (A) and its Hit@1 equals the fast path's;
negatives also 96-100% A. 2.2B answers NONE 84% (K1) / 42% (K3) of the time on positives (blanket rejection; 90% NONE on negatives is therefore not discrimination); when it does commit its accuracy
is 69% (n=16, K1) / 53% (n=15, K3). DeepSeek spreads answers A 37 / B 22 / C 19 / D 15 / E 6 % and says NONE 1% on positives, 46% on negatives, which is the only genuinely discriminating behaviour.
Parse: 0 invalid answers after one retry. Neither small model follows the JSON format (JSON-format failure 100% for 500M, 87-100% for 2.2B; they reply with a bare letter or NONE), so the local backend
accepts a bare letter / NONE (`parse_response(lenient=True)`; the DeepSeek run used the strict parser). Confidence is therefore not available locally.
Caveats: prompt not tuned for SmolVLM2; low image resolution (384 px, no splitting) and bf16; 2.2B K=3 stopped early on request (26 positives / 12 negatives only); Hit@1 CIs on 100 queries are wide (+-10 pts).
