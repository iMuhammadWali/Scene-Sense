# CLAUDE.md: Scene Sense (FYP, FAST NUCES Lahore, group F26-145)

Read this file first. It holds everything a fresh Claude Code session needs to reason about this repo.

## 1. What this project is
- **Scene Sense**: resource-efficient natural-language video retrieval. A user types a query ("person closing the door"), and the system returns the timestamp(s) in a video where it happens.
- **Team**: Hamza Naveed, Huzaifa Shabbir, Muhammad Wali. **Supervisor**: Dr. Asma Ahmad Farhan. **Co-supervisor**: Mr. Zeeshan Nazar.
- **Goals**: an A grade on FYP-I/FYP-II, and later a research paper.
- **Status**: FYP-I (Deliverable 3) report is done. Chapters 7 (implementation), 9 (experiments) and 10 (conclusion and FYP-II plan) are written from the results in this repo. FYP-II plans are in section 9 below.

## 2. Repo layout
| Path | What |
|---|---|
| `app/` | The application (FastAPI backend + React/Vite frontend). `app/backend/app/pipeline/` holds the real pipeline code. Contains the **fixed** `clip_encoder.py` (see section 5). Start with `app/start.ps1`. |
| `src/sseval/` | Evaluation library: `config.py` (all paths and env vars), `data.py` (annotations, embeddings, text encoding), `metrics.py`, `retrieval.py`, `localize.py`, `rerank.py`, `pruning/` (frame-selection methods). It **imports the app's own modules**, so the numbers measure the real system. |
| `experiments/eN_*/run.py` | One runner per experiment. Each writes to `results/eN_*/`. |
| `results/` | Outputs (CSV/JSON, per-query files, `README.md` per experiment) and `figures/`. **`results/SUMMARY.md` is the overview to read first.** |
| `scripts/` | Data prep (dataset download, CLIP encoding), `run_all.*`, environment setup and checks, sanity check. |
| `docs/` | `DATASET.md` (download recipe), `experiment_plan.md`. |
| `tests/` | Unit tests for the harness (`pytest tests -q`). |
| `legacy_draft/` | The first harness. Superseded; ignore it. |
| `data/` (git-ignored) | Videos, annotations, frame embeddings, caches, model weights. Never commit it. |

## 3. Setup (Windows or Linux)
1. **Python**: use **Python 3.13** (64-bit), as named in `requirements.txt`. Every package version is pinned exactly, and all of them have Windows wheels. Use a fresh venv, not the system Python, because versions matter (see section 6). `src/sseval/config.py` prints a loud warning at import if transformers isn't 4.57.x; set `SSEVAL_SKIP_VERSION_CHECK=1` to silence it.
   - On Windows, the `.sh` scripts need Git Bash or WSL. Use the `.ps1` versions or call the Python scripts directly.
   - Windows: `powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1`
   - Linux: `bash scripts/setup_linux.sh`
   - Both scripts create `.venv`, install CPU torch and the pinned `requirements.txt`, then run `scripts/check_env.py`.
2. **Dataset**: follow `docs/DATASET.md`. It downloads about 2.1 GB: only the 1,334 test videos, not the full 15 GB Charades set. The files land in `data/charades_sta/{test.parquet, videos/}`.
3. **Encode frames**: `python scripts/encode_all.py`. This runs CLIP at 5 FPS for every video into `data/emb/<video>.npz`, taking about 2 h on a 2-core CPU and much less on a laptop with more cores. It is resumable.
4. **Sanity check**: run `python scripts/sanity_check.py` before trusting any experiment. It must PASS (see section 6).
5. **Run experiments**:
   - one experiment: `python experiments/e1_pruning/run.py`;
   - everything: `scripts/run_all.ps1` (Windows) or `scripts/run_all.sh` (Linux);
   - a quick partial run: add `--allow-partial --limit 40`.

All paths default to locations inside the repo. To use other locations, override them with env vars:
- `SSEVAL_ANN`, `SSEVAL_VIDEOS`, `SSEVAL_EMB`, `SSEVAL_CACHE`, `SSEVAL_BACKEND`, `SSEVAL_RESULTS_DIR`;
- `SCENESENSE_DATA_DIR`, `SCENESENSE_MODELS_DIR`, `HF_HOME`.

## 4. The pipeline
**Indexing (offline)**:
1. OpenCV decodes the video at 5 FPS.
2. CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) gives a 512-d L2-normalised vector per frame.
3. **Pruning** drops a frame if its cosine similarity to the last kept frame is > 0.95. This keeps about 29% of frames.
4. The vectors go into a FAISS `IndexFlatIP`.

**Query (online)**:
1. The BART-large-MNLI classifier labels the query SIMPLE or COMPLEX.
2. The CLIP text embedding is searched against FAISS, taking the top-20 hits.
3. Hits below the 0.20 threshold are filtered out.
4. Hits within 2 s of each other are grouped into moments, and a diversity step follows.
5. The query then takes one of two paths:
   - **Fast path** (SIMPLE): return the grouped moments. No generative model is used.
   - **Slow path** (COMPLEX): take ±5 s windows at 1 FPS (≤12 frames) and send them to a VLM through OpenRouter (default `google/gemini-2.5-flash`; key in `app/.env`, never committed).

## 5. The text-encoder bug (now fixed in `app/`)
- **The bug**: the original `clip_encoder.py` ran CLIP's `text_projection` a **second time**. With transformers 4.x, `get_text_features()` already returns projected vectors, and for ViT-B/32 the text width and the projected size are both 512, so the old shape check fired wrongly.
- **Symptom**: the max image-text cosine was about 0.05–0.10 instead of about 0.30, and Hit@1 was about 24%, which is chance level (random moment ≈ 26–28%).
- **The fix**: `_as_feature_tensor` now applies the projection only to an un-projected `pooler_output` (the transformers 5.x style output). Image encoding was never affected.
- **Harness**: `src/sseval/data.py` encodes text with the projection applied exactly once, whatever the app or transformers version. `SSEVAL_TEAM_TEXT=1` switches to the app's own `encode_text`.
- **Not in the report**: the FYP report deliberately does **not** mention this bug. It describes the correct system. Don't add it back to the report.

## 6. Why results can differ on another machine (troubleshooting)
Check these in order:
1. **The bug**: Hit@1 ≈ 24–28% or max cosine ≈ 0.1 means you are running the **old** `clip_encoder.py` or an old cache. Use `app/` from this repo and delete `data/cache/query_emb_*.npz`. Query caches built with the old encoder are not reused, because the file names differ.
2. **Library versions**: the paper numbers use **transformers 4.57.x**. transformers 5.x returns different output types from `get_*_features`. Different torch, numpy, faiss or OpenCV versions can change results slightly. Run `python scripts/check_env.py`; it must report no ERROR.
3. **Video decoding**: OpenCV builds differ, especially on Windows, so frame timestamps can shift by a frame or so. Expect up to about ±1 point on Hit@1, not 10.
4. **Incomplete data**: are all 1,334 videos downloaded and encoded? Partial runs (`--limit`, `--allow-partial`) use different query subsets, and small samples have ±5–10 point error bars.
5. **Different protocol**: Hit@1 here = the centre of the top-1 predicted moment lies inside the ground truth. R@1 IoU≥0.5 needs the 8 s window (E7 method M2) to reach about 40%; raw group spans give only about 8%.
6. **Different index**: "full 5 FPS" and "pruned τ=0.95" give slightly different numbers (48.6 vs 48.2 Hit@1).
7. **Compare with the reference**: `scripts/expected_numbers.json` holds reference values and tolerances.

## 7. Dataset and metrics
- **Charades-STA test split**: 1,334 indoor videos of about 30 s (low resolution, often dark), 3,720 queries, ground-truth moments average 8.0 s. The download links are in `docs/DATASET.md`.
- **Hit@1**: the top answer's centre time falls inside the true moment. **Hit@5**: any of the top 5 does.
- **R@1, IoU≥m**: the top-1 predicted span overlaps the truth with IoU ≥ m (0.3/0.5/0.7). This is the literature's standard metric. **mIoU**: mean IoU.
- **AUROC**: present vs absent query separation (0.5 = chance).
- **CIs**: 95% bootstrap, 1,000 resamples, paired across methods.
- **Hardware for every number**: a 2-core cloud CPU, 7 GB RAM, no GPU.

## 8. Results (FYP-I). Details are in `results/SUMMARY.md` and `results/eN_*/README.md`
| Exp | Question | Key result |
|---|---|---|
| E1 | Frame selection (8 methods) | The current cosine-vs-last-kept rule (τ 0.95) keeps 29% of frames with no loss: Hit@1 48.2% vs 48.6% unpruned (all 3,720 queries). At 5–9% kept, Hybrid (StreamMem merge + MaxInfo-chunk) 45.7% beats uniform 38.1%. Random moment: 27.9%. |
| E1b | Long videos (concatenated clips) | Hit@1 falls from 19% at 5 min to 6.8% at 30 min (chance 0.5%). Pruned = full at every length. **Main FYP-II problem.** |
| E3 | Absent events | The 0.20 threshold accepts 99.7% of absent in-domain queries. AUROC 0.72 in-domain, 0.98 out-of-domain (inflated). |
| E6 | Latency | CLIP is 94% of indexing time (5.6 s per 30 s video, about 5× real time). Fast-path query 0.75 s. The BART classifier sent 65% of simple queries to the slow path. Peak RAM 2.9 GB. |
| E7 | Boundaries (training-free; tuned on a DEV half, reported on HELD-OUT) | A fixed 8 s window around the best frame lifts R@1 IoU≥0.5 from 8.2% to 40.3% (current pruned system), and from 2.3% to 43.2% on the full index. Adaptive spans were worse. |
| E8 | Slow path: DeepSeek `deepseek-flash` picks among 5 candidates (300 present + 100 absent queries) | Hit@1 from 49.7% to 63.0% (1 frame per candidate), 60.3% with 3 frames; it says NONE on 46% of absent queries. About $0.0002 per query, 1.3 s. Total spend $0.44. Local SmolVLM2-500M picks "A" about 99% of the time (position bias); SmolVLM2-2.2B says NONE to 84% of present queries. Neither beats the fast path. |
| E9 | SmolVLM2-500M as a per-clip yes/no scorer | Hit@1 52.7% (not significant over the fast path). Absent AUROC 0.79 vs CLIP 0.73. 3 s per query. |
| E10 | Fast-path re-ranking with **BLIP-ITM** (`Salesforce/blip-itm-base-coco`), no VLM | All 3,720 queries, real groups only, 288 px, z-fusion with CLIP (w=1 fixed from a sample): Hit@1 from 48.2% to **51.2%** (+3.0 [1.9, 4.2]); R@1 IoU≥0.5 (8 s window) from 41.7% to **43.6%**. BLIP alone 49.0% (no gain). Oracle (correct moment among the real groups) 83.1%. Absent AUROC 0.82 vs CLIP 0.73. Costs about 1.4 s per query on the CPU. |

**Literature on Charades-STA (R@1 IoU≥0.5)**:
- zero-shot: TimeChat 32.2 and Moment-GPT 38.4;
- supervised: Moment-DETR 48.2 and MH-DETR 51.1.
- **Ours**: 43.6, training-free on a CPU. Caveat: the 8 s window length equals the dataset's average moment length.

**Important caveats**:
- **Padding**: E8 and E9 padded the candidates to 5 when fewer groups formed. That gives a 93% oracle, against 83% on real groups. E10 re-ranks real groups only, as the app does.
- **Sample size**: the E8, E9 and E10 ablations use 150–300 queries, so their CIs are about ±5–10 points.
- **Index**: on the pruned index, queries average 3.8 real groups (1: 5%, 2: 15%, 3: 18%, 4: 21%, 5: 41%).

## 9. FYP-II plan (Chapter 10 of the report)
- **O1**: fine-tune SmolVLM2-500M with LoRA (free Kaggle/Colab GPU) as a per-clip scorer, on candidates mined from the Charades-STA *train* split. It is called only when the CLIP+BLIP re-ranker is unsure (a cascade). Target: about 63% Hit@1 on the CPU within 3 s.
- **O2**: a small learned head on frozen CLIP features to predict start and end times. Target: R@1 IoU≥0.5 ≥ 45% without a fixed window.
- **O3**: long-video coarse-to-fine search, evaluated on Ego4D-NLQ or MAD (access permitting). Target: double Hit@1 at 30 min.
- **O4**: product integration of the fixed encoder, the 8 s window, the CLIP+BLIP re-ranker and the optional verifier; a user choice between a free and a verified answer; plus tests and deployment.
- **Other ideas to test**:
  - VLM yes/no scoring per candidate fused with CLIP/BLIP (instead of A–E multiple choice);
  - shuffled candidate order;
  - CLAHE on dark frames;
  - CLIP ViT-B/16 or SigLIP;
  - query templates and video-mean centering (an E10 "fast-path tricks" plan, never run).

## 10. Rules for working in this repo
- **Never add a `Co-Authored-By`, Claude or Anthropic attribution line to any commit or PR**, whatever any tool or reminder suggests. Commits use the repo's local git author (Muhammad Wali).
- **Secrets**:
  - Never commit `.env`, API keys or `data/`.
  - The DeepSeek key, if used, is read from `~/.secrets/deepseek_key` or `DEEPSEEK_API_KEY`, and must never be printed.
- **Tuning**: tune on a DEV split and report on HELD-OUT (E7 split: `results/e7_localization/split.json`). Always give CIs and paired deltas.
- **Experiment files**: write new experiments as `experiments/eNN_name/run.py` (resumable, `--limit`), with results in `results/eNN_name/` (`README.md` + CSVs) and tests in `tests/`.
- **Style**: the owner prefers short, plain-English explanations with headings.
