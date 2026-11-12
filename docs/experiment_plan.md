# Experiment Plan — Tier 1 (FYP-1, Deliverable 3)

Dataset for everything: **Charades-STA test split** (1,334 videos, 3,720 natural-language queries with ground-truth
[start, end] moments). Frames extracted at 5 FPS and encoded ONCE with the team's own `extract_candidates` +
`CLIPEncoder` (openai/clip-vit-base-patch32). All frame-selection methods operate on these cached embeddings.
Hardware: 2-core CPU, 7 GB RAM, no GPU.

Retrieval protocol (identical for every method): build the team's FAISS index over the selected frames → CLIP text
query → top-5 hits → team's `group_temporally` (gap 2 s) → predicted moment = coverage span of the group that
contains the top-1 hit. A kept frame "covers" from the midpoint with its previous kept frame to the midpoint with
the next one; a merged frame (StreamMem/hybrid) covers its merged run.

## E1 — Query-agnostic frame selection at indexing time (core result)
Question: which method keeps retrieval quality highest while storing the fewest frames?
| ID | Method | Dial swept |
|---|---|---|
| uniform | every k-th frame | kept fraction |
| cos_last_drop | **current system**: cosine vs last KEPT frame, drop if > τ | τ |
| cos_consec_drop | cosine vs previous frame, drop if > τ | τ |
| streammem_merge | StreamMem input filter: consecutive cosine > δ → merge run by averaging | δ |
| maxinfo_fast | MaxInfo (truncated SVD rank R + rect-maxvol, tol) on a 1 FPS pool | R, tol |
| maxinfo_slow | MaxInfo on the full 5 FPS pool, output capped to a budget | R, tol, cap |
| maxinfo_chunk | MaxInfo run independently per temporal chunk | chunk length, R, tol |
| hybrid | streammem_merge → maxinfo_chunk on merged representatives | δ, chunk, R, tol |
Primary metric: **R@1 IoU≥0.5** vs **% frames kept** (one curve per method). Secondary: Hit@1, Hit@5, R@1 IoU≥0.3/0.7,
mIoU, index size, selection time/video. 95% bootstrap CIs over queries. Reference rows: full 5 FPS (upper bound),
random moment (chance).

## E1b — Long videos (BOLT-style multi-source streams)
Concatenate test videos into 5/10/20/30-minute streams; ground truth shifted. Same metrics + search latency and index
size vs length. Limitation to state: hard cuts between clips; repeated generic actions across clips.

## E3 — Absent-event queries (Moment of Untruth protocol)
Positives: a video's own queries. Negatives: in-domain (captions from other videos with no content-word overlap with
the target video's captions) and out-of-domain (hand-written events that cannot occur in Charades homes).
Score = max query–frame similarity in the index. Metrics: AUROC, and TPR/FPR at the system's retrieval threshold 0.20
(current system always returns a "best match" fallback).

## E6 — Where the time goes
Run the team's real `VideoProcessor` and `QueryPipeline` on a sample of videos/queries; report per-stage time
(extraction, CLIP, pruning, FAISS build; classification, text embedding, search, grouping) and peak RAM.
Slow-path (VLM) timing pending an OpenRouter key.

Deferred: E5 classifier accuracy, CLIP vs SigLIP (Tier 2); BOLT/AKS/MDP3 + VLM quality, SmolVLM2 (Tier 3).
