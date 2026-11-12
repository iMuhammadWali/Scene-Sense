# Scene Sense - Tier-1 evaluation summary

## Setup

- Dataset: Charades-STA test split (full: 1,334 videos, 3,720 queries). Frames extracted at 5 FPS and CLIP-encoded once with the team's `extract_candidates` + `CLIPEncoder` (openai/clip-vit-base-patch32).
- Hardware: Intel(R) Xeon(R) Processor @ 2.10GHz x2 cores, 8.2 GB RAM, no GPU; torch 2.15.0+cpu, transformers 4.57.6, faiss 1.16.0.
- Videos / queries actually used: E1 1334 / 3720; E1b 1334 videos; E3 1334 videos; E6 10 videos / 31 queries.
- Retrieval protocol (all methods): team's FAISS flat index -> top-5 -> team's `group_temporally` (gap 2 s) -> predicted moment = union of coverage spans of the group containing the top-1 hit. 95% CIs: 1,000 bootstrap resamples over queries (fixed seed).
- † MaxInfo and StreamMem are **our re-implementation for retrieval indexing, adapted from methods designed for VLM frame selection**; they are not the authors' pipelines.
- Text embeddings use a single-projection CLIP text encoder (see `docs`/report: the team's `encode_text` double-projects under transformers 4.57). Image embeddings come from the team's encoder unchanged.

## E1 - frame selection at indexing time

Reference and system-default rows (mean over videos; R@1 = recall at top-1 with IoU threshold):

| method | % kept | Hit@1 [95% CI] | Hit@5 | R@1 IoU0.3 | R@1 IoU0.5 [95% CI] | R@1 IoU0.7 | mIoU | mean pred. span (s) |
|---|---|---|---|---|---|---|---|---|
| Full 5 FPS | 100.0 | 48.6 [47.0, 50.2] | 72.9 | 9.9 | 2.4 [1.9, 2.9] | 0.5 | 9.7 | 1.3 |
| Cosine vs last kept (tau=0.95) | 29.2 | 48.2 [46.5, 49.8] | 81.6 | 20.6 | 7.8 [7.0, 8.7] | 2.4 | 15.0 | 2.5 |
| Chance (random moment) | nan | 27.9 [26.3, 29.4] | 77.5 | 33.0 | 20.1 [18.8, 21.4] | 7.0 | 21.4 | 8.0 |

(system default = cosine vs last kept, tau=0.95)

Mean ground-truth moment length: 8.0 s.

### Budget table - best operating point per method by R@1 IoU>=0.5 (plan)

| budget (% frames) | method | params | % kept | R@1 IoU0.5 [95% CI] | Hit@1 [95% CI] |
|---|---|---|---|---|---|
| 10 | Cosine vs last kept | tau=0.85 | 4.4 | 16.5 [15.3, 17.7] | 40.2 [38.6, 41.6] |
| 10 | Cosine vs previous | tau=0.92 | 5.8 | 13.4 [12.4, 14.6] | 35.6 [34.2, 37.1] |
| 10 | Hybrid † | delta=0.95;chunk_s=8;R=1;tol=0.6 | 5.9 | 19.9 [18.6, 21.1] | 43.8 [42.2, 45.5] |
| 10 | MaxInfo-chunk † | chunk_s=8;R=1;tol=1.0 | 2.8 | 29.5 [28.2, 31.1] | 40.5 [39.0, 42.0] |
| 10 | MaxInfo-fast † | R=4;tol=1.0 | 3.8 | 25.3 [23.9, 26.7] | 40.4 [38.9, 42.0] |
| 10 | MaxInfo-slow † | R=4;tol=1.0;cap_frac=0.05 | 3.7 | 26.0 [24.7, 27.5] | 43.4 [41.8, 44.8] |
| 10 | StreamMem merge † | delta=0.92 | 5.8 | 16.2 [15.0, 17.3] | 41.4 [39.9, 43.1] |
| 10 | Uniform | fraction=0.02 | 2.0 | 29.2 [27.7, 30.7] | 28.5 [27.0, 30.0] |
| 20 | Cosine vs last kept | tau=0.85 | 4.4 | 16.5 [15.3, 17.7] | 40.2 [38.6, 41.6] |
| 20 | Cosine vs previous | tau=0.92 | 5.8 | 13.4 [12.4, 14.6] | 35.6 [34.2, 37.1] |
| 20 | Hybrid † | delta=0.95;chunk_s=8;R=1;tol=0.6 | 5.9 | 19.9 [18.6, 21.1] | 43.8 [42.2, 45.5] |
| 20 | MaxInfo-chunk † | chunk_s=8;R=1;tol=1.0 | 2.8 | 29.5 [28.2, 31.1] | 40.5 [39.0, 42.0] |
| 20 | MaxInfo-fast † | R=4;tol=1.0 | 3.8 | 25.3 [23.9, 26.7] | 40.4 [38.9, 42.0] |
| 20 | MaxInfo-slow † | R=4;tol=1.0;cap_frac=0.05 | 3.7 | 26.0 [24.7, 27.5] | 43.4 [41.8, 44.8] |
| 20 | StreamMem merge † | delta=0.92 | 5.8 | 16.2 [15.0, 17.3] | 41.4 [39.9, 43.1] |
| 20 | Uniform | fraction=0.02 | 2.0 | 29.2 [27.7, 30.7] | 28.5 [27.0, 30.0] |
| 30 | Cosine vs last kept | tau=0.85 | 4.4 | 16.5 [15.3, 17.7] | 40.2 [38.6, 41.6] |
| 30 | Cosine vs previous | tau=0.92 | 5.8 | 13.4 [12.4, 14.6] | 35.6 [34.2, 37.1] |
| 30 | Hybrid † | delta=0.95;chunk_s=8;R=1;tol=0.6 | 5.9 | 19.9 [18.6, 21.1] | 43.8 [42.2, 45.5] |
| 30 | MaxInfo-chunk † | chunk_s=8;R=1;tol=1.0 | 2.8 | 29.5 [28.2, 31.1] | 40.5 [39.0, 42.0] |
| 30 | MaxInfo-fast † | R=4;tol=1.0 | 3.8 | 25.3 [23.9, 26.7] | 40.4 [38.9, 42.0] |
| 30 | MaxInfo-slow † | R=4;tol=1.0;cap_frac=0.05 | 3.7 | 26.0 [24.7, 27.5] | 43.4 [41.8, 44.8] |
| 30 | StreamMem merge † | delta=0.92 | 5.8 | 16.2 [15.0, 17.3] | 41.4 [39.9, 43.1] |
| 30 | Uniform | fraction=0.02 | 2.0 | 29.2 [27.7, 30.7] | 28.5 [27.0, 30.0] |

### Budget table - best operating point per method by Hit@1 (protocol-robust companion)

| budget (% frames) | method | params | % kept | R@1 IoU0.5 [95% CI] | Hit@1 [95% CI] |
|---|---|---|---|---|---|
| 10 | Cosine vs last kept | tau=0.85 | 4.4 | 16.5 [15.3, 17.7] | 40.2 [38.6, 41.6] |
| 10 | Cosine vs previous | tau=0.92 | 5.8 | 13.4 [12.4, 14.6] | 35.6 [34.2, 37.1] |
| 10 | Hybrid † | delta=0.95;chunk_s=4;R=1;tol=0.6 | 8.8 | 15.7 [14.5, 16.8] | 45.7 [44.0, 47.4] |
| 10 | MaxInfo-chunk † | chunk_s=4;R=1;tol=1.0 | 5.3 | 15.6 [14.5, 16.7] | 42.7 [41.0, 44.2] |
| 10 | MaxInfo-fast † | R=8;tol=1.0 | 7.9 | 16.5 [15.3, 17.7] | 43.8 [42.3, 45.4] |
| 10 | MaxInfo-slow † | R=8;tol=1.0;cap_frac=0.1 | 7.9 | 15.3 [14.2, 16.5] | 44.4 [42.7, 45.9] |
| 10 | StreamMem merge † | delta=0.92 | 5.8 | 16.2 [15.0, 17.3] | 41.4 [39.9, 43.1] |
| 10 | Uniform | fraction=0.05 | 5.0 | 12.4 [11.3, 13.5] | 38.1 [36.5, 39.5] |
| 20 | Cosine vs last kept | tau=0.92 | 14.8 | 11.5 [10.5, 12.5] | 45.5 [43.9, 47.1] |
| 20 | Cosine vs previous | tau=0.95 | 17.8 | 11.9 [10.9, 12.9] | 44.3 [42.8, 45.9] |
| 20 | Hybrid † | delta=0.95;chunk_s=4;R=2;tol=0.3 | 17.3 | 13.5 [12.5, 14.7] | 46.8 [45.3, 48.5] |
| 20 | MaxInfo-chunk † | chunk_s=2;R=1;tol=1.0 | 10.3 | 11.4 [10.5, 12.4] | 45.6 [44.1, 47.2] |
| 20 | MaxInfo-fast † | R=16;tol=1.0 | 14.4 | 10.6 [9.7, 11.6] | 45.1 [43.6, 46.9] |
| 20 | MaxInfo-slow † | R=16;tol=1.0;cap_frac=0.2 | 16.3 | 9.1 [8.2, 10.0] | 46.2 [44.7, 47.7] |
| 20 | StreamMem merge † | delta=0.95 | 17.8 | 13.5 [12.4, 14.6] | 46.7 [45.2, 48.5] |
| 20 | Uniform | fraction=0.15 | 15.0 | 12.6 [11.6, 13.7] | 43.4 [42.0, 45.0] |
| 30 | Cosine vs last kept | tau=0.95 | 29.2 | 7.8 [7.0, 8.7] | 48.2 [46.5, 49.8] |
| 30 | Cosine vs previous | tau=0.95 | 17.8 | 11.9 [10.9, 12.9] | 44.3 [42.8, 45.9] |
| 30 | Hybrid † | delta=0.95;chunk_s=4;R=2;tol=0.3 | 17.3 | 13.5 [12.5, 14.7] | 46.8 [45.3, 48.5] |
| 30 | MaxInfo-chunk † | chunk_s=4;R=2;tol=0.6 | 30.0 | 5.0 [4.3, 5.8] | 45.7 [44.1, 47.4] |
| 30 | MaxInfo-fast † | R=16;tol=1.0 | 14.4 | 10.6 [9.7, 11.6] | 45.1 [43.6, 46.9] |
| 30 | MaxInfo-slow † | R=16;tol=0.45;cap_frac=0.2 | 20.3 | 8.5 [7.6, 9.4] | 47.7 [46.2, 49.4] |
| 30 | StreamMem merge † | delta=0.95 | 17.8 | 13.5 [12.4, 14.6] | 46.7 [45.2, 48.5] |
| 30 | Uniform | fraction=0.3 | 30.0 | 7.7 [6.9, 8.6] | 46.8 [45.2, 48.5] |

Selection cost (mean ms per video, per method family, over all swept configs): 

| method | mean ms/video | max ms/video (any config) |
|---|---|---|
| Cosine vs previous | 0.13 | 0.14 |
| Cosine vs last kept | 1.47 | 1.67 |
| Full 5 FPS | 0.1 | 0.1 |
| Hybrid † | 0.96 | 1.51 |
| MaxInfo-chunk † | 4.82 | 5.86 |
| MaxInfo-fast † | 1.38 | 1.61 |
| MaxInfo-slow † | 9.21 | 10.74 |
| StreamMem merge † | 0.32 | 0.49 |
| Uniform | 0.09 | 0.12 |

## E1b - long streams (concatenated test videos)

Limitations: hard cuts between clips; generic actions repeat across clips (many plausible distractors); one index per stream, every member query searched against the whole stream.

| L (min) | streams | method | % kept | Hit@1 [95% CI] | R@1 IoU0.5 [95% CI] | search ms/query | index MB/stream | selection s/stream |
|---|---|---|---|---|---|---|---|---|
| 5 | 125 | Full 5 FPS | 100.0 | 19.0 [17.7, 20.2] | 0.4 [0.2, 0.6] | 0.166 | 3.23 | 0.001 |
| 5 | 125 | Cosine vs last kept (tau=0.95) | 29.2 | 18.3 [17.1, 19.5] | 1.5 [1.1, 1.9] | 0.049 | 0.94 | 0.013 |
| 5 | 125 | Cosine vs last kept (tau=0.9) | 10.1 | 16.3 [15.2, 17.5] | 2.4 [1.9, 2.9] | 0.037 | 0.33 | 0.013 |
| 5 | 125 | Uniform (match=cos_last_drop[tau=0.95]) | 29.2 | 18.0 [16.8, 19.2] | 1.6 [1.2, 2.0] | 0.048 | 0.94 | 0.0 |
| 5 | 125 | Uniform (match=cos_last_drop[tau=0.9]) | 10.1 | 16.6 [15.4, 17.8] | 3.0 [2.4, 3.5] | 0.037 | 0.33 | 0.0 |
| 5 | 125 | Uniform (fraction=0.1) | 10.0 | 16.4 [15.1, 17.6] | 2.9 [2.4, 3.4] | 0.036 | 0.32 | 0.0 |
| 5 | 125 | StreamMem merge † (delta=0.95) | 18.0 | 17.0 [15.9, 18.3] | 3.7 [3.1, 4.2] | 0.041 | 0.58 | 0.004 |
| 5 | 125 | MaxInfo-chunk † (chunk_s=8,R=2,tol=0.6) | 15.0 | 16.9 [15.6, 18.0] | 3.3 [2.7, 3.8] | 0.041 | 0.48 | 0.048 |
| 5 | 125 | MaxInfo-slow † (R=8,tol=0.1,cap_frac=0.1) | 10.0 | 12.8 [11.8, 13.8] | 3.8 [3.2, 4.4] | 0.039 | 0.32 | 0.074 |
| 5 | 125 | Hybrid † (delta=0.95,chunk_s=8,R=2,tol=0.6) | 8.5 | 16.1 [14.9, 17.2] | 4.7 [4.0, 5.3] | 0.036 | 0.27 | 0.014 |
| 5 | 125 | Chance (random moment) | nan | 2.5 [2.0, 3.0] | 1.1 [0.8, 1.5] | nan | nan | nan |
| 10 | 63 | Full 5 FPS | 100.0 | 13.4 [12.2, 14.5] | 0.3 [0.1, 0.4] | 0.282 | 6.34 | 0.001 |
| 10 | 63 | Cosine vs last kept (tau=0.95) | 29.2 | 12.9 [11.9, 14.0] | 1.0 [0.7, 1.3] | 0.088 | 1.85 | 0.026 |
| 10 | 63 | Cosine vs last kept (tau=0.9) | 10.1 | 11.9 [10.9, 12.8] | 1.5 [1.1, 1.9] | 0.044 | 0.64 | 0.025 |
| 10 | 63 | Uniform (match=cos_last_drop[tau=0.95]) | 29.2 | 12.5 [11.5, 13.7] | 1.0 [0.7, 1.3] | 0.088 | 1.85 | 0.0 |
| 10 | 63 | Uniform (match=cos_last_drop[tau=0.9]) | 10.1 | 11.9 [11.0, 13.0] | 1.7 [1.3, 2.2] | 0.044 | 0.64 | 0.0 |
| 10 | 63 | Uniform (fraction=0.1) | 10.0 | 10.6 [9.7, 11.6] | 1.6 [1.2, 2.0] | 0.041 | 0.63 | 0.0 |
| 10 | 63 | StreamMem merge † (delta=0.95) | 17.9 | 12.2 [11.2, 13.2] | 2.4 [1.9, 2.9] | 0.055 | 1.14 | 0.009 |
| 10 | 63 | MaxInfo-chunk † (chunk_s=8,R=2,tol=0.6) | 14.9 | 11.7 [10.7, 12.7] | 2.7 [2.2, 3.2] | 0.048 | 0.95 | 0.096 |
| 10 | 63 | MaxInfo-slow † (R=8,tol=0.1,cap_frac=0.1) | 10.0 | 7.5 [6.7, 8.4] | 1.5 [1.1, 1.9] | 0.042 | 0.63 | 0.11 |
| 10 | 63 | Hybrid † (delta=0.95,chunk_s=8,R=2,tol=0.6) | 8.5 | 11.5 [10.5, 12.5] | 3.4 [2.8, 3.9] | 0.04 | 0.54 | 0.027 |
| 10 | 63 | Chance (random moment) | nan | 1.0 [0.7, 1.4] | 0.7 [0.5, 1.0] | nan | nan | nan |
| 20 | 32 | Full 5 FPS | 100.0 | 9.0 [8.1, 10.0] | 0.1 [0.0, 0.3] | 0.509 | 12.47 | 0.003 |
| 20 | 32 | Cosine vs last kept (tau=0.95) | 29.2 | 9.1 [8.1, 10.0] | 0.5 [0.3, 0.8] | 0.186 | 3.64 | 0.053 |
| 20 | 32 | Cosine vs last kept (tau=0.9) | 10.1 | 7.5 [6.7, 8.5] | 0.7 [0.5, 1.0] | 0.051 | 1.26 | 0.05 |
| 20 | 32 | Uniform (match=cos_last_drop[tau=0.95]) | 29.2 | 8.7 [7.8, 9.6] | 0.5 [0.3, 0.8] | 0.181 | 3.64 | 0.001 |
| 20 | 32 | Uniform (match=cos_last_drop[tau=0.9]) | 10.1 | 8.2 [7.3, 9.1] | 0.8 [0.5, 1.1] | 0.054 | 1.26 | 0.0 |
| 20 | 32 | Uniform (fraction=0.1) | 10.0 | 7.2 [6.3, 8.0] | 1.0 [0.7, 1.4] | 0.053 | 1.25 | 0.0 |
| 20 | 32 | StreamMem merge † (delta=0.95) | 17.9 | 8.3 [7.4, 9.2] | 1.4 [1.1, 1.9] | 0.108 | 2.24 | 0.02 |
| 20 | 32 | MaxInfo-chunk † (chunk_s=8,R=2,tol=0.6) | 14.9 | 7.6 [6.7, 8.4] | 1.4 [1.0, 1.7] | 0.089 | 1.86 | 0.19 |
| 20 | 32 | MaxInfo-slow † (R=8,tol=0.1,cap_frac=0.1) | 10.0 | 4.6 [3.9, 5.3] | 0.9 [0.6, 1.2] | 0.055 | 1.25 | 0.198 |
| 20 | 32 | Hybrid † (delta=0.95,chunk_s=8,R=2,tol=0.6) | 8.4 | 7.2 [6.3, 8.0] | 1.9 [1.4, 2.3] | 0.051 | 1.05 | 0.054 |
| 20 | 32 | Chance (random moment) | nan | 0.5 [0.3, 0.8] | 0.4 [0.2, 0.5] | nan | nan | nan |
| 30 | 21 | Full 5 FPS | 100.0 | 6.8 [6.0, 7.6] | 0.1 [0.0, 0.2] | 0.707 | 18.64 | 0.004 |
| 30 | 21 | Cosine vs last kept (tau=0.95) | 29.2 | 6.8 [6.0, 7.6] | 0.4 [0.2, 0.7] | 0.246 | 5.44 | 0.077 |
| 30 | 21 | Cosine vs last kept (tau=0.9) | 10.1 | 6.0 [5.2, 6.7] | 0.5 [0.3, 0.7] | 0.085 | 1.88 | 0.073 |
| 30 | 21 | Uniform (match=cos_last_drop[tau=0.95]) | 29.2 | 6.5 [5.7, 7.3] | 0.2 [0.1, 0.3] | 0.242 | 5.44 | 0.001 |
| 30 | 21 | Uniform (match=cos_last_drop[tau=0.9]) | 10.1 | 5.5 [4.9, 6.3] | 0.5 [0.2, 0.7] | 0.086 | 1.88 | 0.001 |
| 30 | 21 | Uniform (fraction=0.1) | 10.0 | 5.5 [4.8, 6.3] | 0.7 [0.4, 1.0] | 0.083 | 1.86 | 0.0 |
| 30 | 21 | StreamMem merge † (delta=0.95) | 17.9 | 6.6 [5.9, 7.4] | 1.0 [0.7, 1.3] | 0.174 | 3.34 | 0.03 |
| 30 | 21 | MaxInfo-chunk † (chunk_s=8,R=2,tol=0.6) | 14.9 | 5.9 [5.1, 6.7] | 1.2 [0.8, 1.5] | 0.141 | 2.77 | 0.276 |
| 30 | 21 | MaxInfo-slow † (R=8,tol=0.1,cap_frac=0.1) | 8.3 | 3.7 [3.1, 4.3] | 0.3 [0.1, 0.5] | 0.066 | 1.54 | 0.298 |
| 30 | 21 | Hybrid † (delta=0.95,chunk_s=8,R=2,tol=0.6) | 8.4 | 5.8 [5.0, 6.5] | 1.3 [0.9, 1.7] | 0.066 | 1.57 | 0.082 |
| 30 | 21 | Chance (random moment) | nan | 0.5 [0.3, 0.8] | 0.2 [0.1, 0.3] | nan | nan | nan |

## E3 - absent-event queries

AUROC is pooled across videos (positives = a video's own queries vs negatives scored against that video's index). 'in-domain' negatives are captions of other videos with no content-word overlap with any caption of the target video - lexical, not verified, absence. OOD queries are hand-written (`experiments/e3_absent/ood_queries.txt`) and also differ linguistically from Charades captions.

| index | negatives | n pos/neg | AUROC [95% CI] | TPR @0.20 | FPR @0.20 | thr @FPR10% | TPR @FPR10% |
|---|---|---|---|---|---|---|---|
| full_5fps | indomain | 3720/4002 | 0.721 [0.711, 0.733] | 99.9 | 99.7 | 0.291 | 38.4 |
| full_5fps | ood | 3720/4002 | 0.981 [0.978, 0.983] | 99.9 | 53.0 | 0.236 | 95.3 |
| full_5fps | all | 3720/8004 | 0.851 [0.844, 0.857] | 99.9 | 76.3 | 0.281 | 52.7 |
| cos_last_drop_tau0.95 | indomain | 3720/4002 | 0.714 [0.704, 0.726] | 99.8 | 99.6 | 0.289 | 37.9 |
| cos_last_drop_tau0.95 | ood | 3720/4002 | 0.979 [0.977, 0.982] | 99.8 | 49.9 | 0.235 | 94.9 |
| cos_last_drop_tau0.95 | all | 3720/8004 | 0.847 [0.841, 0.853] | 99.8 | 74.7 | 0.279 | 51.5 |

## E6 - where the time goes (team's real pipeline)

Wall-clock on a shared 2-core CPU (a background encoding job may be running; see `results/e6_latency/meta.json` loadavg). First video and first query are warm-ups and excluded from means. COMPLEX-routed queries stop at the VLM stage (no OpenRouter key), so no end-to-end slow-path time is reported.

| phase | stage | n | mean_s | median_s | std_s | min_s | max_s |
|---|---|---|---|---|---|---|---|
| preprocess (per video) | frame_extraction | 10 | 0.323 | 0.287 | 0.134 | 0.147 | 0.581 |
| preprocess (per video) | clip_encoding | 10 | 5.27 | 5.457 | 1.219 | 2.412 | 7.067 |
| preprocess (per video) | frame_pruning | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| preprocess (per video) | visual_memory | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.001 |
| preprocess (per video) | faiss_index | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| preprocess (per video) | TOTAL (wall) | 10 | 5.61 | 5.69 | 1.296 | 2.575 | 7.558 |
| query | query_classification | 31 | 0.422 | 0.42 | 0.018 | 0.393 | 0.464 |
| query | clip_text_embedding | 31 | 0.016 | 0.016 | 0.002 | 0.014 | 0.021 |
| query | faiss_search | 31 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| query | similarity_filtering | 31 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| query | temporal_grouping | 31 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| query | diversity_selection | 31 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| query | evidence_frame_extraction | 31 | 0.307 | 0.284 | 0.286 | 0.0 | 1.012 |
| query | answer_ready | 11 | 0.435 | 0.432 | 0.024 | 0.407 | 0.483 |
| query | TOTAL (wall) | 31 | 0.754 | 0.737 | 0.289 | 0.414 | 1.489 |
| model load (cold, once) | clip_load | 1 | 3.081 | 3.081 | nan | 3.081 | 3.081 |
| model load (cold, once) | classifier_load | 1 | 0.419 | 0.419 | nan | 0.419 | 0.419 |

Routing decisions:

| route | n | pct | failed_at_vlm |
|---|---|---|---|
| COMPLEX | 20 | 64.5 | 20 |
| SIMPLE | 11 | 35.5 | 0 |

Memory / resources:

| item | value |
|---|---|
| rss_at_start_mb | 521.8 |
| clip_load_s | 3.1 |
| classifier_load_s | 0.4 |
| rss_after_load_mb | 979.4 |
| rss_end_mb | 2965.4 |
| ru_maxrss_mb | 2895.8 |
| video_peak_rss_mb_max | 2965.4 |
| query_peak_rss_mb_max | 2965.5 |

## Observations (derived only from the numbers above)

1. Full 5 FPS indexing reaches Hit@1 = 48.6% (95% CI 47.0-50.2) against chance 27.9% - the CI lies above chance.
2. The current system default (cosine vs last kept, tau=0.95) keeps 29.2% of frames; its Hit@1 48.2% vs 48.6% for full 5 FPS - within overlapping CIs (not a significant difference).
3. Among operating points keeping <=20% of frames, the highest Hit@1 is Hybrid † (delta=0.95, chunk_s=4, R=2, tol=0.3) at 17.3% kept: 46.8% [45.3, 48.5]; best uniform within the same budget: 43.4% [42.0, 45.0] - non-overlapping CIs.
4. R@1 IoU>=0.5 is strongly span-dependent under this protocol: full 5 FPS scores 2.4% with a mean predicted span of 1.3 s (ground truth 8.0 s), whereas the best sparse point (MaxInfo-chunk † (chunk_s=8, R=1, tol=1.0), 2.8% kept) scores 29.5% with span 7.4 s, and the random-interval chance reference scores 20.1%. Sparser indexes inherit longer coverage spans, so R@1 gains need not reflect better matching; Hit@k is the cleaner comparison.
5. At tau=0.95, comparing against the last kept frame keeps 29.2% of frames vs 17.8% when comparing consecutive frames (Hit@1 48.2% vs 44.3%).
6. MaxInfo (rect-maxvol) selects roughly K ~ R/tol^2 rows independent of clip length, so on ~30 s clips small tol saturates: 11 of 20 MaxInfo-fast configs select the whole 1 FPS pool (20.3% of 5 FPS frames). Mean selection cost over all MaxInfo configs 6.6 ms/video vs 1.5 ms/video for the team's cosine pruning.
7. Full 5 FPS Hit@1 falls from 19.0% at 5 min to 6.8% at 30 min (3601 queries, 21 stream(s)); search latency grows 0.17 -> 0.71 ms/query and the index 3.2 -> 18.6 MB.
8. At 30 min the system default (tau=0.95) keeps 29.2% of frames: Hit@1 6.8% vs 6.8% for full 5 FPS (overlapping CIs). Latency 0.25 ms/query.
