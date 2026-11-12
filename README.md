# Scene Sense — Research Experiments (FYP-1 Deliverable 3)

Evaluation of the Scene Sense indexing/retrieval pipeline on Charades-STA (test split).
The team's application code lives in `fyp-new-main`; this repo only *imports* its modules
(`app.pipeline.*`) so every number here measures the real system.

## Layout
- `src/sseval/` — shared library: data loading, metrics, frame-selection methods (`pruning/`)
- `scripts/` — one-time data prep (video download, CLIP encoding of frames and queries), `run_all.sh`
- `experiments/<id>/run.py` — one runner per experiment; writes to `results/<id>/`
- `results/` — CSV/JSON outputs, `figures/`, and `SUMMARY.md` (the file to read first)
- `logs/` — run logs
- `docs/experiment_plan.md` — what each experiment tests and how it is measured
- `legacy_draft/` — first-draft harness (superseded)

Large data (videos, embeddings, model weights) are NOT in git; paths are set in `src/sseval/config.py`.
