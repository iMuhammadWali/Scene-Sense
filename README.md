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

## Running
```
scripts/run_all.sh                  # full run (needs all 1,334 embeddings; add --allow-partial otherwise)
scripts/run_all.sh --allow-partial --out /tmp/subset --limit 40 --e6-videos 2 --e6-queries 2   # quick validation
python -m pytest tests -q           # unit tests
```
Each step writes `results/<id>/DONE` and resumes from shards on rerun. Set `SSEVAL_TEAM_TEXT=1` to use the team's `encode_text`
unchanged (double text projection under transformers 4.57 - see `src/sseval/data.py`).
