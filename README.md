# Scene Sense — Research Experiments (FYP-1 Deliverable 3)

Evaluation of the Scene Sense indexing/retrieval pipeline on Charades-STA (test split).
The team's application code is vendored in `app/` (with the text-encoder double-projection fix); the harness only *imports* its modules
(`app.pipeline.*`) so every number here measures the real system.

## Layout
- `app/` — the team's application (backend, frontend, scripts); see `app/README.md`
- `src/sseval/` — shared library: data loading, metrics, frame-selection methods (`pruning/`)
- `scripts/` — one-time data prep (video download, CLIP encoding of frames and queries), `run_all.sh`
- `experiments/<id>/run.py` — one runner per experiment; writes to `results/<id>/`
- `results/` — CSV/JSON outputs, `figures/`, and `SUMMARY.md` (the file to read first)
- `logs/` — run logs
- `docs/experiment_plan.md` — what each experiment tests and how it is measured
- `legacy_draft/` — first-draft harness (superseded)

Large data (videos, embeddings, model weights) are NOT in git. By default everything lives under `data/` in the repo
(`data/charades_sta`, `data/emb`, `data/cache`, `data/models`); override with `SSEVAL_ANN`, `SSEVAL_VIDEOS`, `SSEVAL_EMB`, `SSEVAL_CACHE`,
`SSEVAL_BACKEND`, `SCENESENSE_DATA_DIR`, `SCENESENSE_MODELS_DIR`, `HF_HOME` (see `src/sseval/config.py`).

## Setup
```
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                  # transformers MUST stay 4.57.x
# get Charades-STA (docs/DATASET.md), then:
python scripts/sanity_check.py                   # encodes 20 videos if needed, checks cosine / Hit@1
```
The `.sh` scripts need bash (Git Bash / WSL on Windows); the Python entry points work everywhere.

## Running
```
scripts/run_all.sh                  # full run (needs all 1,334 embeddings; add --allow-partial otherwise)
scripts/run_all.sh --allow-partial --out /tmp/subset --limit 40 --e6-videos 2 --e6-queries 2   # quick validation
python -m pytest tests -q           # unit tests
```
Each step writes `results/<id>/DONE` and resumes from shards on rerun. Set `SSEVAL_TEAM_TEXT=1` to use the team's `encode_text`
unchanged (double text projection under transformers 4.57 - see `src/sseval/data.py`).

## Dataset

Charades-STA (test split). Download links and a 2.1 GB selective download script: [docs/DATASET.md](docs/DATASET.md).
