#!/usr/bin/env bash
# Hands-off chain: finish CLIP encoding (resumable), then run all Tier-1 experiments, then commit results.
# Safe to re-run any number of times; every stage resumes where it stopped.
# Arg 1 = encoding time budget in seconds (default 6600).
set -uo pipefail
REPO=/home/claude/scenesense-research
export SCENESENSE_DATA_DIR=/home/claude/work/data SCENESENSE_MODELS_DIR=/home/claude/work/models SCENESENSE_DEVICE=cpu
count() { ls /home/claude/work/emb/*.npz 2>/dev/null | grep -v '\.tmp\.npz$' | wc -l; }
echo "[$(date +%H:%M:%S)] start, embeddings: $(count)/1334"
if [ "$(count)" -lt 1334 ]; then
  (cd /home/claude/experiments && /home/claude/venv/bin/python -W ignore encode_all.py "${1:-6600}" 2>&1 | grep -v -i "slow image processor")
fi
echo "[$(date +%H:%M:%S)] embeddings: $(count)/1334"
[ "$(count)" -lt 1334 ] && { echo "ENCODING INCOMPLETE - rerun this script"; exit 1; }
cd "$REPO" && scripts/run_all.sh; rc=$?
git add -A results scripts && git commit -q -m "Add Tier-1 experiment results" && echo "committed"
echo "[$(date +%H:%M:%S)] OVERNIGHT END run_all rc=$rc"
