#!/usr/bin/env bash
# Hands-off chain: finish CLIP encoding (resumable), then run all Tier-1 experiments.
# Safe to re-run any number of times; every stage resumes where it stopped.
# Arg 1 = encoding time budget in seconds (default 6600). Env: PY (default python), SSEVAL_EMB (default REPO/data/emb).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-python}
EMB=${SSEVAL_EMB:-$REPO/data/emb}
count() { ls "$EMB"/*.npz 2>/dev/null | grep -v '\.tmp\.npz$' | wc -l; }
echo "[$(date +%H:%M:%S)] start, embeddings: $(count)/1334"
if [ "$(count)" -lt 1334 ]; then
  "$PY" -W ignore "$REPO/scripts/encode_all.py" "${1:-6600}" 2>&1 | grep -v -i "slow image processor"
fi
echo "[$(date +%H:%M:%S)] embeddings: $(count)/1334"
[ "$(count)" -lt 1334 ] && { echo "ENCODING INCOMPLETE - rerun this script"; exit 1; }
cd "$REPO" && PY="$PY" scripts/run_all.sh; rc=$?
echo "[$(date +%H:%M:%S)] OVERNIGHT END run_all rc=$rc"
