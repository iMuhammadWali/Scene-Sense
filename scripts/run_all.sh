#!/usr/bin/env bash
# Idempotent, resumable pipeline for the Tier-1 experiments. Each step is skipped if its DONE marker exists.
# Usage: scripts/run_all.sh [--allow-partial] [--out DIR] [--limit N] [--e6-videos N] [--e6-queries Q] [--force]
#   --allow-partial  continue even if fewer than 1,334 video embeddings exist
#   --out DIR        write results to DIR instead of ./results (e.g. for subset validation)
#   --limit N        use only the first N videos in E1/E1b/E3 (quick run)
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
PY=${PY:-python}
EMB=${SSEVAL_EMB:-$REPO/data/emb}
# data/model dirs default to REPO/data/* inside sseval.config; override with SCENESENSE_DATA_DIR / SCENESENSE_MODELS_DIR / SSEVAL_*
export SCENESENSE_DEVICE=${SCENESENSE_DEVICE:-cpu}
ALLOW=0; LIMIT=""; E6V=""; E6Q=""; FORCE=""
while [ $# -gt 0 ]; do case "$1" in
  --allow-partial) ALLOW=1;; --out) export SSEVAL_RESULTS_DIR="$(realpath -m "$2")"; shift;;
  --limit) LIMIT="--limit $2"; shift;; --e6-videos) E6V="--limit $2"; shift;; --e6-queries) E6Q="--max-queries $2"; shift;;
  --force) FORCE="--force";; *) echo "unknown arg $1"; exit 2;; esac; shift; done
RES="${SSEVAL_RESULTS_DIR:-$REPO/results}"; mkdir -p "$RES" logs
log() { echo "[$(date +%H:%M:%S)] $*"; }

# (1) embeddings check only (encoding itself is run separately: python scripts/encode_all.py)
N=$(ls "$EMB"/*.npz 2>/dev/null | grep -v '\.tmp\.npz$' | wc -l)
log "step 1: $N / 1334 video embeddings present"
if [ "$N" -lt 1334 ] && [ $ALLOW -eq 0 ]; then log "ERROR: embeddings incomplete (use --allow-partial to run anyway)"; exit 1; fi

step() {  # name, done-marker-dir, command...
  local name=$1 dir=$2; shift 2
  if [ -f "$RES/$dir/DONE" ] && [ -z "$FORCE" ]; then log "$name: DONE exists, skipping"; return 0; fi
  log "$name: running"; local t0=$(date +%s)
  "$@" > "logs/$name.log" 2>&1; local rc=$?
  log "$name: exit $rc after $(( $(date +%s) - t0 ))s (log: logs/$name.log)"
  return $rc
}
step e1  e1_pruning   $PY -W ignore experiments/e1_pruning/run.py $LIMIT $FORCE
step e1b e1b_long     $PY -W ignore experiments/e1b_long/run.py $LIMIT $FORCE
step e3  e3_absent    $PY -W ignore experiments/e3_absent/run.py $LIMIT $FORCE
step e6  e6_latency   $PY -W ignore experiments/e6_latency/run.py $E6V $E6Q $FORCE
step figures  figures $PY -W ignore scripts/make_figures.py
step summary  summary $PY -W ignore scripts/make_summary.py
log "all steps finished; results in $RES (read $RES/SUMMARY.md)"
