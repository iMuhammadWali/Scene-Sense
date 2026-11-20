#!/usr/bin/env bash
# Create ./.venv with the pinned environment. Usage: scripts/setup_linux.sh   (env PYTHON=python3.13 to choose the interpreter)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
PYTHON=${PYTHON:-python3}
REQ=$(sed -n 's/^# Python: *\([0-9]*\.[0-9]*\).*/\1/p' requirements.txt | head -1)
HAVE=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[ "$HAVE" = "$REQ" ] || { echo "ERROR: need Python $REQ, '$PYTHON' is $HAVE. Install it or set PYTHON=pythonX.Y"; exit 1; }
"$PYTHON" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
# CPU torch from the PyTorch index (small wheel, no CUDA), everything else pinned by requirements.txt
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python scripts/check_env.py
echo "Done. Activate with:  . .venv/bin/activate"
