"""Compare installed package versions with requirements.txt. Exit 1 on ERROR-level mismatch.
ERROR: transformers, torch, faiss-cpu, numpy, opencv-python-headless. WARN: everything else. Usage: python scripts/check_env.py"""
import platform, re, sys
from importlib import metadata
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
CRITICAL = {"transformers", "torch", "faiss-cpu", "numpy", "opencv-python-headless"}
req = Path(__file__).resolve().parents[1] / "requirements.txt"
head = req.read_text(encoding="utf-8")
m = re.search(r"Python:\s*(\d+)\.(\d+)", head)
print(f"python {platform.python_version()} | {platform.platform()} | {platform.machine()}")
rows, errors, warns = [], 0, 0
if m and (sys.version_info.major, sys.version_info.minor) != (int(m[1]), int(m[2])):
    print(f"ERROR: Python {m[1]}.{m[2]} required, found {sys.version_info.major}.{sys.version_info.minor}"); errors += 1
for line in head.splitlines():
    line = line.split("#")[0].strip()
    if "==" not in line:
        continue
    name, want = [x.strip() for x in line.split("==")]
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        have = "MISSING"
    ok = have.split("+")[0] == want.split("+")[0]
    if not ok:
        lvl = "ERROR" if name.lower() in CRITICAL else "WARN"
        errors += lvl == "ERROR"; warns += lvl == "WARN"
        rows.append((lvl, name, want, have))
if rows:
    w = max(len(r[1]) for r in rows)
    print(f"\n{'level':6} {'package':{w}} {'required':14} installed")
    for r in rows:
        print(f"{r[0]:6} {r[1]:{w}} {r[2]:14} {r[3]}")
print(f"\n{errors} error(s), {warns} warning(s)")
if not rows and not errors:
    print("OK: environment matches requirements.txt")
if any(r[1] == "transformers" for r in rows):
    print("transformers MUST be 4.57.x: pip install transformers==4.57.6")
sys.exit(1 if errors else 0)
