"""All paths and environment setup in one place.

Importing this module (a) sets the env vars the team's code reads, (b) puts the team's backend on sys.path,
(c) silences the team's per-call INFO logging and pins FAISS to one thread (stable latency, leaves CPU for other jobs).
The results directory can be redirected with SSEVAL_RESULTS_DIR (used by `run_all.sh --out`).
"""
from __future__ import annotations
import os, sys, logging, warnings
from pathlib import Path

def utf8_console():
    """Make stdout/stderr UTF-8 (Windows consoles default to cp1252 and crash on characters like >=, +/-, em dash)."""
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

def check_versions(warn=True) -> list[str]:
    """Cheap (no heavy imports) check of the pinned stack; prints a loud warning to stderr once. Returns the problem list."""
    from importlib import metadata
    want = {"transformers": "4.57", "numpy": "2", "faiss-cpu": "1"}     # prefix to match (see requirements.txt)
    bad = []
    for pkg, pre in want.items():
        try:
            v = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            bad.append(f"{pkg} not installed (need {pre}.x)"); continue
        if not (v == pre or v.startswith(pre + ".")):
            bad.append(f"{pkg} {v} != {pre}.x")
    if bad and warn:
        print("\n" + "!" * 78 + "\nWARNING: dependency versions differ from requirements.txt - results may NOT match the paper:\n  "
              + "\n  ".join(bad) + "\n  run: python scripts/check_env.py\n" + "!" * 78 + "\n", file=sys.stderr, flush=True)
    return bad

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"      # git-ignored; every default below lives under it unless the env var says otherwise
os.environ.setdefault("SCENESENSE_DATA_DIR", str(DATA_ROOT / "app_data"))
os.environ.setdefault("SCENESENSE_MODELS_DIR", str(DATA_ROOT / "models"))
os.environ.setdefault("SCENESENSE_DEVICE", "cpu")
warnings.filterwarnings("ignore")
utf8_console()
if not os.environ.get("SSEVAL_SKIP_VERSION_CHECK"):
    check_versions()

BACKEND = Path(os.environ.get("SSEVAL_BACKEND", str(REPO / "app" / "backend")))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

ANNOTATIONS = Path(os.environ.get("SSEVAL_ANN", str(DATA_ROOT / "charades_sta" / "test.parquet")))
VIDEOS_DIR = Path(os.environ.get("SSEVAL_VIDEOS", str(DATA_ROOT / "charades_sta" / "videos")))
EMB_DIR = Path(os.environ.get("SSEVAL_EMB", str(DATA_ROOT / "emb")))
CACHE_DIR = Path(os.environ.get("SSEVAL_CACHE", str(DATA_ROOT / "cache")))
_TEAM_TEXT = os.environ.get("SSEVAL_TEAM_TEXT") == "1"
# caches written with the single-projection text encoder (see data._encode_with_team); the legacy query_emb.npz was made with the
# team's double-projected encode_text and is NOT reused.
QUERY_EMB_CACHE = CACHE_DIR / ("query_emb_teamtext.npz" if _TEAM_TEXT else "query_emb_fixed.npz")
EXTRA_TEXT_CACHE = CACHE_DIR / ("extra_text_emb_teamtext.npz" if _TEAM_TEXT else "extra_text_emb_fixed.npz")
RESULTS_DIR = Path(os.environ.get("SSEVAL_RESULTS_DIR", str(REPO / "results")))
LOGS_DIR = REPO / "logs"

N_VIDEOS_FULL = 1334      # Charades-STA test split
N_QUERIES_FULL = 3720
EMB_DIM = 512
FPS = 5.0
FRAME_PERIOD = 1.0 / FPS
TOP_K = 5
GROUP_GAP_S = 2.0
RETRIEVAL_THRESHOLD = 0.20   # team's settings.retrieval_threshold
SEED = 0

def results_dir(exp: str) -> Path:
    p = RESULTS_DIR / exp
    p.mkdir(parents=True, exist_ok=True)
    return p

def setup_team():
    """Quiet the team's logger and pin FAISS threads (idempotent)."""
    import faiss
    faiss.omp_set_num_threads(1)
    logging.disable(logging.INFO)

def peak_rss_gb() -> float:
    """Peak resident memory of this process in GB (Unix: resource; Windows: psutil peak working set)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # KB on Linux
    except ImportError:
        import psutil
        return psutil.Process().memory_info().peak_wset / 1e9

def hardware_info() -> dict:
    import platform, multiprocessing
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": multiprocessing.cpu_count(), "device": os.environ.get("SCENESENSE_DEVICE")}
    if hasattr(os, "getloadavg"):          # Unix only
        try:
            info["loadavg"] = os.getloadavg()
        except OSError:
            pass
    try:
        import torch, transformers, faiss
        info.update(torch=torch.__version__, transformers=transformers.__version__, faiss=faiss.__version__)
    except Exception:
        pass
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip(); break
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemTotal"):
                info["mem_gb"] = round(int(line.split()[1]) / 1e6, 1); break
    except Exception:
        pass
    return info
