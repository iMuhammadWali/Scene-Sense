"""All paths and environment setup in one place.

Importing this module (a) sets the env vars the team's code reads, (b) puts the team's backend on sys.path,
(c) silences the team's per-call INFO logging and pins FAISS to one thread (stable latency, leaves CPU for other jobs).
The results directory can be redirected with SSEVAL_RESULTS_DIR (used by `run_all.sh --out`).
"""
from __future__ import annotations
import os, sys, logging, warnings
from pathlib import Path

os.environ.setdefault("SCENESENSE_DATA_DIR", "/home/claude/work/data")
os.environ.setdefault("SCENESENSE_MODELS_DIR", "/home/claude/work/models")
os.environ.setdefault("SCENESENSE_DEVICE", "cpu")
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
BACKEND = Path(os.environ.get("SSEVAL_BACKEND", "/home/claude/fyp/backend"))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

ANNOTATIONS = Path(os.environ.get("SSEVAL_ANN", "/home/claude/datasets/charades_sta/test.parquet"))
VIDEOS_DIR = Path(os.environ.get("SSEVAL_VIDEOS", "/home/claude/datasets/charades_sta/videos"))
EMB_DIR = Path(os.environ.get("SSEVAL_EMB", "/home/claude/work/emb"))
CACHE_DIR = Path(os.environ.get("SSEVAL_CACHE", "/home/claude/experiments/cache"))
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

def hardware_info() -> dict:
    import platform, multiprocessing
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": multiprocessing.cpu_count(), "loadavg": os.getloadavg(), "device": os.environ.get("SCENESENSE_DEVICE")}
    try:
        import torch, transformers, faiss
        info.update(torch=torch.__version__, transformers=transformers.__version__, faiss=faiss.__version__)
    except Exception:
        pass
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip(); break
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                info["mem_gb"] = round(int(line.split()[1]) / 1e6, 1); break
    except Exception:
        pass
    return info
