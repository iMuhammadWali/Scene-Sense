"""Small helpers shared by experiment runners."""
from __future__ import annotations
import argparse, json, subprocess, time, datetime
from pathlib import Path
from . import config as C

def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N available videos (quick run)")
    ap.add_argument("--force", action="store_true", help="ignore existing shards/DONE and recompute")
    return ap

def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(C.REPO), "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"

def write_meta(out: Path, **kw):
    meta = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), git=git_rev(),
                hardware=C.hardware_info(), **kw)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

def mark_done(out: Path):
    (out / "DONE").write_text(datetime.datetime.now().isoformat(timespec="seconds") + "\n")

def is_done(out: Path, force: bool = False) -> bool:
    return (out / "DONE").exists() and not force

def progress(i, n, t0, what="videos"):
    el = time.time() - t0
    print(f"  [{i}/{n} {what}] {el:.0f}s elapsed, ETA {el / max(i, 1) * (n - i):.0f}s", flush=True)
