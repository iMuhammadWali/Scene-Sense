#!/usr/bin/env python3
"""End-to-end verification script for SceneSense (CLI, no UI)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings
from app.pipeline.query_pipeline import QueryPipeline
from app.pipeline.video_processor import VideoProcessor
from app.services.upload import UploadService
from app.storage.session import SessionStore
from app.utils.hardware import detect_hardware


def main() -> int:
    settings = get_settings()
    video = ROOT / "data" / "videos" / "sample_demo.mp4"
    if not video.exists():
        print(f"Missing sample video: {video}")
        return 1

    print("Hardware:", json.dumps(detect_hardware(), indent=2))
    print("Uploading", video)
    with video.open("rb") as f:
        upload = UploadService().save_upload(video.name, f, size_hint=video.stat().st_size)
    sid = upload["session_id"]
    print("session_id=", sid)

    print("Processing video (one-time)...")
    t0 = time.perf_counter()
    meta = VideoProcessor().process(sid)
    print(f"VIDEO READY in {time.perf_counter()-t0:.2f}s")
    print("retained=", meta.get("retained_frame_count"), "candidates=", meta.get("candidate_frame_count"))

    store = SessionStore(settings)
    assert store.is_video_ready(sid)

    questions = [
        "Is there a black object?",
        "What happened after the scene changed?",
    ]
    for q in questions:
        print("\n=== QUESTION:", q)
        try:
            result = QueryPipeline().ask(sid, q)
            print("type=", result["query_type"], "path=", result["path"], "vlm=", result["vlm"])
            print("answer=", result["answer"])
            print("evidence=", [e["timestamp"] for e in result["evidence"]])
            print("timings=", result["timings"])
        except Exception as exc:
            print("QUESTION FAILED:", exc)
            return 1

    # Confirm second question did not reprocess
    stages = store.read_json(sid, "metadata.json").get("stages", [])
    preprocess = [s for s in stages if s.get("stage") == "frame_extraction"]
    assert len(preprocess) == 1, "Video was reprocessed — FAIL"
    print("\nVerified: second question did NOT reprocess video.")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
