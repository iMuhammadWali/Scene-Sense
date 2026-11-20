"""Frame extraction at a fixed FPS without permanently storing images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Optional

import cv2
import numpy as np

from app.utils.logging_setup import get_logger
from app.utils.timing import timed

logger = get_logger("scenesense.frame_extraction")


@dataclass
class CandidateFrame:
    frame_id: int
    timestamp: float
    source_frame_index: int
    image: Optional[np.ndarray] = None


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    return cap


def read_video_metadata(path: str) -> dict[str, Any]:
    cap = open_video(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        if duration <= 0:
            # Fallback probe by seeking to end
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            duration = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        return {
            "path": path,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": round(duration, 4),
        }
    finally:
        cap.release()


def iter_frames_at_fps(
    path: str,
    target_fps: float = 5.0,
    keep_images: bool = True,
) -> Generator[CandidateFrame, None, None]:
    """Yield candidate frames sampled at approximately target_fps."""
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    cap = open_video(path)
    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if native_fps <= 0:
            native_fps = 25.0
        interval = max(native_fps / target_fps, 1.0)
        next_take = 0.0
        source_idx = 0
        frame_id = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if source_idx + 0.5 >= next_take:
                timestamp = source_idx / native_fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if keep_images else None
                yield CandidateFrame(
                    frame_id=frame_id,
                    timestamp=round(timestamp, 4),
                    source_frame_index=source_idx,
                    image=rgb,
                )
                frame_id += 1
                next_take += interval
            source_idx += 1
    finally:
        cap.release()


def extract_candidates(
    path: str, target_fps: float = 5.0
) -> tuple[list[CandidateFrame], dict[str, Any]]:
    """Extract all candidate frames into memory (images kept for CLIP encoding)."""
    meta = read_video_metadata(path)
    with timed() as t:
        frames = list(iter_frames_at_fps(path, target_fps=target_fps, keep_images=True))
    info = {
        "stage": "frame_extraction",
        "status": "completed",
        "input_count": 1,
        "output_count": len(frames),
        "video_duration_seconds": meta["duration_seconds"],
        "original_fps": meta["fps"],
        "extracted_fps": target_fps,
        "candidate_frame_count": len(frames),
        "width": meta["width"],
        "height": meta["height"],
        "processing_time_seconds": t[0],
    }
    logger.info(
        "[FRAME_EXTRACTION] fps=%s candidate_frames=%s duration=%.2fs",
        target_fps,
        len(frames),
        t[0],
    )
    return frames, info


def _max_seek_timestamp(cap: cv2.VideoCapture) -> float:
    """Last timestamp that OpenCV can usually decode (not past EOF)."""
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps > 0 and frame_count > 0:
        return max(0.0, (frame_count - 1) / fps)
    return 0.0


def extract_frame_at_timestamp(path: str, timestamp: float) -> np.ndarray:
    """Extract a single RGB frame from the original MP4 on demand.

    Timestamps past EOF (e.g. Slow-path ±5s windows on short clips) are
    clamped to the last readable frame instead of failing.
    """
    cap = open_video(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        max_t = _max_seek_timestamp(cap)
        requested = float(timestamp)
        t = max(0.0, requested)
        if max_t > 0:
            t = min(t, max_t)

        # Seek in milliseconds
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not ok or frame is None:
            idx = int(t * fps)
            if frame_count > 0:
                idx = min(max(idx, 0), frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
        if (not ok or frame is None) and frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Failed to extract frame at t={requested}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def extract_frames_in_range(
    path: str,
    start: float,
    end: float,
    sampling_fps: float = 1.0,
) -> list[tuple[float, np.ndarray]]:
    """Extract frames in [start, end] at sampling_fps from original MP4.

    Range is clamped to the video duration so Slow-path context windows
    never request timestamps past EOF.
    """
    if end < start:
        start, end = end, start
    if sampling_fps <= 0:
        sampling_fps = 1.0

    meta = read_video_metadata(path)
    duration = float(meta.get("duration_seconds") or 0.0)
    fps = float(meta.get("fps") or 0.0)
    frame_count = int(meta.get("frame_count") or 0)
    if fps > 0 and frame_count > 0:
        max_t = max(0.0, (frame_count - 1) / fps)
    else:
        max_t = max(0.0, duration - 1e-3) if duration > 0 else end

    start = max(0.0, float(start))
    end = min(float(end), max_t) if max_t > 0 else float(end)
    if end < start:
        return []

    step = 1.0 / sampling_fps
    out: list[tuple[float, np.ndarray]] = []
    t = start
    while t <= end + 1e-9:
        img = extract_frame_at_timestamp(path, t)
        out.append((round(t, 4), img))
        t += step
    return out
