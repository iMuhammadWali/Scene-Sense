"""Session persistence tests."""

from __future__ import annotations

import numpy as np

from app.config import Settings
from app.storage.session import SessionStore


def test_session_persistence(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
    )
    settings.ensure_directories()
    store = SessionStore(settings)
    sid = store.create_session("demo.mp4", 1234)
    assert store.exists(sid)

    # Fake video file
    store.video_path(sid).write_bytes(b"\x00\x00fake")

    frames = [{"frame_id": 1, "timestamp": 0.2, "source_frame_index": 5}]
    emb = np.random.randn(1, 8).astype(np.float32)
    store.save_visual_memory(sid, frames, emb)
    store.save_faiss_mapping(sid, [{"faiss_index": 0, "frame_id": 1, "timestamp": 0.2}])
    # Create empty faiss index file marker for readiness check (not a real index)
    # is_video_ready requires faiss.index exists — write placeholder
    store.faiss_index_path(sid).write_bytes(b"x")
    store.update_metadata(sid, video_ready=True, status="ready")

    frames2, emb2 = store.load_visual_memory(sid)
    assert frames2[0]["frame_id"] == 1
    assert emb2.shape == (1, 8)
    assert store.is_video_ready(sid)
