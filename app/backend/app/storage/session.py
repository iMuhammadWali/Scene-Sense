"""Video session persistence for SceneSense."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.config import Settings, get_settings
from app.utils.logging_setup import get_logger

logger = get_logger("scenesense.session")

ALLOWED_EXTENSIONS = {".mp4", ".mpeg4", ".m4v"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Filesystem-backed video session manager."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()

    def session_dir(self, session_id: str) -> Path:
        safe = self._safe_id(session_id)
        return self.settings.sessions_dir / safe

    def _safe_id(self, session_id: str) -> str:
        sid = (session_id or "").strip()
        if not sid or ".." in sid or "/" in sid or "\\" in sid:
            raise ValueError("Invalid session_id")
        # Restrict to hex/uuid-like chars
        allowed = set("abcdefABCDEF0123456789-_")
        if not all(c in allowed for c in sid):
            raise ValueError("Invalid session_id characters")
        return sid

    def create_session(self, original_filename: str, size_bytes: int) -> str:
        session_id = uuid.uuid4().hex
        path = self.session_dir(session_id)
        path.mkdir(parents=True, exist_ok=False)
        (path / "tmp").mkdir(exist_ok=True)
        metadata = {
            "session_id": session_id,
            "original_filename": Path(original_filename).name,
            "size_bytes": size_bytes,
            "created_at": utc_now(),
            "status": "uploaded",
            "stage": "upload",
            "progress": 0.05,
            "video_ready": False,
            "stages": [],
            "error": None,
            "question_history": [],
        }
        self.write_json(session_id, "metadata.json", metadata)
        self.write_json(session_id, "processing_log.json", {"stages": [], "events": []})
        logger.info("[SESSION_CREATE] session=%s file=%s", session_id, original_filename)
        return session_id

    def exists(self, session_id: str) -> bool:
        try:
            return self.session_dir(session_id).exists()
        except ValueError:
            return False

    def video_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "original_video.mp4"

    def path(self, session_id: str, name: str) -> Path:
        # Prevent path traversal via name
        clean = Path(name).name
        return self.session_dir(session_id) / clean

    def write_json(self, session_id: str, name: str, data: Any) -> None:
        target = self.path(session_id, name)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def read_json(self, session_id: str, name: str) -> Any:
        target = self.path(session_id, name)
        if not target.exists():
            raise FileNotFoundError(f"Missing {name} for session {session_id}")
        return json.loads(target.read_text(encoding="utf-8"))

    def update_metadata(self, session_id: str, **fields: Any) -> dict[str, Any]:
        meta = self.read_json(session_id, "metadata.json")
        meta.update(fields)
        meta["updated_at"] = utc_now()
        self.write_json(session_id, "metadata.json", meta)
        return meta

    def append_stage_log(self, session_id: str, stage_info: dict[str, Any]) -> None:
        log = self.read_json(session_id, "processing_log.json")
        log.setdefault("stages", []).append(stage_info)
        log.setdefault("events", []).append(
            {"ts": utc_now(), "type": "stage", "data": stage_info}
        )
        self.write_json(session_id, "processing_log.json", log)
        meta = self.read_json(session_id, "metadata.json")
        stages = meta.get("stages") or []
        # Replace stage entry if same stage name exists
        replaced = False
        for i, s in enumerate(stages):
            if s.get("stage") == stage_info.get("stage"):
                stages[i] = stage_info
                replaced = True
                break
        if not replaced:
            stages.append(stage_info)
        meta["stages"] = stages
        self.write_json(session_id, "metadata.json", meta)

    def save_visual_memory(
        self,
        session_id: str,
        frames: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """Persist retained frame metadata + embeddings (not images)."""
        emb = np.asarray(embeddings, dtype=np.float32)
        np.save(self.path(session_id, "embeddings.npy"), emb)
        memory = []
        for i, frame in enumerate(frames):
            memory.append(
                {
                    "frame_id": int(frame["frame_id"]),
                    "timestamp": float(frame["timestamp"]),
                    "source_frame_index": frame.get("source_frame_index"),
                    "embedding_index": i,
                }
            )
        self.write_json(session_id, "visual_memory.json", {"frames": memory})

    def load_visual_memory(
        self, session_id: str
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        memory = self.read_json(session_id, "visual_memory.json")
        emb_path = self.path(session_id, "embeddings.npy")
        if not emb_path.exists():
            raise FileNotFoundError("embeddings.npy missing")
        embeddings = np.load(emb_path)
        return memory["frames"], embeddings

    def save_faiss_mapping(self, session_id: str, mapping: list[dict[str, Any]]) -> None:
        self.write_json(session_id, "faiss_mapping.json", mapping)

    def load_faiss_mapping(self, session_id: str) -> list[dict[str, Any]]:
        return self.read_json(session_id, "faiss_mapping.json")

    def faiss_index_path(self, session_id: str) -> Path:
        return self.path(session_id, "faiss.index")

    def append_question(self, session_id: str, record: dict[str, Any]) -> None:
        meta = self.read_json(session_id, "metadata.json")
        history = meta.get("question_history") or []
        history.append(record)
        meta["question_history"] = history
        self.write_json(session_id, "metadata.json", meta)
        # Also persist dedicated questions log
        qpath = self.path(session_id, "questions.json")
        if qpath.exists():
            questions = json.loads(qpath.read_text(encoding="utf-8"))
        else:
            questions = []
        questions.append(record)
        qpath.write_text(json.dumps(questions, indent=2), encoding="utf-8")

    def reset_query_progress(
        self, session_id: str, question: str
    ) -> dict[str, Any]:
        """Start a fresh live query-pipeline progress file for the UI."""
        payload = {
            "session_id": session_id,
            "question": question,
            "status": "running",
            "stage": "starting",
            "progress": 0.0,
            "message": "Starting query pipeline",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "error": None,
            "stages": [],
            "query_type": None,
            "path": None,
            "result_question_id": None,
        }
        self.write_json(session_id, "query_progress.json", payload)
        return payload

    def update_query_progress(
        self, session_id: str, **fields: Any
    ) -> dict[str, Any]:
        path = self.path(session_id, "query_progress.json")
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {
                "session_id": session_id,
                "status": "idle",
                "stage": None,
                "progress": 0.0,
                "stages": [],
            }
        data.update(fields)
        data["updated_at"] = utc_now()
        self.write_json(session_id, "query_progress.json", data)
        return data

    def upsert_query_stage(
        self,
        session_id: str,
        stage_info: dict[str, Any],
        *,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict[str, Any]:
        """Replace/append a query stage entry and mark the current stage."""
        path = self.path(session_id, "query_progress.json")
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {
                "session_id": session_id,
                "status": "running",
                "stages": [],
            }
        stages = list(data.get("stages") or [])
        stage_name = stage_info.get("stage")
        replaced = False
        for i, s in enumerate(stages):
            if s.get("stage") == stage_name:
                stages[i] = {**s, **stage_info}
                replaced = True
                break
        if not replaced:
            stages.append(stage_info)
        data["stages"] = stages
        data["stage"] = stage_name
        if progress is not None:
            data["progress"] = progress
        if message is not None:
            data["message"] = message
        if status is not None:
            data["status"] = status
        data["updated_at"] = utc_now()
        self.write_json(session_id, "query_progress.json", data)
        return data

    def get_query_progress(self, session_id: str) -> dict[str, Any]:
        path = self.path(session_id, "query_progress.json")
        if not path.exists():
            return {
                "session_id": session_id,
                "status": "idle",
                "stage": None,
                "progress": 0.0,
                "message": "No query in progress",
                "stages": [],
                "error": None,
            }
        return json.loads(path.read_text(encoding="utf-8"))

    def list_sessions(self) -> list[str]:
        root = self.settings.sessions_dir
        if not root.exists():
            return []
        return sorted([p.name for p in root.iterdir() if p.is_dir()])

    def is_video_ready(self, session_id: str) -> bool:
        if not self.exists(session_id):
            return False
        meta = self.read_json(session_id, "metadata.json")
        if not meta.get("video_ready"):
            return False
        required = [
            self.video_path(session_id),
            self.path(session_id, "visual_memory.json"),
            self.path(session_id, "embeddings.npy"),
            self.faiss_index_path(session_id),
            self.path(session_id, "faiss_mapping.json"),
        ]
        return all(p.exists() for p in required)

    def cleanup_tmp(self, session_id: str) -> None:
        tmp = self.session_dir(session_id) / "tmp"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(exist_ok=True)

    def validate_extension(self, filename: str) -> None:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Only MP4 is allowed.")
