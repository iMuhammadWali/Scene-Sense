"""Video upload validation and storage."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO

from app.config import get_settings
from app.pipeline.frame_extraction import read_video_metadata
from app.storage.session import SessionStore
from app.utils.logging_setup import get_logger

logger = get_logger("scenesense.upload")


class UploadService:
    def __init__(self, store: SessionStore | None = None) -> None:
        self.settings = get_settings()
        self.store = store or SessionStore(self.settings)

    def save_upload(self, filename: str, fileobj: BinaryIO, size_hint: int | None = None) -> dict:
        self.store.validate_extension(filename)
        safe_name = Path(filename).name
        # Create session first
        session_id = self.store.create_session(safe_name, size_bytes=size_hint or 0)
        dest = self.store.video_path(session_id)

        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        written = 0
        with dest.open("wb") as out:
            while True:
                chunk = fileobj.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    dest.unlink(missing_ok=True)
                    shutil.rmtree(self.store.session_dir(session_id), ignore_errors=True)
                    raise ValueError(
                        f"File exceeds max upload size of {self.settings.max_upload_mb} MB"
                    )
                out.write(chunk)

        if written < 1024:
            shutil.rmtree(self.store.session_dir(session_id), ignore_errors=True)
            raise ValueError("Uploaded file is too small or empty")

        # Validate can open as video
        try:
            meta = read_video_metadata(str(dest))
            if meta["duration_seconds"] <= 0 and meta["frame_count"] <= 0:
                raise ValueError("Video appears to have zero duration")
        except Exception as exc:
            shutil.rmtree(self.store.session_dir(session_id), ignore_errors=True)
            raise ValueError(f"Invalid or corrupted video: {exc}") from exc

        self.store.update_metadata(
            session_id,
            size_bytes=written,
            status="uploaded",
            stage="upload",
            progress=0.08,
            message="Upload complete; ready to process",
            video_metadata=meta,
        )
        self.store.append_stage_log(
            session_id,
            {
                "stage": "upload",
                "status": "completed",
                "filename": safe_name,
                "size_bytes": written,
                "video_metadata": meta,
            },
        )
        logger.info(
            "[UPLOAD] session=%s bytes=%s duration=%s",
            session_id,
            written,
            meta.get("duration_seconds"),
        )
        return {
            "session_id": session_id,
            "filename": safe_name,
            "size_bytes": written,
            "status": "uploaded",
            "message": "Upload successful",
            "video_metadata": meta,
        }
