"""REST API routes for SceneSense."""

from __future__ import annotations

import asyncio
import io
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from app import __version__
from app.config import get_settings
from app.pipeline.frame_extraction import extract_frame_at_timestamp
from app.pipeline.query_pipeline import QueryPipeline
from app.pipeline.video_processor import VideoProcessor
from app.pipeline.vlm import VLMError
from app.schemas.api import QuestionRequest
from app.services.model_status import get_model_status
from app.services.upload import UploadService
from app.storage.session import SessionStore
from app.utils.hardware import detect_hardware
from app.utils.logging_setup import get_logger

logger = get_logger("scenesense.api")
router = APIRouter()

# In-process lock to avoid concurrent processing of same session
_processing: set[str] = set()


def _store() -> SessionStore:
    return SessionStore(get_settings())


@router.get("/health")
async def health() -> dict[str, Any]:
    settings = get_settings()
    model_status = get_model_status(probe_api=True)
    return {
        "status": "ok",
        "version": __version__,
        "hardware": detect_hardware(),
        "models": model_status,
        "config": {
            "frame_extraction_fps": settings.frame_extraction_fps,
            "pruning_threshold": settings.pruning_threshold,
            "faiss_top_k": settings.faiss_top_k,
            "retrieval_threshold": settings.retrieval_threshold,
            "temporal_group_gap_seconds": settings.temporal_group_gap_seconds,
            "complex_context_window_seconds": settings.complex_context_window_seconds,
            "complex_sampling_fps": settings.complex_sampling_fps,
            "clip_model": settings.clip_model,
            "classifier_model": settings.classifier_model,
            "vlm_provider": settings.vlm_provider,
            "vlm_model": settings.vlm_model,
            "privacy": {
                "retrieval_local": True,
                "vlm_local": False,
                "vlm_provider": "openrouter",
                "vlm_only_for_complex": True,
                "vlm_sends_selected_frames_only": True,
                "simple_never_calls_vlm": True,
            },
        },
    }


@router.get("/models/status")
async def models_status() -> dict[str, Any]:
    """Internal/debug model status — never includes API keys."""
    status = get_model_status(probe_api=True)
    vlm = status.get("vlm") or status.get("fast_vlm") or {}
    return {
        "clip": {
            "local": True,
            "mode": "local",
            "status": status["clip"].get("status"),
            "loaded": status["clip"]["loaded"],
            "cached": status["clip"]["cached"],
            "model": status["clip"]["model"],
        },
        "classifier": {
            "local": True,
            "mode": "local",
            "status": status["classifier"].get("status"),
            "loaded": status["classifier"]["loaded"],
            "cached": status["classifier"]["cached"],
            "model": status["classifier"]["model"],
        },
        "vlm": {
            "local": False,
            "mode": "api",
            "provider": "openrouter",
            "status": vlm.get("status"),
            "model": vlm.get("model"),
            "cached": False,
            "loaded": False,
            "configured": vlm.get("configured"),
            "api_key_configured": vlm.get("api_key_configured"),
            "message": vlm.get("message"),
            "note": "SIMPLE never uses VLM; COMPLEX uses OpenRouter only",
        },
        "message": status.get("message"),
        "hardware": status.get("hardware"),
    }


@router.get("/sessions")
async def list_sessions() -> dict[str, Any]:
    store = _store()
    sessions = []
    for sid in store.list_sessions():
        try:
            meta = store.read_json(sid, "metadata.json")
            sessions.append(
                {
                    "session_id": sid,
                    "status": meta.get("status"),
                    "video_ready": store.is_video_ready(sid),
                    "original_filename": meta.get("original_filename"),
                    "created_at": meta.get("created_at"),
                }
            )
        except Exception:
            continue
    return {"sessions": sessions}


@router.post("/videos/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_process: bool = True,
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    try:
        result = UploadService().save_upload(
            filename=file.filename,
            fileobj=file.file,
            size_hint=None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    if auto_process:
        background_tasks.add_task(_run_processing, result["session_id"])
        result["message"] = "Upload successful; processing started"
        result["status"] = "processing"
    return result


@router.post("/videos/{session_id}/process")
async def start_processing(session_id: str, background_tasks: BackgroundTasks) -> dict[str, Any]:
    store = _store()
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    if store.is_video_ready(session_id):
        return {
            "session_id": session_id,
            "status": "ready",
            "message": "Already VIDEO READY; not reprocessing",
            "video_ready": True,
        }
    if session_id in _processing:
        return {
            "session_id": session_id,
            "status": "processing",
            "message": "Processing already in progress",
        }
    background_tasks.add_task(_run_processing, session_id)
    return {
        "session_id": session_id,
        "status": "processing",
        "message": "Processing started",
    }


def _run_processing(session_id: str) -> None:
    if session_id in _processing:
        return
    _processing.add(session_id)
    try:
        VideoProcessor().process(session_id)
    except Exception:
        logger.exception("Background processing failed session=%s", session_id)
    finally:
        _processing.discard(session_id)


@router.get("/videos/{session_id}/status")
async def get_status(session_id: str) -> dict[str, Any]:
    store = _store()
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    meta = store.read_json(session_id, "metadata.json")
    return {
        "session_id": session_id,
        "status": meta.get("status"),
        "stage": meta.get("stage"),
        "progress": meta.get("progress", 0),
        "message": meta.get("message"),
        "video_ready": store.is_video_ready(session_id),
        "stages": meta.get("stages", []),
        "error": meta.get("error"),
        "video_metadata": meta.get("video_metadata"),
        "retained_frame_count": meta.get("retained_frame_count"),
        "candidate_frame_count": meta.get("candidate_frame_count"),
    }


@router.get("/videos/{session_id}/debug")
async def get_debug(session_id: str) -> dict[str, Any]:
    store = _store()
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    meta = store.read_json(session_id, "metadata.json")
    try:
        proc_log = store.read_json(session_id, "processing_log.json")
    except FileNotFoundError:
        proc_log = {}
    visual_summary = {"exists": False}
    if store.path(session_id, "visual_memory.json").exists():
        frames, emb = store.load_visual_memory(session_id)
        visual_summary = {
            "exists": True,
            "retained_frames": len(frames),
            "stored_images": 0,
            "embeddings": int(emb.shape[0]),
            "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
            "faiss_vectors": len(store.load_faiss_mapping(session_id))
            if store.path(session_id, "faiss_mapping.json").exists()
            else 0,
        }
    return {
        "session_id": session_id,
        "metadata": meta,
        "processing_log": proc_log,
        "visual_memory_summary": visual_summary,
        "video_ready": store.is_video_ready(session_id),
        "config": {
            "pruning_threshold": get_settings().pruning_threshold,
            "retrieval_threshold": get_settings().retrieval_threshold,
            "faiss_top_k": get_settings().faiss_top_k,
            "frame_extraction_fps": get_settings().frame_extraction_fps,
        },
    }


@router.get("/videos/{session_id}/metadata")
async def get_metadata(session_id: str) -> dict[str, Any]:
    store = _store()
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    meta = store.read_json(session_id, "metadata.json")
    summary = {"retained_frames": meta.get("retained_frame_count", 0), "stored_images": 0}
    if store.path(session_id, "visual_memory.json").exists():
        frames, emb = store.load_visual_memory(session_id)
        summary = {
            "retained_frames": len(frames),
            "stored_images": 0,
            "embeddings": int(emb.shape[0]),
            "faiss_vectors": len(store.load_faiss_mapping(session_id))
            if store.path(session_id, "faiss_mapping.json").exists()
            else 0,
        }
    return {
        "session_id": session_id,
        "metadata": meta,
        "visual_memory_summary": summary,
        "video_ready": store.is_video_ready(session_id),
    }


@router.post("/videos/{session_id}/questions")
async def ask_question(session_id: str, body: QuestionRequest) -> dict[str, Any]:
    store = _store()
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    if not store.is_video_ready(session_id):
        raise HTTPException(
            status_code=409,
            detail="VIDEO READY required before asking questions. Video is still processing or failed.",
        )
    try:
        # Run in thread to avoid blocking event loop during model inference
        result = await asyncio.to_thread(QueryPipeline().ask, session_id, body.question)
        return result
    except VLMError as exc:
        # Structured VLM failure — do not fall back to another model/path
        raise HTTPException(
            status_code=502,
            detail=exc.as_dict(),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Question failed")
        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {exc}",
        ) from exc


@router.get("/videos/{session_id}/query-progress")
async def query_progress(session_id: str) -> dict[str, Any]:
    """Live query-pipeline stages (poll while a question is running)."""
    store = _store()
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return store.get_query_progress(session_id)


@router.get("/videos/{session_id}/questions")
async def list_questions(session_id: str) -> dict[str, Any]:
    store = _store()
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    meta = store.read_json(session_id, "metadata.json")
    return {
        "session_id": session_id,
        "questions": meta.get("question_history", []),
    }


@router.get("/videos/{session_id}/evidence/{timestamp}")
async def get_evidence_frame(session_id: str, timestamp: float) -> Response:
    store = _store()
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    video = store.video_path(session_id)
    if not video.exists():
        raise HTTPException(status_code=404, detail="Video missing")
    try:
        rgb = extract_frame_at_timestamp(str(video), float(timestamp))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Evidence extraction failed: {exc}") from exc


@router.get("/videos/{session_id}/stream")
async def stream_video(session_id: str) -> FileResponse:
    store = _store()
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    path = store.video_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video missing")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=store.read_json(session_id, "metadata.json").get(
            "original_filename", "video.mp4"
        ),
    )
