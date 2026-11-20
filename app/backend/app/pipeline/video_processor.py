"""One-time video processing pipeline: extract → CLIP → prune → memory → FAISS."""

from __future__ import annotations

from typing import Any, Callable, Optional

from app.config import get_settings
from app.pipeline.clip_encoder import CLIPEncoder
from app.pipeline.faiss_index import build_faiss_index, load_index, save_index
from app.pipeline.frame_extraction import extract_candidates, read_video_metadata
from app.pipeline.pruning import prune_frames
from app.storage.session import SessionStore
from app.utils.logging_setup import get_logger
from app.utils.timing import timed

logger = get_logger("scenesense.video_processor")

ProgressCb = Optional[Callable[[str, float, str], None]]


class VideoProcessor:
    def __init__(self, store: Optional[SessionStore] = None) -> None:
        self.settings = get_settings()
        self.store = store or SessionStore(self.settings)

    def process(self, session_id: str, progress: ProgressCb = None) -> dict[str, Any]:
        """Run the full one-time preprocessing pipeline for a session."""
        if not self.store.exists(session_id):
            raise FileNotFoundError("Invalid session")

        video_path = str(self.store.video_path(session_id))
        if not self.store.video_path(session_id).exists():
            raise FileNotFoundError("Original video missing")

        def report(stage: str, pct: float, message: str) -> None:
            self.store.update_metadata(
                session_id,
                status="processing",
                stage=stage,
                progress=pct,
                message=message,
                video_ready=False,
            )
            if progress:
                progress(stage, pct, message)

        with timed() as total_t:
            report("frame_extraction", 0.1, "Extracting frames at 5 FPS")
            try:
                meta = read_video_metadata(video_path)
                self.store.update_metadata(session_id, video_metadata=meta)

                candidates, extract_info = extract_candidates(
                    video_path, target_fps=self.settings.frame_extraction_fps
                )
                self.store.append_stage_log(session_id, extract_info)

                report("clip_encoding", 0.35, "Encoding candidate frames with CLIP")
                clip = CLIPEncoder.get()
                images = [c.image for c in candidates if c.image is not None]
                with timed() as clip_t:
                    embeddings = clip.encode_images(images)
                clip_info = {
                    "stage": "clip_encoding",
                    "status": "completed",
                    "input_count": len(images),
                    "output_count": int(embeddings.shape[0]),
                    "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                    "model": clip.model_name,
                    "processing_time_seconds": clip_t[0],
                }
                self.store.append_stage_log(session_id, clip_info)

                # Free raw images from candidate objects before pruning result keep
                for c in candidates:
                    c.image = None

                report("frame_pruning", 0.55, "Pruning redundant frames (threshold 0.95)")
                retained, retained_emb, prune_info = prune_frames(
                    candidates,
                    embeddings,
                    threshold=self.settings.pruning_threshold,
                )
                self.store.append_stage_log(session_id, prune_info)

                report("visual_memory", 0.75, "Writing visual memory (embeddings only)")
                with timed() as mem_t:
                    frame_meta = [
                        {
                            "frame_id": f.frame_id,
                            "timestamp": f.timestamp,
                            "source_frame_index": f.source_frame_index,
                        }
                        for f in retained
                    ]
                    self.store.save_visual_memory(session_id, frame_meta, retained_emb)
                mem_info = {
                    "stage": "visual_memory",
                    "status": "completed",
                    "input_count": len(retained),
                    "output_count": len(retained),
                    "stored_images": 0,
                    "embeddings": len(retained),
                    "processing_time_seconds": mem_t[0],
                    "note": "Images are NOT permanently stored; original MP4 is source of truth",
                }
                self.store.append_stage_log(session_id, mem_info)

                report("faiss_index", 0.88, "Building FAISS index")
                index, mapping, faiss_info = build_faiss_index(retained_emb, frame_meta)
                save_index(index, self.store.faiss_index_path(session_id))
                self.store.save_faiss_mapping(session_id, mapping)
                # Verify reload
                reloaded = load_index(self.store.faiss_index_path(session_id))
                if int(reloaded.ntotal) != len(mapping):
                    raise RuntimeError("FAISS reload verification failed")
                faiss_info["verified_reload"] = True
                self.store.append_stage_log(session_id, faiss_info)

                self.store.cleanup_tmp(session_id)

                ready_info = {
                    "stage": "video_ready",
                    "status": "completed",
                    "retained_frames": len(retained),
                    "faiss_vectors": int(reloaded.ntotal),
                    "stored_images": 0,
                    "processing_time_seconds": total_t[0],
                }
                self.store.append_stage_log(session_id, ready_info)

                self.store.update_metadata(
                    session_id,
                    status="ready",
                    stage="video_ready",
                    progress=1.0,
                    message="VIDEO READY",
                    video_ready=True,
                    retained_frame_count=len(retained),
                    candidate_frame_count=len(candidates),
                    total_preprocessing_seconds=total_t[0],
                )
                logger.info(
                    "[VIDEO_READY] session=%s retained=%s total=%.2fs",
                    session_id,
                    len(retained),
                    total_t[0],
                )
                return self.store.read_json(session_id, "metadata.json")
            except Exception as exc:
                logger.exception("[VIDEO_PROCESS_FAIL] session=%s", session_id)
                self.store.update_metadata(
                    session_id,
                    status="error",
                    stage="error",
                    progress=1.0,
                    message="Processing failed",
                    video_ready=False,
                    error=str(exc),
                )
                self.store.append_stage_log(
                    session_id,
                    {
                        "stage": "error",
                        "status": "failed",
                        "error": str(exc),
                    },
                )
                raise
