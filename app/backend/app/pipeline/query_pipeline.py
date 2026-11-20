"""Per-question pipeline: classify → CLIP/FAISS → Fast (no VLM) or Slow (OpenRouter)."""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

import numpy as np

from app.config import get_settings
from app.pipeline.classifier import QueryClassifier
from app.pipeline.clip_encoder import CLIPEncoder
from app.pipeline.diversity import select_diverse_evidence
from app.pipeline.faiss_index import load_index, search
from app.pipeline.frame_extraction import (
    extract_frame_at_timestamp,
    extract_frames_in_range,
    read_video_metadata,
)
from app.pipeline.retrieval import filter_by_similarity
from app.pipeline.temporal import group_temporally, merge_context_windows
from app.pipeline.vlm import VLMService
from app.storage.session import SessionStore, utc_now
from app.utils.logging_setup import get_logger
from app.utils.timing import StageTimer, timed

logger = get_logger("scenesense.query")

_STAGE_PROGRESS = {
    "query_classification": 0.08,
    "clip_text_embedding": 0.18,
    "faiss_search": 0.28,
    "similarity_filtering": 0.38,
    "temporal_grouping": 0.48,
    "diversity_selection": 0.58,
    "evidence_frame_extraction": 0.72,
    "vlm_inference": 0.92,
    "answer_ready": 1.0,
}


class QueryPipeline:
    def __init__(self, store: Optional[SessionStore] = None) -> None:
        self.settings = get_settings()
        self.store = store or SessionStore(self.settings)

    def _begin_stage(self, session_id: str, stage: str, message: str) -> float:
        self.store.upsert_query_stage(
            session_id,
            {
                "stage": stage,
                "status": "running",
                "message": message,
                "started_at": utc_now(),
            },
            progress=_STAGE_PROGRESS.get(stage, 0.5),
            message=message,
            status="running",
        )
        return time.perf_counter()

    def _end_stage(
        self,
        session_id: str,
        stage: str,
        started: float,
        extra: Optional[dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> float:
        elapsed = round(time.perf_counter() - started, 4)
        info: dict[str, Any] = {
            "stage": stage,
            "status": "completed",
            "processing_time_seconds": elapsed,
            "message": message or f"{stage} completed",
        }
        if extra:
            info.update(extra)
        self.store.upsert_query_stage(
            session_id,
            info,
            progress=_STAGE_PROGRESS.get(stage, 0.5),
            message=info["message"],
            status="running",
        )
        return elapsed

    def ask(self, session_id: str, question: str) -> dict[str, Any]:
        if not self.store.exists(session_id):
            raise FileNotFoundError("Invalid session")
        if not self.store.is_video_ready(session_id):
            raise RuntimeError("Video is not READY. Wait for processing to finish.")

        q = (question or "").strip()
        if not q:
            raise ValueError("Question cannot be empty")

        self.store.reset_query_progress(session_id, q)
        timer = StageTimer()
        video_path = str(self.store.video_path(session_id))
        mapping = self.store.load_faiss_mapping(session_id)
        index = load_index(self.store.faiss_index_path(session_id))

        try:
            # 1) Classification — absolute routing
            t0 = self._begin_stage(
                session_id, "query_classification", "Classifying SIMPLE vs COMPLEX"
            )
            with timer.track("query_classification"):
                query_type, class_info = QueryClassifier.get().classify(q)
            path = "FAST" if query_type == "SIMPLE" else "SLOW"
            self._end_stage(
                session_id,
                "query_classification",
                t0,
                extra={"query_type": query_type, "path": path, "model": class_info.get("model")},
                message=f"Classified as {query_type} → {path}",
            )
            self.store.update_query_progress(
                session_id, query_type=query_type, path=path
            )

            # 2) CLIP text embedding (both paths)
            t0 = self._begin_stage(
                session_id, "clip_text_embedding", "Encoding question with CLIP"
            )
            with timer.track("clip_text_embedding"):
                with timed() as emb_t:
                    query_emb = CLIPEncoder.get().encode_text(q)
                emb_info = {
                    "stage": "clip_text_embedding",
                    "status": "completed",
                    "embedding_dim": int(query_emb.shape[0]),
                    "normalized": True,
                    "processing_time_seconds": emb_t[0],
                }
            self._end_stage(
                session_id,
                "clip_text_embedding",
                t0,
                extra={"embedding_dim": int(query_emb.shape[0]), "input_count": 1, "output_count": 1},
            )

            # 3) FAISS Top-K
            t0 = self._begin_stage(
                session_id, "faiss_search", f"FAISS Top-{self.settings.faiss_top_k}"
            )
            with timer.track("faiss_search"):
                with timed() as faiss_t:
                    top_hits = search(
                        index, query_emb, mapping, top_k=self.settings.faiss_top_k
                    )
                faiss_info = {
                    "stage": "faiss_retrieval",
                    "status": "completed",
                    "top_k": self.settings.faiss_top_k,
                    "returned": len(top_hits),
                    "hits": top_hits,
                    "processing_time_seconds": faiss_t[0],
                }
            self._end_stage(
                session_id,
                "faiss_search",
                t0,
                extra={
                    "top_k": self.settings.faiss_top_k,
                    "input_count": 1,
                    "output_count": len(top_hits),
                },
            )

            # 4) Similarity filtering
            t0 = self._begin_stage(
                session_id, "similarity_filtering", "Filtering by similarity"
            )
            with timer.track("similarity_filtering"):
                with timed() as filt_t:
                    filtered, filt_info = filter_by_similarity(
                        top_hits, threshold=self.settings.retrieval_threshold
                    )
                filt_info["processing_time_seconds"] = filt_t[0]
            self._end_stage(
                session_id,
                "similarity_filtering",
                t0,
                extra={
                    "input_count": len(top_hits),
                    "output_count": len(filtered),
                    "threshold": self.settings.retrieval_threshold,
                    "candidate_count": filt_info.get("candidate_count"),
                    "passed_count": filt_info.get("passed_count"),
                    "fallback_used": filt_info.get("fallback_used"),
                    "best_similarity": filt_info.get("best_similarity"),
                },
                message=(
                    f"Fallback BEST MATCH (passed={filt_info.get('passed_count')})"
                    if filt_info.get("fallback_used")
                    else f"Passed threshold: {filt_info.get('passed_count')}"
                ),
            )

            # 5) Temporal grouping
            t0 = self._begin_stage(
                session_id, "temporal_grouping", "Grouping frames temporally"
            )
            with timer.track("temporal_grouping"):
                with timed() as grp_t:
                    groups, group_info = group_temporally(
                        filtered, gap_seconds=self.settings.temporal_group_gap_seconds
                    )
                group_info["processing_time_seconds"] = grp_t[0]
            self._end_stage(
                session_id,
                "temporal_grouping",
                t0,
                extra={"input_count": len(filtered), "output_count": len(groups)},
            )

            # 6) Diversity selection
            t0 = self._begin_stage(
                session_id, "diversity_selection", "Selecting diverse evidence"
            )
            with timer.track("diversity_selection"):
                with timed() as div_t:
                    selected, div_info = select_diverse_evidence(
                        groups, query_type=query_type
                    )
                div_info["processing_time_seconds"] = div_t[0]
            self._end_stage(
                session_id,
                "diversity_selection",
                t0,
                extra={
                    "input_count": len(groups),
                    "output_count": len(selected),
                    "query_type": query_type,
                },
            )

            # ---- FAST PATH: CLIP retrieval only — NEVER call VLM ----
            if query_type == "SIMPLE":
                t0 = self._begin_stage(
                    session_id,
                    "evidence_frame_extraction",
                    "Preparing retrieval evidence (no VLM)",
                )
                with timer.track("evidence_frame_extraction"):
                    evidence_meta = [
                        {
                            "evidence_id": i + 1,
                            "timestamp": float(item["timestamp"]),
                            "retrieved_frame_id": item.get("frame_id"),
                            "retrieved_timestamp": float(item["timestamp"]),
                            "source_frame_id": item.get("source_frame_index")
                            or item.get("frame_id"),
                            "frame_id": item.get("frame_id"),
                            "similarity": item.get("similarity"),
                            "passed_threshold": item.get("passed_threshold"),
                            "fallback": item.get("fallback", False),
                            "source": item.get("source", "selected"),
                        }
                        for i, item in enumerate(selected)
                    ]
                    extract_debug = {
                        "mode": "simple_retrieval_only",
                        "count": len(evidence_meta),
                        "vlm_called": False,
                    }
                self._end_stage(
                    session_id,
                    "evidence_frame_extraction",
                    t0,
                    extra={
                        "input_count": len(selected),
                        "output_count": len(evidence_meta),
                        "path": "FAST",
                        "vlm_called": False,
                        "timestamps": [e["timestamp"] for e in evidence_meta],
                    },
                    message="Retrieval evidence ready — VLM NOT USED",
                )

                # Explicit skip of VLM stage in progress UI
                self.store.upsert_query_stage(
                    session_id,
                    {
                        "stage": "vlm_inference",
                        "status": "skipped",
                        "processing_time_seconds": 0.0,
                        "message": "VLM skipped — SIMPLE uses CLIP/FAISS only",
                        "vlm_called": False,
                        "used": False,
                    },
                    progress=0.95,
                    message="VLM skipped (SIMPLE / FAST)",
                    status="running",
                )

                answer = _format_simple_retrieval_answer(evidence_meta)
                vlm_block = {
                    "used": False,
                    "provider": None,
                    "model": None,
                    "input_frames": [],
                    "answer": None,
                    "latency_ms": 0,
                    "cost": 0,
                    "reason": "SIMPLE query",
                    "note": "SIMPLE path never calls OpenRouter / any VLM",
                }
                vlm_debug = {
                    "used": False,
                    "provider": None,
                    "model": None,
                    "vlm": "NOT USED",
                    "path": "FAST",
                    "num_images": 0,
                    "processing_time_seconds": 0.0,
                    "status": "skipped",
                    "reason": "SIMPLE query",
                    "note": "CLIP retrieval only — no VLM",
                }
                short_vlm = "NOT USED"

            else:
                # ---- SLOW PATH: temporal evidence + OpenRouter ----
                t0 = self._begin_stage(
                    session_id,
                    "evidence_frame_extraction",
                    "Extracting temporal evidence frames for VLM",
                )
                with timer.track("evidence_frame_extraction"):
                    images, evidence_meta, extract_debug = self._build_complex_vlm_inputs(
                        video_path=video_path,
                        selected=selected,
                    )
                self._end_stage(
                    session_id,
                    "evidence_frame_extraction",
                    t0,
                    extra={
                        "input_count": len(selected),
                        "output_count": len(images),
                        "path": "SLOW",
                        "timestamps": [e.get("timestamp") for e in evidence_meta],
                    },
                )

                t0 = self._begin_stage(
                    session_id,
                    "vlm_inference",
                    f"OpenRouter VLM ({self.settings.vlm_model}, {len(images)} frames)",
                )
                with timer.track("vlm_inference"):
                    # Guard: only COMPLEX reaches here
                    answer, vlm_debug = VLMService.get().generate(
                        path="SLOW",
                        question=q,
                        images=images,
                        timestamps=[e["timestamp"] for e in evidence_meta],
                    )
                self._end_stage(
                    session_id,
                    "vlm_inference",
                    t0,
                    extra={
                        "provider": "openrouter",
                        "model": self.settings.vlm_model,
                        "num_images": len(images),
                        "path": "SLOW",
                        "vlm_called": True,
                    },
                )
                for i, e in enumerate(evidence_meta):
                    if i < len(vlm_debug.get("frames", [])):
                        vlm_debug["frames"][i]["evidence_id"] = e.get("evidence_id")
                        vlm_debug["frames"][i]["timestamp"] = e.get("timestamp")
                        vlm_debug["frames"][i]["source_frame_id"] = e.get(
                            "source_frame_id"
                        )
                        vlm_debug["frames"][i]["retrieved_frame_id"] = e.get(
                            "retrieved_frame_id"
                        )

                usage = vlm_debug.get("usage") or {}
                vlm_block = {
                    "used": True,
                    "provider": "openrouter",
                    "model": self.settings.vlm_model,
                    "reason": "COMPLEX query",
                    "input_frames": [
                        {
                            "evidence_id": e.get("evidence_id"),
                            "timestamp": e.get("timestamp"),
                            "source_frame_id": e.get("source_frame_id"),
                            "retrieved_frame_id": e.get("retrieved_frame_id"),
                        }
                        for e in evidence_meta
                    ],
                    "answer": answer,
                    "latency_ms": vlm_debug.get("latency_ms")
                    or round(float(vlm_debug.get("processing_time_seconds") or 0) * 1000, 1),
                    "usage": usage,
                    "cost": usage.get("estimated_cost"),
                }
                short_vlm = "OpenRouter"
                extract_debug["vlm_called"] = True

            question_id = uuid.uuid4().hex[:12]
            timings = timer.as_dict()
            timings["vlm_called"] = query_type == "COMPLEX"
            timings["num_retrieved"] = len(top_hits)
            timings["num_evidence_frames"] = len(evidence_meta)

            record = {
                "question_id": question_id,
                "query": q,
                "question": q,
                "answer": answer,
                "query_type": query_type,
                "path": path,
                "vlm": short_vlm,
                "evidence": evidence_meta,
                "retrieval": {
                    "top_k": self.settings.faiss_top_k,
                    "retrieval_threshold": self.settings.retrieval_threshold,
                    "pruning_threshold_not_used_here": self.settings.pruning_threshold,
                    "top_hits": top_hits,
                    "results": top_hits,
                    "filtered": filtered,
                    "temporal_groups": group_info.get("groups", []),
                    "selected": div_info.get("selected", []),
                    "selected_evidence": evidence_meta,
                },
                "vlm_info": vlm_block,
                "timings": timings,
                "debug": {
                    "classification": class_info,
                    "clip_query_embedding": emb_info,
                    "faiss": faiss_info,
                    "filtering": filt_info,
                    "temporal_grouping": group_info,
                    "diversity_selection": div_info,
                    "evidence_extraction": extract_debug,
                    "vlm": vlm_debug,
                    "routing": {
                        "query_type": query_type,
                        "path": path,
                        "override_allowed": False,
                        "vlm_allowed": query_type == "COMPLEX",
                        "note": "Retrieval confidence must NOT override SIMPLE/COMPLEX routing",
                    },
                },
            }
            self.store.append_question(session_id, record)
            self.store.upsert_query_stage(
                session_id,
                {
                    "stage": "answer_ready",
                    "status": "completed",
                    "processing_time_seconds": timings["total_seconds"],
                    "message": "Answer ready" if query_type == "COMPLEX" else "Retrieval ready",
                },
                progress=1.0,
                message="Done",
                status="completed",
            )
            self.store.update_query_progress(
                session_id,
                status="completed",
                stage="answer_ready",
                progress=1.0,
                result_question_id=question_id,
                message="Done",
            )
            logger.info(
                "[QUESTION] session=%s type=%s path=%s vlm=%s total=%.2fs",
                session_id,
                query_type,
                path,
                timings.get("vlm_called"),
                timings["total_seconds"],
            )
            return record
        except Exception as exc:
            self.store.update_query_progress(
                session_id,
                status="error",
                error=str(exc),
                message=f"Query failed: {exc}",
            )
            raise

    def _build_complex_vlm_inputs(
        self,
        video_path: str,
        selected: list[dict[str, Any]],
    ) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
        """COMPLEX: ±context windows, ~1 FPS, merge, chronological frames only.

        Each temporal sample gets its own evidence_id / timestamp / source_frame_id
        derived from the original MP4 timeline — NOT the triggering FAISS frame_id.
        """
        with timed() as t:
            images: list[np.ndarray] = []
            evidence: list[dict[str, Any]] = []
            meta = read_video_metadata(video_path)
            fps = float(meta.get("fps") or 0.0)
            frame_count = int(meta.get("frame_count") or 0)
            timestamps = [float(s["timestamp"]) for s in selected]
            windows = merge_context_windows(
                timestamps, window_seconds=self.settings.complex_context_window_seconds
            )
            max_frames = 12
            evidence_seq = 0
            for start, end in windows:
                sampled = extract_frames_in_range(
                    video_path,
                    start=start,
                    end=end,
                    sampling_fps=self.settings.complex_sampling_fps,
                )
                for ts, img in sampled:
                    if len(images) >= max_frames:
                        break
                    images.append(img)
                    evidence_seq += 1
                    # Triggering FAISS hit (relevance), kept separate from source frame
                    nearest = min(
                        selected,
                        key=lambda s: abs(float(s["timestamp"]) - ts),
                        default=None,
                    )
                    source_frame_id = _timestamp_to_source_frame_id(
                        ts, fps=fps, frame_count=frame_count
                    )
                    evidence.append(
                        {
                            "evidence_id": evidence_seq,
                            "timestamp": float(ts),
                            "evidence_timestamp": float(ts),
                            "source_frame_id": source_frame_id,
                            # Do NOT reuse FAISS frame_id as this sample's identity
                            "frame_id": source_frame_id,
                            "retrieved_frame_id": (
                                nearest.get("frame_id") if nearest else None
                            ),
                            "retrieved_timestamp": (
                                float(nearest["timestamp"]) if nearest else None
                            ),
                            "similarity": (
                                nearest.get("similarity") if nearest else None
                            ),
                            "source": "complex_context_window",
                        }
                    )
                if len(images) >= max_frames:
                    break

            # Chronological order for VLM (never by similarity)
            paired = sorted(
                zip(evidence, images), key=lambda x: float(x[0]["timestamp"])
            )
            evidence = [p[0] for p in paired]
            images = [p[1] for p in paired]
            # Re-number evidence_id after chronological sort
            for i, e in enumerate(evidence, start=1):
                e["evidence_id"] = i

            source_ids = [e.get("source_frame_id") for e in evidence]
            debug = {
                "mode": "complex_merged_context",
                "windows": [{"start": a, "end": b} for a, b in windows],
                "context_window_seconds": self.settings.complex_context_window_seconds,
                "sampling_fps": self.settings.complex_sampling_fps,
                "count": len(evidence),
                "max_frames_cap": max_frames,
                "video_fps": fps,
                "processing_time_seconds": t[0],
                "sent_full_video": False,
                "chronological": True,
                "unique_source_frame_ids": len(set(source_ids)),
                "evidence_items": [
                    {
                        "evidence_id": e.get("evidence_id"),
                        "timestamp": e.get("timestamp"),
                        "source_frame_id": e.get("source_frame_id"),
                        "retrieved_frame_id": e.get("retrieved_frame_id"),
                        "retrieved_timestamp": e.get("retrieved_timestamp"),
                    }
                    for e in evidence
                ],
            }
            return images, evidence, debug


def _timestamp_to_source_frame_id(
    timestamp: float, *, fps: float, frame_count: int
) -> Optional[int]:
    """Map an evidence timestamp to a native MP4 frame index."""
    if fps <= 0:
        return None
    idx = int(round(float(timestamp) * fps))
    if frame_count > 0:
        idx = max(0, min(idx, frame_count - 1))
    else:
        idx = max(0, idx)
    return idx


def _format_simple_retrieval_answer(evidence: list[dict[str, Any]]) -> str:
    """CLIP retrieval result — not a fabricated VLM natural-language answer."""
    if not evidence:
        return (
            "No matching visual evidence found.\n"
            "Visual evidence retrieved — VLM not used."
        )
    lines = [
        "Visual evidence found",
        "Visual evidence retrieved — VLM not used.",
        f"{len(evidence)} relevant frame(s):",
    ]
    for i, e in enumerate(evidence, 1):
        sim = e.get("similarity")
        sim_s = f"{float(sim):.3f}" if sim is not None else "—"
        fb = " [FALLBACK — did not pass threshold]" if e.get("fallback") else ""
        lines.append(
            f"  {i}. t={float(e['timestamp']):.2f}s  "
            f"retrieved_frame_id={e.get('retrieved_frame_id', e.get('frame_id'))}  "
            f"similarity={sim_s}{fb}"
        )
    lines.append("Click a timestamp to seek the original video.")
    return "\n".join(lines)
