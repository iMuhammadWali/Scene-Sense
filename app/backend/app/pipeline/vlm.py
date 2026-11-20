"""VLM service — OpenRouter only, COMPLEX / Slow path exclusively."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np

from app.config import get_settings
from app.pipeline.vlm_providers.base_types import VLMAPIError
from app.pipeline.vlm_providers.openai_compatible import (
    COMPLEX_PROMPT,
    OpenAICompatibleVLMProvider,
)
from app.utils.logging_setup import get_logger

logger = get_logger("scenesense.vlm")


# Back-compat alias used by routes
VLMError = VLMAPIError


class VLMService:
    """
    OpenRouter VLM for COMPLEX (Slow) queries only.

    SIMPLE / Fast path must NEVER call this service.
    There is no local VLM.
    """

    _instance: Optional["VLMService"] = None

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        provider = (settings.vlm_provider or "openrouter").lower().strip()
        if provider != "openrouter":
            raise VLMAPIError(
                provider=provider,
                model=settings.vlm_model,
                reason=(
                    f"VLM_PROVIDER={provider!r} is not supported. "
                    "Only openrouter is allowed."
                ),
            )
        self.model_id = settings.vlm_model
        self.provider_name = "openrouter"
        self._provider = OpenAICompatibleVLMProvider(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout_seconds=settings.vlm_timeout_seconds,
            provider_name="openrouter",
        )

    @classmethod
    def get(cls) -> "VLMService":
        if cls._instance is None:
            cls._instance = VLMService()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def is_configured(self) -> bool:
        return bool((self.settings.openrouter_api_key or "").strip())

    def generate(
        self,
        path: Literal["FAST", "SLOW"],
        question: str,
        images: list[np.ndarray],
        timestamps: Optional[list[float]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate an answer. FAST path is rejected by design."""
        if path == "FAST":
            raise VLMAPIError(
                provider=self.provider_name,
                model=self.model_id,
                path="FAST",
                reason=(
                    "VLM must not be called for SIMPLE/FAST queries. "
                    "Use CLIP/FAISS retrieval evidence only."
                ),
            )
        if path != "SLOW":
            raise VLMAPIError(
                provider=self.provider_name,
                model=self.model_id,
                path=str(path),
                reason=f"Unsupported path {path!r}; only SLOW (COMPLEX) may call VLM",
            )

        result = self._provider.generate_answer(
            question=question,
            frames=images,
            prompt=COMPLEX_PROMPT,
            model=self.model_id,
            timestamps=timestamps,
            max_tokens=max_new_tokens or self.settings.vlm_max_new_tokens,
            path="SLOW",
        )

        frames_meta = []
        for i, _ in enumerate(images):
            frames_meta.append(
                {
                    "timestamp": (
                        timestamps[i] if timestamps and i < len(timestamps) else None
                    ),
                    "index": i,
                }
            )

        debug = {
            "vlm": "OpenRouter",
            "vlm_model_id": result.model,
            "path": "SLOW",
            "mode": "api",
            "provider": result.provider,
            "used": True,
            "frames": frames_meta,
            "num_images": len(images),
            "timestamps": list(timestamps or []),
            "processing_time_seconds": result.latency_seconds,
            "latency_ms": round(result.latency_seconds * 1000, 1),
            "status": result.status,
            "http_status": result.http_status,
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.total_tokens,
                "estimated_cost": result.usage.estimated_cost,
                "cost_note": result.usage.cost_note,
            },
            "request_log": {
                "provider": result.provider,
                "model": result.model,
                "path": "SLOW",
                "question": question,
                "frame_count": len(images),
                "timestamps": list(timestamps or []),
                "latency_seconds": result.latency_seconds,
                "status": "success",
                "sent_full_video": False,
            },
            "raw_response": result.answer,
            "note": (
                "Only selected evidence frames were sent to OpenRouter "
                "(not the full video). SIMPLE queries never call this service."
            ),
        }
        logger.info(
            "[VLM] path=SLOW provider=openrouter model=%s frames=%s duration=%.2fs",
            result.model,
            len(images),
            result.latency_seconds,
        )
        return result.answer, debug

    def status(self) -> dict[str, Any]:
        cfg = self._provider.check_configuration(self.model_id)
        configured = bool(cfg.get("configured"))
        return {
            "provider": "openrouter",
            "mode": "api",
            "model": self.model_id,
            "local": False,
            "cached": False,
            "loaded": False,
            "configured": configured,
            "reachable": configured,
            "api_key_configured": configured,
            "message": cfg.get("message"),
            "fast": {
                "provider": None,
                "model": None,
                "used": False,
                "path": "FAST",
                "note": "SIMPLE/FAST never calls VLM",
            },
            "slow": {
                "provider": "openrouter",
                "model": self.model_id,
                "path": "SLOW",
                "configured": configured,
                "message": cfg.get("message"),
            },
        }
