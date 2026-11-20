"""Legacy stub — local VLM removed. Use OpenRouter via app.pipeline.vlm.VLMService."""

from __future__ import annotations
from typing import Any, Optional
import numpy as np
from app.pipeline.vlm_providers.base_types import VLMAPIError, VLMProvider, VLMResult


class LocalStubVLMProvider(VLMProvider):
    name = "local_disabled"

    def check_configuration(self, model: str) -> dict[str, Any]:
        return {
            "provider": "local_disabled",
            "model": model,
            "configured": False,
            "message": "Local VLM removed. COMPLEX path uses OpenRouter only.",
        }

    def generate_answer(
        self,
        question: str,
        frames: list[np.ndarray],
        prompt: str,
        model: str,
        *,
        timestamps: Optional[list[float]] = None,
        max_tokens: int = 256,
    ) -> VLMResult:
        raise VLMAPIError(
            provider="local_disabled",
            model=model,
            reason="Local SmolVLM is removed. Use OpenRouter for COMPLEX queries.",
        )
