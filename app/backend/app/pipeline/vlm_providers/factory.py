"""VLM factory — OpenRouter for COMPLEX/Slow only."""

from __future__ import annotations

from typing import Literal

from app.config import Settings, get_settings
from app.pipeline.vlm import VLMError, VLMService


def create_provider_for_path(
    path: Literal["FAST", "SLOW"],
    settings: Settings | None = None,
):
    """
    FAST → raises (VLM must not run for SIMPLE).
    SLOW → OpenRouter VLMService.
    """
    settings = settings or get_settings()
    if path == "FAST":
        raise VLMError(
            provider=settings.vlm_provider,
            model=settings.vlm_model,
            path="FAST",
            reason="VLM is disabled for SIMPLE/FAST path",
        )
    service = VLMService.get()
    return service, settings.vlm_model, "openrouter"
