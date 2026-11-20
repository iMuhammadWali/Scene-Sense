"""Model availability — local CLIP + BART; OpenRouter VLM (COMPLEX only)."""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.utils.hardware import detect_hardware
from app.utils.model_cache import model_cached

_model_cached = model_cached


def get_model_status(probe_api: bool = True) -> dict[str, Any]:
    settings = get_settings()
    clip = {
        "local": True,
        "mode": "local",
        "loaded": False,
        "cached": model_cached(settings.clip_model),
        "model": settings.clip_model,
        "status": "missing",
        "required_download": True,
    }
    classifier = {
        "local": True,
        "mode": "local",
        "loaded": False,
        "cached": model_cached(settings.classifier_model),
        "model": settings.classifier_model,
        "status": "missing",
        "required_download": True,
    }

    try:
        from app.pipeline.clip_encoder import CLIPEncoder

        clip["loaded"] = CLIPEncoder.get().is_ready()
    except Exception:
        pass
    try:
        from app.pipeline.classifier import QueryClassifier

        classifier["loaded"] = QueryClassifier.get()._loaded  # noqa: SLF001
    except Exception:
        pass

    clip["status"] = "ready" if clip["cached"] else "missing"
    classifier["status"] = "ready" if classifier["cached"] else "missing"

    key_ok = bool((settings.openrouter_api_key or "").strip())
    vlm_message = (
        "OpenRouter API key configured (COMPLEX path only)"
        if key_ok
        else "OPENROUTER_API_KEY not set — COMPLEX queries will fail"
    )
    if probe_api:
        try:
            from app.pipeline.vlm import VLMService

            st = VLMService.get().status()
            vlm_message = st.get("message") or vlm_message
            key_ok = bool(st.get("configured", key_ok))
        except Exception as exc:
            vlm_message = f"VLM status check failed: {exc}"

    vlm = {
        "local": False,
        "mode": "api",
        "provider": "openrouter",
        "model": settings.vlm_model,
        "cached": False,
        "loaded": False,
        "configured": key_ok,
        "reachable": key_ok,
        "api_key_configured": key_ok,
        "status": "configured" if key_ok else "not_configured",
        "message": vlm_message,
        "required_download": False,
        "download": "API model — not downloaded locally",
        "fast_path_model": None,
        "slow_path_model": settings.vlm_model,
        "note": "SIMPLE never uses VLM; COMPLEX uses OpenRouter only",
    }

    local_ok = clip["cached"] and classifier["cached"]
    if not local_ok:
        message = "Local models missing. Run: python scripts/download_models.py"
    elif not key_ok:
        message = (
            "CLIP + BART ready. Set OPENROUTER_API_KEY for COMPLEX queries. "
            "SIMPLE queries work without a VLM key."
        )
    else:
        message = "CLIP + BART ready. OpenRouter configured for COMPLEX path."

    return {
        "clip": clip,
        "classifier": classifier,
        "vlm": vlm,
        "fast_vlm": {
            "used": False,
            "provider": None,
            "model": None,
            "path": "FAST",
            "note": "SIMPLE/FAST never calls VLM",
            "status": "not_used",
            "configured": True,
        },
        "slow_vlm": {
            **vlm,
            "path": "SLOW",
            "note": "COMPLEX/SLOW uses OpenRouter",
        },
        "local_models_ready": local_ok,
        "vlm_api_configured": key_ok,
        "hf_cache_dir": str(settings.hf_cache_dir),
        "hardware": detect_hardware(),
        "message": message,
        "models": {
            "clip": {"id": clip["model"], "cached": clip["cached"], "local": True},
            "classifier": {
                "id": classifier["model"],
                "cached": classifier["cached"],
                "local": True,
            },
            "vlm": {
                "id": vlm["model"],
                "cached": False,
                "local": False,
                "provider": "openrouter",
                "configured": key_ok,
            },
        },
        "all_required_cached": local_ok,
    }
