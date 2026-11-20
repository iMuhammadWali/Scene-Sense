"""Shared helpers for encoding evidence frames for VLM APIs."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image


def numpy_to_jpeg_bytes(image: np.ndarray, quality: int = 85) -> bytes:
    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def numpy_to_data_url(image: np.ndarray, quality: int = 85) -> str:
    raw = numpy_to_jpeg_bytes(image, quality=quality)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_vision_messages(
    question: str,
    prompt: str,
    frames: list[np.ndarray],
) -> list[dict[str, Any]]:
    """OpenAI-compatible multimodal chat messages (chronological frame order).

    Text first, then images — recommended by OpenRouter multimodal docs.
    """
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"Question:\n{question}\n\n{prompt}",
        }
    ]
    for frame in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": numpy_to_data_url(frame)},
            }
        )
    return [{"role": "user", "content": content}]


def extract_usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "estimated_cost": None,
            "cost_note": "Not provided by provider",
            "raw": {},
        }
    return {
        "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "estimated_cost": usage.get("cost") or usage.get("total_cost"),
        "cost_note": (
            "Provided by provider"
            if (usage.get("cost") is not None or usage.get("total_cost") is not None)
            else "Not provided by provider"
        ),
        "raw": usage,
    }


PROVIDER_DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
}
