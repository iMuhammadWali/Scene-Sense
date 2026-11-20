"""OpenRouter (OpenAI-compatible) multimodal VLM provider — COMPLEX path only."""

from __future__ import annotations

from typing import Any, Optional

import httpx
import numpy as np

from app.pipeline.vlm_providers.base_types import (
    VLMAPIError,
    VLMProvider,
    VLMResult,
    VLMUsage,
)
from app.pipeline.vlm_providers.encoding import (
    PROVIDER_DEFAULT_BASE_URLS,
    build_vision_messages,
    extract_usage,
)
from app.utils.timing import timed


COMPLEX_PROMPT = (
    "You are given selected frames from a video in chronological order.\n"
    "Use the temporal order of the frames to answer the question.\n"
    "Base your answer only on the provided visual evidence.\n"
    "Do not assume events that are not visible.\n"
    'If the answer cannot be determined from the provided evidence, respond: '
    '"Not enough visual evidence."\n'
    "Give a concise answer."
)


class OpenAICompatibleVLMProvider(VLMProvider):
    """OpenRouter chat/completions with image_url content parts."""

    name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 90.0,
        provider_name: str = "openrouter",
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or PROVIDER_DEFAULT_BASE_URLS["openrouter"]).rstrip(
            "/"
        )
        self.timeout_seconds = timeout_seconds
        self.name = provider_name

    def check_configuration(self, model: str) -> dict[str, Any]:
        configured = bool(self.api_key)
        return {
            "provider": self.name,
            "model": model,
            "configured": configured,
            "api_key_configured": configured,
            "base_url": self.base_url,
            "message": (
                "OpenRouter API key configured"
                if configured
                else "OPENROUTER_API_KEY is missing — set it in .env"
            ),
            "local": False,
            "mode": "api",
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
        path: str = "SLOW",
    ) -> VLMResult:
        if not self.api_key:
            raise VLMAPIError(
                provider=self.name,
                model=model,
                path=path,
                reason="OPENROUTER_API_KEY is missing. Add it to .env (backend only).",
            )
        if not frames:
            raise VLMAPIError(
                provider=self.name,
                model=model,
                path=path,
                reason="No evidence frames provided to VLM",
            )
        if not model:
            raise VLMAPIError(
                provider=self.name,
                model=model or "(empty)",
                path=path,
                reason="VLM_MODEL is not configured",
            )

        messages = build_vision_messages(question, prompt or COMPLEX_PROMPT, frames)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://scenesense.local",
            "X-Title": "SceneSense",
        }
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        http_status: Optional[int] = None
        try:
            with timed() as t:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    resp = client.post(url, headers=headers, json=body)
                http_status = resp.status_code
                try:
                    payload = resp.json()
                except Exception:
                    payload = {"raw_text": resp.text[:2000]}

                if resp.status_code == 401:
                    raise VLMAPIError(
                        provider=self.name,
                        model=model,
                        path=path,
                        http_status=401,
                        reason="Invalid or unauthorized OpenRouter API key",
                    )
                if resp.status_code == 429:
                    raise VLMAPIError(
                        provider=self.name,
                        model=model,
                        path=path,
                        http_status=429,
                        reason="OpenRouter rate limit exceeded",
                    )
                if resp.status_code == 404:
                    raise VLMAPIError(
                        provider=self.name,
                        model=model,
                        path=path,
                        http_status=404,
                        reason=f"Model not found or unavailable: {model}",
                    )
                if resp.status_code >= 400:
                    err = payload.get("error") if isinstance(payload, dict) else None
                    detail = (
                        err.get("message")
                        if isinstance(err, dict)
                        else (str(err) if err else resp.text[:500])
                    )
                    raise VLMAPIError(
                        provider=self.name,
                        model=model,
                        path=path,
                        http_status=resp.status_code,
                        reason=f"OpenRouter request failed: {detail}",
                    )

                answer = _extract_answer_text(payload)
                if not answer:
                    raise VLMAPIError(
                        provider=self.name,
                        model=model,
                        path=path,
                        http_status=http_status,
                        reason="Malformed OpenRouter response (empty answer)",
                    )
        except VLMAPIError:
            raise
        except httpx.TimeoutException as exc:
            raise VLMAPIError(
                provider=self.name,
                model=model,
                path=path,
                reason=f"OpenRouter request timed out after {self.timeout_seconds}s",
            ) from exc
        except httpx.HTTPError as exc:
            raise VLMAPIError(
                provider=self.name,
                model=model,
                path=path,
                reason=f"Network error talking to OpenRouter: {exc}",
            ) from exc

        usage_dict = extract_usage(payload if isinstance(payload, dict) else {})
        usage = VLMUsage(
            input_tokens=usage_dict.get("input_tokens"),
            output_tokens=usage_dict.get("output_tokens"),
            total_tokens=usage_dict.get("total_tokens"),
            estimated_cost=usage_dict.get("estimated_cost"),
            cost_note=usage_dict.get("cost_note") or "Not provided by provider",
            raw=usage_dict.get("raw") or {},
        )
        preview = str(payload)[:800] if payload is not None else ""
        return VLMResult(
            answer=answer.strip(),
            provider=self.name,
            model=model,
            path=path,
            frame_count=len(frames),
            timestamps=list(timestamps or []),
            latency_seconds=t[0],
            status="success",
            http_status=http_status,
            usage=usage,
            raw_response_preview=preview,
            extra={"sent_full_video": False, "sent_embeddings": False},
        )


def _extract_answer_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p).strip()
    return ""
