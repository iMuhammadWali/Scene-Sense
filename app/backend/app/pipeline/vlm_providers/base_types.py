"""Provider-independent VLM types and interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class VLMUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    cost_note: str = "Not provided by provider"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class VLMResult:
    answer: str
    provider: str
    model: str
    path: str
    frame_count: int
    timestamps: list[float]
    latency_seconds: float
    status: str = "success"
    http_status: Optional[int] = None
    usage: VLMUsage = field(default_factory=VLMUsage)
    raw_response_preview: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class VLMAPIError(RuntimeError):
    """Structured VLM API failure — never silently fall back to another model."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        reason: str,
        http_status: Optional[int] = None,
        path: str = "",
    ) -> None:
        self.provider = provider
        self.model = model
        self.reason = reason
        self.http_status = http_status
        self.path = path
        parts = [
            "VLM API ERROR",
            f"Provider: {provider}",
            f"Model: {model}",
        ]
        if path:
            parts.append(f"Path: {path}")
        if http_status is not None:
            parts.append(f"HTTP status: {http_status}")
        parts.append(f"Reason: {reason}")
        super().__init__("\n".join(parts))

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": "VLM_API_ERROR",
            "provider": self.provider,
            "model": self.model,
            "path": self.path,
            "http_status": self.http_status,
            "reason": self.reason,
        }


class VLMProvider(ABC):
    """Abstract VLM backend (API or future local)."""

    name: str = "base"

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def check_configuration(self, model: str) -> dict[str, Any]:
        """Return configuration/reachability diagnostics (never includes API keys)."""
        raise NotImplementedError
