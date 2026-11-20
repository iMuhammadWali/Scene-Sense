"""Timing helpers for research instrumentation."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Iterator


@dataclass
class StageTimer:
    """Collect wall-clock timings for named pipeline stages."""

    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def track(self, name: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round(time.perf_counter() - start, 4)

    def total(self) -> float:
        return round(sum(self.stages.values()), 4)

    def as_dict(self) -> dict[str, Any]:
        return {"stages": dict(self.stages), "total_seconds": self.total()}


@contextmanager
def timed() -> Iterator[list[float]]:
    """Yield a one-element list that receives elapsed seconds."""
    box: list[float] = [0.0]
    start = time.perf_counter()
    try:
        yield box
    finally:
        box[0] = round(time.perf_counter() - start, 4)
