#!/usr/bin/env python3
"""Optional OpenRouter smoke test (requires OPENROUTER_API_KEY).

Usage:
  python scripts/smoke_vlm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> int:
    from app.config import get_settings
    from app.pipeline.vlm import VLMService

    settings = get_settings()
    print("OpenRouter VLM smoke test (COMPLEX path only)")
    print("=" * 40)
    print("Provider:", settings.vlm_provider)
    print("Model:", settings.vlm_model)
    if not (settings.openrouter_api_key or "").strip():
        print("FAILED: OPENROUTER_API_KEY not set")
        return 1

    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :112] = (40, 80, 200)
    img[:, 112:] = (200, 60, 40)

    VLMService.reset()
    svc = VLMService.get()
    try:
        # FAST must fail
        try:
            svc.generate("FAST", "x", images=[img])
            print("FAILED: FAST path should reject VLM call")
            return 1
        except Exception as exc:
            print("OK: FAST rejected —", str(exc).splitlines()[0])

        answer, debug = svc.generate(
            "SLOW",
            "What colors are visible?",
            images=[img],
            timestamps=[0.0],
            max_new_tokens=64,
        )
    except Exception as exc:
        print("FAILED during OpenRouter call:")
        print(exc)
        return 1

    print("Answer:", answer)
    print("Latency:", debug.get("processing_time_seconds"), "s")
    print("Provider:", debug.get("provider"))
    if not answer.strip():
        print("FAILED: empty answer")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
