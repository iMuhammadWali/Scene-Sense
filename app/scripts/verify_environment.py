#!/usr/bin/env python3
"""Verify SceneSense environment (CLIP + BART local; OpenRouter for COMPLEX)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> int:
    from app.config import get_settings
    from app.services.model_status import get_model_status
    from app.utils.hardware import detect_hardware
    from app.utils.model_cache import model_cached

    settings = get_settings()
    hw = detect_hardware()
    status = get_model_status(probe_api=False)
    failed = False

    print("SceneSense Environment")
    print("=" * 40)
    print(f"Python: {hw.get('python_version')}")
    try:
        import torch

        print(f"PyTorch: OK — {torch.__version__}")
    except Exception as exc:
        print(f"PyTorch: FAIL — {exc}")
        failed = True

    try:
        import transformers

        print(f"Transformers: OK — {transformers.__version__}")
    except Exception as exc:
        print(f"Transformers: FAIL — {exc}")
        failed = True

    print(f"CUDA: {'Available' if hw.get('cuda_available') else 'Not available'}")
    print(f"GPU: {hw.get('gpu_name') or 'none'}")
    print(f"RAM: {hw.get('ram_gb')} GB")

    ffmpeg = shutil.which("ffmpeg")
    print(f"FFmpeg: {'OK — ' + ffmpeg if ffmpeg else 'not on PATH'}")

    try:
        import faiss  # noqa: F401

        print("FAISS: OK")
    except Exception as e:
        print(f"FAISS: FAIL — {e}")
        failed = True

    try:
        import cv2

        print(f"OpenCV: OK — {cv2.__version__}")
    except Exception as exc:
        print(f"OpenCV: FAIL — {exc}")
        failed = True

    print()
    print("CLIP:")
    if model_cached(settings.clip_model):
        print(f"    [OK] Cached — {settings.clip_model}")
    else:
        print(f"    [FAIL] Missing — {settings.clip_model}")
        failed = True

    print("Classifier:")
    if model_cached(settings.classifier_model):
        print(f"    [OK] Cached — {settings.classifier_model}")
    else:
        print(f"    [FAIL] Missing — {settings.classifier_model}")
        failed = True

    print("VLM (OpenRouter — COMPLEX only):")
    key_ok = bool((settings.openrouter_api_key or "").strip())
    print(f"    Provider: {settings.vlm_provider}")
    print(f"    Model: {settings.vlm_model}")
    if key_ok:
        print("    [OK] OPENROUTER_API_KEY configured")
    else:
        print("    [WARN] OPENROUTER_API_KEY missing — SIMPLE still works; COMPLEX will fail")

    print()
    print("NOTE: SIMPLE = CLIP/FAISS only (no VLM).")
    print("      COMPLEX = CLIP/FAISS + OpenRouter VLM.")
    print()
    if failed:
        print("Environment incomplete. Run: python scripts/download_models.py")
        return 1
    print("Environment looks ready for local retrieval.")
    print(f"Message: {status.get('message')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
