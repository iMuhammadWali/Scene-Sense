#!/usr/bin/env python3
"""Download ONLY local SceneSense models:

- openai/clip-vit-base-patch32
- facebook/bart-large-mnli

OpenRouter VLM models are API-only and are NOT downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.utils.model_cache import model_cached  # noqa: E402


def download_repo(repo_id: str, cache_dir: str) -> str:
    from huggingface_hub import snapshot_download

    ignore = [
        "*.h5",
        "*tf_model*",
        "*flax*",
        "*.msgpack",
        "*.ot",
        "rust_model.ot",
    ]
    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        ignore_patterns=ignore,
    )


def ensure_model(label: str, repo_id: str, cache: str) -> None:
    if model_cached(repo_id):
        print(f"[OK] {label} found")
        print(f"    {repo_id}")
        return
    print(f"[ ] {label} missing — downloading...")
    print(f"    {repo_id}")
    path = download_repo(repo_id, cache)
    if not model_cached(repo_id):
        raise RuntimeError(
            f"{label} download finished but cache check still fails for {repo_id}. "
            f"Path reported: {path}"
        )
    print(f"[OK] {label} downloaded")
    print(f"    {path}")


def main() -> int:
    settings = get_settings()
    settings.ensure_directories()
    cache = str(settings.hf_cache_dir)

    print("SceneSense Model Setup (LOCAL: CLIP + BART)")
    print("=" * 40)
    print(f"Cache: {cache}\n")

    plan = [
        ("CLIP", settings.clip_model),
        ("Classifier (BART-large-MNLI)", settings.classifier_model),
    ]

    failures: list[str] = []
    for label, repo in plan:
        try:
            ensure_model(label, repo, cache)
        except Exception as exc:
            print(f"[FAIL] {label}: {exc}")
            failures.append(f"{label}: {exc}")
        print()

    print("[INFO] No local VLM is downloaded.")
    print(f"       COMPLEX path uses OpenRouter: {settings.vlm_model}")
    print("       SIMPLE path uses CLIP/FAISS only (no VLM).")
    print("       Set OPENROUTER_API_KEY in .env for COMPLEX queries.\n")

    if failures:
        print("Setup incomplete:")
        for f in failures:
            print(f" - {f}")
        return 1

    print("Setup complete.")
    print("Next: python scripts/verify_environment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
