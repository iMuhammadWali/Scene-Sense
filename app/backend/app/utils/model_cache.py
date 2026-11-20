"""Shared Hugging Face cache presence checks."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import try_to_load_from_cache

from app.config import get_settings


def model_cached(repo_id: str, filename: str = "config.json") -> bool:
    """True when config + at least one weight file exist in the local HF cache."""
    settings = get_settings()
    cache = str(settings.hf_cache_dir.resolve())

    def _ok(name: str, min_size: int = 1) -> bool:
        try:
            path = try_to_load_from_cache(
                repo_id=repo_id, filename=name, cache_dir=cache
            )
            if path is None or path == "PENDING":
                return False
            p = Path(str(path))
            return p.exists() and p.stat().st_size >= min_size
        except Exception:
            return False

    if not _ok(filename, min_size=1):
        return False

    weight_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return any(_ok(w, min_size=1024) for w in weight_names)


# Back-compat alias
_model_cached = model_cached
