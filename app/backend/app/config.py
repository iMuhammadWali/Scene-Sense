"""Central configuration for SceneSense research MVP."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All tunable research and runtime parameters live here."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Paths
    data_dir: Path = Field(default=ROOT_DIR / "data", alias="SCENESENSE_DATA_DIR")
    models_dir: Path = Field(default=ROOT_DIR / "models", alias="SCENESENSE_MODELS_DIR")
    max_upload_mb: int = Field(default=500, alias="SCENESENSE_MAX_UPLOAD_MB")

    # Pipeline parameters
    frame_extraction_fps: float = Field(default=5.0, alias="FRAME_EXTRACTION_FPS")
    pruning_threshold: float = Field(default=0.95, alias="PRUNING_THRESHOLD")
    faiss_top_k: int = Field(default=20, alias="FAISS_TOP_K")
    retrieval_threshold: float = Field(default=0.20, alias="RETRIEVAL_THRESHOLD")
    temporal_group_gap_seconds: float = Field(
        default=2.0, alias="TEMPORAL_GROUP_GAP_SECONDS"
    )
    complex_context_window_seconds: float = Field(
        default=5.0, alias="COMPLEX_CONTEXT_WINDOW_SECONDS"
    )
    complex_sampling_fps: float = Field(default=1.0, alias="COMPLEX_SAMPLING_FPS")

    # Local models
    clip_model: str = Field(
        default="openai/clip-vit-base-patch32", alias="CLIP_MODEL"
    )
    classifier_model: str = Field(
        default="facebook/bart-large-mnli",
        alias="CLASSIFIER_MODEL",
    )

    # OpenRouter VLM — COMPLEX / Slow path ONLY (SIMPLE never uses VLM)
    vlm_provider: str = Field(default="openrouter", alias="VLM_PROVIDER")
    vlm_model: str = Field(
        default="google/gemini-2.5-flash",
        alias="VLM_MODEL",
    )
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    vlm_max_new_tokens: int = Field(default=256, alias="VLM_MAX_NEW_TOKENS")
    vlm_timeout_seconds: float = Field(default=90.0, alias="VLM_TIMEOUT_SECONDS")

    # Deprecated aliases ignored if present (extra=ignore already)
    # VLM_MODE=local is no longer used

    device_preference: str = Field(default="auto", alias="SCENESENSE_DEVICE")
    host: str = Field(default="127.0.0.1", alias="SCENESENSE_HOST")
    port: int = Field(default=8000, alias="SCENESENSE_PORT")
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="SCENESENSE_CORS_ORIGINS",
    )

    @property
    def videos_dir(self) -> Path:
        return self._abs_data_dir / "videos"

    @property
    def sessions_dir(self) -> Path:
        return self._abs_data_dir / "sessions"

    @property
    def hf_cache_dir(self) -> Path:
        return self._abs_models_dir / "hf_cache"

    @property
    def _abs_data_dir(self) -> Path:
        p = self.data_dir
        return p.resolve() if p.is_absolute() else (ROOT_DIR / p).resolve()

    @property
    def _abs_models_dir(self) -> Path:
        p = self.models_dir
        return p.resolve() if p.is_absolute() else (ROOT_DIR / p).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_directories(self) -> None:
        data = self._abs_data_dir
        models = self._abs_models_dir
        data.mkdir(parents=True, exist_ok=True)
        (data / "videos").mkdir(parents=True, exist_ok=True)
        (data / "sessions").mkdir(parents=True, exist_ok=True)
        models.mkdir(parents=True, exist_ok=True)
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.hf_cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.hf_cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.hf_cache_dir))


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
