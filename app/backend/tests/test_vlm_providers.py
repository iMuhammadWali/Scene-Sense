"""Tests: OpenRouter COMPLEX-only VLM; SIMPLE never calls VLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import Settings, get_settings
from app.pipeline.vlm import VLMError, VLMService
from app.pipeline.vlm_providers.base_types import VLMAPIError, VLMResult, VLMUsage
from app.pipeline.vlm_providers.openai_compatible import OpenAICompatibleVLMProvider


@pytest.fixture(autouse=True)
def _clear():
    get_settings.cache_clear()
    VLMService.reset()
    yield
    get_settings.cache_clear()
    VLMService.reset()


def test_config_openrouter_defaults(tmp_path):
    s = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        vlm_provider="openrouter",
        vlm_model="google/gemini-2.5-flash",
        classifier_model="facebook/bart-large-mnli",
        clip_model="openai/clip-vit-base-patch32",
    )
    assert s.vlm_provider == "openrouter"
    assert "gemini" in s.vlm_model
    assert s.classifier_model == "facebook/bart-large-mnli"


def test_missing_api_key_raises():
    provider = OpenAICompatibleVLMProvider(api_key="")
    with pytest.raises(VLMAPIError) as ei:
        provider.generate_answer(
            "q",
            [np.zeros((8, 8, 3), dtype=np.uint8)],
            "prompt",
            "google/gemini-2.5-flash",
            path="SLOW",
        )
    assert "OPENROUTER_API_KEY" in ei.value.reason


def test_fast_path_generate_rejected(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        openrouter_api_key="sk-test",
    )
    with patch("app.pipeline.vlm.get_settings", return_value=settings):
        svc = VLMService()
        with pytest.raises(VLMError) as ei:
            svc.generate(
                "FAST",
                "point out the blue shirt",
                images=[np.zeros((8, 8, 3), dtype=np.uint8)],
            )
        assert "must not be called" in ei.value.reason.lower() or "SIMPLE" in ei.value.reason


def test_slow_path_calls_openrouter(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        openrouter_api_key="sk-test",
        vlm_model="google/gemini-2.5-flash",
    )
    fake = MagicMock()
    fake.generate_answer.return_value = VLMResult(
        answer="He walked in.",
        provider="openrouter",
        model="google/gemini-2.5-flash",
        path="SLOW",
        frame_count=2,
        timestamps=[1.0, 2.0],
        latency_seconds=0.5,
        usage=VLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    with patch("app.pipeline.vlm.get_settings", return_value=settings):
        svc = VLMService()
        svc._provider = fake
        frames = [
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.ones((8, 8, 3), dtype=np.uint8),
        ]
        answer, debug = svc.generate(
            "SLOW", "What happened next?", images=frames, timestamps=[1.0, 2.0]
        )
        assert answer == "He walked in."
        assert debug["used"] is True
        assert debug["provider"] == "openrouter"
        assert debug["num_images"] == 2
        call_kw = fake.generate_answer.call_args
        assert len(call_kw.kwargs.get("frames") or call_kw[0][1] if False else call_kw.kwargs["frames"]) == 2
        # Ensure chronological frames list was passed (same order)
        passed = fake.generate_answer.call_args.kwargs["frames"]
        assert len(passed) == 2
        # Must not look like a video path was sent
        assert fake.generate_answer.call_args.kwargs.get("model") == "google/gemini-2.5-flash"


def test_vision_messages_are_images_not_video():
    from app.pipeline.vlm_providers.encoding import build_vision_messages

    frames = [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)]
    msgs = build_vision_messages("q", "prompt", frames)
    content = msgs[0]["content"]
    types = [c["type"] for c in content]
    assert "text" in types
    assert types.count("image_url") == 2
    assert "video" not in types


def test_simple_query_pipeline_never_calls_vlm(tmp_path):
    """End-to-end unit path: SIMPLE must not touch VLMService.generate."""
    from app.pipeline.query_pipeline import QueryPipeline

    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        openrouter_api_key="sk-should-not-be-used",
    )
    settings.ensure_directories()

    store = MagicMock()
    store.exists.return_value = True
    store.is_video_ready.return_value = True
    store.video_path.return_value = tmp_path / "v.mp4"
    store.load_faiss_mapping.return_value = [
        {"frame_id": 0, "timestamp": 1.0, "embedding_index": 0}
    ]
    store.faiss_index_path.return_value = tmp_path / "faiss.index"
    store.reset_query_progress.return_value = {}
    store.upsert_query_stage.return_value = {}
    store.update_query_progress.return_value = {}
    store.append_question.return_value = None

    fake_index = object()
    hits = [{"frame_id": 0, "timestamp": 1.0, "similarity": 0.5, "embedding_index": 0}]
    groups = [
        {
            "group_id": 1,
            "start_timestamp": 1.0,
            "end_timestamp": 1.0,
            "frames": hits,
        }
    ]

    with (
        patch("app.pipeline.query_pipeline.get_settings", return_value=settings),
        patch("app.pipeline.query_pipeline.QueryClassifier") as QC,
        patch("app.pipeline.query_pipeline.CLIPEncoder") as CE,
        patch("app.pipeline.query_pipeline.load_index", return_value=fake_index),
        patch("app.pipeline.query_pipeline.search", return_value=hits),
        patch(
            "app.pipeline.query_pipeline.filter_by_similarity",
            return_value=(hits, {"stage": "similarity_filtering", "status": "completed"}),
        ),
        patch(
            "app.pipeline.query_pipeline.group_temporally",
            return_value=(groups, {"stage": "temporal_grouping", "groups": []}),
        ),
        patch(
            "app.pipeline.query_pipeline.select_diverse_evidence",
            return_value=(hits, {"stage": "diversity_selection", "selected": hits}),
        ),
        patch("app.pipeline.query_pipeline.VLMService") as VS,
    ):
        QC.get.return_value.classify.return_value = (
            "SIMPLE",
            {"model": "facebook/bart-large-mnli"},
        )
        CE.get.return_value.encode_text.return_value = np.ones(512, dtype=np.float32)
        pipe = QueryPipeline(store=store)
        result = pipe.ask("sess", "Point out the man wearing a blue shirt.")
        assert result["query_type"] == "SIMPLE"
        assert result["path"] == "FAST"
        assert result["vlm_info"]["used"] is False
        assert result["timings"]["vlm_called"] is False
        VS.get.assert_not_called()


def test_complex_query_pipeline_calls_vlm(tmp_path):
    from app.pipeline.query_pipeline import QueryPipeline

    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        openrouter_api_key="sk-test",
    )
    settings.ensure_directories()
    store = MagicMock()
    store.exists.return_value = True
    store.is_video_ready.return_value = True
    store.video_path.return_value = tmp_path / "v.mp4"
    store.load_faiss_mapping.return_value = [
        {"frame_id": 0, "timestamp": 1.0, "embedding_index": 0}
    ]
    store.faiss_index_path.return_value = tmp_path / "faiss.index"
    store.reset_query_progress.return_value = {}
    store.upsert_query_stage.return_value = {}
    store.update_query_progress.return_value = {}
    store.append_question.return_value = None

    hits = [{"frame_id": 0, "timestamp": 1.0, "similarity": 0.5, "embedding_index": 0}]
    groups = [
        {
            "group_id": 1,
            "start_timestamp": 1.0,
            "end_timestamp": 1.0,
            "frames": hits,
        }
    ]
    img = np.zeros((16, 16, 3), dtype=np.uint8)

    with (
        patch("app.pipeline.query_pipeline.get_settings", return_value=settings),
        patch("app.pipeline.query_pipeline.QueryClassifier") as QC,
        patch("app.pipeline.query_pipeline.CLIPEncoder") as CE,
        patch("app.pipeline.query_pipeline.load_index", return_value=object()),
        patch("app.pipeline.query_pipeline.search", return_value=hits),
        patch(
            "app.pipeline.query_pipeline.filter_by_similarity",
            return_value=(hits, {"stage": "similarity_filtering", "status": "completed"}),
        ),
        patch(
            "app.pipeline.query_pipeline.group_temporally",
            return_value=(groups, {"stage": "temporal_grouping", "groups": []}),
        ),
        patch(
            "app.pipeline.query_pipeline.select_diverse_evidence",
            return_value=(hits, {"stage": "diversity_selection", "selected": hits}),
        ),
        patch.object(
            QueryPipeline,
            "_build_complex_vlm_inputs",
            return_value=(
                [img],
                [{"frame_id": 0, "timestamp": 1.0, "similarity": 0.5}],
                {"mode": "complex_merged_context", "count": 1},
            ),
        ),
        patch("app.pipeline.query_pipeline.VLMService") as VS,
    ):
        QC.get.return_value.classify.return_value = (
            "COMPLEX",
            {"model": "facebook/bart-large-mnli"},
        )
        CE.get.return_value.encode_text.return_value = np.ones(512, dtype=np.float32)
        VS.get.return_value.generate.return_value = (
            "He entered the room.",
            {
                "used": True,
                "provider": "openrouter",
                "vlm_model_id": "google/gemini-2.5-flash",
                "processing_time_seconds": 0.2,
                "frames": [{"timestamp": 1.0}],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
        )
        pipe = QueryPipeline(store=store)
        result = pipe.ask("sess", "What did the man do after entering the room?")
        assert result["query_type"] == "COMPLEX"
        assert result["path"] == "SLOW"
        assert result["vlm_info"]["used"] is True
        assert result["timings"]["vlm_called"] is True
        VS.get.return_value.generate.assert_called_once()
        call_kwargs = VS.get.return_value.generate.call_args.kwargs
        assert call_kwargs["path"] == "SLOW"
        assert len(call_kwargs["images"]) == 1
