"""Pretrained zero-shot SIMPLE/COMPLEX query classifier (no training)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from app.config import get_settings
from app.utils.hardware import resolve_device
from app.utils.logging_setup import get_logger
from app.utils.timing import timed

logger = get_logger("scenesense.classifier")

# Output-modality distinction (not single-frame vs multi-frame):
# SIMPLE = retrieve visual evidence only; COMPLEX = natural-language answer via VLM.
LABEL_SIMPLE = (
    "A visual retrieval request where the user only needs relevant image or "
    "video-frame evidence shown or located, without a natural-language "
    "explanation or answer."
)
LABEL_COMPLEX = (
    "A visual question where the user expects a natural-language answer "
    "requiring visual understanding, description, reasoning, counting, "
    "action recognition, temporal understanding, or aggregation across the video."
)
HYPOTHESIS_TEMPLATE = "This example is {}."


class QueryClassifier:
    """Zero-shot text classification using a pretrained NLI model."""

    _instance: Optional["QueryClassifier"] = None

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.classifier_model
        self.device = device or resolve_device(settings.device_preference)
        self.cache_dir = str(settings.hf_cache_dir)
        self.pipe = None
        self._loaded = False

    @classmethod
    def get(cls) -> "QueryClassifier":
        if cls._instance is None:
            cls._instance = QueryClassifier()
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        from transformers import pipeline

        logger.info(
            "[CLASSIFIER] Loading model=%s device=%s", self.model_name, self.device
        )
        device_idx = 0 if self.device == "cuda" else -1
        with timed() as t:
            self.pipe = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=device_idx,
                model_kwargs={"cache_dir": self.cache_dir},
            )
        self._loaded = True
        logger.info("[CLASSIFIER] Loaded in %.2fs", t[0])

    def classify(self, question: str) -> tuple[Literal["SIMPLE", "COMPLEX"], dict[str, Any]]:
        self.load()
        assert self.pipe is not None
        with timed() as t:
            result = self.pipe(
                question,
                candidate_labels=[LABEL_SIMPLE, LABEL_COMPLEX],
                multi_label=False,
                hypothesis_template=HYPOTHESIS_TEMPLATE,
            )
            labels = list(result["labels"])
            scores = list(result["scores"])
            top = labels[0]
            query_type: Literal["SIMPLE", "COMPLEX"] = (
                "COMPLEX" if top == LABEL_COMPLEX else "SIMPLE"
            )
            score_map = {
                lab: round(float(sc), 4) for lab, sc in zip(labels, scores)
            }
            info = {
                "stage": "query_classification",
                "status": "completed",
                "result": query_type,
                "classification": query_type,
                "model": self.model_name,
                "labels": labels,
                "scores": [round(float(s), 4) for s in scores],
                "simple_confidence": score_map.get(LABEL_SIMPLE),
                "complex_confidence": score_map.get(LABEL_COMPLEX),
                "label_definitions": {
                    "SIMPLE": LABEL_SIMPLE,
                    "COMPLEX": LABEL_COMPLEX,
                },
                "processing_time_seconds": t[0],
                "routing_rule": "SIMPLE->FAST, COMPLEX->SLOW (retrieval cannot override)",
            }
        logger.info(
            "[CLASSIFIER] q=%r => %s (%.2fs)",
            question[:80],
            query_type,
            t[0],
        )
        return query_type, info

    def status(self) -> dict:
        return {
            "name": self.model_name,
            "loaded": self._loaded,
            "device": self.device,
            "role": "zero-shot SIMPLE/COMPLEX classifier",
        }
