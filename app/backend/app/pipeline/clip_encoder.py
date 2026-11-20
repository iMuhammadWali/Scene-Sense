"""Shared CLIP image/text encoder (same embedding space)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from app.config import get_settings
from app.utils.hardware import resolve_device
from app.utils.logging_setup import get_logger
from app.utils.math_ops import l2_normalize
from app.utils.timing import timed

logger = get_logger("scenesense.clip")


def _as_feature_tensor(output: Any, projection: Optional[torch.nn.Module] = None) -> torch.Tensor:
    """
    Normalize CLIP feature outputs across Transformers versions.

    Transformers 4.x: get_*_features returns a Tensor that is ALREADY projected
    into the shared image-text space.
    Transformers >=5.x: may return BaseModelOutputWithPooling whose pooler_output
    is NOT yet projected.

    The projection head is applied only to the un-projected pooler_output.
    (Bug fix: the old code decided by checking feats.shape[-1] == in_features.
    For ViT-B/32 the text width and the projected size are both 512, so text
    embeddings that were already projected got projected a second time and no
    longer lined up with the image embeddings.)
    """
    needs_projection = False
    if torch.is_tensor(output):
        feats = output  # already projected
    elif getattr(output, "image_embeds", None) is not None:
        feats = output.image_embeds  # already projected
    elif getattr(output, "text_embeds", None) is not None:
        feats = output.text_embeds  # already projected
    elif getattr(output, "pooler_output", None) is not None:
        feats = output.pooler_output  # pre-projection pooled hidden state
        needs_projection = True
    elif isinstance(output, (tuple, list)) and output:
        feats = output[0]
    else:
        raise TypeError(f"Unexpected CLIP feature output type: {type(output)}")

    if not torch.is_tensor(feats):
        raise TypeError(f"CLIP features are not a tensor: {type(feats)}")

    if needs_projection and projection is not None:
        feats = projection(feats)
    return feats


class CLIPEncoder:
    """Load CLIP once and reuse for image + text embeddings."""

    _instance: Optional["CLIPEncoder"] = None

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.clip_model
        self.device = device or resolve_device(settings.device_preference)
        self.cache_dir = str(settings.hf_cache_dir)
        self.model = None
        self.processor = None
        self._loaded = False

    @classmethod
    def get(cls) -> "CLIPEncoder":
        if cls._instance is None:
            cls._instance = CLIPEncoder()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def load(self) -> None:
        if self._loaded:
            return
        from transformers import CLIPModel, CLIPProcessor

        logger.info("[CLIP] Loading model=%s device=%s", self.model_name, self.device)
        with timed() as t:
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            self.model = CLIPModel.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
        self._loaded = True
        logger.info("[CLIP] Loaded in %.2fs", t[0])

    def is_ready(self) -> bool:
        return self._loaded and self.model is not None

    def encode_images(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """Encode RGB numpy images to L2-normalized CLIP embeddings."""
        self.load()
        assert self.model is not None and self.processor is not None
        all_emb: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                pil_batch = [Image.fromarray(img.astype(np.uint8)) for img in batch]
                inputs = self.processor(images=pil_batch, return_tensors="pt", padding=True)
                pixel_values = inputs["pixel_values"].to(self.device)
                raw = self.model.get_image_features(pixel_values=pixel_values)
                feats = _as_feature_tensor(raw, projection=self.model.visual_projection)
                all_emb.append(feats.detach().cpu().numpy().astype(np.float32))
        if not all_emb:
            return np.zeros((0, 512), dtype=np.float32)
        stacked = np.vstack(all_emb)
        return l2_normalize(stacked)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        return self.encode_images([image])[0]

    def encode_text(self, text: str) -> np.ndarray:
        """Encode query text with the SAME CLIP text encoder."""
        self.load()
        assert self.model is not None and self.processor is not None
        with torch.no_grad():
            inputs = self.processor(
                text=[text], return_tensors="pt", padding=True, truncation=True
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                raw = self.model.get_text_features(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            else:
                raw = self.model.get_text_features(input_ids=input_ids)
            feats = _as_feature_tensor(raw, projection=self.model.text_projection)
            emb = feats.detach().cpu().numpy().astype(np.float32)[0]
        return l2_normalize(emb)

    def status(self) -> dict:
        return {
            "name": self.model_name,
            "loaded": self._loaded,
            "device": self.device,
            "role": "image+text embeddings",
        }
