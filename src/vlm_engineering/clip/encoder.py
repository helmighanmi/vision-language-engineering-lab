# Path: src/vlm_engineering/clip/encoder.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Reusable CLIP encoder for text/image embeddings and zero-shot classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..config import DEFAULT_CLIP_MODEL
from ..exceptions import OptionalDependencyError


@dataclass(frozen=True, slots=True)
class ZeroShotPrediction:
    """One zero-shot classification result."""

    label: str
    probability: float


class CLIPEncoder:
    """Thin production wrapper around Hugging Face CLIP.

    Heavy ML dependencies are loaded lazily so imports and unit tests stay lightweight.
    A prebuilt model and processor may be injected for testing or custom runtimes.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_CLIP_MODEL,
        *,
        device: str | None = None,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = model
        self._processor = processor

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise OptionalDependencyError(
                'Install CLIP dependencies with: pip install -e ".[clip]"'
            ) from exc

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id).to(device)
        self.device = device

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(array, axis=-1, keepdims=True)
        return array / np.clip(denom, 1e-12, None)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return normalized CLIP text embeddings."""
        self._ensure_loaded()
        import torch

        inputs = self._processor(text=list(texts), padding=True, return_tensors="pt")
        if self.device:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            embeddings = self._model.get_text_features(**inputs)
        return self._normalize(embeddings.detach().cpu().numpy())

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        """Return normalized CLIP image embeddings."""
        self._ensure_loaded()
        import torch

        inputs = self._processor(images=list(images), return_tensors="pt")
        if self.device:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            embeddings = self._model.get_image_features(**inputs)
        return self._normalize(embeddings.detach().cpu().numpy())

    def similarity(self, texts: Sequence[str], images: Sequence[Any]) -> np.ndarray:
        """Return cosine-similarity matrix with shape (len(texts), len(images))."""
        return self.encode_text(texts) @ self.encode_images(images).T

    def zero_shot_classify(self, image: Any, labels: Sequence[str]) -> list[ZeroShotPrediction]:
        """Classify one image against arbitrary text labels without fine-tuning."""
        scores = self.similarity(labels, [image])[:, 0]
        exp = np.exp(scores - scores.max())
        probs = exp / exp.sum()
        results = [ZeroShotPrediction(label, float(prob)) for label, prob in zip(labels, probs)]
        return sorted(results, key=lambda item: item.probability, reverse=True)
