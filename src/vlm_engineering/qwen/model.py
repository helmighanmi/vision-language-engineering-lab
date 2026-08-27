# Path: src/vlm_engineering/qwen/model.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Qwen3-VL inference wrapper supporting Hub/cache and explicit local models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DEFAULT_QWEN_MODEL
from ..exceptions import ModelLoadError, OptionalDependencyError
from ..utils import to_image_reference


class QwenVLModel:
    """Run Qwen3-VL from a Hugging Face model ID or a local model directory.

    ``trust_remote_code`` defaults to ``False``. Current Qwen3-VL is integrated
    into Transformers and does not require remote custom Python code for normal use.
    """

    def __init__(
        self,
        model_source: str | Path = DEFAULT_QWEN_MODEL,
        *,
        local_files_only: bool = False,
        device_map: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.model_source = str(model_source)
        self.local_files_only = local_files_only
        self.device_map = device_map
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self._model = model
        self._processor = processor

    @classmethod
    def from_hub(cls, model_id: str = DEFAULT_QWEN_MODEL, **kwargs: Any) -> "QwenVLModel":
        """Create a Hub/cache-backed model loader."""
        return cls(model_id, local_files_only=False, **kwargs)

    @classmethod
    def from_local(cls, model_path: str | Path, **kwargs: Any) -> "QwenVLModel":
        """Create an offline loader that never falls back to the Hub."""
        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return cls(str(path), local_files_only=True, **kwargs)

    def _ensure_loaded(self) -> tuple[Any, Any]:
        """Load and return ``(model, processor)`` exactly once."""
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise OptionalDependencyError(
                'Install Qwen dependencies with: pip install -e ".[qwen]"'
            ) from exc

        common = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_source, **common)
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.model_source,
                device_map=self.device_map,
                dtype=self.dtype,
                **common,
            )
        except Exception as exc:  # pragma: no cover - backend/hardware specific
            raise ModelLoadError(f"Unable to load model from {self.model_source!r}: {exc}") from exc

        if self._model is None or self._processor is None:  # defensive guard
            raise ModelLoadError(f"Unable to load model from {self.model_source!r}.")

        return self._model, self._processor

    def generate(
        self,
        image: str | Path,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate a text response grounded in one image."""
        model, processor = self._ensure_loaded()
        image_ref = to_image_reference(image)
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_ref},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return processor.decode(generated, skip_special_tokens=True).strip()
