# Path: src/vlm_engineering/qwen/model.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Qwen3-VL inference wrapper supporting presets, Hub/cache and local models."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..exceptions import ModelLoadError, OptionalDependencyError
from .registry import resolve_qwen_model_id


def _normalize_image_source(image: str | Path) -> str:
    """Normalize an image source for the Hugging Face processor.

    Local images are passed as absolute filesystem paths rather than ``file://``
    URIs because the Transformers image-loading backend expects a normal local
    path, an HTTP(S) URL, or an encoded image payload.

    HTTP(S) URLs are preserved unchanged.

    ``file://`` URIs are accepted for backwards compatibility and converted
    back into normal filesystem paths.
    """

    if isinstance(image, Path):
        path = image.expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(path)

        if not path.is_file():
            raise ValueError(f"Image path is not a file: {path}")

        return str(path)

    value = image.strip()

    if not value:
        raise ValueError("image must not be empty.")

    if value.startswith(("http://", "https://")):
        return value

    if value.startswith("file://"):
        parsed = urlparse(value)
        path = Path(unquote(parsed.path)).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(path)

        if not path.is_file():
            raise ValueError(f"Image path is not a file: {path}")

        return str(path)

    candidate = Path(value).expanduser()

    try:
        if candidate.exists():
            path = candidate.resolve()

            if not path.is_file():
                raise ValueError(f"Image path is not a file: {path}")

            return str(path)
    except OSError:
        # The value may be an encoded image payload rather than a filesystem
        # path. Leave backend-specific validation to Transformers.
        pass

    # Preserve non-path values such as supported encoded image payloads.
    return value


class QwenVLModel:
    """Run Qwen3-VL from a preset, Hugging Face model ID, or local directory.

    With no arguments the package uses ``Qwen/Qwen3-VL-2B-Instruct``.
    Friendly ``model_size`` aliases are available for ``2b``, ``4b`` and ``8b``.

    ``trust_remote_code`` defaults to ``False``. Current Qwen3-VL is integrated
    into Transformers and does not require remote custom Python code for normal
    use.
    """

    def __init__(
        self,
        model_source: str | Path | None = None,
        *,
        model_size: str | None = None,
        local_files_only: bool = False,
        device_map: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        if model_source is not None and model_size is not None:
            raise ValueError("Provide either model_source or model_size, not both.")

        if model_source is None:
            resolved_source = resolve_qwen_model_id(model_size=model_size)
        else:
            resolved_source = str(model_source)

            if not resolved_source.strip():
                raise ValueError("model_source must not be empty.")

        self.model_source = resolved_source
        self.local_files_only = local_files_only
        self.device_map = device_map
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self._model = model
        self._processor = processor

    @classmethod
    def from_preset(
        cls,
        model_size: str = "2b",
        **kwargs: Any,
    ) -> "QwenVLModel":
        """Create a loader from the supported ``2b``, ``4b`` or ``8b`` presets."""
        return cls(
            model_size=model_size,
            local_files_only=False,
            **kwargs,
        )

    @classmethod
    def from_hub(
        cls,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "QwenVLModel":
        """Create a Hub/cache-backed loader for any compatible model ID."""
        return cls(
            resolve_qwen_model_id(model_id=model_id),
            local_files_only=False,
            **kwargs,
        )

    @classmethod
    def from_local(
        cls,
        model_path: str | Path,
        **kwargs: Any,
    ) -> "QwenVLModel":
        """Create an offline loader that never falls back to the Hub."""
        path = Path(model_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(path)

        if not path.is_dir():
            raise ValueError(f"Model path is not a directory: {path}")

        return cls(
            str(path),
            local_files_only=True,
            **kwargs,
        )

    def _ensure_loaded(self) -> tuple[Any, Any]:
        """Load and return ``(model, processor)`` exactly once."""
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise OptionalDependencyError(
                'Install Qwen dependencies with: pip install "vision-language-engineering-lab[qwen]"'
            ) from exc

        common = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_source,
                **common,
            )

            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.model_source,
                device_map=self.device_map,
                dtype=self.dtype,
                **common,
            )

        except Exception as exc:  # pragma: no cover - backend/hardware specific
            raise ModelLoadError(
                f"Unable to load model from {self.model_source!r}: {exc}"
            ) from exc

        if self._model is None or self._processor is None:
            raise ModelLoadError(
                f"Unable to load model from {self.model_source!r}."
            )

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
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")

        image_source = _normalize_image_source(image)

        model, processor = self._ensure_loaded()

        messages: list[dict[str, Any]] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                        }
                    ],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": image_source,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
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

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

        generated = outputs[0][inputs["input_ids"].shape[-1] :]

        return processor.decode(
            generated,
            skip_special_tokens=True,
        ).strip()