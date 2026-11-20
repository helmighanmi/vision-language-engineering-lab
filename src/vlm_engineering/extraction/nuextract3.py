# Path: src/vlm_engineering/extraction/nuextract3.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""NuExtract3 template-guided document extraction.

NuExtract3 (``numind/NuExtract3``) is a ~4B-parameter Qwen3.5-based VLM
specialised for document understanding. Unlike the general-purpose
``Qwen3-VL Instruct`` presets in this package, NuExtract3 expects a *template*
that describes the exact JSON shape to return, plus optional free-form
instructions. It also supports a second "content extraction" mode that
transcribes a page to Markdown.

This module does **not** download or load model weights. It composes the
prompt and validates the output; generation is delegated to an injected
generator. That keeps the module usable with:

* :class:`~vlm_engineering.providers.openai_compatible.OpenAICompatibleVLMProvider`
  backed by LM Studio, vLLM, Ollama or any ``/v1/chat/completions`` server;
* :class:`~vlm_engineering.providers.qwen.QwenVLMProvider` for a local
  Transformers-based Qwen-compatible model;
* a simple fake generator in tests.

Example
-------
>>> from vlm_engineering.extraction import NuExtract3Extractor, create_nuextract3_extractor
>>> extractor = create_nuextract3_extractor(
...     base_url="http://localhost:1234/v1",
...     model="NuExtract3",
... )
>>> data = extractor.extract(
...     ["http://localhost:1234/screenshots/invoice.png"],
...     {"invoice_number": "", "total": 0.0, "line_items": []},
...     instructions="Return amounts as numbers without currency symbols.",
... )

.. note::
   When serving a Qwen-VL-family GGUF through llama-server / LM Studio, set
   the server option ``--image-min-tokens`` to a positive value (for example
   ``--image-min-tokens 8192``). Without it, decimal numbers in extracted
   fields can be silently wrong (a known NuExtract3 / Qwen-VL GGUF gotcha).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..exceptions import InputValidationError, StructuredOutputError
from ..utils import extract_json_object

__all__ = [
    "NUEXTRACT3_MODEL",
    "NuExtract3Extractor",
    "NuExtract3Generator",
    "NuExtract3ModelPreset",
    "create_nuextract3_extractor",
    "load_extraction_template",
]


# ---------------------------------------------------------------------------
# Model metadata (kept separate from the Qwen3-VL Instruct registry)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NuExtract3ModelPreset:
    """Metadata for the NuExtract3 extraction model.

    This is intentionally *not* a :class:`~vlm_engineering.qwen.registry.QwenModelPreset`
    entry: NuExtract3 is Qwen3.5-based and fine-tuned for extraction, so it is
    not interchangeable with the 2B/4B/8B Qwen3-VL Instruct generative presets.
    """

    model_id: str
    gguf_model_id: str
    parameter_class: str
    description: str
    default_max_new_tokens: int
    recommended_image_min_tokens: int


NUEXTRACT3_MODEL = NuExtract3ModelPreset(
    model_id="numind/NuExtract3",
    gguf_model_id="numind/NuExtract3-GGUF",
    parameter_class="~4B",
    description=(
        "Qwen3.5-based document-extraction VLM; template-guided JSON extraction "
        "and Markdown content extraction, local GGUF / LM Studio / vLLM / Ollama."
    ),
    default_max_new_tokens=2048,
    recommended_image_min_tokens=8192,
)


# ---------------------------------------------------------------------------
# Generator contract
# ---------------------------------------------------------------------------


@runtime_checkable
class NuExtract3Generator(Protocol):
    """Minimal generation interface required by :class:`NuExtract3Extractor`.

    The signature mirrors ``VLMProvider.generate`` so the existing Qwen and
    OpenAI-compatible providers satisfy this protocol without modification.
    """

    def generate(
        self,
        images: Sequence[str | Path],
        prompt: str,
        *,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

STRUCTURED_SYSTEM_PROMPT = """You are NuExtract3, a document extraction engine.
You receive a document (image and/or text) together with a JSON template and optional instructions.
Return ONLY one valid JSON object that follows the template exactly.
Rules:
- Preserve keys from the template; do not add keys that are not in the template.
- Use null for values that are not present or unreadable in the document.
- Use numbers (not strings) for numeric fields, and never include currency symbols in numbers.
- Do not wrap the JSON in Markdown fences and do not add any commentary.
"""

CONTENT_SYSTEM_PROMPT = """You are NuExtract3, a document content extraction engine.
Transcribe ALL visible content of the document into clean Markdown.
Rules:
- Preserve headings, lists, tables, equations and reading order.
- Keep exact identifiers, numbers and labels; never invent or summarise content.
- Use fenced code blocks for code and math where appropriate.
- Return only the Markdown document, without commentary.
"""


def _build_structured_prompt(
    template: Mapping[str, Any],
    instructions: str | None,
    *,
    enable_thinking: bool,
) -> str:
    template_json = json.dumps(template, ensure_ascii=False, indent=2)
    parts = [
        "### Template\n" + template_json,
    ]
    if instructions:
        parts.append("### Instructions\n" + instructions.strip())
    if enable_thinking:
        parts.append(
            "Think carefully about the document before answering, "
            "but output only the final JSON object."
        )
    parts.append("### Output\nThe JSON object:")
    return "\n\n".join(parts)


def _build_content_prompt(instructions: str | None, *, enable_thinking: bool) -> str:
    parts: list[str] = []
    if instructions:
        parts.append("### Instructions\n" + instructions.strip())
    if enable_thinking:
        parts.append(
            "Read the document carefully before answering, "
            "but output only the final Markdown content."
        )
    parts.append("### Output\nThe Markdown content:")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


def load_extraction_template(template: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Normalize an extraction template from a dict, JSON string, or file path.

    Args:
        template:
            A mapping describing the desired JSON shape, a JSON object string,
            or a path to a ``.json`` file containing a JSON object.

    Returns:
        A plain ``dict`` copy of the template.

    Raises:
        InputValidationError:
            If the template is empty, not a JSON object, or an unreadable path.
    """
    if isinstance(template, Mapping):
        data: Any = dict(template)
    elif isinstance(template, (str, Path)):
        raw = str(template)
        looks_like_json = raw.lstrip().startswith(("{", "["))
        if looks_like_json:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise InputValidationError(f"Invalid JSON template string: {exc}") from exc
        else:
            path = Path(raw).expanduser()
            if not path.is_file():
                # A bare string that is not a path or JSON is not a valid template.
                raise InputValidationError(
                    "template must be a JSON object, a JSON string, or an existing file path."
                )
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise InputValidationError(f"Could not load template from {path}: {exc}") from exc
    else:
        raise InputValidationError(
            "template must be a mapping, a JSON string, or a path to a JSON file."
        )

    if not isinstance(data, dict):
        raise InputValidationError("Extraction template must be a JSON object.")
    if not data:
        raise InputValidationError("Extraction template must not be empty.")
    return data


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class NuExtract3Extractor:
    """Template-guided NuExtract3 extraction over an injected generator.

    Args:
        generator:
            Any object implementing :class:`NuExtract3Generator`.
        preset:
            Metadata describing the model; defaults to :data:`NUEXTRACT3_MODEL`.
        max_new_tokens:
            Default generation budget, overridable per call.
        retries:
            Number of bounded retries when the model returns malformed JSON
            (default 1, i.e. at most two attempts).
    """

    def __init__(
        self,
        generator: NuExtract3Generator,
        *,
        preset: NuExtract3ModelPreset = NUEXTRACT3_MODEL,
        max_new_tokens: int | None = None,
        retries: int = 1,
    ) -> None:
        if not isinstance(generator, NuExtract3Generator):
            raise InputValidationError("generator must implement generate(images, prompt, ...).")
        if type(retries) is not int or retries < 0:
            raise InputValidationError("retries must be a non-negative integer.")
        budget = max_new_tokens if max_new_tokens is not None else preset.default_max_new_tokens
        if type(budget) is not int or budget <= 0:
            raise InputValidationError("max_new_tokens must be a positive integer.")
        self.generator = generator
        self.preset = preset
        self.max_new_tokens = budget
        self.retries = retries

    # -- helpers -----------------------------------------------------------

    def _normalize_images(self, images: Sequence[str | Path] | str | Path) -> list[str | Path]:
        if isinstance(images, (str, Path)):
            images = [images]
        images = list(images)
        if not images:
            raise InputValidationError("At least one image or document input is required.")
        return images

    @staticmethod
    def _normalize_instructions(instructions: str | None) -> str | None:
        if instructions is None:
            return None
        if not isinstance(instructions, str):
            raise InputValidationError("instructions must be a string or None.")
        stripped = instructions.strip()
        return stripped or None

    # -- public API --------------------------------------------------------

    def extract(
        self,
        images: Sequence[str | Path] | str | Path,
        template: Mapping[str, Any] | str | Path,
        *,
        instructions: str | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int | None = None,
        retries: int | None = None,
    ) -> dict[str, Any]:
        """Extract structured JSON matching ``template``.

        The model response is parsed with
        :func:`~vlm_engineering.utils.extract_json_object`, which tolerates
        Markdown fences and surrounding prose. Malformed output triggers a
        bounded retry before :class:`StructuredOutputError` is raised.

        Args:
            images: One or more image paths or URLs (or a single one).
            template: JSON shape described as a dict, JSON string, or file path.
            instructions: Optional task-specific guidance.
            enable_thinking: Ask the model to reason before emitting the JSON.
            max_new_tokens: Per-call generation budget override.
            retries: Per-call retry override.

        Returns:
            The parsed JSON object as a ``dict``.
        """
        resolved_images = self._normalize_images(images)
        resolved_template = load_extraction_template(template)
        resolved_instructions = self._normalize_instructions(instructions)
        attempt_budget = self.retries if retries is None else retries
        if type(attempt_budget) is not int or attempt_budget < 0:
            raise InputValidationError("retries must be a non-negative integer.")
        tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if type(tokens) is not int or tokens <= 0:
            raise InputValidationError("max_new_tokens must be a positive integer.")

        prompt = _build_structured_prompt(
            resolved_template, resolved_instructions, enable_thinking=enable_thinking
        )
        last_error: Exception | None = None
        for _attempt in range(attempt_budget + 1):
            raw = self.generator.generate(
                resolved_images,
                prompt,
                max_new_tokens=tokens,
                system_prompt=STRUCTURED_SYSTEM_PROMPT,
            )
            try:
                data = extract_json_object(raw)
                if not isinstance(data, dict):
                    raise StructuredOutputError("Structured extraction must be a JSON object.")
                return data
            except (StructuredOutputError, ValueError, TypeError) as exc:
                last_error = exc

        raise StructuredOutputError(
            f"Invalid structured extraction after {attempt_budget + 1} attempt(s): {last_error}"
        ) from last_error

    def extract_markdown(
        self,
        images: Sequence[str | Path] | str | Path,
        *,
        instructions: str | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int | None = None,
    ) -> str:
        """Extract all document content as clean Markdown.

        This is NuExtract3's second core capability (content extraction /
        OCR-like transcription). No JSON parsing is performed.
        """
        resolved_images = self._normalize_images(images)
        resolved_instructions = self._normalize_instructions(instructions)
        tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if type(tokens) is not int or tokens <= 0:
            raise InputValidationError("max_new_tokens must be a positive integer.")
        prompt = _build_content_prompt(resolved_instructions, enable_thinking=enable_thinking)
        raw = self.generator.generate(
            resolved_images,
            prompt,
            max_new_tokens=tokens,
            system_prompt=CONTENT_SYSTEM_PROMPT,
        )
        if not isinstance(raw, str) or not raw.strip():
            raise StructuredOutputError("Content extraction returned an empty response.")
        return raw.strip()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_nuextract3_extractor(
    *,
    base_url: str,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    max_new_tokens: int | None = None,
    retries: int = 1,
) -> NuExtract3Extractor:
    """Build a :class:`NuExtract3Extractor` on an OpenAI-compatible server.

    This targets the servers used by the NuExtract3 application project:

    * LM Studio: ``http://localhost:1234/v1``
    * vLLM:       ``http://localhost:8000/v1``
    * Ollama:     ``http://localhost:11434/v1``

    Args:
        base_url: Base URL of the OpenAI-compatible API (including ``/v1``).
        model: Served model name/alias. Defaults to ``"NuExtract3"``; replace
            it with whatever alias your local server exposes (for example
            ``"numind/NuExtract3-GGUF"``).
        api_key: Optional API key for the local server.
        timeout: HTTP timeout in seconds (extraction on CPU/GPU can be slow).
        max_new_tokens: Default generation budget.
        retries: Bounded JSON parse retries.

    .. note::
       ``OpenAICompatibleVLMProvider`` requires images that the *server* can
       reach. Use HTTP URLs, or local paths only if the server process can
       read the same filesystem. For local files served by LM Studio you can
       start it with ``--served-model-name NuExtract3 --image-min-tokens 8192``
       and pass a reachable URL for each image.
    """
    from ..providers.openai_compatible import OpenAICompatibleVLMProvider

    served_model = (model or "NuExtract3").strip()
    if not served_model:
        raise InputValidationError("model must be a non-empty string when provided.")
    provider = OpenAICompatibleVLMProvider(
        base_url=base_url,
        model=served_model,
        api_key=api_key,
        timeout=timeout,
    )
    return NuExtract3Extractor(
        provider, max_new_tokens=max_new_tokens, retries=retries
    )
