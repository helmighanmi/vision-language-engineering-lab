# Path: src/vlm_engineering/extraction/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Document extraction capabilities (NuExtract3).

NuExtract3 is a Qwen3.5-based vision-language model purpose-built for
document understanding. It is deliberately kept separate from the
``Qwen3-VL Instruct`` presets because it is fine-tuned for two specific tasks:

* template-guided structured JSON extraction;
* full-document content extraction (Markdown / OCR-like transcription).

The integration is backend-agnostic: it talks to any generator that implements
the small :class:`NuExtract3Generator` protocol, including the existing
``QwenVLMProvider`` and ``OpenAICompatibleVLMProvider`` (LM Studio, vLLM,
Ollama, ...).
"""

from __future__ import annotations

from .nuextract3 import (
    NUEXTRACT3_MODEL,
    NuExtract3Extractor,
    NuExtract3Generator,
    NuExtract3ModelPreset,
    create_nuextract3_extractor,
    load_extraction_template,
)

__all__ = [
    "NUEXTRACT3_MODEL",
    "NuExtract3Extractor",
    "NuExtract3Generator",
    "NuExtract3ModelPreset",
    "create_nuextract3_extractor",
    "load_extraction_template",
]
