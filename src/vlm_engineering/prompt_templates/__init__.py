# Path: src/vlm_engineering/prompt_templates/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Domain-specific prompt templates for VLM analysis and RAG."""

from .templates import (
    DIAGRAM_PROMPT,
    RAG_ANSWER_PROMPT,
    TABLE_PROMPT,
    TEXT_PROMPT,
    prompt_for_page_type,
)

__all__ = [
    "DIAGRAM_PROMPT",
    "RAG_ANSWER_PROMPT",
    "TABLE_PROMPT",
    "TEXT_PROMPT",
    "prompt_for_page_type",
]
