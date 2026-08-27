# Path: src/vlm_engineering/documents/visual_analysis.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Structured VLM prompting for diagrams, screenshots, tables and mixed pages."""

from __future__ import annotations

from typing import Protocol

from ..exceptions import StructuredOutputError
from ..utils import extract_json_object
from .schemas import VisualAnalysis, VisualRelation

VISUAL_RAG_PROMPT = """You analyze a technical document image for a future RAG system.
Return ONLY one valid JSON object with this schema:
{
  "page_type": "diagram|table|text|mixed|photo|screenshot",
  "title": "short title",
  "summary": "standalone factual summary",
  "entities": ["exact entity labels"],
  "relations": [
    {"source": "...", "relation": "...", "target": "...", "evidence": "..."}
  ],
  "important_text": ["exact text worth preserving"],
  "uncertainties": ["anything unreadable or ambiguous"]
}
Rules:
- Do not invent an unreadable arrow or relationship.
- Preserve exact component names, identifiers and labels when visible.
- Explain spatial or directional relations that OCR alone would lose.
- Keep the summary useful as a text-search representation of the image.
"""


class VisionGenerator(Protocol):
    def generate(self, image: str, prompt: str, *, max_new_tokens: int = 512, **kwargs: object) -> str: ...


def analyze_visual_document(generator: VisionGenerator, image_ref: str) -> VisualAnalysis:
    """Run structured VLM analysis and validate the response into typed objects."""
    raw = generator.generate(image_ref, VISUAL_RAG_PROMPT, max_new_tokens=1200)
    try:
        data = extract_json_object(raw)
        relations = tuple(
            VisualRelation(
                source=str(item.get("source", "")),
                relation=str(item.get("relation", "")),
                target=str(item.get("target", "")),
                evidence=str(item.get("evidence", "")),
            )
            for item in data.get("relations", [])
            if isinstance(item, dict)
        )
        return VisualAnalysis(
            page_type=str(data.get("page_type", "mixed")),
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            entities=tuple(map(str, data.get("entities", []))),
            relations=relations,
            important_text=tuple(map(str, data.get("important_text", []))),
            uncertainties=tuple(map(str, data.get("uncertainties", []))),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise StructuredOutputError(f"Invalid structured VLM response: {exc}") from exc
