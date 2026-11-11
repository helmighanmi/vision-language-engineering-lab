# Path: src/vlm_engineering/documents/visual_analysis.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Structured VLM prompting for diagrams, screenshots, tables and mixed pages."""

from __future__ import annotations

from collections.abc import Mapping
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

PAGE_TYPES = frozenset({"diagram", "table", "text", "mixed", "photo", "screenshot"})


class VisionGenerator(Protocol):
    def generate(self, image: str, prompt: str, *, max_new_tokens: int = 512) -> str: ...


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    raise StructuredOutputError(f"Expected a string value, received {type(value).__name__}.")


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    """Coerce a scalar or list into a tuple of strings.

    A single string is one item, never a sequence of characters. ``None`` is
    treated as empty so partially populated model output stays usable.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if not isinstance(value, (list, tuple)):
        raise StructuredOutputError(f"{name} must be a string or a list of strings.")
    result = []
    for item in value:
        if item is None:
            continue
        if not isinstance(item, (str, int, float)) or isinstance(item, bool):
            raise StructuredOutputError(f"{name} must contain only strings.")
        text = str(item).strip()
        if text:
            result.append(text)
    return tuple(result)


def _relations(value: object) -> tuple[tuple[VisualRelation, ...], tuple[str, ...]]:
    """Validate relation objects, skipping incomplete entries with a note."""
    if value is None:
        return (), ()
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise StructuredOutputError("relations must be a list of objects.")

    relations: list[VisualRelation] = []
    notes: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            notes.append(f"Skipped relation {index + 1}: expected an object.")
            continue
        source = _text(item.get("source"))
        relation = _text(item.get("relation"))
        target = _text(item.get("target"))
        evidence = _text(item.get("evidence"))
        if not (source and relation and target):
            notes.append(f"Skipped relation {index + 1}: missing source, relation or target.")
            continue
        relations.append(VisualRelation(source, relation, target, evidence))
    return tuple(relations), tuple(notes)


def _parse_analysis(raw: str) -> VisualAnalysis:
    data = extract_json_object(raw)
    if not isinstance(data, dict):
        raise StructuredOutputError("Structured response must be a JSON object.")

    page_type = _text(data.get("page_type"), "mixed").lower() or "mixed"
    notes: list[str] = []
    if page_type not in PAGE_TYPES:
        notes.append(f"Unrecognized page_type {page_type!r}; recorded as mixed.")
        page_type = "mixed"

    relations, relation_notes = _relations(data.get("relations"))
    notes.extend(relation_notes)

    uncertainties = list(_text_tuple(data.get("uncertainties"), "uncertainties"))
    uncertainties.extend(notes)

    return VisualAnalysis(
        page_type=page_type,
        title=_text(data.get("title")),
        summary=_text(data.get("summary")),
        entities=_text_tuple(data.get("entities"), "entities"),
        relations=relations,
        important_text=_text_tuple(data.get("important_text"), "important_text"),
        uncertainties=tuple(uncertainties),
    )


def analyze_visual_document(
    generator: VisionGenerator,
    image_ref: str,
    *,
    retries: int = 1,
) -> VisualAnalysis:
    """Run structured VLM analysis and validate the response into typed objects.

    Malformed or partially invalid JSON triggers a bounded retry (one by
    default) before ``StructuredOutputError`` is raised. Generator failures
    such as model-load errors are not retried and propagate unchanged.
    """
    if type(retries) is not int or retries < 0:
        raise ValueError("retries must be a non-negative integer.")

    last_error: Exception | None = None
    for _attempt in range(retries + 1):
        raw = generator.generate(image_ref, VISUAL_RAG_PROMPT, max_new_tokens=1200)
        try:
            return _parse_analysis(raw)
        except (StructuredOutputError, ValueError, TypeError, KeyError) as exc:
            last_error = exc

    raise StructuredOutputError(
        f"Invalid structured VLM response after {retries + 1} attempt(s): {last_error}"
    ) from last_error
