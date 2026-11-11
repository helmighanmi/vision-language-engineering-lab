# Path: src/vlm_engineering/evaluation/dataset.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""JSONL/JSON evaluation datasets and chunk loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..documents.schemas import VisualChunk
from ..exceptions import InputValidationError

_CASE_FIELDS = {
    "case_id",
    "query",
    "relevant_ids",
    "query_image",
    "keypoints",
    "metadata",
}


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, (list, tuple)):
        raise InputValidationError(f"{name} must be a string or a list of strings.")
    result = []
    for item in value:
        if not isinstance(item, str):
            raise InputValidationError(f"{name} must contain only strings.")
        if item.strip():
            result.append(item)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One retrieval/answer evaluation case.

    ``relevant_ids`` are ``VisualChunk.chunk_id`` values. Cases with no relevant
    IDs are skipped from relevance aggregates instead of being counted as zero.
    """

    case_id: str
    query: str
    relevant_ids: tuple[str, ...] = ()
    query_image: str | None = None
    keypoints: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise InputValidationError("case_id must be a non-empty string.")
        if not isinstance(self.query, str) or not self.query.strip():
            raise InputValidationError("query must be a non-empty string.")
        if not isinstance(self.relevant_ids, tuple) or not all(
            isinstance(item, str) and item.strip() for item in self.relevant_ids
        ):
            raise InputValidationError("relevant_ids must be a tuple of non-empty strings.")
        if self.query_image is not None and (
            not isinstance(self.query_image, str) or not self.query_image.strip()
        ):
            raise InputValidationError("query_image must be a non-empty string or None.")
        if not isinstance(self.keypoints, tuple) or not all(
            isinstance(item, str) and item.strip() for item in self.keypoints
        ):
            raise InputValidationError("keypoints must be a tuple of non-empty strings.")
        if not isinstance(self.metadata, dict):
            raise InputValidationError("metadata must be a dictionary.")

    def retrieval_input(self) -> dict[str, str]:
        value = {"text": self.query}
        if self.query_image is not None:
            value["image"] = self.query_image
        return value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationCase":
        if not isinstance(value, Mapping):
            raise InputValidationError("evaluation case must be a mapping.")
        unknown = set(value) - _CASE_FIELDS
        if unknown:
            raise InputValidationError(f"Unknown evaluation case keys: {', '.join(sorted(unknown))}.")
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise InputValidationError("metadata must be a mapping.")
        return cls(
            case_id=str(value.get("case_id") or ""),
            query=str(value.get("query") or ""),
            relevant_ids=_string_tuple(value.get("relevant_ids"), "relevant_ids"),
            query_image=(
                str(value["query_image"]) if value.get("query_image") is not None else None
            ),
            keypoints=_string_tuple(value.get("keypoints"), "keypoints"),
            metadata=dict(metadata),
        )


def _load_records(path: str | Path) -> list[Any]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise InputValidationError(f"Invalid JSON dataset {file_path}: {exc}") from exc
        if not isinstance(value, list):
            raise InputValidationError(f"JSON dataset {file_path} must contain a list.")
        return value

    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise InputValidationError(
                f"Invalid JSON on line {line_number} of {file_path}: {exc}"
            ) from exc
    return records


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """Load cases from JSONL or a JSON array file."""
    cases = []
    for record in _load_records(path):
        if not isinstance(record, Mapping):
            raise InputValidationError("Each evaluation record must be a JSON object.")
        cases.append(EvaluationCase.from_dict(record))
    return cases


def load_chunks(path: str | Path) -> list[VisualChunk]:
    """Load indexed chunks from JSONL or a JSON array of ``VisualChunk`` objects."""
    chunks = []
    for record in _load_records(path):
        chunks.append(VisualChunk.from_dict(record))
    return chunks
