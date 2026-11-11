# Path: src/vlm_engineering/documents/schemas.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Typed data structures for visual document understanding and RAG chunks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ..exceptions import InputValidationError


def _require_string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise InputValidationError(f"{name} must be a string.")
    if not allow_empty and not value.strip():
        raise InputValidationError(f"{name} must be a non-empty string.")
    return value


def _positive_page(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise InputValidationError("page must be a positive integer.")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    """Coerce a scalar or sequence into a tuple of strings.

    A scalar string is one value, never a sequence of characters. ``None`` is
    treated as an empty list so partially populated model output is usable.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, (list, tuple)):
        raise InputValidationError(f"{name} must be a string or a list of strings.")
    return tuple(str(item) for item in value)


@dataclass(frozen=True, slots=True)
class VisualRelation:
    source: str
    relation: str
    target: str
    evidence: str = ""

    def __post_init__(self) -> None:
        _require_string(self.source, "relation.source")
        _require_string(self.relation, "relation.relation")
        _require_string(self.target, "relation.target")
        _require_string(self.evidence, "relation.evidence", allow_empty=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualRelation":
        if not isinstance(value, Mapping):
            raise InputValidationError("relation entries must be mappings.")
        return cls(
            source=str(value.get("source", "")).strip(),
            relation=str(value.get("relation", "")).strip(),
            target=str(value.get("target", "")).strip(),
            evidence=str(value.get("evidence", "")).strip(),
        )


@dataclass(frozen=True, slots=True)
class VisualAnalysis:
    page_type: str
    title: str
    summary: str
    entities: tuple[str, ...] = ()
    relations: tuple[VisualRelation, ...] = ()
    important_text: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_string(self.page_type, "page_type")
        _require_string(self.title, "title", allow_empty=True)
        _require_string(self.summary, "summary", allow_empty=True)
        for name, values in (
            ("entities", self.entities),
            ("important_text", self.important_text),
            ("uncertainties", self.uncertainties),
        ):
            if not isinstance(values, tuple) or not all(isinstance(item, str) for item in values):
                raise InputValidationError(f"{name} must be a tuple of strings.")
        if not isinstance(self.relations, tuple) or not all(
            isinstance(item, VisualRelation) for item in self.relations
        ):
            raise InputValidationError("relations must be a tuple of VisualRelation objects.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualAnalysis":
        if not isinstance(value, Mapping):
            raise InputValidationError("analysis must be a mapping.")
        relations = value.get("relations") or ()
        if isinstance(relations, Mapping):
            relations = (relations,)
        if not isinstance(relations, (list, tuple)):
            raise InputValidationError("relations must be a list.")
        return cls(
            page_type=str(value.get("page_type") or "mixed"),
            title=str(value.get("title") or ""),
            summary=str(value.get("summary") or ""),
            entities=_string_tuple(value.get("entities"), "entities"),
            relations=tuple(VisualRelation.from_dict(item) for item in relations),
            important_text=_string_tuple(value.get("important_text"), "important_text"),
            uncertainties=_string_tuple(value.get("uncertainties"), "uncertainties"),
        )


@dataclass(frozen=True, slots=True)
class NativePageContent:
    document_id: str
    page: int
    source_file: str
    native_text: str = ""
    title: str = ""
    image_ref: str = ""

    def __post_init__(self) -> None:
        _require_string(self.document_id, "document_id")
        _positive_page(self.page)
        _require_string(self.source_file, "source_file")
        _require_string(self.native_text, "native_text", allow_empty=True)
        _require_string(self.title, "title", allow_empty=True)
        _require_string(self.image_ref, "image_ref", allow_empty=True)


@dataclass(frozen=True, slots=True)
class VisualChunk:
    chunk_id: str
    document_id: str
    source_file: str
    page: int
    chunk_type: str
    title: str
    text: str
    image_ref: str
    parent_context: str = ""
    entities: tuple[str, ...] = ()
    relations: tuple[VisualRelation, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_string(self.chunk_id, "chunk_id")
        _require_string(self.document_id, "document_id")
        _require_string(self.source_file, "source_file")
        _positive_page(self.page)
        _require_string(self.chunk_type, "chunk_type")
        _require_string(self.title, "title", allow_empty=True)
        _require_string(self.text, "text", allow_empty=True)
        _require_string(self.image_ref, "image_ref", allow_empty=True)
        _require_string(self.parent_context, "parent_context", allow_empty=True)
        if not isinstance(self.entities, tuple) or not all(
            isinstance(item, str) for item in self.entities
        ):
            raise InputValidationError("entities must be a tuple of strings.")
        if not isinstance(self.relations, tuple) or not all(
            isinstance(item, VisualRelation) for item in self.relations
        ):
            raise InputValidationError("relations must be a tuple of VisualRelation objects.")
        if not isinstance(self.metadata, dict):
            raise InputValidationError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualChunk":
        if not isinstance(value, Mapping):
            raise InputValidationError("chunk must be a mapping.")
        relations = value.get("relations") or ()
        if isinstance(relations, Mapping):
            relations = (relations,)
        if not isinstance(relations, (list, tuple)):
            raise InputValidationError("chunk.relations must be a list.")
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise InputValidationError("chunk.metadata must be a mapping.")
        page = value.get("page")
        if isinstance(page, bool) or not isinstance(page, int):
            raise InputValidationError("page must be a positive integer.")
        return cls(
            chunk_id=str(value.get("chunk_id") or ""),
            document_id=str(value.get("document_id") or ""),
            source_file=str(value.get("source_file") or ""),
            page=page,
            chunk_type=str(value.get("chunk_type") or ""),
            title=str(value.get("title") or ""),
            text=str(value.get("text") or ""),
            image_ref=str(value.get("image_ref") or ""),
            parent_context=str(value.get("parent_context") or ""),
            entities=_string_tuple(value.get("entities"), "chunk.entities"),
            relations=tuple(VisualRelation.from_dict(item) for item in relations),
            metadata=dict(metadata),
        )
