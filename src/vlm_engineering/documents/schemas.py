# Path: src/vlm_engineering/documents/schemas.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Typed data structures for visual document understanding and RAG chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class VisualRelation:
    source: str
    relation: str
    target: str
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class VisualAnalysis:
    page_type: str
    title: str
    summary: str
    entities: tuple[str, ...] = ()
    relations: tuple[VisualRelation, ...] = ()
    important_text: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NativePageContent:
    document_id: str
    page: int
    source_file: str
    native_text: str = ""
    title: str = ""
    image_ref: str = ""


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
