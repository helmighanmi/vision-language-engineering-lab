# Path: tests/unit/test_documents_extended.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

import pytest

from vlm_engineering.documents.chunking import TextChunker, VisualChunkBuilder
from vlm_engineering.documents.fusion import fuse_native_and_visual
from vlm_engineering.documents.schemas import NativePageContent, VisualAnalysis, VisualRelation
from vlm_engineering.documents.visual_analysis import analyze_visual_document
from vlm_engineering.exceptions import StructuredOutputError


class StaticGenerator:
    def __init__(self, response: str) -> None:
        self.response = response
        self.max_new_tokens: int | None = None

    def generate(self, _image: str, _prompt: str, *, max_new_tokens: int = 512) -> str:
        self.max_new_tokens = max_new_tokens
        return self.response


def _visual() -> VisualAnalysis:
    return VisualAnalysis(
        page_type="diagram",
        title="Payment architecture",
        summary="Gateway calls Payment Service.",
        entities=("Gateway", "Payment Service", "Redis"),
        relations=(VisualRelation("Gateway", "calls", "Payment Service", "arrow"),),
        important_text=("POST /payments",),
        uncertainties=("Cache protocol unreadable",),
    )


def test_text_chunker_returns_empty_for_blank_text() -> None:
    assert TextChunker().split("   \n\n   ") == []


def test_text_chunker_splits_oversized_paragraph_with_progress() -> None:
    chunks = TextChunker(max_chars=10, overlap_chars=2).split("abcdefghijklmnopqrstuvwxyz")
    assert len(chunks) > 1
    assert all(len(chunk) <= 10 for chunk in chunks)
    assert chunks[0] == "abcdefghij"


def test_text_chunker_preserves_small_paragraphs_together() -> None:
    chunks = TextChunker(max_chars=100, overlap_chars=10).split("alpha\n\nbeta\n\ngamma")
    assert chunks == ["alpha\n\nbeta\n\ngamma"]


def test_fusion_preserves_native_and_visual_evidence() -> None:
    native = NativePageContent(
        "doc-1",
        4,
        "architecture.pdf",
        native_text="Exact ID: PAY-42",
        title="Backend",
        image_ref="page.png",
    )
    fused = fuse_native_and_visual(native, _visual())
    assert "Exact ID: PAY-42" in fused
    assert "Gateway -> Payment Service: calls (arrow)" in fused
    assert "Cache protocol unreadable" in fused


def test_visual_chunk_id_is_deterministic() -> None:
    native = NativePageContent("doc", 2, "demo.pdf", native_text="A", image_ref="page.png")
    builder = VisualChunkBuilder()
    first = builder.build(native, _visual())
    second = builder.build(native, _visual())
    assert first.chunk_id == second.chunk_id


def test_visual_chunk_metadata_and_parent_context_are_preserved() -> None:
    native = NativePageContent("doc", 2, "demo.pdf", image_ref="page.png")
    chunk = VisualChunkBuilder().build(native, _visual(), parent_context="Payments")
    assert chunk.parent_context == "Payments"
    assert chunk.metadata["has_visual_evidence"] is True
    assert chunk.entities == ("Gateway", "Payment Service", "Redis")


def test_structured_analysis_preserves_relations_and_uncertainties() -> None:
    response = """```json
    {
      "page_type": "diagram",
      "title": "Architecture",
      "summary": "A calls B",
      "entities": ["A", "B"],
      "relations": [{"source": "A", "relation": "calls", "target": "B", "evidence": "arrow"}],
      "important_text": ["API"],
      "uncertainties": ["protocol"]
    }
    ```"""
    generator = StaticGenerator(response)
    result = analyze_visual_document(generator, "page.png")
    assert result.relations[0].evidence == "arrow"
    assert result.uncertainties == ("protocol",)
    assert generator.max_new_tokens == 1200


def test_structured_analysis_rejects_non_json_output() -> None:
    with pytest.raises(StructuredOutputError):
        analyze_visual_document(StaticGenerator("This is not JSON"), "page.png")
