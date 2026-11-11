# Path: tests/unit/test_ingestion_pipeline.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Tests for the ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from vlm_engineering.documents.schemas import VisualChunk
from vlm_engineering.ingestion.ocr import NullOCR
from vlm_engineering.ingestion.pipeline import (
    SUPPORTED_IMAGE_EXTENSIONS,
    IngestionPipeline,
    IngestionResult,
    deterministic_chunk_id,
)


class FakeGenerator:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[Any] = []

    def generate(self, image: str, prompt: str, *, max_new_tokens: int = 512) -> str:
        self.calls.append((image, prompt, max_new_tokens))
        return self.response


class OCRFake:
    def __init__(self, text: str) -> None:
        self.text = text

    def recognize(self, image_path: str | Path) -> str:
        return self.text


def test_supported_extensions_include_common_formats() -> None:
    assert ".png" in SUPPORTED_IMAGE_EXTENSIONS
    assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS
    assert ".webp" in SUPPORTED_IMAGE_EXTENSIONS


def test_ingest_single_image(tmp_path) -> None:
    image = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "blue").save(image)

    generator = FakeGenerator(
        '{"page_type":"diagram","title":"T","summary":"S","entities":["A"],"relations":[],'
        '"important_text":[],"uncertainties":[]}'
    )
    pipeline = IngestionPipeline(generator, document_id="doc", source_file="src.pdf")
    results = pipeline.ingest_images([image], start_page=3)

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, IngestionResult)
    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert isinstance(chunk, VisualChunk)
    assert chunk.page == 3
    assert chunk.document_id == "doc"
    assert chunk.source_file == "src.pdf"
    assert chunk.title == "T"
    assert generator.calls[0][0] == str(image.resolve())


def test_ingest_with_ocr(tmp_path) -> None:
    image = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "blue").save(image)

    generator = FakeGenerator(
        '{"page_type":"text","title":"T","summary":"S","entities":[],"relations":[],'
        '"important_text":[],"uncertainties":[]}'
    )
    pipeline = IngestionPipeline(
        generator, ocr=OCRFake("native text here"), document_id="doc", source_file="src.pdf"
    )
    results = pipeline.ingest_images([image], start_page=1)

    chunk = results[0].chunks[0]
    assert "native text here" in chunk.text


def test_ingest_directory(tmp_path) -> None:
    for name in ("a.png", "b.png"):
        Image.new("RGB", (8, 8), "red").save(tmp_path / name)
    # Non-image file should be ignored.
    (tmp_path / "c.txt").write_text("not an image", encoding="utf-8")

    generator = FakeGenerator(
        '{"page_type":"photo","title":"T","summary":"S","entities":[],"relations":[],'
        '"important_text":[],"uncertainties":[]}'
    )
    pipeline = IngestionPipeline(generator)
    results = pipeline.ingest_directory(tmp_path)

    assert len(results) == 2  # only .png images
    assert all(len(r.chunks) == 1 for r in results)


def test_ingest_visual_analysis_failure_records_error(tmp_path) -> None:
    image = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "blue").save(image)

    class BrokenGenerator:
        def generate(self, _image, _prompt, *, max_new_tokens=512):
            raise RuntimeError("model down")

    pipeline = IngestionPipeline(BrokenGenerator())
    results = pipeline.ingest_images([image], start_page=1)

    assert len(results[0].errors) == 1
    assert "model down" in results[0].errors[0]
    # Still produces a chunk with degraded analysis.
    assert results[0].chunks[0].page == 1


def test_deterministic_chunk_id_is_stable() -> None:
    id1 = deterministic_chunk_id("doc", 1, "/path/to/page.png")
    id2 = deterministic_chunk_id("doc", 1, "/path/to/page.png")
    assert id1 == id2
    assert id1.startswith("doc-p0001-")


def test_null_ocr_returns_empty() -> None:
    assert NullOCR().recognize("any.png") == ""
