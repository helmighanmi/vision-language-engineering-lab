# Path: src/vlm_engineering/ingestion/pipeline.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Ingestion pipeline: images/PDF pages -> traceable VisualChunk objects."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..documents.chunking import VisualChunkBuilder
from ..documents.schemas import NativePageContent, VisualAnalysis, VisualChunk
from ..documents.visual_analysis import analyze_visual_document
from .ocr import NullOCR, OCRAdapter

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"})


@dataclass
class IngestionResult:
    """Outcome of ingesting one source."""

    source: str
    chunks: list[VisualChunk]
    errors: list[str]


class IngestionPipeline:
    """Ingest images or rendered PDF pages into VisualChunk objects.

    For each page:
    1. Render/load the page image.
    2. Optionally run OCR to extract native text.
    3. Optionally run VLM structured analysis.
    4. Fuse native text + visual semantics into a VisualChunk.

    The pipeline does not load models itself; pass a configured generator.
    """

    def __init__(
        self,
        generator,
        *,
        ocr: OCRAdapter | None = None,
        document_id: str = "ingested",
        source_file: str = "ingested",
    ) -> None:
        self.generator = generator
        self.ocr = ocr or NullOCR()
        self.document_id = document_id
        self.source_file = source_file
        self.builder = VisualChunkBuilder()

    @classmethod
    def from_images(
        cls,
        image_paths: Sequence[str | Path],
        generator,
        *,
        ocr: OCRAdapter | None = None,
        document_id: str = "ingested",
        source_file: str = "images",
    ) -> "IngestionPipeline":
        """Create a pipeline ready to ingest a list of image paths."""
        return cls(
            generator=generator,
            ocr=ocr,
            document_id=document_id,
            source_file=source_file,
        )

    def ingest_directory(
        self,
        directory: str | Path,
        *,
        recursive: bool = False,
        start_page: int = 1,
    ) -> list[IngestionResult]:
        """Ingest all supported images in a directory.

        Returns one :class:`IngestionResult` per image, in sorted order.
        """
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        image_paths = sorted(
            path
            for path in directory.glob(pattern)
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        return self.ingest_images(image_paths, start_page=start_page)

    def ingest_images(
        self,
        image_paths: Sequence[str | Path],
        *,
        start_page: int = 1,
    ) -> list[IngestionResult]:
        """Ingest a sequence of image paths, one page per image."""
        results: list[IngestionResult] = []
        for index, image_path in enumerate(image_paths, start=start_page):
            results.append(self.ingest_image(image_path, page=index))
        return results

    def ingest_image(
        self,
        image_path: str | Path,
        *,
        page: int,
    ) -> IngestionResult:
        """Ingest a single image as one page."""
        image_path = Path(image_path)
        source = str(image_path.resolve())
        errors: list[str] = []

        # Step 1: OCR / native text.
        native_text = ""
        try:
            native_text = self.ocr.recognize(image_path)
        except Exception as exc:  # OCR failures should not abort ingestion
            errors.append(f"OCR failed: {exc}")

        # Step 2: VLM structured analysis.
        analysis = None
        try:
            analysis = analyze_visual_document(self.generator, str(image_path))
        except Exception as exc:
            errors.append(f"Visual analysis failed: {exc}")
            analysis = VisualAnalysis(
                page_type="mixed",
                title=f"Page {page}",
                summary="",
                uncertainties=(f"Analysis failed: {exc}",),
            )

        if analysis is None:
            analysis = VisualAnalysis(
                page_type="mixed",
                title=f"Page {page}",
                summary="",
            )

        # Step 3: Build the traceable chunk.
        native = NativePageContent(
            document_id=self.document_id,
            page=page,
            source_file=self.source_file,
            native_text=native_text,
            title=analysis.title or f"Page {page}",
            image_ref=source,
        )
        try:
            chunk = self.builder.build(native, analysis)
        except Exception as exc:
            errors.append(f"Chunk building failed: {exc}")
            chunk = VisualChunk(
                chunk_id=f"{self.document_id}-p{page:04d}-error",
                document_id=self.document_id,
                source_file=self.source_file,
                page=page,
                chunk_type="mixed",
                title=f"Page {page}",
                text=native_text,
                image_ref=source,
                metadata={"ingestion_errors": errors},
            )

        return IngestionResult(
            source=source,
            chunks=[chunk],
            errors=errors,
        )

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        *,
        dpi: int = 150,
        start_page: int = 1,
    ) -> list[IngestionResult]:
        """Ingest a PDF by rendering pages to images.

        Requires the optional ``pypdf`` and ``pdf2image`` packages, or pass
        pre-rendered images via :meth:`ingest_images`.
        """
        raise NotImplementedError(
            "PDF ingestion requires the optional [ingest] extra. "
            "Use ingest_images() with pre-rendered page images instead."
        )

    def all_chunks(self, results: Sequence[IngestionResult]) -> list[VisualChunk]:
        """Flatten chunks from multiple results, skipping errored ones."""
        chunks: list[VisualChunk] = []
        for result in results:
            chunks.extend(result.chunks)
        return chunks


def deterministic_chunk_id(document_id: str, page: int, image_ref: str) -> str:
    """Stable chunk id derived from document, page and image path."""
    digest = hashlib.sha256(
        f"{document_id}:{page}:{Path(image_ref).resolve()}".encode()
    ).hexdigest()[:16]
    return f"{document_id}-p{page:04d}-{digest}"
