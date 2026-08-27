# Path: src/vlm_engineering/documents/chunking.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Semantic-friendly text chunking and page-aware visual chunk construction."""

from __future__ import annotations

import hashlib

from .fusion import fuse_native_and_visual
from .schemas import NativePageContent, VisualAnalysis, VisualChunk


class TextChunker:
    """Paragraph-aware text chunker with a bounded character budget."""

    def __init__(self, max_chars: int = 5000, overlap_chars: int = 400) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive.")
        if overlap_chars < 0 or overlap_chars >= max_chars:
            raise ValueError("overlap_chars must satisfy 0 <= overlap_chars < max_chars.")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def split(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if len(candidate) <= self.max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
                prefix = current[-self.overlap_chars :] if self.overlap_chars else ""
                current = f"{prefix}\n\n{paragraph}".strip() if prefix else paragraph
            else:
                # Hard fallback for a single oversized paragraph while guaranteeing progress.
                start = 0
                step = self.max_chars - self.overlap_chars
                while start < len(paragraph):
                    chunks.append(paragraph[start : start + self.max_chars])
                    start += step
                current = ""
        if current:
            chunks.append(current)
        return chunks


class VisualChunkBuilder:
    """Build one traceable visual chunk from native and VLM page evidence."""

    def build(
        self,
        native: NativePageContent,
        visual: VisualAnalysis,
        *,
        parent_context: str = "",
    ) -> VisualChunk:
        text = fuse_native_and_visual(native, visual)
        digest = hashlib.sha256(
            f"{native.document_id}:{native.page}:{native.image_ref}:{visual.summary}".encode()
        ).hexdigest()[:16]
        return VisualChunk(
            chunk_id=f"{native.document_id}-p{native.page:04d}-{digest}",
            document_id=native.document_id,
            source_file=native.source_file,
            page=native.page,
            chunk_type=visual.page_type,
            title=visual.title or native.title or f"Page {native.page}",
            text=text,
            image_ref=native.image_ref,
            parent_context=parent_context,
            entities=visual.entities,
            relations=visual.relations,
            metadata={"has_visual_evidence": bool(native.image_ref)},
        )
