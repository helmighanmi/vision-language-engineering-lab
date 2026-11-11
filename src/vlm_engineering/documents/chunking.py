# Path: src/vlm_engineering/documents/chunking.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Semantic-friendly text chunking and page-aware visual chunk construction."""

from __future__ import annotations

import hashlib
import json

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
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        if not paragraphs:
            return []
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if not current:
                current = paragraph
            else:
                candidate = f"{current}\n\n{paragraph}"
                if len(candidate) <= self.max_chars:
                    current = candidate
                    continue

                chunks.append(current)
                current = self._next_chunk(current, paragraph)

            # A paragraph can be larger than the whole budget; keep slicing it
            # so every emitted chunk satisfies the max_chars contract.
            step = self.max_chars - self.overlap_chars
            while len(current) > self.max_chars:
                chunks.append(current[: self.max_chars])
                current = current[step:]
        if current:
            chunks.append(current)
        return chunks

    def _next_chunk(self, previous: str, paragraph: str) -> str:
        """Start a chunk with the largest overlap that keeps the paragraph intact."""
        if not self.overlap_chars:
            return paragraph
        if len(paragraph) >= self.max_chars:
            # The paragraph must be hard-split; prepend the configured overlap.
            return f"{previous[-self.overlap_chars :]}\n\n{paragraph}"
        room = self.max_chars - len(paragraph) - 2
        if room <= 0:
            return paragraph
        prefix = previous[-min(self.overlap_chars, room) :]
        return f"{prefix}\n\n{paragraph}"


class VisualChunkBuilder:
    """Build one traceable visual chunk from native and VLM page evidence."""

    def build(
        self,
        native: NativePageContent,
        visual: VisualAnalysis,
        *,
        parent_context: str = "",
    ) -> VisualChunk:
        """Build a traceable chunk.

        ``chunk_id`` stays stable for the same document/page/image so that
        re-analysis can replace a chunk instead of duplicating it. A separate
        ``content_hash`` covers all fused evidence, allowing callers to detect
        changed text, entities or relations and invalidate caches explicitly.
        """
        text = fuse_native_and_visual(native, visual)
        identity = hashlib.sha256(
            f"{native.document_id}:{native.page}:{native.image_ref}".encode()
        ).hexdigest()[:16]
        content_hash = hashlib.sha256(
            json.dumps(
                {
                    "native": {
                        "title": native.title,
                        "text": native.native_text,
                        "image_ref": native.image_ref,
                    },
                    "visual": visual.to_dict(),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        return VisualChunk(
            chunk_id=f"{native.document_id}-p{native.page:04d}-{identity}",
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
            metadata={
                "has_visual_evidence": bool(native.image_ref),
                "content_hash": content_hash,
            },
        )
