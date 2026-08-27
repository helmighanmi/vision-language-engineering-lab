# Path: src/vlm_engineering/retrieval/rag.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Simple visual RAG orchestration with retrieval and grounded generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from ..documents.schemas import VisualChunk
from .in_memory import InMemoryVectorIndex


class Embedder(Protocol):
    def encode(self, inputs: Sequence[Any], **kwargs: Any) -> np.ndarray: ...


class Generator(Protocol):
    def generate(self, image: str, prompt: str, **kwargs: Any) -> str: ...


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    answer: str
    chunks: tuple[VisualChunk, ...]


class VisualRAGPipeline:
    """Educational production-shaped pipeline for text-described visual chunks."""

    def __init__(self, embedder: Embedder, generator: Generator) -> None:
        self.embedder = embedder
        self.generator = generator
        self.index = InMemoryVectorIndex()
        self._chunks: list[VisualChunk] = []

    def index_chunks(self, chunks: Sequence[VisualChunk]) -> None:
        self._chunks = list(chunks)
        embeddings = self.embedder.encode([chunk.text for chunk in chunks])
        self.index.add(embeddings, chunks)

    def retrieve(self, question: str, *, top_k: int = 5) -> list[VisualChunk]:
        query = self.embedder.encode([question])[0]
        return [result.item for result in self.index.search(query, top_k=top_k)]

    def answer(self, question: str, *, top_k: int = 3) -> RAGAnswer:
        chunks = self.retrieve(question, top_k=top_k)
        if not chunks:
            return RAGAnswer(answer="No relevant evidence found.", chunks=())
        evidence = "\n\n".join(
            f"SOURCE: {chunk.source_file}, page {chunk.page}\n{chunk.text}" for chunk in chunks
        )
        prompt = (
            "Answer only from the provided evidence. If evidence is ambiguous, say so. "
            "Cite the source file and page in the answer.\n\n"
            f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence}"
        )
        answer = self.generator.generate(chunks[0].image_ref, prompt, max_new_tokens=700)
        return RAGAnswer(answer=answer, chunks=tuple(chunks))
