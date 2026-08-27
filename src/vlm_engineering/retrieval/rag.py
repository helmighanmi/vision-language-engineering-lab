# Path: src/vlm_engineering/retrieval/rag.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Simple visual RAG orchestration with retrieval and grounded generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..documents.schemas import VisualChunk
from .in_memory import InMemoryVectorIndex


class Embedder(Protocol):
    """Minimal text-embedding interface required by the RAG pipeline."""

    def encode(
        self,
        inputs: Sequence[str],
    ) -> np.ndarray:
        """Encode a sequence of text inputs into embedding vectors."""
        ...


class Generator(Protocol):
    """Minimal vision-language generation interface required by the RAG pipeline."""

    def generate(
        self,
        image: str | Path,
        prompt: str,
        *,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate a grounded response from an image and text prompt."""
        ...


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    """Answer returned by the visual RAG pipeline."""

    answer: str
    chunks: tuple[VisualChunk, ...]


class VisualRAGPipeline:
    """Retrieve text-described visual chunks and generate grounded answers.

    The pipeline deliberately depends on small Protocol interfaces rather
    than concrete model implementations.

    This allows the same orchestration layer to work with:
    - text-only embedders;
    - Qwen3-VL or another compatible VLM;
    - mocked components during unit testing.

    Heavy model dependencies therefore remain outside the core RAG logic.
    """

    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
    ) -> None:
        """Initialize the visual RAG pipeline.

        Args:
            embedder:
                Text embedding implementation used for indexing and retrieval.
            generator:
                Vision-language generator used to produce the final answer.
        """
        self.embedder = embedder
        self.generator = generator

        self.index = InMemoryVectorIndex()
        self._chunks: list[VisualChunk] = []

    def index_chunks(
        self,
        chunks: Sequence[VisualChunk],
    ) -> None:
        """Embed and index visual-document chunks.

        Args:
            chunks:
                Visual chunks whose textual representations will be indexed.
        """
        self._chunks = list(chunks)

        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode(texts)

        self.index.add(
            embeddings,
            chunks,
        )

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
    ) -> list[VisualChunk]:
        """Retrieve the most relevant visual chunks for a question.

        Args:
            question:
                User query expressed as text.
            top_k:
                Maximum number of chunks to retrieve.

        Returns:
            Ranked list of relevant visual chunks.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not question.strip():
            raise ValueError("question must not be empty.")

        query_embedding = self.embedder.encode([question])[0]

        results = self.index.search(
            query_embedding,
            top_k=top_k,
        )

        return [result.item for result in results]

    def answer(
        self,
        question: str,
        *,
        top_k: int = 3,
    ) -> RAGAnswer:
        """Answer a question using retrieved textual and visual evidence.

        Retrieval is performed using the textual representation of each
        visual chunk. The original image associated with the highest-ranked
        chunk is then supplied to the VLM together with the retrieved
        evidence.

        Args:
            question:
                User question.
            top_k:
                Number of chunks used as evidence.

        Returns:
            Grounded answer together with the retrieved chunks.
        """
        chunks = self.retrieve(
            question,
            top_k=top_k,
        )

        if not chunks:
            return RAGAnswer(
                answer="No relevant evidence found.",
                chunks=(),
            )

        evidence = "\n\n".join(
            (
                f"SOURCE: {chunk.source_file}, page {chunk.page}\n"
                f"{chunk.text}"
            )
            for chunk in chunks
        )

        prompt = (
            "Answer only from the provided evidence. "
            "If the evidence is ambiguous or insufficient, say so. "
            "Cite the source file and page used in the answer.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"EVIDENCE:\n{evidence}"
        )

        answer = self.generator.generate(
            chunks[0].image_ref,
            prompt,
            max_new_tokens=700,
        )

        return RAGAnswer(
            answer=answer,
            chunks=tuple(chunks),
        )