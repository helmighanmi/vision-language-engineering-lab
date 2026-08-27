# Path: tests/unit/test_retrieval_extended.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Extended unit tests for retrieval and visual RAG components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from vlm_engineering.documents.schemas import VisualChunk
from vlm_engineering.retrieval.in_memory import InMemoryVectorIndex
from vlm_engineering.retrieval.multimodal_embedding import QwenMultimodalEmbedder
from vlm_engineering.retrieval.rag import VisualRAGPipeline
from vlm_engineering.retrieval.reranker import QwenMultimodalReranker
from vlm_engineering.retrieval.text_embedding import TextEmbedder


class FakeSentenceModel:
    """Lightweight SentenceTransformer replacement used in tests."""

    def __init__(self) -> None:
        self.calls: list[
            tuple[
                list[Any],
                dict[str, Any],
            ]
        ] = []

    def encode(
        self,
        inputs: list[Any],
        **kwargs: Any,
    ) -> np.ndarray:
        """Return deterministic embeddings without loading a real model."""
        self.calls.append(
            (
                inputs,
                kwargs,
            )
        )

        return np.asarray(
            [
                [1.0, 0.0]
                for _ in inputs
            ],
            dtype=np.float64,
        )


class FakeCrossEncoder:
    """Lightweight CrossEncoder replacement used in tests."""

    def __init__(self) -> None:
        self.pairs: list[
            tuple[
                Any,
                Any,
            ]
        ] = []

    def predict(
        self,
        pairs: list[tuple[Any, Any]],
    ) -> list[float]:
        """Return deterministic reranking scores."""
        self.pairs = pairs

        return [
            0.2,
            0.9,
        ][: len(pairs)]


class KeywordEmbedder:
    """Deterministic text embedder for RAG pipeline tests."""

    def encode(
        self,
        inputs: Sequence[str],
    ) -> np.ndarray:
        """Map Redis-related text and other text into separate vectors."""
        rows: list[list[float]] = []

        for value in inputs:
            text = value.lower()

            if "redis" in text or "cache" in text:
                rows.append([1.0, 0.0])
            else:
                rows.append([0.0, 1.0])

        return np.asarray(
            rows,
            dtype=np.float32,
        )


class RecordingGenerator:
    """Fake VLM generator that records the generation request."""

    def __init__(self) -> None:
        self.image: str | Path | None = None
        self.prompt: str = ""
        self.max_new_tokens: int | None = None

    def generate(
        self,
        image: str | Path,
        prompt: str,
        *,
        max_new_tokens: int = 512,
    ) -> str:
        """Record the request and return a deterministic answer."""
        self.image = image
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

        return "Redis is used for caching (architecture.pdf, page 1)."


def _chunk(
    chunk_id: str,
    text: str,
    image_ref: str = "page.png",
    *,
    page: int = 1,
) -> VisualChunk:
    """Create a valid VisualChunk test fixture."""

    return VisualChunk(
        chunk_id=chunk_id,
        document_id="doc",
        source_file="architecture.pdf",
        page=page,
        chunk_type="diagram",
        title="Architecture",
        text=text,
        image_ref=image_ref,
        parent_context="Application architecture",
        entities=(),
        relations=(),
        metadata={
            "test_fixture": True,
        },
    )


def test_vector_index_rejects_mismatched_items() -> None:
    """The index must receive exactly one item per embedding."""

    index = InMemoryVectorIndex()

    with pytest.raises(
        ValueError,
        match="one vector per item",
    ):
        index.add(
            np.asarray(
                [[1.0, 0.0]],
                dtype=np.float32,
            ),
            [
                "a",
                "b",
            ],
        )


def test_vector_index_rejects_search_before_add() -> None:
    """Searching an empty vector index must fail explicitly."""

    index = InMemoryVectorIndex()

    with pytest.raises(
        RuntimeError,
        match="Index is empty",
    ):
        index.search(
            np.asarray(
                [1.0, 0.0],
                dtype=np.float32,
            )
        )


def test_vector_index_returns_rank_and_score() -> None:
    """The nearest item should be returned with rank and cosine score."""

    index = InMemoryVectorIndex()

    index.add(
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        [
            "a",
            "b",
        ],
    )

    results = index.search(
        np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        top_k=2,
    )

    assert len(results) == 2

    assert [
        result.rank
        for result in results
    ] == [
        1,
        2,
    ]

    assert results[0].item == "a"
    assert results[0].score == pytest.approx(1.0)


def test_text_embedder_uses_normalized_sentence_transformer_contract() -> None:
    """Text embeddings should request normalized vectors."""

    model = FakeSentenceModel()

    embedder = TextEmbedder(
        model=model,
    )

    result = embedder.encode(
        [
            "one",
            "two",
        ]
    )

    assert result.shape == (2, 2)
    assert result.dtype == np.float32

    assert model.calls[0][0] == [
        "one",
        "two",
    ]

    assert model.calls[0][1] == {
        "normalize_embeddings": True,
    }


def test_multimodal_embedder_forwards_prompt() -> None:
    """The multimodal embedding adapter should forward retrieval prompts."""

    model = FakeSentenceModel()

    embedder = QwenMultimodalEmbedder(
        model=model,
    )

    result = embedder.encode(
        [
            {
                "text": "query",
            }
        ],
        prompt="Retrieve relevant visual evidence",
    )

    assert result.shape == (1, 2)
    assert result.dtype == np.float32

    assert model.calls[0][1] == {
        "prompt": "Retrieve relevant visual evidence",
    }


def test_multimodal_embedder_without_prompt() -> None:
    """No prompt argument should be forwarded when one is not provided."""

    model = FakeSentenceModel()

    embedder = QwenMultimodalEmbedder(
        model=model,
    )

    result = embedder.encode(
        [
            {
                "text": "architecture",
            }
        ]
    )

    assert result.shape == (1, 2)
    assert model.calls[0][1] == {}


def test_reranker_builds_query_document_pairs() -> None:
    """The reranker should pair the query with every candidate document."""

    model = FakeCrossEncoder()

    reranker = QwenMultimodalReranker(
        model=model,
    )

    scores = reranker.score(
        "question",
        [
            "doc-a",
            "doc-b",
        ],
    )

    assert model.pairs == [
        (
            "question",
            "doc-a",
        ),
        (
            "question",
            "doc-b",
        ),
    ]

    assert scores.dtype == np.float32

    np.testing.assert_allclose(
        scores,
        np.asarray(
            [
                0.2,
                0.9,
            ],
            dtype=np.float32,
        ),
    )


def test_visual_rag_retrieves_matching_chunk() -> None:
    """Text retrieval should rank the semantically matching visual chunk."""

    pipeline = VisualRAGPipeline(
        KeywordEmbedder(),
        RecordingGenerator(),
    )

    redis_chunk = _chunk(
        "redis",
        "Redis cache architecture",
    )

    database_chunk = _chunk(
        "database",
        "PostgreSQL persistence database",
    )

    pipeline.index_chunks(
        [
            redis_chunk,
            database_chunk,
        ]
    )

    result = pipeline.retrieve(
        "Which component uses Redis?",
        top_k=1,
    )

    assert result == [
        redis_chunk,
    ]


def test_visual_rag_answer_includes_grounded_evidence() -> None:
    """Generation should receive both the source evidence and original image."""

    generator = RecordingGenerator()

    pipeline = VisualRAGPipeline(
        KeywordEmbedder(),
        generator,
    )

    redis_chunk = _chunk(
        "redis",
        "Redis cache architecture",
        image_ref="redis.png",
        page=1,
    )

    pipeline.index_chunks(
        [
            redis_chunk,
        ]
    )

    result = pipeline.answer(
        "Where is the cache?",
        top_k=1,
    )

    assert result.answer == (
        "Redis is used for caching "
        "(architecture.pdf, page 1)."
    )

    assert result.chunks == (
        redis_chunk,
    )

    assert generator.image == "redis.png"

    assert "Where is the cache?" in generator.prompt

    assert (
        "SOURCE: architecture.pdf, page 1"
        in generator.prompt
    )

    assert (
        "Answer only from the provided evidence"
        in generator.prompt
    )

    assert generator.max_new_tokens == 700


def test_visual_rag_answer_preserves_retrieved_chunk_metadata() -> None:
    """Retrieved evidence should preserve VisualChunk metadata."""

    generator = RecordingGenerator()

    pipeline = VisualRAGPipeline(
        KeywordEmbedder(),
        generator,
    )

    chunk = _chunk(
        "redis",
        "Redis is used as the application cache.",
        image_ref="redis.png",
        page=4,
    )

    pipeline.index_chunks(
        [
            chunk,
        ]
    )

    result = pipeline.answer(
        "What component provides caching?",
        top_k=1,
    )

    assert result.chunks == (
        chunk,
    )

    assert result.chunks[0].metadata == {
        "test_fixture": True,
    }

    assert result.chunks[0].page == 4

    assert result.chunks[0].source_file == "architecture.pdf"


def test_visual_rag_rejects_empty_question() -> None:
    """An empty query should be rejected before embedding."""

    pipeline = VisualRAGPipeline(
        KeywordEmbedder(),
        RecordingGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        pipeline.retrieve(
            "   ",
        )


def test_visual_rag_rejects_non_positive_top_k() -> None:
    """Retrieval must require a positive top_k value."""

    pipeline = VisualRAGPipeline(
        KeywordEmbedder(),
        RecordingGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        pipeline.retrieve(
            "Redis",
            top_k=0,
        )


def test_visual_rag_indexes_empty_collection_without_model_call() -> None:
    """Indexing an empty collection should be a safe no-op."""

    pipeline = VisualRAGPipeline(
        KeywordEmbedder(),
        RecordingGenerator(),
    )

    pipeline.index_chunks([])