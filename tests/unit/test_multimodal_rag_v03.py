# Path: tests/unit/test_multimodal_rag_v03.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Pipeline state, evidence, ranking and generation contracts."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from vlm_engineering import MultimodalRAGPipeline
from vlm_engineering.documents import VisualChunk
from vlm_engineering.retrieval import InMemoryVectorIndex, VisualRAGPipeline


class Embedder:
    def __init__(self):
        self.calls = []
        self.fail = False

    def encode(self, inputs, **kwargs):
        self.calls.append(inputs)
        if self.fail:
            raise RuntimeError("embedding failed")
        return np.array(
            [
                [1, 0] if "cache" in (v if isinstance(v, str) else v.get("text", "")) else [0, 1]
                for v in inputs
            ]
        )


class Reranker:
    def __init__(self):
        self.calls = []

    def score(self, query, docs):
        self.calls.append((query, docs))
        return np.array([0.5] * len(docs))


class Generator:
    def __init__(self):
        self.calls = []

    def generate_images(self, images, prompt, **kwargs):
        self.calls.append((images, prompt, kwargs))
        return "grounded answer"


@pytest.fixture
def chunks(tmp_path):
    result = []
    for i in range(3):
        path = tmp_path / f"page{i}.png"
        Image.new("RGB", (8, 8), "blue").save(path)
        result.append(
            VisualChunk(
                str(i),
                "document",
                "source.pdf",
                i + 1,
                "diagram",
                "title",
                "cache" if i != 1 else "database",
                str(path),
                metadata={"owner": "demo"},
            )
        )
    return result


def test_retrieve_rerank_and_generate_original_evidence(chunks):
    embedder, generator, reranker = Embedder(), Generator(), Reranker()
    pipe = MultimodalRAGPipeline(embedder, generator, reranker, candidate_k=3, top_k=2)
    pipe.index_chunks(chunks)
    assert embedder.calls[0][0] == {"text": "cache", "image": chunks[0].image_ref}
    answer = pipe.answer("cache", max_new_tokens=73)
    assert [c.chunk_id for c in answer.chunks] == ["0", "2"]
    assert answer.chunks[0] is chunks[0]
    assert answer.chunks[0].metadata == {"owner": "demo"}
    assert len(reranker.calls[0][1]) == 3
    images, prompt, kwargs = generator.calls[0]
    assert images == [chunks[0].image_ref, chunks[2].image_ref]
    assert "source.pdf, page 1" in prompt and "source.pdf, page 3" in prompt
    assert "IMAGE: 1" in prompt and "IMAGE: 2" in prompt
    assert "Answer only from the provided evidence" in prompt
    assert kwargs == {"max_new_tokens": 73}


def test_candidate_k_limits_second_stage(chunks):
    scorer = Reranker()
    pipe = MultimodalRAGPipeline(Embedder(), Generator(), scorer, top_k=1, candidate_k=2)
    pipe.index_chunks(chunks)
    assert len(pipe.retrieve("cache")) == 1
    assert len(scorer.calls[0][1]) == 2
    pipe.retrieve({"image": chunks[0].image_ref}, candidate_k=3)
    assert scorer.calls[-1][0] == {"image": chunks[0].image_ref}


def test_reranker_changes_retrieval_order(chunks, monkeypatch):
    scorer = Reranker()
    monkeypatch.setattr(scorer, "score", lambda query, docs: np.arange(len(docs)))
    pipe = MultimodalRAGPipeline(Embedder(), Generator(), scorer, top_k=1)
    pipe.index_chunks(chunks)
    assert pipe.retrieve("cache")[0] is chunks[1]


@pytest.mark.parametrize("cls", [MultimodalRAGPipeline, VisualRAGPipeline])
def test_reindex_replaces_empty_clears_and_failure_retains_old(cls, chunks):
    embedder = Embedder()
    pipe = cls(embedder, Generator())
    pipe.index_chunks(chunks)
    pipe.index_chunks([chunks[1]])
    assert pipe.retrieve("cache") == [chunks[1]]
    embedder.fail = True
    with pytest.raises(RuntimeError, match="embedding failed"):
        pipe.index_chunks([chunks[0]])
    embedder.fail = False
    assert pipe.retrieve("cache") == [chunks[1]]
    pipe.index_chunks([])
    call_count = len(embedder.calls)
    assert pipe.retrieve("cache") == []
    assert pipe.answer("cache").chunks == ()
    assert len(embedder.calls) == call_count


def test_query_image_forwarded_and_repeated_evidence_deduplicated(chunks):
    generator = Generator()
    pipe = MultimodalRAGPipeline(Embedder(), generator, top_k=2)
    pipe.index_chunks([chunks[0], replace(chunks[0], chunk_id="copy")])
    pipe.answer("cache", query_image=chunks[1].image_ref)
    assert generator.calls[0][0] == [chunks[1].image_ref, chunks[0].image_ref]
    assert "Image 1 is the query image" in generator.calls[0][1]
    assert "IMAGE: 2" in generator.calls[0][1]


def test_visual_evidence_required_and_query_image_is_not_evidence(chunks):
    pipe = MultimodalRAGPipeline(Embedder(), Generator())
    pipe.index_chunks([replace(chunks[0], image_ref="")])
    with pytest.raises(ValueError, match="requires an original evidence image"):
        pipe.answer("cache", query_image=chunks[1].image_ref)


def test_deleted_evidence_image_fails(chunks):
    from pathlib import Path

    pipe = MultimodalRAGPipeline(Embedder(), Generator())
    pipe.index_chunks(chunks)
    Path(chunks[0].image_ref).unlink()
    with pytest.raises(FileNotFoundError):
        pipe.answer("cache")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"top_k": True},
        {"candidate_k": 0},
        {"candidate_k": 2, "top_k": 3},
        {"max_new_tokens": 0},
    ],
)
def test_invalid_pipeline_config(kwargs):
    with pytest.raises(ValueError):
        MultimodalRAGPipeline(Embedder(), Generator(), **kwargs)


def test_invalid_call_limits_and_query():
    pipe = MultimodalRAGPipeline(Embedder(), Generator())
    for kwargs in ({"top_k": 0}, {"candidate_k": 1}, {"top_k": True}):
        with pytest.raises(ValueError):
            pipe.retrieve("q", **kwargs)
    with pytest.raises(ValueError):
        pipe.answer(" ")
    with pytest.raises(ValueError):
        pipe.answer("q", max_new_tokens=0)


@pytest.mark.parametrize("values", [[1], [float("nan")] * 3, [float("inf")] * 3])
def test_pipeline_checks_injected_scorer(chunks, values, monkeypatch):
    scorer = Reranker()
    monkeypatch.setattr(scorer, "score", lambda query, docs: values)
    pipe = MultimodalRAGPipeline(Embedder(), Generator(), scorer)
    pipe.index_chunks(chunks)
    with pytest.raises(ValueError, match="finite score"):
        pipe.retrieve("cache")


@pytest.mark.parametrize("vectors", [[[0, 0]], [[np.nan, 1]], [[np.inf, 1]], [[]]])
def test_index_rejects_invalid_vectors(vectors):
    with pytest.raises(ValueError):
        InMemoryVectorIndex().add(vectors, ["x"])


def test_index_query_contract_and_ties():
    index = InMemoryVectorIndex()
    index.add(np.ones((20, 2)), list(range(20)))
    assert [r.item for r in index.search(np.ones(2), top_k=20)] == list(range(20))
    for vector in (np.zeros(2), np.array([np.nan, 1]), np.ones(3), np.ones((1, 2))):
        with pytest.raises(ValueError):
            index.search(vector)
    index.add(np.empty((0, 2)), [])
    with pytest.raises(RuntimeError, match="empty"):
        index.search(np.ones(2))
