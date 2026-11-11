# Path: tests/unit/test_evaluation_v04.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Tests for the deterministic evaluation datasets, metrics and runner."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vlm_engineering.documents.schemas import VisualChunk, VisualRelation
from vlm_engineering.evaluation import (
    EvaluationCase,
    evaluate,
    keypoint_coverage,
    load_chunks,
    load_evaluation_cases,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from vlm_engineering.evaluation.runner import mean
from vlm_engineering.exceptions import InputValidationError


def test_recall_at_k_uses_unique_ranked_results() -> None:
    assert recall_at_k(["a", "a", "b"], ["b"], 2) == 1.0
    assert recall_at_k(["a", "b", "c"], ["a", "c"], 1) == 0.5
    assert recall_at_k(["a", "b", "c"], ["a", "c"], 3) == 1.0
    assert recall_at_k([], ["a"], 3) == 0.0
    assert recall_at_k(["a"], [], 3) == 0.0


def test_mrr_at_k() -> None:
    assert mrr_at_k(["x", "a"], ["a"], 2) == 0.5
    assert mrr_at_k(["a"], ["a"], 2) == 1.0
    assert mrr_at_k(["x"], ["a"], 2) == 0.0


def test_ndcg_at_k_rewards_earlier_relevance() -> None:
    assert ndcg_at_k(["a"], ["a"], 1) == 1.0
    first = ndcg_at_k(["a", "x"], ["a"], 2)
    second = ndcg_at_k(["x", "a"], ["a"], 2)
    assert first > second
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)
    assert ndcg_at_k([], ["a"], 1) == 0.0


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_metrics_reject_invalid_k(k) -> None:
    for metric in (recall_at_k, mrr_at_k, ndcg_at_k):
        with pytest.raises(InputValidationError):
            metric(["a"], ["a"], k)


def test_keypoint_coverage_is_case_insensitive() -> None:
    assert keypoint_coverage("Redis CACHES session data.", ["redis", "session"]) == 1.0
    assert keypoint_coverage("Redis caches data.", ["redis", "postgres"]) == 0.5
    assert keypoint_coverage("anything", []) == 1.0
    with pytest.raises(InputValidationError):
        keypoint_coverage(1, ["x"])  # type: ignore[arg-type]


def test_mean_handles_empty_input() -> None:
    assert mean([1.0, 3.0]) == 2.0
    assert mean([]) == 0.0


def test_evaluation_case_contract() -> None:
    case = EvaluationCase("c1", "find cache", ("chunk-1",), query_image="query.png")
    assert case.retrieval_input() == {"text": "find cache", "image": "query.png"}
    assert EvaluationCase.from_dict(case.to_dict()) == case
    assert EvaluationCase("c2", "text only").retrieval_input() == {"text": "text only"}
    with pytest.raises(InputValidationError):
        EvaluationCase("", "query")
    with pytest.raises(InputValidationError):
        EvaluationCase("c", "  ")
    with pytest.raises(InputValidationError):
        EvaluationCase("c", "query", relevant_ids=("",))


def test_load_evaluation_cases_supports_jsonl_and_json_arrays(tmp_path) -> None:
    records = [
        {"case_id": "c1", "query": "q1", "relevant_ids": ["a"]},
        {"case_id": "c2", "query": "q2"},
    ]
    jsonl = tmp_path / "cases.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    loaded = load_evaluation_cases(jsonl)
    assert [case.case_id for case in loaded] == ["c1", "c2"]

    array = tmp_path / "cases.json"
    array.write_text(json.dumps(records), encoding="utf-8")
    assert load_evaluation_cases(array) == loaded

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    assert load_evaluation_cases(empty) == []


def test_load_evaluation_cases_rejects_unknown_keys_and_bad_json(tmp_path) -> None:
    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text(json.dumps({"case_id": "c", "query": "q", "surprise": 1}), encoding="utf-8")
    with pytest.raises(InputValidationError, match="Unknown"):
        load_evaluation_cases(unknown)

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"case_id": "c",\n', encoding="utf-8")
    with pytest.raises(InputValidationError, match="line 1"):
        load_evaluation_cases(invalid)


def test_load_chunks_round_trips_visual_chunks(tmp_path) -> None:
    chunk = VisualChunk(
        "id",
        "doc",
        "source.pdf",
        1,
        "diagram",
        "Title",
        "Text",
        "page.png",
        entities=("A",),
        relations=(VisualRelation("A", "calls", "B"),),
        metadata={"owner": "team"},
    )
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(chunk.to_dict()) + "\n", encoding="utf-8")
    assert load_chunks(path) == [chunk]


def test_evaluate_computes_aggregates_and_answer_metrics(tmp_path) -> None:
    cases = [
        EvaluationCase("c1", "q1", ("a", "c"), keypoints=("redis", "cache")),
        EvaluationCase("c2", "q2", ("b",)),
        EvaluationCase("c3", "q3"),
    ]

    def retriever(case: EvaluationCase):
        if case.case_id == "c1":
            return [SimpleNamespace(chunk_id="a"), SimpleNamespace(chunk_id="b")]
        return [SimpleNamespace(chunk_id="b")]

    report = evaluate(
        retriever,
        cases,
        k_values=(1, 2),
        answerer=lambda _case, _items: "Redis is used as a cache.",
    )
    assert report.k_values == (1, 2)
    assert [result.case_id for result in report.cases] == ["c1", "c2", "c3"]
    assert report.cases[0].metrics["recall@1"] == 0.5
    assert report.cases[0].metrics["recall@2"] == 0.5
    assert report.cases[0].metrics["keypoint_coverage"] == 1.0
    assert report.cases[1].metrics["recall@1"] == 1.0
    assert "keypoint_coverage" not in report.cases[2].metrics
    assert report.aggregate["recall@1"] == 0.75
    assert report.aggregate["keypoint_coverage"] == 1.0
    assert report.aggregate["latency_ms_mean"] >= 0.0

    destination = report.to_json(tmp_path / "report.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["aggregate"]["recall@1"] == 0.75
    assert payload["case_count"] == 3


def test_evaluate_supports_custom_id_getter_and_empty_cases() -> None:
    report = evaluate(lambda _case: [{"id": "x"}], [EvaluationCase("c", "q", ("x",))], k_values=[1], id_getter=lambda item: item["id"])
    assert report.cases[0].retrieved_ids == ("x",)
    assert report.aggregate["recall@1"] == 1.0

    empty = evaluate(lambda _case: [], [])
    assert empty.cases == ()
    assert empty.aggregate == {"latency_ms_mean": 0.0}


@pytest.mark.parametrize("k_values", [(), (0,), (1, 1.5), (True,)])
def test_evaluate_rejects_invalid_k_values(k_values) -> None:
    with pytest.raises(InputValidationError):
        evaluate(lambda _case: [], [], k_values=k_values)
