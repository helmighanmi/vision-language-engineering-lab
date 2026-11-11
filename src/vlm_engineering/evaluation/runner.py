# Path: src/vlm_engineering/evaluation/runner.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Run retrieval/answer evaluation cases and serialize a deterministic report."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..exceptions import InputValidationError
from .dataset import EvaluationCase
from .metrics import keypoint_coverage, mean, mrr_at_k, ndcg_at_k, recall_at_k

Retriever = Callable[[EvaluationCase], Sequence[Any]]
Answerer = Callable[[EvaluationCase, Sequence[Any]], Any]


def _default_id(item: Any) -> str:
    chunk_id = getattr(item, "chunk_id", None)
    return str(chunk_id) if chunk_id is not None else str(item)


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    retrieved_ids: tuple[str, ...]
    relevant_ids: tuple[str, ...]
    metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    k_values: tuple[int, ...]
    cases: tuple[CaseResult, ...]
    aggregate: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_values": list(self.k_values),
            "case_count": len(self.cases),
            "aggregate": self.aggregate,
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        return destination


def _validate_k_values(k_values: Sequence[int]) -> tuple[int, ...]:
    if not k_values:
        raise InputValidationError("k_values must contain at least one value.")
    result = []
    for value in k_values:
        if type(value) is not int or value <= 0:
            raise InputValidationError("k_values must contain positive integers.")
        if value not in result:
            result.append(value)
    return tuple(sorted(result))


def evaluate(
    retriever: Retriever,
    cases: Sequence[EvaluationCase],
    *,
    k_values: Sequence[int] = (1, 3, 5),
    answerer: Answerer | None = None,
    id_getter: Callable[[Any], str] | None = None,
) -> EvaluationReport:
    """Evaluate retrieval (and optionally answers) over labelled cases.

    ``retriever`` receives an :class:`EvaluationCase` and returns ranked items.
    ``answerer`` receives the case and the already retrieved items, and returns
    either a string or an object with an ``answer`` attribute. Relevance metrics
    are only computed for cases with non-empty ``relevant_ids``.
    """
    ks = _validate_k_values(k_values)
    get_id = id_getter or _default_id
    max_k = max(ks)

    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        items = list(retriever(case) or [])
        retrieved_ids = tuple(str(get_id(item)) for item in items)
        latency_ms = (time.perf_counter() - started) * 1000.0

        metrics: dict[str, float] = {}
        if case.relevant_ids:
            for k in ks:
                metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, case.relevant_ids, k)
                metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, case.relevant_ids, k)
            metrics[f"mrr@{max_k}"] = mrr_at_k(retrieved_ids, case.relevant_ids, max_k)

        answer_text: str | None = None
        if answerer is not None and case.keypoints:
            answer_started = time.perf_counter()
            value = answerer(case, items)
            answer_text = value if isinstance(value, str) else str(getattr(value, "answer", value))
            metrics["keypoint_coverage"] = keypoint_coverage(answer_text, case.keypoints)
            latency_ms += (time.perf_counter() - answer_started) * 1000.0

        results.append(
            CaseResult(
                case_id=case.case_id,
                retrieved_ids=retrieved_ids,
                relevant_ids=case.relevant_ids,
                metrics=metrics,
                latency_ms=latency_ms,
                answer=answer_text,
            )
        )

    metric_names = sorted({name for result in results for name in result.metrics})
    aggregate = {
        name: mean([result.metrics[name] for result in results if name in result.metrics])
        for name in metric_names
    }
    aggregate["latency_ms_mean"] = mean([result.latency_ms for result in results])
    return EvaluationReport(k_values=ks, cases=tuple(results), aggregate=aggregate)
