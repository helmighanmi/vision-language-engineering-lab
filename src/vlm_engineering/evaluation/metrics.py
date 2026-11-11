# Path: src/vlm_engineering/evaluation/metrics.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Deterministic retrieval and answer-quality metrics (no ML dependencies)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..exceptions import InputValidationError


def _validate_k(k: int) -> None:
    if type(k) is not int or k <= 0:
        raise InputValidationError("k must be a positive integer.")


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant IDs found in the first ``k`` unique results."""
    _validate_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for item in _unique(retrieved)[:k] if item in relevant_set)
    return hits / len(relevant_set)


def mrr_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Reciprocal rank of the first relevant result, or zero when absent."""
    _validate_k(k)
    relevant_set = set(relevant)
    for rank, item in enumerate(_unique(retrieved)[:k], start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Binary-relevance normalized discounted cumulative gain at ``k``."""
    _validate_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(_unique(retrieved)[:k], start=1)
        if item in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def keypoint_coverage(answer: str, keypoints: Sequence[str]) -> float:
    """Fraction of expected keypoints present in an answer (case-insensitive).

    The check is deliberately simple and deterministic: evaluators should use
    short, unambiguous facts. An empty keypoint list is treated as fully covered.
    """
    if not isinstance(answer, str):
        raise InputValidationError("answer must be a string.")
    expected = [item.strip().casefold() for item in keypoints if item.strip()]
    if not expected:
        return 1.0
    normalized = " ".join(answer.casefold().split())
    hits = sum(1 for item in expected if item in normalized)
    return hits / len(expected)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; returns 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0
