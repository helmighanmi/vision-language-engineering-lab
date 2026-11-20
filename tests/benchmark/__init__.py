# Path: tests/benchmark/__init__.py
# Minor package init for the v0.4 benchmark suite.

from __future__ import annotations

from .fixtures import make_ndcg_benchmark_fixture, make_recall_benchmark_fixture
from .ndcg_at_k import average_ndcg_at_k, ndcg_at_k
from .recall_at_k import average_recall_at_k, recall_at_k

__all__ = [
    "recall_at_k",
    "ndcg_at_k",
    "average_recall_at_k",
    "average_ndcg_at_k",
    "make_recall_benchmark_fixture",
    "make_ndcg_benchmark_fixture",
]