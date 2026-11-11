# Path: src/vlm_engineering/evaluation/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Deterministic evaluation datasets, metrics and runners."""

from .dataset import EvaluationCase, load_chunks, load_evaluation_cases
from .metrics import keypoint_coverage, mrr_at_k, ndcg_at_k, recall_at_k
from .runner import CaseResult, EvaluationReport, evaluate

__all__ = [
    "CaseResult",
    "EvaluationCase",
    "EvaluationReport",
    "evaluate",
    "keypoint_coverage",
    "load_chunks",
    "load_evaluation_cases",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
]
