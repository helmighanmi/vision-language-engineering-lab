# Path: tests/test_retrieval.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

import numpy as np

from vlm_engineering.retrieval import InMemoryVectorIndex


def test_in_memory_retrieval_orders_by_cosine() -> None:
    index = InMemoryVectorIndex()
    index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), ["x", "y"])
    assert index.search(np.array([0.9, 0.1], dtype=np.float32), top_k=1)[0].item == "x"
