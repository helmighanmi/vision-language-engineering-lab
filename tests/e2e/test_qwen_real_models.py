# Path: tests/e2e/test_qwen_real_models.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Opt-in real inference; no model imports or downloads during ordinary collection."""

import os

import numpy as np
import pytest
from PIL import Image

from vlm_engineering import QwenVLEmbedder, QwenVLReranker

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        os.getenv("VLM_RUN_REAL_MODEL_TESTS") != "1",
        reason="Set VLM_RUN_REAL_MODEL_TESTS=1 explicitly.",
    ),
]


def options(env):
    path = os.getenv(env)
    return {"model_path": path} if path else {"model_size": "2b"}


@pytest.fixture
def real_image(tmp_path):
    path = tmp_path / "red.png"
    Image.new("RGB", (128, 128), "red").save(path)
    return str(path)


def test_embedding_real_text_image_and_mixed(real_image):
    model = QwenVLEmbedder(**options("VLM_QWEN_EMBEDDING_MODEL_PATH"), dimensions=128)
    try:
        values = model.encode(
            [{"text": "A red square"}, {"image": real_image}, {"text": "red square", "image": real_image}]
        )
        assert values.shape == (3, 128)
        assert np.isfinite(values).all()
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5)
        repeat = model.embed_text("A red square")
        np.testing.assert_allclose(values[0], repeat, atol=1e-3)
    finally:
        model.unload()


def test_reranker_real_text_image_and_mixed(real_image):
    model = QwenVLReranker(**options("VLM_QWEN_RERANKER_MODEL_PATH"))
    docs = [{"text": "A red square"}, {"image": real_image}, {"text": "red", "image": real_image}]
    try:
        raw = model.score({"text": "red", "image": real_image}, docs)
        assert raw.shape == (3,) and np.isfinite(raw).all()
        model.normalize_scores = True
        normalized = model.score({"text": "red", "image": real_image}, docs)
        np.testing.assert_allclose(normalized, 1 / (1 + np.exp(-raw)), atol=1e-3)
    finally:
        model.unload()
