# Path: src/vlm_engineering/providers/openai_compatible.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""OpenAI-compatible HTTP providers (vLLM, LM Studio, Ollama, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import httpx
import numpy as np

from ..inputs import MultimodalInput, normalize_input, normalize_inputs
from ..operational import backend_errors
from ..exceptions import InputValidationError


class OpenAICompatibleVLMProvider:
    """VLM provider calling an OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Images must be reachable by the server (HTTP URLs or local paths the
    server can read). This provider sends image URLs as ``image_url`` content.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def generate(
        self,
        images: Sequence[str | Path],
        prompt: str,
        *,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        content: list[dict[str, Any]] = []
        for image in images:
            url = str(image)
            if not url.startswith(("http://", "https://")):
                raise InputValidationError(
                    "OpenAI-compatible generation requires image URLs reachable by the remote server; "
                    f"local paths such as {url!r} are not automatically readable remotely. "
                    "Host the image at a public URL or use the local Qwen provider instead."
                )
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": content})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        with backend_errors():
            response = self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        with backend_errors():
            response = self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "OpenAICompatibleVLMProvider":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class OpenAICompatibleEmbeddingProvider:
    """Embedding provider calling an OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def encode(
        self,
        inputs: Sequence[MultimodalInput],
        *,
        prompt: str | None = None,
    ) -> np.ndarray:
        prepared = normalize_inputs(inputs)
        vectors = []
        for item in prepared:
            payload: dict[str, Any] = {
                "model": self.model,
                "input": item.get("text", ""),
            }
            if "image" in item:
                payload["input"] = {
                    "text": item.get("text", ""),
                    "image": item["image"],
                }
            with backend_errors():
                response = self.client.post("/v1/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
            vectors.append(data["data"][0]["embedding"])
        return np.asarray(vectors, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return self.encode([{"text": text}])[0]

    def embed_image(self, image: str | Path) -> np.ndarray:
        return self.encode([{"image": str(image)}])[0]

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "OpenAICompatibleEmbeddingProvider":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class OpenAICompatibleRerankerProvider:
    """Reranker provider calling a custom ``/v1/rerank`` endpoint.

    By default, errors from the remote endpoint are surfaced to the caller
    rather than silently falling back to a local model. Set
    ``fallback_to_local=True`` to retain the legacy silent-fallback behavior
    (which requires local Qwen model dependencies and downloads).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        fallback_to_local: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback_to_local = fallback_to_local
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def score(
        self,
        query: MultimodalInput,
        documents: Sequence[MultimodalInput],
        *,
        prompt: str | None = None,
    ) -> np.ndarray:
        prepared_query = normalize_input(query)
        prepared_docs = normalize_inputs(documents)
        payload: dict[str, Any] = {
            "model": self.model,
            "query": prepared_query,
            "documents": prepared_docs,
        }
        if prompt:
            payload["prompt"] = prompt
        try:
            with backend_errors():
                response = self.client.post("/v1/rerank", json=payload)
            response.raise_for_status()
            data = response.json()
            raw_results = data.get("results", [])
            if len(raw_results) != len(prepared_docs):
                raise ValueError("reranker returned wrong number of scores")
            scores: list[float] = [0.0] * len(prepared_docs)
            seen_indices: set[int] = set()
            for item in raw_results:
                index = item.get("index", len(seen_indices))
                if not isinstance(index, int) or isinstance(index, bool):
                    raise ValueError("reranker result index must be an integer.")
                if index < 0 or index >= len(prepared_docs):
                    raise ValueError("reranker result index out of range.")
                if index in seen_indices:
                    raise ValueError("reranker returned duplicate indices.")
                seen_indices.add(index)
                score = float(item["score"])
                if not np.isfinite(score):
                    raise ValueError("reranker returned a non-finite score.")
                scores[index] = score
            if len(seen_indices) != len(prepared_docs):
                raise ValueError("reranker returned missing indices.")
            return np.asarray(scores, dtype=np.float32)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            if not self.fallback_to_local:
                # Surface authentication/status/parsing failures instead of
                # silently switching to a local model that may not be installed.
                raise
            # Fallback: cosine similarity between query and document text.
            from ..retrieval.multimodal_embedding import QwenVLEmbedder

            embedder = QwenVLEmbedder()
            query_vector = embedder.embed_text(prepared_query.get("text", ""))
            doc_vectors = embedder.encode(prepared_docs)
            norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            query_norm_value = float(np.linalg.norm(query_vector))
            query_norm = query_norm_value if query_norm_value > 0 else 1e-12
            scores = (doc_vectors @ query_vector) / (norms.flatten() * query_norm)
            return scores.astype(np.float32)

    def rerank(
        self,
        query: MultimodalInput,
        documents: Sequence[MultimodalInput],
        *,
        top_k: int | None = None,
        prompt: str | None = None,
    ) -> list[Any]:
        from ..retrieval.in_memory import SearchResult

        scores = self.score(query, documents, prompt=prompt)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            SearchResult(documents[int(i)], float(scores[i]), rank)
            for rank, i in enumerate(order, 1)
        ]

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "OpenAICompatibleRerankerProvider":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
