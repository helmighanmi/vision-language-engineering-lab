# Path: src/vlm_engineering/service/runtime.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Lazy runtime wiring for the HTTP service.

Constructing :class:`ServiceRuntime` never imports torch, transformers, FastAPI
or a vector-store adapter. Providers and stores are built on first use so that
``import vlm_engineering.service`` and ``create_app()`` stay cheap, and so a
base installation without model extras can still serve ``/health``.

Provider instances are cached behind a lock because Phase 2 handlers are plain
``def`` functions executed on Starlette's threadpool.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..exceptions import InputValidationError, OptionalDependencyError
from .config import ServiceConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..runtime_config import RuntimeConfig
    from ..vectorstore.protocol import VectorStore

_SERVICE_CONFIG_HINT = (
    'Runtime YAML profiles need the [config] extra: '
    'pip install "vision-language-engineering-lab[service,config]"'
)
_STORES_HINT = (
    'Persistent vector stores need the [stores] extra: '
    'pip install "vision-language-engineering-lab[service,stores]"'
)


def build_store(kind: str, *, path: str | Path | None = None, **options: Any) -> "VectorStore":
    """Instantiate a vector store by name without importing adapters eagerly."""
    if kind == "memory":
        from ..vectorstore.memory import MemoryVectorStore

        return MemoryVectorStore()
    if kind == "sqlite-vec":
        if path is None:
            raise InputValidationError("sqlite-vec stores require a path.")
        try:
            from ..vectorstore.sqlite_vec import SqliteVecVectorStore
        except OptionalDependencyError as exc:
            raise OptionalDependencyError(_STORES_HINT) from exc
        return SqliteVecVectorStore(path, **options)
    if kind == "qdrant":
        try:
            from ..vectorstore.qdrant import QdrantVectorStore
        except OptionalDependencyError as exc:
            raise OptionalDependencyError(_STORES_HINT) from exc
        return QdrantVectorStore(**options)
    raise InputValidationError(
        f"Unknown vector store {kind!r}; expected memory, sqlite-vec, or qdrant."
    )


class ServiceRuntime:
    """Own and lazily construct the service's model providers and stores.

    Args:
        config:
            Validated :class:`ServiceConfig`.
        generator:
            Optional pre-built VLM provider (tests, offline deployments).
        embedder:
            Optional pre-built embedding provider.
        reranker:
            Optional pre-built reranking provider.
        stores:
            Optional pre-built stores keyed by name; these bypass construction
            and are returned as-is.
        runtime_profile:
            Optional pre-loaded :class:`RuntimeConfig` YAML profile.
    """

    def __init__(
        self,
        config: ServiceConfig,
        *,
        generator: Any | None = None,
        embedder: Any | None = None,
        reranker: Any | None = None,
        stores: Mapping[str, Any] | None = None,
        runtime_profile: "RuntimeConfig | None" = None,
    ) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._generator = generator
        self._embedder = embedder
        self._reranker = reranker
        self._provided_stores = dict(stores or {})
        self._built_stores: dict[tuple[str, str | None], Any] = {}
        self._runtime_profile = runtime_profile

    # -- Providers ---------------------------------------------------------

    @property
    def generator(self) -> Any:
        """Return the VLM provider, building it on first access."""
        return self._get_or_build("_generator", self._build_generator)

    @property
    def embedder(self) -> Any:
        """Return the embedding provider, building it on first access."""
        return self._get_or_build("_embedder", self._build_embedder)

    @property
    def reranker(self) -> Any:
        """Return the reranking provider, building it on first access."""
        return self._get_or_build("_reranker", self._build_reranker)

    def _get_or_build(self, attribute: str, builder: Any) -> Any:
        with self._lock:
            value = getattr(self, attribute)
            if value is None:
                value = builder()
                setattr(self, attribute, value)
            return value

    def _build_generator(self) -> Any:
        config = self.config
        if config.vlm_provider == "openai-compatible":
            # Config validation guarantees these are set.
            from ..providers.openai_compatible import OpenAICompatibleVLMProvider

            return OpenAICompatibleVLMProvider(
                base_url=str(config.vlm_base_url),
                model=str(config.vlm_model),
                api_key=config.vlm_api_key,
                timeout=config.request_timeout,
            )
        from ..providers.qwen import QwenVLMProvider

        if config.model_path is not None:
            return QwenVLMProvider(model_source=config.model_path, local_files_only=True)
        if config.model_id is not None:
            return QwenVLMProvider(model_source=config.model_id)
        if config.model_size is not None:
            return QwenVLMProvider(model_size=config.model_size)
        return QwenVLMProvider()

    def _build_embedder(self) -> Any:
        from ..providers.qwen import QwenEmbeddingProvider

        kwargs = self._retrieval_model_kwargs()
        profile = self.runtime_profile
        if profile is not None:
            kwargs.setdefault("batch_size", profile.embedding.batch_size)
            if profile.embedding.dimensions is not None:
                kwargs.setdefault("dimensions", profile.embedding.dimensions)
            kwargs.setdefault("normalize_embeddings", profile.embedding.normalize_embeddings)
        return QwenEmbeddingProvider(**kwargs)

    def _build_reranker(self) -> Any:
        from ..providers.qwen import QwenRerankerProvider

        kwargs = self._retrieval_model_kwargs()
        profile = self.runtime_profile
        if profile is not None:
            kwargs.setdefault("batch_size", profile.reranker.batch_size)
            kwargs.setdefault("normalize_scores", profile.reranker.normalize_scores)
        return QwenRerankerProvider(**kwargs)

    def _retrieval_model_kwargs(self) -> dict[str, Any]:
        """Map the shared model selection onto embedder/reranker keyword names."""
        config = self.config
        if config.model_path is not None:
            return {"model_path": config.model_path, "local_files_only": True}
        if config.model_id is not None:
            return {"model_id": config.model_id}
        if config.model_size is not None:
            return {"model_size": config.model_size}
        return {}

    # -- Runtime profile ---------------------------------------------------

    @property
    def runtime_profile(self) -> "RuntimeConfig | None":
        """Load the optional YAML profile once, on first use."""
        with self._lock:
            if self._runtime_profile is not None:
                return self._runtime_profile
            path = self.config.runtime_config_path
            if path is None:
                return None
            from ..runtime_config import load_runtime_config

            try:
                self._runtime_profile = load_runtime_config(path)
            except OptionalDependencyError as exc:
                raise OptionalDependencyError(_SERVICE_CONFIG_HINT) from exc
            return self._runtime_profile

    # -- Vector stores -----------------------------------------------------

    def store(
        self, kind: str = "memory", *, path: str | Path | None = None, **options: Any
    ) -> "VectorStore":
        """Return a store by kind, constructing and caching it on first use.

        Pre-built stores supplied through ``stores=`` win over construction.
        Persistent stores are cached per ``(kind, path)`` pair.
        """
        provided = self._provided_stores.get(kind)
        if provided is not None:
            return provided
        key = (kind, str(path) if path is not None else None)
        with self._lock:
            if key not in self._built_stores:
                self._built_stores[key] = build_store(kind, path=path, **options)
            return self._built_stores[key]

    # -- Lifecycle ---------------------------------------------------------

    def loaded_backends(self) -> dict[str, bool]:
        """Report which lazy backends have been constructed (for readiness)."""
        with self._lock:
            return {
                "generator": self._generator is not None,
                "embedder": self._embedder is not None,
                "reranker": self._reranker is not None,
            }

    def unload(self) -> None:
        """Release providers and stores; safe to call more than once."""
        with self._lock:
            for attribute in ("_generator", "_embedder", "_reranker"):
                provider = getattr(self, attribute)
                if provider is None:
                    continue
                unload = getattr(provider, "unload", None)
                if callable(unload):
                    unload()
                close = getattr(provider, "close", None)
                if callable(close):
                    close()
                setattr(self, attribute, None)
            self._built_stores.clear()

    def close(self) -> None:
        """Alias for :meth:`unload`, used by the ASGI lifespan."""
        self.unload()
