# Path: src/vlm_engineering/runtime_config.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Versioned YAML runtime profiles; parsing never imports a model backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .exceptions import InputValidationError, OptionalDependencyError
from .inputs import boolean, positive_int
from .retrieval.model_options import EMBEDDING_DIMENSIONS, RetrievalModelOptions, validate_dimensions


@dataclass(frozen=True)
class EmbeddingConfig:
    model: RetrievalModelOptions = field(default_factory=RetrievalModelOptions)
    batch_size: int = 1
    dimensions: int | None = None
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model, RetrievalModelOptions):
            raise InputValidationError("embedding.model must be RetrievalModelOptions.")
        positive_int(self.batch_size, "embedding.batch_size")
        boolean(self.normalize_embeddings, "normalize_embeddings")
        validate_dimensions(self.dimensions, EMBEDDING_DIMENSIONS.get(self.model.source("Embedding")))


@dataclass(frozen=True)
class RerankerConfig:
    model: RetrievalModelOptions = field(default_factory=RetrievalModelOptions)
    batch_size: int = 1
    normalize_scores: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model, RetrievalModelOptions):
            raise InputValidationError("reranker.model must be RetrievalModelOptions.")
        positive_int(self.batch_size, "reranker.batch_size")
        boolean(self.normalize_scores, "normalize_scores")


@dataclass(frozen=True)
class RAGConfig:
    top_k: int = 3
    candidate_k: int = 12
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        for name in ("top_k", "candidate_k", "max_new_tokens"):
            positive_int(getattr(self, name), f"rag.{name}")
        if self.candidate_k < self.top_k:
            raise InputValidationError("candidate_k must be greater than or equal to top_k.")


@dataclass(frozen=True)
class RuntimeConfig:
    version: int = 1
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise InputValidationError("Unsupported config version; expected integer 1.")
        for name, cls in (("embedding", EmbeddingConfig), ("reranker", RerankerConfig), ("rag", RAGConfig)):
            if not isinstance(getattr(self, name), cls):
                raise InputValidationError(f"{name} has an invalid configuration type.")

    def build_pipeline(self, generator: Any) -> Any:
        """Construct lazy wrappers using this profile and a caller-supplied generator."""
        from .retrieval import MultimodalRAGPipeline, QwenVLEmbedder, QwenVLReranker

        embedding = asdict(self.embedding)
        reranker = asdict(self.reranker)
        embedder = QwenVLEmbedder(**embedding.pop("model"), **embedding)
        scorer = QwenVLReranker(**reranker.pop("model"), **reranker)
        return MultimodalRAGPipeline(embedder, generator, scorer, **asdict(self.rag))


def _mapping(value: Any, cls: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputValidationError(f"{location} must be a mapping.")
    unknown = set(value) - {item.name for item in fields(cls)}
    if unknown:
        raise InputValidationError(f"Unknown keys in {location}: {', '.join(map(str, unknown))}.")
    return dict(value)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    try:
        import yaml
    except ImportError as exc:
        raise OptionalDependencyError(
            'Install "vision-language-engineering-lab[config]" for YAML profiles.'
        ) from exc
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InputValidationError(f"Invalid YAML configuration: {exc}") from exc
    values = _mapping(data, RuntimeConfig, "root")
    if "version" not in values:
        raise InputValidationError("Configuration requires an explicit version: 1.")
    for name, cls in (("embedding", EmbeddingConfig), ("reranker", RerankerConfig)):
        if name in values:
            section = _mapping(values[name], cls, name)
            if "model" in section:
                section["model"] = RetrievalModelOptions(
                    **_mapping(section["model"], RetrievalModelOptions, f"{name}.model")
                )
            values[name] = cls(**section)
    if "rag" in values:
        values["rag"] = RAGConfig(**_mapping(values["rag"], RAGConfig, "rag"))
    return RuntimeConfig(**values)
