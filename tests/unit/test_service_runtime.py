# Path: tests/unit/test_service_runtime.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for lazy service runtime wiring."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from vlm_engineering.exceptions import InputValidationError, OptionalDependencyError
from vlm_engineering.service.config import ServiceConfig
from vlm_engineering.service.runtime import ServiceRuntime, build_store

_SRC = Path(__file__).resolve().parents[2] / "src"


class _FakeProvider:
    def __init__(self) -> None:
        self.unload_calls = 0
        self.close_calls = 0

    def unload(self) -> None:
        self.unload_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _BareProvider:
    """Provider exposing no lifecycle methods at all."""


def _run(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_SRC)
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=120,
    )


# -- Lazy construction -------------------------------------------------------

def test_injected_providers_are_returned_verbatim() -> None:
    generator, embedder, reranker = _FakeProvider(), _FakeProvider(), _FakeProvider()
    runtime = ServiceRuntime(
        ServiceConfig(),
        generator=generator,
        embedder=embedder,
        reranker=reranker,
    )
    assert runtime.generator is generator
    assert runtime.embedder is embedder
    assert runtime.reranker is reranker


def test_no_model_backend_is_built_at_construction() -> None:
    runtime = ServiceRuntime(ServiceConfig())
    assert runtime.loaded_backends() == {
        "generator": False,
        "embedder": False,
        "reranker": False,
    }


def test_builders_run_once_under_concurrency() -> None:
    class _CountingRuntime(ServiceRuntime):
        def __init__(self) -> None:
            super().__init__(ServiceConfig())
            self.builds = 0

        def _build_generator(self) -> _FakeProvider:
            time.sleep(0.01)
            self.builds += 1
            return _FakeProvider()

    runtime = _CountingRuntime()
    results: list[_FakeProvider] = []

    def resolve() -> None:
        results.append(runtime.generator)

    threads = [threading.Thread(target=resolve) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert runtime.builds == 1
    assert len(results) == 8
    assert all(item is results[0] for item in results)
    assert runtime.loaded_backends()["generator"] is True


def test_qwen_generator_builds_without_loading_weights() -> None:
    from vlm_engineering.providers.qwen import QwenVLMProvider

    runtime = ServiceRuntime(ServiceConfig(model_size="2b"))
    provider = runtime.generator
    assert isinstance(provider, QwenVLMProvider)
    assert provider.model.model_source.endswith("Qwen3-VL-2B-Instruct")


@pytest.mark.parametrize(
    ("config", "expected_suffix"),
    [
        (ServiceConfig(model_size="8b"), "Qwen3-VL-8B-Instruct"),
        (ServiceConfig(model_id="Qwen/custom"), "Qwen/custom"),
        (ServiceConfig(model_path=Path("/models/local")), "/models/local"),
        (ServiceConfig(), "Qwen3-VL-2B-Instruct"),
    ],
)
def test_generator_model_selection_maps_to_qwen_model_source(
    config: ServiceConfig, expected_suffix: str
) -> None:
    provider = ServiceRuntime(config).generator
    assert provider.model.model_source.endswith(expected_suffix)


def test_embedder_and_reranker_build_without_loading_weights() -> None:
    runtime = ServiceRuntime(ServiceConfig(model_size="2b"))
    assert runtime.embedder.model.model_id == "Qwen/Qwen3-VL-Embedding-2B"
    assert runtime.reranker.model.model_id == "Qwen/Qwen3-VL-Reranker-2B"


def test_embedder_and_reranker_apply_profile_defaults(tmp_path: Path) -> None:
    pytest.importorskip("yaml", reason="the config extra is not installed")
    profile_path = tmp_path / "runtime.yaml"
    profile_path.write_text(
        "version: 1\n"
        "embedding:\n"
        "  batch_size: 4\n"
        "  normalize_embeddings: false\n"
        "reranker:\n"
        "  batch_size: 3\n"
        "  normalize_scores: true\n",
        encoding="utf-8",
    )
    runtime = ServiceRuntime(ServiceConfig(runtime_config_path=profile_path))
    assert runtime.embedder.model.batch_size == 4
    assert runtime.embedder.model.normalize_embeddings is False
    assert runtime.reranker.model.batch_size == 3
    assert runtime.reranker.model.normalize_scores is True


def test_openai_compatible_generator_uses_base_url_and_model() -> None:
    pytest.importorskip("httpx", reason="the service extra is not installed")
    runtime = ServiceRuntime(
        ServiceConfig(
            vlm_provider="openai-compatible",
            vlm_base_url="http://localhost:8000",
            vlm_model="qwen-vl",
            vlm_api_key="secret",
        )
    )
    provider = runtime.generator
    try:
        assert provider.model == "qwen-vl"
        assert provider.base_url == "http://localhost:8000"
        assert provider.client.headers["Authorization"] == "Bearer secret"
    finally:
        provider.close()


def test_retrieval_model_kwargs_follow_model_selection() -> None:
    assert ServiceRuntime(ServiceConfig())._retrieval_model_kwargs() == {}
    assert ServiceRuntime(ServiceConfig(model_size="8b"))._retrieval_model_kwargs() == {
        "model_size": "8b"
    }
    assert ServiceRuntime(ServiceConfig(model_id="Qwen/test"))._retrieval_model_kwargs() == {
        "model_id": "Qwen/test"
    }
    path = Path("/models/qwen")
    assert ServiceRuntime(ServiceConfig(model_path=path))._retrieval_model_kwargs() == {
        "model_path": path,
        "local_files_only": True,
    }


# -- Runtime profile ---------------------------------------------------------

def test_runtime_profile_is_none_without_a_path() -> None:
    assert ServiceRuntime(ServiceConfig()).runtime_profile is None


def test_runtime_profile_loads_yaml_once(tmp_path: Path) -> None:
    pytest.importorskip("yaml", reason="the config extra is not installed")
    profile_path = tmp_path / "runtime.yaml"
    profile_path.write_text(
        "version: 1\n"
        "embedding:\n"
        "  batch_size: 2\n"
        "  normalize_embeddings: false\n"
        "reranker:\n"
        "  normalize_scores: true\n"
        "rag:\n"
        "  top_k: 2\n"
        "  candidate_k: 5\n",
        encoding="utf-8",
    )
    runtime = ServiceRuntime(ServiceConfig(runtime_config_path=profile_path))
    profile = runtime.runtime_profile
    assert profile is not None
    assert profile.embedding.batch_size == 2
    assert profile.embedding.normalize_embeddings is False
    assert profile.reranker.normalize_scores is True
    assert runtime.runtime_profile is profile


def test_runtime_profile_missing_extra_names_both_extras(tmp_path: Path) -> None:
    profile_path = tmp_path / "runtime.yaml"
    profile_path.write_text("version: 1\n", encoding="utf-8")
    code = """
import importlib.abc
import sys
from pathlib import Path


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "yaml":
            raise ImportError("blocked: yaml")
        return None


sys.meta_path.insert(0, _Blocker())
from vlm_engineering.service.config import ServiceConfig
from vlm_engineering.service.runtime import ServiceRuntime

runtime = ServiceRuntime(ServiceConfig(runtime_config_path=Path(sys.argv[1])))
try:
    runtime.runtime_profile
except Exception as exc:
    assert "[service,config]" in str(exc), str(exc)
    raise SystemExit(0)
raise SystemExit(1)
"""
    result = _run(code, str(profile_path))
    assert result.returncode == 0, result.stderr


def test_runtime_profile_error_keeps_both_extra_hints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vlm_engineering.runtime_config as runtime_config

    profile_path = tmp_path / "runtime.yaml"
    profile_path.write_text("version: 1\n", encoding="utf-8")

    def _raise(_: object) -> None:
        raise OptionalDependencyError("missing PyYAML")

    monkeypatch.setattr(runtime_config, "load_runtime_config", _raise)
    runtime = ServiceRuntime(ServiceConfig(runtime_config_path=profile_path))
    with pytest.raises(OptionalDependencyError, match=r"\[service,config\]"):
        _ = runtime.runtime_profile


# -- Vector stores -----------------------------------------------------------

def test_injected_store_wins_over_construction() -> None:
    store = object()
    runtime = ServiceRuntime(ServiceConfig(), stores={"qdrant": store})
    assert runtime.store("qdrant") is store


def test_memory_store_is_cached() -> None:
    runtime = ServiceRuntime(ServiceConfig())
    first = runtime.store("memory")
    assert runtime.store("memory") is first


def test_unknown_store_kind_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="Unknown vector store"):
        build_store("redis")


def test_sqlite_store_requires_a_path() -> None:
    with pytest.raises(InputValidationError, match="require a path"):
        build_store("sqlite-vec")


def test_sqlite_store_is_built_with_the_given_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vlm_engineering.vectorstore.sqlite_vec as adapter

    captured: dict[str, object] = {}

    class _FakeStore:
        def __init__(self, path: object, **options: object) -> None:
            captured["path"] = path
            captured["options"] = options

    monkeypatch.setattr(adapter, "SqliteVecVectorStore", _FakeStore)
    path = tmp_path / "index.db"
    store = build_store("sqlite-vec", path=path, table="docs")
    assert isinstance(store, _FakeStore)
    assert captured == {"path": path, "options": {"table": "docs"}}


def test_qdrant_store_is_built_from_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import vlm_engineering.vectorstore.qdrant as adapter

    captured: dict[str, object] = {}

    class _FakeStore:
        def __init__(self, **options: object) -> None:
            captured.update(options)

    monkeypatch.setattr(adapter, "QdrantVectorStore", _FakeStore)
    store = build_store("qdrant", collection="docs")
    assert isinstance(store, _FakeStore)
    assert captured == {"collection": "docs"}


def test_persistent_stores_are_cached_per_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vlm_engineering.vectorstore.sqlite_vec as adapter

    class _FakeStore:
        def __init__(self, path: object, **options: object) -> None:
            self.path = path

    monkeypatch.setattr(adapter, "SqliteVecVectorStore", _FakeStore)
    runtime = ServiceRuntime(ServiceConfig())
    first = runtime.store("sqlite-vec", path=tmp_path / "a.db")
    second = runtime.store("sqlite-vec", path=tmp_path / "b.db")
    assert first is not second
    assert runtime.store("sqlite-vec", path=tmp_path / "a.db") is first


def test_missing_store_extra_names_the_stores_extra() -> None:
    code = """
import importlib.abc
import sys


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"sqlite_vec", "qdrant_client"}:
            raise ImportError(f"blocked: {fullname}")
        return None


sys.meta_path.insert(0, _Blocker())
from vlm_engineering.service.config import ServiceConfig
from vlm_engineering.service.runtime import ServiceRuntime

runtime = ServiceRuntime(ServiceConfig())
for kind in ("sqlite-vec", "qdrant"):
    try:
        runtime.store(kind, path="index.db")
    except Exception as exc:
        assert "[stores]" in str(exc), (kind, str(exc))
    else:
        raise SystemExit(1)
raise SystemExit(0)
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr


# -- Configured store --------------------------------------------------------

def test_configured_store_uses_injected_memory_store() -> None:
    store = object()
    runtime = ServiceRuntime(ServiceConfig(), stores={"memory": store})
    assert runtime.configured_store() is store


def test_configured_sqlite_store_resolves_inside_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vlm_engineering.service.runtime as runtime_module

    captured: dict[str, object] = {}

    def _fake_build_store(kind: str, *, path: object = None, **options: object) -> object:
        captured.update(kind=kind, path=path, options=options)
        return object()

    monkeypatch.setattr(runtime_module, "build_store", _fake_build_store)
    config = ServiceConfig(store="sqlite-vec", output_root=tmp_path, store_path="indexes/main")
    ServiceRuntime(config).configured_store()
    assert captured["kind"] == "sqlite-vec"
    assert captured["path"] == (tmp_path / "indexes" / "main").resolve()


def test_configured_qdrant_store_passes_connection_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vlm_engineering.service.runtime as runtime_module

    captured: dict[str, object] = {}

    def _fake_build_store(kind: str, *, path: object = None, **options: object) -> object:
        captured.update(kind=kind, options=options)
        return object()

    monkeypatch.setattr(runtime_module, "build_store", _fake_build_store)
    config = ServiceConfig(
        store="qdrant",
        qdrant_host="qdrant.internal",
        qdrant_port=6334,
        qdrant_collection="documents",
        qdrant_api_key="secret",
        qdrant_https=True,
    )
    ServiceRuntime(config).configured_store()
    assert captured["kind"] == "qdrant"
    assert captured["options"] == {
        "host": "qdrant.internal",
        "port": 6334,
        "collection": "documents",
        "api_key": "secret",
        "https": True,
    }


def test_runtime_exposes_the_configured_concurrency_limit() -> None:
    assert ServiceRuntime(ServiceConfig(max_concurrency=3)).limiter.limit == 3


# -- Lifecycle ---------------------------------------------------------------

def test_unload_releases_providers_and_is_idempotent() -> None:
    generator, embedder, reranker = _FakeProvider(), _FakeProvider(), _FakeProvider()
    runtime = ServiceRuntime(
        ServiceConfig(),
        generator=generator,
        embedder=embedder,
        reranker=reranker,
    )
    runtime.unload()
    runtime.close()

    for provider in (generator, embedder, reranker):
        assert provider.unload_calls == 1
        assert provider.close_calls == 1
    assert runtime.loaded_backends() == {
        "generator": False,
        "embedder": False,
        "reranker": False,
    }


def test_unload_tolerates_providers_without_lifecycle_methods() -> None:
    runtime = ServiceRuntime(ServiceConfig(), generator=_BareProvider())
    runtime.unload()
    assert runtime.loaded_backends()["generator"] is False


def test_constructing_runtime_does_not_import_torch() -> None:
    code = """
import sys
from vlm_engineering.service.config import ServiceConfig
from vlm_engineering.service.runtime import ServiceRuntime

runtime = ServiceRuntime(ServiceConfig())
assert runtime.loaded_backends() == {
    "generator": False,
    "embedder": False,
    "reranker": False,
}
assert "torch" not in sys.modules, "runtime construction imported torch"
print("ok")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_missing_store_errors_are_optional_dependency_errors() -> None:
    assert issubclass(OptionalDependencyError, RuntimeError)
