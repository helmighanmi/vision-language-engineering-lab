# Path: tests/unit/test_hardening_v04.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Regression tests for v0.4 reliability hardening."""

from __future__ import annotations

import json
import os

import pytest
from PIL import Image

from vlm_engineering import inputs
from vlm_engineering.documents.chunking import VisualChunkBuilder
from vlm_engineering.documents.schemas import (
    NativePageContent,
    VisualAnalysis,
    VisualChunk,
    VisualRelation,
)
from vlm_engineering.documents.visual_analysis import analyze_visual_document
from vlm_engineering.exceptions import InputValidationError, OptionalDependencyError, StructuredOutputError
from vlm_engineering.operational import backend_errors
from vlm_engineering.utils import extract_json_object


class SequenceGenerator:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls = 0

    def generate(self, _image: str, _prompt: str, *, max_new_tokens: int = 512) -> str:
        self.calls += 1
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return str(value)


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        "prefix\n```json\n{\"a\": 1}\n```\ntrailing prose",
        'first {"a": 1} second {"b": 2}',
        '{"outer": {"inner": "closing } inside a string"}}',
    ],
)
def test_extract_json_object_handles_model_output_variants(raw: str) -> None:
    assert isinstance(extract_json_object(raw), dict)


@pytest.mark.parametrize("raw", ["", "prose only", "[1, 2]", "```json\nnot json\n```"])
def test_extract_json_object_rejects_non_object_output(raw: str) -> None:
    with pytest.raises(ValueError):
        extract_json_object(raw)


def test_page_content_requires_a_positive_integer_page() -> None:
    for page in (0, -1, "1", True, None):
        with pytest.raises(InputValidationError, match="page"):
            NativePageContent("doc", page, "source.pdf")  # type: ignore[arg-type]


def test_visual_chunk_validates_identity_and_types() -> None:
    with pytest.raises(InputValidationError, match="chunk_id"):
        VisualChunk("", "doc", "source.pdf", 1, "diagram", "t", "text", "")
    with pytest.raises(InputValidationError, match="page"):
        VisualChunk("id", "doc", "source.pdf", 0, "diagram", "t", "text", "")
    with pytest.raises(InputValidationError, match="metadata"):
        VisualChunk("id", "doc", "source.pdf", 1, "diagram", "t", "text", "", metadata=[])  # type: ignore[arg-type]


def test_visual_chunk_round_trips_through_dict() -> None:
    chunk = VisualChunk(
        "id",
        "doc",
        "source.pdf",
        2,
        "diagram",
        "Title",
        "Text",
        "page.png",
        parent_context="Section",
        entities=("A", "B"),
        relations=(VisualRelation("A", "calls", "B", "arrow"),),
        metadata={"owner": "team"},
    )
    assert VisualChunk.from_dict(chunk.to_dict()) == chunk


def test_analysis_from_dict_does_not_split_scalar_strings() -> None:
    analysis = VisualAnalysis.from_dict(
        {
            "page_type": "diagram",
            "title": "T",
            "summary": "S",
            "entities": "Redis",
            "important_text": "POST /payments",
            "uncertainties": None,
        }
    )
    assert analysis.entities == ("Redis",)
    assert analysis.important_text == ("POST /payments",)
    assert analysis.uncertainties == ()


def test_chunk_id_is_stable_while_content_hash_tracks_changes() -> None:
    native = NativePageContent("doc", 3, "source.pdf", native_text="Exact ID: A-1", image_ref="page.png")
    builder = VisualChunkBuilder()
    first = builder.build(native, VisualAnalysis("diagram", "T", "S", entities=("A",)))
    second = builder.build(native, VisualAnalysis("diagram", "T", "S", entities=("B",)))
    assert first.chunk_id == second.chunk_id
    assert first.metadata["content_hash"] != second.metadata["content_hash"]
    assert first.metadata["has_visual_evidence"] is True


def test_structured_analysis_coerces_scalars_and_skips_incomplete_relations() -> None:
    response = json.dumps(
        {
            "page_type": "chart",
            "title": "Architecture",
            "summary": "A calls B",
            "entities": "Redis",
            "relations": [
                {"source": "A", "relation": "calls", "target": "B", "evidence": "arrow"},
                {"source": "A"},
                "not-an-object",
            ],
            "important_text": "POST /payments",
            "uncertainties": None,
        }
    )
    result = analyze_visual_document(SequenceGenerator(response), "page.png")
    assert result.page_type == "mixed"
    assert result.entities == ("Redis",)
    assert result.important_text == ("POST /payments",)
    assert len(result.relations) == 1
    assert any("Unrecognized page_type" in note for note in result.uncertainties)
    assert any("Skipped relation 2" in note for note in result.uncertainties)


def test_structured_analysis_retries_malformed_output_once() -> None:
    generator = SequenceGenerator(
        "not json",
        json.dumps(
            {
                "page_type": "diagram",
                "title": "T",
                "summary": "S",
                "entities": ["A"],
                "relations": [],
                "important_text": [],
                "uncertainties": [],
            }
        ),
    )
    result = analyze_visual_document(generator, "page.png")
    assert result.title == "T"
    assert generator.calls == 2


def test_structured_analysis_exhausts_retries_with_context() -> None:
    generator = SequenceGenerator("bad", "still bad", "bad again")
    with pytest.raises(StructuredOutputError, match="3 attempt"):
        analyze_visual_document(generator, "page.png", retries=2)
    assert generator.calls == 3


def test_structured_analysis_does_not_retry_generator_failures() -> None:
    generator = SequenceGenerator(RuntimeError("model crashed"))
    with pytest.raises(RuntimeError, match="model crashed"):
        analyze_visual_document(generator, "page.png")
    assert generator.calls == 1


def test_structured_analysis_rejects_invalid_retry_count() -> None:
    for retries in (-1, 1.5, True):
        with pytest.raises(ValueError, match="retries"):
            analyze_visual_document(SequenceGenerator("{}"), "page.png", retries=retries)  # type: ignore[arg-type]


def test_image_validation_cache_reuses_unchanged_files_and_detects_deletion(tmp_path) -> None:
    path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "blue").save(path)
    inputs.clear_image_validation_cache()

    assert inputs.validate_image(path) == str(path.resolve())
    info = inputs._validate_cached.cache_info()
    assert info.misses == 1 and info.hits == 0

    assert inputs.validate_image(str(path)) == str(path.resolve())
    info = inputs._validate_cached.cache_info()
    assert info.hits == 1

    Image.new("RGB", (9, 9), "red").save(path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns + 1_000_000, stat.st_mtime_ns + 1_000_000))
    inputs.validate_image(path)
    assert inputs._validate_cached.cache_info().misses == 2

    path.unlink()
    with pytest.raises(FileNotFoundError):
        inputs.validate_image(path)


def test_image_validation_accepts_local_file_uri(tmp_path) -> None:
    path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "blue").save(path)
    assert inputs.validate_image(path.as_uri()) == str(path.resolve())


@pytest.mark.parametrize(
    ("error", "fragment"),
    [
        (ImportError("No module named 'yaml'"), "config"),
        (ImportError("No module named 'sentence_transformers'"), "retrieval"),
        (ImportError("No module named 'torch'"), "qwen"),
        (ImportError("No module named 'mystery'"), "relevant extra"),
    ],
)
def test_import_error_hint_matches_missing_dependency(error: ImportError, fragment: str) -> None:
    with pytest.raises(OptionalDependencyError, match=fragment):
        with backend_errors():
            raise error


def test_operational_import_error_keeps_original_cause() -> None:
    error = ImportError("No module named 'yaml'")
    with pytest.raises(OptionalDependencyError) as caught:
        with backend_errors():
            raise error
    assert caught.value.__cause__ is error
