# Path: tests/unit/test_nuextract3.py
# Unit tests for the NuExtract3 document-extraction integration.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vlm_engineering.exceptions import InputValidationError, StructuredOutputError
from vlm_engineering.extraction import (
    NUEXTRACT3_MODEL,
    NuExtract3Extractor,
    NuExtract3Generator,
    create_nuextract3_extractor,
    load_extraction_template,
)


class FakeGenerator:
    """Deterministic generator that records calls and returns queued responses."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        images: Any,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "images": list(images),
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "system_prompt": system_prompt,
            }
        )
        if not self.responses:
            raise AssertionError("No queued response left for generate()")
        return self.responses.pop(0)


def test_preset_metadata_is_distinct_from_qwen_registry() -> None:
    assert NUEXTRACT3_MODEL.model_id == "numind/NuExtract3"
    assert NUEXTRACT3_MODEL.gguf_model_id == "numind/NuExtract3-GGUF"
    assert NUEXTRACT3_MODEL.parameter_class == "~4B"
    assert NUEXTRACT3_MODEL.recommended_image_min_tokens > 0


def test_generator_protocol_recognizes_compatible_objects() -> None:
    assert isinstance(FakeGenerator('{"ok": true}'), NuExtract3Generator)


@pytest.mark.parametrize(
    "template",
    [
        {"invoice_number": "", "total": 0.0},
        '{"name": "", "age": 0}',
    ],
)
def test_load_template_from_mapping_and_json_string(template: Any) -> None:
    assert load_extraction_template(template) == (
        template if isinstance(template, dict) else json.loads(template)
    )


def test_load_template_from_file(tmp_path: Path) -> None:
    path = tmp_path / "template.json"
    path.write_text('{"title": "", "pages": []}', encoding="utf-8")
    assert load_extraction_template(path) == {"title": "", "pages": []}
    assert load_extraction_template(str(path)) == {"title": "", "pages": []}


@pytest.mark.parametrize(
    "bad_template",
    [
        [],
        "",
        "not-json-and-not-a-path",
        "[]",
        {},
    ],
)
def test_load_template_rejects_invalid_inputs(bad_template: Any) -> None:
    with pytest.raises(InputValidationError):
        load_extraction_template(bad_template)


def test_extract_returns_parsed_json_and_builds_prompt() -> None:
    generator = FakeGenerator('```json\n{"name": "Ada", "score": 9}\n```')
    extractor = NuExtract3Extractor(generator)

    result = extractor.extract(
        ["https://example.com/page.png"],
        {"name": "", "score": 0},
        instructions="Use exact values from the page.",
    )

    assert result == {"name": "Ada", "score": 9}
    assert len(generator.calls) == 1
    call = generator.calls[0]
    assert call["images"] == ["https://example.com/page.png"]
    assert '"name": ""' in call["prompt"]
    assert '"score": 0' in call["prompt"]
    assert "Use exact values from the page." in call["prompt"]
    assert "NuExtract3" in call["system_prompt"]
    assert call["max_new_tokens"] == NUEXTRACT3_MODEL.default_max_new_tokens


def test_extract_supports_single_image_and_thinking_flag() -> None:
    generator = FakeGenerator('{"ok": true}')
    extractor = NuExtract3Extractor(generator)

    result = extractor.extract("page.png", {"ok": False}, enable_thinking=True)

    assert result == {"ok": True}
    assert "Think carefully" in generator.calls[0]["prompt"]


def test_extract_retries_malformed_json_then_succeeds() -> None:
    generator = FakeGenerator("not json", '{"value": 42}')
    extractor = NuExtract3Extractor(generator, retries=1)

    assert extractor.extract(["page.png"], {"value": 0}) == {"value": 42}
    assert len(generator.calls) == 2


def test_extract_raises_structured_error_after_retries() -> None:
    generator = FakeGenerator("nope", "still nope")
    extractor = NuExtract3Extractor(generator, retries=1)

    with pytest.raises(StructuredOutputError, match="Invalid structured extraction"):
        extractor.extract(["page.png"], {"value": 0})
    assert len(generator.calls) == 2


def test_extract_rejects_array_json_as_structured_output() -> None:
    generator = FakeGenerator("[1, 2, 3]", "[1, 2, 3]")
    extractor = NuExtract3Extractor(generator, retries=1)

    with pytest.raises(StructuredOutputError, match="JSON object"):
        extractor.extract(["page.png"], {"value": 0})
    assert len(generator.calls) == 2


def test_extract_markdown_returns_clean_text() -> None:
    generator = FakeGenerator("  # Invoice\n\nTotal: 10.50  ")
    extractor = NuExtract3Extractor(generator)

    markdown = extractor.extract_markdown(
        ["page.png"], instructions="Preserve table structure."
    )

    assert markdown == "# Invoice\n\nTotal: 10.50"
    call = generator.calls[0]
    assert "Transcribe ALL visible content" in call["system_prompt"]
    assert "Preserve table structure." in call["prompt"]


def test_extract_markdown_rejects_empty_response() -> None:
    generator = FakeGenerator("   ")
    extractor = NuExtract3Extractor(generator)

    with pytest.raises(StructuredOutputError, match="empty response"):
        extractor.extract_markdown(["page.png"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"images": []},
    ],
)
def test_extract_validates_images(kwargs: Any) -> None:
    generator = FakeGenerator('{"x": 1}')
    extractor = NuExtract3Extractor(generator)
    with pytest.raises(InputValidationError):
        extractor.extract(kwargs["images"], {"x": 0})


def test_extractor_validates_construction_arguments() -> None:
    with pytest.raises(InputValidationError):
        NuExtract3Extractor(object())  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        NuExtract3Extractor(FakeGenerator(), retries=-1)
    with pytest.raises(InputValidationError):
        NuExtract3Extractor(FakeGenerator(), max_new_tokens=0)


def test_extract_validates_per_call_arguments() -> None:
    generator = FakeGenerator('{"x": 1}')
    extractor = NuExtract3Extractor(generator)
    with pytest.raises(InputValidationError):
        extractor.extract(["page.png"], {"x": 0}, max_new_tokens=0)
    with pytest.raises(InputValidationError):
        extractor.extract(["page.png"], {"x": 0}, retries=-1)
    with pytest.raises(InputValidationError):
        extractor.extract(["page.png"], {"x": 0}, instructions=123)  # type: ignore[arg-type]


def test_factory_builds_extractor_with_openai_compatible_provider() -> None:
    extractor = create_nuextract3_extractor(
        base_url="http://localhost:1234/v1", model="NuExtract3"
    )
    assert isinstance(extractor, NuExtract3Extractor)
    assert isinstance(extractor.generator, NuExtract3Generator)
    assert extractor.generator.model == "NuExtract3"  # type: ignore[attr-defined]
    extractor.generator.client.close()  # type: ignore[attr-defined]


def test_factory_rejects_blank_model() -> None:
    with pytest.raises(InputValidationError):
        create_nuextract3_extractor(base_url="http://localhost:1234/v1", model="   ")


def test_cli_extract_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any) -> None:
    from vlm_engineering import cli

    calls: list[dict[str, Any]] = []

    def fake_factory(**kwargs: Any) -> NuExtract3Extractor:
        calls.append(kwargs)
        return NuExtract3Extractor(FakeGenerator('{"ok": true}'))

    monkeypatch.setattr(cli, "create_nuextract3_extractor", fake_factory)
    output = tmp_path / "result.json"

    assert cli.main(
        [
            "extract",
            "https://example.com/page.png",
            "--template",
            '{"ok": false}',
            "--base-url",
            "http://localhost:1234/v1",
            "--output",
            str(output),
        ]
    ) == 0

    assert calls[0]["base_url"] == "http://localhost:1234/v1"
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
    assert "result.json" in capsys.readouterr().out


def test_cli_extract_requires_template_or_markdown(capsys: Any) -> None:
    from vlm_engineering import cli

    assert cli.main(["extract", "page.png"]) == 2
    assert "--template" in capsys.readouterr().err
