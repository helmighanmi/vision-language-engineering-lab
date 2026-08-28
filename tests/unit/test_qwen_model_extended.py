# Path: tests/unit/test_qwen_model_extended.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Extended unit tests for the Qwen3-VL model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vlm_engineering.qwen.model import QwenVLModel, _normalize_image_source


class FakeInputIDs:
    """Minimal input-id object exposing the shape contract used by QwenVLModel."""

    shape = (1, 3)


class FakeInputs(dict[str, Any]):
    """Mapping returned by the fake processor."""

    def __init__(self) -> None:
        super().__init__(
            {
                "input_ids": FakeInputIDs(),
            }
        )
        self.device: str | None = None

    def to(self, device: str) -> "FakeInputs":
        """Record the requested device and return the same mapping."""
        self.device = device
        return self


class FakeProcessor:
    """Minimal processor implementing the interface used by QwenVLModel."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] | None = None
        self.apply_kwargs: dict[str, Any] = {}
        self.decoded_tokens: Any = None
        self.skip_special_tokens: bool | None = None
        self.inputs = FakeInputs()

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> FakeInputs:
        """Record chat-template inputs and return deterministic fake tensors."""
        self.messages = messages
        self.apply_kwargs = kwargs
        return self.inputs

    def decode(
        self,
        tokens: Any,
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        """Record generated tokens and return deterministic text."""
        self.decoded_tokens = tokens
        self.skip_special_tokens = skip_special_tokens
        return "  Grounded visual answer.  "


class FakeModel:
    """Minimal model implementing the generation interface."""

    device = "cpu"

    def __init__(self) -> None:
        self.generate_kwargs: dict[str, Any] = {}

    def generate(
        self,
        **kwargs: Any,
    ) -> list[list[int]]:
        """Return prompt tokens followed by deterministic generated tokens."""
        self.generate_kwargs = kwargs

        # FakeInputIDs reports an input sequence length of 3.
        # QwenVLModel should therefore keep only [20, 21].
        return [
            [
                10,
                11,
                12,
                20,
                21,
            ]
        ]


def test_default_model_is_qwen3_vl_2b() -> None:
    """The default constructor should resolve to the lightweight 2B preset."""
    model = QwenVLModel()

    assert model.model_source == "Qwen/Qwen3-VL-2B-Instruct"
    assert model.local_files_only is False
    assert model.trust_remote_code is False


@pytest.mark.parametrize(
    ("model_size", "expected_model_id"),
    [
        (
            "2b",
            "Qwen/Qwen3-VL-2B-Instruct",
        ),
        (
            "4b",
            "Qwen/Qwen3-VL-4B-Instruct",
        ),
        (
            "8b",
            "Qwen/Qwen3-VL-8B-Instruct",
        ),
    ],
)
def test_model_size_presets_resolve_expected_model_ids(
    model_size: str,
    expected_model_id: str,
) -> None:
    """Friendly size aliases should resolve to official Qwen3-VL models."""
    model = QwenVLModel(
        model_size=model_size,
    )

    assert model.model_source == expected_model_id


def test_from_preset_creates_hub_backed_model() -> None:
    """from_preset should use Hub/cache mode."""
    model = QwenVLModel.from_preset(
        "4b",
    )

    assert model.model_source == "Qwen/Qwen3-VL-4B-Instruct"
    assert model.local_files_only is False


def test_from_hub_preserves_explicit_model_id() -> None:
    """An explicit compatible Hugging Face model ID should be preserved."""
    model = QwenVLModel.from_hub(
        "Qwen/Qwen3-VL-4B-Instruct",
    )

    assert model.model_source == "Qwen/Qwen3-VL-4B-Instruct"
    assert model.local_files_only is False


def test_from_local_resolves_model_directory(
    tmp_path: Path,
) -> None:
    """Local model paths should become absolute offline-only directories."""
    model_directory = tmp_path / "Qwen3-VL-2B-Instruct"
    model_directory.mkdir()

    model = QwenVLModel.from_local(
        model_directory,
    )

    assert model.model_source == str(model_directory.resolve())
    assert model.local_files_only is True


def test_from_local_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """A missing local model directory should fail immediately."""
    model_directory = tmp_path / "missing-model"

    with pytest.raises(FileNotFoundError):
        QwenVLModel.from_local(
            model_directory,
        )


def test_from_local_rejects_file_instead_of_directory(
    tmp_path: Path,
) -> None:
    """A normal file must not be accepted as a local model directory."""
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"model")

    with pytest.raises(
        ValueError,
        match="Model path is not a directory",
    ):
        QwenVLModel.from_local(
            model_file,
        )


def test_constructor_rejects_model_source_and_model_size_together() -> None:
    """Users must choose either an explicit source or a preset alias."""
    with pytest.raises(
        ValueError,
        match="Provide either model_source or model_size",
    ):
        QwenVLModel(
            "Qwen/custom-model",
            model_size="4b",
        )


def test_constructor_rejects_empty_model_source() -> None:
    """An explicitly supplied model source must contain a value."""
    with pytest.raises(
        ValueError,
        match="model_source must not be empty",
    ):
        QwenVLModel(
            "   ",
        )


def test_injected_model_and_processor_are_reused() -> None:
    """Injected components should avoid loading the heavy Transformers backend."""
    fake_model = FakeModel()
    fake_processor = FakeProcessor()

    wrapper = QwenVLModel(
        model=fake_model,
        processor=fake_processor,
    )

    model, processor = wrapper._ensure_loaded()

    assert model is fake_model
    assert processor is fake_processor


def test_generate_rejects_empty_prompt() -> None:
    """Generation requires a meaningful user prompt."""
    wrapper = QwenVLModel(
        model=FakeModel(),
        processor=FakeProcessor(),
    )

    with pytest.raises(
        ValueError,
        match="prompt must not be empty",
    ):
        wrapper.generate(
            "image.png",
            "   ",
        )


@pytest.mark.parametrize(
    "max_new_tokens",
    [
        0,
        -1,
        -100,
    ],
)
def test_generate_rejects_non_positive_max_new_tokens(
    max_new_tokens: int,
) -> None:
    """Generation length must be strictly positive."""
    wrapper = QwenVLModel(
        model=FakeModel(),
        processor=FakeProcessor(),
    )

    with pytest.raises(
        ValueError,
        match="max_new_tokens must be greater than zero",
    ):
        wrapper.generate(
            "image.png",
            "Describe the image.",
            max_new_tokens=max_new_tokens,
        )


def test_generate_uses_absolute_local_image_path(
    tmp_path: Path,
) -> None:
    """Local images must be sent to Transformers as normal filesystem paths."""
    image = tmp_path / "diagram.png"
    image.write_bytes(b"fake-image")

    fake_model = FakeModel()
    fake_processor = FakeProcessor()

    wrapper = QwenVLModel(
        model=fake_model,
        processor=fake_processor,
    )

    result = wrapper.generate(
        image,
        "Describe this architecture.",
        max_new_tokens=64,
    )

    assert result == "Grounded visual answer."

    assert fake_processor.messages is not None

    user_message = fake_processor.messages[-1]
    image_content = user_message["content"][0]

    assert image_content["type"] == "image"
    assert image_content["url"] == str(image.resolve())
    assert not image_content["url"].startswith("file://")


def test_generate_preserves_http_image_url() -> None:
    """Remote HTTP(S) image URLs should not be converted into local paths."""
    image_url = "https://example.com/architecture.png"

    fake_model = FakeModel()
    fake_processor = FakeProcessor()

    wrapper = QwenVLModel(
        model=fake_model,
        processor=fake_processor,
    )

    wrapper.generate(
        image_url,
        "Describe the image.",
    )

    assert fake_processor.messages is not None

    image_content = fake_processor.messages[-1]["content"][0]

    assert image_content == {
        "type": "image",
        "url": image_url,
    }


def test_generate_adds_system_prompt_when_provided(
    tmp_path: Path,
) -> None:
    """A supplied system prompt should precede the user image message."""
    image = tmp_path / "diagram.png"
    image.write_bytes(b"fake-image")

    fake_processor = FakeProcessor()

    wrapper = QwenVLModel(
        model=FakeModel(),
        processor=fake_processor,
    )

    wrapper.generate(
        image,
        "Describe the architecture.",
        system_prompt="Answer only from visible evidence.",
    )

    assert fake_processor.messages is not None
    assert len(fake_processor.messages) == 2

    assert fake_processor.messages[0] == {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Answer only from visible evidence.",
            }
        ],
    }

    assert fake_processor.messages[1]["role"] == "user"


def test_generate_passes_expected_chat_template_options(
    tmp_path: Path,
) -> None:
    """The processor should receive the generation-oriented chat-template flags."""
    image = tmp_path / "diagram.png"
    image.write_bytes(b"fake-image")

    fake_processor = FakeProcessor()

    wrapper = QwenVLModel(
        model=FakeModel(),
        processor=fake_processor,
    )

    wrapper.generate(
        image,
        "Describe the diagram.",
    )

    assert fake_processor.apply_kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }


def test_generate_moves_processor_inputs_to_model_device(
    tmp_path: Path,
) -> None:
    """Prepared model inputs should be moved to the model device."""
    image = tmp_path / "diagram.png"
    image.write_bytes(b"fake-image")

    fake_processor = FakeProcessor()

    wrapper = QwenVLModel(
        model=FakeModel(),
        processor=fake_processor,
    )

    wrapper.generate(
        image,
        "Describe the image.",
    )

    assert fake_processor.inputs.device == "cpu"


def test_generate_forwards_max_new_tokens_to_model(
    tmp_path: Path,
) -> None:
    """Generation length must be forwarded to the backend model."""
    image = tmp_path / "diagram.png"
    image.write_bytes(b"fake-image")

    fake_model = FakeModel()

    wrapper = QwenVLModel(
        model=fake_model,
        processor=FakeProcessor(),
    )

    wrapper.generate(
        image,
        "Describe the image.",
        max_new_tokens=123,
    )

    assert fake_model.generate_kwargs["max_new_tokens"] == 123


def test_generate_decodes_only_new_tokens(
    tmp_path: Path,
) -> None:
    """Prompt tokens should be removed before decoding the model response."""
    image = tmp_path / "diagram.png"
    image.write_bytes(b"fake-image")

    fake_processor = FakeProcessor()

    wrapper = QwenVLModel(
        model=FakeModel(),
        processor=fake_processor,
    )

    result = wrapper.generate(
        image,
        "Describe the image.",
    )

    assert result == "Grounded visual answer."
    assert fake_processor.decoded_tokens == [
        20,
        21,
    ]
    assert fake_processor.skip_special_tokens is True


# ---------------------------------------------------------------------------
# Image source normalization regression tests
# ---------------------------------------------------------------------------


def test_normalize_local_image_path(
    tmp_path: Path,
) -> None:
    """Path objects must remain normal local filesystem paths."""
    image = tmp_path / "diagram.png"
    image.write_bytes(b"test")

    result = _normalize_image_source(
        image,
    )

    assert result == str(image.resolve())
    assert not result.startswith("file://")


def test_normalize_local_image_string(
    tmp_path: Path,
) -> None:
    """String paths should resolve to absolute local filesystem paths."""
    image = tmp_path / "diagram.png"
    image.write_bytes(b"test")

    result = _normalize_image_source(
        str(image),
    )

    assert result == str(image.resolve())
    assert not result.startswith("file://")


def test_normalize_http_image_url() -> None:
    """HTTP URLs should pass through unchanged."""
    url = "http://example.com/diagram.png"

    assert _normalize_image_source(url) == url


def test_normalize_https_image_url() -> None:
    """HTTPS URLs should pass through unchanged."""
    url = "https://example.com/diagram.png"

    assert _normalize_image_source(url) == url


def test_normalize_file_uri_to_local_path(
    tmp_path: Path,
) -> None:
    """Legacy file:// references should be converted to filesystem paths."""
    image = tmp_path / "diagram.png"
    image.write_bytes(b"test")

    result = _normalize_image_source(
        image.resolve().as_uri(),
    )

    assert result == str(image.resolve())
    assert not result.startswith("file://")


def test_normalize_file_uri_decodes_escaped_path(
    tmp_path: Path,
) -> None:
    """Percent-escaped file URIs should resolve to their original path."""
    image = tmp_path / "diagram with spaces.png"
    image.write_bytes(b"test")

    result = _normalize_image_source(
        image.resolve().as_uri(),
    )

    assert result == str(image.resolve())


def test_missing_path_object_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Missing Path inputs should fail before backend inference."""
    image = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        _normalize_image_source(
            image,
        )


def test_directory_path_is_rejected(
    tmp_path: Path,
) -> None:
    """A directory should never be accepted as an image."""
    with pytest.raises(
        ValueError,
        match="Image path is not a file",
    ):
        _normalize_image_source(
            tmp_path,
        )


def test_empty_image_source_is_rejected() -> None:
    """An empty image source should fail with a clear validation error."""
    with pytest.raises(
        ValueError,
        match="image must not be empty",
    ):
        _normalize_image_source(
            "   ",
        )


def test_unknown_non_path_string_is_preserved() -> None:
    """Non-path strings are left for Transformers to interpret or validate."""
    encoded_or_backend_specific_value = "not-a-local-file"

    assert (
        _normalize_image_source(
            encoded_or_backend_specific_value,
        )
        == encoded_or_backend_specific_value
    )