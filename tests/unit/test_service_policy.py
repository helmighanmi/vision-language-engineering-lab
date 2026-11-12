# Path: tests/unit/test_service_policy.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for media, output and model-selection request policy."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vlm_engineering.exceptions import InputValidationError
from vlm_engineering.service.config import ServiceConfig
from vlm_engineering.service.policy import (
    reject_model_overrides,
    resolve_media_reference,
    resolve_output_path,
)


@pytest.fixture(name="roots")
def roots_fixture(tmp_path: Path) -> tuple[ServiceConfig, Path, Path]:
    media = tmp_path / "media"
    output = tmp_path / "output"
    media.mkdir()
    output.mkdir()
    return ServiceConfig(media_root=media, output_root=output), media, output


# -- Media -------------------------------------------------------------------

def test_local_image_inside_media_root_is_returned(
    roots: tuple[ServiceConfig, Path, Path],
) -> None:
    config, media, _ = roots
    image = media / "page.png"
    image.write_bytes(b"not-decoded-here")
    assert resolve_media_reference(str(image), config) == str(image.resolve())


def test_local_image_relative_to_media_root(
    roots: tuple[ServiceConfig, Path, Path],
) -> None:
    config, media, _ = roots
    image = media / "nested" / "page.png"
    image.parent.mkdir()
    image.write_bytes(b"x")
    assert resolve_media_reference(str(image), config) == str(image.resolve())


def test_media_reference_rejects_blank_values() -> None:
    with pytest.raises(InputValidationError, match="non-empty"):
        resolve_media_reference("   ", ServiceConfig())


def test_local_image_requires_media_root(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"x")
    with pytest.raises(InputValidationError, match="VLM_SERVICE_MEDIA_ROOT"):
        resolve_media_reference(str(image), ServiceConfig())


def test_local_image_outside_media_root_is_rejected(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"x")
    config = ServiceConfig(media_root=media)
    with pytest.raises(InputValidationError, match="escapes VLM_SERVICE_MEDIA_ROOT"):
        resolve_media_reference(str(outside), config)


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"x")
    link = media / "link.png"
    os.symlink(outside, link)
    config = ServiceConfig(media_root=media)
    with pytest.raises(InputValidationError, match="escapes VLM_SERVICE_MEDIA_ROOT"):
        resolve_media_reference(str(link), config)


def test_missing_or_directory_media_paths_are_rejected(
    roots: tuple[ServiceConfig, Path, Path],
) -> None:
    config, media, _ = roots
    with pytest.raises(InputValidationError, match="does not exist"):
        resolve_media_reference(str(media / "missing.png"), config)
    with pytest.raises(InputValidationError, match="not a file"):
        resolve_media_reference(str(media), config)


def test_remote_images_are_opt_in() -> None:
    config = ServiceConfig(allow_remote_images=False)
    with pytest.raises(InputValidationError, match="ALLOW_REMOTE_IMAGES"):
        resolve_media_reference("https://example.com/cat.png", config)
    enabled = ServiceConfig(allow_remote_images=True)
    assert (
        resolve_media_reference("https://example.com/cat.png", enabled)
        == "https://example.com/cat.png"
    )


def test_remote_url_requires_a_host() -> None:
    config = ServiceConfig(allow_remote_images=True)
    with pytest.raises(InputValidationError, match="host"):
        resolve_media_reference("https:///cat.png", config)


def test_unsupported_scheme_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="Unsupported image scheme"):
        resolve_media_reference("ftp://example.com/cat.png", ServiceConfig())


def test_file_uri_is_accepted_inside_media_root(
    roots: tuple[ServiceConfig, Path, Path],
) -> None:
    config, media, _ = roots
    image = media / "page.png"
    image.write_bytes(b"x")
    assert resolve_media_reference(image.as_uri(), config) == str(image.resolve())


# -- Output ------------------------------------------------------------------

def test_relative_output_path_resolves_inside_root(
    roots: tuple[ServiceConfig, Path, Path],
) -> None:
    config, _, output = roots
    assert resolve_output_path("index/main", config) == (output / "index" / "main").resolve()


def test_output_rejects_absolute_paths(roots: tuple[ServiceConfig, Path, Path]) -> None:
    config, _, _ = roots
    with pytest.raises(InputValidationError, match="relative"):
        resolve_output_path("/tmp/index", config)


def test_output_rejects_traversal(roots: tuple[ServiceConfig, Path, Path]) -> None:
    config, _, _ = roots
    with pytest.raises(InputValidationError, match="'..'"):
        resolve_output_path("../index", config)


def test_output_requires_configured_root() -> None:
    with pytest.raises(InputValidationError, match="VLM_SERVICE_OUTPUT_ROOT"):
        resolve_output_path("index", ServiceConfig())


def test_output_symlink_escape_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, output / "link")
    config = ServiceConfig(output_root=output)
    with pytest.raises(InputValidationError, match="escapes VLM_SERVICE_OUTPUT_ROOT"):
        resolve_output_path("link/index", config)


def test_output_rejects_blank_reference(roots: tuple[ServiceConfig, Path, Path]) -> None:
    config, _, _ = roots
    with pytest.raises(InputValidationError, match="non-empty"):
        resolve_output_path(" ", config)


# -- Model overrides ---------------------------------------------------------

def test_model_overrides_are_allowed_when_unset() -> None:
    reject_model_overrides(model_size=None, model_id=None, model_path=None)


@pytest.mark.parametrize(
    "selection",
    [
        {"model_size": "8b"},
        {"model_id": "Qwen/custom"},
        {"model_path": "/models/qwen"},
    ],
)
def test_any_per_request_model_override_is_rejected(selection: dict[str, str]) -> None:
    with pytest.raises(InputValidationError, match="Per-request model selection"):
        reject_model_overrides(**selection)  # type: ignore[arg-type]
