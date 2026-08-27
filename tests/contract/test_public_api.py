# Path: tests/contract/test_public_api.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Contract tests for the public package API."""

from __future__ import annotations

import argparse

import vlm_engineering
from vlm_engineering.cli import build_parser


def test_public_package_exports_expected_symbols() -> None:
    """The main package should expose its supported public API."""

    expected_exports = {
        "CLIPEncoder",
        "QwenVLModel",
        "VisualRAGPipeline",
    }

    missing = [
        name
        for name in expected_exports
        if not hasattr(vlm_engineering, name)
    ]

    assert not missing, f"Missing public exports: {missing}"


def test_cli_parser_can_be_created() -> None:
    """The CLI parser should be constructible without loading ML models."""

    parser = build_parser()

    assert isinstance(parser, argparse.ArgumentParser)


def test_cli_exposes_expected_commands() -> None:
    """The CLI should expose its documented public commands."""

    parser = build_parser()

    subparsers = next(
        action
        for action in parser._actions
        if action.dest == "command"
    )

    choices = subparsers.choices

    assert choices is not None

    assert set(choices) == {
        "describe",
        "analyze",
        "chunk",
        "download-model",
    }


def test_package_has_documentation() -> None:
    """The package module should include top-level documentation."""

    assert vlm_engineering.__doc__ is not None
    assert vlm_engineering.__doc__.strip()