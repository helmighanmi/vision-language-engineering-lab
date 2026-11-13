# Path: tests/unit/test_cli_serve.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""CLI tests for the ``vlm-lab serve`` command."""

from __future__ import annotations

import sys
import types
from argparse import Namespace

import pytest

from vlm_engineering import cli
from vlm_engineering.exceptions import OptionalDependencyError


def test_serve_parser_defaults() -> None:
    args = cli.build_parser().parse_args(["serve"])
    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.reload is False
    assert args.log_level == "info"


def test_serve_parser_accepts_explicit_bind_options() -> None:
    args = cli.build_parser().parse_args(
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--reload",
            "--log-level",
            "debug",
        ]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.reload is True
    assert args.log_level == "debug"


def test_cmd_serve_runs_uvicorn_with_the_app_import_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    module = types.ModuleType("uvicorn")

    def fake_run(app: str, **kwargs: object) -> None:
        captured["app"] = app
        captured.update(kwargs)

    module.run = fake_run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", module)

    args = Namespace(host="0.0.0.0", port=9000, reload=True, log_level="debug")
    assert cli.cmd_serve(args) == 0
    assert captured == {
        "app": "vlm_engineering.service.app:app",
        "host": "0.0.0.0",
        "port": 9000,
        "reload": True,
        "log_level": "debug",
    }


def test_cmd_serve_without_the_extra_names_the_service_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    args = Namespace(host="127.0.0.1", port=8000, reload=False, log_level="info")
    with pytest.raises(OptionalDependencyError, match=r"\[service\]"):
        cli.cmd_serve(args)


def test_main_reports_a_missing_service_extra(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert cli.main(["serve"]) == 2
    assert "[service]" in capsys.readouterr().err
