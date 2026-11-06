# Path: tests/integration/test_module_entrypoint.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Subprocess coverage for ``python -m vlm_engineering``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "vlm_engineering", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_module_entrypoint_prints_help() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "vlm-lab" in result.stdout


def test_module_entrypoint_reports_input_errors_with_exit_code_2() -> None:
    result = _run("embed")
    assert result.returncode == 2
    assert "vlm-lab:" in result.stderr
