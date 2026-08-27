# Path: tests/contract/test_scenarios.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Contract tests for the runnable scenario cookbook."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "scenarios"
SCRIPTS = [
    "scenario_01_image_captioning.py",
    "scenario_02_visual_question_answering.py",
    "scenario_03_diagram_to_json.py",
    "scenario_04_document_page_to_rag_chunk.py",
    "scenario_05_text_only_visual_rag.py",
    "scenario_06_true_multimodal_retrieval.py",
    "scenario_07_compare_qwen_presets.py",
    "scenario_08_hub_vs_local_loading.py",
    "scenario_09_batch_structured_analysis.py",
]


def test_scenario_readme_exists() -> None:
    assert (SCENARIOS / "README.md").is_file()


@pytest.mark.parametrize("filename", SCRIPTS)
def test_scenario_help_is_runnable_without_loading_models(filename: str) -> None:
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(SCENARIOS / filename), "--help"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
