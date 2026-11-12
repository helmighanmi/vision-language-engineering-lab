# Path: tests/contract/test_service_contract.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Contract tests for the optional HTTP service package."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"

_BLOCK_CODE = """
import importlib.abc
import sys


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"fastapi", "pydantic", "uvicorn", "httpx"}:
            raise ImportError(f"blocked extra dependency: {fullname}")
        return None


sys.meta_path.insert(0, _Blocker())
try:
    import vlm_engineering.service

    vlm_engineering.service.create_app
except ImportError as exc:
    assert "[service]" in str(exc), str(exc)
    raise SystemExit(0) from None
raise SystemExit(1)
"""


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )


def test_service_imports_without_models_and_exposes_create_app() -> None:
    """Importing the service requires no model backends and exports the API."""
    proc = _run(
        "import vlm_engineering.service as service;"
        " print(service.create_app.__name__)"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "create_app"


def test_missing_service_extra_raises_helpful_import_error() -> None:
    """Without fastapi/pydantic, the lazy export names the [service] extra."""
    proc = _run(_BLOCK_CODE)
    assert proc.returncode == 0, proc.stderr


def test_unknown_service_attribute_raises_attribute_error() -> None:
    proc = _run(
        "import vlm_engineering.service as service\n"
        "try:\n"
        "    service.not_a_real_name\n"
        "except AttributeError as exc:\n"
        "    assert 'not_a_real_name' in str(exc)\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_config_and_runtime_import_without_service_dependencies() -> None:
    """Config/runtime stay stdlib-only and construct no backends on import."""
    proc = _run(
        "import importlib.abc\n"
        "import sys\n"
        "\n"
        "\n"
        "class _Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in {'fastapi', 'pydantic', 'uvicorn', 'httpx', 'torch'}:\n"
        "            raise ImportError(f'blocked: {fullname}')\n"
        "        return None\n"
        "\n"
        "\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "from vlm_engineering.service.config import ServiceConfig\n"
        "from vlm_engineering.service.runtime import ServiceRuntime\n"
        "\n"
        "runtime = ServiceRuntime(ServiceConfig())\n"
        "assert runtime.loaded_backends() == {\n"
        "    'generator': False,\n"
        "    'embedder': False,\n"
        "    'reranker': False,\n"
        "}\n"
        "print('ok')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
