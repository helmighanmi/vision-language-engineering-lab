# Path: src/vlm_engineering/service/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Vision-Language Engineering Lab HTTP service.

Exposes the package's retrieval, ingestion and generation capabilities as a
FastAPI application with auth, rate limiting, structured logging and metrics.

The implementation ships behind the ``service`` extra:

    pip install -e ".[service]"

``app`` and ``create_app`` resolve lazily (PEP 562) so importing this
package without the extra raises a helpful ``ImportError`` naming the
extra, instead of an opaque missing-module error.
"""

from __future__ import annotations

from typing import Any

__all__ = ["app", "create_app"]

_INSTALL_HINT = (
    "The HTTP service requires the [service] extra: "
    'pip install "vision-language-engineering-lab[service]"'
)


def __getattr__(name: str) -> Any:
    """Resolve ``app`` and ``create_app`` lazily from the app module."""
    if name in ("app", "create_app"):
        try:
            from .app import app as _app
            from .app import create_app as _create_app
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        globals()["create_app"] = _create_app
        globals()["app"] = _app
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily exported names in ``dir()`` output."""
    return sorted(set(globals()) | set(__all__))
