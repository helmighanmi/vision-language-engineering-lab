# Path: src/vlm_engineering/service/security.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Optional API-key authentication.

Authentication is off unless ``VLM_SERVICE_API_KEY`` is set, which is the
local-development default and is called out in every doc example. When a key is
configured, callers present either ``Authorization: Bearer <key>`` or
``X-API-Key: <key>``; comparison uses :func:`secrets.compare_digest` so a wrong
key cannot be recovered from timing.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping

from fastapi import Request

from .config import ServiceConfig

API_KEY_HEADER = "X-API-Key"


class AuthenticationError(RuntimeError):
    """Raised when a protected route is called without the configured key."""

    def __init__(self, message: str = "Missing or invalid API key.") -> None:
        super().__init__(message)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Look up a header case-insensitively for plain mappings and ASGI dicts."""
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def extract_api_key(headers: Mapping[str, str]) -> str | None:
    """Read the presented key from either supported header."""
    authorization = _header(headers, "authorization")
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    api_key = _header(headers, API_KEY_HEADER)
    if api_key and api_key.strip():
        return api_key.strip()
    return None


def authenticate(headers: Mapping[str, str], config: ServiceConfig) -> None:
    """Validate the request against ``config.api_key``; no-op when unset.

    Keys are compared as UTF-8 bytes because :func:`secrets.compare_digest`
    rejects non-ASCII ``str`` values; encoding keeps the comparison
    constant-time for any Unicode key instead of raising.
    """
    if config.api_key is None:
        return
    presented = extract_api_key(headers)
    if presented is None or not secrets.compare_digest(
        presented.encode("utf-8"),
        config.api_key.encode("utf-8"),
    ):
        raise AuthenticationError()


def require_api_key(request: Request) -> None:
    """FastAPI dependency enforcing authentication for a protected route."""
    config: ServiceConfig = request.app.state.config
    authenticate(request.headers, config)
