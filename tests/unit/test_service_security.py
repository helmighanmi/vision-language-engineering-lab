# Path: tests/unit/test_service_security.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for optional API-key authentication."""

from __future__ import annotations

import pytest

from vlm_engineering.service.config import ServiceConfig
from vlm_engineering.service.security import (
    API_KEY_HEADER,
    AuthenticationError,
    authenticate,
    extract_api_key,
)


def test_extract_bearer_token() -> None:
    assert extract_api_key({"authorization": "Bearer secret"}) == "secret"
    assert extract_api_key({"Authorization": "bearer secret"}) == "secret"


def test_extract_x_api_key_header() -> None:
    assert extract_api_key({API_KEY_HEADER: "secret"}) == "secret"
    assert extract_api_key({"x-api-key": "  secret  "}) == "secret"


def test_extract_prefers_bearer_and_ignores_other_schemes() -> None:
    headers = {"authorization": "Basic abc", API_KEY_HEADER: "secret"}
    assert extract_api_key(headers) == "secret"
    assert extract_api_key({"authorization": "Basic abc"}) is None


def test_extract_returns_none_for_blank_values() -> None:
    assert extract_api_key({}) is None
    assert extract_api_key({"authorization": "Bearer "}) is None
    assert extract_api_key({API_KEY_HEADER: "   "}) is None


def test_authenticate_is_a_noop_without_a_configured_key() -> None:
    authenticate({}, ServiceConfig())


def test_authenticate_accepts_a_matching_key_from_either_header() -> None:
    config = ServiceConfig(api_key="secret")
    authenticate({"authorization": "Bearer secret"}, config)
    authenticate({API_KEY_HEADER: "secret"}, config)


def test_authenticate_rejects_missing_or_wrong_key() -> None:
    config = ServiceConfig(api_key="secret")
    with pytest.raises(AuthenticationError):
        authenticate({}, config)
    with pytest.raises(AuthenticationError):
        authenticate({"authorization": "Bearer nope"}, config)
    with pytest.raises(AuthenticationError):
        authenticate({API_KEY_HEADER: "nope"}, config)


def test_authenticate_handles_non_ascii_keys() -> None:
    """Non-ASCII keys must compare safely instead of raising TypeError."""
    config = ServiceConfig(api_key="clé-secrète")
    authenticate({API_KEY_HEADER: "clé-secrète"}, config)
    with pytest.raises(AuthenticationError):
        authenticate({API_KEY_HEADER: "clé-fausse"}, config)


def test_config_redacts_the_api_key_in_repr() -> None:
    config = ServiceConfig(api_key="secret")
    assert "secret" not in repr(config)
