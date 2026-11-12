# Path: src/vlm_engineering/service/policy.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Request-boundary policy: media paths, output paths and model selection.

These checks run before any model or store is touched. Local references are
canonicalised with :meth:`Path.resolve`, so ``..`` segments and symlinks that
escape the configured root are rejected rather than silently followed.

Remote images are denied by default (``VLM_SERVICE_ALLOW_REMOTE_IMAGES=1``
opts in). Response size and content-type limits belong to the download step
when remote support is implemented end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlsplit

from ..exceptions import InputValidationError
from .config import MEDIA_ROOT_ENV, OUTPUT_ROOT_ENV, ServiceConfig

_REMOTE_SCHEMES = frozenset({"http", "https"})
_REMOTE_DISABLED = (
    "Remote images are disabled; set VLM_SERVICE_ALLOW_REMOTE_IMAGES=1 to enable them."
)


def _relative_to(path: Path, root: Path, root_name: str, label: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError:
        raise InputValidationError(f"{label} escapes {root_name}: {path}") from None
    return path


def resolve_media_reference(reference: str, config: ServiceConfig) -> str:
    """Return a safe local path string or an allowed remote URL.

    Raises:
        InputValidationError: blank input, unsupported scheme, disabled remote
            access, missing ``media_root``, or a path outside that root.
    """
    if not isinstance(reference, str) or not reference.strip():
        raise InputValidationError("image must be a non-empty path or URL.")
    value = reference.strip()
    parts = urlsplit(value)
    scheme = parts.scheme.lower()

    if scheme in _REMOTE_SCHEMES:
        if not config.allow_remote_images:
            raise InputValidationError(_REMOTE_DISABLED)
        if not parts.netloc:
            raise InputValidationError("Remote image URL must include a host.")
        return value
    if scheme and scheme != "file":
        raise InputValidationError(
            f"Unsupported image scheme {scheme!r}; use a local path or an http(s) URL."
        )

    raw_path = Path(unquote(parts.path)) if scheme == "file" else Path(value).expanduser()
    root = config.media_root
    if root is None:
        raise InputValidationError(f"Local image paths require {MEDIA_ROOT_ENV} to be configured.")
    resolved = _relative_to(raw_path.resolve(), root.resolve(), MEDIA_ROOT_ENV, "image")
    if not resolved.exists():
        raise InputValidationError(f"Image path does not exist: {resolved}")
    if not resolved.is_file():
        raise InputValidationError(f"Image path is not a file: {resolved}")
    return str(resolved)


def resolve_output_path(reference: str, config: ServiceConfig) -> Path:
    """Resolve a server-side output path inside ``VLM_SERVICE_OUTPUT_ROOT``.

    Absolute paths and ``..`` segments are rejected before resolution; the
    resolved path must still be inside the configured root, which also covers
    symlink escapes.
    """
    if not isinstance(reference, str) or not reference.strip():
        raise InputValidationError("output must be a non-empty relative path.")
    value = reference.strip()
    candidate = Path(value)
    if candidate.is_absolute():
        raise InputValidationError("output must be relative to VLM_SERVICE_OUTPUT_ROOT.")
    if ".." in candidate.parts:
        raise InputValidationError("output must not contain '..' segments.")
    root = config.output_root
    if root is None:
        raise InputValidationError(f"Server-side output requires {OUTPUT_ROOT_ENV} to be configured.")
    return _relative_to(
        (root / candidate).resolve(),
        root.resolve(),
        OUTPUT_ROOT_ENV,
        "output",
    )


def reject_model_overrides(
    *,
    model_size: str | None = None,
    model_id: str | None = None,
    model_path: str | None = None,
) -> None:
    """Reject per-request model selection; the service serves one configured model.

    Phase 2 keeps a single provider per capability. Honouring request-level
    overrides would either reload models per call or silently ignore the
    caller, so the API returns a 400 instead of pretending.
    """
    if model_size is not None or model_id is not None or model_path is not None:
        raise InputValidationError(
            "Per-request model selection is not supported; "
            "configure VLM_SERVICE_MODEL_SIZE, VLM_SERVICE_MODEL_ID or "
            "VLM_SERVICE_MODEL_PATH instead."
        )
