# Path: src/vlm_engineering/operational.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Map recognized operational failures; preserve unknown backend/programming errors."""

from __future__ import annotations

import errno
from collections.abc import Iterator
from contextlib import contextmanager

from .exceptions import ModelLoadError, OptionalDependencyError, VLMEngineeringError


def operational_error(exc: Exception, *, loading: bool = False) -> VLMEngineeringError | None:
    message = str(exc).lower()
    name = type(exc).__name__
    hint = None
    if isinstance(exc, ImportError):
        return OptionalDependencyError(
            'Missing optional dependency. Install "vision-language-engineering-lab[qwen-retrieval]" '
            "(includes sentence-transformers[image], torchvision and qwen-vl-utils). "
            "Check the chained exception for the missing module."
        )
    if isinstance(exc, MemoryError) or "out of memory" in message or "cannot allocate memory" in message:
        hint = (
            "Memory exhausted. Use 2B, batch_size=1, smaller images, unload unused models, or add RAM/VRAM."
        )
    elif isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
        hint = "Disk full or quota exceeded. Free space in your model/cache directory and retry the download."
    elif "torchvision" in message or "operator torchvision::" in message:
        hint = (
            "Torch/torchvision installation is incompatible. Install a matching pair for your CPU/CUDA build."
        )
    elif "qwen_vl_utils" in message or "qwen-vl-utils" in message:
        hint = "Install qwen-vl-utils using the package [qwen-retrieval] extra."
    elif "offload" in message and "disk" in message:
        hint = "Disk offloading requires Accelerate configuration and writable storage; use more memory or configure the backend explicitly."
    elif name == "RevisionNotFoundError":
        hint = (
            "Model revision not found. Check revision against the model repository's branches/tags/commits."
        )
    elif name == "LocalEntryNotFoundError" or "offline" in message:
        hint = "Offline model files are missing. Download a complete snapshot online, then use model_path."
    elif name in {"GatedRepoError", "RepositoryNotFoundError", "HfHubHTTPError"}:
        hint = "Check the model ID, access rights and revision. For 401/403, log in with hf auth login or set HF_TOKEN."
    elif name == "SafetensorError" or "safetensor" in message:
        hint = "Model weights are corrupt or incomplete. Download the affected snapshot again into your chosen model directory."
    elif loading and isinstance(exc, OSError):
        hint = "Cannot load the model snapshot. Check local config/tokenizer/processor/weight files, model ID, revision, access and offline settings."
    if hint is None:
        return None
    return ModelLoadError(hint) if loading else VLMEngineeringError(hint)


@contextmanager
def backend_errors(*, loading: bool = False) -> Iterator[None]:
    try:
        yield
    except Exception as exc:
        # Classification needs optional backend exception types without importing
        # every backend. Unknown exceptions are re-raised unchanged, with traceback.
        mapped = operational_error(exc, loading=loading)
        if mapped is None:
            raise
        raise mapped from exc
