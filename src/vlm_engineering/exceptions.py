# Path: src/vlm_engineering/exceptions.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Project-specific exceptions."""


class VLMEngineeringError(RuntimeError):
    """Base exception for project failures."""


class OptionalDependencyError(VLMEngineeringError):
    """Raised when an optional runtime dependency is missing."""


class ModelLoadError(VLMEngineeringError):
    """Raised when a model cannot be loaded."""


class StructuredOutputError(VLMEngineeringError):
    """Raised when model output cannot be validated as structured data."""
