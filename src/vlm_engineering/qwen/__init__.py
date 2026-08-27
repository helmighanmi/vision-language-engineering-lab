# Path: src/vlm_engineering/qwen/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Qwen3-VL model loading and inference."""

from .downloader import download_model_snapshot
from .model import QwenVLModel

__all__ = ["QwenVLModel", "download_model_snapshot"]
