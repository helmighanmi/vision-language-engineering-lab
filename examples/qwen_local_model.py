# Path: examples/qwen_local_model.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Offline Qwen3-VL inference from an explicitly downloaded local model."""

from vlm_engineering.qwen import QwenVLModel

model = QwenVLModel.from_local("models/Qwen3-VL-2B-Instruct")
print(model.generate("data/example.jpg", "Describe this image in one paragraph."))
