# Path: examples/qwen_describe_image.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Qwen3-VL image description using the Hub/cache-backed loader."""

from vlm_engineering.qwen import QwenVLModel

model = QwenVLModel.from_hub("Qwen/Qwen3-VL-2B-Instruct")
print(model.generate("data/example.jpg", "Describe the image and list important visual details."))
