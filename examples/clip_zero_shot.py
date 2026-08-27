# Path: examples/clip_zero_shot.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Zero-shot image classification with CLIP."""

from PIL import Image

from vlm_engineering.clip import CLIPEncoder

image = Image.open("data/example.jpg").convert("RGB")
model = CLIPEncoder()
for prediction in model.zero_shot_classify(image, ["cat", "dog", "car", "airplane"]):
    print(prediction)
