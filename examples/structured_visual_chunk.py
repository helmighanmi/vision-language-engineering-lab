# Path: examples/structured_visual_chunk.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Create a RAG-ready structured chunk from one rendered document page."""

import json

from vlm_engineering.documents import NativePageContent, VisualChunkBuilder, analyze_visual_document
from vlm_engineering.qwen import QwenVLModel

vlm = QwenVLModel.from_hub()
image = "data/page_001.png"
analysis = analyze_visual_document(vlm, image)
native = NativePageContent(
    document_id="architecture-v1",
    page=1,
    source_file="architecture.pdf",
    native_text="Exact text extracted by the native parser can be inserted here.",
    image_ref=image,
)
chunk = VisualChunkBuilder().build(native, analysis)
print(json.dumps(chunk.to_dict(), indent=2, ensure_ascii=False))
