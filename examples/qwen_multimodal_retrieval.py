# Path: examples/qwen_multimodal_retrieval.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Multimodal retrieval with Qwen3-VL-Embedding-2B through SentenceTransformers."""

from vlm_engineering.retrieval import QwenMultimodalEmbedder

embedder = QwenMultimodalEmbedder()
queries = ["Find the architecture diagram that shows a cache and a database."]
documents = [
    "A plain text architecture note.",
    "data/architecture_diagram.png",
    {"text": "Payment architecture", "image": "data/architecture_diagram.png"},
]
query_vectors = embedder.encode(queries, prompt="Retrieve relevant technical document evidence.")
doc_vectors = embedder.encode(documents)
print(query_vectors @ doc_vectors.T)
