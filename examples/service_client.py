# Path: examples/service_client.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Minimal client for the HTTP service API.

Start the service first:

    vlm-lab serve --port 8000

Then run:

    python examples/service_client.py
"""

import os

import httpx

base_url = os.environ.get("VLM_SERVICE_URL", "http://127.0.0.1:8000")
headers: dict[str, str] = {}
api_key = os.environ.get("VLM_SERVICE_API_KEY")
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

with httpx.Client(base_url=base_url, headers=headers, timeout=120.0) as client:
    print(client.get("/health").json())
    print(client.get("/v1/models").json())

    response = client.post("/v1/embed", json={"text": "architecture diagram"})
    response.raise_for_status()
    embedding = response.json()
    print(f"{len(embedding['vector'])}-d vector from {embedding['model']}")
