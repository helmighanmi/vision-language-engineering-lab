# Path: Dockerfile
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim

LABEL org.opencontainers.image.title="Vision-Language Engineering Lab"
LABEL org.opencontainers.image.description="CLIP, Qwen3-VL, visual document understanding and multimodal RAG"
LABEL org.opencontainers.image.source="https://github.com/helmighanmi/vision-language-engineering-lab"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface

RUN useradd \
    --create-home \
    --uid 10001 \
    appuser

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install ".[qwen,retrieval]"

RUN mkdir -p \
    /app/data \
    /app/models \
    /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser \
    /app \
    /home/appuser

USER appuser

ENTRYPOINT ["vlm-lab"]
CMD ["--help"]