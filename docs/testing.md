<!--
Path: docs/testing.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Validation strategy

Normal tests are deterministic and use fake ML backends plus real small image
files. They validate package contracts, not learned model quality or GPU behavior.

```bash
python -m pip install -e ".[dev,config]"
python -m ruff check .
python -m mypy .
python -m pytest -m "not real_model" --cov=vlm_engineering --cov-report=term-missing
python -m pip check
python -m pip_audit .
```

The 80% branch-aware coverage gate is unchanged. Normal CI runs Python 3.11,
3.12 and 3.13 without ML weights or the heavy optional runtime. Manual
`runtime-compatibility.yml` installs CPU runtime extras and smoke-imports ML
libraries on that same matrix; it does not load models. It must be run before
claiming full optional-stack compatibility. Docker is not built by normal CI.

## Opt-in real model tests

Install the runtime, download complete snapshots, then use:

```bash
python -m pip install -e ".[qwen-retrieval,dev]"
vlm-lab download-model --embedding-size 2b --output models/Qwen3-VL-Embedding-2B
vlm-lab download-model --reranker-size 2b --output models/Qwen3-VL-Reranker-2B

HF_HUB_OFFLINE=1 \
VLM_RUN_REAL_MODEL_TESTS=1 \
VLM_QWEN_EMBEDDING_MODEL_PATH=models/Qwen3-VL-Embedding-2B \
VLM_QWEN_RERANKER_MODEL_PATH=models/Qwen3-VL-Reranker-2B \
pytest -m real_model tests/e2e -v
```

For Hub-backed downloads/inference instead:

```bash
VLM_RUN_REAL_MODEL_TESTS=1 pytest -m real_model tests/e2e -v
```

Unset local-path variables for the Hub command. Without the opt-in variable,
`pytest -m real_model tests/e2e -v` skips both tests before loading a backend.
Tests cover real text/image/mixed outputs, normalization and raw/sigmoid score
consistency. They do not certify ranking quality, multi-image generator inference,
latency or peak memory; those require representative GPU evaluation separately.

The project audit command `pip_audit .` audits resolved core project dependencies.
For optional ML runtime security, also run `python -m pip_audit` inside the
installed runtime environment. A passing core audit is not an all-extras audit.
