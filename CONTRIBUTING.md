<!--
Path: CONTRIBUTING.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Contributing

Support Python 3.11–3.13; Python 3.12 is the default Docker/development baseline.
Create a feature branch from current `main`, keep changes focused, and retain
source attribution. Reusable code belongs in `src/vlm_engineering`; notebooks
and scenarios are clients.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,config]"
python -m ruff check .
python -m mypy .
python -m pytest -m "not real_model" --cov=vlm_engineering --cov-report=term-missing
python -m pip check
python -m pip_audit .
```

Use `.[qwen-retrieval]` or `.[all]` only for optional runtime work. Do not download
models in normal CI. See [testing](docs/testing.md) for opt-in real-model commands.
Preserve the coverage threshold. Cover input validation, malformed backend outputs,
state replacement, source/image provenance and error propagation.

Before committing, review `git status`, `git diff --stat` and the patch. Keep
weights, models, caches, virtual environments, build products, tokens and secrets
out of version control. Open a PR explaining behavior, compatibility changes,
validation results and remaining hardware checks. Do not conflate deterministic
backend mocks with real inference validation.
