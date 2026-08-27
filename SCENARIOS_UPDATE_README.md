<!--
Path: SCENARIOS_UPDATE_README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Scenario Cookbook Update

Copy this bundle over the repository root. It adds nine runnable VLM application scenarios, scenario contract tests, and README/CHANGELOG documentation.

After copying, run:

```bash
python -m ruff check .
python -m mypy .
python -m pytest --cov=vlm_engineering --cov-report=term-missing
python -m pip_audit .
```

The scenarios do not load or download model weights when invoked with `--help`, so the contract tests remain lightweight.
