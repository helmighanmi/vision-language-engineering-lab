<!--
Path: CONTRIBUTING.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Contributing

1. Create and activate a Python 3.12 virtual environment.
2. Install `python -m pip install -e ".[dev]"`.
3. Keep reusable logic under `src/vlm_engineering`; notebooks should import it.
4. Run `make quality` before opening a pull request.
5. Do not add model weights, caches, secrets or third-party training material to Git.
