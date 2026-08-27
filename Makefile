# Path: Makefile
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

.PHONY: install install-all test lint typecheck security quality

install:
	python -m pip install -e ".[dev]"

install-all:
	python -m pip install -e ".[all,dev,notebooks]"

test:
	python -m pytest --cov=vlm_engineering --cov-report=term-missing

lint:
	python -m ruff check src tests examples

typecheck:
	python -m mypy src/vlm_engineering

security:
	python -m pip_audit .

quality: lint typecheck test
