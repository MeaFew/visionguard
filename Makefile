.PHONY: help install install-dev lint format test data download clean

PYTHON := python
PIP := python -m pip

help:
	@echo "VisionGuard Makefile targets:"
	@echo "  install      - Install runtime dependencies"
	@echo "  install-dev  - Install dev dependencies"
	@echo "  lint         - Run ruff linter"
	@echo "  format       - Run ruff formatter"
	@echo "  test         - Run pytest"
	@echo "  data         - Download and convert NEU-DET dataset"
	@echo "  download     - Download NEU-DET only"
	@echo "  clean        - Remove caches and generated files"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	$(PYTHON) -m pre_commit install

lint:
	$(PYTHON) -m ruff check visionguard scripts tests

format:
	$(PYTHON) -m ruff format visionguard scripts tests

test:
	$(PYTHON) -m pytest tests -v

download:
	$(PYTHON) scripts/download_neu_det.py

data: download
	$(PYTHON) scripts/convert_annotations.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache *.egg-info build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
