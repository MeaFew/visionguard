.PHONY: help install install-dev lint format test train evaluate data download clean

PYTHON := python
PIP := python -m pip

help:
	@echo "VisionGuard Makefile targets:"
	@echo "  install      - Install runtime dependencies"
	@echo "  install-dev  - Install dev dependencies"
	@echo "  lint         - Run ruff linter"
	@echo "  format       - Run ruff formatter"
	@echo "  test         - Run pytest"
	@echo "  train        - Train YOLOv8 on NEU-DET"
	@echo "  evaluate     - Evaluate trained YOLOv8 model"
	@echo "  data         - Download and convert NEU-DET dataset"
	@echo "  download     - Download NEU-DET only (Google Drive)"
	@echo "  download-kaggle - Download NEU-DET only (Kaggle)"
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

train:
	$(PYTHON) scripts/train_yolo.py

evaluate:
	$(PYTHON) scripts/evaluate.py

download:
	$(PYTHON) scripts/download_neu_det.py

download-kaggle:
	$(PYTHON) scripts/download_neu_det.py --source kaggle

data: download
	$(PYTHON) scripts/convert_annotations.py

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('__pycache__'), pathlib.Path('.pytest_cache'), pathlib.Path('.ruff_cache'), pathlib.Path('build'), pathlib.Path('dist')] + list(pathlib.Path('.').glob('*.egg-info'))]"
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
