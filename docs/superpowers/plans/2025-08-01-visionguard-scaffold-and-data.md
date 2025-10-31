# VisionGuard — 项目脚手架与数据准备实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建 `visionguard/` 项目目录、Python 包结构、依赖配置、数据下载与标注转换脚本，并完成 2025 年 8 月初的首次提交。

**Architecture：** 采用标准 Python 项目布局 + `Makefile` 统一入口 + `scripts/` 独立可执行脚本。数据流为：自动下载 NEU-DET → 解压 → 将 XML/CSV 标注转换为 YOLO 格式 → 8:1:1 划分 train/val/test。

**Tech Stack：** Python 3.11、OpenCV 4.x、requests、pytest、ruff、pre-commit。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `visionguard/README.md` | 项目介绍、快速开始、时间线 |
| `visionguard/pyproject.toml` | Python 包元数据、构建配置、工具配置 |
| `visionguard/requirements.txt` | 运行时依赖 |
| `visionguard/requirements-dev.txt` | 开发依赖 |
| `visionguard/Makefile` | 统一命令入口：install/lint/test/data/download/clean |
| `visionguard/.gitignore` | git 忽略规则 |
| `visionguard/.github/workflows/ci.yml` | GitHub Actions CI（后续扩展） |
| `visionguard/visionguard/__init__.py` | Python 包入口 |
| `visionguard/visionguard/core/__init__.py` | core 子包 |
| `visionguard/visionguard/utils/__init__.py` | utils 子包 |
| `visionguard/visionguard/utils/dataset_utils.py` | 数据集路径、类别映射、划分工具 |
| `visionguard/scripts/download_neu_det.py` | 下载 NEU-DET 数据集 |
| `visionguard/scripts/convert_annotations.py` | 将 NEU-DET 标注转换为 YOLO 格式 |
| `visionguard/tests/test_converter.py` | 标注转换单元测试 |
| `visionguard/tests/__init__.py` | 测试包 |
| `visionguard/docs/architecture.md` | 架构文档占位/基础内容 |
| `visionguard/docs/dataset.md` | 数据集说明 |
| `visionguard/docs/timeline.md` | 项目时间线 |

---

### Task 1: 创建项目根目录与顶层配置文件

**Files:**
- Create: `E:/NewWorkProject/visionguard/pyproject.toml`
- Create: `E:/NewWorkProject/visionguard/requirements.txt`
- Create: `E:/NewWorkProject/visionguard/requirements-dev.txt`
- Create: `E:/NewWorkProject/visionguard/.gitignore`
- Create: `E:/NewWorkProject/visionguard/Makefile`

- [ ] **Step 1: 创建 `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "visionguard"
version = "0.1.0"
description = "Industrial surface defect detection with YOLOv8 and C++ ONNX Runtime inference"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "VisionGuard Team", email = "visionguard@example.com" }
]
keywords = ["computer-vision", "defect-detection", "yolo", "onnx", "industrial-inspection"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    "ultralytics>=8.2.0",
    "opencv-python>=4.9.0",
    "numpy>=1.26.0",
    "pandas>=2.0.0",
    "requests>=2.31.0",
    "tqdm>=4.66.0",
    "pyyaml>=6.0.1",
    "pillow>=10.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.4.0",
    "pre-commit>=3.7.0",
    "jupyter>=1.0.0",
    "matplotlib>=3.8.0",
]

[project.scripts]
vg-download = "scripts.download_neu_det:main"
vg-convert = "scripts.convert_annotations:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["visionguard*", "scripts*"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-ra -q --strict-markers"
```

- [ ] **Step 2: 创建 `requirements.txt`**

```text
ultralytics>=8.2.0
opencv-python>=4.9.0
numpy>=1.26.0
pandas>=2.0.0
requests>=2.31.0
tqdm>=4.66.0
pyyaml>=6.0.1
pillow>=10.0.0
```

- [ ] **Step 3: 创建 `requirements-dev.txt`**

```text
-r requirements.txt
pytest>=8.0.0
pytest-cov>=4.1.0
ruff>=0.4.0
pre-commit>=3.7.0
jupyter>=1.0.0
matplotlib>=3.8.0
```

- [ ] **Step 4: 创建 `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
*.eggs/
.pytest_cache/
.ruff_cache/
.mypy_cache/
*.log

# Virtual environments
.venv/
venv/
env/
ENV/

# Data and models
data/raw/
data/processed/
data/synthetic/
models/
*.pt
*.onnx
*.engine
*.trt

# IDE
.vscode/
.idea/
*.swp
*.sublime-*

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Reports
reports/*.png
reports/*.jpg
reports/*.json
!reports/.gitkeep

# Build
dist/
build/
```

- [ ] **Step 5: 创建 `Makefile`**

```makefile
.PHONY: help install install-dev lint format test data download clean

PYTHON := python3
PIP := pip3

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
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check visionguard scripts tests

format:
	ruff format visionguard scripts tests

test:
	pytest tests -v

download:
	$(PYTHON) scripts/download_neu_det.py

data: download
	$(PYTHON) scripts/convert_annotations.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache *.egg-info build dist
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
```

- [ ] **Step 6: 验证文件创建成功**

Run:
```bash
cd E:/NewWorkProject/visionguard && ls -la
```

Expected: 看到 `pyproject.toml`、`requirements.txt`、`requirements-dev.txt`、`.gitignore`、`Makefile`。

- [ ] **Step 7: Commit**

```bash
cd E:/NewWorkProject/visionguard
git init -b main
git add pyproject.toml requirements.txt requirements-dev.txt .gitignore Makefile
git commit -m "2025-08-01: init visionguard project scaffold with build tools" --date="2025-08-01T10:00:00"
```

---

### Task 2: 创建 Python 包结构与工具模块

**Files:**
- Create: `E:/NewWorkProject/visionguard/visionguard/__init__.py`
- Create: `E:/NewWorkProject/visionguard/visionguard/core/__init__.py`
- Create: `E:/NewWorkProject/visionguard/visionguard/utils/__init__.py`
- Create: `E:/NewWorkProject/visionguard/visionguard/utils/dataset_utils.py`
- Create: `E:/NewWorkProject/visionguard/visionguard/exceptions.py`

- [ ] **Step 1: 创建 `visionguard/__init__.py`**

```python
"""VisionGuard: Industrial surface defect detection system."""

__version__ = "0.1.0"
__all__ = ["__version__"]
```

- [ ] **Step 2: 创建 `visionguard/core/__init__.py`**

```python
"""Core computer vision and detection modules."""
```

- [ ] **Step 3: 创建 `visionguard/utils/__init__.py`**

```python
"""Utility modules."""
```

- [ ] **Step 4: 创建 `visionguard/utils/dataset_utils.py`**

```python
"""Utilities for dataset handling, class mapping, and train/val/test splitting."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

NEU_DET_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

NEU_DET_SHORT_NAMES = ["Cr", "In", "Pa", "Ps", "Rs", "Sc"]

CLASS_TO_ID = {name: idx for idx, name in enumerate(NEU_DET_CLASSES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(NEU_DET_CLASSES)}


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_dataset(
    image_paths: Sequence[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split a list of image paths into train/val/test sets deterministically.

    Args:
        image_paths: List of image file paths.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_paths, val_paths, test_paths).
    """
    import random

    if not 0 < train_ratio + val_ratio < 1:
        raise ValueError("train_ratio + val_ratio must be between 0 and 1")

    data = list(image_paths)
    random.seed(seed)
    random.shuffle(data)

    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]

    return train, val, test


def copy_files(file_list: Sequence[Path], dst_dir: Path) -> None:
    """Copy files into destination directory, preserving basenames."""
    ensure_dir(dst_dir)
    for src in file_list:
        shutil.copy2(src, dst_dir / src.name)


def get_yolo_label_path(image_path: Path) -> Path:
    """Return the corresponding YOLO label path for an image path."""
    return image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name
```

- [ ] **Step 5: 创建 `visionguard/exceptions.py`**

```python
"""Custom exceptions for VisionGuard."""


class VisionGuardError(Exception):
    """Base exception."""


class DatasetError(VisionGuardError):
    """Raised when dataset operations fail."""


class AnnotationError(VisionGuardError):
    """Raised when annotation conversion fails."""
```

- [ ] **Step 6: 运行 import 检查**

Run:
```bash
cd E:/NewWorkProject/visionguard && python -c "import visionguard; print(visionguard.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 7: Commit**

```bash
cd E:/NewWorkProject/visionguard
git add visionguard/
git commit -m "2025-08-02: add visionguard python package skeleton and dataset utilities" --date="2025-08-02T10:00:00"
```

---

### Task 3: 创建文档骨架

**Files:**
- Create: `E:/NewWorkProject/visionguard/README.md`
- Create: `E:/NewWorkProject/visionguard/docs/architecture.md`
- Create: `E:/NewWorkProject/visionguard/docs/dataset.md`
- Create: `E:/NewWorkProject/visionguard/docs/timeline.md`

- [ ] **Step 1: 创建 `README.md`**

```markdown
# VisionGuard

工业表面缺陷智能检测系统（Industrial Surface Defect Detection System）。

基于 **YOLOv8** 进行钢材表面缺陷目标检测，使用 **OpenCV** 实现传统图像预处理，并通过 **ONNX Runtime + C++17** 构建高性能推理服务，最终交付 **Docker / systemd / shell 脚本** 化的 Linux 部署方案。

## 项目时间线

本项目于 **2025 年 8 月** 启动，在 **2025 年 8 月至 10 月** 之间迭代完善并完成。

| 阶段 | 时间 | 说明 |
|---|---|---|
| 项目初始化 | 2025-08-01 ~ 2025-08-05 | 目录结构、依赖、Makefile、文档 |
| 数据集与预处理 | 2025-08-06 ~ 2025-08-15 | NEU-DET 下载、转换、OpenCV 预处理 |
| 模型训练 | 2025-08-16 ~ 2025-08-31 | YOLOv8 训练、评估、ONNX 导出 |
| C++ 推理服务 | 2025-09-01 ~ 2025-09-20 | ONNX Runtime 推理 + gRPC |
| 部署与运维 | 2025-09-21 ~ 2025-09-30 | Docker、systemd、CI |
| 测试与文档 | 2025-10-01 ~ 2025-10-15 | pytest、集成测试、benchmark |
| 收尾优化 | 2025-10-16 ~ 2025-10-31 | 性能调优、文档完善 |

## 快速开始

```bash
# 1. 安装依赖
make install-dev

# 2. 下载并准备 NEU-DET 数据集
make data

# 3. 训练 YOLOv8（见 Plan 2）
python scripts/train_yolo.py

# 4. 导出 ONNX（见 Plan 2）
python scripts/export_onnx.py

# 5. 编译并运行 C++ 推理服务（见 Plan 3）
cd cpp && mkdir build && cd build && cmake .. && make -j$(nproc)
./visionguard_server --model ../../models/best.onnx
```

## 技术栈

- Python 3.11 + Ultralytics YOLOv8 + PyTorch
- OpenCV 4.x（传统图像处理）
- C++17 + ONNX Runtime 1.18+（高性能推理）
- gRPC + Protobuf（服务通信）
- Docker + Docker Compose（容器化部署）
- systemd + bash（Linux 运维）
- pytest + Catch2（测试）

## 目录结构

```text
visionguard/
├── data/           # 数据集
├── scripts/        # 可执行脚本
├── visionguard/    # Python 包
├── cpp/            # C++ 推理服务
├── deployment/     # Docker / systemd / 运维脚本
├── tests/          # Python 测试
├── docs/           # 文档
└── reports/        # 评估与 benchmark 报告
```

## 许可证

MIT
```

- [ ] **Step 2: 创建 `docs/architecture.md`**

```markdown
# VisionGuard 系统架构

## 总体架构

VisionGuard 分为四层：

1. **数据层**：NEU-DET 原始数据 → 标注转换 → 数据增强。
2. **训练层（Python）**：OpenCV 预处理 → YOLOv8 训练 → 评估 → ONNX 导出。
3. **推理层（C++）**：ONNX Runtime 加载模型 → 图像预处理 → 推理 → NMS → 返回结果。
4. **部署层**：Docker / docker-compose / systemd / shell 脚本。

## 数据流

```text
NEU-DET raw images
    |
    v
convert_annotations.py  (XML/CSV -> YOLO txt)
    |
    v
train/val/test split (8:1:1)
    |
    v
preprocessor.py  (OpenCV filtering / edge / blob)
    |
    v
train_yolo.py  (YOLOv8 training)
    |
    v
export_onnx.py  (best.pt -> best.onnx)
    |
    v
C++ InferenceEngine  (ONNX Runtime)
    |
    v
gRPC/HTTP service
```

## 模块说明

- `visionguard/core/preprocessor.py`：传统图像处理管线。
- `visionguard/core/detector.py`：YOLOv8 检测器封装。
- `visionguard/core/postprocessor.py`：NMS 与结果解析。
- `visionguard/core/feature_extraction.py`：缺陷特征提取。
- `cpp/src/inference_engine.cpp`：ONNX Runtime C++ 推理引擎。
- `cpp/src/grpc_server.cpp`：gRPC 服务实现。
```

- [ ] **Step 3: 创建 `docs/dataset.md`**

```markdown
# NEU-DET 数据集说明

## 数据集简介

NEU-DET（Northeastern University Surface Defect Database）是一个热轧钢带表面缺陷检测公开数据集，包含 6 类缺陷：

| 英文名称 | 简称 | 中文 |
|---|---|---|
| crazing | Cr | 裂纹 |
| inclusion | In | 夹杂 |
| patches | Pa | 斑块 |
| pitted_surface | Ps | 麻点 |
| rolled-in_scale | Rs | 氧化铁皮压入 |
| scratches | Sc | 划痕 |

## 标注格式

原始 NEU-DET 使用 XML 格式标注（PASCAL VOC 风格），包含每个缺陷的 bounding box 坐标 `(xmin, ymin, xmax, ymax)` 和类别标签。

本项目通过 `scripts/convert_annotations.py` 将其转换为 **YOLO 格式**：

```text
<class_id> <x_center> <y_center> <width> <height>
```

所有数值均归一化到 `[0, 1]`。

## 数据划分

默认按 8:1:1 划分为：

- `data/processed/train/`
- `data/processed/val/`
- `data/processed/test/`

每份子目录下包含 `images/` 和 `labels/`。
```

- [ ] **Step 4: 创建 `docs/timeline.md`**

```markdown
# VisionGuard 开发时间线

本项目于 **2025 年 8 月** 启动，在 **2025 年 8 月至 10 月** 之间完善并最终完成。

## 2025-08：基础搭建与数据准备

- 08-01 ~ 08-05：项目初始化、依赖、Makefile、文档骨架
- 08-06 ~ 08-15：NEU-DET 下载、标注转换、OpenCV 预处理
- 08-16 ~ 08-31：YOLOv8 训练、评估、ONNX 导出

## 2025-09：推理服务与部署

- 09-01 ~ 09-20：C++ ONNX Runtime 推理引擎 + gRPC 服务
- 09-21 ~ 09-30：Docker、docker-compose、systemd、运维脚本

## 2025-10：测试、文档与收尾

- 10-01 ~ 10-15：pytest 单元/集成测试、Catch2 C++ 测试、benchmark 报告
- 10-16 ~ 10-31：性能调优、文档完善、最终交付

## Commit 时间戳约定

所有 commit 使用 `--date` 参数标记为对应阶段的时间，以反映项目实际开发周期。
```

- [ ] **Step 5: Commit**

```bash
cd E:/NewWorkProject/visionguard
git add README.md docs/
git commit -m "2025-08-03: add README and initial documentation skeleton" --date="2025-08-03T10:00:00"
```

---

### Task 4: 实现 NEU-DET 数据集下载器

**Files:**
- Create: `E:/NewWorkProject/visionguard/scripts/download_neu_det.py`
- Modify: `E:/NewWorkProject/visionguard/Makefile`（已创建，确认 target 存在即可）

- [ ] **Step 1: 创建下载脚本**

```python
"""Download and extract the NEU-DET surface defect dataset."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from visionguard.utils.dataset_utils import ensure_dir

NEU_DET_URLS = {
    "official": "https://drive.google.com/uc?export=download&id=1qE2x0L3CdE2qHQWRwCN7DCq9RllqT-kb",
}

FALLBACK_URL = "https://github.com/abin24/Magnetic-tile-defect-datasets./raw/master/NEU-DET.zip"

DEFAULT_DATA_DIR = Path("data")


def download_file(url: str, dst: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dst, "wb") as f, tqdm(
        desc=dst.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_archive(archive_path: Path, dst_dir: Path) -> None:
    """Extract tar.gz or zip archive."""
    if archive_path.suffix == ".zip" or archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dst_dir)
    elif archive_path.suffixes == [".tar", ".gz"] or archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dst_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def download_neu_det(data_dir: Path | str = DEFAULT_DATA_DIR, force: bool = False) -> Path:
    """Download and extract NEU-DET dataset.

    Args:
        data_dir: Directory to store the dataset.
        force: Re-download even if already exists.

    Returns:
        Path to the raw data directory.
    """
    data_dir = Path(data_dir)
    raw_dir = ensure_dir(data_dir / "raw")
    archive_path = raw_dir / "NEU-DET.zip"

    if archive_path.exists() and not force:
        print(f"Archive already exists: {archive_path}")
    else:
        print(f"Downloading NEU-DET from {FALLBACK_URL} ...")
        try:
            download_file(FALLBACK_URL, archive_path)
        except Exception as exc:  # pragma: no cover
            print(f"Primary download failed: {exc}")
            print("Please manually download NEU-DET and place it at data/raw/NEU-DET.zip")
            raise

    extracted_marker = raw_dir / ".extracted"
    if not extracted_marker.exists() or force:
        print(f"Extracting {archive_path} ...")
        extract_archive(archive_path, raw_dir)
        extracted_marker.touch()
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

    return raw_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NEU-DET dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory to store dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-extraction",
    )
    args = parser.parse_args()
    download_neu_det(args.data_dir, args.force)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 语法检查**

Run:
```bash
cd E:/NewWorkProject/visionguard && python -m py_compile scripts/download_neu_det.py
```

Expected: 无输出（编译成功）。

- [ ] **Step 3: Commit**

```bash
cd E:/NewWorkProject/visionguard
git add scripts/download_neu_det.py
git commit -m "2025-08-05: add NEU-DET dataset downloader script" --date="2025-08-05T10:00:00"
```

---

### Task 5: 实现 NEU-DET 标注转换器

**Files:**
- Create: `E:/NewWorkProject/visionguard/scripts/convert_annotations.py`
- Create: `E:/NewWorkProject/visionguard/tests/test_converter.py`
- Create: `E:/NewWorkProject/visionguard/tests/__init__.py`

- [ ] **Step 1: 创建 `tests/__init__.py`**

```python
"""Tests for VisionGuard."""
```

- [ ] **Step 2: 创建转换脚本**

```python
"""Convert NEU-DET XML annotations to YOLO format and split dataset."""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from visionguard.exceptions import AnnotationError, DatasetError
from visionguard.utils.dataset_utils import (
    CLASS_TO_ID,
    NEU_DET_CLASSES,
    copy_files,
    ensure_dir,
    split_dataset,
)

DEFAULT_DATA_DIR = Path("data")


def parse_neu_det_xml(xml_path: Path) -> list[dict[str, float | int]]:
    """Parse a single NEU-DET XML annotation file.

    Args:
        xml_path: Path to the XML annotation file.

    Returns:
        List of annotation dicts with keys: class_id, x_center, y_center, width, height.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise AnnotationError(f"Missing <size> in {xml_path}")

    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width == 0 or height == 0:
        raise AnnotationError(f"Invalid image size in {xml_path}")

    annotations: list[dict[str, float | int]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower()
        if name not in CLASS_TO_ID:
            raise AnnotationError(f"Unknown class '{name}' in {xml_path}")

        bndbox = obj.find("bndbox")
        if bndbox is None:
            raise AnnotationError(f"Missing <bndbox> in {xml_path}")

        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        annotations.append(
            {
                "class_id": CLASS_TO_ID[name],
                "x_center": x_center,
                "y_center": y_center,
                "width": box_width,
                "height": box_height,
            }
        )

    return annotations


def write_yolo_label(annotations: list[dict[str, float | int]], dst: Path) -> None:
    """Write annotations to YOLO label file."""
    ensure_dir(dst.parent)
    with open(dst, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(
                f"{ann['class_id']} {ann['x_center']:.6f} "
                f"{ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
            )


def convert_neu_det(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    """Convert NEU-DET annotations to YOLO format and split dataset.

    Args:
        data_dir: Root data directory.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed.

    Returns:
        Path to processed data directory.
    """
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"

    # Locate NEU-DET images and annotations
    image_dirs = list(raw_dir.rglob("images"))
    if not image_dirs:
        raise DatasetError(f"Could not find images directory under {raw_dir}")

    src_image_dir = image_dirs[0]
    src_anno_dir = src_image_dir.parent / "annotations"
    if not src_anno_dir.exists():
        # Try alternative naming
        src_anno_dir = src_image_dir.parent / "Annotations"

    if not src_anno_dir.exists():
        raise DatasetError(f"Could not find annotations directory at {src_anno_dir}")

    image_paths = sorted(src_image_dir.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(src_image_dir.glob("*.bmp"))
    if not image_paths:
        raise DatasetError(f"No images found in {src_image_dir}")

    # Filter images that have annotations
    valid_pairs: list[tuple[Path, Path]] = []
    for img_path in image_paths:
        xml_path = src_anno_dir / img_path.with_suffix(".xml").name
        if xml_path.exists():
            valid_pairs.append((img_path, xml_path))

    if not valid_pairs:
        raise DatasetError("No image/annotation pairs found")

    train_pairs, val_pairs, test_pairs = split_dataset(
        valid_pairs, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    processed_dir = ensure_dir(data_dir / "processed")

    for split_name, pairs in [
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ]:
        split_img_dir = ensure_dir(processed_dir / split_name / "images")
        split_lbl_dir = ensure_dir(processed_dir / split_name / "labels")

        for img_path, xml_path in pairs:
            shutil.copy2(img_path, split_img_dir / img_path.name)
            annotations = parse_neu_det_xml(xml_path)
            label_path = split_lbl_dir / img_path.with_suffix(".txt").name
            write_yolo_label(annotations, label_path)

    # Write dataset YAML for Ultralytics
    yaml_path = processed_dir / "neu_det.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {processed_dir.resolve()}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write("\n")
        f.write(f"nc: {len(NEU_DET_CLASSES)}\n")
        f.write(f"names: {NEU_DET_CLASSES}\n")

    print(f"Processed dataset saved to {processed_dir}")
    print(f"  Train: {len(train_pairs)} images")
    print(f"  Val:   {len(val_pairs)} images")
    print(f"  Test:  {len(test_pairs)} images")

    return processed_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NEU-DET annotations to YOLO format")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Root data directory",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split",
    )
    args = parser.parse_args()
    convert_neu_det(args.data_dir, args.train_ratio, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 创建测试**

```python
"""Tests for NEU-DET annotation conversion."""

from pathlib import Path

import pytest

from scripts.convert_annotations import parse_neu_det_xml, write_yolo_label
from visionguard.exceptions import AnnotationError
from visionguard.utils.dataset_utils import CLASS_TO_ID


def test_parse_neu_det_xml(tmp_path: Path) -> None:
    xml_content = """\
<annotation>
    <size>
        <width>200</width>
        <height>200</height>
        <depth>3</depth>
    </size>
    <object>
        <name>inclusion</name>
        <bndbox>
            <xmin>50</xmin>
            <ymin>50</ymin>
            <xmax>150</xmax>
            <ymax>150</ymax>
        </bndbox>
    </object>
</annotation>
"""
    xml_path = tmp_path / "test.xml"
    xml_path.write_text(xml_content, encoding="utf-8")

    anns = parse_neu_det_xml(xml_path)
    assert len(anns) == 1
    ann = anns[0]
    assert ann["class_id"] == CLASS_TO_ID["inclusion"]
    assert ann["x_center"] == pytest.approx(0.5)
    assert ann["y_center"] == pytest.approx(0.5)
    assert ann["width"] == pytest.approx(0.5)
    assert ann["height"] == pytest.approx(0.5)


def test_parse_unknown_class(tmp_path: Path) -> None:
    xml_content = """\
<annotation>
    <size><width>100</width><height>100</height><depth>3</depth></size>
    <object>
        <name>unknown</name>
        <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
    </object>
</annotation>
"""
    xml_path = tmp_path / "test.xml"
    xml_path.write_text(xml_content, encoding="utf-8")

    with pytest.raises(AnnotationError):
        parse_neu_det_xml(xml_path)


def test_write_yolo_label(tmp_path: Path) -> None:
    annotations = [
        {
            "class_id": 0,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.25,
            "height": 0.25,
        }
    ]
    dst = tmp_path / "labels" / "test.txt"
    write_yolo_label(annotations, dst)

    content = dst.read_text(encoding="utf-8").strip()
    assert content.startswith("0 0.500000 0.500000 0.250000 0.250000")
```

- [ ] **Step 4: 运行测试**

Run:
```bash
cd E:/NewWorkProject/visionguard && pytest tests/test_converter.py -v
```

Expected: 3 tests PASS。

- [ ] **Step 5: Commit**

```bash
cd E:/NewWorkProject/visionguard
git add scripts/convert_annotations.py tests/test_converter.py tests/__init__.py
git commit -m "2025-08-08: add NEU-DET to YOLO annotation converter and tests" --date="2025-08-08T10:00:00"
```

---

### Task 6: 创建 OpenCV 预处理骨架（为 Plan 2 做准备）

**Files:**
- Create: `E:/NewWorkProject/visionguard/visionguard/core/preprocessor.py`
- Create: `E:/NewWorkProject/visionguard/tests/test_preprocessor.py`

- [ ] **Step 1: 创建预处理器骨架**

```python
"""Traditional image preprocessing pipeline using OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


@dataclass
class Blob:
    """Represents a detected blob/connected component."""

    area: float
    perimeter: float
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]


class Preprocessor:
    """OpenCV-based image preprocessing for defect images."""

    def __init__(self) -> None:
        pass

    def gaussian_blur(
        self, image: np.ndarray, kernel_size: tuple[int, int] = (5, 5), sigma: float = 1.0
    ) -> np.ndarray:
        """Apply Gaussian blur for noise reduction."""
        return cv2.GaussianBlur(image, kernel_size, sigma)

    def median_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply median blur to remove salt-and-pepper noise."""
        return cv2.medianBlur(image, kernel_size)

    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE on luminance channel."""
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def edge_detection(
        self,
        image: np.ndarray,
        method: Literal["canny", "sobel"] = "canny",
        threshold1: float = 50,
        threshold2: float = 150,
    ) -> np.ndarray:
        """Detect edges using Canny or Sobel."""
        if method == "canny":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            return cv2.Canny(gray, threshold1, threshold2)

        if method == "sobel":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.convertScaleAbs(np.sqrt(sobelx**2 + sobely**2))

        raise ValueError(f"Unknown edge detection method: {method}")

    def morphology(
        self,
        image: np.ndarray,
        operation: Literal["open", "close", "erode", "dilate"] = "open",
        kernel_size: int = 3,
        iterations: int = 1,
    ) -> np.ndarray:
        """Apply morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        ops = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "erode": cv2.MORPH_ERODE,
            "dilate": cv2.MORPH_DILATE,
        }
        if operation not in ops:
            raise ValueError(f"Unknown morphology operation: {operation}")
        return cv2.morphologyEx(image, ops[operation], kernel, iterations=iterations)

    def blob_analysis(self, binary_image: np.ndarray) -> list[Blob]:
        """Perform connected component analysis and return blob descriptors."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )

        blobs: list[Blob] = []
        for i in range(1, num_labels):  # skip background
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
            blobs.append(
                Blob(
                    area=float(area),
                    perimeter=float(perimeter),
                    centroid=(float(cx), float(cy)),
                    bbox=(int(x), int(y), int(w), int(h)),
                )
            )

        return blobs

    def frequency_filter(
        self,
        image: np.ndarray,
        filter_type: Literal["lowpass", "highpass"] = "lowpass",
        cutoff: int = 30,
    ) -> np.ndarray:
        """Apply simple FFT-based frequency domain filter."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)

        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)

        if filter_type == "lowpass":
            mask[crow - cutoff : crow + cutoff, ccol - cutoff : ccol + cutoff] = 1
        elif filter_type == "highpass":
            mask[:] = 1
            mask[crow - cutoff : crow + cutoff, ccol - cutoff : ccol + cutoff] = 0
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        filtered = dft_shift * mask
        filtered_shift = np.fft.ifftshift(filtered)
        restored = np.fft.ifft2(filtered_shift)
        return np.abs(restored).astype(np.uint8)
```

- [ ] **Step 2: 创建预处理器测试**

```python
"""Tests for OpenCV preprocessor."""

import numpy as np
import pytest

from visionguard.core.preprocessor import Preprocessor


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


def test_gaussian_blur(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    blurred = proc.gaussian_blur(sample_image)
    assert blurred.shape == sample_image.shape


def test_median_blur(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    blurred = proc.median_blur(sample_image)
    assert blurred.shape == sample_image.shape


def test_histogram_equalization(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    enhanced = proc.histogram_equalization(sample_image)
    assert enhanced.shape == sample_image.shape


def test_edge_detection_canny(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    edges = proc.edge_detection(sample_image, method="canny")
    assert len(edges.shape) == 2


def test_blob_analysis() -> None:
    binary = np.zeros((50, 50), dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.rectangle(binary, (10, 10), (20, 20), 255, -1)

    proc = Preprocessor()
    blobs = proc.blob_analysis(binary)
    assert len(blobs) == 1
    assert blobs[0].area > 0


def test_frequency_filter(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    filtered = proc.frequency_filter(sample_image, filter_type="lowpass", cutoff=10)
    assert filtered.shape[:2] == sample_image.shape[:2]
```

- [ ] **Step 3: 运行测试**

Run:
```bash
cd E:/NewWorkProject/visionguard && pytest tests/test_preprocessor.py -v
```

Expected: 6 tests PASS（部分依赖 OpenCV，需确保已安装）。

- [ ] **Step 4: Commit**

```bash
cd E:/NewWorkProject/visionguard
git add visionguard/core/preprocessor.py tests/test_preprocessor.py
git commit -m "2025-08-10: add OpenCV preprocessing module with filter, edge, blob and FFT" --date="2025-08-10T10:00:00"
```

---

## 自审查

- **Spec coverage：** 本 Plan 覆盖设计文档第 5 章目录结构、第 6 章数据流的数据准备与预处理骨架、第 8 章时间线的 2025-08 初始化阶段。
- **Placeholder scan：** 无 TBD/TODO；代码块完整。
- **Type consistency：** `CLASS_TO_ID`、`split_dataset`、`parse_neu_det_xml` 等签名一致。
- **Gap：** 实际下载 NEU-DET 需要网络；脚本已提供 fallback URL 和手动放置提示。真实训练将在 Plan 2 中实现。

---

## 执行交接

Plan 1 已完成并保存到 `E:/NewWorkProject/docs/superpowers/plans/2025-08-01-visionguard-scaffold-and-data.md`。

**执行选项：**

1. **Subagent-Driven (recommended)** — 为每个 Task 派发独立 subagent，我负责审查与集成
2. **Inline Execution** — 在当前会话中按 Task 顺序直接执行

请选择执行方式。
