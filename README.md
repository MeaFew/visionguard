**English** | [中文](README.zh-CN.md)

<div align="center">

# VisionGuard

**Industrial surface-defect detection from training to a C++ inference service**

*YOLOv8 · ONNX Runtime · C++17 gRPC · NEU-DET*

<img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-Ultralytics-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch / Ultralytics">
<img src="https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white" alt="ONNX">
<img src="https://img.shields.io/badge/Ruff-checked-261230?logo=ruff&logoColor=white" alt="Ruff">
<img src="https://img.shields.io/badge/Tests-pytest-0a9ed4?logo=pytest&logoColor=white" alt="Tests">
<a href="https://github.com/MeaFew/visionguard/actions"><img src="https://github.com/MeaFew/visionguard/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">

</div>

---

## Headline

> **YOLOv8n reaches test mAP50 = 0.750 and mAP50–95 = 0.422 on the real NEU-DET dataset.** The delivery path continues beyond Python: trained weights are exported to ONNX and served by a C++17 gRPC application with Docker and systemd deployment.

<p align="center">
  <img src="assets/deployment_summary.svg" width="900" alt="VisionGuard per-class AP50 and deployment path">
</p>

### Overall metrics

| Metric | Value |
|---|---|
| val mAP50 | 0.780 |
| **test mAP50** | **0.750** |
| **test mAP50-95** | **0.422** |

### Per-class test AP50

| Class | AP50 |
|-------|-----:|
| patches | **0.944** |
| scratches | **0.914** |
| inclusion | 0.865 |
| pitted_surface | 0.769 |
| rolled-in_scale | 0.604 |
| crazing | 0.406 |

> Protocol: 1,800 NEU-DET images; train/validation/test = 1,440/180/180; 100 epochs. Full metrics: [`assets/real_evaluation_test.json`](assets/real_evaluation_test.json).

<p align="center">
  <img src="assets/real_demo_detection.jpg" alt="ONNX inference on real NEU-DET test image inclusion_14.jpg">
</p>

<p align="center">
  <img src="runs/detect/models/real_train/confusion_matrix_normalized.png" width="720" alt="Normalized confusion matrix on real NEU-DET data">
</p>

## Architecture

```mermaid
flowchart LR
    A[NEU-DET XML] --> B[YOLO labels]
    B --> C[YOLOv8n training]
    C --> D[best.pt]
    D --> E[ONNX export]
    E --> F[C++17 + ONNX Runtime]
    F --> G[gRPC service]
    G --> H[Docker / systemd]
```

<details>
<summary><b>Project timeline</b></summary>

This project started in **August 2025** and was iteratively built and completed between **August and October 2025**.

| Phase | Period | Scope |
|---|---|---|
| Project scaffold | 2025-08-01 ~ 2025-08-05 | Directory layout, dependencies, Makefile, documentation |
| Dataset & preprocessing | 2025-08-06 ~ 2025-08-15 | NEU-DET download, conversion, OpenCV preprocessing |
| Model training | 2025-08-16 ~ 2025-08-31 | YOLOv8 training, evaluation, ONNX export |
| C++ inference service | 2025-09-01 ~ 2025-09-20 | ONNX Runtime inference + gRPC |
| Deployment & operations | 2025-09-21 ~ 2025-09-30 | Docker, systemd, CI |
| Testing & documentation | 2025-10-01 ~ 2025-10-15 | pytest, integration tests, benchmark |
| Final polish | 2025-10-16 ~ 2025-10-31 | Performance tuning, documentation |

</details>

## Quick start

```bash
git clone https://github.com/MeaFew/visionguard.git
cd visionguard

python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

make install-dev
make data

# Produces the default model path used by evaluation/export/deployment
python scripts/train_yolo.py --epochs 100 --batch 8 --device 0 \
  --project runs/detect/models --name real_train

python scripts/evaluate.py --split test
python scripts/export_onnx.py
python scripts/demo_inference.py --output reports/demo_detection.jpg
```

The repository commits evaluation JSON and showcase images, but not `.pt` or `.onnx` weights. Evaluation, export, and deployment therefore require the training step above to create `runs/detect/models/real_train/weights/best.pt`; a clean clone cannot silently reuse a local model.

## C++ inference service

```bash
cd cpp && mkdir build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH="/opt/onnxruntime/lib/cmake/onnxruntime" \
  -DONNXRuntime_DIR="/opt/onnxruntime/lib/cmake/onnxruntime"
make -j$(nproc)
./visionguard_server \
  --model ../../runs/detect/models/real_train/weights/best.onnx
```

## Deployment

### Docker Compose (serving only)

Export the ONNX model first, then start the gRPC service:

```bash
python scripts/export_onnx.py --model runs/detect/models/real_train/weights/best.pt
docker compose up --build
```

Compose mounts the weights **directory** read-only into the service container; with a single-file mount Docker would silently create an empty host directory when `best.onnx` is missing.

### Linux native (systemd)

`deploy.sh` requires the exported ONNX model (`python scripts/export_onnx.py`): the script explicitly installs `runs/detect/models/real_train/weights/best.onnx` into `/opt/visionguard` (matching the `ExecStart` path of the systemd unit) and fails fast with an error if it is missing.

```bash
sudo bash deployment/scripts/deploy.sh
sudo bash deployment/scripts/health_check.sh
```

## Quality gates

```bash
pytest tests/ -v
ruff check .
ruff format --check .
```

## Tech stack

- Python 3.11 + Ultralytics YOLOv8 + PyTorch
- OpenCV 4.x (classical image processing)
- C++17 + ONNX Runtime 1.18+ (high-performance inference)
- gRPC + Protobuf (service communication)
- Docker + Docker Compose (containerized deployment)
- systemd + bash (Linux operations)
- pytest + Catch2 (testing)
- GitHub Actions (CI/CD)

## Project structure

```text
visionguard/
├── assets/           # committed evidence and README visuals
├── configs/          # YOLOv8 dataset/training configuration
├── cpp/              # C++ ONNX Runtime + gRPC service
├── data/             # NEU-DET dataset (git-ignored)
├── deployment/       # Docker, systemd, and operations scripts
├── docs/             # documentation
├── notebooks/        # Jupyter data exploration
├── reports/          # evaluation and benchmark reports
├── scripts/          # data, training, evaluation, export, demo
├── tests/            # Python tests
└── visionguard/      # Python package
```

<details>
<summary><b>Synthetic-data smoke test</b></summary>

When NEU-DET cannot be downloaded, `scripts/generate_synthetic_data.py` can validate the pipeline. Synthetic data is for execution checks only and does **not** support real-world performance claims.

Training YOLOv8n for 50 epochs (CPU) on 100 synthetic samples yields the following test metrics:

```json
{
  "map50": 0.679,
  "map75": 0.542,
  "map": 0.487
}
```

![Synthetic demo detection](assets/demo_detection.jpg)

</details>

## License

MIT
