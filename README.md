<div align="center">

# VisionGuard

**工业表面缺陷检测**

*YOLOv8 · ONNX · C++ gRPC · NEU-DET*

<img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-Ultralytics-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch / Ultralytics">
<img src="https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white" alt="ONNX">
<img src="https://img.shields.io/badge/Ruff-checked-261230?logo=ruff&logoColor=white" alt="Ruff">
<img src="https://img.shields.io/badge/Tests-pytest-0a9ed4?logo=pytest&logoColor=white" alt="Tests">
<a href="https://github.com/MeaFew/visionguard/actions"><img src="https://img.shields.io/badge/CI-passing-brightgreen?logo=githubactions&logoColor=white" alt="CI"></a>
<img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">

</div>

---

## 核心结论

> **YOLOv8n 在真实 NEU-DET 测试集上：mAP50 = 0.750、mAP50-95 = 0.422**（1800 张，train/val/test = 1440/180/180，100 epoch）。
> 端到端全栈链路——**Python 训练 + C++ gRPC 部署**，从模型到生产推理服务一条龙交付。

本项目以 YOLOv8 完成钢材表面 6 类缺陷目标检测，训练结果经 ONNX 导出后由 **C++17 + ONNX Runtime** 构建高性能推理服务，最终交付 Docker / systemd / shell 脚本化的 Linux 部署方案。

### 总体指标（test 集）

| 指标 | 数值 |
|---|---|
| val mAP50 | 0.783 |
| **test mAP50** | **0.750** |
| **test mAP50-95** | **0.422** |

### 各类别 test AP50

| 类别 | 中文 | AP50 |
|---|---|---|
| patches | 斑块 | **0.944** |
| scratches | 划痕 | **0.914** |
| inclusion | 夹杂 | 0.865 |
| pitted_surface | 麻点 | 0.769 |
| rolled-in_scale | 氧化铁皮压入 | 0.604 |
| crazing | 裂纹 | 0.406 |

> 完整指标见 [`reports/real_evaluation_test.json`](reports/real_evaluation_test.json)。

<p align="center">
  <img src="assets/real_demo_detection.jpg" alt="ONNX 推理可视化（真实测试图 inclusion_14.jpg）">
</p>

---

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

# 3. 训练 YOLOv8
python scripts/train_yolo.py --epochs 100 --batch 8 --device 0

# 4. 评估模型（默认使用真实 NEU-DET 训练结果）
python scripts/evaluate.py --split test

# 5. 导出 ONNX
python scripts/export_onnx.py

# 6. Python 端到端推理 demo
python scripts/demo_inference.py --output reports/demo_detection.jpg

# 7. 编译并运行 C++ 推理服务
cd cpp && mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/opt/onnxruntime/lib/cmake/onnxruntime" -DONNXRuntime_DIR="/opt/onnxruntime/lib/cmake/onnxruntime"
make -j$(nproc)
./visionguard_server --model ../../runs/detect/models/real_train/weights/best.onnx
```

## 测试

```bash
# Python 测试
pytest tests/ -v

# 代码格式检查
ruff check .
ruff format --check .
```

## 部署

### Docker Compose（仅推理服务）

先导出 ONNX 模型，再启动 gRPC 服务：

```bash
python scripts/export_onnx.py --model runs/detect/models/real_train/weights/best.pt
docker compose up --build
```

### Linux 原生部署

```bash
sudo bash deployment/scripts/deploy.sh
sudo bash deployment/scripts/health_check.sh
```

## 技术栈

- Python 3.11 + Ultralytics YOLOv8 + PyTorch
- OpenCV 4.x（传统图像处理）
- C++17 + ONNX Runtime 1.18+（高性能推理）
- gRPC + Protobuf（服务通信）
- Docker + Docker Compose（容器化部署）
- systemd + bash（Linux 运维）
- pytest + Catch2（测试）
- GitHub Actions（CI/CD）

## 目录结构

```text
visionguard/
├── configs/        # YOLOv8 训练配置
├── cpp/            # C++ 推理服务
├── data/           # 数据集
├── deployment/     # Docker / systemd / 运维脚本
├── docs/           # 文档
├── notebooks/      # Jupyter 数据探索
├── reports/        # 评估与 benchmark 报告
├── scripts/        # 可执行脚本
├── tests/          # Python 测试
└── visionguard/    # Python 包
```

<details>
<summary><b>🧪 合成数据 Demo（无 NEU-DET 时验证 pipeline）</b></summary>

如果无法下载真实 NEU-DET，可使用 `scripts/generate_synthetic_data.py` 生成合成数据验证 pipeline。合成数据仅用于验证 pipeline 正确性，**不代表真实场景性能**。

使用 YOLOv8n 在 100 张合成样本上训练 50 epoch（CPU），测试集指标：

```json
{
  "map50": 0.679,
  "map75": 0.542,
  "map": 0.487
}
```

![Synthetic demo detection](assets/demo_detection.jpg)

</details>

## 许可证

MIT
