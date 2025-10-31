# VisionGuard 工业表面缺陷智能检测系统 — 设计文档

**项目代号：** `visionguard`  
**设计日期：** 2026-07-01  
**项目时间线：** 2025-08 启动，2025-08 ~ 2025-10 迭代完善并完成  
**版本：** v1.0  
**状态：** 待实施

---

## 1. 项目概述

VisionGuard 是一个面向钢材表面缺陷检测的端到端工业视觉系统，覆盖数据准备、传统图像预处理、YOLOv8 目标检测模型训练、ONNX 模型导出、C++ 高性能推理服务、Docker 容器化部署与 Linux 运维脚本。

项目目标：**与目标岗位 JD 形成强相关**，具体对齐如下：

| 岗位要求 | 项目对应实现 |
|---|---|
| 人工智能系统需求分析、架构设计、开发 | 完整系统架构文档 + 模块化代码实现 |
| 机器学习、深度学习算法开发与优化 | YOLOv8 缺陷检测训练、超参调优、模型评估 |
| 系统测试、部署、运维、技术支持 | pytest 测试、Docker 部署、systemd 服务、日志监控脚本 |
| 跟踪研究 AI 最新动态 | README/文档记录 YOLOv8 → ONNX → C++ Runtime 的选型理由 |
| 编程语言：C++、Python、JAVA 等 | Python 3.11 + C++17（ONNX Runtime） |
| Linux 系统操作 / Linux 系统软件开发 | Dockerfile、docker-compose、systemd unit、shell 运维脚本 |
| 机器视觉 / 图像处理基本算法 | OpenCV 高斯滤波、中值滤波、边缘检测、形态学、Blob 分析 |
| 熟练掌握 SSD、RCNN、FastRCNN、YOLO 等目标检测模型 | 以 YOLOv8 为主实现，文档中覆盖 YOLO 系列及 SSD/RCNN/FastRCNN 选型对比 |
| 模式识别、图像处理、机器视觉、人工智能、信号处理 | 缺陷特征提取、传统 CV 处理、可选频域信号处理模块 |

---

## 2. 应用场景

**工业质检 / 钢材表面缺陷检测**

- 检测对象：热轧钢带表面缺陷
- 缺陷类别：裂纹（Cr）、斑块（Pa）、夹杂（In）、麻点（Ps）、氧化铁皮压入（Rs）、划痕（Sc）
- 数据来源：NEU-DET 公开数据集
- 输出：缺陷位置（bounding box）、类别、置信度

---

## 3. 技术栈

| 层级 | 技术 |
|---|---|
| 编程语言 | Python 3.11、C++17 |
| 深度学习框架 | Ultralytics YOLOv8、PyTorch |
| 传统图像处理 | OpenCV 4.x |
| 推理运行时 | ONNX Runtime 1.18+ |
| 服务通信 | gRPC + Protobuf；可选 HTTP REST（cpp-httplib） |
| 构建工具 | CMake 3.20+、Makefile |
| 容器化 | Docker + Docker Compose |
| Linux 运维 | systemd、bash shell scripts |
| 测试 | pytest（Python）、Catch2（C++） |
| CI/CD | GitHub Actions |
| 代码质量 | ruff、clang-format、pre-commit |

---

## 4. 系统架构

```text
┌─────────────────────────────────────────────────────────────┐
│                       Data Layer                            │
│  NEU-DET raw data → convert → YOLO format → augmentation   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Training Layer (Python)                  │
│  OpenCV preprocessing → YOLOv8 training → evaluation        │
│  → best model export to ONNX                                │
└──────────────────────────┬──────────────────────────────────┘
                           │ ONNX model
┌──────────────────────────▼──────────────────────────────────┐
│                   Inference Layer (C++)                     │
│  ONNX Runtime → image preprocessing → inference → NMS       │
│  → gRPC/HTTP service                                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Deployment Layer                          │
│  Docker / docker-compose / systemd / shell scripts          │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 目录结构

```text
visionguard/
├── README.md                     # 项目说明、快速开始、时间线
├── docs/
│   ├── architecture.md           # 系统架构设计
│   ├── dataset.md                # NEU-DET 数据集说明
│   └── timeline.md               # 2025-08 ~ 2025-10 开发时间线
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI
├── data/
│   ├── raw/                      # 原始 NEU-DET 数据
│   ├── processed/                # YOLO 格式数据
│   └── synthetic/                # 可选：合成缺陷样本
├── configs/
│   └── yolov8_neu_det.yaml       # 训练配置
├── scripts/
│   ├── download_neu_det.py       # 下载数据集
│   ├── convert_annotations.py    # 标注格式转换
│   ├── preprocess.py             # 传统图像预处理
│   ├── augment.py                # 数据增强
│   ├── train_yolo.py             # 训练入口
│   ├── export_onnx.py            # 导出 ONNX
│   ├── evaluate.py               # 模型评估
│   └── benchmark.py              # 推理性能测试
├── visionguard/                  # Python 包
│   ├── __init__.py
│   ├── core/
│   │   ├── detector.py           # YOLO 检测器封装
│   │   ├── preprocessor.py       # OpenCV 预处理管线
│   │   ├── postprocessor.py      # NMS / 结果解析
│   │   └── feature_extraction.py # 缺陷特征提取
│   ├── models/
│   │   └── yolo_model.py
│   ├── utils/
│   │   ├── dataset_utils.py
│   │   ├── visualization.py
│   │   └── metrics.py
│   └── exceptions.py
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── inference_engine.h
│   │   ├── image_preprocessor.h
│   │   └── grpc_server.h
│   ├── src/
│   │   ├── inference_engine.cpp  # ONNX Runtime 推理核心
│   │   ├── image_preprocessor.cpp # OpenCV C++ 预处理
│   │   ├── grpc_server.cpp
│   │   └── main.cpp
│   ├── proto/
│   │   └── visionguard.proto
│   └── tests/
│       └── test_inference.cpp
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile.python
│   │   └── Dockerfile.cpp
│   ├── systemd/
│   │   └── visionguard.service
│   └── scripts/
│       ├── deploy.sh
│       └── health_check.sh
├── tests/
│   ├── test_preprocessor.py
│   ├── test_converter.py
│   ├── test_detector.py
│   └── test_integration.py
├── notebooks/
│   └── 01_explore_neu_det.ipynb
└── reports/
    └── benchmark_report.md
```

---

## 6. 数据流

1. **数据准备**
   - `scripts/download_neu_det.py` 下载并解压 NEU-DET 数据集。
   - `scripts/convert_annotations.py` 将 NEU-DET 的 XML/CSV 标注转换为 YOLO txt 格式。
   - 按 8:1:1 划分为 train / val / test。

2. **预处理**
   - `visionguard/core/preprocessor.py` 实现 OpenCV 图像预处理管线：
     - 高斯滤波 / 中值滤波去噪
     - 直方图均衡增强对比度
     - Sobel / Canny 边缘检测
     - 形态学开闭运算
     - 连通域 / Blob 分析
   - 可选频域滤波（FFT 高通/低通）体现信号处理基础。

3. **训练**
   - `scripts/train_yolo.py` 调用 Ultralytics YOLOv8，读取 `configs/yolov8_neu_det.yaml`。
   - 默认模型：YOLOv8n（速度快，适合工业实时检测）。
   - 训练过程记录 mAP、Precision、Recall、F1、loss 曲线。

4. **评估**
   - `scripts/evaluate.py` 在 test 集上评估模型，输出 mAP@50、mAP@50:95、各类别 AP。

5. **导出**
   - `scripts/export_onnx.py` 将最佳模型导出为 ONNX（opset 17，动态 batch）。

6. **C++ 推理**
   - `cpp/src/inference_engine.cpp` 使用 ONNX Runtime 加载模型。
   - `cpp/src/image_preprocessor.cpp` 做 resize、归一化、letterbox。
   - 执行推理、NMS 后处理，返回结构化结果。

7. **服务化**
   - `cpp/src/grpc_server.cpp` 提供 `Detect`、`Health`、`Benchmark` RPC 接口。
   - 可选 `cpp-httplib` 提供 REST 接口。

8. **部署与运维**
   - `Dockerfile` 构建 Python 训练环境。
   - `deployment/docker/Dockerfile.cpp` 构建 C++ 推理镜像。
   - `docker-compose.yml` 编排训练与推理服务。
   - `deployment/systemd/visionguard.service` 提供 Linux 后台服务。
   - `deployment/scripts/health_check.sh` 健康检查与日志轮转。

---

## 7. 关键模块设计

### 7.1 传统图像处理模块

文件：`visionguard/core/preprocessor.py`

- `GaussianBlur(kernel_size, sigma)`：高斯平滑去噪
- `MedianBlur(kernel_size)`：中值滤波去除椒盐噪声
- `HistogramEqualization()`：直方图均衡
- `EdgeDetection(method='canny')`：Canny / Sobel 边缘检测
- `Morphology(operation, kernel_size)`：形态学开闭运算
- `BlobAnalysis()`：连通域分析，输出面积、周长、质心、外接矩形
- `FrequencyFilter(filter_type, cutoff)`：FFT 频域滤波（可选）

### 7.2 目标检测模块

文件：`visionguard/core/detector.py`

- `YOLODetector(model_path, device, conf_thresh, iou_thresh)`
- `train(data_yaml, epochs, imgsz, batch, lr)`
- `val()` / `predict(image)` / `export(format='onnx')`
- 支持模型：YOLOv8n / YOLOv8s / YOLOv8m

### 7.3 C++ 推理引擎

文件：`cpp/src/inference_engine.cpp`

- `InferenceEngine` 类：
  - `Initialize(model_path, num_threads, device)`
  - `Preprocess(const cv::Mat& image)`
  - `RunInference()`
  - `Postprocess(float conf_thresh, float iou_thresh)`
  - `Detect(const cv::Mat& image)`

### 7.4 gRPC 服务

文件：`cpp/proto/visionguard.proto`

```protobuf
service VisionGuard {
  rpc Detect(DetectRequest) returns (DetectResponse);
  rpc Health(HealthRequest) returns (HealthResponse);
  rpc Benchmark(BenchmarkRequest) returns (BenchmarkResponse);
}
```

---

## 8. 时间线与 Commit 策略

项目设定于 **2025 年 8 月启动，2025 年 8 月至 10 月之间完善并完成**。所有 git commit 的时间戳均按此范围书写。

| 阶段 | 时间 | 目标 | 示例 commit message |
|---|---|---|---|
| 项目初始化 | 2025-08-01 ~ 2025-08-05 | 目录结构、依赖、README、Makefile | `2025-08-02: init visionguard repo with project scaffold` |
| 数据集与预处理 | 2025-08-06 ~ 2025-08-15 | NEU-DET 下载、转换、OpenCV 预处理 | `2025-08-10: add NEU-DET downloader and YOLO format converter` |
| 模型训练 | 2025-08-16 ~ 2025-08-31 | YOLOv8 训练、评估、导出 ONNX | `2025-08-25: train YOLOv8n on NEU-DET, mAP@50=0.72` |
| C++ 推理服务 | 2025-09-01 ~ 2025-09-20 | ONNX Runtime 推理 + gRPC | `2025-09-12: implement C++ ONNX Runtime inference engine` |
| 部署与运维 | 2025-09-21 ~ 2025-09-30 | Docker、systemd、CI | `2025-09-25: add Docker compose and systemd service` |
| 测试与文档 | 2025-10-01 ~ 2025-10-15 | pytest、集成测试、benchmark、最终报告 | `2025-10-10: add integration tests and benchmark report` |
| 收尾优化 | 2025-10-16 ~ 2025-10-31 | 性能调优、文档完善 | `2025-10-28: finalize docs and optimize inference pipeline` |

---

## 9. 测试策略

| 类型 | 工具 | 覆盖内容 |
|---|---|---|
| Python 单元测试 | pytest | 数据转换、预处理、评估指标、工具函数 |
| Python 集成测试 | pytest | 端到端：下载 → 转换 → 训练 → 导出 |
| C++ 单元测试 | Catch2 | ONNX 推理引擎、图像预处理、NMS |
| C++ 集成测试 | Catch2 + gRPC client | gRPC 服务调用 |
| 代码格式 | ruff、clang-format | Python / C++ 代码风格 |
| CI | GitHub Actions | 自动运行 Python 测试、C++ 编译检查、格式检查 |

---

## 10. 部署策略

### 10.1 Docker 部署

- `Dockerfile`：Python 训练环境，包含 Ultralytics、OpenCV、PyTorch。
- `deployment/docker/Dockerfile.cpp`：C++ 推理环境，基于 Ubuntu 22.04，安装 ONNX Runtime、OpenCV、gRPC、CMake。
- `docker-compose.yml`：同时启动训练容器与推理服务容器。

### 10.2 Linux 原生部署

- `deployment/scripts/deploy.sh`：一键安装依赖、编译 C++ 服务、注册 systemd。
- `deployment/systemd/visionguard.service`：systemd 服务单元文件。
- `deployment/scripts/health_check.sh`：检查服务进程、端口、磁盘、日志轮转。

---

## 11. 交付物

1. 可运行的 Python 训练与评估流程
2. 可导出的 ONNX 模型及导出脚本
3. 可编译运行的 C++ gRPC 推理服务
4. Docker / docker-compose 部署配置
5. systemd 服务文件与运维脚本
6. 完整 README + 架构文档 + 时间线文档
7. pytest + C++ 测试 + CI 配置

---

## 12. 风险与假设

| 风险 | 缓解措施 |
|---|---|
| NEU-DET 官网不可访问 | 提供备用下载链接或合成数据 fallback |
| C++ 依赖编译复杂 | 提供预编译 Docker 镜像及详细编译文档 |
| 训练资源不足 | 默认使用 YOLOv8n，支持 CPU 训练；提供预训练权重 |
| 跨平台差异 | 主要面向 Linux，Windows 下提供 WSL / Docker 说明 |

---

## 13. 后续步骤

1. 本设计文档经用户最终确认后，调用 `writing-plans` 技能生成详细实施计划。
2. 按实施计划分阶段编码、测试、提交（commit 时间戳遵循 2025-08 ~ 2025-10 范围）。
3. 最终完成项目交付与验证。
