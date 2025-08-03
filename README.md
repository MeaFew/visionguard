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

## 完整流程（开发完成后）

> 注：以下命令展示 VisionGuard 的完整使用流程。项目按 2025-08 ~ 2025-10 分阶段实现，部分脚本/服务在后续 Plan 中才会加入。

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
