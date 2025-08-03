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

> 注：以下模块列表反映目标架构，各模块按项目计划分阶段增量实现。

- `visionguard/core/preprocessor.py`：传统图像处理管线。
- `visionguard/core/detector.py`：YOLOv8 检测器封装。
- `visionguard/core/postprocessor.py`：NMS 与结果解析。
- `visionguard/core/feature_extraction.py`：缺陷特征提取。
- `cpp/src/inference_engine.cpp`：ONNX Runtime C++ 推理引擎。
- `cpp/src/grpc_server.cpp`：gRPC 服务实现。
