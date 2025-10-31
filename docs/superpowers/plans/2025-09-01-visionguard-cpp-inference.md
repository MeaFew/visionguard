# VisionGuard — C++ 推理服务实施计划

> **Goal：** 使用 C++17 + ONNX Runtime + gRPC 构建高性能推理服务，能够加载 YOLOv8 ONNX 模型并对输入图像执行目标检测。

**Architecture：** `cpp/src/inference_engine.cpp` 负责 ONNX Runtime 模型加载、预处理、推理、NMS 后处理；`cpp/src/grpc_server.cpp` 通过 gRPC 暴露 `Detect`/`Health`/`Benchmark` 接口；`cpp/proto/visionguard.proto` 定义服务协议。

**Tech Stack：** C++17、ONNX Runtime 1.18+、OpenCV 4.x、gRPC 1.65+、Protobuf、CMake 3.20+。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `cpp/CMakeLists.txt` | CMake 构建配置 |
| `cpp/proto/visionguard.proto` | gRPC 服务定义 |
| `cpp/include/inference_engine.h` | 推理引擎头文件 |
| `cpp/include/image_preprocessor.h` | 图像预处理头文件 |
| `cpp/include/grpc_server.h` | gRPC 服务头文件 |
| `cpp/src/inference_engine.cpp` | ONNX Runtime 推理实现 |
| `cpp/src/image_preprocessor.cpp` | OpenCV 预处理实现 |
| `cpp/src/grpc_server.cpp` | gRPC 服务实现 |
| `cpp/src/main.cpp` | 服务端入口 |
| `cpp/tests/test_inference.cpp` | C++ 推理单元测试骨架 |
| `docs/cpp_build.md` | C++ 编译与运行说明 |

---

## Task 1: 创建 C++ 目录结构与 Proto 定义

**Files:**
- Create: `cpp/CMakeLists.txt`
- Create: `cpp/proto/visionguard.proto`
- Create: `cpp/include/inference_engine.h`
- Create: `cpp/include/image_preprocessor.h`
- Create: `cpp/include/grpc_server.h`

**Commit:** `2025-09-01: add C++ inference service headers and gRPC proto` `--date="2025-09-01T10:00:00"`

---

## Task 2: 实现推理引擎与图像预处理

**Files:**
- Create: `cpp/src/inference_engine.cpp`
- Create: `cpp/src/image_preprocessor.cpp`

**Commit:** `2025-09-08: implement C++ ONNX Runtime inference engine and OpenCV preprocessor` `--date="2025-09-08T10:00:00"`

---

## Task 3: 实现 gRPC 服务与入口

**Files:**
- Create: `cpp/src/grpc_server.cpp`
- Create: `cpp/src/main.cpp`
- Create: `docs/cpp_build.md`

**Commit:** `2025-09-15: add gRPC server and C++ service entry point` `--date="2025-09-15T10:00:00"`

---

## Task 4: 添加 C++ 测试骨架与编译文档

**Files:**
- Create: `cpp/tests/test_inference.cpp`
- Update: `cpp/CMakeLists.txt` to add tests

**Commit:** `2025-09-18: add C++ inference tests and build documentation` `--date="2025-09-18T10:00:00"`

---

## 执行方式

Inline execution（当前 subagent 配额已满）。
