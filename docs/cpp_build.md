# VisionGuard C++ 推理服务编译说明

## 依赖

- CMake >= 3.20
- C++17 编译器（GCC 11+ / Clang 14+ / MSVC 2022）
- OpenCV 4.x
- ONNX Runtime 1.18+
- gRPC 1.65+
- Protobuf

## Ubuntu 22.04 安装示例

```bash
# OpenCV
sudo apt-get install -y libopencv-dev

# ONNX Runtime (download from GitHub releases)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -xzf onnxruntime-linux-x64-1.18.0.tgz
sudo cp -r onnxruntime-linux-x64-1.18.0 /opt/onnxruntime

# gRPC + Protobuf (via vcpkg or apt)
sudo apt-get install -y protobuf-compiler libprotobuf-dev grpc-plugin
```

## 编译

```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/opt/onnxruntime/lib/cmake/onnxruntime" \
         -DONNXRuntime_DIR="/opt/onnxruntime/lib/cmake/onnxruntime"
make -j$(nproc)
```

## 运行

```bash
# 默认监听 0.0.0.0:50051
./visionguard_server --model ../../runs/detect/train/weights/best.onnx

# 指定地址
./visionguard_server --model ../../runs/detect/train/weights/best.onnx --address 0.0.0.0:50052
```

## 测试 gRPC

使用 Python 客户端示例（需 `grpcio` 和 `grpcio-tools`）：

```python
import grpc
import visionguard_pb2
import visionguard_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = visionguard_pb2_grpc.VisionGuardStub(channel)

with open("test.jpg", "rb") as f:
    response = stub.Detect(visionguard_pb2.DetectRequest(image_data=f.read()))
print(response.detections)
```

## Windows 说明

在 Windows 上建议使用 WSL2 / Docker，或手动安装 vcpkg 依赖后使用 MSVC 编译。
