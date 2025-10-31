FROM ubuntu:22.04 AS cpp-build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    libopencv-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgrpc++-dev \
    protobuf-compiler-grpc \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz \
    && tar -xzf onnxruntime-linux-x64-1.18.0.tgz \
    && cp -r onnxruntime-linux-x64-1.18.0 /opt/onnxruntime \
    && rm -rf onnxruntime-linux-x64-1.18.0.tgz

WORKDIR /app
COPY cpp ./cpp

RUN cd cpp && mkdir build && cd build \
    && cmake .. -DCMAKE_PREFIX_PATH="/opt/onnxruntime/lib/cmake/onnxruntime" \
    && make -j$(nproc)

FROM ubuntu:22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core-dev \
    libopencv-imgcodecs-dev \
    libopencv-imgproc-dev \
    libprotobuf-dev \
    libgrpc++1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=cpp-build /opt/onnxruntime /opt/onnxruntime
COPY --from=cpp-build /app/cpp/build/visionguard_server /usr/local/bin/visionguard_server

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

EXPOSE 50051

ENTRYPOINT ["visionguard_server"]
CMD ["--model", "/app/models/best.onnx", "--address", "0.0.0.0:50051"]
