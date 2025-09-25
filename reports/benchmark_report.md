# VisionGuard Benchmark Report

> Template — fill in after running benchmarks on target hardware.

## Environment

| Item | Value |
|---|---|
| CPU | e.g. Intel Xeon E5-2680 v4 @ 2.4GHz |
| GPU | e.g. NVIDIA RTX 4090 / CPU only |
| RAM | e.g. 32 GB |
| OS | Ubuntu 22.04 LTS |
| ONNX Runtime | 1.18.0 |
| OpenCV | 4.9.0 |
| Model | YOLOv8n trained on NEU-DET |
| Input size | 640 x 640 |

## Metrics

| Metric | Value |
|---|---|
| mAP@50 | TBD |
| mAP@50:95 | TBD |
| Precision | TBD |
| Recall | TBD |
| Inference latency (single image) | TBD ms |
| Throughput (batch=1) | TBD FPS |
| C++ gRPC Benchmark (100 iters) | TBD ms avg |

## Commands Used

```bash
# Python evaluation
python scripts/evaluate.py --model models/train/weights/best.pt

# C++ benchmark via gRPC
python scripts/benchmark_client.py --iterations 100

# Python benchmark
python scripts/benchmark.py --model models/train/weights/best.onnx --iterations 100
```

## Observations

- TBD
