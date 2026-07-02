# VisionGuard Benchmark Report

Benchmark results for YOLOv8n trained on NEU-DET and exported to ONNX.

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
| mAP@50 | 0.750 |
| mAP@50:95 | 0.422 |
| Precision | TBD |
| Recall | TBD |
| Inference latency (single image) | TBD ms |
| Throughput (batch=1) | TBD FPS |
| C++ gRPC Benchmark (100 iters) | TBD ms avg |

## Commands Used

```bash
# Python evaluation
python scripts/evaluate.py --model runs/detect/train/weights/best.pt

# Python/ONNX benchmark
python scripts/benchmark.py --model runs/detect/train/weights/best.onnx --iterations 100
```

## Observations

- Update this section with numbers measured on your target hardware.
