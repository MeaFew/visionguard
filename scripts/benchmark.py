"""Benchmark Python/ONNX Runtime inference latency."""

from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort

from visionguard.core.preprocessor import letterbox_tensor

DEFAULT_MODEL = "runs/detect/train/weights/best.onnx"


def benchmark(model_path: str, iterations: int, size: int) -> dict[str, float]:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    dummy = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    input_tensor, _scale, _pad_left, _pad_top = letterbox_tensor(dummy, size)

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: input_tensor})

    start = time.perf_counter()
    for _ in range(iterations):
        session.run(None, {input_name: input_tensor})
    total = time.perf_counter() - start

    return {
        "iterations": iterations,
        "total_ms": total * 1000.0,
        "avg_ms": total * 1000.0 / iterations,
        "fps": iterations / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ONNX inference")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    results = benchmark(args.model, args.iterations, args.imgsz)
    print(results)


if __name__ == "__main__":
    main()
