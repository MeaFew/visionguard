"""Benchmark Python/ONNX Runtime inference latency."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(image: np.ndarray, size: int = 640) -> np.ndarray:
    """Resize with letterbox and normalize."""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def benchmark(model_path: str, iterations: int, size: int) -> dict[str, float]:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    dummy = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    input_tensor = preprocess(dummy, size)

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
    parser.add_argument("--model", type=str, default="models/train/weights/best.onnx")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    results = benchmark(args.model, args.iterations, args.imgsz)
    print(results)


if __name__ == "__main__":
    main()
