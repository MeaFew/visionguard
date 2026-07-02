"""Benchmark Python/ONNX Runtime inference latency."""

from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort

from visionguard.core.preprocessor import letterbox_tensor

DEFAULT_MODEL = "runs/detect/train/weights/best.onnx"


def benchmark(
    model_path: str, iterations: int, size: int, providers: list[str] | None = None
) -> dict[str, float]:
    # providers=None lets ONNX Runtime auto-select (GPU when available). The old
    # hardcoded CPUExecutionProvider silently ignored a usable GPU, producing
    # misleading latency numbers for a "performance" benchmark.
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    active_providers = session.get_providers()

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
        "providers": ",".join(active_providers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ONNX inference")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--providers",
        type=str,
        default=None,
        help=(
            "Comma-separated ONNX Runtime providers (default: auto — GPU if "
            "available). e.g. 'CUDAExecutionProvider,CPUExecutionProvider'"
        ),
    )
    args = parser.parse_args()

    providers = args.providers.split(",") if args.providers else None
    results = benchmark(args.model, args.iterations, args.imgsz, providers=providers)
    print(results)


if __name__ == "__main__":
    main()
