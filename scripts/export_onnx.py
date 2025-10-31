"""Export trained YOLOv8 model to ONNX format."""

from __future__ import annotations

import argparse
from pathlib import Path

from visionguard.core.detector import YOLODetector
from visionguard.exceptions import VisionGuardError

DEFAULT_MODEL = Path("runs/detect/models/real_train/weights/best.pt")
DEFAULT_IMGSZ = 640
DEFAULT_OPSET = 17


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to trained YOLOv8 weights",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Input image size for ONNX export",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Disable dynamic batch/axes",
    )
    args = parser.parse_args()

    dynamic = not args.no_dynamic

    detector = YOLODetector()
    try:
        detector.load(args.model)
        onnx_path = detector.export(
            format="onnx",
            imgsz=args.imgsz,
            dynamic=dynamic,
            opset=args.opset,
        )
        print(f"ONNX model exported to {onnx_path}")
    except VisionGuardError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
