"""Train YOLOv8 on NEU-DET dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from visionguard.core.detector import YOLODetector
from visionguard.exceptions import VisionGuardError

DEFAULT_CONFIG = Path("configs/yolov8_neu_det.yaml")
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_PROJECT = "models"
DEFAULT_NAME = "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on NEU-DET")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to YOLO dataset config YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="YOLOv8 model name or path (e.g. yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Input image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help="Project directory for training outputs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_NAME,
        help="Experiment name",
    )
    args = parser.parse_args()

    detector = YOLODetector(device=args.device)
    try:
        detector.load(args.model)
        best_path = detector.train(
            data_yaml=args.config,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
        )
        print(f"Training complete. Best model: {best_path}")
    except VisionGuardError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
