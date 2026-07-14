"""Train YOLOv8 on NEU-DET dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from visionguard.core.detector import YOLODetector
from visionguard.exceptions import VisionGuardError
from visionguard.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("configs/yolov8_neu_det.yaml")
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_PROJECT = "runs/detect/models"
DEFAULT_NAME = "real_train"


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
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Overwrite existing experiment directory",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use augmented hyperparameters for real NEU-DET training",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="disk",
        choices=["none", "ram", "disk"],
        help="Dataset cache strategy",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers",
    )
    args = parser.parse_args()

    cache = None if args.cache == "none" else args.cache

    train_kwargs = {
        "exist_ok": args.exist_ok,
        "cache": cache,
        "workers": args.workers,
    }
    if args.augment:
        train_kwargs.update(
            {
                "mixup": 0.1,
                "copy_paste": 0.1,
                "degrees": 5.0,
                "translate": 0.1,
                "scale": 0.5,
            }
        )

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
            **train_kwargs,
        )
        logger.info(f"Training complete. Best model: {best_path}")
    except VisionGuardError as exc:
        logger.error(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    setup_logging()
    main()
