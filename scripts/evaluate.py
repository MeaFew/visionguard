"""Evaluate a trained YOLOv8 model on NEU-DET test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visionguard.core.detector import YOLODetector
from visionguard.exceptions import VisionGuardError
from visionguard.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("configs/yolov8_neu_det.yaml")
DEFAULT_MODEL = Path("runs/detect/models/real_train/weights/best.pt")
DEFAULT_OUTPUT = Path("reports/evaluation.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 on NEU-DET")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to YOLO dataset config YAML",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to write evaluation metrics JSON",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    args = parser.parse_args()

    detector = YOLODetector(device=args.device)
    try:
        detector.load(args.model)
        metrics = detector.validate(data_yaml=args.config, split=args.split)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Evaluation metrics: {metrics}")
        logger.info(f"Metrics saved to {output_path}")
    except VisionGuardError as exc:
        logger.error(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    setup_logging()
    main()
