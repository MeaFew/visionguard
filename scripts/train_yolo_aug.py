"""Train YOLOv8n on real NEU-DET with augmented hyperparameters."""

from __future__ import annotations

import argparse
from pathlib import Path

from visionguard.core.detector import YOLODetector
from visionguard.exceptions import VisionGuardError

DEFAULT_CONFIG = Path("configs/yolov8_neu_det.yaml")
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 150
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_PROJECT = "models"
DEFAULT_NAME = "real_train_aug"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n on NEU-DET with augmentation")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--name", type=str, default=DEFAULT_NAME)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--cache", type=str, default="disk", choices=["none", "ram", "disk"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cache = None if args.cache == "none" else args.cache

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
            exist_ok=args.exist_ok,
            cache=cache,
            workers=args.workers,
            mixup=0.1,
            copy_paste=0.1,
            degrees=5.0,
            translate=0.1,
            scale=0.5,
        )
        print(f"Training complete. Best model: {best_path}")
    except VisionGuardError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
