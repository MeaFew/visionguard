"""Run a quick end-to-end inference demo using the exported ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import onnxruntime as ort

from visionguard.core.postprocessor import decode_yolov8_output
from visionguard.core.preprocessor import letterbox_tensor
from visionguard.logging_setup import get_logger, setup_logging
from visionguard.utils.dataset_utils import NEU_DET_CLASSES
from visionguard.utils.visualization import save_detection_image

logger = get_logger(__name__)

DEFAULT_MODEL = "runs/detect/models/real_train/weights/best.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX inference demo")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/processed/test/images/inclusion_14.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/demo_detection.jpg",
        help="Path to save visualization",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default=None,
        help=(
            "Comma-separated ONNX Runtime providers, e.g. "
            "'CUDAExecutionProvider,CPUExecutionProvider'. Default: auto "
            "(use GPU if available, else CPU)."
        ),
    )
    args = parser.parse_args()

    # Resolve execution providers: honor --providers, otherwise let ONNX Runtime
    # pick (GPU when available). Hardcoding CPU previously wasted a usable GPU.
    providers = args.providers.split(",") if args.providers else None
    try:
        session = ort.InferenceSession(args.model, providers=providers)
    except Exception as exc:
        raise SystemExit(
            f"Failed to load ONNX model '{args.model}': {exc}. "
            "Export it first with `python scripts/export_onnx.py`."
        ) from exc
    input_name = session.get_inputs()[0].name
    active = session.get_providers()
    logger.info(f"ONNX Runtime providers in use: {active}")

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    input_tensor, scale, pad_left, pad_top = letterbox_tensor(image)
    outputs = session.run(None, {input_name: input_tensor})
    output = outputs[0]

    detections = decode_yolov8_output(
        output,
        class_names=NEU_DET_CLASSES,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        img_width=image.shape[1],
        img_height=image.shape[0],
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )

    logger.info(f"Found {len(detections)} detections:")
    for det in detections:
        logger.info(
            f"  {det.class_name}: {det.confidence:.2f} at ({det.x1:.1f}, {det.y1:.1f}) - ({det.x2:.1f}, {det.y2:.1f})"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_detection_image(args.image, detections, output_path)
    logger.info(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    setup_logging()
    main()
