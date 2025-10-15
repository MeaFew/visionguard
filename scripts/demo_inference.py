"""Run a quick end-to-end inference demo using the exported ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from visionguard.core.postprocessor import decode_yolov8_output
from visionguard.utils.dataset_utils import NEU_DET_CLASSES
from visionguard.utils.visualization import save_detection_image


def preprocess(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, int, int]:
    """Resize with letterbox and normalize; return CHW tensor plus padding info."""
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
    tensor = np.expand_dims(chw, axis=0)
    return tensor, scale, pad_left, pad_top


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX inference demo")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/models/demo_train/weights/best.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/processed/test/images/synthetic_0080.jpg",
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
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    input_tensor, scale, pad_left, pad_top = preprocess(image)
    outputs = session.run(None, {input_name: input_tensor})
    output = outputs[0]

    detections = decode_yolov8_output(
        output,
        class_names=NEU_DET_CLASSES,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=640,
        img_width=image.shape[1],
        img_height=image.shape[0],
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )

    print(f"Found {len(detections)} detections:")
    for det in detections:
        print(f"  {det.class_name}: {det.confidence:.2f} at ({det.x1:.1f}, {det.y1:.1f}) - ({det.x2:.1f}, {det.y2:.1f})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_detection_image(args.image, detections, output_path)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
