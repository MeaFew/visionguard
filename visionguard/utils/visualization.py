"""Visualization utilities for detection results."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from visionguard.core.postprocessor import Detection

COLOR_PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image."""
    vis_image = image.copy()
    for det in detections:
        color = COLOR_PALETTE[det.class_id % len(COLOR_PALETTE)]
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.class_name}: {det.confidence:.2f}"
        cv2.putText(
            vis_image,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )
    return vis_image


def save_detection_image(
    image_path: str | Path,
    detections: list[Detection],
    output_path: str | Path,
) -> None:
    """Load an image, draw detections, and save the result."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    vis_image = draw_detections(image, detections)
    cv2.imwrite(str(output_path), vis_image)
