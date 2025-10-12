"""Post-processing for YOLOv8 detection outputs: NMS, confidence filtering, formatting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single detection result."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [x_center, y_center, w, h] to [x1, y1, x2, y2]."""
    out = np.copy(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Apply greedy Non-Maximum Suppression.

    Args:
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Array of shape (N,) with confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        List of indices to keep.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        order = order[np.where(iou < iou_threshold)[0] + 1]

    return keep


def decode_yolov8_output(
    output: np.ndarray,
    class_names: list[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_width: int = 640,
    img_height: int = 640,
) -> list[Detection]:
    """Decode YOLOv8 ONNX output (shape: 1 x 84 x 8400) into detection list.

    The output format from Ultralytics YOLOv8 ONNX export is:
        [batch, 4 + num_classes, num_anchors]
    where the first 4 channels are box coordinates (xywh, normalized) and
    the remaining channels are class logits.
    """
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"Unexpected output shape: {output.shape}")

    predictions = output[0]  # shape: (84, 8400)
    predictions = predictions.transpose(1, 0)  # shape: (8400, 84)

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    boxes_xyxy = xywh2xyxy(boxes)
    boxes_xyxy[:, [0, 2]] *= img_width
    boxes_xyxy[:, [1, 3]] *= img_height

    keep = nms(boxes_xyxy, scores, iou_threshold)

    detections: list[Detection] = []
    for idx in keep:
        class_id = int(class_ids[idx])
        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_names[class_id] if class_id < len(class_names) else "unknown",
                confidence=float(scores[idx]),
                x1=float(boxes_xyxy[idx, 0]),
                y1=float(boxes_xyxy[idx, 1]),
                x2=float(boxes_xyxy[idx, 2]),
                y2=float(boxes_xyxy[idx, 3]),
            )
        )

    return detections
