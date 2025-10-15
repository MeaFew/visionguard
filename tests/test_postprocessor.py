"""Tests for YOLOv8 post-processing."""

import numpy as np
import pytest

from visionguard.core.postprocessor import (
    Detection,
    decode_yolov8_output,
    nms,
    xywh2xyxy,
)


def test_xywh2xyxy() -> None:
    boxes = np.array([[0.5, 0.5, 0.2, 0.4]])
    out = xywh2xyxy(boxes)
    expected = np.array([[0.4, 0.3, 0.6, 0.7]])
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_nms_empty() -> None:
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5) == []


def test_nms_simple() -> None:
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]])
    scores = np.array([0.9, 0.75, 0.8])
    keep = nms(boxes, scores, 0.5)
    # The first box should suppress the second because IoU is high.
    # The third box should remain.
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep


def test_decode_yolov8_output_realistic() -> None:
    output = np.full((1, 84, 8400), -100.0, dtype=np.float32)
    anchor = 1234
    class_id = 3
    class_names = ["background"] * 80
    class_names[class_id] = "person"

    # YOLOv8 ONNX export returns xywh in model-input pixel coordinates.
    output[0, :4, anchor] = np.array([320.0, 320.0, 128.0, 192.0], dtype=np.float32)
    output[0, 4 + class_id, anchor] = 0.9

    dets = decode_yolov8_output(output, class_names=class_names)
    assert len(dets) == 1
    det = dets[0]
    assert det.class_id == class_id
    assert det.class_name == "person"
    assert det.confidence == pytest.approx(0.9, abs=1e-5)
    np.testing.assert_allclose(
        [det.x1, det.y1, det.x2, det.y2],
        [256.0, 224.0, 384.0, 416.0],
        rtol=1e-5,
    )


def test_decode_yolov8_output_shape_error() -> None:
    with pytest.raises(ValueError):
        decode_yolov8_output(np.zeros((2, 84, 8400)), class_names=["a"])


def test_decode_yolov8_output_empty() -> None:
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    dets = decode_yolov8_output(output, class_names=["a"] * 80, conf_threshold=0.25)
    assert dets == []


def test_detection_dataclass() -> None:
    det = Detection(class_id=0, class_name="test", confidence=0.9, x1=0, y1=0, x2=10, y2=10)
    assert det.class_id == 0
