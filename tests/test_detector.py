"""Tests for YOLOv8 detector wrapper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from visionguard.core.detector import DetectorError, YOLODetector


def test_detector_load_invalid() -> None:
    detector = YOLODetector()
    with pytest.raises(DetectorError, match="Failed to load YOLO model from nonexistent_model.pt"):
        detector.load("nonexistent_model.pt")


def test_detector_predict_without_load() -> None:
    detector = YOLODetector()
    with pytest.raises(DetectorError, match="Model not loaded"):
        detector.predict("nonexistent.jpg")


def test_detector_validate_without_load() -> None:
    detector = YOLODetector()
    with pytest.raises(DetectorError, match="Model not loaded"):
        detector.validate("data.yaml")


def test_detector_export_without_load() -> None:
    detector = YOLODetector()
    with pytest.raises(DetectorError, match="Model not loaded"):
        detector.export()


def test_detector_train_without_load() -> None:
    detector = YOLODetector()
    with pytest.raises(DetectorError, match="Model not loaded"):
        detector.train("data.yaml")


def test_detector_init_invalid_conf_thresh() -> None:
    with pytest.raises(DetectorError, match="conf_thresh must be between 0 and 1"):
        YOLODetector(conf_thresh=0)
    with pytest.raises(DetectorError, match="conf_thresh must be between 0 and 1"):
        YOLODetector(conf_thresh=1)


def test_detector_init_invalid_iou_thresh() -> None:
    with pytest.raises(DetectorError, match="iou_thresh must be between 0 and 1"):
        YOLODetector(iou_thresh=0)
    with pytest.raises(DetectorError, match="iou_thresh must be between 0 and 1"):
        YOLODetector(iou_thresh=1)


@patch("visionguard.core.detector.YOLO")
def test_detector_load_calls_yolo(mock_yolo: MagicMock) -> None:
    detector = YOLODetector()
    detector.load("yolov8n.pt")
    mock_yolo.assert_called_once_with("yolov8n.pt")
    assert detector.model is mock_yolo.return_value


@patch("visionguard.core.detector.YOLO")
def test_detector_train_calls_train_and_returns_best_path(
    mock_yolo: MagicMock, tmp_path: Path
) -> None:
    detector = YOLODetector()
    detector.load("yolov8n.pt")
    mock_model = mock_yolo.return_value

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("data")
    project = tmp_path / "train_proj"
    name = "exp"
    best_path = project / name / "weights" / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.touch()

    result = detector.train(
        data_yaml=data_yaml,
        epochs=10,
        imgsz=320,
        batch=8,
        project=str(project),
        name=name,
    )

    mock_model.train.assert_called_once_with(
        data=str(data_yaml),
        epochs=10,
        imgsz=320,
        batch=8,
        lr0=0.01,
        project=str(project),
        name=name,
        device=detector.device,
    )
    assert result == best_path


@patch("visionguard.core.detector.YOLO")
def test_detector_validate_returns_metrics(mock_yolo: MagicMock) -> None:
    detector = YOLODetector()
    detector.load("yolov8n.pt")
    mock_model = mock_yolo.return_value

    mock_metrics = MagicMock()
    mock_metrics.box.map50 = 0.5
    mock_metrics.box.map75 = 0.6
    mock_metrics.box.map = 0.7
    mock_model.val.return_value = mock_metrics

    metrics = detector.validate("data.yaml")

    mock_model.val.assert_called_once_with(data="data.yaml", device=detector.device)
    assert metrics == {"map50": 0.5, "map75": 0.6, "map": 0.7}


@patch("visionguard.core.detector.YOLO")
def test_detector_predict_returns_results(mock_yolo: MagicMock) -> None:
    detector = YOLODetector()
    detector.load("yolov8n.pt")
    mock_model = mock_yolo.return_value

    mock_result = MagicMock()
    mock_model.return_value = [mock_result]

    results = detector.predict("image.jpg")

    mock_model.assert_called_once_with(
        source="image.jpg",
        conf=detector.conf_thresh,
        iou=detector.iou_thresh,
        device=detector.device,
        save=False,
    )
    assert results == [mock_result]


@patch("visionguard.core.detector.YOLO")
def test_detector_export_returns_path(mock_yolo: MagicMock) -> None:
    detector = YOLODetector()
    detector.load("yolov8n.pt")
    mock_model = mock_yolo.return_value

    mock_model.export.return_value = "/tmp/model.onnx"

    result = detector.export()

    mock_model.export.assert_called_once_with(
        format="onnx",
        imgsz=640,
        dynamic=True,
        opset=17,
    )
    assert result == Path("/tmp/model.onnx")


def test_detector_train_invalid_params() -> None:
    detector = YOLODetector()
    detector.model = MagicMock()
    with pytest.raises(DetectorError, match="epochs must be a positive integer"):
        detector.train("data.yaml", epochs=0)
    with pytest.raises(DetectorError, match="imgsz must be a positive integer"):
        detector.train("data.yaml", imgsz=0)
    with pytest.raises(DetectorError, match="batch must be a positive integer"):
        detector.train("data.yaml", batch=0)
