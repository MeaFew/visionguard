"""YOLOv8 detector wrapper using Ultralytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.engine.results import Results

from visionguard.exceptions import VisionGuardError


class DetectorError(VisionGuardError):
    """Raised when detector operations fail."""


class YOLODetector:
    """Wrapper around Ultralytics YOLOv8 for training, validation, prediction, and export."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str | None = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ) -> None:
        if not 0 < conf_thresh < 1:
            raise DetectorError("conf_thresh must be between 0 and 1 (exclusive)")
        if not 0 < iou_thresh < 1:
            raise DetectorError("iou_thresh must be between 0 and 1 (exclusive)")
        self.model_name = model_name
        self.device = device or "cpu"
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model: YOLO | None = None

    def load(self, model_path: str | Path) -> YOLODetector:
        """Load a YOLO model from a path or pretrained name."""
        try:
            self.model = YOLO(str(model_path))
        except Exception as exc:
            raise DetectorError(f"Failed to load YOLO model from {model_path}: {exc}") from exc
        return self

    def train(
        self,
        data_yaml: str | Path,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        lr0: float = 0.01,
        project: str = "models",
        name: str = "train",
        **kwargs: Any,
    ) -> Path:
        """Train YOLOv8 on the given dataset YAML."""
        if self.model is None:
            raise DetectorError("Model not loaded. Call load() first.")
        if epochs <= 0:
            raise DetectorError("epochs must be a positive integer")
        if imgsz <= 0:
            raise DetectorError("imgsz must be a positive integer")
        if batch <= 0:
            raise DetectorError("batch must be a positive integer")

        try:
            self.model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                lr0=lr0,
                project=project,
                name=name,
                device=self.device,
                **kwargs,
            )
        except Exception as exc:
            raise DetectorError(f"Training failed: {exc}") from exc

        # Ultralytics may place results under runs/detect/ depending on settings.
        trainer = getattr(self.model, "trainer", None)
        trainer_save_dir = getattr(trainer, "save_dir", None)
        if isinstance(trainer_save_dir, (str, Path)):
            save_dir = Path(trainer_save_dir)
        else:
            save_dir = Path(project) / name
        best_path = save_dir / "weights" / "best.pt"
        if not best_path.exists():
            raise DetectorError(f"Expected best model not found at {best_path}")
        return best_path

    def validate(self, data_yaml: str | Path, **kwargs: Any) -> dict[str, Any]:
        """Validate the model on a dataset."""
        if self.model is None:
            raise DetectorError("Model not loaded. Call load() first.")
        try:
            metrics = self.model.val(data=str(data_yaml), device=self.device, **kwargs)
        except Exception as exc:
            raise DetectorError(f"Validation failed: {exc}") from exc
        try:
            return {
                "map50": float(metrics.box.map50),
                "map75": float(metrics.box.map75),
                "map": float(metrics.box.map),
            }
        except Exception as exc:
            raise DetectorError(f"Failed to extract validation metrics: {exc}") from exc

    def predict(self, image_path: str | Path, save: bool = False, **kwargs: Any) -> list[Results]:
        """Run inference on a single image or list of images."""
        if self.model is None:
            raise DetectorError("Model not loaded. Call load() first.")
        try:
            results = self.model(
                source=str(image_path),
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                device=self.device,
                save=save,
                **kwargs,
            )
        except Exception as exc:
            raise DetectorError(f"Prediction failed: {exc}") from exc
        return results

    def export(
        self,
        format: str = "onnx",
        imgsz: int = 640,
        dynamic: bool = True,
        opset: int = 17,
        **kwargs: Any,
    ) -> Path:
        """Export the loaded model to the specified format."""
        if self.model is None:
            raise DetectorError("Model not loaded. Call load() first.")
        try:
            path = self.model.export(
                format=format,
                imgsz=imgsz,
                dynamic=dynamic,
                opset=opset,
                **kwargs,
            )
        except Exception as exc:
            raise DetectorError(f"Export failed: {exc}") from exc
        return Path(path)
