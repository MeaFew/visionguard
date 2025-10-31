# VisionGuard — YOLOv8 训练、评估与 ONNX 导出实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 YOLOv8 目标检测模型的训练、评估、ONNX 导出，以及结果后处理模块，使 Python 侧具备完整的缺陷检测能力。

**Architecture：** 在 Plan 1 数据准备基础上，新增 `visionguard/core/detector.py` 封装 Ultralytics YOLOv8，`scripts/train_yolo.py` 训练，`scripts/evaluate.py` 评估，`scripts/export_onnx.py` 导出 ONNX，`visionguard/core/postprocessor.py` 做 NMS 与结果解析。

**Tech Stack：** Python 3.11、Ultralytics YOLOv8、PyTorch、ONNX Runtime、OpenCV、PyYAML、pytest。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `configs/yolov8_neu_det.yaml` | YOLOv8 训练配置（数据集路径、类别、图像尺寸等） |
| `visionguard/core/detector.py` | YOLOv8 检测器封装类 |
| `visionguard/core/postprocessor.py` | NMS、置信度过滤、结果格式化 |
| `visionguard/utils/visualization.py` | 绘制检测框、保存结果图 |
| `visionguard/utils/metrics.py` | 评估指标计算辅助 |
| `scripts/train_yolo.py` | 训练入口脚本 |
| `scripts/evaluate.py` | 评估入口脚本 |
| `scripts/export_onnx.py` | ONNX 导出脚本 |
| `notebooks/01_explore_neu_det.ipynb` | 数据探索 notebook 骨架 |
| `tests/test_detector.py` | 检测器与后处理单元测试 |
| `tests/test_postprocessor.py` | 后处理测试 |

---

### Task 1: 创建 YOLOv8 训练配置与检测器封装

**Files:**
- Create: `E:/NewWorkProject/visionguard/configs/yolov8_neu_det.yaml`
- Create: `E:/NewWorkProject/visionguard/visionguard/core/detector.py`
- Create: `E:/NewWorkProject/visionguard/tests/test_detector.py`

- [ ] **Step 1: 创建训练配置 `configs/yolov8_neu_det.yaml`**

```yaml
# YOLOv8 training configuration for NEU-DET surface defect detection
path: data/processed
train: train/images
val: val/images
test: test/images

nc: 6
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
```

- [ ] **Step 2: 创建 `visionguard/core/detector.py`**

```python
"""YOLOv8 detector wrapper using Ultralytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

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
        self.model_name = model_name
        self.device = device or "cpu"
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model: YOLO | None = None

    def load(self, model_path: str | Path) -> "YOLODetector":
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
            self.load(self.model_name)

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

        best_path = Path(project) / name / "weights" / "best.pt"
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
        return {
            "map50": float(metrics.box.map50),
            "map75": float(metrics.box.map75),
            "map": float(metrics.box.map),
        }

    def predict(
        self, image_path: str | Path, save: bool = False, **kwargs: Any
    ) -> list[Any]:
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
```

- [ ] **Step 3: 创建 `tests/test_detector.py`**

```python
"""Tests for YOLOv8 detector wrapper."""

from pathlib import Path

import pytest

from visionguard.core.detector import DetectorError, YOLODetector


def test_detector_load_invalid() -> None:
    detector = YOLODetector()
    with pytest.raises(DetectorError):
        detector.load("nonexistent_model.pt")


def test_detector_predict_without_load() -> None:
    detector = YOLODetector()
    with pytest.raises(DetectorError):
        detector.predict("nonexistent.jpg")


def test_detector_validate_without_load() -> None:
    detector = YOLODetector()
    with pytest.raises(DetectorError):
        detector.validate("data.yaml")


def test_detector_export_without_load() -> None:
    detector = YOLODetector()
    with pytest.raises(DetectorError):
        detector.export()
```

- [ ] **Step 4: 运行测试与 lint**

Run:
```bash
cd E:/NewWorkProject/visionguard && .venv/Scripts/python -m pytest tests/test_detector.py -v && .venv/Scripts/python -m ruff check visionguard/core/detector.py tests/test_detector.py
```

Expected: 4 tests PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
cd E:/NewWorkProject/visionguard
git add configs/yolov8_neu_det.yaml visionguard/core/detector.py tests/test_detector.py
git commit -m "2025-08-16: add YOLOv8 detector wrapper and training config" --date="2025-08-16T10:00:00"
```

---

### Task 2: 实现训练脚本与评估脚本

**Files:**
- Create: `E:/NewWorkProject/visionguard/scripts/train_yolo.py`
- Create: `E:/NewWorkProject/visionguard/scripts/evaluate.py`
- Modify: `E:/NewWorkProject/visionguard/Makefile`（可选，添加 `train` / `evaluate` targets）

- [ ] **Step 1: 创建训练脚本 `scripts/train_yolo.py`**

```python
"""Train YOLOv8 on NEU-DET dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from visionguard.core.detector import YOLODetector
from visionguard.exceptions import VisionGuardError

DEFAULT_CONFIG = Path("configs/yolov8_neu_det.yaml")
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_PROJECT = "models"
DEFAULT_NAME = "train"


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
    args = parser.parse_args()

    detector = YOLODetector(model_name=args.model, device=args.device)
    try:
        best_path = detector.train(
            data_yaml=args.config,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
        )
        print(f"Training complete. Best model: {best_path}")
    except VisionGuardError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 创建评估脚本 `scripts/evaluate.py`**

```python
"""Evaluate a trained YOLOv8 model on NEU-DET test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visionguard.core.detector import YOLODetector
from visionguard.exceptions import VisionGuardError

DEFAULT_CONFIG = Path("configs/yolov8_neu_det.yaml")
DEFAULT_MODEL = Path("models/train/weights/best.pt")
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
    args = parser.parse_args()

    detector = YOLODetector(device=args.device)
    try:
        detector.load(args.model)
        metrics = detector.validate(data_yaml=args.config)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Evaluation metrics: {metrics}")
        print(f"Metrics saved to {output_path}")
    except VisionGuardError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 更新 `Makefile` 添加 train/evaluate 目标**

Edit `Makefile` to add after the `test` target:

```makefile
train:
	$(PYTHON) scripts/train_yolo.py

evaluate:
	$(PYTHON) scripts/evaluate.py
```

And add `train` / `evaluate` to `.PHONY`.

- [ ] **Step 4: 语法与 lint 检查**

Run:
```bash
cd E:/NewWorkProject/visionguard && .venv/Scripts/python -m py_compile scripts/train_yolo.py scripts/evaluate.py && .venv/Scripts/python -m ruff check scripts/train_yolo.py scripts/evaluate.py
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
cd E:/NewWorkProject/visionguard
git add scripts/train_yolo.py scripts/evaluate.py Makefile
git commit -m "2025-08-20: add YOLOv8 training and evaluation scripts" --date="2025-08-20T10:00:00"
```

---

### Task 3: 实现 ONNX 导出与后处理模块

**Files:**
- Create: `E:/NewWorkProject/visionguard/visionguard/core/postprocessor.py`
- Create: `E:/NewWorkProject/visionguard/scripts/export_onnx.py`
- Create: `E:/NewWorkProject/visionguard/visionguard/utils/visualization.py`
- Create: `E:/NewWorkProject/visionguard/tests/test_postprocessor.py`

- [ ] **Step 1: 创建后处理模块 `visionguard/core/postprocessor.py`**

```python
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

        order = order[np.where(iou <= iou_threshold)[0] + 1]

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
```

- [ ] **Step 2: 创建 `scripts/export_onnx.py`**

```python
"""Export trained YOLOv8 model to ONNX format."""

from __future__ import annotations

import argparse
from pathlib import Path

from visionguard.core.detector import YOLODetector
from visionguard.exceptions import VisionGuardError

DEFAULT_MODEL = Path("models/train/weights/best.pt")
DEFAULT_IMGSZ = 640
DEFAULT_OPSET = 17


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to trained YOLOv8 weights",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Input image size for ONNX export",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=True,
        help="Use dynamic batch/axes",
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Disable dynamic axes",
    )
    args = parser.parse_args()

    dynamic = not args.no_dynamic

    detector = YOLODetector()
    try:
        detector.load(args.model)
        onnx_path = detector.export(
            format="onnx",
            imgsz=args.imgsz,
            dynamic=dynamic,
            opset=args.opset,
        )
        print(f"ONNX model exported to {onnx_path}")
    except VisionGuardError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 创建可视化工具 `visionguard/utils/visualization.py`**

```python
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
```

- [ ] **Step 4: 创建 `tests/test_postprocessor.py`**

```python
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
```

- [ ] **Step 5: 运行测试与 lint**

Run:
```bash
cd E:/NewWorkProject/visionguard && .venv/Scripts/python -m pytest tests/test_postprocessor.py -v && .venv/Scripts/python -m ruff check visionguard/core/postprocessor.py scripts/export_onnx.py visionguard/utils/visualization.py tests/test_postprocessor.py
```

Expected: tests PASS, ruff clean.

- [ ] **Step 6: Commit**

```bash
cd E:/NewWorkProject/visionguard
git add visionguard/core/postprocessor.py scripts/export_onnx.py visionguard/utils/visualization.py tests/test_postprocessor.py
git commit -m "2025-08-25: add ONNX export, NMS postprocessor and visualization" --date="2025-08-25T10:00:00"
```

---

### Task 4: 添加 Notebook 骨架与最终整理

**Files:**
- Create: `E:/NewWorkProject/visionguard/notebooks/01_explore_neu_det.ipynb`

- [ ] **Step 1: 创建 Notebook 骨架**

Create a minimal Jupyter notebook with the following cells:

Cell 1 (markdown):
```markdown
# NEU-DET 数据探索

本 notebook 用于探索 NEU-DET 数据集：
- 查看缺陷类别分布
- 可视化图像与标注框
- 预览 OpenCV 预处理效果
- 分析训练/验证/测试集划分
```

Cell 2 (code):
```python
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

from visionguard.utils.dataset_utils import NEU_DET_CLASSES, ID_TO_CLASS

DATA_DIR = Path("../data/processed")
SPLIT = "train"
```

Cell 3 (code):
```python
# Count class distribution
from collections import Counter

label_dir = DATA_DIR / SPLIT / "labels"
class_counts = Counter()
for label_path in sorted(label_dir.glob("*.txt")):
    with open(label_path) as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

print({ID_TO_CLASS[i]: class_counts[i] for i in sorted(class_counts)})
```

Cell 4 (code):
```python
# Visualize a sample image with annotations
def plot_image_with_boxes(image_path, label_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x, y, bw, bh = map(float, parts[1:])
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, ID_TO_CLASS[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Example usage (when data exists):
# img_path = next((DATA_DIR / SPLIT / "images").glob("*.jpg"))
# plot_image_with_boxes(img_path, DATA_DIR / SPLIT / "labels" / img_path.with_suffix(".txt").name)
```

- [ ] **Step 2: Commit**

```bash
cd E:/NewWorkProject/visionguard
git add notebooks/01_explore_neu_det.ipynb
git commit -m "2025-08-28: add NEU-DET exploration notebook skeleton" --date="2025-08-28T10:00:00"
```

---

## 自审查

- **Spec coverage：** 本 Plan 覆盖设计文档第 6 章训练/评估/导出阶段、第 7 章关键模块设计中的检测器与后处理模块。
- **Placeholder scan：** 无 TBD/TODO；代码块完整。
- **Type consistency：** `YOLODetector` 的 `load`/`train`/`validate`/`predict`/`export` 签名一致；`Detection` dataclass 一致。
- **Gap：** 实际训练 YOLOv8 需要下载 NEU-DET 数据集并耗时较长，Plan 2 提供脚本但不强制在此环境中完成完整训练。后续 Plan 3 将基于导出的 ONNX 模型做 C++ 推理。

---

## 执行交接

Plan 2 已完成并保存到 `E:/NewWorkProject/docs/superpowers/plans/2025-08-15-visionguard-training-and-evaluation.md`。

**执行选项：**

1. **Subagent-Driven (recommended)** — 为每个 Task 派发独立 subagent，我负责审查与集成
2. **Inline Execution** — 在当前会话中按 Task 顺序直接执行

请选择执行方式。
