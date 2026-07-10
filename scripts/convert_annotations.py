"""Convert NEU-DET XML annotations to YOLO format and split dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from visionguard.exceptions import AnnotationError, DatasetError, VisionGuardError
from visionguard.logging_setup import get_logger, setup_logging
from visionguard.utils.dataset_utils import (
    CLASS_TO_ID,
    NEU_DET_CLASSES,
    ensure_dir,
    split_dataset,
)

logger = get_logger(__name__)

DEFAULT_DATA_DIR = Path("data")


def parse_neu_det_xml(xml_path: Path) -> list[dict[str, float | int]]:
    """Parse a single NEU-DET XML annotation file.

    Args:
        xml_path: Path to the XML annotation file.

    Returns:
        List of annotation dicts with keys: class_id, x_center, y_center, width, height.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise AnnotationError(f"Malformed XML in {xml_path}: {exc}") from exc
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise AnnotationError(f"Missing <size> in {xml_path}")

    width_elem = size.find("width")
    if width_elem is None or width_elem.text is None:
        raise AnnotationError(f"Missing <width> in {xml_path}")
    try:
        width = int(width_elem.text.strip())
    except ValueError as exc:
        raise AnnotationError(f"Non-numeric <width> in {xml_path}") from exc

    height_elem = size.find("height")
    if height_elem is None or height_elem.text is None:
        raise AnnotationError(f"Missing <height> in {xml_path}")
    try:
        height = int(height_elem.text.strip())
    except ValueError as exc:
        raise AnnotationError(f"Non-numeric <height> in {xml_path}") from exc

    if width <= 0 or height <= 0:
        raise AnnotationError(f"Invalid image size in {xml_path}")

    annotations: list[dict[str, float | int]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower()
        if name not in CLASS_TO_ID:
            raise AnnotationError(f"Unknown class '{name}' in {xml_path}")

        bndbox = obj.find("bndbox")
        if bndbox is None:
            raise AnnotationError(f"Missing <bndbox> in {xml_path}")

        def _get_float(box: ET.Element, tag: str, path: Path) -> float:
            elem = box.find(tag)
            if elem is None or elem.text is None:
                raise AnnotationError(f"Missing <{tag}> in {path}")
            try:
                return float(elem.text.strip())
            except ValueError as exc:
                raise AnnotationError(f"Non-numeric <{tag}> in {path}") from exc

        xmin = _get_float(bndbox, "xmin", xml_path)
        ymin = _get_float(bndbox, "ymin", xml_path)
        xmax = _get_float(bndbox, "xmax", xml_path)
        ymax = _get_float(bndbox, "ymax", xml_path)

        if not (0 <= xmin < xmax <= width):
            raise AnnotationError(f"Invalid x bounds in {xml_path}")
        if not (0 <= ymin < ymax <= height):
            raise AnnotationError(f"Invalid y bounds in {xml_path}")

        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        annotations.append(
            {
                "class_id": CLASS_TO_ID[name],
                "x_center": x_center,
                "y_center": y_center,
                "width": box_width,
                "height": box_height,
            }
        )

    return annotations


def write_yolo_label(annotations: list[dict[str, float | int]], dst: Path) -> None:
    """Write annotations to YOLO label file."""
    ensure_dir(dst.parent)
    with open(dst, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(
                f"{ann['class_id']} {ann['x_center']:.6f} "
                f"{ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
            )


def convert_neu_det(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    """Convert NEU-DET annotations to YOLO format and split dataset.

    Args:
        data_dir: Root data directory.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed.

    Returns:
        Path to processed data directory.
    """
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        # Only remove expected subdirectories to avoid accidental data loss.
        for sub in ["train", "val", "test", "neu_det.yaml"]:
            path = processed_dir / sub
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

    # Locate all NEU-DET images/annotations directories (handles nested archives
    # such as archive/NEU-DET/train and archive/NEU-DET/validation, where images
    # may be grouped into class subdirectories).
    image_dirs = sorted(raw_dir.rglob("images"), key=lambda p: str(p))
    if not image_dirs:
        raise DatasetError(f"Could not find images directory under {raw_dir}")

    supported_exts = {".jpg", ".jpeg", ".bmp", ".png"}
    valid_pairs: list[tuple[Path, Path]] = []
    for src_image_dir in image_dirs:
        src_anno_dir = src_image_dir.parent / "annotations"
        if not src_anno_dir.exists():
            src_anno_dir = src_image_dir.parent / "Annotations"
        if not src_anno_dir.exists():
            continue

        for img_path in src_image_dir.rglob("*"):
            if img_path.suffix.lower() not in supported_exts:
                continue
            xml_path = src_anno_dir / img_path.with_suffix(".xml").name
            if xml_path.exists():
                valid_pairs.append((img_path, xml_path))

    if not valid_pairs:
        raise DatasetError("No image/annotation pairs found")

    # Deduplicate by image name in case the same image appears in multiple splits.
    seen: set[str] = set()
    unique_pairs: list[tuple[Path, Path]] = []
    for img_path, xml_path in valid_pairs:
        if img_path.name in seen:
            continue
        seen.add(img_path.name)
        unique_pairs.append((img_path, xml_path))
    valid_pairs = unique_pairs

    train_pairs, val_pairs, test_pairs = split_dataset(
        valid_pairs, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    processed_dir = ensure_dir(processed_dir)

    for split_name, pairs in [
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ]:
        split_img_dir = ensure_dir(processed_dir / split_name / "images")
        split_lbl_dir = ensure_dir(processed_dir / split_name / "labels")

        for img_path, xml_path in pairs:
            shutil.copy2(img_path, split_img_dir / img_path.name)
            annotations = parse_neu_det_xml(xml_path)
            label_path = split_lbl_dir / img_path.with_suffix(".txt").name
            write_yolo_label(annotations, label_path)

    # Write dataset YAML for Ultralytics
    yaml_path = processed_dir / "neu_det.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {processed_dir.resolve()}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write("\n")
        f.write(f"nc: {len(NEU_DET_CLASSES)}\n")
        f.write(f"names: {NEU_DET_CLASSES}\n")

    logger.info(f"Processed dataset saved to {processed_dir}")
    logger.info(f"  Train: {len(train_pairs)} images")
    logger.info(f"  Val:   {len(val_pairs)} images")
    logger.info(f"  Test:  {len(test_pairs)} images")

    return processed_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NEU-DET annotations to YOLO format")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Root data directory",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split",
    )
    args = parser.parse_args()
    try:
        convert_neu_det(args.data_dir, args.train_ratio, args.val_ratio, args.seed)
    except VisionGuardError as exc:
        logger.error(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    setup_logging()
    main()
