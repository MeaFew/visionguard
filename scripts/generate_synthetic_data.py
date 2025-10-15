"""Generate a small synthetic NEU-DET-like dataset for demo/training pipeline verification.

This is intended as a fallback when the real NEU-DET dataset cannot be downloaded
(e.g. Google Drive permission/quota issues). The synthetic images simulate steel
surface defects with bounding-box annotations in PASCAL VOC XML format, which can
be converted to YOLO format by scripts/convert_annotations.py.
"""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from visionguard.utils.dataset_utils import NEU_DET_CLASSES, ensure_dir

DEFAULT_DATA_DIR = Path("data")
DEFAULT_NUM_IMAGES = 100
DEFAULT_IMAGE_SIZE = 640


def generate_steel_background(size: int) -> np.ndarray:
    """Generate a gray steel-like background with subtle noise."""
    bg = np.full((size, size, 3), 160, dtype=np.uint8)
    noise = np.random.randint(-15, 15, (size, size, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 100, 220).astype(np.uint8)
    return bg


def draw_scratch(image: np.ndarray, bbox: list[int]) -> None:
    """Draw a scratch-like defect."""
    x1, y1, x2, y2 = bbox
    thickness = max(1, random.randint(1, 3))
    color = (random.randint(30, 80),) * 3
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def draw_patch(image: np.ndarray, bbox: list[int]) -> None:
    """Draw a patch/scale defect."""
    x1, y1, x2, y2 = bbox
    color = (random.randint(40, 90),) * 3
    cv2.ellipse(image, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, color, -1)


def draw_inclusion(image: np.ndarray, bbox: list[int]) -> None:
    """Draw an inclusion defect."""
    x1, y1, x2, y2 = bbox
    color = (random.randint(20, 70),) * 3
    cv2.circle(image, ((x1 + x2) // 2, (y1 + y2) // 2), (x2 - x1) // 2, color, -1)


def draw_pitted(image: np.ndarray, bbox: list[int]) -> None:
    """Draw pitted surface defect."""
    x1, y1, x2, y2 = bbox
    for _ in range(random.randint(3, 8)):
        cx = random.randint(x1, x2)
        cy = random.randint(y1, y2)
        r = random.randint(2, 5)
        color = (random.randint(40, 80),) * 3
        cv2.circle(image, (cx, cy), r, color, -1)


def draw_crazing(image: np.ndarray, bbox: list[int]) -> None:
    """Draw crazing (network of fine cracks)."""
    x1, y1, x2, y2 = bbox
    for _ in range(random.randint(2, 5)):
        sx, sy = random.randint(x1, x2), random.randint(y1, y2)
        ex, ey = sx + random.randint(10, 40), sy + random.randint(-20, 20)
        color = (random.randint(40, 80),) * 3
        cv2.line(image, (sx, sy), (ex, ey), color, 1)


DEFECT_DRAWERS = {
    "crazing": draw_crazing,
    "inclusion": draw_inclusion,
    "patches": draw_patch,
    "pitted_surface": draw_pitted,
    "rolled-in_scale": draw_patch,
    "scratches": draw_scratch,
}


def generate_image(
    idx: int,
    size: int,
    data_dir: Path,
    max_defects: int = 3,
) -> None:
    """Generate one synthetic image and its XML annotation."""
    raw_dir = ensure_dir(data_dir / "raw" / "synthetic")
    images_dir = ensure_dir(raw_dir / "images")
    annos_dir = ensure_dir(raw_dir / "annotations")

    image = generate_steel_background(size)
    num_defects = random.randint(1, max_defects)

    annotations: list[dict[str, int | str]] = []
    for _ in range(num_defects):
        class_name = random.choice(NEU_DET_CLASSES)
        w = random.randint(30, 120)
        h = random.randint(20, 80)
        x1 = random.randint(10, size - w - 10)
        y1 = random.randint(10, size - h - 10)
        x2 = x1 + w
        y2 = y1 + h

        drawer = DEFECT_DRAWERS[class_name]
        drawer(image, [x1, y1, x2, y2])

        annotations.append(
            {
                "name": class_name,
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
            }
        )

    image_path = images_dir / f"synthetic_{idx:04d}.jpg"
    cv2.imwrite(str(image_path), image)

    # Write PASCAL VOC XML
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = image_path.name
    size_elem = ET.SubElement(annotation, "size")
    ET.SubElement(size_elem, "width").text = str(size)
    ET.SubElement(size_elem, "height").text = str(size)
    ET.SubElement(size_elem, "depth").text = "3"

    for ann in annotations:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = str(ann["name"])
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(ann["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(ann["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(ann["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(ann["ymax"])

    xml_path = annos_dir / f"synthetic_{idx:04d}.xml"
    ET.ElementTree(annotation).write(xml_path, encoding="utf-8", xml_declaration=True)


def generate_synthetic_dataset(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    num_images: int = DEFAULT_NUM_IMAGES,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = 42,
) -> None:
    """Generate the full synthetic dataset."""
    random.seed(seed)
    np.random.seed(seed)
    data_dir = Path(data_dir)

    for i in range(num_images):
        generate_image(i, image_size, data_dir)

    print(f"Generated {num_images} synthetic images in {data_dir / 'raw' / 'synthetic'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic NEU-DET-like dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Root data directory",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="Number of synthetic images to generate",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Image width/height in pixels",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    generate_synthetic_dataset(args.data_dir, args.num_images, args.image_size, args.seed)


if __name__ == "__main__":
    main()
