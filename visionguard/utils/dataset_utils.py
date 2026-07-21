"""Utilities for dataset handling, class mapping, and train/val/test splitting."""

from __future__ import annotations

import random
import shutil
from collections.abc import Sequence
from pathlib import Path

NEU_DET_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

NEU_DET_SHORT_NAMES = ["Cr", "In", "Pa", "Ps", "Rs", "Sc"]

CLASS_TO_ID = {name: idx for idx, name in enumerate(NEU_DET_CLASSES)}
ID_TO_CLASS = dict(enumerate(NEU_DET_CLASSES))


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_dataset(
    image_paths: Sequence[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split a list of image paths into train/val/test sets deterministically.

    Args:
        image_paths: List of image file paths.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_paths, val_paths, test_paths). If ``image_paths`` is
        empty, all three returned lists are empty.
    """
    if train_ratio < 0:
        raise ValueError("train_ratio must be non-negative")
    if val_ratio < 0:
        raise ValueError("val_ratio must be non-negative")
    if train_ratio + val_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be greater than 0")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    data = list(image_paths)
    # Use a local RNG so the global random state is not disturbed.
    rng = random.Random(seed)
    rng.shuffle(data)

    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]

    return train, val, test


def copy_files(file_list: Sequence[Path], dst_dir: Path) -> None:
    """Copy files into destination directory, preserving basenames."""
    ensure_dir(dst_dir)
    for src in file_list:
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst_dir / src.name)
