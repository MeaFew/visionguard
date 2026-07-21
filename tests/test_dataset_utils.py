"""Tests for dataset splitting and file copying utilities."""

import random
from pathlib import Path

import pytest

from visionguard.utils.dataset_utils import copy_files, split_dataset


def _make_images(root: Path, count: int = 10) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        path = root / f"img_{i:02d}.jpg"
        path.touch()
        paths.append(path)
    return paths


def test_split_dataset_ratios_and_partition(tmp_path: Path) -> None:
    images = _make_images(tmp_path, 10)
    train, val, test = split_dataset(images, train_ratio=0.6, val_ratio=0.2, seed=0)

    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
    # The three sets partition the input: disjoint and covering everything.
    combined = train + val + test
    assert len(set(combined)) == len(images)
    assert set(combined) == set(images)


def test_split_dataset_deterministic_for_same_seed(tmp_path: Path) -> None:
    images = _make_images(tmp_path, 10)
    first = split_dataset(images, train_ratio=0.6, val_ratio=0.2, seed=7)
    second = split_dataset(images, train_ratio=0.6, val_ratio=0.2, seed=7)
    assert first == second


def test_split_dataset_does_not_touch_global_rng(tmp_path: Path) -> None:
    images = _make_images(tmp_path, 10)
    random.seed(1234)
    expected = [random.random() for _ in range(3)]

    random.seed(1234)
    split_dataset(images, seed=99)
    assert [random.random() for _ in range(3)] == expected


def test_split_dataset_invalid_ratios(tmp_path: Path) -> None:
    images = _make_images(tmp_path, 4)
    with pytest.raises(ValueError):
        split_dataset(images, train_ratio=-0.1)
    with pytest.raises(ValueError):
        split_dataset(images, train_ratio=0.9, val_ratio=0.2)


def test_copy_files_copies_preserving_names(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    files = []
    for name in ("a.jpg", "b.txt"):
        path = src_dir / name
        path.write_text(name)
        files.append(path)

    dst = tmp_path / "dst"
    copy_files(files, dst)

    for path in files:
        copied = dst / path.name
        assert copied.exists()
        assert copied.read_text() == path.name


def test_copy_files_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        copy_files([tmp_path / "missing.jpg"], tmp_path / "dst")
