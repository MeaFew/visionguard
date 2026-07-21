"""Tests for the NEU-DET download script's archive extraction safeguards."""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.download_neu_det import extract_archive


def _make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def _make_tar_gz(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_extract_zip_skips_path_traversal_members(tmp_path: Path) -> None:
    archive = tmp_path / "NEU-DET.zip"
    _make_zip(
        archive,
        {
            "images/ok.jpg": b"ok",
            "../evil.txt": b"pwned",
            "nested/../../evil2.txt": b"pwned",
        },
    )
    dst = tmp_path / "raw"

    extract_archive(archive, dst)

    # Safe members are extracted under the destination directory.
    assert (dst / "images" / "ok.jpg").read_bytes() == b"ok"
    # Traversal members are rejected: nothing written outside or inside dst.
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "evil2.txt").exists()
    assert not (dst / "evil.txt").exists()
    assert not (dst / "evil2.txt").exists()


def test_extract_tar_gz_skips_path_traversal_members(tmp_path: Path) -> None:
    archive = tmp_path / "NEU-DET.tar.gz"
    _make_tar_gz(
        archive,
        {
            "images/ok.jpg": b"ok",
            "../evil.txt": b"pwned",
        },
    )
    dst = tmp_path / "raw"

    extract_archive(archive, dst)

    assert (dst / "images" / "ok.jpg").read_bytes() == b"ok"
    assert not (tmp_path / "evil.txt").exists()
    assert not (dst / "evil.txt").exists()


def test_extract_archive_rejects_unsupported_format(tmp_path: Path) -> None:
    archive = tmp_path / "NEU-DET.rar"
    archive.touch()
    with pytest.raises(ValueError, match="Unsupported archive format"):
        extract_archive(archive, tmp_path / "raw")
