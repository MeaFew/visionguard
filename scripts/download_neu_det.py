"""Download and extract the NEU-DET surface defect dataset."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

import gdown

from visionguard.utils.dataset_utils import ensure_dir

GOOGLE_DRIVE_FILE_ID = "1qE2x0L3CdE2qHQWRwCN7DCq9RllqT-kb"
GOOGLE_DRIVE_URL = (
    "https://drive.google.com/uc?export=download&id=1qE2x0L3CdE2qHQWRwCN7DCq9RllqT-kb"
)

DEFAULT_DATA_DIR = Path("data")


def extract_archive(archive_path: Path, dst_dir: Path) -> None:
    """Extract zip or tar.gz archive safely, guarding against path traversal."""
    dst_dir = dst_dir.resolve()

    def _is_safe(target: Path) -> bool:
        try:
            target.relative_to(dst_dir)
            return True
        except ValueError:
            return False

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                target = (dst_dir / member.filename).resolve()
                if not _is_safe(target):
                    print(f"Skipping unsafe archive member: {member.filename}")
                    continue
                zf.extract(member, dst_dir)
    elif archive_path.suffixes == [".tar", ".gz"]:
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                target = (dst_dir / member.name).resolve()
                if not _is_safe(target):
                    print(f"Skipping unsafe archive member: {member.name}")
                    continue
                tf.extract(member, dst_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def download_neu_det(data_dir: Path | str = DEFAULT_DATA_DIR, force: bool = False) -> Path:
    """Download and extract NEU-DET dataset.

    Args:
        data_dir: Directory to store the dataset.
        force: Re-download even if already exists.

    Returns:
        Path to the raw data directory.
    """
    data_dir = Path(data_dir)
    raw_dir = ensure_dir(data_dir / "raw")
    archive_path = raw_dir / "NEU-DET.zip"

    if archive_path.exists() and not force:
        print(f"Archive already exists: {archive_path}")
    else:
        print("Downloading NEU-DET from Google Drive ...")
        try:
            gdown.download(
                id=GOOGLE_DRIVE_FILE_ID,
                output=str(archive_path),
                quiet=False,
                fuzzy=True,
            )
        except Exception as exc:  # pragma: no cover
            print(f"Google Drive download failed: {exc}")
            print(
                "Please manually download NEU-DET from "
                f"{GOOGLE_DRIVE_URL} and place it at {archive_path}"
            )
            raise

    extracted_marker = raw_dir / ".extracted"
    if not extracted_marker.exists() or force:
        print(f"Extracting {archive_path} ...")
        extract_archive(archive_path, raw_dir)
        extracted_marker.touch()
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

    return raw_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NEU-DET dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory to store dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-extraction",
    )
    args = parser.parse_args()
    download_neu_det(args.data_dir, args.force)


if __name__ == "__main__":
    main()
