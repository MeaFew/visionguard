"""Download and extract the NEU-DET surface defect dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import gdown

from visionguard.logging_setup import get_logger, setup_logging
from visionguard.utils.dataset_utils import ensure_dir

logger = get_logger(__name__)

GOOGLE_DRIVE_FILE_ID = "1qE2x0L3CdE2qHQWRwCN7DCq9RllqT-kb"
GOOGLE_DRIVE_URL = (
    "https://drive.google.com/uc?export=download&id=1qE2x0L3CdE2qHQWRwCN7DCq9RllqT-kb"
)

# A reliable NEU-DET mirror on Kaggle.
DEFAULT_KAGGLE_DATASET = "danielfinez/neu-det-steel-surface-defect-detection-dataset"

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
                    logger.info(f"Skipping unsafe archive member: {member.filename}")
                    continue
                zf.extract(member, dst_dir)
    elif archive_path.suffixes == [".tar", ".gz"]:
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                target = (dst_dir / member.name).resolve()
                if not _is_safe(target):
                    logger.info(f"Skipping unsafe archive member: {member.name}")
                    continue
                tf.extract(member, dst_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def _find_archive(raw_dir: Path) -> Path | None:
    """Return the most recently modified zip/tar.gz archive in raw_dir."""
    archives = sorted(
        raw_dir.glob("*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in archives:
        if candidate.suffix == ".zip" or candidate.suffixes == [".tar", ".gz"]:
            return candidate
    return None


def _download_from_google_drive(archive_path: Path) -> None:
    """Download NEU-DET archive from the official Google Drive mirror."""
    logger.info("Downloading NEU-DET from Google Drive ...")
    try:
        gdown.download(
            id=GOOGLE_DRIVE_FILE_ID,
            output=str(archive_path),
            quiet=False,
        )
    except Exception as exc:  # pragma: no cover
        logger.error(f"Google Drive download failed: {exc}")
        logger.error(
            "Please manually download NEU-DET from "
            f"{GOOGLE_DRIVE_URL} and place it at {archive_path}"
        )
        raise


def _download_from_kaggle(raw_dir: Path, dataset: str) -> Path:
    """Download NEU-DET archive from a Kaggle dataset mirror.

    Returns the path to the downloaded archive.
    """
    logger.info(f"Downloading NEU-DET from Kaggle dataset {dataset} ...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset,
                "-p",
                str(raw_dir),
                "--force",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Kaggle download failed. Common causes:")
        logger.error("  1. API token not configured (~/.kaggle/kaggle.json).")
        logger.error("  2. Not accepted the dataset's terms on kaggle.com.")
        logger.error("  3. Network / VPN cannot reach Kaggle.")
        raise RuntimeError(
            "Kaggle download failed. See docs/dataset.md for setup instructions."
        ) from exc

    archive_path = _find_archive(raw_dir)
    if archive_path is None:
        raise RuntimeError(f"Kaggle download did not produce an archive in {raw_dir}")
    return archive_path


def download_neu_det(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    source: str = "google-drive",
    kaggle_dataset: str = DEFAULT_KAGGLE_DATASET,
    force: bool = False,
) -> Path:
    """Download and extract NEU-DET dataset.

    Args:
        data_dir: Directory to store the dataset.
        source: Download source, either "google-drive" or "kaggle".
        kaggle_dataset: Kaggle dataset slug (owner/name).
        force: Re-download even if already exists.

    Returns:
        Path to the raw data directory.
    """
    data_dir = Path(data_dir)
    raw_dir = ensure_dir(data_dir / "raw")

    extracted_marker = raw_dir / ".extracted"
    if force:
        extracted_marker.unlink(missing_ok=True)

    if source == "google-drive":
        archive_path = raw_dir / "NEU-DET.zip"
        if archive_path.exists() and not force:
            logger.info(f"Archive already exists: {archive_path}")
        else:
            _download_from_google_drive(archive_path)
    elif source == "kaggle":
        # Re-use an existing archive unless force is set.
        archive_path = _find_archive(raw_dir)
        if archive_path is None or force:
            archive_path = _download_from_kaggle(raw_dir, kaggle_dataset)
        else:
            logger.info(f"Reusing existing archive: {archive_path}")
    else:
        raise ValueError(f"Unknown download source: {source}")

    if not extracted_marker.exists():
        logger.info(f"Extracting {archive_path} ...")
        extract_archive(archive_path, raw_dir)
        extracted_marker.touch()
        logger.info("Extraction complete.")
    else:
        logger.info("Dataset already extracted.")

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
        "--source",
        type=str,
        default="google-drive",
        choices=["google-drive", "kaggle"],
        help="Download source",
    )
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default=DEFAULT_KAGGLE_DATASET,
        help="Kaggle dataset slug to use when --source kaggle",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-extraction",
    )
    args = parser.parse_args()
    download_neu_det(
        args.data_dir,
        source=args.source,
        kaggle_dataset=args.kaggle_dataset,
        force=args.force,
    )


if __name__ == "__main__":
    setup_logging()
    main()
