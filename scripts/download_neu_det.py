"""Download and extract the NEU-DET surface defect dataset."""

from __future__ import annotations

import argparse
import tarfile
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from visionguard.utils.dataset_utils import ensure_dir

FALLBACK_URL = "https://github.com/abin24/Magnetic-tile-defect-datasets./raw/master/NEU-DET.zip"

DEFAULT_DATA_DIR = Path("data")


def download_file(
    url: str, dst: Path, chunk_size: int = 8192, max_retries: int = 3
) -> None:
    """Download a file with progress bar and retry on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with open(dst, "wb") as f, tqdm(
                desc=dst.name,
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            return
        except requests.RequestException as exc:
            last_exc = exc
            print(f"Download attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise last_exc or RuntimeError("Download failed after all retries")


def extract_archive(archive_path: Path, dst_dir: Path) -> None:
    """Extract zip or tar.gz archive safely, guarding against path traversal."""
    dst_dir = dst_dir.resolve()

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                target = (dst_dir / member.filename).resolve()
                if not str(target).startswith(str(dst_dir) + "\\") and not str(target).startswith(str(dst_dir) + "/"):
                    print(f"Skipping unsafe archive member: {member.filename}")
                    continue
                zf.extract(member, dst_dir)
    elif archive_path.suffixes == [".tar", ".gz"]:
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                target = (dst_dir / member.name).resolve()
                if not str(target).startswith(str(dst_dir) + "\\") and not str(target).startswith(str(dst_dir) + "/"):
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
        print(f"Downloading NEU-DET from {FALLBACK_URL} ...")
        try:
            download_file(FALLBACK_URL, archive_path)
        except (requests.RequestException, OSError) as exc:  # pragma: no cover
            print(f"Primary download failed: {exc}")
            print("Please manually download NEU-DET and place it at data/raw/NEU-DET.zip")
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
