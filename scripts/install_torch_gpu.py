"""Download PyTorch CUDA wheels with a progress bar and install them locally."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import requests

from visionguard.logging_setup import setup_logging

logger = logging.getLogger(__name__)

WHEELS = [
    (
        "torch",
        "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl",
    ),
    (
        "torchvision",
        "https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp311-cp311-win_amd64.whl",
    ),
]

CACHE_DIR = Path(".cache") / "torch_wheels"
CHUNK_SIZE = 1024 * 1024  # 1 MB


def _draw_bar(percent: float, width: int = 40) -> str:
    filled = int(round(width * percent / 100.0))
    bar = "=" * filled + ">" + " " * (width - filled - 1)
    return f"[{bar}] {percent:5.1f}%"


def download(url: str, dst: Path) -> None:
    """Download a file with a simple progress bar."""
    if dst.exists():
        local_size = dst.stat().st_size
        head = requests.head(url, timeout=30)
        remote_size = int(head.headers.get("content-length", 0))
        if local_size == remote_size and remote_size > 0:
            logger.info(f"Using cached {dst.name}")
            return

    logger.info(f"Downloading {dst.name} ...")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dst, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = downloaded / total * 100
                    bar = _draw_bar(percent)
                    print(
                        f"\r{bar}  {downloaded / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB",
                        end="",
                        flush=True,
                    )
        print()  # newline after completion


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    wheel_paths: list[Path] = []
    for _, url in WHEELS:
        filename = url.split("/")[-1]
        dst = CACHE_DIR / filename
        download(url, dst)
        wheel_paths.append(dst)

    logger.info("Installing wheels ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps"]
        + [str(p) for p in wheel_paths],
        check=True,
    )
    logger.info("PyTorch GPU installation complete.")
    logger.info("Verifying CUDA availability:")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())",
        ]
    )


if __name__ == "__main__":
    setup_logging()
    main()
