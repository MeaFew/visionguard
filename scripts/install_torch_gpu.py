"""Install PyTorch with CUDA support using pip's official index.

This script uses `pip install --index-url` to fetch the correct wheels
for the current platform and Python version automatically, avoiding
hard-coded wheel URLs.
"""

from __future__ import annotations

import logging
import subprocess
import sys

from visionguard.logging_setup import setup_logging

logger = logging.getLogger(__name__)

TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
PACKAGES = ["torch", "torchvision"]


def main() -> None:
    """Install PyTorch GPU packages via pip with the official CUDA index."""
    logger.info(
        "Installing PyTorch GPU (platform=%s, python=%s) ...",
        sys.platform,
        sys.version.split()[0],
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--index-url",
            TORCH_INDEX_URL,
            *PACKAGES,
        ],
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
