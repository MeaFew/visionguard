"""Compatibility wrapper for training YOLOv8n with augmented hyperparameters.

This script is kept for backward compatibility. It delegates to
``scripts/train_yolo.py --augment``.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Run ``train_yolo.py`` with the ``--augment`` flag enabled."""
    args = list(argv) if argv is not None else sys.argv[1:]

    # Default to a distinct experiment name unless the caller already supplied one.
    has_name = any(a == "--name" or a.startswith("--name=") for a in args)
    if not has_name:
        args.extend(["--name", "train_aug"])

    args.append("--augment")

    # Import and invoke the unified training script's main entry point.
    from scripts.train_yolo import main as train_main

    sys.argv = ["scripts/train_yolo.py", *args]
    train_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
