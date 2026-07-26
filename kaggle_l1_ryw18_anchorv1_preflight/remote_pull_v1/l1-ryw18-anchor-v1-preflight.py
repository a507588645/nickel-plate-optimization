#!/usr/bin/env python3
"""Validate the immutable RYW18 anchor on Kaggle without requesting a GPU."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CONFIG_DATASET_ROOT = Path("/kaggle/input/dianjienie-ryw18-anchor-v1-config")
IMAGE_DATASET_ROOT = Path("/kaggle/input/dianjienie")
BOOTSTRAP = CONFIG_DATASET_ROOT / "kaggle_dianjienie_ryw18_archive_bootstrap.py"


def main() -> None:
    if not BOOTSTRAP.is_file():
        raise FileNotFoundError(f"Missing RYW18 bootstrap: {BOOTSTRAP}")
    if not IMAGE_DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Missing source image Dataset: {IMAGE_DATASET_ROOT}")

    command = [
        sys.executable,
        str(BOOTSTRAP),
        "--model",
        "yolo11m.pt",
        "--fold",
        "1",
        "--image-dataset-root",
        str(IMAGE_DATASET_ROOT),
        "--dry-run",
    ]
    print("[preflight]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
