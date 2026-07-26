#!/usr/bin/env python3
"""Validate the immutable RYW18 anchor on Kaggle without requesting a GPU."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
IMAGE_DATASET_ROOT = Path("/kaggle/input/dianjienie")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_config_payload() -> tuple[Path, Path]:
    package_roots = [
        path for path in INPUT_ROOT.iterdir() if (path / "config_archive_manifest.json").is_file()
    ]
    if len(package_roots) != 1:
        inventory = {path.name: sorted(item.name for item in path.iterdir()) for path in INPUT_ROOT.iterdir()}
        raise RuntimeError(f"Cannot uniquely locate RYW18 configuration Dataset: {inventory}")

    package_root = package_roots[0]
    manifest = json.loads((package_root / "config_archive_manifest.json").read_text(encoding="utf-8"))
    archive_name = manifest.get("config_archive")
    expected_files = manifest.get("payload_files")
    expected_config_hash = manifest.get("source_manifest_sha256")
    if not isinstance(archive_name, str) or not isinstance(expected_files, list) or not isinstance(expected_config_hash, str):
        raise ValueError("Invalid RYW18 configuration archive manifest")

    # Kaggle expands the uploaded zip into this directory instead of exposing
    # the original zip.  The runtime runner can use the verified payload directly.
    payload_root = package_root / Path(archive_name).stem
    actual_files = {path.name for path in payload_root.iterdir() if path.is_file()} if payload_root.is_dir() else set()
    if actual_files != set(expected_files):
        raise ValueError(
            f"Expanded RYW18 payload mismatch: expected={len(expected_files)}, actual={len(actual_files)}"
        )
    config_manifest = payload_root / "config_manifest.json"
    if sha256_file(config_manifest) != expected_config_hash:
        raise ValueError("Expanded RYW18 config_manifest hash mismatch")
    runner = payload_root / "runner__kaggle_dianjienie_ryw18_existing_data_runner.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Missing RYW18 runtime runner: {runner}")
    print(f"[preflight] package={package_root}, payload={payload_root}", flush=True)
    return payload_root, runner


def main() -> None:
    if not IMAGE_DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Missing source image Dataset: {IMAGE_DATASET_ROOT}")
    payload_root, runner = resolve_config_payload()

    command = [
        sys.executable,
        str(runner),
        "--config-root",
        str(payload_root),
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
