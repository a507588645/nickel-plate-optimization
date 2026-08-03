#!/usr/bin/env python3
"""Run the registered YOLO26m Fold 3 baseline on the immutable RYW18 anchor."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath


INPUT_ROOT = Path("/kaggle/input")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mounted_dataset_roots() -> list[Path]:
    """Support both legacy flat and current owner-scoped Kaggle mount layouts."""
    roots = [path for path in INPUT_ROOT.iterdir() if path.is_dir()]
    scoped_root = INPUT_ROOT / "datasets"
    if scoped_root.is_dir():
        roots.extend(path for path in scoped_root.rglob("*") if path.is_dir())
    return roots


def input_inventory() -> dict[str, list[str]]:
    return {
        str(path.relative_to(INPUT_ROOT)): sorted(item.name for item in path.iterdir())[:12]
        for path in mounted_dataset_roots()
    }


def resolve_config_payload() -> tuple[Path, Path]:
    package_roots = [
        path for path in mounted_dataset_roots() if (path / "config_archive_manifest.json").is_file()
    ]
    if len(package_roots) != 1:
        raise RuntimeError(f"Cannot uniquely locate RYW18 configuration Dataset: {input_inventory()}")

    package_root = package_roots[0]
    manifest = json.loads((package_root / "config_archive_manifest.json").read_text(encoding="utf-8"))
    archive_name = manifest.get("config_archive")
    expected_files = manifest.get("payload_files")
    expected_config_hash = manifest.get("source_manifest_sha256")
    if not isinstance(archive_name, str) or not isinstance(expected_files, list) or not isinstance(expected_config_hash, str):
        raise ValueError("Invalid RYW18 configuration archive manifest")

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
    print(f"[train] package={package_root}, payload={payload_root}", flush=True)
    return payload_root, runner


def resolve_image_dataset_root(payload_root: Path) -> Path:
    config_manifest = json.loads((payload_root / "config_manifest.json").read_text(encoding="utf-8"))
    images = config_manifest.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], dict):
        raise ValueError("RYW18 config manifest has no source image probe")
    source_relative = images[0].get("source_relative")
    if not isinstance(source_relative, str):
        raise ValueError("RYW18 config manifest source image probe is invalid")
    probe_parts = PurePosixPath(source_relative).parts
    candidates = [
        root for root in mounted_dataset_roots()
        if root.is_dir() and root != payload_root.parent and root.joinpath(*probe_parts).is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cannot uniquely locate source image Dataset for {source_relative}: "
            f"matches={[str(path.relative_to(INPUT_ROOT)) for path in candidates]}, inputs={input_inventory()}"
        )
    print(f"[train] image_dataset={candidates[0]}", flush=True)
    return candidates[0]


def main() -> None:
    payload_root, runner = resolve_config_payload()
    image_dataset_root = resolve_image_dataset_root(payload_root)
    command = [
        sys.executable,
        str(runner),
        "--config-root",
        str(payload_root),
        "--model",
        "yolo26m.pt",
        "--fold",
        "3",
        "--image-dataset-root",
        str(image_dataset_root),
        "--epochs",
        "300",
        "--imgsz",
        "640",
        "--batch",
        "8",
        "--workers",
        "8",
        "--device",
        "0",
        "--seed",
        "0",
        "--patience",
        "100",
    ]
    print("[train]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
