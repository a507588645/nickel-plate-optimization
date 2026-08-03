#!/usr/bin/env python3
"""Validate machine-local dataset mappings without reading image contents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from workspace_config import ROOT, load_local_paths


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local logical dataset mappings.")
    parser.add_argument("--catalog", type=Path, default=ROOT / "config" / "dataset_catalog.json")
    parser.add_argument("--config", type=Path, help="Defaults to NICKEL_WORKSPACE_CONFIG or config/workspace.local.json.")
    parser.add_argument("--dataset", action="append", help="Logical dataset ID; repeatable. Defaults to all catalog entries.")
    return parser.parse_args()


def count_files(root: Path, suffixes: set[str]) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def main() -> int:
    args = parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("datasets"), dict):
        raise ValueError(f"unsupported dataset catalog: {args.catalog}")
    mappings = load_local_paths(args.config)
    requested = args.dataset or list(catalog["datasets"])
    failures: list[str] = []

    for logical_id in requested:
        spec = catalog["datasets"].get(logical_id)
        if not isinstance(spec, dict):
            failures.append(f"unknown catalog ID: {logical_id}")
            continue
        root = mappings.get(logical_id)
        if root is None:
            failures.append(f"{logical_id}: missing local path mapping")
            continue
        if not root.is_dir():
            failures.append(f"{logical_id}: directory does not exist: {root}")
            continue
        missing = [item for item in spec.get("required_paths", []) if not (root / item).exists()]
        if missing:
            failures.append(f"{logical_id}: missing required paths: {', '.join(missing)}")
            continue
        counts = spec.get("expected_counts", {})
        actual_images = count_files(root / "images", IMAGE_SUFFIXES)
        actual_labels = count_files(root / "labels", {".txt"})
        if actual_images != counts.get("images") or actual_labels != counts.get("labels"):
            failures.append(
                f"{logical_id}: expected images/labels={counts.get('images')}/{counts.get('labels')}, "
                f"actual={actual_images}/{actual_labels}"
            )
            continue
        print(f"[ok] {logical_id}: {root} ({actual_images} images, {actual_labels} labels)")

    if failures:
        for item in failures:
            print(f"[fail] {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
