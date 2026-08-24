#!/usr/bin/env python3
"""Restore a development-only flattened L1 config package into a new local view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore a verified flattened OOF-consensus development config without changing its source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> Path:
    candidate = PurePosixPath(value)
    if (
        "\\" in value
        or candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"Unsafe config relative path: {value!r}")
    return Path(*candidate.parts)


def read_manifest(source_root: Path) -> tuple[Path, dict[str, Any], list[dict[str, str]]]:
    manifest_path = source_root / "config_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing flattened config manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("data_version") != "oofconsensusv1" or manifest.get("development_only") is not True:
        raise ValueError("Only the development-only OOF-consensus config package is permitted")
    if manifest.get("classes") is None or len(manifest["classes"]) != 18:
        raise ValueError("The config package must declare exactly 18 fixed RYW classes")
    records = manifest.get("labels", []) + manifest.get("metadata", [])
    if not isinstance(records, list) or len(records) != 1139:
        raise ValueError("The config package must contain 1127 labels and 12 metadata records")
    normalized: list[dict[str, str]] = []
    destinations: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Config record is not an object")
        upload_name = item.get("upload_name")
        relative_path = item.get("relative_path")
        if not isinstance(upload_name, str) or Path(upload_name).name != upload_name:
            raise ValueError(f"Invalid config upload name: {upload_name!r}")
        if not isinstance(relative_path, str):
            raise ValueError("Config record is missing a relative path")
        relative = safe_relative(relative_path)
        key = relative.as_posix()
        if key in destinations:
            raise ValueError(f"Duplicate config destination: {key}")
        destinations.add(key)
        source = source_root / upload_name
        if not source.is_file():
            raise FileNotFoundError(f"Flattened config payload is missing: {source}")
        normalized.append({"upload_name": upload_name, "relative_path": key})
    return manifest_path, manifest, normalized


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path, manifest, records = read_manifest(source_root)
    summary = {
        "schema_version": 1,
        "source_root": str(source_root),
        "source_manifest_sha256": sha256_file(manifest_path),
        "development_only": True,
        "frozen_test_set": None,
        "labels": len(manifest["labels"]),
        "metadata": len(manifest["metadata"]),
        "output_root": str(output_root),
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output view: {output_root}")
    output_root.mkdir(parents=True)
    try:
        shutil.copy2(manifest_path, output_root / "config_manifest.json")
        hardlinks = 0
        for record in records:
            source = source_root / record["upload_name"]
            target = output_root / Path(*PurePosixPath(record["relative_path"]).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
            hardlinks += 1
        expected = {
            output_root / "labels",
            output_root / "splits" / "folds" / "fold_01" / "train.txt",
            output_root / "splits" / "folds" / "fold_01" / "val.txt",
        }
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Restored config is incomplete: {missing}")
        if (output_root / "splits" / "folds" / "fold_01" / "test.txt").exists():
            raise ValueError("Development-only runtime unexpectedly contains a Fold 1 test list")
        summary["hardlinks_created"] = hardlinks
        (output_root / "restore_manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        # Never remove a partial directory automatically: it is evidence for diagnosis.
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
