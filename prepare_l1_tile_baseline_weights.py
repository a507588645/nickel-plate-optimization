#!/usr/bin/env python3
"""Package the fixed L1 Fold 1 baseline weight as a private Kaggle Dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(
    r"E:\PythonProject\models\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold01\weights\best.pt"
)
DEFAULT_SLUG = "dianjienie-ryw18-oof-yolo11m-fold01-640-weight"
WEIGHT_NAME = "baseline_fold01_640_best.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a private Kaggle Dataset package containing only the fixed L1 Fold 1 baseline weight.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--owner", default="lly3323w")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing baseline source weight: {source}")
    if not args.slug or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in args.slug):
        raise ValueError("--slug must contain lowercase letters, digits, and hyphens only")
    output_root = (args.output_root or ROOT / f"kaggle_{args.slug.replace('-', '_')}").resolve()
    metadata = {
        "title": "RYW18 OOF YOLO11m Fold01 640 Weight",
        "id": f"{args.owner}/{args.slug}",
        "licenses": [{"name": "other"}],
    }
    manifest = {
        "schema_version": 1,
        "data_version": "oofconsensusv1",
        "development_only": True,
        "fold": 1,
        "model": "YOLO11m",
        "imgsz": 640,
        "source_weight": str(source),
        "source_weight_sha256": sha256_file(source),
        "purpose": "Fixed Fold 1 baseline for the original-image tiled evaluation protocol; not a final release model.",
    }
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "metadata": metadata, "manifest": manifest}, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Dataset package: {output_root}")
    output_root.mkdir(parents=True)
    shutil.copy2(source, output_root / WEIGHT_NAME)
    (output_root / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "baseline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "# L1 Fold 1 640 baseline\n\n"
        "Private comparison input for the L1 tile experiment. It contains the archived OOF-consensus Fold 1 YOLO11m best weight only.\n",
        encoding="utf-8",
    )
    print(f"[created] {output_root}")
    print(f"[push] kaggle datasets create -p {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
