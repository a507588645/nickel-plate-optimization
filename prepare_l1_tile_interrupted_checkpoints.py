#!/usr/bin/env python3
"""Package recovered L1 tile checkpoints as a private Kaggle Dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = ROOT / "kaggle_l1_ryw18_y11m_oof_v1_tile1024_o25_b8_e300_f01_r3" / "recovered_interrupted_v3"
DEFAULT_SLUG = "dianjienie-ryw18-oof-y11m-tile-f01-v3-ckpt"
BEST_NAME = "tile_fold01_v3_best.pt"
LAST_NAME = "tile_fold01_v3_last.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a private Kaggle Dataset package for recovered L1 tile checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--owner", default="lly3323w")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = args.source_dir.resolve()
    best = source_dir / "best.pt"
    last = source_dir / "last.pt"
    recovery_manifest = source_dir / "recovery_manifest.json"
    for path in (best, last, recovery_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Missing recovered checkpoint evidence: {path}")
    if not args.slug or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in args.slug):
        raise ValueError("--slug must contain lowercase letters, digits, and hyphens only")
    output_root = (args.output_root or ROOT / f"kaggle_{args.slug.replace('-', '_')}").resolve()
    recovered = json.loads(recovery_manifest.read_text(encoding="utf-8"))
    checkpoint_manifest = {
        "schema_version": 1,
        "data_version": "oofconsensusv1",
        "development_only": True,
        "fold": 1,
        "experiment": "tile1024o25_focus6_mix",
        "source_kernel": recovered.get("kernel"),
        "source_status": recovered.get("status_at_recovery"),
        "last_completed_epoch": recovered.get("last_completed_epoch"),
        "best_training_epoch": recovered.get("best_training_epoch"),
        "best_training_metrics": recovered.get("best_training_metrics"),
        "weights": {
            BEST_NAME: {"source": str(best), "sha256": sha256_file(best)},
            LAST_NAME: {"source": str(last), "sha256": sha256_file(last)},
        },
        "purpose": "Evaluation of recovered interrupted tile-training checkpoint; not a final release model.",
    }
    metadata = {
        "title": "RYW18 OOF Tile Fold01 V3 Checkpoints",
        "id": f"{args.owner}/{args.slug}",
        "licenses": [{"name": "other"}],
    }
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "metadata": metadata, "manifest": checkpoint_manifest}, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Dataset package: {output_root}")
    output_root.mkdir(parents=True)
    shutil.copy2(best, output_root / BEST_NAME)
    shutil.copy2(last, output_root / LAST_NAME)
    (output_root / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "# Recovered L1 tile checkpoints\n\n"
        "Private inputs for the development-only original-image tile evaluation. "
        "These checkpoints came from an interrupted Kaggle training run and are not final release weights.\n",
        encoding="utf-8",
    )
    print(f"[created] {output_root}")
    print(f"[push] kaggle datasets create -p {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
