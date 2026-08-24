#!/usr/bin/env python3
"""Package the recovered L1 E4 Copy-Paste best checkpoint for Kaggle evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUN_NAME = "dianjienie_medium_yolo11m_cp_640_kaggle_seed0_oofconsensusv1_fold01"
DEFAULT_RECOVERY_ROOT = ROOT / "l1_e4_copypaste_v4_recovery_20260824_r2"
DEFAULT_SLUG = "dianjienie-ryw18-oof-e4-cp-f01-v4-best"
WEIGHT_NAME = "e4_copypaste_fold01_v4_best.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a private Kaggle Dataset package for the recovered E4 Copy-Paste best checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    parser.add_argument("--owner", default="lly3323w")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def read_training_summary(results_path: Path) -> dict[str, Any]:
    with results_path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("Recovered results.csv has no epoch rows")
    metric_key = "metrics/mAP50-95(B)"
    if metric_key not in rows[0]:
        raise ValueError(f"Recovered results.csv lacks {metric_key}")
    best = max(rows, key=lambda row: float(row[metric_key]))
    last = rows[-1]
    return {
        "recorded_epochs": len(rows),
        "best_training_epoch": int(float(best["epoch"])),
        "best_training_mAP50_95": float(best[metric_key]),
        "last_recorded_epoch": int(float(last["epoch"])),
        "last_training_mAP50_95": float(last[metric_key]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.slug or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in args.slug):
        raise ValueError("--slug must contain lowercase letters, digits, and hyphens only")
    recovery_root = args.recovery_root.resolve()
    run_root = recovery_root / RUN_NAME
    best = run_root / "weights" / "best.pt"
    args_yaml = run_root / "args.yaml"
    results = run_root / "results.csv"
    recovery_manifest_path = recovery_root / "recovery_manifest.json"
    for path in (best, args_yaml, results, recovery_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing recovered E4 evidence: {path}")
    recovery_manifest = json.loads(recovery_manifest_path.read_text(encoding="utf-8"))
    if recovery_manifest.get("run_name") != RUN_NAME:
        raise ValueError("Recovered run name does not match the fixed E4 Fold 1 run")
    if recovery_manifest.get("development_only") is not True or recovery_manifest.get("frozen_test_set") is not None:
        raise ValueError("Recovered input does not prove development-only frozen-test isolation")
    training_summary = read_training_summary(results)
    output_root = (args.output_root or ROOT / f"kaggle_{args.slug.replace('-', '_')}").resolve()
    checkpoint_manifest = {
        "schema_version": 1,
        "data_version": "oofconsensusv1",
        "development_only": True,
        "auto_only": True,
        "frozen_test_set": None,
        "fold": 1,
        "experiment": "L1 E4 Copy-Paste Fold 1 recovered best checkpoint",
        "source_kernel": recovery_manifest.get("kernel"),
        "source_kernel_status": recovery_manifest.get("kernel_status"),
        "source_run_name": RUN_NAME,
        "source_recovery_manifest_sha256": sha256_file(recovery_manifest_path),
        "training_summary": training_summary,
        "weights": {
            WEIGHT_NAME: {
                "source": str(best),
                "bytes": best.stat().st_size,
                "sha256": sha256_file(best),
            }
        },
        "purpose": (
            "Development-only original-image Fold 1 evaluation after the training completed "
            "but the original Kaggle notebook failed in its post-training evaluator. "
            "This is not a final release model."
        ),
    }
    metadata = {
        "title": "RYW18 OOF E4 CopyPaste Fold01 V4 Best",
        "id": f"{args.owner}/{args.slug}",
        "licenses": [{"name": "other"}],
    }
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "metadata": metadata, "manifest": checkpoint_manifest}, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Dataset package: {output_root}")
    output_root.mkdir(parents=True)
    shutil.copy2(best, output_root / WEIGHT_NAME)
    shutil.copy2(args_yaml, output_root / "training_args.yaml")
    shutil.copy2(results, output_root / "training_results.csv")
    shutil.copy2(recovery_manifest_path, output_root / "recovery_manifest.json")
    (output_root / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "# Recovered L1 E4 Copy-Paste checkpoint\n\n"
        "This private Dataset contains only the recovered best checkpoint and its training evidence. "
        "It contains no images, labels, source dataset, or frozen historical release-validation data.\n",
        encoding="utf-8",
    )
    print(f"[created] {output_root}")
    print(f"[push] kaggle datasets create -p {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
