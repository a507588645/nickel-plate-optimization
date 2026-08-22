"""Derive a Fold-1 threshold profile from Fold-2/3 OOF predictions only.

Phase 0 first scans all three archived development folds to describe the
dataset.  This companion tool creates the threshold table used by the Phase 1
Fold-1 tiled evaluation without selecting it on Fold 1 itself.  It reads only
the versioned raw prediction JSONL files written by ``run_l1_phase0_diagnostics``;
no model inference, source image, or frozen release-validation data is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from l1_tile_pipeline import CLASS_NAMES
from run_l1_phase0_diagnostics import DEFAULT_THRESHOLDS, scan_thresholds, select_recommendations, write_csv


DEFAULT_INPUT_ROOT = Path("l1_phase0_diagnostics_20260822_1455")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a held-out-fold per-class confidence profile from Phase 0 raw OOF predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--held-out-fold", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--default-threshold", type=float, default=0.25)
    parser.add_argument("--minimum-support", type=int, default=100)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to <input-root>/foldXX_holdout_thresholds; refuses to overwrite an existing directory.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rows(values: list[list[float]], columns: int) -> np.ndarray:
    if not values:
        return np.empty((0, columns), dtype=np.float64)
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != columns:
        raise ValueError(f"Expected rows with shape (*, {columns}), got {rows.shape}")
    return rows


def load_fold_jsonl(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    predictions: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    prediction_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            image_id = str(payload["image_id"])
            if image_id in predictions:
                raise ValueError(f"Duplicate image_id in {path} line {line_number}: {image_id}")
            prediction_rows = normalize_rows(payload["predictions_xyxy_conf_class"], 6)
            target_rows = normalize_rows(payload["targets_class_xyxy"], 5)
            predictions[image_id] = prediction_rows
            targets[image_id] = target_rows
            prediction_count += len(prediction_rows)
    if not predictions:
        raise ValueError(f"No prediction rows found in {path}")
    return predictions, targets, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "images": len(predictions),
        "predictions_after_model_nms": prediction_count,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_profile(
    recommendations: list[dict[str, Any]], default_threshold: float, minimum_support: int
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    overrides: dict[str, float] = {}
    audit_rows: list[dict[str, Any]] = []
    for row in recommendations:
        class_name = str(row["class_name"])
        support = int(row["targets"])
        recommended = row.get("recommended_threshold")
        eligible = row["status"] == "evaluated" and support >= minimum_support and recommended is not None
        selected = float(recommended) if eligible else default_threshold
        if eligible and abs(selected - default_threshold) > 1e-12:
            overrides[class_name] = selected
        audit_rows.append(
            {
                **row,
                "minimum_support": minimum_support,
                "phase1_threshold": selected,
                "phase1_selection": "crossfold_override" if class_name in overrides else "default_025",
                "phase1_selection_reason": (
                    "max_f1_on_nonheldout_oof_folds_with_support_at_least_minimum"
                    if class_name in overrides
                    else "insufficient_support_or_no_change_from_default"
                ),
            }
        )
    return overrides, audit_rows


def render_report(
    held_out_fold: int,
    included_folds: list[int],
    source_audit: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    overrides: dict[str, float],
    default_threshold: float,
    minimum_support: int,
) -> str:
    lines = [
        "# Phase 1 Fold-specific confidence profile",
        "",
        f"- Held-out evaluation fold: Fold {held_out_fold}",
        f"- Calibration folds: {', '.join(f'Fold {fold}' for fold in included_folds)}",
        "- Scope: archived OOF development predictions only; `release-validation` was not read.",
        "- This is a research-side post-processing profile, not a deployed threshold or a model-selection result.",
        "",
        "## Source files",
        "",
    ]
    for source in source_audit:
        lines.append(
            f"- `{Path(source['path']).name}`: {source['images']} images, "
            f"{source['predictions_after_model_nms']} model-NMS predictions, SHA-256 `{source['sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Phase 1 profile",
            "",
            f"- Default confidence: `{default_threshold:.2f}`",
            f"- Minimum non-held-out target support for an override: `{minimum_support}`",
            f"- Overrides: `{json.dumps(overrides, ensure_ascii=False, sort_keys=True)}`",
            "",
            "## Selection audit",
            "",
            "| 类别 | 非 Fold 1 支持数 | 建议阈值 | Phase 1 阈值 | 选择 | F1@建议 |",
            "| --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in audit_rows:
        recommended = row.get("recommended_threshold")
        recommended_text = "n/a" if recommended is None else f"{float(recommended):.2f}"
        f1 = row.get("recommended_f1")
        f1_text = "n/a" if f1 is None else f"{float(f1):.4f}"
        lines.append(
            f"| {row['class_name']} | {row['targets']} | {recommended_text} | "
            f"{float(row['phase1_threshold']):.2f} | {row['phase1_selection']} | {f1_text} |"
        )
    lines.extend(
        [
            "",
            "The table deliberately excludes Fold 1 from threshold selection. "
            "Classes below the support threshold remain at the fixed default instead of inheriting noisy fold-specific optima.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.default_threshold <= 1:
        raise ValueError("--default-threshold must be in (0, 1]")
    if args.minimum_support < 1:
        raise ValueError("--minimum-support must be positive")
    thresholds = sorted({float(value) for value in args.thresholds})
    if args.default_threshold not in thresholds or any(not 0 < value <= 1 for value in thresholds):
        raise ValueError("--thresholds must include --default-threshold and stay within (0, 1]")

    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Phase 0 input root not found: {input_root}")
    included_folds = [fold for fold in (1, 2, 3) if fold != args.held_out_fold]
    raw_paths = [input_root / "raw_predictions" / f"fold{fold:02d}.jsonl" for fold in included_folds]
    for raw_path in raw_paths:
        if not raw_path.is_file():
            raise FileNotFoundError(f"Required Phase 0 raw prediction file not found: {raw_path}")
    output_root = (args.output_root or input_root / f"fold{args.held_out_fold:02d}_holdout_thresholds").resolve()
    plan = {
        "held_out_fold": args.held_out_fold,
        "included_folds": included_folds,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "default_threshold": args.default_threshold,
        "minimum_support": args.minimum_support,
        "thresholds": thresholds,
        "development_only": True,
        "frozen_test_set": None,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output_root}")

    pooled_predictions: dict[str, np.ndarray] = {}
    pooled_targets: dict[str, np.ndarray] = {}
    source_audit: list[dict[str, Any]] = []
    for fold, raw_path in zip(included_folds, raw_paths, strict=True):
        predictions, targets, audit = load_fold_jsonl(raw_path)
        audit["fold"] = fold
        duplicate_ids = set(pooled_predictions) & set(predictions)
        if duplicate_ids:
            raise ValueError(f"Cross-fold image_id collision: {sorted(duplicate_ids)[:3]}")
        pooled_predictions.update(predictions)
        pooled_targets.update(targets)
        source_audit.append(audit)

    curves = scan_thresholds(pooled_predictions, pooled_targets, thresholds, f"folds_{'_'.join(map(str, included_folds))}_for_fold{args.held_out_fold:02d}")
    recommendations = select_recommendations(curves)
    overrides, audit_rows = build_profile(recommendations, args.default_threshold, args.minimum_support)
    output_root.mkdir(parents=True)
    write_csv(output_root / "per_class_threshold_curves.csv", curves)
    write_csv(output_root / "selection_audit.csv", audit_rows)
    config = {
        "schema_version": 1,
        "purpose": "Phase 1 development-only tiled evaluation post-processing",
        "held_out_evaluation_fold": args.held_out_fold,
        "calibration_folds": included_folds,
        "selection_objective": "max_f1_at_iou50_then_higher_threshold",
        "default_confidence": args.default_threshold,
        "minimum_support": args.minimum_support,
        "class_thresholds": overrides,
        "class_names": list(CLASS_NAMES),
        "development_only": True,
        "frozen_test_set": None,
        "source_raw_predictions": source_audit,
        "limitations": [
            "Cross-fold calibration uses distinct model weights and is intended only to avoid selecting thresholds on Fold 1.",
            "This profile does not establish a final deployment threshold or manual-label performance.",
        ],
    }
    write_json(output_root / "phase1_per_class_thresholds.json", config)
    write_json(output_root / "selection_audit.json", audit_rows)
    (output_root / "report.md").write_text(
        render_report(
            args.held_out_fold,
            included_folds,
            source_audit,
            audit_rows,
            overrides,
            args.default_threshold,
            args.minimum_support,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "phase1_per_class_thresholds": str(output_root / "phase1_per_class_thresholds.json"),
                "overrides": overrides,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
