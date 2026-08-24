"""Scan L1 OOF confidence overrides without loading models or new images.

E1b deliberately consumes only persisted development-fold prediction dumps.
It evaluates one focus-class override at a time against an all-class 0.25
operating control.  This isolates the decision-layer variable and avoids
presenting an untested six-class joint profile as a result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from l1_tile_pipeline import CLASS_NAMES, IOU_THRESHOLDS, average_precision, box_iou_one_to_many
from run_l1_oof_dualmodel_wbf import RawRecord, load_phase0_fold, parse_phase0_record, score_with_f1, sha256_file


ROOT = Path(__file__).resolve().parent
DEFAULT_PHASE0_ROOT = ROOT / "l1_phase0_diagnostics_20260822_1455"
DEFAULT_E1_ROOT = ROOT / "l1_oof_dualmodel_wbf_20260822_1830"
DEFAULT_OUTPUT_ROOT = ROOT / "l1_oof_perclass_conf_scan_20260823_r3"
FOCUS_CLASSES = ("wuzi", "waterstain", "airhole", "bubble", "greenstain", "watermark")
DEFAULT_THRESHOLDS = tuple(round(value / 100.0, 2) for value in range(5, 51, 5))
FIXED_RAW_ANCHORS = {
    "yolo26m": 0.148939031666318,
    "wbf_iou_0.55": 0.16751980771806158,
}
EXPECTED_COUNTS = {
    "phase0_yolo11m": 149826,
    "yolo26m": 100292,
    "wbf_iou_0.55": 169187,
}


@dataclass(frozen=True)
class ClassTrace:
    """Confidence-sorted candidates and their greedy TP state at each IoU."""

    class_id: int
    class_name: str
    support: int
    confidences: np.ndarray
    true_positive_by_iou: dict[float, np.ndarray]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E1b CPU-only scan of persisted L1 OOF per-class confidence overrides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase0-root", type=Path, default=DEFAULT_PHASE0_ROOT)
    parser.add_argument("--e1-root", type=Path, default=DEFAULT_E1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-threshold", type=float, default=0.25)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--raw-conf-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--no-plots", action="store_true", help="Skip optional matplotlib PR curve images.")
    parser.add_argument("--dry-run", action="store_true", help="Audit source dumps and leakage boundary without writing output.")
    return parser.parse_args(argv)


def assert_development_path(path: Path, purpose: str) -> Path:
    resolved = path.resolve()
    if "release-validation" in resolved.as_posix().lower():
        raise ValueError(f"Forbidden release-validation path for {purpose}: {resolved}")
    return resolved


def assert_project_output(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Output must stay under the project root {ROOT}: {resolved}") from error
    if resolved == ROOT.resolve():
        raise ValueError("Output root cannot be the project root itself")
    return resolved


def fold_from_image_id(image_id: object) -> int:
    if not isinstance(image_id, str):
        raise ValueError("JSONL image_id must be a string")
    match = re.fullmatch(r"fold(0[1-3])/.+", image_id)
    if match is None:
        raise ValueError(f"Unexpected OOF image_id: {image_id!r}")
    return int(match.group(1))


def load_jsonl_records(path: Path) -> tuple[dict[str, RawRecord], dict[str, Any]]:
    path = assert_development_path(path, "persisted prediction dump")
    if not path.is_file():
        raise FileNotFoundError(f"Missing persisted prediction dump: {path}")
    records: dict[str, RawRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"Blank JSONL line: {path}:{line_number}")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON: {path}:{line_number}") from error
            fold = fold_from_image_id(payload.get("image_id"))
            record = parse_phase0_record(payload, fold)
            if record.image_id in records:
                raise ValueError(f"Duplicate image_id in {path}: {record.image_id}")
            records[record.image_id] = record
    if not records:
        raise ValueError(f"Empty prediction dump: {path}")
    return records, {
        "path": str(path),
        "sha256": sha256_file(path),
        "images": len(records),
        "predictions_after_model_nms": int(sum(len(record.predictions) for record in records.values())),
        "targets": int(sum(len(record.targets) for record in records.values())),
    }


def load_phase0_records(root: Path) -> tuple[dict[str, RawRecord], list[dict[str, Any]]]:
    root = assert_development_path(root, "Phase 0 root")
    if not root.is_dir():
        raise FileNotFoundError(f"Missing Phase 0 root: {root}")
    records: dict[str, RawRecord] = {}
    audits: list[dict[str, Any]] = []
    for fold in (1, 2, 3):
        fold_records, audit = load_phase0_fold(root, fold)
        for record in fold_records:
            if record.image_id in records:
                raise ValueError(f"Duplicate Phase 0 image across folds: {record.image_id}")
            records[record.image_id] = record
        audits.append({"fold": fold, **audit})
    if len(records) != 1127:
        raise ValueError(f"Unexpected Phase 0 OOF image count: {len(records)}")
    return records, audits


def load_yolo26_records(e1_root: Path) -> tuple[dict[str, RawRecord], list[dict[str, Any]]]:
    e1_root = assert_development_path(e1_root, "E1 root")
    records: dict[str, RawRecord] = {}
    audits: list[dict[str, Any]] = []
    for fold in (1, 2, 3):
        path = e1_root / "raw_predictions_yolo26" / f"fold{fold:02d}.jsonl"
        fold_records, audit = load_jsonl_records(path)
        expected_prefix = f"fold{fold:02d}/"
        if any(not image_id.startswith(expected_prefix) for image_id in fold_records):
            raise ValueError(f"YOLO26m dump contains an unexpected fold member: {path}")
        overlap = set(records).intersection(fold_records)
        if overlap:
            raise ValueError(f"Duplicate YOLO26m image across fold dumps: {sorted(overlap)[:3]}")
        records.update(fold_records)
        audits.append({"fold": fold, **audit})
    if len(records) != 1127:
        raise ValueError(f"Unexpected YOLO26m OOF image count: {len(records)}")
    return records, audits


def validate_record_alignment(
    reference: dict[str, RawRecord], candidate: dict[str, RawRecord], candidate_name: str
) -> None:
    if set(reference) != set(candidate):
        only_reference = sorted(set(reference) - set(candidate))
        only_candidate = sorted(set(candidate) - set(reference))
        raise ValueError(
            f"{candidate_name} membership differs from Phase 0; "
            f"only_phase0={only_reference[:3]}, only_candidate={only_candidate[:3]}"
        )
    for image_id in sorted(reference):
        left = reference[image_id]
        right = candidate[image_id]
        if (left.fold, left.width, left.height) != (right.fold, right.width, right.height):
            raise ValueError(f"{candidate_name} image metadata differs: {image_id}")
        if left.targets.shape != right.targets.shape or not np.allclose(left.targets, right.targets, rtol=0, atol=1e-6):
            raise ValueError(f"{candidate_name} target differs from Phase 0: {image_id}")


def build_class_traces(
    predictions_by_image: dict[str, np.ndarray], targets_by_image: dict[str, np.ndarray]
) -> dict[str, ClassTrace]:
    traces: dict[str, ClassTrace] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        ground_truth = {
            image_id: targets_by_image[image_id][targets_by_image[image_id][:, 0] == class_id, 1:]
            for image_id in sorted(targets_by_image)
        }
        candidates: list[tuple[float, str, np.ndarray]] = []
        for image_id in sorted(predictions_by_image):
            rows = predictions_by_image[image_id]
            selected = rows[rows[:, 5] == class_id]
            candidates.extend((float(row[4]), image_id, row[:4]) for row in selected)
        candidates.sort(key=lambda item: item[0], reverse=True)
        confidences = np.asarray([item[0] for item in candidates], dtype=np.float64)
        true_positive_by_iou: dict[float, np.ndarray] = {}
        for iou_threshold in IOU_THRESHOLDS:
            matched: dict[str, set[int]] = defaultdict(set)
            true_positive: list[float] = []
            for _, image_id, box in candidates:
                boxes = ground_truth[image_id]
                overlaps = box_iou_one_to_many(box, boxes)
                best_index = int(np.argmax(overlaps)) if len(overlaps) else -1
                is_true_positive = (
                    best_index >= 0
                    and overlaps[best_index] >= iou_threshold
                    and best_index not in matched[image_id]
                )
                if is_true_positive:
                    matched[image_id].add(best_index)
                true_positive.append(1.0 if is_true_positive else 0.0)
            true_positive_by_iou[float(iou_threshold)] = np.asarray(true_positive, dtype=np.float64)
        traces[class_name] = ClassTrace(
            class_id=class_id,
            class_name=class_name,
            support=int(sum(len(boxes) for boxes in ground_truth.values())),
            confidences=confidences,
            true_positive_by_iou=true_positive_by_iou,
        )
    return traces


def count_above_threshold(confidences: np.ndarray, threshold: float) -> int:
    if len(confidences) == 0:
        return 0
    return int(np.count_nonzero(confidences >= threshold))


def class_metrics(trace: ClassTrace, threshold: float) -> dict[str, Any]:
    prediction_count = count_above_threshold(trace.confidences, threshold)
    ap_by_iou: list[float] = []
    for iou_threshold in IOU_THRESHOLDS:
        true_positive = trace.true_positive_by_iou[float(iou_threshold)][:prediction_count]
        confidence = trace.confidences[:prediction_count]
        ap_by_iou.append(average_precision(true_positive, confidence, trace.support))
    true_positive_050 = float(trace.true_positive_by_iou[0.5][:prediction_count].sum())
    precision = true_positive_050 / prediction_count if prediction_count else 0.0
    recall = true_positive_050 / trace.support if trace.support else 0.0
    return {
        "class_id": trace.class_id,
        "class_name": trace.class_name,
        "threshold": float(threshold),
        "support": trace.support,
        "predictions": prediction_count,
        "true_positives": int(true_positive_050),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "ap50": ap_by_iou[0] if trace.support else None,
        "ap50_95": float(np.mean(ap_by_iou)) if trace.support else None,
    }


def pooled_metrics(metrics_by_class: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evaluable = [metric for metric in metrics_by_class.values() if int(metric["support"]) > 0]
    total_tp = sum(int(metric["true_positives"]) for metric in metrics_by_class.values())
    total_predictions = sum(int(metric["predictions"]) for metric in metrics_by_class.values())
    total_targets = sum(int(metric["support"]) for metric in metrics_by_class.values())
    precision = total_tp / total_predictions if total_predictions else 0.0
    recall = total_tp / total_targets if total_targets else 0.0
    return {
        "images": 1127,
        "classes_with_support": len(evaluable),
        "post_threshold_prediction_count": total_predictions,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mAP50": float(np.mean([float(metric["ap50"]) for metric in evaluable])) if evaluable else 0.0,
        "mAP50_95": float(np.mean([float(metric["ap50_95"]) for metric in evaluable])) if evaluable else 0.0,
    }


def canonical_raw_score(
    predictions_by_image: dict[str, np.ndarray], targets_by_image: dict[str, np.ndarray], nms_iou: float
) -> dict[str, Any]:
    return score_with_f1(predictions_by_image, targets_by_image, nms_iou)


def build_candidate_rows(
    variant: str,
    traces: dict[str, ClassTrace],
    thresholds: tuple[float, ...],
    base_threshold: float,
    raw_anchor: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    baseline_by_class = {class_name: class_metrics(trace, base_threshold) for class_name, trace in traces.items()}
    baseline_pooled = pooled_metrics(baseline_by_class)
    baseline = {
        "variant": variant,
        "base_threshold": base_threshold,
        "per_class": baseline_by_class,
        "pooled": baseline_pooled,
        "fixed_raw_anchor_mAP50_95": float(raw_anchor["mAP50_95"]),
        "thresholded_control_delta_vs_fixed_raw_anchor": (
            float(baseline_pooled["mAP50_95"]) - float(raw_anchor["mAP50_95"])
        ),
    }
    rows: list[dict[str, Any]] = []
    candidate_profiles: dict[str, dict[str, Any]] = {}
    for class_name in FOCUS_CLASSES:
        control = baseline_by_class[class_name]
        for threshold in thresholds:
            metric = class_metrics(traces[class_name], threshold)
            candidate_per_class = dict(baseline_by_class)
            candidate_per_class[class_name] = metric
            pooled = pooled_metrics(candidate_per_class)
            recall_delta = float(metric["recall"]) - float(control["recall"])
            map_delta = float(pooled["mAP50_95"]) - float(baseline_pooled["mAP50_95"])
            fixed_anchor_delta = float(pooled["mAP50_95"]) - float(raw_anchor["mAP50_95"])
            passes_operating_control = recall_delta >= 0.05 - 1e-12 and map_delta >= -0.005 - 1e-12
            passes_strict_fixed_anchor = recall_delta >= 0.05 - 1e-12 and fixed_anchor_delta >= -0.005 - 1e-12
            row = {
                "variant": variant,
                "scan_mode": "one_focus_class_override_other_classes_fixed_0.25",
                "class_id": metric["class_id"],
                "class_name": class_name,
                "threshold": threshold,
                "support": metric["support"],
                "predictions": metric["predictions"],
                "true_positives": metric["true_positives"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "ap50": metric["ap50"],
                "ap50_95": metric["ap50_95"],
                "control_threshold": base_threshold,
                "control_precision": control["precision"],
                "control_recall": control["recall"],
                "control_f1": control["f1"],
                "recall_delta_vs_control": recall_delta,
                "pooled_precision": pooled["precision"],
                "pooled_recall": pooled["recall"],
                "pooled_f1": pooled["f1"],
                "pooled_mAP50": pooled["mAP50"],
                "pooled_mAP50_95": pooled["mAP50_95"],
                "pooled_mAP50_95_delta_vs_control": map_delta,
                "pooled_mAP50_95_delta_vs_fixed_raw_anchor": fixed_anchor_delta,
                "passes_operating_control_gate": passes_operating_control,
                "passes_strict_fixed_anchor_gate": passes_strict_fixed_anchor,
            }
            rows.append(row)
            if passes_operating_control:
                profile = {
                    "class_name": class_name,
                    "threshold": threshold,
                    "recall_delta_vs_control": recall_delta,
                    "f1": metric["f1"],
                    "pooled_mAP50_95_delta_vs_control": map_delta,
                }
                current = candidate_profiles.get(class_name)
                if current is None or (
                    float(profile["f1"]),
                    float(profile["pooled_mAP50_95_delta_vs_control"]),
                    float(profile["recall_delta_vs_control"]),
                    float(profile["threshold"]),
                ) > (
                    float(current["f1"]),
                    float(current["pooled_mAP50_95_delta_vs_control"]),
                    float(current["recall_delta_vs_control"]),
                    float(current["threshold"]),
                ):
                    candidate_profiles[class_name] = profile
    return baseline, rows, candidate_profiles


def plot_curves(rows: list[dict[str, Any]], output_root: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        return f"matplotlib unavailable: {error}"
    for variant in sorted({str(row["variant"]) for row in rows}):
        figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for axis, class_name in zip(axes.flat, FOCUS_CLASSES, strict=True):
            curve = [row for row in rows if row["variant"] == variant and row["class_name"] == class_name]
            curve.sort(key=lambda row: float(row["threshold"]))
            axis.plot([row["recall"] for row in curve], [row["precision"] for row in curve], marker="o", markersize=3)
            for row in curve:
                axis.annotate(f"{float(row['threshold']):.2f}", (row["recall"], row["precision"]), fontsize=7)
            axis.set_title(class_name)
            axis.set_xlabel("Recall at class threshold")
            axis.set_ylabel("Precision at class threshold")
            axis.grid(alpha=0.3)
        figure.savefig(output_root / f"{variant}_focus_pr_curves.png", dpi=180)
        plt.close(figure)
    return None


def write_report(
    path: Path,
    preflight: dict[str, Any],
    raw_scores: dict[str, dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> None:
    lines = [
        "# L1 E1b Per-Class Confidence Scan",
        "",
        "## Boundary",
        "",
        "- `auto_only` development OOF research only; it is not manual truth, a final model selection, or a deployment threshold decision.",
        "- No new inference or training ran. The scan only filters persisted post-model-NMS OOF dumps.",
        "- The frozen `release-validation` set was not read, mounted, linked, or evaluated.",
        "- Each row changes one focus class only; a six-class joint threshold profile has not been evaluated.",
        "",
        "## Raw Anchor Reproduction",
        "",
        "| Variant | Persisted boxes | Fixed E1 mAP50-95 | Reproduced mAP50-95 | Delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in ("yolo26m", "wbf_iou_0.55"):
        score = raw_scores[variant]
        fixed = FIXED_RAW_ANCHORS[variant]
        lines.append(
            f"| {variant} | {int(score['post_nms_prediction_count'])} | {fixed:.6f} | "
            f"{float(score['mAP50_95']):.6f} | {float(score['mAP50_95']) - fixed:+.9f} |"
        )
    lines.extend(
        [
            "",
            "The fixed E1 anchors use all persisted predictions down to the original model floor `0.001`. "
            "The scan gate instead uses an all-class `0.25` operating control, because a runtime confidence override "
            "must be compared with the same filtered decision path.",
            "",
            "## Thresholded Controls",
            "",
            "| Variant | P@0.25 | R@0.25 | F1@0.25 | mAP50 | mAP50-95 | Delta vs raw anchor |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in ("yolo26m", "wbf_iou_0.55"):
        control = baselines[variant]["pooled"]
        lines.append(
            f"| {variant} | {float(control['precision']):.6f} | {float(control['recall']):.6f} | "
            f"{float(control['f1']):.6f} | {float(control['mAP50']):.6f} | "
            f"{float(control['mAP50_95']):.6f} | "
            f"{float(baselines[variant]['thresholded_control_delta_vs_fixed_raw_anchor']):+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Gate Candidates",
            "",
            "Operating-control gate: focus-class Recall increase `>= +0.05` versus its `0.25` operating control and pooled mAP50-95 change `>= -0.005` versus that same control.",
            "",
            "| Variant | Class | Threshold | P | R | F1 | Recall delta | Delta vs `0.25` control | Delta vs raw anchor | Control gate | Strict anchor gate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    passing = [row for row in rows if bool(row["passes_operating_control_gate"])]
    for row in passing:
        lines.append(
            f"| {row['variant']} | {row['class_name']} | {float(row['threshold']):.2f} | "
            f"{float(row['precision']):.6f} | {float(row['recall']):.6f} | {float(row['f1']):.6f} | "
            f"{float(row['recall_delta_vs_control']):+.6f} | "
            f"{float(row['pooled_mAP50_95_delta_vs_control']):+.6f} | "
            f"{float(row['pooled_mAP50_95_delta_vs_fixed_raw_anchor']):+.6f} | pass | "
            f"{'pass' if row['passes_strict_fixed_anchor_gate'] else 'fail'} |"
        )
    if not passing:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | no candidate passed | n/a |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Final plan-gate status (strict fixed raw anchor): **`{gate['final_plan_gate']['status']}`**.",
            f"- Operating-control candidates: `{gate['operating_control_gate']['passing_candidate_count']}`.",
            f"- Strict fixed-anchor candidates: `{gate['strict_fixed_anchor_gate']['passing_candidate_count']}`.",
            "- Detailed curves and every threshold row are in `focus_threshold_curves.csv`; no candidate row is a tested joint profile.",
            "",
            "## Limitations",
            "",
            "- Thresholds are selected and measured on the same OOF development labels, so selection bias remains.",
            "- The OOF labels are `auto_only`; no result proves human-label accuracy, generalization, or a stable multi-seed benefit.",
            "- The stored source predictions were already model-NMS output. This scan cannot recover candidates removed by model NMS.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = tuple(sorted({round(float(value), 10) for value in args.thresholds}))
    if not thresholds or any(not 0 < value <= 1 for value in thresholds):
        raise ValueError("--thresholds must be non-empty values in (0, 1]")
    if not 0 < args.base_threshold <= 1 or not 0 < args.raw_conf_floor <= 1 or not 0 < args.nms_iou <= 1:
        raise ValueError("Confidence thresholds and --nms-iou must be in (0, 1]")
    if args.base_threshold not in thresholds:
        raise ValueError("--thresholds must include --base-threshold for a comparable control")
    output_root = assert_project_output(args.output_root)

    phase0_records, phase0_audits = load_phase0_records(args.phase0_root)
    yolo26_records, yolo26_audits = load_yolo26_records(args.e1_root)
    wbf_path = assert_development_path(args.e1_root / "fusion_predictions" / "wbf_iou_0.55.jsonl", "E1 WBF dump")
    wbf_records, wbf_audit = load_jsonl_records(wbf_path)
    validate_record_alignment(phase0_records, yolo26_records, "YOLO26m")
    validate_record_alignment(phase0_records, wbf_records, "WBF IoU 0.55")

    actual_counts = {
        "phase0_yolo11m": int(sum(len(record.predictions) for record in phase0_records.values())),
        "yolo26m": int(sum(len(record.predictions) for record in yolo26_records.values())),
        "wbf_iou_0.55": int(sum(len(record.predictions) for record in wbf_records.values())),
    }
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(f"Persisted prediction counts changed: actual={actual_counts}, expected={EXPECTED_COUNTS}")
    targets_by_image = {image_id: phase0_records[image_id].targets for image_id in sorted(phase0_records)}
    variants = {
        "yolo26m": {image_id: yolo26_records[image_id].predictions for image_id in sorted(yolo26_records)},
        "wbf_iou_0.55": {image_id: wbf_records[image_id].predictions for image_id in sorted(wbf_records)},
    }
    preflight = {
        "schema_version": 1,
        "purpose": "E1b per-class confidence threshold scan from persisted OOF predictions",
        "development_only": True,
        "frozen_test_set": None,
        "release_validation_access": "not read; not mounted; not linked; not evaluated",
        "no_new_inference": True,
        "no_training": True,
        "scan_mode": "one_focus_class_override_other_classes_fixed_at_base_threshold",
        "base_threshold": args.base_threshold,
        "thresholds": list(thresholds),
        "focus_classes": list(FOCUS_CLASSES),
        "raw_conf_floor": args.raw_conf_floor,
        "nms_iou": args.nms_iou,
        "pooled_images": len(phase0_records),
        "expected_prediction_counts": EXPECTED_COUNTS,
        "actual_prediction_counts": actual_counts,
        "phase0_folds": phase0_audits,
        "yolo26_folds": yolo26_audits,
        "wbf_iou_0.55": wbf_audit,
        "sources_aligned_to_phase0_targets": True,
    }
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "preflight": preflight}, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)
    write_json(output_root / "preflight.json", preflight)
    write_json(
        output_root / "requested_config.json",
        {
            "command": " ".join([sys.executable, Path(__file__).name, *sys.argv[1:]]),
            "phase0_root": str(args.phase0_root.resolve()),
            "e1_root": str(args.e1_root.resolve()),
            "base_threshold": args.base_threshold,
            "thresholds": list(thresholds),
            "raw_conf_floor": args.raw_conf_floor,
            "nms_iou": args.nms_iou,
            "focus_classes": list(FOCUS_CLASSES),
            "fixed_raw_anchors": FIXED_RAW_ANCHORS,
            "development_only": True,
            "frozen_test_set": None,
            "new_inference": False,
            "new_training": False,
        },
    )

    raw_scores = {
        variant: canonical_raw_score(predictions, targets_by_image, args.nms_iou) for variant, predictions in variants.items()
    }
    anchor_reproduction: dict[str, dict[str, float | bool]] = {}
    for variant, score in raw_scores.items():
        fixed = FIXED_RAW_ANCHORS[variant]
        reproduced = float(score["mAP50_95"])
        delta = reproduced - fixed
        anchor_reproduction[variant] = {
            "fixed_mAP50_95": fixed,
            "reproduced_mAP50_95": reproduced,
            "delta": delta,
            "within_1e_9": abs(delta) <= 1e-9,
        }
        if abs(delta) > 1e-9:
            raise RuntimeError(f"E1 fixed raw anchor did not reproduce for {variant}: {anchor_reproduction[variant]}")

    baselines: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    recommended: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, predictions in variants.items():
        traces = build_class_traces(predictions, targets_by_image)
        baseline, rows, candidate_profiles = build_candidate_rows(
            variant, traces, thresholds, args.base_threshold, raw_scores[variant]
        )
        baselines[variant] = baseline
        all_rows.extend(rows)
        recommended[variant] = candidate_profiles
    operating_passing = [row for row in all_rows if bool(row["passes_operating_control_gate"])]
    strict_passing = [row for row in all_rows if bool(row["passes_strict_fixed_anchor_gate"])]
    gate = {
        "operating_control_gate": {
            "status": "pass_keep_operating_control_candidate" if operating_passing else "not_passed_no_operating_control_candidate",
            "rule": {
                "focus_recall_delta_min": 0.05,
                "pooled_mAP50_95_delta_vs_thresholded_control_min": -0.005,
                "control": "all classes at 0.25; only one focus class overridden per row",
            },
            "passing_candidate_count": len(operating_passing),
            "passing_candidates": operating_passing,
        },
        "strict_fixed_anchor_gate": {
            "status": "pass_keep_candidate" if strict_passing else "not_passed_strict_fixed_anchor",
            "rule": {
                "focus_recall_delta_min": 0.05,
                "pooled_mAP50_95_delta_vs_fixed_raw_anchor_min": -0.005,
                "fixed_raw_anchors": FIXED_RAW_ANCHORS,
                "recall_control": "class-specific 0.25 P/R operating control",
            },
            "passing_candidate_count": len(strict_passing),
            "passing_candidates": strict_passing,
        },
        "final_plan_gate": {
            "status": "pass_keep_candidate" if strict_passing else "not_passed_strict_fixed_anchor",
            "policy": "The written E1b plan names fixed E1 raw mAP50-95 anchors; this strict gate controls the final plan decision.",
        },
        "recommended_single_overrides_passing_operating_control_only": recommended,
        "joint_profile_evaluated": False,
        "selection_bias": "same development OOF labels used for scanning and scoring; auto_only only",
    }
    plot_note = None if args.no_plots else plot_curves(all_rows, output_root)
    environment = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "device": "cpu_only_no_model_loaded",
        "new_inference": False,
        "new_training": False,
        "plot_note": plot_note,
    }
    write_json(output_root / "raw_anchor_reproduction.json", anchor_reproduction)
    write_json(output_root / "raw_scores.json", raw_scores)
    write_json(output_root / "thresholded_controls.json", baselines)
    write_csv(output_root / "focus_threshold_curves.csv", all_rows)
    write_json(output_root / "gate_decision.json", gate)
    write_json(output_root / "environment.json", environment)
    write_report(output_root / "report.md", preflight, raw_scores, baselines, all_rows, gate)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "final_plan_gate": gate["final_plan_gate"]["status"],
                "operating_control_passing_candidate_count": len(operating_passing),
                "strict_fixed_anchor_passing_candidate_count": len(strict_passing),
                "report": str(output_root / "report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
