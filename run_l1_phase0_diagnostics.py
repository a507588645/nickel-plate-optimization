#!/usr/bin/env python3
"""Run the CPU-only Phase 0 diagnostics for the L1 small-object plan.

The script intentionally reads only the three archived development-fold
re-evaluations. It never reads a release-validation list or writes to source
images, source labels, model runs, or the dataset root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from l1_tile_pipeline import CLASS_NAMES, box_iou_one_to_many, class_aware_nms


ROOT = Path(__file__).resolve().parent
DEFAULT_FOLD_ROOTS = (
    ROOT / "l1_reeval_yolo11m_oofconsensusv1_fold01",
    ROOT / "l1_reeval_yolo11m_oofconsensusv1_fold02",
    ROOT / "l1_reeval_yolo11m_oofconsensusv1_fold03",
)
DEFAULT_G4B_ARCHIVE = (
    ROOT
    / "kaggle_l1_ryw18_y11m_oof_v1_tile1024_o25_eval_f01_20260822_r2"
    / "remote_output_v1_extracted"
)
DEFAULT_THRESHOLDS = (
    0.001,
    0.01,
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
)
FOCUS_CLASSES = ("wuzi", "waterstain", "airhole", "bubble", "greenstain", "watermark")


@dataclass(frozen=True)
class FoldSource:
    fold: int
    root: Path
    metrics_path: Path
    weights: Path
    view_root: Path
    val_list: Path
    val_images: tuple[Path, ...]
    labels_root: Path
    source_metrics: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only Phase 0 per-class threshold scan for L1 OOF development folds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fold-roots", type=Path, nargs="+", default=list(DEFAULT_FOLD_ROOTS))
    parser.add_argument("--g4b-archive", type=Path, default=DEFAULT_G4B_ARCHIVE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cpu", help="Phase 0 is intentionally CPU-only.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--conf-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def normalized_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(value) for value in names]
    raise TypeError(f"Unsupported model names type: {type(names)!r}")


def read_val_images(val_list: Path, view_root: Path) -> tuple[Path, ...]:
    images_root = (view_root / "images").resolve()
    images: list[Path] = []
    for raw in val_list.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        candidate = Path(value)
        image = (candidate if candidate.is_absolute() else view_root / candidate).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Validation image does not exist: {image}")
        try:
            image.relative_to(images_root)
        except ValueError as error:
            raise ValueError(f"Validation image is outside the development-only view: {image}") from error
        if "release-validation" in image.as_posix().lower():
            raise ValueError(f"Release-validation image is forbidden in Phase 0: {image}")
        images.append(image)
    if not images:
        raise ValueError(f"Empty validation list: {val_list}")
    if len({image.name for image in images}) != len(images):
        raise ValueError(f"Duplicate image names in flat development view: {val_list}")
    return tuple(images)


def load_fold_source(root: Path) -> FoldSource:
    root = root.resolve()
    metrics_path = root / "metrics_per_class_reeval.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing archived development re-evaluation metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("development_only") is not True:
        raise ValueError(f"Phase 0 requires development_only=true: {metrics_path}")
    if metrics.get("frozen_test_list_not_evaluated") is not None:
        raise ValueError(f"Phase 0 refuses a source that references a frozen test list: {metrics_path}")
    fold = metrics.get("fold")
    if fold not in {1, 2, 3}:
        raise ValueError(f"Unexpected fold in {metrics_path}: {fold!r}")
    weights = Path(str(metrics.get("weights", ""))).resolve()
    view_root = Path(str(metrics.get("local_view_root", ""))).resolve()
    val_list = Path(str(metrics.get("local_val_list", ""))).resolve()
    labels_root = (view_root / "labels").resolve()
    if not weights.is_file() or not view_root.is_dir() or not val_list.is_file() or not labels_root.is_dir():
        raise FileNotFoundError(f"Missing weight or development-view input for Fold {fold}: {metrics_path}")
    images = read_val_images(val_list, view_root)
    for image in images:
        label = labels_root / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing development-view label: {label}")
    if int(metrics.get("val_images", 0)) != len(images):
        raise ValueError(f"Validation image count changed since archived re-evaluation: {metrics_path}")
    return FoldSource(
        fold=int(fold),
        root=root,
        metrics_path=metrics_path,
        weights=weights,
        view_root=view_root,
        val_list=val_list,
        val_images=images,
        labels_root=labels_root,
        source_metrics=metrics,
    )


def parse_yolo_label(path: Path, width: int, height: int) -> np.ndarray:
    rows: list[list[float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        values = raw.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"Expected a five-column YOLO label at {path}:{line_number}")
        class_id, x_center, y_center, box_width, box_height = map(float, values)
        if not class_id.is_integer() or not 0 <= int(class_id) < len(CLASS_NAMES):
            raise ValueError(f"Invalid class id at {path}:{line_number}")
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < box_width <= 1 and 0 < box_height <= 1):
            raise ValueError(f"Invalid normalized box at {path}:{line_number}")
        x1 = max(0.0, (x_center - box_width / 2.0) * width)
        y1 = max(0.0, (y_center - box_height / 2.0) * height)
        x2 = min(float(width), (x_center + box_width / 2.0) * width)
        y2 = min(float(height), (y_center + box_height / 2.0) * height)
        rows.append([class_id, x1, y1, x2, y2])
    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 5), dtype=np.float64)


def result_to_rows(result: Any) -> np.ndarray:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float64)
    coordinates = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    rows = np.column_stack((coordinates, confidence, classes)).astype(np.float64, copy=False)
    return class_aware_nms(rows.tolist(), 0.70)


def predict_fold(source: FoldSource, args: argparse.Namespace, output_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    model = YOLO(str(source.weights))
    if normalized_names(model.names) != list(CLASS_NAMES):
        raise ValueError(f"Class order mismatch for Fold {source.fold}: {source.weights}")
    predictions_by_image: dict[str, np.ndarray] = {}
    targets_by_image: dict[str, np.ndarray] = {}
    raw_file = output_path.open("w", encoding="utf-8")
    raw_count = 0
    try:
        for start in range(0, len(source.val_images), args.batch):
            batch_paths = source.val_images[start : start + args.batch]
            results = model.predict(
                source=[str(path) for path in batch_paths],
                imgsz=args.imgsz,
                conf=args.conf_floor,
                iou=args.nms_iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )
            if len(results) != len(batch_paths):
                raise RuntimeError(f"Prediction result count mismatch in Fold {source.fold}")
            for image_path, result in zip(batch_paths, results, strict=True):
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"OpenCV failed to read development image: {image_path}")
                height, width = image.shape[:2]
                image_key = f"fold{source.fold:02d}/{image_path.name}"
                predictions = result_to_rows(result)
                targets = parse_yolo_label(source.labels_root / f"{image_path.stem}.txt", width, height)
                predictions_by_image[image_key] = predictions
                targets_by_image[image_key] = targets
                raw_file.write(
                    json.dumps(
                        {
                            "image_id": image_key,
                            "source_image": str(image_path),
                            "width": width,
                            "height": height,
                            "predictions_xyxy_conf_class": predictions.tolist(),
                            "targets_class_xyxy": targets.tolist(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raw_count += len(predictions)
    finally:
        raw_file.close()
    audit = {
        "fold": source.fold,
        "weights": str(source.weights),
        "weights_sha256": sha256_file(source.weights),
        "images": len(source.val_images),
        "raw_prediction_count_after_model_nms": raw_count,
        "raw_prediction_file": str(output_path),
    }
    return predictions_by_image, targets_by_image, audit


def class_match_counts(
    predictions_by_image: dict[str, np.ndarray],
    targets_by_image: dict[str, np.ndarray],
    class_id: int,
    confidence_threshold: float,
    iou_threshold: float = 0.50,
) -> tuple[int, int, int]:
    targets: dict[str, np.ndarray] = {}
    candidates: list[tuple[float, str, np.ndarray]] = []
    for image_id, rows in targets_by_image.items():
        targets[image_id] = rows[rows[:, 0] == class_id, 1:]
    for image_id, rows in predictions_by_image.items():
        selected = rows[(rows[:, 5] == class_id) & (rows[:, 4] >= confidence_threshold)]
        candidates.extend((float(row[4]), image_id, row[:4]) for row in selected)
    candidates.sort(key=lambda value: value[0], reverse=True)
    matched: dict[str, set[int]] = defaultdict(set)
    true_positive = 0
    for _, image_id, box in candidates:
        ground_truth = targets[image_id]
        overlaps = box_iou_one_to_many(box, ground_truth)
        best_index = int(np.argmax(overlaps)) if len(overlaps) else -1
        if best_index >= 0 and overlaps[best_index] >= iou_threshold and best_index not in matched[image_id]:
            matched[image_id].add(best_index)
            true_positive += 1
    target_count = sum(len(rows) for rows in targets.values())
    return true_positive, len(candidates), target_count


def scan_thresholds(
    predictions_by_image: dict[str, np.ndarray],
    targets_by_image: dict[str, np.ndarray],
    thresholds: list[float],
    scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        for threshold in thresholds:
            true_positive, prediction_count, target_count = class_match_counts(
                predictions_by_image, targets_by_image, class_id, threshold
            )
            precision = true_positive / prediction_count if prediction_count else 0.0
            recall = true_positive / target_count if target_count else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(
                {
                    "scope": scope,
                    "class_id": class_id,
                    "class_name": class_name,
                    "threshold": threshold,
                    "iou": 0.50,
                    "targets": target_count,
                    "predictions": prediction_count,
                    "true_positives": true_positive,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return rows


def select_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["class_name"]].append(row)
    recommendations: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        curve = by_class[class_name]
        baseline = next(row for row in curve if abs(float(row["threshold"]) - 0.25) < 1e-12)
        targets = int(baseline["targets"])
        if targets == 0:
            recommendations.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "status": "not_evaluable_no_ground_truth",
                    "targets": 0,
                    "recommended_threshold": None,
                    "selection_objective": "max_f1_at_iou50",
                }
            )
            continue
        # Break F1 ties toward a higher threshold to avoid needless false positives.
        selected = max(curve, key=lambda row: (float(row["f1"]), float(row["threshold"])))
        recommendations.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "status": "evaluated",
                "targets": targets,
                "selection_objective": "max_f1_at_iou50_then_higher_threshold",
                "recommended_threshold": selected["threshold"],
                "recommended_precision": selected["precision"],
                "recommended_recall": selected["recall"],
                "recommended_f1": selected["f1"],
                "baseline_threshold": 0.25,
                "baseline_precision": baseline["precision"],
                "baseline_recall": baseline["recall"],
                "baseline_f1": baseline["f1"],
                "recall_delta_vs_025": float(selected["recall"]) - float(baseline["recall"]),
                "precision_delta_vs_025": float(selected["precision"]) - float(baseline["precision"]),
                "tier": "low" if float(selected["threshold"]) <= 0.20 else "default_or_high",
            }
        )
    return recommendations


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_focus_curves(rows: list[dict[str, Any]], output_path: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        return f"matplotlib unavailable: {error}"
    focus_rows = [row for row in rows if row["class_name"] in FOCUS_CLASSES and row["targets"]]
    if not focus_rows:
        return "No focus classes with ground truth for PR plotting."
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in focus_rows:
        grouped[str(row["class_name"])].append(row)
    for axis, class_name in zip(axes.flat, FOCUS_CLASSES, strict=True):
        curve = grouped.get(class_name, [])
        if not curve:
            axis.set_title(f"{class_name} (no ground truth)")
            axis.axis("off")
            continue
        curve.sort(key=lambda row: float(row["threshold"]))
        axis.plot([row["recall"] for row in curve], [row["precision"] for row in curve], marker="o", markersize=3)
        axis.set_title(class_name)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.grid(alpha=0.3)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return None


def inspect_g4b_attribution_gap(archive_root: Path) -> dict[str, Any]:
    archive_root = archive_root.resolve()
    if not archive_root.is_dir():
        raise FileNotFoundError(f"Missing G4b evaluation archive: {archive_root}")
    files = sorted(path.relative_to(archive_root).as_posix() for path in archive_root.rglob("*") if path.is_file())
    detail_markers = ("prediction", "predictions", "per_image", "raw_boxes", "detections")
    detail_files = [path for path in files if any(marker in path.lower() for marker in detail_markers)]
    report_paths = [path for path in files if path.endswith("tile_evaluation/baseline_tiled.json") or path.endswith("tile_evaluation/tile_model_tiled.json")]
    aggregate: dict[str, Any] = {}
    for relative in report_paths:
        payload = json.loads((archive_root / relative).read_text(encoding="utf-8"))
        aggregate[Path(relative).stem] = {
            "post_nms_prediction_count": payload.get("post_nms_prediction_count"),
            "precision_conf025_iou50_micro": payload.get("precision_conf025_iou50_micro"),
            "recall_conf025_iou50_micro": payload.get("recall_conf025_iou50_micro"),
            "mAP50_95": payload.get("mAP50_95"),
        }
    return {
        "status": "inconclusive_no_per_box_details" if not detail_files else "details_present",
        "archive_root": str(archive_root),
        "archive_files": files,
        "detail_files": detail_files,
        "unavailable_statistics": []
        if detail_files
        else [
            "overlap_region_duplicate_ratio",
            "confidence_below_025_ratio",
            "edge_truncated_box_ratio",
        ],
        "aggregate_tiled_reports": aggregate,
        "phase1_requirement": (
            "Persist per-tile raw boxes before and after cross-tile merging so the Phase 0 attribution gate can be measured."
            if not detail_files
            else None
        ),
    }


def build_report(
    output_root: Path,
    fold_audit: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    attribution: dict[str, Any],
    plot_error: str | None,
) -> None:
    lines = [
        "# L1 Phase 0 CPU Diagnostics",
        "",
        "## Boundary",
        "",
        "- CPU-only prediction over the three archived OOF development validation folds.",
        "- `release-validation` was not read, mounted, or evaluated.",
        "- Results are `auto_only`; they tune no deployed threshold and select no final model.",
        "",
        "## Fold Inputs",
        "",
        "| Fold | Images | Raw predictions after model NMS | Weight SHA-256 |",
        "| --- | ---: | ---: | --- |",
    ]
    for audit in fold_audit:
        lines.append(
            f"| {audit['fold']} | {audit['images']} | {audit['raw_prediction_count_after_model_nms']} | `{audit['weights_sha256']}` |"
        )
    lines.extend(["", "## Pooled Per-Class Threshold Recommendations", "", "| Class | Targets | Threshold | P@0.5 | R@0.5 | F1 | Delta recall vs 0.25 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in recommendations:
        if row["status"] != "evaluated":
            lines.append(f"| {row['class_name']} | 0 | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            "| {class_name} | {targets} | {recommended_threshold:.3f} | {recommended_precision:.4f} | "
            "{recommended_recall:.4f} | {recommended_f1:.4f} | {recall_delta_vs_025:+.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Thresholds are selected by maximum pooled F1 at IoU=0.50; exact ties prefer the higher threshold.",
            "The full threshold curves are in `per_class_threshold_curves.csv`; focus-class PR plots are in `focus_class_pr_curves.png` when matplotlib is available.",
            "",
            "## G4b Attribution Availability",
            "",
            f"- Status: `{attribution['status']}`.",
            f"- Detail files: {len(attribution['detail_files'])}.",
            f"- Phase 1 requirement: {attribution.get('phase1_requirement') or 'Detailed records are available.'}",
            "",
            "The existing G4b ZIP only contains aggregate reports, so it cannot prove whether overlap duplicates or low-confidence boxes exceed 50%.",
            "Phase 1 must persist raw per-tile predictions before evaluating WBF, edge filtering, or per-class thresholds.",
        ]
    )
    if plot_error:
        lines.extend(["", f"Plot note: {plot_error}"])
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device.lower() != "cpu":
        raise ValueError("Phase 0 must run with --device cpu and consume no GPU.")
    if args.batch <= 0 or args.imgsz <= 0 or args.cpu_threads <= 0 or args.max_det <= 0:
        raise ValueError("Batch, image size, CPU threads, and max-det must be positive.")
    if not 0 < args.conf_floor <= 1 or not 0 < args.nms_iou <= 1:
        raise ValueError("Invalid confidence floor or NMS IoU.")
    thresholds = sorted({float(value) for value in args.thresholds})
    if 0.25 not in thresholds or any(not 0 < value <= 1 for value in thresholds):
        raise ValueError("Thresholds must include 0.25 and stay within (0, 1].")
    fold_sources = [load_fold_source(path) for path in args.fold_roots]
    if sorted(source.fold for source in fold_sources) != [1, 2, 3]:
        raise ValueError("Phase 0 requires exactly one development source for Fold 1, Fold 2, and Fold 3.")
    attribution = inspect_g4b_attribution_gap(args.g4b_archive)
    output_root = (args.output_root or ROOT / f"l1_phase0_diagnostics_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    preflight = {
        "phase": "L1_small_object_phase0",
        "device": args.device,
        "development_only": True,
        "frozen_test_set": None,
        "folds": [
            {
                "fold": source.fold,
                "metrics_path": str(source.metrics_path),
                "weights": str(source.weights),
                "weights_sha256": sha256_file(source.weights),
                "view_root": str(source.view_root),
                "val_list": str(source.val_list),
                "val_images": len(source.val_images),
            }
            for source in fold_sources
        ],
        "g4b_attribution": attribution,
        "thresholds": thresholds,
        "inference": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "conf_floor": args.conf_floor,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
        },
    }
    if args.dry_run:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing Phase 0 output root: {output_root}")
    output_root.mkdir(parents=True)
    raw_root = output_root / "raw_predictions"
    raw_root.mkdir()
    torch.set_num_threads(args.cpu_threads)
    all_predictions: dict[str, np.ndarray] = {}
    all_targets: dict[str, np.ndarray] = {}
    fold_audit: list[dict[str, Any]] = []
    fold_curves: list[dict[str, Any]] = []
    for source in fold_sources:
        predictions, targets, audit = predict_fold(source, args, raw_root / f"fold{source.fold:02d}.jsonl")
        all_predictions.update(predictions)
        all_targets.update(targets)
        fold_audit.append(audit)
        fold_curves.extend(scan_thresholds(predictions, targets, thresholds, f"fold{source.fold:02d}"))
    pooled_curves = scan_thresholds(all_predictions, all_targets, thresholds, "pooled_oof_folds123")
    recommendations = select_recommendations(pooled_curves)
    write_csv(output_root / "per_class_threshold_curves.csv", [*fold_curves, *pooled_curves])
    write_csv(output_root / "threshold_recommendations.csv", recommendations)
    write_json(output_root / "threshold_recommendations.json", recommendations)
    write_json(output_root / "g4b_attribution_gap.json", attribution)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available_but_unused": bool(torch.cuda.is_available()),
        "ultralytics": __import__("ultralytics").__version__,
        "cpu_threads": args.cpu_threads,
    }
    write_json(output_root / "environment.json", environment)
    summary = {
        **preflight,
        "output_root": str(output_root),
        "fold_prediction_audit": fold_audit,
        "pooled_images": len(all_predictions),
        "pooled_raw_predictions_after_model_nms": sum(audit["raw_prediction_count_after_model_nms"] for audit in fold_audit),
        "phase1_gate": {
            "status": "inconclusive",
            "reason": "G4b result ZIP does not retain per-box detail needed to measure duplicate, low-confidence, or edge-truncation shares.",
            "next_requirement": "Phase 1 must persist those records before applying post-processing variants.",
        },
    }
    write_json(output_root / "phase0_summary.json", summary)
    plot_error = plot_focus_curves(pooled_curves, output_root / "focus_class_pr_curves.png")
    build_report(output_root, fold_audit, recommendations, attribution, plot_error)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[created] {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
