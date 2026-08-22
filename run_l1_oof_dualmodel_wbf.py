#!/usr/bin/env python3
"""Evaluate leakage-safe YOLO11m/YOLO26m OOF weighted-box fusion on CPU only.

The input YOLO11m predictions are the persisted Phase 0 records.  Each
YOLO26m prediction is generated only for the corresponding validation fold,
then fused image-by-image with the paired YOLO11m record.  The script never
reads a release-validation list and never writes into a source dataset or a
model-run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
from ultralytics import YOLO

from l1_tile_pipeline import CLASS_NAMES, box_iou_one_to_many, class_aware_nms, score_predictions


ROOT = Path(__file__).resolve().parent
DEFAULT_PHASE0_ROOT = ROOT / "l1_phase0_diagnostics_20260822_1455"
DEFAULT_YOLO26_REEVAL_ROOTS = (
    ROOT / "l1_reeval_yolo26m_oofconsensusv1_fold01",
    ROOT / "l1_reeval_yolo26m_oofconsensusv1_fold02",
    ROOT / "l1_reeval_yolo26m_oofconsensusv1_fold03",
)
FOCUS_CLASSES = ("wuzi", "waterstain", "airhole", "bubble", "greenstain", "watermark")
DEFAULT_WBF_IOUS = (0.25, 0.35, 0.45, 0.55)
RAW_SCHEMA_KEYS = (
    "image_id",
    "source_image",
    "width",
    "height",
    "predictions_xyxy_conf_class",
    "targets_class_xyxy",
)


@dataclass(frozen=True)
class RawRecord:
    """One Phase 0-compatible original-image prediction record."""

    image_id: str
    fold: int
    source_image: Path
    width: int
    height: int
    predictions: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class FoldInputs:
    """Validated paired inputs for exactly one OOF validation fold."""

    fold: int
    yolo11_records: tuple[RawRecord, ...]
    yolo26_weights: Path
    yolo26_weight_sha256: str
    yolo26_images_by_name: dict[str, Path]
    yolo26_labels_root: Path
    yolo26_metrics_path: Path
    yolo26_metrics: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only, leakage-safe dual-model OOF WBF evaluation for L1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase0-root", type=Path, default=DEFAULT_PHASE0_ROOT)
    parser.add_argument(
        "--yolo26-reeval-roots",
        type=Path,
        nargs="+",
        default=list(DEFAULT_YOLO26_REEVAL_ROOTS),
        help="Three archived YOLO26m development-fold re-evaluation roots.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cpu", help="This experiment is deliberately CPU-only.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--conf-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--wbf-ious", type=float, nargs="+", default=list(DEFAULT_WBF_IOUS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-images-per-fold",
        type=int,
        help="Optional smoke-test limit. A limited run is explicitly marked partial and cannot pass the gate.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not create output or run inference.")
    return parser.parse_args(argv)


def normalized_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(value) for value in names]
    raise TypeError(f"Unsupported model class-name type: {type(names)!r}")


def ensure_array(value: Any, columns: int, field: str, record_id: str) -> np.ndarray:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list in {record_id}")
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, columns), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{field} has invalid shape {array.shape!r} in {record_id}; expected (*, {columns})")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains non-finite values in {record_id}")
    return array


def parse_phase0_record(payload: Any, expected_fold: int) -> RawRecord:
    if not isinstance(payload, dict):
        raise ValueError("Phase 0 JSONL record must be an object")
    missing = [key for key in RAW_SCHEMA_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Phase 0 JSONL record is missing required keys: {missing}")
    image_id = payload["image_id"]
    if not isinstance(image_id, str) or not image_id.startswith(f"fold{expected_fold:02d}/"):
        raise ValueError(f"Phase 0 image_id does not belong to Fold {expected_fold}: {image_id!r}")
    source_image = Path(str(payload["source_image"])).resolve()
    if "release-validation" in source_image.as_posix().lower():
        raise ValueError(f"Forbidden release-validation source image: {source_image}")
    if not source_image.is_file():
        raise FileNotFoundError(f"Phase 0 source image no longer exists: {source_image}")
    if source_image.name != Path(image_id).name:
        raise ValueError(f"Phase 0 source-image name mismatch: {image_id} vs {source_image.name}")
    width, height = payload["width"], payload["height"]
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {image_id}")
    predictions = ensure_array(payload["predictions_xyxy_conf_class"], 6, "predictions", image_id)
    targets = ensure_array(payload["targets_class_xyxy"], 5, "targets", image_id)
    if len(predictions):
        if np.any(predictions[:, :2] < 0) or np.any(predictions[:, 2] > width) or np.any(predictions[:, 3] > height):
            raise ValueError(f"Prediction coordinates outside image bounds in {image_id}")
        # Phase 0 retained post-NMS candidates exactly as emitted by the model.
        # At an image boundary Ultralytics can emit a zero-area clipped box;
        # the established scorer treats it as an unmatched candidate. Preserve
        # it instead of changing the archived prediction-count protocol.
        if np.any(predictions[:, 2] < predictions[:, 0]) or np.any(predictions[:, 3] < predictions[:, 1]):
            raise ValueError(f"Inverted prediction box in {image_id}")
        if np.any((predictions[:, 4] < 0) | (predictions[:, 4] > 1)):
            raise ValueError(f"Prediction confidence outside [0, 1] in {image_id}")
        if np.any((predictions[:, 5] < 0) | (predictions[:, 5] >= len(CLASS_NAMES))):
            raise ValueError(f"Prediction class outside RYW18 range in {image_id}")
    if len(targets):
        if np.any(targets[:, :1] < 0) or np.any(targets[:, :1] >= len(CLASS_NAMES)):
            raise ValueError(f"Target class outside RYW18 range in {image_id}")
        if np.any(targets[:, 1:3] < 0) or np.any(targets[:, 3] > width) or np.any(targets[:, 4] > height):
            raise ValueError(f"Target coordinates outside image bounds in {image_id}")
        if np.any(targets[:, 3] <= targets[:, 1]) or np.any(targets[:, 4] <= targets[:, 2]):
            raise ValueError(f"Degenerate target box in {image_id}")
    return RawRecord(
        image_id=image_id,
        fold=expected_fold,
        source_image=source_image,
        width=width,
        height=height,
        predictions=predictions,
        targets=targets,
    )


def load_phase0_fold(phase0_root: Path, fold: int) -> tuple[list[RawRecord], dict[str, Any]]:
    raw_path = phase0_root / "raw_predictions" / f"fold{fold:02d}.jsonl"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing Phase 0 raw prediction file: {raw_path}")
    records: list[RawRecord] = []
    seen_ids: set[str] = set()
    with raw_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"Blank line in Phase 0 JSONL: {raw_path}:{line_number}")
            try:
                record = parse_phase0_record(json.loads(raw), fold)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {raw_path}:{line_number}") from error
            if record.image_id in seen_ids:
                raise ValueError(f"Duplicate Phase 0 image_id: {record.image_id}")
            seen_ids.add(record.image_id)
            records.append(record)
    if not records:
        raise ValueError(f"Phase 0 raw prediction file is empty: {raw_path}")
    return records, {
        "path": str(raw_path),
        "sha256": sha256_file(raw_path),
        "images": len(records),
        "predictions_after_model_nms": int(sum(len(record.predictions) for record in records)),
        "targets": int(sum(len(record.targets) for record in records)),
    }


def parse_yolo_label(path: Path, width: int, height: int) -> np.ndarray:
    rows: list[list[float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Expected five-column YOLO label: {path}:{line_number}")
        try:
            class_id, x_center, y_center, box_width, box_height = map(float, fields)
        except ValueError as error:
            raise ValueError(f"Invalid YOLO label value: {path}:{line_number}") from error
        if not class_id.is_integer() or not 0 <= int(class_id) < len(CLASS_NAMES):
            raise ValueError(f"Invalid YOLO class ID: {path}:{line_number}")
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < box_width <= 1 and 0 < box_height <= 1):
            raise ValueError(f"Invalid normalized YOLO box: {path}:{line_number}")
        x1 = max(0.0, (x_center - box_width / 2.0) * width)
        y1 = max(0.0, (y_center - box_height / 2.0) * height)
        x2 = min(float(width), (x_center + box_width / 2.0) * width)
        y2 = min(float(height), (y_center + box_height / 2.0) * height)
        rows.append([class_id, x1, y1, x2, y2])
    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 5), dtype=np.float64)


def read_validation_images(val_list: Path, view_root: Path) -> dict[str, Path]:
    images_root = (view_root / "images").resolve()
    if not images_root.is_dir():
        raise FileNotFoundError(f"Missing archived re-evaluation images: {images_root}")
    images: dict[str, Path] = {}
    for line_number, raw in enumerate(val_list.read_text(encoding="utf-8").splitlines(), start=1):
        value = raw.strip()
        if not value:
            continue
        candidate = Path(value)
        image = (candidate if candidate.is_absolute() else view_root / candidate).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Missing YOLO26m validation image: {image}")
        try:
            image.relative_to(images_root)
        except ValueError as error:
            raise ValueError(f"YOLO26m validation image is outside its development view: {image}") from error
        if "release-validation" in image.as_posix().lower():
            raise ValueError(f"Forbidden release-validation image in re-evaluation: {image}")
        if image.name in images:
            raise ValueError(f"Duplicate YOLO26m validation image name: {image.name} at {val_list}:{line_number}")
        images[image.name] = image
    if not images:
        raise ValueError(f"YOLO26m validation list is empty: {val_list}")
    return images


def validate_image_and_target(record: RawRecord, image: Path, labels_root: Path) -> None:
    # `cv2.imread` decodes ~20 GiB of source pixels just to inspect dimensions.
    # PIL reads PNG headers only, keeping the preflight CPU-only and practical.
    from PIL import Image

    with Image.open(image) as source:
        width, height = source.size
    if (width, height) != (record.width, record.height):
        raise ValueError(
            f"Image dimensions differ between Phase 0 and YOLO26m views for {record.image_id}: "
            f"Phase0={(record.width, record.height)}, YOLO26={(width, height)}"
        )
    label = labels_root / f"{image.stem}.txt"
    if not label.is_file():
        raise FileNotFoundError(f"Missing YOLO26m development label: {label}")
    targets = parse_yolo_label(label, width, height)
    if targets.shape != record.targets.shape or not np.allclose(targets, record.targets, rtol=0, atol=1e-5):
        raise ValueError(f"YOLO26m label target differs from Phase 0 target for {record.image_id}")


def load_yolo26_fold(reeval_root: Path, fold: int, yolo11_records: list[RawRecord]) -> FoldInputs:
    reeval_root = reeval_root.resolve()
    metrics_path = reeval_root / "metrics_per_class_reeval.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing YOLO26m re-evaluation metrics: {metrics_path}")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid YOLO26m re-evaluation JSON: {metrics_path}") from error
    if metrics.get("fold") != fold or metrics.get("development_only") is not True:
        raise ValueError(f"YOLO26m re-evaluation is not Fold {fold} development-only: {metrics_path}")
    if metrics.get("frozen_test_list_not_evaluated") is not None or metrics.get("frozen_test_images_not_evaluated") is not None:
        raise ValueError(f"YOLO26m re-evaluation references a frozen test set: {metrics_path}")
    weights = Path(str(metrics.get("weights", ""))).resolve()
    view_root = Path(str(metrics.get("local_view_root", ""))).resolve()
    val_list = Path(str(metrics.get("local_val_list", ""))).resolve()
    labels_root = (view_root / "labels").resolve()
    if not weights.is_file() or not view_root.is_dir() or not val_list.is_file() or not labels_root.is_dir():
        raise FileNotFoundError(f"YOLO26m Fold {fold} archived inputs are incomplete: {metrics_path}")
    if "dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_oofconsensusv1_fold" not in weights.as_posix():
        raise ValueError(f"Unexpected YOLO26m OOF weight path: {weights}")
    yolo26_images_by_name = read_validation_images(val_list, view_root)
    yolo11_names = {record.source_image.name for record in yolo11_records}
    if yolo11_names != set(yolo26_images_by_name):
        only_yolo11 = sorted(yolo11_names - set(yolo26_images_by_name))
        only_yolo26 = sorted(set(yolo26_images_by_name) - yolo11_names)
        raise ValueError(
            f"Fold {fold} YOLO11m/YOLO26m validation membership differs; "
            f"only_yolo11={only_yolo11[:3]}, only_yolo26={only_yolo26[:3]}"
        )
    if int(metrics.get("val_images", 0)) != len(yolo11_records):
        raise ValueError(f"YOLO26m Fold {fold} val count differs from Phase 0 records")
    for record in yolo11_records:
        validate_image_and_target(record, yolo26_images_by_name[record.source_image.name], labels_root)
    return FoldInputs(
        fold=fold,
        yolo11_records=tuple(yolo11_records),
        yolo26_weights=weights,
        yolo26_weight_sha256=sha256_file(weights),
        yolo26_images_by_name=yolo26_images_by_name,
        yolo26_labels_root=labels_root,
        yolo26_metrics_path=metrics_path,
        yolo26_metrics=metrics,
    )


def result_to_rows(result: Any, nms_iou: float) -> np.ndarray:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float64)
    coordinates = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    rows = np.column_stack((coordinates, confidence, classes)).astype(np.float64, copy=False)
    return class_aware_nms(rows.tolist(), nms_iou)


def emit_raw_record(handle: Any, record: RawRecord, source_image: Path, predictions: np.ndarray) -> None:
    handle.write(
        json.dumps(
            {
                "image_id": record.image_id,
                "source_image": str(source_image),
                "width": record.width,
                "height": record.height,
                "predictions_xyxy_conf_class": predictions.tolist(),
                "targets_class_xyxy": record.targets.tolist(),
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def predict_yolo26_fold(
    inputs: FoldInputs,
    args: argparse.Namespace,
    output_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected = list(inputs.yolo11_records)
    if args.max_images_per_fold is not None:
        selected = selected[: args.max_images_per_fold]
    model = YOLO(str(inputs.yolo26_weights))
    if normalized_names(model.names) != list(CLASS_NAMES):
        raise ValueError(f"YOLO26m class order does not match RYW18: {inputs.yolo26_weights}")
    predictions_by_image: dict[str, np.ndarray] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_count = 0
    with output_path.open("x", encoding="utf-8") as handle:
        for start in range(0, len(selected), args.batch):
            batch_records = selected[start : start + args.batch]
            batch_paths = [inputs.yolo26_images_by_name[record.source_image.name] for record in batch_records]
            results = model.predict(
                source=[str(path) for path in batch_paths],
                imgsz=args.imgsz,
                conf=args.conf_floor,
                iou=args.nms_iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )
            if len(results) != len(batch_records):
                raise RuntimeError(f"YOLO26m result count mismatch in Fold {inputs.fold}")
            for record, image_path, result in zip(batch_records, batch_paths, results, strict=True):
                predictions = result_to_rows(result, args.nms_iou)
                predictions_by_image[record.image_id] = predictions
                emit_raw_record(handle, record, image_path, predictions)
                raw_count += len(predictions)
    return predictions_by_image, {
        "fold": inputs.fold,
        "images": len(selected),
        "weights": str(inputs.yolo26_weights),
        "weights_sha256": inputs.yolo26_weight_sha256,
        "raw_prediction_file": str(output_path),
        "raw_prediction_file_sha256": sha256_file(output_path),
        "raw_prediction_count_after_model_nms": raw_count,
    }


def pairwise_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Use the established original-image IoU implementation for WBF clustering."""
    return box_iou_one_to_many(box, boxes)


def weighted_box_fusion_two_sources(
    yolo11_predictions: np.ndarray,
    yolo26_predictions: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, Counter[str]]:
    """Greedy per-class WBF with confidence-weighted coordinates and max score.

    `ensemble-boxes` is intentionally not installed into the shared project
    environment.  This deterministic implementation only fuses the two
    already model-NMS-filtered sources for one OOF image.  A single-source
    cluster remains unchanged; a two-source cluster gets weighted coordinates
    and retains its maximum source confidence for the existing score protocol.
    """
    entries: list[tuple[float, int, int, np.ndarray]] = []
    for source_index, predictions in enumerate((yolo11_predictions, yolo26_predictions)):
        for row_index, row in enumerate(predictions):
            entries.append((float(row[4]), source_index, row_index, row))
    entries.sort(key=lambda item: (-item[0], item[1], item[2]))
    clusters_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for confidence, source_index, _, row in entries:
        class_id = int(row[5])
        clusters = clusters_by_class[class_id]
        if clusters:
            cluster_boxes = np.asarray([cluster["xyxy"] for cluster in clusters], dtype=np.float64)
            overlaps = pairwise_iou(row[:4], cluster_boxes)
            best_index = int(np.argmax(overlaps))
            if overlaps[best_index] >= iou_threshold:
                cluster = clusters[best_index]
                cluster["weighted_sum"] += row[:4] * confidence
                cluster["weight_sum"] += confidence
                cluster["xyxy"] = cluster["weighted_sum"] / max(cluster["weight_sum"], 1e-12)
                cluster["confidence"] = max(float(cluster["confidence"]), confidence)
                cluster["sources"].add(source_index)
                cluster["members"] += 1
                continue
        clusters.append(
            {
                "xyxy": row[:4].astype(np.float64, copy=True),
                "weighted_sum": row[:4].astype(np.float64, copy=True) * confidence,
                "weight_sum": confidence,
                "confidence": confidence,
                "sources": {source_index},
                "members": 1,
            }
        )
    fused_rows: list[list[float]] = []
    source_composition: Counter[str] = Counter()
    for class_id in sorted(clusters_by_class):
        for cluster in clusters_by_class[class_id]:
            source_key = "both" if cluster["sources"] == {0, 1} else ("yolo11m_only" if 0 in cluster["sources"] else "yolo26m_only")
            source_composition[source_key] += 1
            fused_rows.append(
                [
                    float(cluster["xyxy"][0]),
                    float(cluster["xyxy"][1]),
                    float(cluster["xyxy"][2]),
                    float(cluster["xyxy"][3]),
                    float(cluster["confidence"]),
                    float(class_id),
                ]
            )
    fused_rows.sort(key=lambda row: (-row[4], int(row[5]), row[0], row[1]))
    return (
        np.asarray(fused_rows, dtype=np.float64) if fused_rows else np.empty((0, 6), dtype=np.float64),
        source_composition,
    )


def score_with_f1(
    predictions_by_image: dict[str, np.ndarray],
    targets_by_image: dict[str, np.ndarray],
    nms_iou: float,
) -> dict[str, Any]:
    scores = score_predictions(predictions_by_image, targets_by_image, SimpleNamespace(nms_iou=nms_iou))
    precision = float(scores["precision_conf025_iou50_micro"])
    recall = float(scores["recall_conf025_iou50_micro"])
    scores["f1_conf025_iou50_micro"] = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    scores["post_nms_prediction_count"] = int(sum(len(rows) for rows in predictions_by_image.values()))
    for values in scores["per_class"].values():
        class_precision = float(values["precision_conf025_iou50"])
        class_recall = float(values["recall_conf025_iou50"])
        values["f1_conf025_iou50"] = (
            2.0 * class_precision * class_recall / (class_precision + class_recall)
            if class_precision + class_recall
            else 0.0
        )
    return scores


def summary_row(variant: str, score: dict[str, Any], wbf_iou: float | None, raw_input_boxes: int | None) -> dict[str, Any]:
    return {
        "variant": variant,
        "wbf_iou": "" if wbf_iou is None else f"{wbf_iou:.2f}",
        "images": score["images"],
        "raw_input_boxes": "" if raw_input_boxes is None else raw_input_boxes,
        "postprocess_boxes": score["post_nms_prediction_count"],
        "precision_conf025_iou50_micro": score["precision_conf025_iou50_micro"],
        "recall_conf025_iou50_micro": score["recall_conf025_iou50_micro"],
        "f1_conf025_iou50_micro": score["f1_conf025_iou50_micro"],
        "mAP50": score["mAP50"],
        "mAP50_95": score["mAP50_95"],
    }


def per_class_rows(variant: str, score: dict[str, Any], wbf_iou: float | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        values = score["per_class"][class_name]
        rows.append(
            {
                "variant": variant,
                "wbf_iou": "" if wbf_iou is None else f"{wbf_iou:.2f}",
                "class_id": class_id,
                "class_name": class_name,
                "support": values["support"],
                "ap50": values["ap50"],
                "ap50_95": values["ap50_95"],
                "precision_conf025_iou50": values["precision_conf025_iou50"],
                "recall_conf025_iou50": values["recall_conf025_iou50"],
                "f1_conf025_iou50": values["f1_conf025_iou50"],
            }
        )
    return rows


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def determine_gate(
    yolo11_score: dict[str, Any], yolo26_score: dict[str, Any], fusion_scores: dict[float, dict[str, Any]], partial: bool
) -> dict[str, Any]:
    best_single_map = max(float(yolo11_score["mAP50_95"]), float(yolo26_score["mAP50_95"]))
    best_single_f1_by_class = {
        class_name: max(
            float(yolo11_score["per_class"][class_name]["f1_conf025_iou50"]),
            float(yolo26_score["per_class"][class_name]["f1_conf025_iou50"]),
        )
        for class_name in CLASS_NAMES
    }
    candidates: list[dict[str, Any]] = []
    for wbf_iou, score in fusion_scores.items():
        focus_deltas = {
            class_name: float(score["per_class"][class_name]["f1_conf025_iou50"]) - best_single_f1_by_class[class_name]
            for class_name in FOCUS_CLASSES
        }
        map_delta = float(score["mAP50_95"]) - best_single_map
        qualifying_focus = [class_name for class_name, delta in focus_deltas.items() if delta >= 0.01]
        passes = map_delta >= 0.01 or bool(qualifying_focus)
        candidates.append(
            {
                "wbf_iou": wbf_iou,
                "mAP50_95": float(score["mAP50_95"]),
                "mAP50_95_delta_vs_best_single": map_delta,
                "focus_f1_delta_vs_best_single": focus_deltas,
                "qualifying_focus_classes": qualifying_focus,
                "passes_keep_threshold": passes,
            }
        )
    qualifying = [candidate for candidate in candidates if candidate["passes_keep_threshold"]]
    if partial:
        status = "partial_smoke_no_gate"
        reason = "A --max-images-per-fold limit was used; no experiment-selection gate is valid."
    elif qualifying:
        status = "pass_keep"
        reason = "At least one WBF IoU reaches +0.01 mAP50-95 versus the best single model or +0.01 F1 for a focus class."
    else:
        status = "reject"
        reason = "No WBF IoU reaches the predeclared +0.01 mAP50-95 or focus-class F1 threshold versus the best single model."
    selected = max(
        candidates,
        key=lambda item: (
            int(item["passes_keep_threshold"]),
            item["mAP50_95_delta_vs_best_single"],
            max(item["focus_f1_delta_vs_best_single"].values()),
            -float(item["wbf_iou"]),
        ),
    )
    return {
        "threshold": 0.01,
        "comparison_baseline": "best of the two pooled single-model OOF scores, per metric",
        "best_single_mAP50_95": best_single_map,
        "best_single_focus_f1": {name: best_single_f1_by_class[name] for name in FOCUS_CLASSES},
        "candidates": candidates,
        "selected_wbf_iou": selected["wbf_iou"],
        "status": status,
        "reason": reason,
    }


def write_report(
    path: Path,
    preflight: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    gate: dict[str, Any],
    fusion_composition: dict[float, Counter[str]],
) -> None:
    lines = [
        "# L1 Dual-Model OOF WBF Evaluation",
        "",
        "## Boundary",
        "",
        "- `auto_only` development OOF evaluation only; this is not manual truth or a deployment decision.",
        "- The frozen 198-image `release-validation` set was not read, mounted, or evaluated.",
        "- Each image uses exactly `YOLO11m_foldK + YOLO26m_foldK`; other-fold weights are rejected during preflight.",
        "- WBF is a deterministic inlined fallback because `ensemble-boxes` is absent from the fixed shared venv and was not installed.",
        "",
        "## Input Coverage",
        "",
        "| Fold | Images | YOLO11m raw boxes | YOLO26m raw boxes | YOLO26m weight SHA-256 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for fold in preflight["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['images']} | {fold['yolo11_raw_prediction_count']} | "
            f"{fold.get('yolo26_raw_prediction_count', 'pending')} | `{fold['yolo26_weights_sha256']}` |"
        )
    lines.extend(
        [
            "",
            f"- Pooled images: `{preflight['pooled_images']}` (expected `{preflight['expected_pooled_images']}`).",
            f"- Partial smoke run: `{preflight['partial_smoke']}`.",
            "- The Phase 0 JSONL SHA-256 values and all YOLO26m raw prediction file SHA-256 values are in `preflight_and_output_audit.json`.",
            "",
            "## Overall Metrics",
            "",
            "Metric protocol: original-image coordinates, model-NMS input floor `0.001`, P/R/F1 at confidence `0.25` and IoU `0.50`, AP at IoU `0.50:0.95` using the established 101-point envelope.",
            "",
            "| Variant | WBF IoU | Input boxes | Output boxes | P@0.25 | R@0.25 | F1@0.25 | mAP50 | mAP50-95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {row['wbf_iou'] or 'n/a'} | {row['raw_input_boxes'] or 'n/a'} | "
            f"{row['postprocess_boxes']} | {float(row['precision_conf025_iou50_micro']):.6f} | "
            f"{float(row['recall_conf025_iou50_micro']):.6f} | {float(row['f1_conf025_iou50_micro']):.6f} | "
            f"{float(row['mAP50']):.6f} | {float(row['mAP50_95']):.6f} |"
        )
    for variant, score in scores.items():
        if not variant.startswith("wbf_iou_"):
            continue
        wbf_iou = variant.removeprefix("wbf_iou_")
        lines.extend(
            [
                "",
                f"## Per-Class Metrics: WBF IoU {wbf_iou}",
                "",
                "| Class | Support | P@0.25 | R@0.25 | F1@0.25 | AP50 | AP50-95 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for class_name in CLASS_NAMES:
            values = score["per_class"][class_name]
            lines.append(
                f"| {class_name} | {values['support']} | {float(values['precision_conf025_iou50']):.6f} | "
                f"{float(values['recall_conf025_iou50']):.6f} | {float(values['f1_conf025_iou50']):.6f} | "
                f"{format_float(values['ap50'])} | {format_float(values['ap50_95'])} |"
            )
        composition = fusion_composition[float(wbf_iou)]
        lines.append(
            f"\nFused-source composition: both=`{composition['both']}`, YOLO11m-only=`{composition['yolo11m_only']}`, "
            f"YOLO26m-only=`{composition['yolo26m_only']}`."
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Status: **`{gate['status']}`**.",
            f"- Selected WBF IoU for the stated rule: `{gate['selected_wbf_iou']:.2f}`.",
            f"- Rule: `{gate['reason']}`",
            f"- Best pooled single-model mAP50-95 baseline: `{gate['best_single_mAP50_95']:.6f}`.",
            "- Candidate deltas and every focus-class comparison are retained in `gate_decision.json`.",
            "",
            "## Reproducibility Files",
            "",
            "- `raw_predictions_yolo26/fold01.jsonl` through `fold03.jsonl`: Phase 0-compatible YOLO26m raw OOF schema.",
            "- `fusion_predictions/wbf_iou_*.jsonl`: exact fused prediction records and original targets.",
            "- `metrics_summary.csv`, `per_class_metrics.csv`, and `metrics/*.json`: machine-readable metric tables.",
            "- `random_pair_audit.json`: deterministic five-image provenance sample; all source folds must match each image fold.",
            "- `environment.json` and `preflight_and_output_audit.json`: runtime, file hashes, and no-leakage checks.",
            "",
            "## Limitations",
            "",
            "- OOF labels are `auto_only`; no result establishes manual-ground-truth accuracy or stable multi-seed benefit.",
            "- WBF IoUs are scanned on this same development OOF set, so a keep result is only a candidate for later frozen evaluation under an approved protocol.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_args(args: argparse.Namespace) -> None:
    if args.device.lower() != "cpu":
        raise ValueError("E1 must run with --device cpu and consume no GPU.")
    if args.batch <= 0 or args.imgsz <= 0 or args.cpu_threads <= 0 or args.max_det <= 0:
        raise ValueError("batch, imgsz, cpu-threads, and max-det must be positive")
    if not 0 < args.conf_floor <= 1 or not 0 < args.nms_iou <= 1:
        raise ValueError("conf-floor and nms-iou must be in (0, 1]")
    wbf_ious = sorted({float(value) for value in args.wbf_ious})
    if wbf_ious != list(DEFAULT_WBF_IOUS):
        raise ValueError(f"E1 requires exactly the predeclared WBF IoUs: {DEFAULT_WBF_IOUS}")
    args.wbf_ious = tuple(wbf_ious)
    if len(args.yolo26_reeval_roots) != 3:
        raise ValueError("E1 requires exactly three YOLO26m re-evaluation roots")
    if args.max_images_per_fold is not None and args.max_images_per_fold <= 0:
        raise ValueError("--max-images-per-fold must be positive")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    verify_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    phase0_root = args.phase0_root.resolve()
    if not phase0_root.is_dir():
        raise FileNotFoundError(f"Missing Phase 0 root: {phase0_root}")

    yolo11_by_image: dict[str, np.ndarray] = {}
    targets_by_image: dict[str, np.ndarray] = {}
    fold_records: dict[int, list[RawRecord]] = {}
    phase0_audit: dict[int, dict[str, Any]] = {}
    all_image_ids: set[str] = set()
    for fold in (1, 2, 3):
        records, audit = load_phase0_fold(phase0_root, fold)
        fold_records[fold] = records
        phase0_audit[fold] = audit
        for record in records:
            if record.image_id in all_image_ids:
                raise ValueError(f"An image appears in more than one Phase 0 OOF fold: {record.image_id}")
            all_image_ids.add(record.image_id)
            yolo11_by_image[record.image_id] = record.predictions
            targets_by_image[record.image_id] = record.targets
    expected_counts = {1: 376, 2: 375, 3: 376}
    actual_counts = {fold: len(records) for fold, records in fold_records.items()}
    if actual_counts != expected_counts or len(all_image_ids) != 1127:
        raise ValueError(f"Unexpected Phase 0 OOF coverage: folds={actual_counts}, total={len(all_image_ids)}")

    yolo26_by_root: dict[int, Path] = {}
    for root in args.yolo26_reeval_roots:
        metrics_path = root.resolve() / "metrics_per_class_reeval.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing re-evaluation metrics: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        fold = metrics.get("fold")
        if fold not in {1, 2, 3} or fold in yolo26_by_root:
            raise ValueError(f"YOLO26m re-evaluation roots must describe folds 1, 2, 3 exactly once: {root}")
        yolo26_by_root[int(fold)] = root
    if sorted(yolo26_by_root) != [1, 2, 3]:
        raise ValueError("YOLO26m roots do not cover folds 1, 2, 3")
    fold_inputs = {fold: load_yolo26_fold(yolo26_by_root[fold], fold, fold_records[fold]) for fold in (1, 2, 3)}

    partial = args.max_images_per_fold is not None
    selected_ids = {
        record.image_id
        for fold in (1, 2, 3)
        for record in (fold_records[fold][: args.max_images_per_fold] if partial else fold_records[fold])
    }
    preflight: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "E1 dual-model OOF original-image WBF; auto_only research evaluation",
        "development_only": True,
        "frozen_test_set": None,
        "release_validation_access": "not read; not mounted; not evaluated",
        "phase0_root": str(phase0_root),
        "expected_pooled_images": 1127,
        "pooled_images": len(selected_ids),
        "partial_smoke": partial,
        "max_images_per_fold": args.max_images_per_fold,
        "inference": {
            "device": "cpu",
            "batch": args.batch,
            "imgsz": args.imgsz,
            "conf_floor": args.conf_floor,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "seed": args.seed,
        },
        "wbf": {
            "ious": list(args.wbf_ious),
            "implementation": "inlined_confidence_weighted_coordinates_max_source_confidence",
            "ensemble_boxes_available": False,
            "ensemble_boxes_installation": "not attempted; shared venv remains unchanged",
        },
        "folds": [],
    }
    for fold in (1, 2, 3):
        source = fold_inputs[fold]
        selected_count = len(fold_records[fold][: args.max_images_per_fold]) if partial else len(fold_records[fold])
        preflight["folds"].append(
            {
                "fold": fold,
                "images": selected_count,
                "full_fold_images": len(fold_records[fold]),
                "yolo11_phase0_raw_file": phase0_audit[fold]["path"],
                "yolo11_phase0_raw_file_sha256": phase0_audit[fold]["sha256"],
                "yolo11_raw_prediction_count": (
                    int(sum(len(record.predictions) for record in fold_records[fold][: args.max_images_per_fold]))
                    if partial
                    else phase0_audit[fold]["predictions_after_model_nms"]
                ),
                "yolo26_metrics_path": str(source.yolo26_metrics_path),
                "yolo26_weights": str(source.yolo26_weights),
                "yolo26_weights_sha256": source.yolo26_weight_sha256,
                "same_validation_membership": True,
                "same_targets_checked": True,
            }
        )
    output_root = (args.output_root or ROOT / f"l1_oof_dualmodel_wbf_{datetime.now():%Y%m%d_%H%M}").resolve()
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "preflight": preflight}, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing E1 output root: {output_root}")
    output_root.mkdir(parents=True)
    write_json(output_root / "preflight.json", preflight)

    yolo26_by_image: dict[str, np.ndarray] = {}
    yolo26_output_audit: dict[int, dict[str, Any]] = {}
    for fold in (1, 2, 3):
        raw_output = output_root / "raw_predictions_yolo26" / f"fold{fold:02d}.jsonl"
        predictions, audit = predict_yolo26_fold(fold_inputs[fold], args, raw_output)
        expected_fold_ids = {
            record.image_id
            for record in (fold_records[fold][: args.max_images_per_fold] if partial else fold_records[fold])
        }
        if set(predictions) != expected_fold_ids:
            raise RuntimeError(f"YOLO26m prediction coverage mismatch for Fold {fold}")
        yolo26_by_image.update(predictions)
        yolo26_output_audit[fold] = audit
        preflight["folds"][fold - 1]["yolo26_raw_prediction_count"] = audit["raw_prediction_count_after_model_nms"]
        preflight["folds"][fold - 1]["yolo26_raw_prediction_file"] = audit["raw_prediction_file"]
        preflight["folds"][fold - 1]["yolo26_raw_prediction_file_sha256"] = audit["raw_prediction_file_sha256"]

    if set(yolo26_by_image) != selected_ids:
        raise RuntimeError("YOLO26m pooled prediction coverage is not identical to the selected Phase 0 OOF records")
    selected_yolo11 = {image_id: yolo11_by_image[image_id] for image_id in sorted(selected_ids)}
    selected_yolo26 = {image_id: yolo26_by_image[image_id] for image_id in sorted(selected_ids)}
    selected_targets = {image_id: targets_by_image[image_id] for image_id in sorted(selected_ids)}
    yolo11_score = score_with_f1(selected_yolo11, selected_targets, args.nms_iou)
    yolo26_score = score_with_f1(selected_yolo26, selected_targets, args.nms_iou)
    scores: dict[str, dict[str, Any]] = {"yolo11m": yolo11_score, "yolo26m": yolo26_score}
    write_json(output_root / "metrics" / "yolo11m.json", yolo11_score)
    write_json(output_root / "metrics" / "yolo26m.json", yolo26_score)

    fusion_counts: dict[float, list[dict[str, Any]]] = {}
    fusion_composition: dict[float, Counter[str]] = {}
    fusion_scores: dict[float, dict[str, Any]] = {}
    random_audit_ids = random.Random(args.seed).sample(sorted(selected_ids), min(5, len(selected_ids)))
    random_audit: dict[str, Any] = {
        "seed": args.seed,
        "sample_image_ids": random_audit_ids,
        "images": {},
        "assertion": "Each sampled image contains only YOLO11m and YOLO26m predictions from its own OOF Fold.",
    }
    for image_id in random_audit_ids:
        fold = int(image_id.split("/", 1)[0].removeprefix("fold"))
        random_audit["images"][image_id] = {
            "fold": fold,
            "yolo11m_source_fold": fold,
            "yolo26m_source_fold": fold,
            "yolo11m_input_boxes": int(len(selected_yolo11[image_id])),
            "yolo26m_input_boxes": int(len(selected_yolo26[image_id])),
        }
    for wbf_iou in args.wbf_ious:
        fused_by_image: dict[str, np.ndarray] = {}
        counts: list[dict[str, Any]] = []
        composition: Counter[str] = Counter()
        fusion_path = output_root / "fusion_predictions" / f"wbf_iou_{wbf_iou:.2f}.jsonl"
        fusion_path.parent.mkdir(parents=True, exist_ok=True)
        with fusion_path.open("x", encoding="utf-8") as handle:
            for image_id in sorted(selected_ids):
                fused, source_composition = weighted_box_fusion_two_sources(
                    selected_yolo11[image_id], selected_yolo26[image_id], wbf_iou
                )
                fused_by_image[image_id] = fused
                composition.update(source_composition)
                record = next(record for record in fold_records[int(image_id[4:6])] if record.image_id == image_id)
                emit_raw_record(handle, record, record.source_image, fused)
                counts.append(
                    {
                        "image_id": image_id,
                        "fold": record.fold,
                        "yolo11m_input_boxes": len(selected_yolo11[image_id]),
                        "yolo26m_input_boxes": len(selected_yolo26[image_id]),
                        "combined_input_boxes": len(selected_yolo11[image_id]) + len(selected_yolo26[image_id]),
                        "wbf_output_boxes": len(fused),
                        "fused_both_sources": source_composition["both"],
                        "fused_yolo11m_only": source_composition["yolo11m_only"],
                        "fused_yolo26m_only": source_composition["yolo26m_only"],
                    }
                )
                if image_id in random_audit["images"]:
                    random_audit["images"][image_id][f"wbf_iou_{wbf_iou:.2f}_output_boxes"] = int(len(fused))
                    random_audit["images"][image_id][f"wbf_iou_{wbf_iou:.2f}_source_composition"] = dict(source_composition)
        write_csv(output_root / "fusion_counts" / f"wbf_iou_{wbf_iou:.2f}.csv", counts)
        score = score_with_f1(fused_by_image, selected_targets, args.nms_iou)
        score["wbf"] = {
            "iou": wbf_iou,
            "input_prediction_count": int(sum(row["combined_input_boxes"] for row in counts)),
            "source_composition": dict(composition),
            "fusion_prediction_file": str(fusion_path),
            "fusion_prediction_file_sha256": sha256_file(fusion_path),
        }
        fusion_scores[wbf_iou] = score
        fusion_counts[wbf_iou] = counts
        fusion_composition[wbf_iou] = composition
        variant = f"wbf_iou_{wbf_iou:.2f}"
        scores[variant] = score
        write_json(output_root / "metrics" / f"{variant}.json", score)

    summary_rows = [
        summary_row("yolo11m", yolo11_score, None, None),
        summary_row("yolo26m", yolo26_score, None, None),
    ]
    per_class_metric_rows = [*per_class_rows("yolo11m", yolo11_score, None), *per_class_rows("yolo26m", yolo26_score, None)]
    for wbf_iou in args.wbf_ious:
        score = fusion_scores[wbf_iou]
        summary_rows.append(
            summary_row(
                f"wbf_iou_{wbf_iou:.2f}",
                score,
                wbf_iou,
                int(sum(row["combined_input_boxes"] for row in fusion_counts[wbf_iou])),
            )
        )
        per_class_metric_rows.extend(per_class_rows(f"wbf_iou_{wbf_iou:.2f}", score, wbf_iou))
    write_csv(output_root / "metrics_summary.csv", summary_rows)
    write_csv(output_root / "per_class_metrics.csv", per_class_metric_rows)
    gate = determine_gate(yolo11_score, yolo26_score, fusion_scores, partial)
    write_json(output_root / "gate_decision.json", gate)
    write_json(output_root / "random_pair_audit.json", random_audit)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available_but_unused": torch.cuda.is_available(),
        "ultralytics": __import__("ultralytics").__version__,
        "image_dimension_probe": "PIL header only during preflight; OpenCV is not used",
        "cpu_threads": args.cpu_threads,
        "device": "cpu",
        "ensemble_boxes_available": False,
    }
    write_json(output_root / "environment.json", environment)
    preflight["pooled_images"] = len(selected_ids)
    preflight["output"] = {
        "yolo26_fold_audit": yolo26_output_audit,
        "total_yolo11_raw_prediction_count": int(sum(len(rows) for rows in selected_yolo11.values())),
        "total_yolo26_raw_prediction_count": int(sum(len(rows) for rows in selected_yolo26.values())),
        "random_pair_audit_file": "random_pair_audit.json",
    }
    write_json(output_root / "preflight_and_output_audit.json", preflight)
    write_report(output_root / "report.md", preflight, summary_rows, scores, gate, fusion_composition)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "images": len(selected_ids),
                "gate_status": gate["status"],
                "selected_wbf_iou": gate["selected_wbf_iou"],
                "report": str(output_root / "report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
