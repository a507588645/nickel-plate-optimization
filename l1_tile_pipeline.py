#!/usr/bin/env python3
"""Build and evaluate a development-only mixed full-image/tile L1 experiment.

The script never modifies source images or source labels.  It creates a new
runtime view containing symlinked full images plus generated tiles, then maps
tiled validation predictions back to each original image before scoring.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DATA_VERSION = "oofconsensusv1"
CLASS_NAMES = (
    "wuzi",
    "waterstain",
    "yellowstain",
    "nodule",
    "airhole",
    "bubble",
    "bump",
    "fawu",
    "overfire",
    "NiOH",
    "greenstain",
    "suanwuran",
    "quejiao",
    "watermark",
    "youpao",
    "heibian",
    "mijixiaokeng",
    "fanxitong",
)
DEFAULT_FOCUS_NAMES = ("wuzi", "waterstain", "airhole", "bubble", "greenstain", "watermark")
IOU_THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))
TILED_EVAL_PROFILES = ("legacy", "perclass_nms", "wbf", "wbf_edge")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a development-only L1 tile training view and original-image tiled evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("preflight", "train", "evaluate"), required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--config-manifest", type=Path, help="Optional manifest path when config-root is an already restored view.")
    parser.add_argument("--image-dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--baseline-weights", type=Path)
    parser.add_argument("--candidate-weights", type=Path, help="Interrupted or completed tile-model weight for evaluate mode.")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument("--min-visible-fraction", type=float, default=0.50)
    parser.add_argument("--focus-classes", default=",".join(DEFAULT_FOCUS_NAMES))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--predict-batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument(
        "--tiled-eval-profiles",
        default="legacy",
        help="Comma-separated tiled post-processing profiles: legacy, perclass_nms, wbf, wbf_edge.",
    )
    parser.add_argument(
        "--primary-tiled-profile",
        default="legacy",
        help="One requested tiled profile to expose as baseline_tiled/tile_model_tiled in the comparison summary.",
    )
    parser.add_argument(
        "--per-class-conf-json",
        type=Path,
        help="Optional JSON profile with default_confidence and class_thresholds keyed by CLASS_NAMES.",
    )
    parser.add_argument("--wbf-iou", type=float, default=0.25)
    parser.add_argument(
        "--edge-margin",
        type=int,
        default=16,
        help="Pixels from an internal crop edge considered truncated by the wbf_edge profile.",
    )
    parser.add_argument(
        "--save-tile-details",
        action="store_true",
        help="Write gzip-compressed raw per-tile predictions and post-processing memberships for attribution review.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.tile_size < 128:
        raise ValueError("--tile-size must be at least 128")
    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be in [0, 1)")
    if args.negative_ratio < 0:
        raise ValueError("--negative-ratio must be non-negative")
    if not 0 < args.min_visible_fraction <= 1:
        raise ValueError("--min-visible-fraction must be in (0, 1]")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.predict_batch <= 0:
        raise ValueError("epochs, imgsz, batch, and predict-batch must be positive")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative")
    if not 0 < args.nms_iou <= 1:
        raise ValueError("--nms-iou must be in (0, 1]")
    if not 0 < args.wbf_iou <= 1:
        raise ValueError("--wbf-iou must be in (0, 1]")
    if args.edge_margin < 0:
        raise ValueError("--edge-margin must be non-negative")
    profiles = tuple(profile.strip() for profile in args.tiled_eval_profiles.split(",") if profile.strip())
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("--tiled-eval-profiles must contain one or more unique profile names")
    unknown_profiles = sorted(set(profiles) - set(TILED_EVAL_PROFILES))
    if unknown_profiles:
        raise ValueError(f"Unsupported tiled evaluation profile(s): {unknown_profiles}")
    if args.primary_tiled_profile not in profiles:
        raise ValueError("--primary-tiled-profile must be included in --tiled-eval-profiles")
    args.tiled_eval_profiles = profiles
    args.per_class_confidence = load_per_class_confidence(args.per_class_conf_json)
    if args.mode in {"train", "evaluate"} and args.baseline_weights is None:
        raise ValueError("--baseline-weights is required for the fixed Fold baseline comparison")
    if args.mode == "evaluate" and args.candidate_weights is None:
        raise ValueError("--candidate-weights is required in evaluate mode")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_per_class_confidence(path: Path | None) -> dict[str, Any]:
    """Load the optional research profile while keeping legacy inference unchanged."""
    if path is None:
        return {
            "source": None,
            "default_confidence": 0.25,
            "class_thresholds": {},
            "metadata": {},
        }
    if not path.is_file():
        raise FileNotFoundError(f"Per-class confidence profile not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Per-class confidence profile must be a JSON object")
    default_confidence = payload.get("default_confidence", 0.25)
    class_thresholds = payload.get("class_thresholds")
    if class_thresholds is None:
        class_thresholds = {key: value for key, value in payload.items() if key in CLASS_NAMES}
    if not isinstance(default_confidence, (int, float)) or not 0 < float(default_confidence) <= 1:
        raise ValueError("Per-class confidence default_confidence must be in (0, 1]")
    if not isinstance(class_thresholds, dict):
        raise ValueError("Per-class confidence class_thresholds must be a JSON object")
    unknown_names = sorted(set(class_thresholds) - set(CLASS_NAMES))
    if unknown_names:
        raise ValueError(f"Per-class confidence profile has unknown classes: {unknown_names}")
    normalized: dict[str, float] = {}
    for class_name, value in class_thresholds.items():
        if not isinstance(value, (int, float)) or not 0 < float(value) <= 1:
            raise ValueError(f"Invalid confidence threshold for {class_name!r}: {value!r}")
        normalized[class_name] = float(value)
    return {
        "source": str(path.resolve()),
        "default_confidence": float(default_confidence),
        "class_thresholds": normalized,
        "metadata": {
            key: value
            for key, value in payload.items()
            if key not in {"default_confidence", "class_thresholds"}
        },
    }


def read_text_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    records: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label field count: {path}:{line_number}")
        try:
            class_id = int(parts[0])
            values = tuple(float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO label value: {path}:{line_number}") from exc
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"Invalid class id: {path}:{line_number}")
        xc, yc, width, height = values
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Invalid normalized YOLO box: {path}:{line_number}")
        records.append((class_id, xc, yc, width, height))
    return records


def normalize_path(path: Path) -> str:
    return path.resolve().as_posix()


def parse_focus_ids(value: str) -> set[int]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("--focus-classes cannot be empty")
    unknown = sorted(set(names) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unknown focus classes: {unknown}")
    return {CLASS_NAMES.index(name) for name in names}


def load_context(args: argparse.Namespace) -> dict[str, Any]:
    config_root = args.config_root.resolve()
    image_root = args.image_dataset_root.resolve()
    manifest_path = args.config_manifest.resolve() if args.config_manifest else config_root / "config_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing config manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("data_version") != DATA_VERSION or manifest.get("development_only") is not True:
        raise ValueError("The config payload is not the required development-only OOF-consensus version")
    classes = manifest.get("classes")
    if classes != list(CLASS_NAMES):
        raise ValueError("Class names or class order do not match the fixed RYW18 contract")
    images = manifest.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Config manifest has no image mapping")
    source_by_anchor: dict[str, str] = {}
    for item in images:
        if not isinstance(item, dict):
            raise ValueError("Invalid image mapping entry")
        anchor_name = item.get("anchor_name")
        source_relative = item.get("source_relative")
        if not isinstance(anchor_name, str) or not isinstance(source_relative, str):
            raise ValueError("Invalid image mapping fields")
        if anchor_name in source_by_anchor:
            raise ValueError(f"Duplicate anchor name in config manifest: {anchor_name}")
        source_by_anchor[anchor_name] = source_relative
    split_root = config_root / "splits" / "folds" / f"fold_{args.fold:02d}"
    train_list = split_root / "train.txt"
    val_list = split_root / "val.txt"
    if not train_list.is_file() or not val_list.is_file():
        raise FileNotFoundError(f"Missing Fold {args.fold} train/val lists")
    test_list = split_root / "test.txt"
    if test_list.exists():
        raise ValueError("Development-only tile experiment must not contain a test.txt list")
    return {
        "config_root": config_root,
        "manifest_path": manifest_path,
        "image_root": image_root,
        "manifest": manifest,
        "source_by_anchor": source_by_anchor,
        "split_root": split_root,
    }


def pixel_boxes(labels: list[tuple[int, float, float, float, float]], width: int, height: int) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    for class_id, xc, yc, box_width, box_height in labels:
        x1 = max(0.0, (xc - box_width / 2) * width)
        y1 = max(0.0, (yc - box_height / 2) * height)
        x2 = min(float(width), (xc + box_width / 2) * width)
        y2 = min(float(height), (yc + box_height / 2) * height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Normalized YOLO label became an empty pixel box")
        output.append({"class_id": class_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return output


def read_records(context: dict[str, Any], split: str) -> list[dict[str, Any]]:
    split_path = context["split_root"] / f"{split}.txt"
    records: list[dict[str, Any]] = []
    for image_relative in read_text_lines(split_path):
        anchor_name = Path(image_relative).name
        source_relative = context["source_by_anchor"].get(anchor_name)
        if source_relative is None:
            raise ValueError(f"Split image is absent from the manifest mapping: {image_relative}")
        source_image = context["image_root"] / Path(source_relative)
        label_path = context["config_root"] / "labels" / f"{Path(anchor_name).stem}.txt"
        if not source_image.is_file():
            raise FileNotFoundError(f"Missing source image: {source_image}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing OOF candidate label: {label_path}")
        with Image.open(source_image) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid source image size: {source_image}")
        labels = read_yolo_labels(label_path)
        records.append(
            {
                "anchor_name": anchor_name,
                "anchor_stem": Path(anchor_name).stem,
                "image_relative": image_relative,
                "source_relative": source_relative,
                "source_image": source_image,
                "label_path": label_path,
                "width": width,
                "height": height,
                "labels": labels,
                "pixel_boxes": pixel_boxes(labels, width, height),
            }
        )
    return records


def grid_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    last = length - tile_size
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


def make_grid(width: int, height: int, tile_size: int, overlap: float) -> list[dict[str, int]]:
    tile_width = min(tile_size, width)
    tile_height = min(tile_size, height)
    return [
        {"x": x, "y": y, "width": tile_width, "height": tile_height}
        for y in grid_origins(height, tile_height, overlap)
        for x in grid_origins(width, tile_width, overlap)
    ]


def clipped_box(box: dict[str, float | int], tile: dict[str, int]) -> tuple[float, float, float, float, float]:
    x1 = max(float(tile["x"]), float(box["x1"]))
    y1 = max(float(tile["y"]), float(box["y1"]))
    x2 = min(float(tile["x"] + tile["width"]), float(box["x2"]))
    y2 = min(float(tile["y"] + tile["height"]), float(box["y2"]))
    visible_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    original_area = (float(box["x2"]) - float(box["x1"])) * (float(box["y2"]) - float(box["y1"]))
    visible_fraction = visible_area / original_area if original_area > 0 else 0.0
    return x1, y1, x2, y2, visible_fraction


def tile_margin(box: dict[str, float | int], tile: dict[str, int]) -> float:
    center_x = (float(box["x1"]) + float(box["x2"])) / 2
    center_y = (float(box["y1"]) + float(box["y2"])) / 2
    return min(
        center_x - float(tile["x"]),
        float(tile["x"] + tile["width"]) - center_x,
        center_y - float(tile["y"]),
        float(tile["y"] + tile["height"]) - center_y,
    )


def select_tiles_for_record(record: dict[str, Any], args: argparse.Namespace, focus_ids: set[int]) -> dict[str, Any]:
    grid = make_grid(record["width"], record["height"], args.tile_size, args.overlap)
    selected: set[int] = set()
    focus_box_counts: Counter[int] = Counter()
    uncovered_focus_counts: Counter[int] = Counter()
    for box in record["pixel_boxes"]:
        class_id = int(box["class_id"])
        if class_id not in focus_ids:
            continue
        candidates: list[tuple[float, float, int]] = []
        for index, tile in enumerate(grid):
            _, _, _, _, visible_fraction = clipped_box(box, tile)
            if visible_fraction >= args.min_visible_fraction:
                candidates.append((visible_fraction, tile_margin(box, tile), index))
        if not candidates:
            # Keep oversized or severely truncated objects in their full-image
            # training sample instead of emitting an unreliable partial tile label.
            uncovered_focus_counts[class_id] += 1
            continue
        _, _, best_index = max(candidates)
        selected.add(best_index)
        focus_box_counts[class_id] += 1

    negative_candidates: list[int] = []
    for index, tile in enumerate(grid):
        has_visible_box = any(
            clipped_box(box, tile)[4] >= args.min_visible_fraction for box in record["pixel_boxes"]
        )
        if not has_visible_box:
            negative_candidates.append(index)
    return {
        "record": record,
        "grid": grid,
        "selected": selected,
        "negative_candidates": negative_candidates,
        "focus_box_counts": focus_box_counts,
        "uncovered_focus_counts": uncovered_focus_counts,
    }


def tile_label_lines(record: dict[str, Any], tile: dict[str, int], min_visible_fraction: float) -> list[str]:
    lines: list[str] = []
    for box in record["pixel_boxes"]:
        x1, y1, x2, y2, visible_fraction = clipped_box(box, tile)
        if visible_fraction < min_visible_fraction or x2 <= x1 or y2 <= y1:
            continue
        local_x1 = x1 - float(tile["x"])
        local_y1 = y1 - float(tile["y"])
        local_x2 = x2 - float(tile["x"])
        local_y2 = y2 - float(tile["y"])
        x_center = (local_x1 + local_x2) / (2 * tile["width"])
        y_center = (local_y1 + local_y2) / (2 * tile["height"])
        box_width = (local_x2 - local_x1) / tile["width"]
        box_height = (local_y2 - local_y1) / tile["height"]
        lines.append(
            f"{int(box['class_id'])} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
        )
    return lines


def plan_dataset(train_records: list[dict[str, Any]], val_records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    focus_ids = parse_focus_ids(args.focus_classes)
    train_plans = [select_tiles_for_record(record, args, focus_ids) for record in train_records]
    val_grids = [make_grid(record["width"], record["height"], args.tile_size, args.overlap) for record in val_records]
    flat_negatives = [
        (plan_index, tile_index)
        for plan_index, plan in enumerate(train_plans)
        for tile_index in plan["negative_candidates"]
    ]
    selected_tile_count = sum(len(plan["selected"]) for plan in train_plans)
    negative_target = min(len(flat_negatives), math.floor(selected_tile_count * args.negative_ratio))
    rng = random.Random(args.seed)
    sampled_negatives = set(rng.sample(flat_negatives, negative_target)) if negative_target else set()
    source_class_counts: Counter[int] = Counter()
    for record in train_records + val_records:
        source_class_counts.update(int(box["class_id"]) for box in record["pixel_boxes"])
    focus_counts: Counter[int] = Counter()
    uncovered_focus_counts: Counter[int] = Counter()
    for plan in train_plans:
        focus_counts.update(plan["focus_box_counts"])
        uncovered_focus_counts.update(plan["uncovered_focus_counts"])
    return {
        "focus_ids": focus_ids,
        "train_plans": train_plans,
        "val_grids": val_grids,
        "sampled_negatives": sampled_negatives,
        "source_class_counts": source_class_counts,
        "focus_counts": focus_counts,
        "uncovered_focus_counts": uncovered_focus_counts,
    }


def serializable_plan_summary(
    context: dict[str, Any], train_records: list[dict[str, Any]], val_records: list[dict[str, Any]], plan: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    selected_tile_count = sum(len(item["selected"]) for item in plan["train_plans"])
    train_grid_tile_count = sum(len(item["grid"]) for item in plan["train_plans"])
    val_grid_tile_count = sum(len(grid) for grid in plan["val_grids"])
    return {
        "schema_version": 1,
        "data_version": DATA_VERSION,
        "development_only": True,
        "fold": args.fold,
        "source_config_manifest_sha256": sha256_file(context["manifest_path"]),
        "source_split_hashes": {
            "train": sha256_file(context["split_root"] / "train.txt"),
            "val": sha256_file(context["split_root"] / "val.txt"),
        },
        "tile": {
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "negative_ratio": args.negative_ratio,
            "min_visible_fraction": args.min_visible_fraction,
            "focus_classes": [CLASS_NAMES[index] for index in sorted(plan["focus_ids"])],
            "selection": "one best-covered tile per focus box; labels retain boxes with visible area >= min_visible_fraction",
        },
        "split_counts": {
            "full_train_images": len(train_records),
            "full_val_images": len(val_records),
            "all_train_grid_tiles": train_grid_tile_count,
            "selected_focus_tiles": selected_tile_count,
            "sampled_negative_tiles": len(plan["sampled_negatives"]),
            "mixed_training_images": len(train_records) + selected_tile_count + len(plan["sampled_negatives"]),
            "val_inference_tiles": val_grid_tile_count,
            "test_frozen": None,
        },
        "focus_source_box_counts": {
            CLASS_NAMES[index]: plan["focus_counts"].get(index, 0) for index in sorted(plan["focus_ids"])
        },
        "focus_boxes_kept_full_image_only": {
            CLASS_NAMES[index]: plan["uncovered_focus_counts"].get(index, 0) for index in sorted(plan["focus_ids"])
        },
        "source_box_counts": {
            CLASS_NAMES[index]: plan["source_class_counts"].get(index, 0) for index in range(len(CLASS_NAMES))
        },
    }


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def write_data_yaml(dataset_root: Path) -> Path:
    lines = [
        f'path: "{normalize_path(dataset_root)}"',
        'train: "train.txt"',
        'val: "val.txt"',
        "nc: 18",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
    ]
    path = dataset_root / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_tile_image_and_labels(
    source: np.ndarray, record: dict[str, Any], tile: dict[str, int], image_path: Path, label_path: Path, args: argparse.Namespace
) -> int:
    import cv2
    crop = source[tile["y"] : tile["y"] + tile["height"], tile["x"] : tile["x"] + tile["width"]]
    if crop.shape[0] != tile["height"] or crop.shape[1] != tile["width"]:
        raise ValueError(f"Tile crop shape mismatch for {record['anchor_name']}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Failed to write tile image: {image_path}")
    label_lines = tile_label_lines(record, tile, args.min_visible_fraction)
    label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    return len(label_lines)


def prepare_runtime_view(
    output_root: Path,
    context: dict[str, Any],
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    plan: dict[str, Any],
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Path | int]:
    import cv2

    if not output_root.is_dir():
        raise FileNotFoundError(f"Tile runtime root was not initialized: {output_root}")
    if any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to write tile runtime into a non-empty output root: {output_root}")
    dataset_root = output_root / "dataset_view"
    dataset_root.mkdir(parents=True, exist_ok=False)
    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for split, records, target_paths in (("train", train_records, train_paths), ("val", val_records, val_paths)):
        for record in records:
            image_path = dataset_root / "images" / f"full_{split}" / record["anchor_name"]
            label_path = dataset_root / "labels" / f"full_{split}" / f"{record['anchor_stem']}.txt"
            link_or_copy(record["source_image"], image_path)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["label_path"], label_path)
            target_paths.append(image_path)

    tile_label_count = 0
    tile_count = 0
    for plan_index, item in enumerate(plan["train_plans"]):
        selected_indices = set(item["selected"])
        selected_indices.update(tile_index for current_plan, tile_index in plan["sampled_negatives"] if current_plan == plan_index)
        record = item["record"]
        source = cv2.imread(str(record["source_image"]), cv2.IMREAD_COLOR)
        if source is None:
            raise RuntimeError(f"OpenCV failed to read source image: {record['source_image']}")
        for tile_index in sorted(selected_indices):
            tile = item["grid"][tile_index]
            stem = f"{record['anchor_stem']}__x{tile['x']:04d}_y{tile['y']:04d}"
            image_path = dataset_root / "images" / "tiles" / f"{stem}.jpg"
            label_path = dataset_root / "labels" / "tiles" / f"{stem}.txt"
            tile_label_count += write_tile_image_and_labels(source, record, tile, image_path, label_path, args)
            train_paths.append(image_path)
            tile_count += 1

    data_yaml = write_data_yaml(dataset_root)
    (dataset_root / "train.txt").write_text(
        "\n".join(normalize_path(path) for path in train_paths) + "\n", encoding="utf-8"
    )
    (dataset_root / "val.txt").write_text(
        "\n".join(normalize_path(path) for path in val_paths) + "\n", encoding="utf-8"
    )
    summary = {**summary, "generated": {"tile_images": tile_count, "tile_label_boxes": tile_label_count}}
    write_json(dataset_root / "tile_manifest.json", summary)
    return {
        "dataset_root": dataset_root,
        "data_yaml": data_yaml,
        "train_txt": dataset_root / "train.txt",
        "val_txt": dataset_root / "val.txt",
        "tile_manifest": dataset_root / "tile_manifest.json",
    }


def prediction_to_rows(result: Any, offset_x: int = 0, offset_y: int = 0) -> list[list[float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    coordinates = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    return [
        [
            float(x1 + offset_x),
            float(y1 + offset_y),
            float(x2 + offset_x),
            float(y2 + offset_y),
            float(confidence),
            float(class_id),
        ]
        for (x1, y1, x2, y2), confidence, class_id in zip(coordinates, confidences, classes, strict=True)
    ]


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float64)
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area_one = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_many = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_one + area_many - inter, 1e-12)


def class_aware_nms(rows: list[list[float]], iou_threshold: float) -> np.ndarray:
    if not rows:
        return np.empty((0, 6), dtype=np.float64)
    predictions = np.asarray(rows, dtype=np.float64)
    kept: list[np.ndarray] = []
    for class_id in sorted({int(value) for value in predictions[:, 5]}):
        remaining = predictions[predictions[:, 5] == class_id]
        remaining = remaining[np.argsort(-remaining[:, 4])]
        while len(remaining):
            current = remaining[0]
            kept.append(current)
            if len(remaining) == 1:
                break
            overlaps = box_iou_one_to_many(current[:4], remaining[1:, :4])
            remaining = remaining[1:][overlaps <= iou_threshold]
    return np.asarray(kept, dtype=np.float64)


def rows_to_array(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 6), dtype=np.float64)


def tile_internal_edge_sides(
    row: list[float], tile: dict[str, int], image_width: int, image_height: int, edge_margin: int
) -> list[str]:
    """Identify crop boundaries that have a neighbouring crop and touch a prediction."""
    x1, y1, x2, y2 = (float(value) for value in row[:4])
    sides: list[str] = []
    if tile["x"] > 0 and x1 - tile["x"] <= edge_margin:
        sides.append("left")
    if tile["x"] + tile["width"] < image_width and tile["x"] + tile["width"] - x2 <= edge_margin:
        sides.append("right")
    if tile["y"] > 0 and y1 - tile["y"] <= edge_margin:
        sides.append("top")
    if tile["y"] + tile["height"] < image_height and tile["y"] + tile["height"] - y2 <= edge_margin:
        sides.append("bottom")
    return sides


def predict_tiled_raw(
    model: Any, image: np.ndarray, grid: list[dict[str, int]], args: argparse.Namespace
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Predict every crop once and retain provenance before cross-tile merging."""
    rows: list[list[float]] = []
    raw_details: list[dict[str, Any]] = []
    image_height, image_width = image.shape[:2]
    for start in range(0, len(grid), args.predict_batch):
        batch_tiles = grid[start : start + args.predict_batch]
        crops = [
            image[tile["y"] : tile["y"] + tile["height"], tile["x"] : tile["x"] + tile["width"]]
            for tile in batch_tiles
        ]
        results = model.predict(
            source=crops,
            imgsz=args.imgsz,
            conf=0.001,
            iou=args.nms_iou,
            max_det=300,
            device=args.device,
            verbose=False,
        )
        if len(results) != len(batch_tiles):
            raise RuntimeError("Tiled prediction result count mismatch")
        for offset, (result, tile) in enumerate(zip(results, batch_tiles, strict=True)):
            tile_index = start + offset
            for row in prediction_to_rows(result, tile["x"], tile["y"]):
                raw_id = len(rows)
                class_id = int(row[5])
                rows.append(row)
                raw_details.append(
                    {
                        "raw_id": raw_id,
                        "tile_index": tile_index,
                        "xyxy": [float(value) for value in row[:4]],
                        "confidence": float(row[4]),
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "touches_internal_edges": tile_internal_edge_sides(
                            row, tile, image_width, image_height, args.edge_margin
                        ),
                    }
                )
    return rows_to_array(rows), raw_details


def per_class_confidence_threshold(class_id: int, args: argparse.Namespace) -> float:
    class_name = CLASS_NAMES[class_id]
    return float(args.per_class_confidence["class_thresholds"].get(class_name, args.per_class_confidence["default_confidence"]))


def confidence_filtered_indices(predictions: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    indices = [
        index
        for index, row in enumerate(predictions)
        if float(row[4]) >= per_class_confidence_threshold(int(row[5]), args)
    ]
    return np.asarray(indices, dtype=np.int64)


def class_aware_nms_indices(
    predictions: np.ndarray, candidate_indices: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the same class-wise NMS result as legacy mode plus raw row provenance."""
    if len(candidate_indices) == 0:
        return np.empty((0, 6), dtype=np.float64), np.empty((0,), dtype=np.int64)
    kept_indices: list[int] = []
    for class_id in sorted({int(predictions[index, 5]) for index in candidate_indices}):
        remaining = candidate_indices[predictions[candidate_indices, 5] == class_id]
        remaining = remaining[np.argsort(-predictions[remaining, 4])]
        while len(remaining):
            current_index = int(remaining[0])
            kept_indices.append(current_index)
            if len(remaining) == 1:
                break
            overlaps = box_iou_one_to_many(predictions[current_index, :4], predictions[remaining[1:], :4])
            remaining = remaining[1:][overlaps <= iou_threshold]
    kept = np.asarray(kept_indices, dtype=np.int64)
    return predictions[kept], kept


def weighted_box_fusion(
    predictions: np.ndarray, candidate_indices: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fuse same-class cross-crop candidates and retain every raw member id.

    Coordinates are confidence-weighted; the maximum member confidence is kept
    so cross-crop duplication does not lower the calibrated detection score.
    """
    if len(candidate_indices) == 0:
        return np.empty((0, 6), dtype=np.float64), []
    completed: list[dict[str, Any]] = []
    for class_id in sorted({int(predictions[index, 5]) for index in candidate_indices}):
        class_indices = candidate_indices[predictions[candidate_indices, 5] == class_id]
        class_indices = class_indices[np.argsort(-predictions[class_indices, 4])]
        clusters: list[dict[str, Any]] = []
        for raw_index in class_indices:
            raw_index = int(raw_index)
            row = predictions[raw_index]
            if clusters:
                cluster_boxes = np.asarray([cluster["xyxy"] for cluster in clusters], dtype=np.float64)
                overlaps = box_iou_one_to_many(row[:4], cluster_boxes)
                best_cluster = int(np.argmax(overlaps))
                if overlaps[best_cluster] >= iou_threshold:
                    cluster = clusters[best_cluster]
                    weight = float(row[4])
                    cluster["weighted_sum"] += row[:4] * weight
                    cluster["weight_sum"] += weight
                    cluster["xyxy"] = cluster["weighted_sum"] / cluster["weight_sum"]
                    cluster["confidence"] = max(float(cluster["confidence"]), weight)
                    cluster["raw_member_ids"].append(raw_index)
                    continue
            weight = float(row[4])
            clusters.append(
                {
                    "class_id": class_id,
                    "xyxy": row[:4].copy(),
                    "weighted_sum": row[:4] * weight,
                    "weight_sum": weight,
                    "confidence": weight,
                    "raw_member_ids": [raw_index],
                }
            )
        completed.extend(clusters)
    fused_rows = [
        [*map(float, cluster["xyxy"]), float(cluster["confidence"]), float(cluster["class_id"])]
        for cluster in completed
    ]
    return rows_to_array(fused_rows), completed


def wbf_detail_rows(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "class_id": int(cluster["class_id"]),
            "class_name": CLASS_NAMES[int(cluster["class_id"])],
            "xyxy": [float(value) for value in cluster["xyxy"]],
            "confidence": float(cluster["confidence"]),
            "raw_member_ids": [int(value) for value in cluster["raw_member_ids"]],
        }
        for cluster in clusters
    ]


def cross_tile_duplicate_diagnostic(
    predictions: np.ndarray, raw_details: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    """Measure high-confidence cross-tile duplicate pressure without scanning low-confidence noise."""
    probe_indices = np.flatnonzero(predictions[:, 4] >= 0.25).astype(np.int64)
    _, clusters = weighted_box_fusion(predictions, probe_indices, args.wbf_iou)
    cross_tile_clusters = [
        cluster
        for cluster in clusters
        if len({int(raw_details[index]["tile_index"]) for index in cluster["raw_member_ids"]}) > 1
    ]
    member_count = sum(len(cluster["raw_member_ids"]) for cluster in cross_tile_clusters)
    excess_count = sum(len(cluster["raw_member_ids"]) - 1 for cluster in cross_tile_clusters)
    return {
        "probe_confidence_floor": 0.25,
        "probe_iou": args.wbf_iou,
        "probe_candidate_count": int(len(probe_indices)),
        "cross_tile_cluster_count": len(cross_tile_clusters),
        "cross_tile_member_count": member_count,
        "cross_tile_duplicate_excess_count": excess_count,
    }


def postprocess_tiled_predictions(
    predictions: np.ndarray, raw_details: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Apply requested profiles to one raw tiled prediction stream."""
    raw_indices = np.arange(len(predictions), dtype=np.int64)
    threshold_indices = confidence_filtered_indices(predictions, args)
    edge_safe_indices = np.asarray(
        [index for index in threshold_indices if not raw_details[int(index)]["touches_internal_edges"]], dtype=np.int64
    )
    results: dict[str, dict[str, Any]] = {}
    for profile in args.tiled_eval_profiles:
        if profile == "legacy":
            candidate_indices = raw_indices
            output, kept_indices = class_aware_nms_indices(predictions, candidate_indices, args.nms_iou)
            detail = {
                "merger": "class_aware_nms",
                "iou": args.nms_iou,
                "raw_input_count": int(len(raw_indices)),
                "after_confidence_filter_count": int(len(raw_indices)),
                "after_edge_filter_count": int(len(raw_indices)),
                "output_count": int(len(output)),
                "output_raw_ids": [int(index) for index in kept_indices],
            }
        elif profile == "perclass_nms":
            candidate_indices = threshold_indices
            output, kept_indices = class_aware_nms_indices(predictions, candidate_indices, args.nms_iou)
            detail = {
                "merger": "class_aware_nms",
                "iou": args.nms_iou,
                "raw_input_count": int(len(raw_indices)),
                "after_confidence_filter_count": int(len(threshold_indices)),
                "after_edge_filter_count": int(len(threshold_indices)),
                "output_count": int(len(output)),
                "output_raw_ids": [int(index) for index in kept_indices],
            }
        elif profile in {"wbf", "wbf_edge"}:
            candidate_indices = threshold_indices if profile == "wbf" else edge_safe_indices
            output, clusters = weighted_box_fusion(predictions, candidate_indices, args.wbf_iou)
            detail = {
                "merger": "confidence_weighted_box_fusion_max_confidence",
                "iou": args.wbf_iou,
                "raw_input_count": int(len(raw_indices)),
                "after_confidence_filter_count": int(len(threshold_indices)),
                "after_edge_filter_count": int(len(candidate_indices)),
                "output_count": int(len(output)),
                "fused_outputs": wbf_detail_rows(clusters),
            }
        else:  # validate_args rejects this before inference; retain a local guard for direct callers.
            raise ValueError(f"Unsupported tiled profile: {profile}")
        results[profile] = {"predictions": output, "detail": detail}
    low_confidence_count = int(np.count_nonzero(predictions[:, 4] < 0.25))
    edge_touch_count = sum(bool(detail["touches_internal_edges"]) for detail in raw_details)
    diagnostic = {
        "raw_premerge_count": int(len(predictions)),
        "raw_confidence_below_025_count": low_confidence_count,
        "raw_touches_internal_edge_count": edge_touch_count,
        "cross_tile_duplicate_probe": cross_tile_duplicate_diagnostic(predictions, raw_details, args),
    }
    return results, diagnostic


def predict_tiled(model: Any, image: np.ndarray, grid: list[dict[str, int]], args: argparse.Namespace) -> np.ndarray:
    rows: list[list[float]] = []
    for start in range(0, len(grid), args.predict_batch):
        batch_tiles = grid[start : start + args.predict_batch]
        crops = [
            image[tile["y"] : tile["y"] + tile["height"], tile["x"] : tile["x"] + tile["width"]]
            for tile in batch_tiles
        ]
        results = model.predict(
            source=crops,
            imgsz=args.imgsz,
            conf=0.001,
            iou=args.nms_iou,
            max_det=300,
            device=args.device,
            verbose=False,
        )
        if len(results) != len(batch_tiles):
            raise RuntimeError("Tiled prediction result count mismatch")
        for result, tile in zip(results, batch_tiles, strict=True):
            rows.extend(prediction_to_rows(result, tile["x"], tile["y"]))
    return class_aware_nms(rows, args.nms_iou)


def predict_full(model: Any, image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    results = model.predict(
        source=[image],
        imgsz=args.imgsz,
        conf=0.001,
        iou=args.nms_iou,
        max_det=300,
        device=args.device,
        verbose=False,
    )
    return class_aware_nms(prediction_to_rows(results[0]), args.nms_iou)


def average_precision(tp: np.ndarray, confidence: np.ndarray, target_count: int) -> float:
    if target_count == 0 or len(tp) == 0:
        return 0.0
    order = np.argsort(-confidence)
    tp = tp[order]
    fp = 1.0 - tp
    recall = np.cumsum(tp) / target_count
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp + fp), 1e-12)
    recall_grid = np.linspace(0.0, 1.0, 101)
    # Explicit sampling avoids duplicate-recall endpoint interpolation errors.
    sampled_precision = [
        float(np.max(precision[recall >= recall_level])) if np.any(recall >= recall_level) else 0.0
        for recall_level in recall_grid
    ]
    return float(np.mean(sampled_precision))


def class_matches(
    predictions_by_image: dict[str, np.ndarray],
    ground_truth_by_image: dict[str, np.ndarray],
    class_id: int,
    iou_threshold: float,
    confidence_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    ground_truth: dict[str, np.ndarray] = {}
    candidates: list[tuple[float, str, np.ndarray]] = []
    for image_id, rows in ground_truth_by_image.items():
        selected = rows[rows[:, 0] == class_id, 1:]
        ground_truth[image_id] = selected
    for image_id, rows in predictions_by_image.items():
        selected = rows[(rows[:, 5] == class_id) & (rows[:, 4] >= confidence_threshold)]
        candidates.extend((float(row[4]), image_id, row[:4]) for row in selected)
    candidates.sort(key=lambda item: item[0], reverse=True)
    matched: dict[str, set[int]] = defaultdict(set)
    true_positive: list[float] = []
    confidences: list[float] = []
    for confidence, image_id, box in candidates:
        gt = ground_truth[image_id]
        overlaps = box_iou_one_to_many(box, gt)
        best_index = int(np.argmax(overlaps)) if len(overlaps) else -1
        is_true_positive = best_index >= 0 and overlaps[best_index] >= iou_threshold and best_index not in matched[image_id]
        if is_true_positive:
            matched[image_id].add(best_index)
        true_positive.append(1.0 if is_true_positive else 0.0)
        confidences.append(confidence)
    target_count = sum(len(rows) for rows in ground_truth.values())
    return np.asarray(true_positive), np.asarray(confidences), target_count, len(candidates)


def score_predictions(predictions_by_image: dict[str, np.ndarray], ground_truth_by_image: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    per_class: dict[str, dict[str, float | int | None]] = {}
    ap50_values: list[float] = []
    map_values: list[float] = []
    precision_true_positive = 0.0
    precision_predictions = 0
    recall_targets = 0
    for class_id, class_name in enumerate(CLASS_NAMES):
        ap_by_iou: list[float] = []
        target_count = 0
        for iou_threshold in IOU_THRESHOLDS:
            tp, confidence, target_count, _ = class_matches(
                predictions_by_image, ground_truth_by_image, class_id, iou_threshold
            )
            ap_by_iou.append(average_precision(tp, confidence, target_count))
        tp_025, _, target_count_025, prediction_count_025 = class_matches(
            predictions_by_image, ground_truth_by_image, class_id, 0.50, confidence_threshold=0.25
        )
        true_positive_025 = float(tp_025.sum())
        precision_true_positive += true_positive_025
        precision_predictions += prediction_count_025
        recall_targets += target_count_025
        if target_count:
            ap50_values.append(ap_by_iou[0])
            map_values.extend(ap_by_iou)
            ap50: float | None = ap_by_iou[0]
            map5095: float | None = float(np.mean(ap_by_iou))
        else:
            ap50 = None
            map5095 = None
        per_class[class_name] = {
            "support": target_count,
            "ap50": ap50,
            "ap50_95": map5095,
            "precision_conf025_iou50": true_positive_025 / prediction_count_025 if prediction_count_025 else 0.0,
            "recall_conf025_iou50": true_positive_025 / target_count_025 if target_count_025 else 0.0,
        }
    return {
        "metric_protocol": {
            "scope": "original-image evaluation after mapping tiled predictions back to source coordinates",
            "iou_thresholds": list(IOU_THRESHOLDS),
            "prediction_confidence_floor": 0.001,
            "precision_recall_confidence": 0.25,
            "global_class_aware_nms_iou": args.nms_iou,
            "ap_interpolation": "101-point precision envelope",
        },
        "images": len(ground_truth_by_image),
        "precision_conf025_iou50_micro": precision_true_positive / precision_predictions if precision_predictions else 0.0,
        "recall_conf025_iou50_micro": precision_true_positive / recall_targets if recall_targets else 0.0,
        "mAP50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "mAP50_95": float(np.mean(map_values)) if map_values else 0.0,
        "per_class": per_class,
    }


def ground_truth_for_record(record: dict[str, Any]) -> np.ndarray:
    rows = [
        [float(box["class_id"]), float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]
        for box in record["pixel_boxes"]
    ]
    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 5), dtype=np.float64)


def evaluate_model(model_path: Path, val_records: list[dict[str, Any]], val_grids: list[list[dict[str, int]]], args: argparse.Namespace, mode: str) -> dict[str, Any]:
    from ultralytics import YOLO
    import cv2

    if mode not in {"full", "tiled"}:
        raise ValueError(f"Unsupported evaluation mode: {mode}")
    model = YOLO(str(model_path))
    predictions_by_image: dict[str, np.ndarray] = {}
    ground_truth_by_image: dict[str, np.ndarray] = {}
    prediction_count = 0
    for record, grid in zip(val_records, val_grids, strict=True):
        image = cv2.imread(str(record["source_image"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV failed to read validation image: {record['source_image']}")
        predictions = predict_tiled(model, image, grid, args) if mode == "tiled" else predict_full(model, image, args)
        predictions_by_image[record["anchor_name"]] = predictions
        ground_truth_by_image[record["anchor_name"]] = ground_truth_for_record(record)
        prediction_count += len(predictions)
    scores = score_predictions(predictions_by_image, ground_truth_by_image, args)
    scores.update(
        {
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "inference_mode": mode,
            "post_nms_prediction_count": prediction_count,
        }
    )
    return scores


def tiled_postprocess_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "requested_profiles": list(args.tiled_eval_profiles),
        "primary_profile": args.primary_tiled_profile,
        "legacy_global_class_aware_nms_iou": args.nms_iou,
        "wbf_iou": args.wbf_iou,
        "edge_margin_pixels": args.edge_margin,
        "per_class_confidence": args.per_class_confidence,
    }


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_tiled_profiles(
    model_path: Path,
    model_label: str,
    val_records: list[dict[str, Any]],
    val_grids: list[list[dict[str, int]]],
    args: argparse.Namespace,
    evaluation_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Run crop inference once per image, then score all requested post-processing profiles."""
    from ultralytics import YOLO
    import cv2

    model = YOLO(str(model_path))
    predictions_by_profile: dict[str, dict[str, np.ndarray]] = {
        profile: {} for profile in args.tiled_eval_profiles
    }
    ground_truth_by_image: dict[str, np.ndarray] = {}
    profile_counts: dict[str, dict[str, int]] = {
        profile: {
            "raw_input_count": 0,
            "after_confidence_filter_count": 0,
            "after_edge_filter_count": 0,
            "output_count": 0,
        }
        for profile in args.tiled_eval_profiles
    }
    attribution = {
        "schema_version": 1,
        "model_label": model_label,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "images": 0,
        "raw_premerge_count": 0,
        "raw_confidence_below_025_count": 0,
        "raw_touches_internal_edge_count": 0,
        "cross_tile_duplicate_probe": {
            "probe_confidence_floor": 0.25,
            "probe_iou": args.wbf_iou,
            "probe_candidate_count": 0,
            "cross_tile_cluster_count": 0,
            "cross_tile_member_count": 0,
            "cross_tile_duplicate_excess_count": 0,
        },
        "profile_counts": profile_counts,
        "postprocess_config": tiled_postprocess_config(args),
        "per_box_detail_file": None,
        "per_box_detail_schema": {
            "raw_predictions": "One row per crop-level, model-NMS prediction with tile provenance and internal-edge flags.",
            "postprocess": "NMS outputs retain raw ids; WBF outputs retain every fused raw member id and fused coordinates.",
        },
    }
    detail_path = evaluation_dir / f"{model_label}_raw_tile_predictions.jsonl.gz"
    detail_handle: Any = None
    if args.save_tile_details:
        detail_handle = gzip.open(detail_path, "wt", encoding="utf-8")
        attribution["per_box_detail_file"] = detail_path.name
    try:
        for record, grid in zip(val_records, val_grids, strict=True):
            image = cv2.imread(str(record["source_image"]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"OpenCV failed to read validation image: {record['source_image']}")
            raw_predictions, raw_details = predict_tiled_raw(model, image, grid, args)
            profile_results, image_diagnostic = postprocess_tiled_predictions(raw_predictions, raw_details, args)
            ground_truth_by_image[record["anchor_name"]] = ground_truth_for_record(record)
            attribution["images"] += 1
            attribution["raw_premerge_count"] += int(image_diagnostic["raw_premerge_count"])
            attribution["raw_confidence_below_025_count"] += int(image_diagnostic["raw_confidence_below_025_count"])
            attribution["raw_touches_internal_edge_count"] += int(image_diagnostic["raw_touches_internal_edge_count"])
            duplicate_probe = image_diagnostic["cross_tile_duplicate_probe"]
            for key in (
                "probe_candidate_count",
                "cross_tile_cluster_count",
                "cross_tile_member_count",
                "cross_tile_duplicate_excess_count",
            ):
                attribution["cross_tile_duplicate_probe"][key] += int(duplicate_probe[key])
            for profile, result in profile_results.items():
                predictions_by_profile[profile][record["anchor_name"]] = result["predictions"]
                for key in profile_counts[profile]:
                    profile_counts[profile][key] += int(result["detail"][key])
            if detail_handle is not None:
                detail_handle.write(
                    json.dumps(
                        {
                            "image_id": record["anchor_name"],
                            "source_relative": record["source_relative"],
                            "image_width": int(record["width"]),
                            "image_height": int(record["height"]),
                            "tiles": [{"tile_index": index, **tile} for index, tile in enumerate(grid)],
                            "raw_predictions": raw_details,
                            "diagnostic": image_diagnostic,
                            "postprocess": {profile: result["detail"] for profile, result in profile_results.items()},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if detail_handle is not None:
            detail_handle.close()
    raw_count = int(attribution["raw_premerge_count"])
    duplicate_probe = attribution["cross_tile_duplicate_probe"]
    duplicate_probe["cross_tile_member_fraction_of_probe"] = ratio(
        int(duplicate_probe["cross_tile_member_count"]), int(duplicate_probe["probe_candidate_count"])
    )
    duplicate_probe["cross_tile_duplicate_excess_fraction_of_probe"] = ratio(
        int(duplicate_probe["cross_tile_duplicate_excess_count"]), int(duplicate_probe["probe_candidate_count"])
    )
    attribution["raw_confidence_below_025_fraction"] = ratio(
        int(attribution["raw_confidence_below_025_count"]), raw_count
    )
    attribution["raw_touches_internal_edge_fraction"] = ratio(
        int(attribution["raw_touches_internal_edge_count"]), raw_count
    )
    reports: dict[str, dict[str, Any]] = {}
    for profile, predictions_by_image in predictions_by_profile.items():
        scores = score_predictions(predictions_by_image, ground_truth_by_image, args)
        scores["metric_protocol"]["tiled_postprocessing"] = {
            **tiled_postprocess_config(args),
            "active_profile": profile,
        }
        scores.update(
            {
                "model_path": str(model_path),
                "model_sha256": sha256_file(model_path),
                "inference_mode": "tiled",
                "tiled_profile": profile,
                "post_nms_prediction_count": profile_counts[profile]["output_count"],
            }
        )
        reports[profile] = scores
    return reports, attribution


def environment_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        payload.update({"torch": torch.__version__, "cuda": torch.version.cuda})
        payload["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        payload["torch_error"] = repr(exc)
    try:
        import ultralytics

        payload["ultralytics"] = ultralytics.__version__
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        payload["ultralytics_error"] = repr(exc)
    return payload


def write_run_metadata(run_dir: Path, runtime: dict[str, Path | int], summary: dict[str, Any], args: argparse.Namespace) -> None:
    dataset_config = run_dir / "dataset_config"
    dataset_config.mkdir(parents=True, exist_ok=True)
    for name in ("data.yaml", "train.txt", "val.txt", "tile_manifest.json"):
        shutil.copy2(Path(runtime["dataset_root"]) / name, dataset_config / name)
    requested_config = {
        "model": args.model,
        "run_name": args.run_name,
        "data_version": DATA_VERSION,
        "development_only": True,
        "fold": args.fold,
        "frozen_test_set": None,
        "tile_view": summary["tile"],
        "split_counts": summary["split_counts"],
        "training": {
            "data": str(runtime["data_yaml"]),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "optimizer": "SGD",
            "seed": args.seed,
            "deterministic": True,
            "patience": args.patience,
            "close_mosaic": 0,
            "resume": False,
        },
        "tiled_evaluation_postprocessing": tiled_postprocess_config(args),
    }
    write_json(run_dir / "requested_config.json", requested_config)
    write_json(run_dir / "environment.json", environment_payload())
    (run_dir / "README.md").write_text(
        "# L1 development-only tile experiment\n\n"
        "This run uses original OOF-consensus development images plus generated training tiles. "
        "The frozen historical release-validation set is not mounted, read, linked, or evaluated. "
        "All generated labels remain auto_only derived views rather than manual truth.\n",
        encoding="utf-8",
    )


def archive_run(run_dir: Path, output_root: Path, run_name: str) -> Path:
    archive_path = output_root / f"{run_name}_results.zip"
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {archive_path}")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, Path(run_name) / path.relative_to(run_dir))
    return archive_path


def write_comparison_reports(
    evaluation_dir: Path,
    baseline_weights: Path,
    candidate_weights: Path,
    val_records: list[dict[str, Any]],
    plan: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Evaluate two fixed weights under the same original-image protocol."""
    if not baseline_weights.is_file():
        raise FileNotFoundError(f"Baseline Fold 1 weights not found: {baseline_weights}")
    if not candidate_weights.is_file():
        raise FileNotFoundError(f"Candidate tile weights not found: {candidate_weights}")
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the archived G4b protocol byte-for-byte in the default legacy mode.
    if args.tiled_eval_profiles == ("legacy",) and not args.save_tile_details:
        reports = {
            "baseline_full": evaluate_model(baseline_weights, val_records, plan["val_grids"], args, "full"),
            "baseline_tiled": evaluate_model(baseline_weights, val_records, plan["val_grids"], args, "tiled"),
            "tile_model_full": evaluate_model(candidate_weights, val_records, plan["val_grids"], args, "full"),
            "tile_model_tiled": evaluate_model(candidate_weights, val_records, plan["val_grids"], args, "tiled"),
        }
        for name, report in reports.items():
            write_json(evaluation_dir / f"{name}.json", report)
        write_json(
            evaluation_dir / "comparison_summary.json",
            {
                "baseline_full_mAP50_95": reports["baseline_full"]["mAP50_95"],
                "baseline_tiled_mAP50_95": reports["baseline_tiled"]["mAP50_95"],
                "tile_model_full_mAP50_95": reports["tile_model_full"]["mAP50_95"],
                "tile_model_tiled_mAP50_95": reports["tile_model_tiled"]["mAP50_95"],
                "primary_comparison": "tile_model_tiled minus baseline_full under the same custom original-image evaluator",
            },
        )
        return
    baseline_tiled_profiles, baseline_attribution = evaluate_tiled_profiles(
        baseline_weights, "baseline", val_records, plan["val_grids"], args, evaluation_dir
    )
    candidate_tiled_profiles, candidate_attribution = evaluate_tiled_profiles(
        candidate_weights, "tile_model", val_records, plan["val_grids"], args, evaluation_dir
    )
    primary_profile = args.primary_tiled_profile
    reports = {
        "baseline_full": evaluate_model(baseline_weights, val_records, plan["val_grids"], args, "full"),
        "tile_model_full": evaluate_model(candidate_weights, val_records, plan["val_grids"], args, "full"),
        "baseline_tiled": baseline_tiled_profiles[primary_profile],
        "tile_model_tiled": candidate_tiled_profiles[primary_profile],
    }
    for name, report in reports.items():
        write_json(evaluation_dir / f"{name}.json", report)
    for profile in args.tiled_eval_profiles:
        write_json(evaluation_dir / f"baseline_tiled_{profile}.json", baseline_tiled_profiles[profile])
        write_json(evaluation_dir / f"tile_model_tiled_{profile}.json", candidate_tiled_profiles[profile])
    write_json(evaluation_dir / "tiled_attribution_baseline.json", baseline_attribution)
    write_json(evaluation_dir / "tiled_attribution_tile_model.json", candidate_attribution)
    summary: dict[str, Any] = {
        "baseline_full_mAP50_95": reports["baseline_full"]["mAP50_95"],
        "baseline_tiled_mAP50_95": reports["baseline_tiled"]["mAP50_95"],
        "tile_model_full_mAP50_95": reports["tile_model_full"]["mAP50_95"],
        "tile_model_tiled_mAP50_95": reports["tile_model_tiled"]["mAP50_95"],
        "primary_comparison": "tile_model_tiled minus baseline_full under the same custom original-image evaluator",
    }
    summary.update(
        {
            "primary_tiled_profile": primary_profile,
            "tiled_profiles": {
                profile: {
                    "baseline_tiled_mAP50_95": baseline_tiled_profiles[profile]["mAP50_95"],
                    "tile_model_tiled_mAP50_95": candidate_tiled_profiles[profile]["mAP50_95"],
                    "tile_model_tiled_minus_baseline_full_mAP50_95": (
                        candidate_tiled_profiles[profile]["mAP50_95"] - reports["baseline_full"]["mAP50_95"]
                    ),
                }
                for profile in args.tiled_eval_profiles
            },
            "attribution_files": {
                "baseline": "tiled_attribution_baseline.json",
                "tile_model": "tiled_attribution_tile_model.json",
            },
        }
    )
    write_json(
        evaluation_dir / "comparison_summary.json",
        summary,
    )


def run_training(
    output_root: Path,
    runtime: dict[str, Path | int],
    summary: dict[str, Any],
    val_records: list[dict[str, Any]],
    plan: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    from ultralytics import YOLO

    run_parent = output_root / "runs"
    run_dir = run_parent / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite training run: {run_dir}")
    model = YOLO(args.model)
    model.train(
        data=str(runtime["data_yaml"]),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(run_parent),
        name=args.run_name,
        exist_ok=False,
        pretrained=True,
        optimizer="SGD",
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        cache=False,
        resume=False,
        close_mosaic=0,
        amp=True,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        iou=0.7,
        max_det=300,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.4,
        save_period=-1,
        plots=True,
    )
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Ultralytics did not create expected run directory: {run_dir}")
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    if not best_weights.is_file() or not last_weights.is_file():
        raise FileNotFoundError("Training run is missing best.pt or last.pt")
    write_run_metadata(run_dir, runtime, summary, args)
    baseline_weights = args.baseline_weights.resolve()
    write_comparison_reports(run_dir / "tile_evaluation", baseline_weights, best_weights, val_records, plan, args)
    return run_dir


def run_evaluation(
    output_root: Path,
    summary: dict[str, Any],
    val_records: list[dict[str, Any]],
    plan: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    """Evaluate a recovered tile checkpoint without rebuilding the training view."""
    baseline_weights = args.baseline_weights.resolve()
    candidate_weights = args.candidate_weights.resolve()
    evaluation_root = output_root / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=False)
    write_json(
        evaluation_root / "requested_config.json",
        {
            "mode": "evaluate",
            "run_name": args.run_name,
            "data_version": DATA_VERSION,
            "development_only": True,
            "fold": args.fold,
            "frozen_test_set": None,
            "baseline_weights": str(baseline_weights),
            "baseline_weights_sha256": sha256_file(baseline_weights),
            "candidate_weights": str(candidate_weights),
            "candidate_weights_sha256": sha256_file(candidate_weights),
            "tile_view": summary["tile"],
            "split_counts": summary["split_counts"],
            "inference": {
                "imgsz": args.imgsz,
                "predict_batch": args.predict_batch,
                "device": args.device,
                "nms_iou": args.nms_iou,
                "tiled_postprocessing": tiled_postprocess_config(args),
            },
        },
    )
    write_json(evaluation_root / "plan_summary.json", summary)
    write_json(evaluation_root / "environment.json", environment_payload())
    (evaluation_root / "README.md").write_text(
        "# L1 development-only interrupted-checkpoint evaluation\n\n"
        "This evaluates a recovered tile checkpoint against the fixed Fold 1 baseline. "
        "It does not rebuild tiles, train a model, or read the frozen historical release-validation set.\n",
        encoding="utf-8",
    )
    write_comparison_reports(
        evaluation_root / "tile_evaluation", baseline_weights, candidate_weights, val_records, plan, args
    )
    return evaluation_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    context = load_context(args)
    train_records = read_records(context, "train")
    val_records = read_records(context, "val")
    train_names = {record["anchor_name"] for record in train_records}
    val_names = {record["anchor_name"] for record in val_records}
    if train_names & val_names:
        raise ValueError("Fold train/val source images overlap")
    plan = plan_dataset(train_records, val_records, args)
    summary = serializable_plan_summary(context, train_records, val_records, plan, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    output_root = args.output_root.resolve()
    if args.dry_run:
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True)
    if args.mode == "preflight":
        write_json(output_root / "preflight_report.json", summary)
        print(f"[preflight-ok] {output_root / 'preflight_report.json'}", flush=True)
        return 0
    if args.mode == "train":
        runtime = prepare_runtime_view(output_root, context, train_records, val_records, plan, summary, args)
        artifact_dir = run_training(output_root, runtime, summary, val_records, plan, args)
        archive_stem = args.run_name
    else:
        artifact_dir = run_evaluation(output_root, summary, val_records, plan, args)
        archive_stem = f"{args.run_name}_evaluation"
    archive_path = archive_run(artifact_dir, output_root, archive_stem)
    print(
        json.dumps(
            {"artifact_dir": str(artifact_dir), "archive": str(archive_path), "archive_sha256": sha256_file(archive_path)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
