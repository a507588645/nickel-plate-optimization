#!/usr/bin/env python3
"""Run one bounded L1 E3/E4 Fold 1 development-only GPU experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


CONFIG = json.loads('{"variant": "E3_P2", "experiment": "L1 E3 YOLO11m native P2 head Fold 1", "slug": "l1-ryw18-y11m-p2-e3-f01", "title": "L1 RYW18 Y11m P2 E3 Fold01", "run_name": "dianjienie_medium_yolo11m_p2_640_kaggle_seed0_oofconsensusv1_fold01", "dataset_sources": ["lly3323w/dianjienie", "lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config", "lly3323w/dianjienie-ryw18-oof-yolo11m-fold01-640-weight"], "training": {"epochs": 300, "time_hours": 9.0, "imgsz": 640, "batch": 8, "optimizer": "SGD", "seed": 0, "deterministic": true, "resume": false, "patience": 100, "save_period": 1}, "gate": {"screening_only": true, "continue_only_if": "original-image Fold 1 mAP50-95 improves by at least 0.02 and focus-class AP/Recall show the same-direction improvement; otherwise do not expand to Fold 2/3.", "comparison": "fixed full-image evaluator and baseline SHA-256 only"}, "runtime_pins": {"torch": "2.6.0+cu124", "torchvision": "0.21.0", "numpy": "1.26.4", "ultralytics": "8.4.66"}, "expected_classes": ["wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble", "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran", "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong"], "focus_classes": ["wuzi", "waterstain", "airhole", "bubble", "greenstain", "watermark"], "baseline_weight_name": "baseline_fold01_640_best.pt", "baseline_weight_sha256": "b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce", "baseline_full_map50_95": 0.16800147577890728, "baseline_tolerance": 1e-06, "pipeline_sha256": "7128b72c1e4fbd30d2cc25ce1daf051692f9da68a2933115540b2a5df863a1b8", "p2_model_yaml": "# YOLO11m with a native-module P2/P3/P4/P5 detection head for E3.\\n# Derived from Ultralytics 8.4.66 yolo11.yaml; no vendor-patched layers.\\n\\nnc: 18\\nscale: m\\nscales:\\n  n: [0.50, 0.25, 1024]\\n  s: [0.50, 0.50, 1024]\\n  m: [0.50, 1.00, 512]\\n  l: [1.00, 1.00, 512]\\n  x: [1.00, 1.50, 512]\\n\\nbackbone:\\n  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2\\n  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4\\n  - [-1, 2, C3k2, [256, False, 0.25]] # 2-P2/4\\n  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8\\n  - [-1, 2, C3k2, [512, False, 0.25]] # 4-P3/8\\n  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16\\n  - [-1, 2, C3k2, [512, True]] # 6-P4/16\\n  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32\\n  - [-1, 2, C3k2, [1024, True]] # 8-P5/32\\n  - [-1, 1, SPPF, [1024, 5]] # 9\\n  - [-1, 2, C2PSA, [1024]] # 10\\n\\nhead:\\n  - [-1, 1, nn.Upsample, [None, 2, \\"nearest\\"]]\\n  - [[-1, 6], 1, Concat, [1]] # 12: backbone P4\\n  - [-1, 2, C3k2, [512, False]] # 13-P4/16\\n\\n  - [-1, 1, nn.Upsample, [None, 2, \\"nearest\\"]]\\n  - [[-1, 4], 1, Concat, [1]] # 15: backbone P3\\n  - [-1, 2, C3k2, [256, False]] # 16-P3/8\\n\\n  - [-1, 1, nn.Upsample, [None, 2, \\"nearest\\"]]\\n  - [[-1, 2], 1, Concat, [1]] # 18: backbone P2\\n  - [-1, 2, C3k2, [256, False]] # 19-P2/4\\n\\n  - [-1, 1, Conv, [256, 3, 2]] # 20: P2 -> P3\\n  - [[-1, 16], 1, Concat, [1]]\\n  - [-1, 2, C3k2, [256, False]] # 22-P3/8\\n\\n  - [-1, 1, Conv, [256, 3, 2]] # 23: P3 -> P4\\n  - [[-1, 13], 1, Concat, [1]]\\n  - [-1, 2, C3k2, [512, False]] # 25-P4/16\\n\\n  - [-1, 1, Conv, [512, 3, 2]] # 26: P4 -> P5\\n  - [[-1, 10], 1, Concat, [1]]\\n  - [-1, 2, C3k2, [1024, True]] # 28-P5/32\\n\\n  - [[19, 22, 25, 28], 1, Detect, [nc]] # Detect(P2, P3, P4, P5)\\n", "p2_model_yaml_sha256": "1738eb00730cd0403f291a954b2e0f19a77d9fb3ad83dc1bce353743e3acf5ac"}')
PIPELINE_SOURCE = '#!/usr/bin/env python3\n"""Build and evaluate a development-only mixed full-image/tile L1 experiment.\n\nThe script never modifies source images or source labels.  It creates a new\nruntime view containing symlinked full images plus generated tiles, then maps\ntiled validation predictions back to each original image before scoring.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport gzip\nimport hashlib\nimport json\nimport math\nimport os\nimport platform\nimport random\nimport shutil\nimport sys\nimport zipfile\nfrom collections import Counter, defaultdict\nfrom pathlib import Path\nfrom typing import Any\n\nimport numpy as np\nfrom PIL import Image\n\n\nDATA_VERSION = "oofconsensusv1"\nCLASS_NAMES = (\n    "wuzi",\n    "waterstain",\n    "yellowstain",\n    "nodule",\n    "airhole",\n    "bubble",\n    "bump",\n    "fawu",\n    "overfire",\n    "NiOH",\n    "greenstain",\n    "suanwuran",\n    "quejiao",\n    "watermark",\n    "youpao",\n    "heibian",\n    "mijixiaokeng",\n    "fanxitong",\n)\nDEFAULT_FOCUS_NAMES = ("wuzi", "waterstain", "airhole", "bubble", "greenstain", "watermark")\nIOU_THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))\nTILED_EVAL_PROFILES = ("legacy", "perclass_nms", "wbf", "wbf_edge")\n\n\ndef parse_args(argv: list[str] | None = None) -> argparse.Namespace:\n    parser = argparse.ArgumentParser(\n        description="Create a development-only L1 tile training view and original-image tiled evaluation.",\n        formatter_class=argparse.ArgumentDefaultsHelpFormatter,\n    )\n    parser.add_argument("--mode", choices=("preflight", "train", "evaluate"), required=True)\n    parser.add_argument("--config-root", type=Path, required=True)\n    parser.add_argument("--config-manifest", type=Path, help="Optional manifest path when config-root is an already restored view.")\n    parser.add_argument("--image-dataset-root", type=Path, required=True)\n    parser.add_argument("--output-root", type=Path, required=True)\n    parser.add_argument("--run-name", required=True)\n    parser.add_argument("--fold", type=int, choices=(1, 2, 3), default=1)\n    parser.add_argument("--model", default="yolo11m.pt")\n    parser.add_argument("--baseline-weights", type=Path)\n    parser.add_argument("--candidate-weights", type=Path, help="Interrupted or completed tile-model weight for evaluate mode.")\n    parser.add_argument("--tile-size", type=int, default=1024)\n    parser.add_argument("--overlap", type=float, default=0.25)\n    parser.add_argument("--negative-ratio", type=float, default=0.25)\n    parser.add_argument("--min-visible-fraction", type=float, default=0.50)\n    parser.add_argument("--focus-classes", default=",".join(DEFAULT_FOCUS_NAMES))\n    parser.add_argument("--epochs", type=int, default=300)\n    parser.add_argument("--imgsz", type=int, default=640)\n    parser.add_argument("--batch", type=int, default=8)\n    parser.add_argument("--predict-batch", type=int, default=8)\n    parser.add_argument("--device", default="0")\n    parser.add_argument("--workers", type=int, default=8)\n    parser.add_argument("--seed", type=int, default=0)\n    parser.add_argument("--patience", type=int, default=100)\n    parser.add_argument("--nms-iou", type=float, default=0.70)\n    parser.add_argument(\n        "--tiled-eval-profiles",\n        default="legacy",\n        help="Comma-separated tiled post-processing profiles: legacy, perclass_nms, wbf, wbf_edge.",\n    )\n    parser.add_argument(\n        "--primary-tiled-profile",\n        default="legacy",\n        help="One requested tiled profile to expose as baseline_tiled/tile_model_tiled in the comparison summary.",\n    )\n    parser.add_argument(\n        "--per-class-conf-json",\n        type=Path,\n        help="Optional JSON profile with default_confidence and class_thresholds keyed by CLASS_NAMES.",\n    )\n    parser.add_argument("--wbf-iou", type=float, default=0.25)\n    parser.add_argument(\n        "--edge-margin",\n        type=int,\n        default=16,\n        help="Pixels from an internal crop edge considered truncated by the wbf_edge profile.",\n    )\n    parser.add_argument(\n        "--save-tile-details",\n        action="store_true",\n        help="Write gzip-compressed raw per-tile predictions and post-processing memberships for attribution review.",\n    )\n    parser.add_argument("--dry-run", action="store_true")\n    return parser.parse_args(argv)\n\n\ndef validate_args(args: argparse.Namespace) -> None:\n    if args.tile_size < 128:\n        raise ValueError("--tile-size must be at least 128")\n    if not 0 <= args.overlap < 1:\n        raise ValueError("--overlap must be in [0, 1)")\n    if args.negative_ratio < 0:\n        raise ValueError("--negative-ratio must be non-negative")\n    if not 0 < args.min_visible_fraction <= 1:\n        raise ValueError("--min-visible-fraction must be in (0, 1]")\n    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.predict_batch <= 0:\n        raise ValueError("epochs, imgsz, batch, and predict-batch must be positive")\n    if args.patience < 0:\n        raise ValueError("--patience must be non-negative")\n    if not 0 < args.nms_iou <= 1:\n        raise ValueError("--nms-iou must be in (0, 1]")\n    if not 0 < args.wbf_iou <= 1:\n        raise ValueError("--wbf-iou must be in (0, 1]")\n    if args.edge_margin < 0:\n        raise ValueError("--edge-margin must be non-negative")\n    profiles = tuple(profile.strip() for profile in args.tiled_eval_profiles.split(",") if profile.strip())\n    if not profiles or len(set(profiles)) != len(profiles):\n        raise ValueError("--tiled-eval-profiles must contain one or more unique profile names")\n    unknown_profiles = sorted(set(profiles) - set(TILED_EVAL_PROFILES))\n    if unknown_profiles:\n        raise ValueError(f"Unsupported tiled evaluation profile(s): {unknown_profiles}")\n    if args.primary_tiled_profile not in profiles:\n        raise ValueError("--primary-tiled-profile must be included in --tiled-eval-profiles")\n    args.tiled_eval_profiles = profiles\n    args.per_class_confidence = load_per_class_confidence(args.per_class_conf_json)\n    if args.mode in {"train", "evaluate"} and args.baseline_weights is None:\n        raise ValueError("--baseline-weights is required for the fixed Fold baseline comparison")\n    if args.mode == "evaluate" and args.candidate_weights is None:\n        raise ValueError("--candidate-weights is required in evaluate mode")\n\n\ndef sha256_file(path: Path) -> str:\n    digest = hashlib.sha256()\n    with path.open("rb") as handle:\n        for chunk in iter(lambda: handle.read(1024 * 1024), b""):\n            digest.update(chunk)\n    return digest.hexdigest()\n\n\ndef write_json(path: Path, payload: Any) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")\n\n\ndef load_per_class_confidence(path: Path | None) -> dict[str, Any]:\n    """Load the optional research profile while keeping legacy inference unchanged."""\n    if path is None:\n        return {\n            "source": None,\n            "default_confidence": 0.25,\n            "class_thresholds": {},\n            "metadata": {},\n        }\n    if not path.is_file():\n        raise FileNotFoundError(f"Per-class confidence profile not found: {path}")\n    payload = json.loads(path.read_text(encoding="utf-8"))\n    if not isinstance(payload, dict):\n        raise ValueError("Per-class confidence profile must be a JSON object")\n    default_confidence = payload.get("default_confidence", 0.25)\n    class_thresholds = payload.get("class_thresholds")\n    if class_thresholds is None:\n        class_thresholds = {key: value for key, value in payload.items() if key in CLASS_NAMES}\n    if not isinstance(default_confidence, (int, float)) or not 0 < float(default_confidence) <= 1:\n        raise ValueError("Per-class confidence default_confidence must be in (0, 1]")\n    if not isinstance(class_thresholds, dict):\n        raise ValueError("Per-class confidence class_thresholds must be a JSON object")\n    unknown_names = sorted(set(class_thresholds) - set(CLASS_NAMES))\n    if unknown_names:\n        raise ValueError(f"Per-class confidence profile has unknown classes: {unknown_names}")\n    normalized: dict[str, float] = {}\n    for class_name, value in class_thresholds.items():\n        if not isinstance(value, (int, float)) or not 0 < float(value) <= 1:\n            raise ValueError(f"Invalid confidence threshold for {class_name!r}: {value!r}")\n        normalized[class_name] = float(value)\n    return {\n        "source": str(path.resolve()),\n        "default_confidence": float(default_confidence),\n        "class_thresholds": normalized,\n        "metadata": {\n            key: value\n            for key, value in payload.items()\n            if key not in {"default_confidence", "class_thresholds"}\n        },\n    }\n\n\ndef read_text_lines(path: Path) -> list[str]:\n    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]\n\n\ndef read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:\n    records: list[tuple[int, float, float, float, float]] = []\n    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):\n        if not line.strip():\n            continue\n        parts = line.split()\n        if len(parts) != 5:\n            raise ValueError(f"Invalid YOLO label field count: {path}:{line_number}")\n        try:\n            class_id = int(parts[0])\n            values = tuple(float(value) for value in parts[1:])\n        except ValueError as exc:\n            raise ValueError(f"Invalid YOLO label value: {path}:{line_number}") from exc\n        if class_id < 0 or class_id >= len(CLASS_NAMES):\n            raise ValueError(f"Invalid class id: {path}:{line_number}")\n        xc, yc, width, height = values\n        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):\n            raise ValueError(f"Invalid normalized YOLO box: {path}:{line_number}")\n        records.append((class_id, xc, yc, width, height))\n    return records\n\n\ndef normalize_path(path: Path) -> str:\n    return path.resolve().as_posix()\n\n\ndef parse_focus_ids(value: str) -> set[int]:\n    names = [item.strip() for item in value.split(",") if item.strip()]\n    if not names:\n        raise ValueError("--focus-classes cannot be empty")\n    unknown = sorted(set(names) - set(CLASS_NAMES))\n    if unknown:\n        raise ValueError(f"Unknown focus classes: {unknown}")\n    return {CLASS_NAMES.index(name) for name in names}\n\n\ndef load_context(args: argparse.Namespace) -> dict[str, Any]:\n    config_root = args.config_root.resolve()\n    image_root = args.image_dataset_root.resolve()\n    manifest_path = args.config_manifest.resolve() if args.config_manifest else config_root / "config_manifest.json"\n    if not manifest_path.is_file():\n        raise FileNotFoundError(f"Missing config manifest: {manifest_path}")\n    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))\n    if manifest.get("data_version") != DATA_VERSION or manifest.get("development_only") is not True:\n        raise ValueError("The config payload is not the required development-only OOF-consensus version")\n    classes = manifest.get("classes")\n    if classes != list(CLASS_NAMES):\n        raise ValueError("Class names or class order do not match the fixed RYW18 contract")\n    images = manifest.get("images")\n    if not isinstance(images, list) or not images:\n        raise ValueError("Config manifest has no image mapping")\n    source_by_anchor: dict[str, str] = {}\n    for item in images:\n        if not isinstance(item, dict):\n            raise ValueError("Invalid image mapping entry")\n        anchor_name = item.get("anchor_name")\n        source_relative = item.get("source_relative")\n        if not isinstance(anchor_name, str) or not isinstance(source_relative, str):\n            raise ValueError("Invalid image mapping fields")\n        if anchor_name in source_by_anchor:\n            raise ValueError(f"Duplicate anchor name in config manifest: {anchor_name}")\n        source_by_anchor[anchor_name] = source_relative\n    split_root = config_root / "splits" / "folds" / f"fold_{args.fold:02d}"\n    train_list = split_root / "train.txt"\n    val_list = split_root / "val.txt"\n    if not train_list.is_file() or not val_list.is_file():\n        raise FileNotFoundError(f"Missing Fold {args.fold} train/val lists")\n    test_list = split_root / "test.txt"\n    if test_list.exists():\n        raise ValueError("Development-only tile experiment must not contain a test.txt list")\n    return {\n        "config_root": config_root,\n        "manifest_path": manifest_path,\n        "image_root": image_root,\n        "manifest": manifest,\n        "source_by_anchor": source_by_anchor,\n        "split_root": split_root,\n    }\n\n\ndef pixel_boxes(labels: list[tuple[int, float, float, float, float]], width: int, height: int) -> list[dict[str, float | int]]:\n    output: list[dict[str, float | int]] = []\n    for class_id, xc, yc, box_width, box_height in labels:\n        x1 = max(0.0, (xc - box_width / 2) * width)\n        y1 = max(0.0, (yc - box_height / 2) * height)\n        x2 = min(float(width), (xc + box_width / 2) * width)\n        y2 = min(float(height), (yc + box_height / 2) * height)\n        if x2 <= x1 or y2 <= y1:\n            raise ValueError("Normalized YOLO label became an empty pixel box")\n        output.append({"class_id": class_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})\n    return output\n\n\ndef read_records(context: dict[str, Any], split: str) -> list[dict[str, Any]]:\n    split_path = context["split_root"] / f"{split}.txt"\n    records: list[dict[str, Any]] = []\n    for image_relative in read_text_lines(split_path):\n        anchor_name = Path(image_relative).name\n        source_relative = context["source_by_anchor"].get(anchor_name)\n        if source_relative is None:\n            raise ValueError(f"Split image is absent from the manifest mapping: {image_relative}")\n        source_image = context["image_root"] / Path(source_relative)\n        label_path = context["config_root"] / "labels" / f"{Path(anchor_name).stem}.txt"\n        if not source_image.is_file():\n            raise FileNotFoundError(f"Missing source image: {source_image}")\n        if not label_path.is_file():\n            raise FileNotFoundError(f"Missing OOF candidate label: {label_path}")\n        with Image.open(source_image) as image:\n            width, height = image.size\n        if width <= 0 or height <= 0:\n            raise ValueError(f"Invalid source image size: {source_image}")\n        labels = read_yolo_labels(label_path)\n        records.append(\n            {\n                "anchor_name": anchor_name,\n                "anchor_stem": Path(anchor_name).stem,\n                "image_relative": image_relative,\n                "source_relative": source_relative,\n                "source_image": source_image,\n                "label_path": label_path,\n                "width": width,\n                "height": height,\n                "labels": labels,\n                "pixel_boxes": pixel_boxes(labels, width, height),\n            }\n        )\n    return records\n\n\ndef grid_origins(length: int, tile_size: int, overlap: float) -> list[int]:\n    if length <= tile_size:\n        return [0]\n    stride = max(1, round(tile_size * (1 - overlap)))\n    last = length - tile_size\n    origins = list(range(0, last + 1, stride))\n    if origins[-1] != last:\n        origins.append(last)\n    return origins\n\n\ndef make_grid(width: int, height: int, tile_size: int, overlap: float) -> list[dict[str, int]]:\n    tile_width = min(tile_size, width)\n    tile_height = min(tile_size, height)\n    return [\n        {"x": x, "y": y, "width": tile_width, "height": tile_height}\n        for y in grid_origins(height, tile_height, overlap)\n        for x in grid_origins(width, tile_width, overlap)\n    ]\n\n\ndef clipped_box(box: dict[str, float | int], tile: dict[str, int]) -> tuple[float, float, float, float, float]:\n    x1 = max(float(tile["x"]), float(box["x1"]))\n    y1 = max(float(tile["y"]), float(box["y1"]))\n    x2 = min(float(tile["x"] + tile["width"]), float(box["x2"]))\n    y2 = min(float(tile["y"] + tile["height"]), float(box["y2"]))\n    visible_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)\n    original_area = (float(box["x2"]) - float(box["x1"])) * (float(box["y2"]) - float(box["y1"]))\n    visible_fraction = visible_area / original_area if original_area > 0 else 0.0\n    return x1, y1, x2, y2, visible_fraction\n\n\ndef tile_margin(box: dict[str, float | int], tile: dict[str, int]) -> float:\n    center_x = (float(box["x1"]) + float(box["x2"])) / 2\n    center_y = (float(box["y1"]) + float(box["y2"])) / 2\n    return min(\n        center_x - float(tile["x"]),\n        float(tile["x"] + tile["width"]) - center_x,\n        center_y - float(tile["y"]),\n        float(tile["y"] + tile["height"]) - center_y,\n    )\n\n\ndef select_tiles_for_record(record: dict[str, Any], args: argparse.Namespace, focus_ids: set[int]) -> dict[str, Any]:\n    grid = make_grid(record["width"], record["height"], args.tile_size, args.overlap)\n    selected: set[int] = set()\n    focus_box_counts: Counter[int] = Counter()\n    uncovered_focus_counts: Counter[int] = Counter()\n    for box in record["pixel_boxes"]:\n        class_id = int(box["class_id"])\n        if class_id not in focus_ids:\n            continue\n        candidates: list[tuple[float, float, int]] = []\n        for index, tile in enumerate(grid):\n            _, _, _, _, visible_fraction = clipped_box(box, tile)\n            if visible_fraction >= args.min_visible_fraction:\n                candidates.append((visible_fraction, tile_margin(box, tile), index))\n        if not candidates:\n            # Keep oversized or severely truncated objects in their full-image\n            # training sample instead of emitting an unreliable partial tile label.\n            uncovered_focus_counts[class_id] += 1\n            continue\n        _, _, best_index = max(candidates)\n        selected.add(best_index)\n        focus_box_counts[class_id] += 1\n\n    negative_candidates: list[int] = []\n    for index, tile in enumerate(grid):\n        has_visible_box = any(\n            clipped_box(box, tile)[4] >= args.min_visible_fraction for box in record["pixel_boxes"]\n        )\n        if not has_visible_box:\n            negative_candidates.append(index)\n    return {\n        "record": record,\n        "grid": grid,\n        "selected": selected,\n        "negative_candidates": negative_candidates,\n        "focus_box_counts": focus_box_counts,\n        "uncovered_focus_counts": uncovered_focus_counts,\n    }\n\n\ndef tile_label_lines(record: dict[str, Any], tile: dict[str, int], min_visible_fraction: float) -> list[str]:\n    lines: list[str] = []\n    for box in record["pixel_boxes"]:\n        x1, y1, x2, y2, visible_fraction = clipped_box(box, tile)\n        if visible_fraction < min_visible_fraction or x2 <= x1 or y2 <= y1:\n            continue\n        local_x1 = x1 - float(tile["x"])\n        local_y1 = y1 - float(tile["y"])\n        local_x2 = x2 - float(tile["x"])\n        local_y2 = y2 - float(tile["y"])\n        x_center = (local_x1 + local_x2) / (2 * tile["width"])\n        y_center = (local_y1 + local_y2) / (2 * tile["height"])\n        box_width = (local_x2 - local_x1) / tile["width"]\n        box_height = (local_y2 - local_y1) / tile["height"]\n        lines.append(\n            f"{int(box[\'class_id\'])} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"\n        )\n    return lines\n\n\ndef plan_dataset(train_records: list[dict[str, Any]], val_records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:\n    focus_ids = parse_focus_ids(args.focus_classes)\n    train_plans = [select_tiles_for_record(record, args, focus_ids) for record in train_records]\n    val_grids = [make_grid(record["width"], record["height"], args.tile_size, args.overlap) for record in val_records]\n    flat_negatives = [\n        (plan_index, tile_index)\n        for plan_index, plan in enumerate(train_plans)\n        for tile_index in plan["negative_candidates"]\n    ]\n    selected_tile_count = sum(len(plan["selected"]) for plan in train_plans)\n    negative_target = min(len(flat_negatives), math.floor(selected_tile_count * args.negative_ratio))\n    rng = random.Random(args.seed)\n    sampled_negatives = set(rng.sample(flat_negatives, negative_target)) if negative_target else set()\n    source_class_counts: Counter[int] = Counter()\n    for record in train_records + val_records:\n        source_class_counts.update(int(box["class_id"]) for box in record["pixel_boxes"])\n    focus_counts: Counter[int] = Counter()\n    uncovered_focus_counts: Counter[int] = Counter()\n    for plan in train_plans:\n        focus_counts.update(plan["focus_box_counts"])\n        uncovered_focus_counts.update(plan["uncovered_focus_counts"])\n    return {\n        "focus_ids": focus_ids,\n        "train_plans": train_plans,\n        "val_grids": val_grids,\n        "sampled_negatives": sampled_negatives,\n        "source_class_counts": source_class_counts,\n        "focus_counts": focus_counts,\n        "uncovered_focus_counts": uncovered_focus_counts,\n    }\n\n\ndef serializable_plan_summary(\n    context: dict[str, Any], train_records: list[dict[str, Any]], val_records: list[dict[str, Any]], plan: dict[str, Any], args: argparse.Namespace\n) -> dict[str, Any]:\n    selected_tile_count = sum(len(item["selected"]) for item in plan["train_plans"])\n    train_grid_tile_count = sum(len(item["grid"]) for item in plan["train_plans"])\n    val_grid_tile_count = sum(len(grid) for grid in plan["val_grids"])\n    return {\n        "schema_version": 1,\n        "data_version": DATA_VERSION,\n        "development_only": True,\n        "fold": args.fold,\n        "source_config_manifest_sha256": sha256_file(context["manifest_path"]),\n        "source_split_hashes": {\n            "train": sha256_file(context["split_root"] / "train.txt"),\n            "val": sha256_file(context["split_root"] / "val.txt"),\n        },\n        "tile": {\n            "tile_size": args.tile_size,\n            "overlap": args.overlap,\n            "negative_ratio": args.negative_ratio,\n            "min_visible_fraction": args.min_visible_fraction,\n            "focus_classes": [CLASS_NAMES[index] for index in sorted(plan["focus_ids"])],\n            "selection": "one best-covered tile per focus box; labels retain boxes with visible area >= min_visible_fraction",\n        },\n        "split_counts": {\n            "full_train_images": len(train_records),\n            "full_val_images": len(val_records),\n            "all_train_grid_tiles": train_grid_tile_count,\n            "selected_focus_tiles": selected_tile_count,\n            "sampled_negative_tiles": len(plan["sampled_negatives"]),\n            "mixed_training_images": len(train_records) + selected_tile_count + len(plan["sampled_negatives"]),\n            "val_inference_tiles": val_grid_tile_count,\n            "test_frozen": None,\n        },\n        "focus_source_box_counts": {\n            CLASS_NAMES[index]: plan["focus_counts"].get(index, 0) for index in sorted(plan["focus_ids"])\n        },\n        "focus_boxes_kept_full_image_only": {\n            CLASS_NAMES[index]: plan["uncovered_focus_counts"].get(index, 0) for index in sorted(plan["focus_ids"])\n        },\n        "source_box_counts": {\n            CLASS_NAMES[index]: plan["source_class_counts"].get(index, 0) for index in range(len(CLASS_NAMES))\n        },\n    }\n\n\ndef link_or_copy(source: Path, destination: Path) -> None:\n    destination.parent.mkdir(parents=True, exist_ok=True)\n    try:\n        destination.symlink_to(source)\n    except OSError:\n        shutil.copy2(source, destination)\n\n\ndef write_data_yaml(dataset_root: Path) -> Path:\n    lines = [\n        f\'path: "{normalize_path(dataset_root)}"\',\n        \'train: "train.txt"\',\n        \'val: "val.txt"\',\n        "nc: 18",\n        "names:",\n        *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],\n    ]\n    path = dataset_root / "data.yaml"\n    path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")\n    return path\n\n\ndef write_tile_image_and_labels(\n    source: np.ndarray, record: dict[str, Any], tile: dict[str, int], image_path: Path, label_path: Path, args: argparse.Namespace\n) -> int:\n    import cv2\n    crop = source[tile["y"] : tile["y"] + tile["height"], tile["x"] : tile["x"] + tile["width"]]\n    if crop.shape[0] != tile["height"] or crop.shape[1] != tile["width"]:\n        raise ValueError(f"Tile crop shape mismatch for {record[\'anchor_name\']}")\n    image_path.parent.mkdir(parents=True, exist_ok=True)\n    label_path.parent.mkdir(parents=True, exist_ok=True)\n    if not cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90]):\n        raise RuntimeError(f"Failed to write tile image: {image_path}")\n    label_lines = tile_label_lines(record, tile, args.min_visible_fraction)\n    label_path.write_text("\\n".join(label_lines) + ("\\n" if label_lines else ""), encoding="utf-8")\n    return len(label_lines)\n\n\ndef prepare_runtime_view(\n    output_root: Path,\n    context: dict[str, Any],\n    train_records: list[dict[str, Any]],\n    val_records: list[dict[str, Any]],\n    plan: dict[str, Any],\n    summary: dict[str, Any],\n    args: argparse.Namespace,\n) -> dict[str, Path | int]:\n    import cv2\n\n    if not output_root.is_dir():\n        raise FileNotFoundError(f"Tile runtime root was not initialized: {output_root}")\n    if any(output_root.iterdir()):\n        raise FileExistsError(f"Refusing to write tile runtime into a non-empty output root: {output_root}")\n    dataset_root = output_root / "dataset_view"\n    dataset_root.mkdir(parents=True, exist_ok=False)\n    train_paths: list[Path] = []\n    val_paths: list[Path] = []\n    for split, records, target_paths in (("train", train_records, train_paths), ("val", val_records, val_paths)):\n        for record in records:\n            image_path = dataset_root / "images" / f"full_{split}" / record["anchor_name"]\n            label_path = dataset_root / "labels" / f"full_{split}" / f"{record[\'anchor_stem\']}.txt"\n            link_or_copy(record["source_image"], image_path)\n            label_path.parent.mkdir(parents=True, exist_ok=True)\n            shutil.copy2(record["label_path"], label_path)\n            target_paths.append(image_path)\n\n    tile_label_count = 0\n    tile_count = 0\n    for plan_index, item in enumerate(plan["train_plans"]):\n        selected_indices = set(item["selected"])\n        selected_indices.update(tile_index for current_plan, tile_index in plan["sampled_negatives"] if current_plan == plan_index)\n        record = item["record"]\n        source = cv2.imread(str(record["source_image"]), cv2.IMREAD_COLOR)\n        if source is None:\n            raise RuntimeError(f"OpenCV failed to read source image: {record[\'source_image\']}")\n        for tile_index in sorted(selected_indices):\n            tile = item["grid"][tile_index]\n            stem = f"{record[\'anchor_stem\']}__x{tile[\'x\']:04d}_y{tile[\'y\']:04d}"\n            image_path = dataset_root / "images" / "tiles" / f"{stem}.jpg"\n            label_path = dataset_root / "labels" / "tiles" / f"{stem}.txt"\n            tile_label_count += write_tile_image_and_labels(source, record, tile, image_path, label_path, args)\n            train_paths.append(image_path)\n            tile_count += 1\n\n    data_yaml = write_data_yaml(dataset_root)\n    (dataset_root / "train.txt").write_text(\n        "\\n".join(normalize_path(path) for path in train_paths) + "\\n", encoding="utf-8"\n    )\n    (dataset_root / "val.txt").write_text(\n        "\\n".join(normalize_path(path) for path in val_paths) + "\\n", encoding="utf-8"\n    )\n    summary = {**summary, "generated": {"tile_images": tile_count, "tile_label_boxes": tile_label_count}}\n    write_json(dataset_root / "tile_manifest.json", summary)\n    return {\n        "dataset_root": dataset_root,\n        "data_yaml": data_yaml,\n        "train_txt": dataset_root / "train.txt",\n        "val_txt": dataset_root / "val.txt",\n        "tile_manifest": dataset_root / "tile_manifest.json",\n    }\n\n\ndef prediction_to_rows(result: Any, offset_x: int = 0, offset_y: int = 0) -> list[list[float]]:\n    boxes = getattr(result, "boxes", None)\n    if boxes is None or len(boxes) == 0:\n        return []\n    coordinates = boxes.xyxy.detach().cpu().numpy()\n    confidences = boxes.conf.detach().cpu().numpy()\n    classes = boxes.cls.detach().cpu().numpy()\n    return [\n        [\n            float(x1 + offset_x),\n            float(y1 + offset_y),\n            float(x2 + offset_x),\n            float(y2 + offset_y),\n            float(confidence),\n            float(class_id),\n        ]\n        for (x1, y1, x2, y2), confidence, class_id in zip(coordinates, confidences, classes, strict=True)\n    ]\n\n\ndef box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:\n    if len(boxes) == 0:\n        return np.empty((0,), dtype=np.float64)\n    inter_x1 = np.maximum(box[0], boxes[:, 0])\n    inter_y1 = np.maximum(box[1], boxes[:, 1])\n    inter_x2 = np.minimum(box[2], boxes[:, 2])\n    inter_y2 = np.minimum(box[3], boxes[:, 3])\n    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)\n    area_one = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])\n    area_many = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])\n    return inter / np.maximum(area_one + area_many - inter, 1e-12)\n\n\ndef class_aware_nms(rows: list[list[float]], iou_threshold: float) -> np.ndarray:\n    if not rows:\n        return np.empty((0, 6), dtype=np.float64)\n    predictions = np.asarray(rows, dtype=np.float64)\n    kept: list[np.ndarray] = []\n    for class_id in sorted({int(value) for value in predictions[:, 5]}):\n        remaining = predictions[predictions[:, 5] == class_id]\n        remaining = remaining[np.argsort(-remaining[:, 4])]\n        while len(remaining):\n            current = remaining[0]\n            kept.append(current)\n            if len(remaining) == 1:\n                break\n            overlaps = box_iou_one_to_many(current[:4], remaining[1:, :4])\n            remaining = remaining[1:][overlaps <= iou_threshold]\n    return np.asarray(kept, dtype=np.float64)\n\n\ndef rows_to_array(rows: list[list[float]]) -> np.ndarray:\n    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 6), dtype=np.float64)\n\n\ndef tile_internal_edge_sides(\n    row: list[float], tile: dict[str, int], image_width: int, image_height: int, edge_margin: int\n) -> list[str]:\n    """Identify crop boundaries that have a neighbouring crop and touch a prediction."""\n    x1, y1, x2, y2 = (float(value) for value in row[:4])\n    sides: list[str] = []\n    if tile["x"] > 0 and x1 - tile["x"] <= edge_margin:\n        sides.append("left")\n    if tile["x"] + tile["width"] < image_width and tile["x"] + tile["width"] - x2 <= edge_margin:\n        sides.append("right")\n    if tile["y"] > 0 and y1 - tile["y"] <= edge_margin:\n        sides.append("top")\n    if tile["y"] + tile["height"] < image_height and tile["y"] + tile["height"] - y2 <= edge_margin:\n        sides.append("bottom")\n    return sides\n\n\ndef predict_tiled_raw(\n    model: Any, image: np.ndarray, grid: list[dict[str, int]], args: argparse.Namespace\n) -> tuple[np.ndarray, list[dict[str, Any]]]:\n    """Predict every crop once and retain provenance before cross-tile merging."""\n    rows: list[list[float]] = []\n    raw_details: list[dict[str, Any]] = []\n    image_height, image_width = image.shape[:2]\n    for start in range(0, len(grid), args.predict_batch):\n        batch_tiles = grid[start : start + args.predict_batch]\n        crops = [\n            image[tile["y"] : tile["y"] + tile["height"], tile["x"] : tile["x"] + tile["width"]]\n            for tile in batch_tiles\n        ]\n        results = model.predict(\n            source=crops,\n            imgsz=args.imgsz,\n            conf=0.001,\n            iou=args.nms_iou,\n            max_det=300,\n            device=args.device,\n            verbose=False,\n        )\n        if len(results) != len(batch_tiles):\n            raise RuntimeError("Tiled prediction result count mismatch")\n        for offset, (result, tile) in enumerate(zip(results, batch_tiles, strict=True)):\n            tile_index = start + offset\n            for row in prediction_to_rows(result, tile["x"], tile["y"]):\n                raw_id = len(rows)\n                class_id = int(row[5])\n                rows.append(row)\n                raw_details.append(\n                    {\n                        "raw_id": raw_id,\n                        "tile_index": tile_index,\n                        "xyxy": [float(value) for value in row[:4]],\n                        "confidence": float(row[4]),\n                        "class_id": class_id,\n                        "class_name": CLASS_NAMES[class_id],\n                        "touches_internal_edges": tile_internal_edge_sides(\n                            row, tile, image_width, image_height, args.edge_margin\n                        ),\n                    }\n                )\n    return rows_to_array(rows), raw_details\n\n\ndef per_class_confidence_threshold(class_id: int, args: argparse.Namespace) -> float:\n    class_name = CLASS_NAMES[class_id]\n    return float(args.per_class_confidence["class_thresholds"].get(class_name, args.per_class_confidence["default_confidence"]))\n\n\ndef confidence_filtered_indices(predictions: np.ndarray, args: argparse.Namespace) -> np.ndarray:\n    indices = [\n        index\n        for index, row in enumerate(predictions)\n        if float(row[4]) >= per_class_confidence_threshold(int(row[5]), args)\n    ]\n    return np.asarray(indices, dtype=np.int64)\n\n\ndef class_aware_nms_indices(\n    predictions: np.ndarray, candidate_indices: np.ndarray, iou_threshold: float\n) -> tuple[np.ndarray, np.ndarray]:\n    """Return the same class-wise NMS result as legacy mode plus raw row provenance."""\n    if len(candidate_indices) == 0:\n        return np.empty((0, 6), dtype=np.float64), np.empty((0,), dtype=np.int64)\n    kept_indices: list[int] = []\n    for class_id in sorted({int(predictions[index, 5]) for index in candidate_indices}):\n        remaining = candidate_indices[predictions[candidate_indices, 5] == class_id]\n        remaining = remaining[np.argsort(-predictions[remaining, 4])]\n        while len(remaining):\n            current_index = int(remaining[0])\n            kept_indices.append(current_index)\n            if len(remaining) == 1:\n                break\n            overlaps = box_iou_one_to_many(predictions[current_index, :4], predictions[remaining[1:], :4])\n            remaining = remaining[1:][overlaps <= iou_threshold]\n    kept = np.asarray(kept_indices, dtype=np.int64)\n    return predictions[kept], kept\n\n\ndef weighted_box_fusion(\n    predictions: np.ndarray, candidate_indices: np.ndarray, iou_threshold: float\n) -> tuple[np.ndarray, list[dict[str, Any]]]:\n    """Fuse same-class cross-crop candidates and retain every raw member id.\n\n    Coordinates are confidence-weighted; the maximum member confidence is kept\n    so cross-crop duplication does not lower the calibrated detection score.\n    """\n    if len(candidate_indices) == 0:\n        return np.empty((0, 6), dtype=np.float64), []\n    completed: list[dict[str, Any]] = []\n    for class_id in sorted({int(predictions[index, 5]) for index in candidate_indices}):\n        class_indices = candidate_indices[predictions[candidate_indices, 5] == class_id]\n        class_indices = class_indices[np.argsort(-predictions[class_indices, 4])]\n        clusters: list[dict[str, Any]] = []\n        for raw_index in class_indices:\n            raw_index = int(raw_index)\n            row = predictions[raw_index]\n            if clusters:\n                cluster_boxes = np.asarray([cluster["xyxy"] for cluster in clusters], dtype=np.float64)\n                overlaps = box_iou_one_to_many(row[:4], cluster_boxes)\n                best_cluster = int(np.argmax(overlaps))\n                if overlaps[best_cluster] >= iou_threshold:\n                    cluster = clusters[best_cluster]\n                    weight = float(row[4])\n                    cluster["weighted_sum"] += row[:4] * weight\n                    cluster["weight_sum"] += weight\n                    cluster["xyxy"] = cluster["weighted_sum"] / cluster["weight_sum"]\n                    cluster["confidence"] = max(float(cluster["confidence"]), weight)\n                    cluster["raw_member_ids"].append(raw_index)\n                    continue\n            weight = float(row[4])\n            clusters.append(\n                {\n                    "class_id": class_id,\n                    "xyxy": row[:4].copy(),\n                    "weighted_sum": row[:4] * weight,\n                    "weight_sum": weight,\n                    "confidence": weight,\n                    "raw_member_ids": [raw_index],\n                }\n            )\n        completed.extend(clusters)\n    fused_rows = [\n        [*map(float, cluster["xyxy"]), float(cluster["confidence"]), float(cluster["class_id"])]\n        for cluster in completed\n    ]\n    return rows_to_array(fused_rows), completed\n\n\ndef wbf_detail_rows(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:\n    return [\n        {\n            "class_id": int(cluster["class_id"]),\n            "class_name": CLASS_NAMES[int(cluster["class_id"])],\n            "xyxy": [float(value) for value in cluster["xyxy"]],\n            "confidence": float(cluster["confidence"]),\n            "raw_member_ids": [int(value) for value in cluster["raw_member_ids"]],\n        }\n        for cluster in clusters\n    ]\n\n\ndef cross_tile_duplicate_diagnostic(\n    predictions: np.ndarray, raw_details: list[dict[str, Any]], args: argparse.Namespace\n) -> dict[str, Any]:\n    """Measure high-confidence cross-tile duplicate pressure without scanning low-confidence noise."""\n    probe_indices = np.flatnonzero(predictions[:, 4] >= 0.25).astype(np.int64)\n    _, clusters = weighted_box_fusion(predictions, probe_indices, args.wbf_iou)\n    cross_tile_clusters = [\n        cluster\n        for cluster in clusters\n        if len({int(raw_details[index]["tile_index"]) for index in cluster["raw_member_ids"]}) > 1\n    ]\n    member_count = sum(len(cluster["raw_member_ids"]) for cluster in cross_tile_clusters)\n    excess_count = sum(len(cluster["raw_member_ids"]) - 1 for cluster in cross_tile_clusters)\n    return {\n        "probe_confidence_floor": 0.25,\n        "probe_iou": args.wbf_iou,\n        "probe_candidate_count": int(len(probe_indices)),\n        "cross_tile_cluster_count": len(cross_tile_clusters),\n        "cross_tile_member_count": member_count,\n        "cross_tile_duplicate_excess_count": excess_count,\n    }\n\n\ndef postprocess_tiled_predictions(\n    predictions: np.ndarray, raw_details: list[dict[str, Any]], args: argparse.Namespace\n) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:\n    """Apply requested profiles to one raw tiled prediction stream."""\n    raw_indices = np.arange(len(predictions), dtype=np.int64)\n    threshold_indices = confidence_filtered_indices(predictions, args)\n    edge_safe_indices = np.asarray(\n        [index for index in threshold_indices if not raw_details[int(index)]["touches_internal_edges"]], dtype=np.int64\n    )\n    results: dict[str, dict[str, Any]] = {}\n    for profile in args.tiled_eval_profiles:\n        if profile == "legacy":\n            candidate_indices = raw_indices\n            output, kept_indices = class_aware_nms_indices(predictions, candidate_indices, args.nms_iou)\n            detail = {\n                "merger": "class_aware_nms",\n                "iou": args.nms_iou,\n                "raw_input_count": int(len(raw_indices)),\n                "after_confidence_filter_count": int(len(raw_indices)),\n                "after_edge_filter_count": int(len(raw_indices)),\n                "output_count": int(len(output)),\n                "output_raw_ids": [int(index) for index in kept_indices],\n            }\n        elif profile == "perclass_nms":\n            candidate_indices = threshold_indices\n            output, kept_indices = class_aware_nms_indices(predictions, candidate_indices, args.nms_iou)\n            detail = {\n                "merger": "class_aware_nms",\n                "iou": args.nms_iou,\n                "raw_input_count": int(len(raw_indices)),\n                "after_confidence_filter_count": int(len(threshold_indices)),\n                "after_edge_filter_count": int(len(threshold_indices)),\n                "output_count": int(len(output)),\n                "output_raw_ids": [int(index) for index in kept_indices],\n            }\n        elif profile in {"wbf", "wbf_edge"}:\n            candidate_indices = threshold_indices if profile == "wbf" else edge_safe_indices\n            output, clusters = weighted_box_fusion(predictions, candidate_indices, args.wbf_iou)\n            detail = {\n                "merger": "confidence_weighted_box_fusion_max_confidence",\n                "iou": args.wbf_iou,\n                "raw_input_count": int(len(raw_indices)),\n                "after_confidence_filter_count": int(len(threshold_indices)),\n                "after_edge_filter_count": int(len(candidate_indices)),\n                "output_count": int(len(output)),\n                "fused_outputs": wbf_detail_rows(clusters),\n            }\n        else:  # validate_args rejects this before inference; retain a local guard for direct callers.\n            raise ValueError(f"Unsupported tiled profile: {profile}")\n        results[profile] = {"predictions": output, "detail": detail}\n    low_confidence_count = int(np.count_nonzero(predictions[:, 4] < 0.25))\n    edge_touch_count = sum(bool(detail["touches_internal_edges"]) for detail in raw_details)\n    diagnostic = {\n        "raw_premerge_count": int(len(predictions)),\n        "raw_confidence_below_025_count": low_confidence_count,\n        "raw_touches_internal_edge_count": edge_touch_count,\n        "cross_tile_duplicate_probe": cross_tile_duplicate_diagnostic(predictions, raw_details, args),\n    }\n    return results, diagnostic\n\n\ndef predict_tiled(model: Any, image: np.ndarray, grid: list[dict[str, int]], args: argparse.Namespace) -> np.ndarray:\n    rows: list[list[float]] = []\n    for start in range(0, len(grid), args.predict_batch):\n        batch_tiles = grid[start : start + args.predict_batch]\n        crops = [\n            image[tile["y"] : tile["y"] + tile["height"], tile["x"] : tile["x"] + tile["width"]]\n            for tile in batch_tiles\n        ]\n        results = model.predict(\n            source=crops,\n            imgsz=args.imgsz,\n            conf=0.001,\n            iou=args.nms_iou,\n            max_det=300,\n            device=args.device,\n            verbose=False,\n        )\n        if len(results) != len(batch_tiles):\n            raise RuntimeError("Tiled prediction result count mismatch")\n        for result, tile in zip(results, batch_tiles, strict=True):\n            rows.extend(prediction_to_rows(result, tile["x"], tile["y"]))\n    return class_aware_nms(rows, args.nms_iou)\n\n\ndef predict_full(model: Any, image: np.ndarray, args: argparse.Namespace) -> np.ndarray:\n    results = model.predict(\n        source=[image],\n        imgsz=args.imgsz,\n        conf=0.001,\n        iou=args.nms_iou,\n        max_det=300,\n        device=args.device,\n        verbose=False,\n    )\n    return class_aware_nms(prediction_to_rows(results[0]), args.nms_iou)\n\n\ndef average_precision(tp: np.ndarray, confidence: np.ndarray, target_count: int) -> float:\n    if target_count == 0 or len(tp) == 0:\n        return 0.0\n    order = np.argsort(-confidence)\n    tp = tp[order]\n    fp = 1.0 - tp\n    recall = np.cumsum(tp) / target_count\n    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp + fp), 1e-12)\n    recall_grid = np.linspace(0.0, 1.0, 101)\n    # Explicit sampling avoids duplicate-recall endpoint interpolation errors.\n    sampled_precision = [\n        float(np.max(precision[recall >= recall_level])) if np.any(recall >= recall_level) else 0.0\n        for recall_level in recall_grid\n    ]\n    return float(np.mean(sampled_precision))\n\n\ndef class_matches(\n    predictions_by_image: dict[str, np.ndarray],\n    ground_truth_by_image: dict[str, np.ndarray],\n    class_id: int,\n    iou_threshold: float,\n    confidence_threshold: float = 0.0,\n) -> tuple[np.ndarray, np.ndarray, int, int]:\n    ground_truth: dict[str, np.ndarray] = {}\n    candidates: list[tuple[float, str, np.ndarray]] = []\n    for image_id, rows in ground_truth_by_image.items():\n        selected = rows[rows[:, 0] == class_id, 1:]\n        ground_truth[image_id] = selected\n    for image_id, rows in predictions_by_image.items():\n        selected = rows[(rows[:, 5] == class_id) & (rows[:, 4] >= confidence_threshold)]\n        candidates.extend((float(row[4]), image_id, row[:4]) for row in selected)\n    candidates.sort(key=lambda item: item[0], reverse=True)\n    matched: dict[str, set[int]] = defaultdict(set)\n    true_positive: list[float] = []\n    confidences: list[float] = []\n    for confidence, image_id, box in candidates:\n        gt = ground_truth[image_id]\n        overlaps = box_iou_one_to_many(box, gt)\n        best_index = int(np.argmax(overlaps)) if len(overlaps) else -1\n        is_true_positive = best_index >= 0 and overlaps[best_index] >= iou_threshold and best_index not in matched[image_id]\n        if is_true_positive:\n            matched[image_id].add(best_index)\n        true_positive.append(1.0 if is_true_positive else 0.0)\n        confidences.append(confidence)\n    target_count = sum(len(rows) for rows in ground_truth.values())\n    return np.asarray(true_positive), np.asarray(confidences), target_count, len(candidates)\n\n\ndef score_predictions(predictions_by_image: dict[str, np.ndarray], ground_truth_by_image: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:\n    per_class: dict[str, dict[str, float | int | None]] = {}\n    ap50_values: list[float] = []\n    map_values: list[float] = []\n    precision_true_positive = 0.0\n    precision_predictions = 0\n    recall_targets = 0\n    for class_id, class_name in enumerate(CLASS_NAMES):\n        ap_by_iou: list[float] = []\n        target_count = 0\n        for iou_threshold in IOU_THRESHOLDS:\n            tp, confidence, target_count, _ = class_matches(\n                predictions_by_image, ground_truth_by_image, class_id, iou_threshold\n            )\n            ap_by_iou.append(average_precision(tp, confidence, target_count))\n        tp_025, _, target_count_025, prediction_count_025 = class_matches(\n            predictions_by_image, ground_truth_by_image, class_id, 0.50, confidence_threshold=0.25\n        )\n        true_positive_025 = float(tp_025.sum())\n        precision_true_positive += true_positive_025\n        precision_predictions += prediction_count_025\n        recall_targets += target_count_025\n        if target_count:\n            ap50_values.append(ap_by_iou[0])\n            map_values.extend(ap_by_iou)\n            ap50: float | None = ap_by_iou[0]\n            map5095: float | None = float(np.mean(ap_by_iou))\n        else:\n            ap50 = None\n            map5095 = None\n        per_class[class_name] = {\n            "support": target_count,\n            "ap50": ap50,\n            "ap50_95": map5095,\n            "precision_conf025_iou50": true_positive_025 / prediction_count_025 if prediction_count_025 else 0.0,\n            "recall_conf025_iou50": true_positive_025 / target_count_025 if target_count_025 else 0.0,\n        }\n    return {\n        "metric_protocol": {\n            "scope": "original-image evaluation after mapping tiled predictions back to source coordinates",\n            "iou_thresholds": list(IOU_THRESHOLDS),\n            "prediction_confidence_floor": 0.001,\n            "precision_recall_confidence": 0.25,\n            "global_class_aware_nms_iou": args.nms_iou,\n            "ap_interpolation": "101-point precision envelope",\n        },\n        "images": len(ground_truth_by_image),\n        "precision_conf025_iou50_micro": precision_true_positive / precision_predictions if precision_predictions else 0.0,\n        "recall_conf025_iou50_micro": precision_true_positive / recall_targets if recall_targets else 0.0,\n        "mAP50": float(np.mean(ap50_values)) if ap50_values else 0.0,\n        "mAP50_95": float(np.mean(map_values)) if map_values else 0.0,\n        "per_class": per_class,\n    }\n\n\ndef ground_truth_for_record(record: dict[str, Any]) -> np.ndarray:\n    rows = [\n        [float(box["class_id"]), float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]\n        for box in record["pixel_boxes"]\n    ]\n    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 5), dtype=np.float64)\n\n\ndef evaluate_model(model_path: Path, val_records: list[dict[str, Any]], val_grids: list[list[dict[str, int]]], args: argparse.Namespace, mode: str) -> dict[str, Any]:\n    from ultralytics import YOLO\n    import cv2\n\n    if mode not in {"full", "tiled"}:\n        raise ValueError(f"Unsupported evaluation mode: {mode}")\n    model = YOLO(str(model_path))\n    predictions_by_image: dict[str, np.ndarray] = {}\n    ground_truth_by_image: dict[str, np.ndarray] = {}\n    prediction_count = 0\n    for record, grid in zip(val_records, val_grids, strict=True):\n        image = cv2.imread(str(record["source_image"]), cv2.IMREAD_COLOR)\n        if image is None:\n            raise RuntimeError(f"OpenCV failed to read validation image: {record[\'source_image\']}")\n        predictions = predict_tiled(model, image, grid, args) if mode == "tiled" else predict_full(model, image, args)\n        predictions_by_image[record["anchor_name"]] = predictions\n        ground_truth_by_image[record["anchor_name"]] = ground_truth_for_record(record)\n        prediction_count += len(predictions)\n    scores = score_predictions(predictions_by_image, ground_truth_by_image, args)\n    scores.update(\n        {\n            "model_path": str(model_path),\n            "model_sha256": sha256_file(model_path),\n            "inference_mode": mode,\n            "post_nms_prediction_count": prediction_count,\n        }\n    )\n    return scores\n\n\ndef tiled_postprocess_config(args: argparse.Namespace) -> dict[str, Any]:\n    return {\n        "requested_profiles": list(args.tiled_eval_profiles),\n        "primary_profile": args.primary_tiled_profile,\n        "legacy_global_class_aware_nms_iou": args.nms_iou,\n        "wbf_iou": args.wbf_iou,\n        "edge_margin_pixels": args.edge_margin,\n        "per_class_confidence": args.per_class_confidence,\n    }\n\n\ndef ratio(numerator: int, denominator: int) -> float:\n    return numerator / denominator if denominator else 0.0\n\n\ndef evaluate_tiled_profiles(\n    model_path: Path,\n    model_label: str,\n    val_records: list[dict[str, Any]],\n    val_grids: list[list[dict[str, int]]],\n    args: argparse.Namespace,\n    evaluation_dir: Path,\n) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:\n    """Run crop inference once per image, then score all requested post-processing profiles."""\n    from ultralytics import YOLO\n    import cv2\n\n    model = YOLO(str(model_path))\n    predictions_by_profile: dict[str, dict[str, np.ndarray]] = {\n        profile: {} for profile in args.tiled_eval_profiles\n    }\n    ground_truth_by_image: dict[str, np.ndarray] = {}\n    profile_counts: dict[str, dict[str, int]] = {\n        profile: {\n            "raw_input_count": 0,\n            "after_confidence_filter_count": 0,\n            "after_edge_filter_count": 0,\n            "output_count": 0,\n        }\n        for profile in args.tiled_eval_profiles\n    }\n    attribution = {\n        "schema_version": 1,\n        "model_label": model_label,\n        "model_path": str(model_path),\n        "model_sha256": sha256_file(model_path),\n        "images": 0,\n        "raw_premerge_count": 0,\n        "raw_confidence_below_025_count": 0,\n        "raw_touches_internal_edge_count": 0,\n        "cross_tile_duplicate_probe": {\n            "probe_confidence_floor": 0.25,\n            "probe_iou": args.wbf_iou,\n            "probe_candidate_count": 0,\n            "cross_tile_cluster_count": 0,\n            "cross_tile_member_count": 0,\n            "cross_tile_duplicate_excess_count": 0,\n        },\n        "profile_counts": profile_counts,\n        "postprocess_config": tiled_postprocess_config(args),\n        "per_box_detail_file": None,\n        "per_box_detail_schema": {\n            "raw_predictions": "One row per crop-level, model-NMS prediction with tile provenance and internal-edge flags.",\n            "postprocess": "NMS outputs retain raw ids; WBF outputs retain every fused raw member id and fused coordinates.",\n        },\n    }\n    detail_path = evaluation_dir / f"{model_label}_raw_tile_predictions.jsonl.gz"\n    detail_handle: Any = None\n    if args.save_tile_details:\n        detail_handle = gzip.open(detail_path, "wt", encoding="utf-8")\n        attribution["per_box_detail_file"] = detail_path.name\n    try:\n        for record, grid in zip(val_records, val_grids, strict=True):\n            image = cv2.imread(str(record["source_image"]), cv2.IMREAD_COLOR)\n            if image is None:\n                raise RuntimeError(f"OpenCV failed to read validation image: {record[\'source_image\']}")\n            raw_predictions, raw_details = predict_tiled_raw(model, image, grid, args)\n            profile_results, image_diagnostic = postprocess_tiled_predictions(raw_predictions, raw_details, args)\n            ground_truth_by_image[record["anchor_name"]] = ground_truth_for_record(record)\n            attribution["images"] += 1\n            attribution["raw_premerge_count"] += int(image_diagnostic["raw_premerge_count"])\n            attribution["raw_confidence_below_025_count"] += int(image_diagnostic["raw_confidence_below_025_count"])\n            attribution["raw_touches_internal_edge_count"] += int(image_diagnostic["raw_touches_internal_edge_count"])\n            duplicate_probe = image_diagnostic["cross_tile_duplicate_probe"]\n            for key in (\n                "probe_candidate_count",\n                "cross_tile_cluster_count",\n                "cross_tile_member_count",\n                "cross_tile_duplicate_excess_count",\n            ):\n                attribution["cross_tile_duplicate_probe"][key] += int(duplicate_probe[key])\n            for profile, result in profile_results.items():\n                predictions_by_profile[profile][record["anchor_name"]] = result["predictions"]\n                for key in profile_counts[profile]:\n                    profile_counts[profile][key] += int(result["detail"][key])\n            if detail_handle is not None:\n                detail_handle.write(\n                    json.dumps(\n                        {\n                            "image_id": record["anchor_name"],\n                            "source_relative": record["source_relative"],\n                            "image_width": int(record["width"]),\n                            "image_height": int(record["height"]),\n                            "tiles": [{"tile_index": index, **tile} for index, tile in enumerate(grid)],\n                            "raw_predictions": raw_details,\n                            "diagnostic": image_diagnostic,\n                            "postprocess": {profile: result["detail"] for profile, result in profile_results.items()},\n                        },\n                        ensure_ascii=False,\n                    )\n                    + "\\n"\n                )\n    finally:\n        if detail_handle is not None:\n            detail_handle.close()\n    raw_count = int(attribution["raw_premerge_count"])\n    duplicate_probe = attribution["cross_tile_duplicate_probe"]\n    duplicate_probe["cross_tile_member_fraction_of_probe"] = ratio(\n        int(duplicate_probe["cross_tile_member_count"]), int(duplicate_probe["probe_candidate_count"])\n    )\n    duplicate_probe["cross_tile_duplicate_excess_fraction_of_probe"] = ratio(\n        int(duplicate_probe["cross_tile_duplicate_excess_count"]), int(duplicate_probe["probe_candidate_count"])\n    )\n    attribution["raw_confidence_below_025_fraction"] = ratio(\n        int(attribution["raw_confidence_below_025_count"]), raw_count\n    )\n    attribution["raw_touches_internal_edge_fraction"] = ratio(\n        int(attribution["raw_touches_internal_edge_count"]), raw_count\n    )\n    reports: dict[str, dict[str, Any]] = {}\n    for profile, predictions_by_image in predictions_by_profile.items():\n        scores = score_predictions(predictions_by_image, ground_truth_by_image, args)\n        scores["metric_protocol"]["tiled_postprocessing"] = {\n            **tiled_postprocess_config(args),\n            "active_profile": profile,\n        }\n        scores.update(\n            {\n                "model_path": str(model_path),\n                "model_sha256": sha256_file(model_path),\n                "inference_mode": "tiled",\n                "tiled_profile": profile,\n                "post_nms_prediction_count": profile_counts[profile]["output_count"],\n            }\n        )\n        reports[profile] = scores\n    return reports, attribution\n\n\ndef environment_payload() -> dict[str, Any]:\n    payload: dict[str, Any] = {\n        "python": sys.version,\n        "platform": platform.platform(),\n    }\n    try:\n        import torch\n\n        payload.update({"torch": torch.__version__, "cuda": torch.version.cuda})\n        payload["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\n    except Exception as exc:  # pragma: no cover - diagnostic fallback\n        payload["torch_error"] = repr(exc)\n    try:\n        import ultralytics\n\n        payload["ultralytics"] = ultralytics.__version__\n    except Exception as exc:  # pragma: no cover - diagnostic fallback\n        payload["ultralytics_error"] = repr(exc)\n    return payload\n\n\ndef write_run_metadata(run_dir: Path, runtime: dict[str, Path | int], summary: dict[str, Any], args: argparse.Namespace) -> None:\n    dataset_config = run_dir / "dataset_config"\n    dataset_config.mkdir(parents=True, exist_ok=True)\n    for name in ("data.yaml", "train.txt", "val.txt", "tile_manifest.json"):\n        shutil.copy2(Path(runtime["dataset_root"]) / name, dataset_config / name)\n    requested_config = {\n        "model": args.model,\n        "run_name": args.run_name,\n        "data_version": DATA_VERSION,\n        "development_only": True,\n        "fold": args.fold,\n        "frozen_test_set": None,\n        "tile_view": summary["tile"],\n        "split_counts": summary["split_counts"],\n        "training": {\n            "data": str(runtime["data_yaml"]),\n            "epochs": args.epochs,\n            "imgsz": args.imgsz,\n            "batch": args.batch,\n            "device": args.device,\n            "workers": args.workers,\n            "optimizer": "SGD",\n            "seed": args.seed,\n            "deterministic": True,\n            "patience": args.patience,\n            "close_mosaic": 0,\n            "resume": False,\n        },\n        "tiled_evaluation_postprocessing": tiled_postprocess_config(args),\n    }\n    write_json(run_dir / "requested_config.json", requested_config)\n    write_json(run_dir / "environment.json", environment_payload())\n    (run_dir / "README.md").write_text(\n        "# L1 development-only tile experiment\\n\\n"\n        "This run uses original OOF-consensus development images plus generated training tiles. "\n        "The frozen historical release-validation set is not mounted, read, linked, or evaluated. "\n        "All generated labels remain auto_only derived views rather than manual truth.\\n",\n        encoding="utf-8",\n    )\n\n\ndef archive_run(run_dir: Path, output_root: Path, run_name: str) -> Path:\n    archive_path = output_root / f"{run_name}_results.zip"\n    if archive_path.exists():\n        raise FileExistsError(f"Refusing to overwrite archive: {archive_path}")\n    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:\n        for path in sorted(run_dir.rglob("*")):\n            if path.is_file():\n                archive.write(path, Path(run_name) / path.relative_to(run_dir))\n    return archive_path\n\n\ndef write_comparison_reports(\n    evaluation_dir: Path,\n    baseline_weights: Path,\n    candidate_weights: Path,\n    val_records: list[dict[str, Any]],\n    plan: dict[str, Any],\n    args: argparse.Namespace,\n) -> None:\n    """Evaluate two fixed weights under the same original-image protocol."""\n    if not baseline_weights.is_file():\n        raise FileNotFoundError(f"Baseline Fold 1 weights not found: {baseline_weights}")\n    if not candidate_weights.is_file():\n        raise FileNotFoundError(f"Candidate tile weights not found: {candidate_weights}")\n    evaluation_dir.mkdir(parents=True, exist_ok=True)\n    # Preserve the archived G4b protocol byte-for-byte in the default legacy mode.\n    if args.tiled_eval_profiles == ("legacy",) and not args.save_tile_details:\n        reports = {\n            "baseline_full": evaluate_model(baseline_weights, val_records, plan["val_grids"], args, "full"),\n            "baseline_tiled": evaluate_model(baseline_weights, val_records, plan["val_grids"], args, "tiled"),\n            "tile_model_full": evaluate_model(candidate_weights, val_records, plan["val_grids"], args, "full"),\n            "tile_model_tiled": evaluate_model(candidate_weights, val_records, plan["val_grids"], args, "tiled"),\n        }\n        for name, report in reports.items():\n            write_json(evaluation_dir / f"{name}.json", report)\n        write_json(\n            evaluation_dir / "comparison_summary.json",\n            {\n                "baseline_full_mAP50_95": reports["baseline_full"]["mAP50_95"],\n                "baseline_tiled_mAP50_95": reports["baseline_tiled"]["mAP50_95"],\n                "tile_model_full_mAP50_95": reports["tile_model_full"]["mAP50_95"],\n                "tile_model_tiled_mAP50_95": reports["tile_model_tiled"]["mAP50_95"],\n                "primary_comparison": "tile_model_tiled minus baseline_full under the same custom original-image evaluator",\n            },\n        )\n        return\n    baseline_tiled_profiles, baseline_attribution = evaluate_tiled_profiles(\n        baseline_weights, "baseline", val_records, plan["val_grids"], args, evaluation_dir\n    )\n    candidate_tiled_profiles, candidate_attribution = evaluate_tiled_profiles(\n        candidate_weights, "tile_model", val_records, plan["val_grids"], args, evaluation_dir\n    )\n    primary_profile = args.primary_tiled_profile\n    reports = {\n        "baseline_full": evaluate_model(baseline_weights, val_records, plan["val_grids"], args, "full"),\n        "tile_model_full": evaluate_model(candidate_weights, val_records, plan["val_grids"], args, "full"),\n        "baseline_tiled": baseline_tiled_profiles[primary_profile],\n        "tile_model_tiled": candidate_tiled_profiles[primary_profile],\n    }\n    for name, report in reports.items():\n        write_json(evaluation_dir / f"{name}.json", report)\n    for profile in args.tiled_eval_profiles:\n        write_json(evaluation_dir / f"baseline_tiled_{profile}.json", baseline_tiled_profiles[profile])\n        write_json(evaluation_dir / f"tile_model_tiled_{profile}.json", candidate_tiled_profiles[profile])\n    write_json(evaluation_dir / "tiled_attribution_baseline.json", baseline_attribution)\n    write_json(evaluation_dir / "tiled_attribution_tile_model.json", candidate_attribution)\n    summary: dict[str, Any] = {\n        "baseline_full_mAP50_95": reports["baseline_full"]["mAP50_95"],\n        "baseline_tiled_mAP50_95": reports["baseline_tiled"]["mAP50_95"],\n        "tile_model_full_mAP50_95": reports["tile_model_full"]["mAP50_95"],\n        "tile_model_tiled_mAP50_95": reports["tile_model_tiled"]["mAP50_95"],\n        "primary_comparison": "tile_model_tiled minus baseline_full under the same custom original-image evaluator",\n    }\n    summary.update(\n        {\n            "primary_tiled_profile": primary_profile,\n            "tiled_profiles": {\n                profile: {\n                    "baseline_tiled_mAP50_95": baseline_tiled_profiles[profile]["mAP50_95"],\n                    "tile_model_tiled_mAP50_95": candidate_tiled_profiles[profile]["mAP50_95"],\n                    "tile_model_tiled_minus_baseline_full_mAP50_95": (\n                        candidate_tiled_profiles[profile]["mAP50_95"] - reports["baseline_full"]["mAP50_95"]\n                    ),\n                }\n                for profile in args.tiled_eval_profiles\n            },\n            "attribution_files": {\n                "baseline": "tiled_attribution_baseline.json",\n                "tile_model": "tiled_attribution_tile_model.json",\n            },\n        }\n    )\n    write_json(\n        evaluation_dir / "comparison_summary.json",\n        summary,\n    )\n\n\ndef run_training(\n    output_root: Path,\n    runtime: dict[str, Path | int],\n    summary: dict[str, Any],\n    val_records: list[dict[str, Any]],\n    plan: dict[str, Any],\n    args: argparse.Namespace,\n) -> Path:\n    from ultralytics import YOLO\n\n    run_parent = output_root / "runs"\n    run_dir = run_parent / args.run_name\n    if run_dir.exists():\n        raise FileExistsError(f"Refusing to overwrite training run: {run_dir}")\n    model = YOLO(args.model)\n    model.train(\n        data=str(runtime["data_yaml"]),\n        epochs=args.epochs,\n        imgsz=args.imgsz,\n        batch=args.batch,\n        device=args.device,\n        workers=args.workers,\n        project=str(run_parent),\n        name=args.run_name,\n        exist_ok=False,\n        pretrained=True,\n        optimizer="SGD",\n        seed=args.seed,\n        deterministic=True,\n        patience=args.patience,\n        cache=False,\n        resume=False,\n        close_mosaic=0,\n        amp=True,\n        lr0=0.01,\n        lrf=0.01,\n        momentum=0.937,\n        weight_decay=0.0005,\n        warmup_epochs=3.0,\n        warmup_momentum=0.8,\n        warmup_bias_lr=0.1,\n        box=7.5,\n        cls=0.5,\n        dfl=1.5,\n        iou=0.7,\n        max_det=300,\n        hsv_h=0.015,\n        hsv_s=0.7,\n        hsv_v=0.4,\n        degrees=0.0,\n        translate=0.1,\n        scale=0.5,\n        shear=0.0,\n        perspective=0.0,\n        flipud=0.0,\n        fliplr=0.5,\n        bgr=0.0,\n        mosaic=1.0,\n        mixup=0.0,\n        cutmix=0.0,\n        copy_paste=0.0,\n        erasing=0.4,\n        save_period=-1,\n        plots=True,\n    )\n    if not run_dir.is_dir():\n        raise FileNotFoundError(f"Ultralytics did not create expected run directory: {run_dir}")\n    best_weights = run_dir / "weights" / "best.pt"\n    last_weights = run_dir / "weights" / "last.pt"\n    if not best_weights.is_file() or not last_weights.is_file():\n        raise FileNotFoundError("Training run is missing best.pt or last.pt")\n    write_run_metadata(run_dir, runtime, summary, args)\n    baseline_weights = args.baseline_weights.resolve()\n    write_comparison_reports(run_dir / "tile_evaluation", baseline_weights, best_weights, val_records, plan, args)\n    return run_dir\n\n\ndef run_evaluation(\n    output_root: Path,\n    summary: dict[str, Any],\n    val_records: list[dict[str, Any]],\n    plan: dict[str, Any],\n    args: argparse.Namespace,\n) -> Path:\n    """Evaluate a recovered tile checkpoint without rebuilding the training view."""\n    baseline_weights = args.baseline_weights.resolve()\n    candidate_weights = args.candidate_weights.resolve()\n    evaluation_root = output_root / "evaluation"\n    evaluation_root.mkdir(parents=True, exist_ok=False)\n    write_json(\n        evaluation_root / "requested_config.json",\n        {\n            "mode": "evaluate",\n            "run_name": args.run_name,\n            "data_version": DATA_VERSION,\n            "development_only": True,\n            "fold": args.fold,\n            "frozen_test_set": None,\n            "baseline_weights": str(baseline_weights),\n            "baseline_weights_sha256": sha256_file(baseline_weights),\n            "candidate_weights": str(candidate_weights),\n            "candidate_weights_sha256": sha256_file(candidate_weights),\n            "tile_view": summary["tile"],\n            "split_counts": summary["split_counts"],\n            "inference": {\n                "imgsz": args.imgsz,\n                "predict_batch": args.predict_batch,\n                "device": args.device,\n                "nms_iou": args.nms_iou,\n                "tiled_postprocessing": tiled_postprocess_config(args),\n            },\n        },\n    )\n    write_json(evaluation_root / "plan_summary.json", summary)\n    write_json(evaluation_root / "environment.json", environment_payload())\n    (evaluation_root / "README.md").write_text(\n        "# L1 development-only interrupted-checkpoint evaluation\\n\\n"\n        "This evaluates a recovered tile checkpoint against the fixed Fold 1 baseline. "\n        "It does not rebuild tiles, train a model, or read the frozen historical release-validation set.\\n",\n        encoding="utf-8",\n    )\n    write_comparison_reports(\n        evaluation_root / "tile_evaluation", baseline_weights, candidate_weights, val_records, plan, args\n    )\n    return evaluation_root\n\n\ndef main(argv: list[str] | None = None) -> int:\n    args = parse_args(argv)\n    validate_args(args)\n    context = load_context(args)\n    train_records = read_records(context, "train")\n    val_records = read_records(context, "val")\n    train_names = {record["anchor_name"] for record in train_records}\n    val_names = {record["anchor_name"] for record in val_records}\n    if train_names & val_names:\n        raise ValueError("Fold train/val source images overlap")\n    plan = plan_dataset(train_records, val_records, args)\n    summary = serializable_plan_summary(context, train_records, val_records, plan, args)\n    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)\n    output_root = args.output_root.resolve()\n    if args.dry_run:\n        return 0\n    if output_root.exists():\n        raise FileExistsError(f"Refusing to overwrite output root: {output_root}")\n    output_root.mkdir(parents=True)\n    if args.mode == "preflight":\n        write_json(output_root / "preflight_report.json", summary)\n        print(f"[preflight-ok] {output_root / \'preflight_report.json\'}", flush=True)\n        return 0\n    if args.mode == "train":\n        runtime = prepare_runtime_view(output_root, context, train_records, val_records, plan, summary, args)\n        artifact_dir = run_training(output_root, runtime, summary, val_records, plan, args)\n        archive_stem = args.run_name\n    else:\n        artifact_dir = run_evaluation(output_root, summary, val_records, plan, args)\n        archive_stem = f"{args.run_name}_evaluation"\n    archive_path = archive_run(artifact_dir, output_root, archive_stem)\n    print(\n        json.dumps(\n            {"artifact_dir": str(artifact_dir), "archive": str(archive_path), "archive_sha256": sha256_file(archive_path)},\n            ensure_ascii=False,\n        ),\n        flush=True,\n    )\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
INPUT_ROOT = Path("/kaggle/input")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mounted_roots() -> list[Path]:
    roots = [path for path in INPUT_ROOT.iterdir() if path.is_dir()]
    scoped = INPUT_ROOT / "datasets"
    if scoped.is_dir():
        roots.extend(path for path in scoped.rglob("*") if path.is_dir())
    return roots


def inventory() -> dict[str, list[str]]:
    return {
        str(root.relative_to(INPUT_ROOT)): sorted(path.name for path in root.iterdir())[:12]
        for root in mounted_roots()
    }


def safe_relative(value: str) -> Path:
    candidate = PurePosixPath(value)
    if "\\" in value or candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return Path(*candidate.parts)


def install_dependencies() -> None:
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "--force-reinstall",
            "torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cu124",
        ],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "torchaudio"], check=False)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "numpy==1.26.4", "ultralytics==8.4.66"],
        check=True,
    )
    import numpy
    import torch
    import torchvision
    import ultralytics

    expected = CONFIG["runtime_pins"]
    if (
        torch.__version__ != expected["torch"]
        or torchvision.__version__.split("+", 1)[0] != expected["torchvision"]
        or numpy.__version__ != expected["numpy"]
        or ultralytics.__version__ != expected["ultralytics"]
        or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Pinned runtime verification failed: "
            f"torch={torch.__version__}, torchvision={torchvision.__version__}, numpy={numpy.__version__}, "
            f"ultralytics={ultralytics.__version__}, cuda={torch.cuda.is_available()}"
        )


def environment_payload() -> dict:
    import numpy
    import torch
    import torchvision
    import ultralytics

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": numpy.__version__,
        "ultralytics": ultralytics.__version__,
    }


def resolve_dev_config_payload() -> tuple[Path, Path]:
    candidates = [root for root in mounted_roots() if (root / "config_archive_manifest.json").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate development-only config Dataset: {inventory()}")
    package_root = candidates[0]
    archive_manifest = json.loads((package_root / "config_archive_manifest.json").read_text(encoding="utf-8"))
    archive_name = archive_manifest.get("config_archive")
    payload_files = archive_manifest.get("payload_files")
    source_manifest_sha256 = archive_manifest.get("source_manifest_sha256")
    if not isinstance(archive_name, str) or not isinstance(payload_files, list) or not isinstance(source_manifest_sha256, str):
        raise ValueError("Invalid development config archive manifest")
    payload_root = package_root / Path(archive_name).stem
    actual = {path.name for path in payload_root.iterdir() if path.is_file()} if payload_root.is_dir() else set()
    if actual != set(payload_files):
        raise ValueError(f"Expanded config payload mismatch: expected={len(payload_files)}, actual={len(actual)}")
    manifest_path = payload_root / "config_manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != source_manifest_sha256:
        raise ValueError("Expanded config manifest hash mismatch")
    return payload_root, manifest_path


def restore_dev_config(payload_root: Path, manifest_path: Path) -> tuple[Path, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("data_version") != "oofconsensusv1" or manifest.get("development_only") is not True:
        raise ValueError("Only the OOF-consensus development-only config may be used")
    if manifest.get("classes") != CONFIG["expected_classes"]:
        raise ValueError("Development config class names do not match the frozen RYW18 contract")
    if manifest.get("counts", {}).get("images") != 1127 or manifest.get("counts", {}).get("labels") != 1127:
        raise ValueError("Development config must declare exactly 1127 images and labels")
    labels, metadata = manifest.get("labels"), manifest.get("metadata")
    if not isinstance(labels, list) or not isinstance(metadata, list):
        raise ValueError("Config manifest does not describe flattened labels and metadata")
    runtime = Path("/kaggle/working/l1_e3e4_dev_config_runtime")
    if runtime.exists():
        raise FileExistsError(f"Refusing to overwrite runtime config: {runtime}")
    runtime.mkdir(parents=True)
    shutil.copy2(manifest_path, runtime / "config_manifest.json")
    for record in labels + metadata:
        upload_name = record.get("upload_name")
        relative_path = record.get("relative_path")
        if not isinstance(upload_name, str) or Path(upload_name).name != upload_name or not isinstance(relative_path, str):
            raise ValueError("Invalid flattened config record")
        source = payload_root / upload_name
        target = runtime / safe_relative(relative_path)
        if not source.is_file():
            raise FileNotFoundError(f"Missing config input: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return runtime, manifest


def resolve_image_dataset(manifest: dict) -> Path:
    images = manifest.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], dict):
        raise ValueError("Config manifest has no source-image probe")
    source_relative = images[0].get("source_relative")
    if not isinstance(source_relative, str):
        raise ValueError("Config source-image probe is invalid")
    parts = PurePosixPath(source_relative).parts
    candidates = [root for root in mounted_roots() if root.joinpath(*parts).is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate source-image Dataset: candidates={candidates}, inputs={inventory()}")
    return candidates[0]


def resolve_baseline_weight() -> Path:
    candidates = [root / CONFIG["baseline_weight_name"] for root in mounted_roots() if (root / CONFIG["baseline_weight_name"]).is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate fixed Fold 1 baseline checkpoint: {inventory()}")
    weight = candidates[0]
    actual = sha256_file(weight)
    if actual != CONFIG["baseline_weight_sha256"]:
        raise ValueError(f"Baseline checkpoint SHA-256 mismatch: expected={CONFIG['baseline_weight_sha256']}, actual={actual}")
    return weight


def source_mapping(manifest: dict) -> dict[str, str]:
    mapping = {item["anchor_name"]: item["source_relative"] for item in manifest["images"]}
    if len(mapping) != 1127:
        raise ValueError("The development config image mapping is incomplete")
    return mapping


def resolve_e4_payload() -> tuple[Path, dict]:
    candidates = []
    for root in mounted_roots():
        manifest_path = root / "payload_manifest.json"
        if not manifest_path.is_file():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("experiment") == "L1 E4 Copy-Paste Fold 1 Kaggle payload":
            candidates.append((root, payload))
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate E4 Copy-Paste payload: {inventory()}")
    root, payload = candidates[0]
    if payload.get("development_only") is not True or payload.get("frozen_test_set") is not None:
        raise ValueError("E4 payload does not declare development-only frozen-test isolation")
    if payload.get("split_counts") != {"train": 787, "val": 376, "synthetic_train": 36}:
        raise ValueError("E4 payload split counts are not the approved Fold 1 view")
    if payload.get("classes") != CONFIG["expected_classes"]:
        raise ValueError("E4 payload classes do not match the frozen RYW18 contract")
    for record in payload.get("files", []):
        relative = record.get("relative_path")
        expected_hash = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("Invalid E4 payload file record")
        path = root / safe_relative(relative)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"E4 payload integrity check failed: {relative}")
    return root, payload


def write_data_yaml(runtime_root: Path, train_images: list[Path], val_images: list[Path]) -> Path:
    train_list = runtime_root / "train.txt"
    val_list = runtime_root / "val.txt"
    train_list.write_text("\n".join(path.as_posix() for path in train_images) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(path.as_posix() for path in val_images) + "\n", encoding="utf-8")
    data_yaml = runtime_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f'path: "{runtime_root.as_posix()}"',
                f'train: "{train_list.as_posix()}"',
                f'val: "{val_list.as_posix()}"',
                "nc: 18",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CONFIG["expected_classes"])],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def materialize_original_pair(
    runtime_root: Path,
    image_root: Path,
    config_root: Path,
    mapping: dict[str, str],
    image_name: str,
    label_source: Path,
) -> tuple[Path, Path]:
    source_relative = mapping.get(image_name)
    if source_relative is None:
        raise ValueError(f"No source-image mapping for {image_name}")
    image = image_root.joinpath(*PurePosixPath(source_relative).parts)
    if not image.is_file() or not label_source.is_file():
        raise FileNotFoundError(f"Runtime source missing: image={image}, label={label_source}")
    linked_image = runtime_root / "images" / image_name
    linked_label = runtime_root / "labels" / label_source.name
    linked_image.parent.mkdir(parents=True, exist_ok=True)
    linked_label.parent.mkdir(parents=True, exist_ok=True)
    linked_image.symlink_to(image)
    shutil.copy2(label_source, linked_label)
    return linked_image, linked_label


def build_e3_runtime(config_root: Path, manifest: dict, image_root: Path) -> tuple[Path, list[Path], list[Path], dict]:
    runtime_root = Path("/kaggle/working/l1_e3_p2_runtime")
    if runtime_root.exists():
        raise FileExistsError(f"Refusing to overwrite E3 runtime view: {runtime_root}")
    runtime_root.mkdir(parents=True)
    mapping = source_mapping(manifest)
    fold_root = config_root / "splits" / "folds" / "fold_01"

    def materialize(name: str) -> list[Path]:
        images: list[Path] = []
        for value in [line.strip() for line in (fold_root / name).read_text(encoding="utf-8").splitlines() if line.strip()]:
            image_name = Path(value).name
            label = config_root / "labels" / f"{Path(image_name).stem}.txt"
            linked_image, _ = materialize_original_pair(runtime_root, image_root, config_root, mapping, image_name, label)
            images.append(linked_image)
        return images

    train_images, val_images = materialize("train.txt"), materialize("val.txt")
    if len(train_images) != 751 or len(val_images) != 376 or set(train_images) & set(val_images):
        raise ValueError("E3 Fold 1 runtime membership is invalid")
    data_yaml = write_data_yaml(runtime_root, train_images, val_images)
    return data_yaml, train_images, val_images, {"synthetic_train": 0, "e4_payload_manifest_sha256": None}


def build_e4_runtime(
    config_root: Path,
    manifest: dict,
    image_root: Path,
    payload_root: Path,
    payload: dict,
) -> tuple[Path, list[Path], list[Path], dict]:
    runtime_root = Path("/kaggle/working/l1_e4_copypaste_runtime")
    if runtime_root.exists():
        raise FileExistsError(f"Refusing to overwrite E4 runtime view: {runtime_root}")
    runtime_root.mkdir(parents=True)
    mapping = source_mapping(manifest)
    fold_root = payload_root / "fold_01"
    labels_root = fold_root / "labels"
    images_root = fold_root / "images"
    train_entries = [line.strip() for line in (fold_root / "splits" / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    val_entries = [line.strip() for line in (fold_root / "splits" / "val.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(train_entries) != 787 or len(val_entries) != 376 or set(train_entries) & set(val_entries):
        raise ValueError("E4 Fold 1 payload membership is invalid")
    train_images: list[Path] = []
    val_images: list[Path] = []
    synthetic_names: list[str] = []

    def materialize(entry: str, target: list[Path], allow_synthetic: bool) -> None:
        relative = safe_relative(entry)
        if relative.parts[0] != "images" or len(relative.parts) != 2:
            raise ValueError(f"Unexpected E4 image entry: {entry}")
        image_name = relative.name
        label = labels_root / f"{Path(image_name).stem}.txt"
        if image_name.startswith("cp_"):
            if not allow_synthetic:
                raise ValueError("Synthetic image entered E4 validation split")
            source_image = images_root / image_name
            if not source_image.is_file() or not label.is_file():
                raise FileNotFoundError(f"E4 synthetic source missing: image={source_image}, label={label}")
            target_image = runtime_root / "images" / image_name
            target_label = runtime_root / "labels" / label.name
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, target_image)
            shutil.copy2(label, target_label)
            target.append(target_image)
            synthetic_names.append(image_name)
            return
        linked_image, _ = materialize_original_pair(runtime_root, image_root, config_root, mapping, image_name, label)
        target.append(linked_image)

    for entry in train_entries:
        materialize(entry, train_images, True)
    for entry in val_entries:
        materialize(entry, val_images, False)
    if len(synthetic_names) != 36 or len(set(synthetic_names)) != 36:
        raise ValueError("E4 runtime must contain exactly 36 unique synthetic training images")
    data_yaml = write_data_yaml(runtime_root, train_images, val_images)
    return data_yaml, train_images, val_images, {
        "synthetic_train": len(synthetic_names),
        "e4_payload_manifest_sha256": sha256_file(payload_root / "payload_manifest.json"),
        "e4_source_build_manifest_sha256": payload.get("source_build_manifest_sha256"),
    }


def build_model() -> tuple[object, dict]:
    from ultralytics import YOLO

    baseline_pretrained = YOLO("yolo11m.pt")
    initial_weight = Path(getattr(baseline_pretrained, "ckpt_path", "yolo11m.pt"))
    if not initial_weight.is_file():
        initial_weight = Path("yolo11m.pt")
    if CONFIG["variant"] == "E3_P2":
        model_yaml = Path("/kaggle/working/yolo11m_p2.yaml")
        model_yaml.write_text(CONFIG["p2_model_yaml"], encoding="utf-8")
        model = YOLO(str(model_yaml))
        model.load(str(initial_weight))
        return model, {
            "architecture": "YOLO11m native P2/P3/P4/P5 Detect",
            "model_yaml_sha256": CONFIG["p2_model_yaml_sha256"],
            "pretrained_source": str(initial_weight),
            "pretrained_source_sha256": sha256_file(initial_weight),
            "transfer": "Ultralytics partial load from official YOLO11m checkpoint; P2-specific parameters remain architecture-initialized.",
        }
    return baseline_pretrained, {
        "architecture": "YOLO11m",
        "model_yaml_sha256": None,
        "pretrained_source": str(initial_weight),
        "pretrained_source_sha256": sha256_file(initial_weight),
        "transfer": "Official YOLO11m pretrained checkpoint.",
    }


def evaluate_full_image(
    pipeline: Path,
    config_root: Path,
    image_root: Path,
    baseline: Path,
    candidate: Path,
    run_dir: Path,
) -> dict:
    workspace = Path("/kaggle/working") / f"{CONFIG['run_name']}_evaluation_workspace"
    command = [
        sys.executable, str(pipeline),
        "--mode", "evaluate",
        "--config-root", str(config_root),
        "--image-dataset-root", str(image_root),
        "--output-root", str(workspace),
        "--run-name", CONFIG["run_name"],
        "--fold", "1",
        "--baseline-weights", str(baseline),
        "--candidate-weights", str(candidate),
        "--tile-size", "4096",
        "--overlap", "0.0",
        "--negative-ratio", "0.0",
        "--min-visible-fraction", "1.0",
        "--focus-classes", ",".join(CONFIG["focus_classes"]),
        "--epochs", str(CONFIG["training"]["epochs"]),
        "--imgsz", str(CONFIG["training"]["imgsz"]),
        "--batch", str(CONFIG["training"]["batch"]),
        "--predict-batch", str(CONFIG["evaluation_predict_batch"]),
        "--device", "0",
        "--workers", "8",
        "--seed", "0",
        "--patience", "100",
        "--tiled-eval-profiles", "legacy",
        "--primary-tiled-profile", "legacy",
    ]
    subprocess.run(command, check=True)
    source = workspace / "evaluation"
    if not source.is_dir():
        raise FileNotFoundError(f"Expected full-image evaluation evidence is missing: {source}")
    destination = run_dir / "evaluation"
    shutil.copytree(source, destination)
    baseline_report = json.loads((destination / "tile_evaluation" / "baseline_full.json").read_text(encoding="utf-8"))
    candidate_report = json.loads((destination / "tile_evaluation" / "tile_model_full.json").read_text(encoding="utf-8"))
    candidate_hash = sha256_file(candidate)
    if candidate_report.get("model_sha256") != candidate_hash:
        raise ValueError("Evaluation candidate report does not match saved best.pt")
    baseline_metric = float(baseline_report["mAP50_95"])
    if abs(baseline_metric - CONFIG["baseline_full_map50_95"]) > CONFIG["baseline_tolerance"]:
        raise ValueError(
            f"Fixed full-image baseline did not reproduce: expected={CONFIG['baseline_full_map50_95']}, actual={baseline_metric}"
        )
    per_class = candidate_report.get("per_class")
    if not isinstance(per_class, dict):
        raise ValueError("Candidate full-image report has no per-class metrics")
    focus = {}
    for class_name in CONFIG["focus_classes"]:
        item = per_class.get(class_name)
        if not isinstance(item, dict):
            raise ValueError(f"Candidate report is missing focus class: {class_name}")
        focus[class_name] = {
            "AP50_95": item.get("AP50_95"),
            "recall_conf025_iou50": item.get("recall_conf025_iou50"),
        }
    return {
        "entry": "l1_tile_pipeline.py --mode evaluate",
        "pipeline_sha256": CONFIG["pipeline_sha256"],
        "mode": "one original full-image crop per source validation image",
        "baseline_full_mAP50_95": baseline_metric,
        "candidate_best_mAP50_95": float(candidate_report["mAP50_95"]),
        "candidate_minus_fixed_baseline_mAP50_95": float(candidate_report["mAP50_95"]) - baseline_metric,
        "candidate_focus": focus,
        "baseline_model_sha256": baseline_report.get("model_sha256"),
        "candidate_best_model_sha256": candidate_hash,
        "tile_size": 4096,
        "overlap": 0.0,
    }


def copy_evidence(run_dir: Path, data_yaml: Path, config_root: Path) -> None:
    evidence = run_dir / "dataset_config"
    evidence.mkdir(parents=True, exist_ok=False)
    shutil.copy2(data_yaml, evidence / "data.yaml")
    shutil.copy2(config_root / "splits" / "folds" / "fold_01" / "train.txt", evidence / "original_fold01_train.txt")
    shutil.copy2(config_root / "splits" / "folds" / "fold_01" / "val.txt", evidence / "original_fold01_val.txt")


def main() -> None:
    payload_root, manifest_path = resolve_dev_config_payload()
    config_root, manifest = restore_dev_config(payload_root, manifest_path)
    image_root = resolve_image_dataset(manifest)
    baseline = resolve_baseline_weight()
    payload_details = {"synthetic_train": 0, "e4_payload_manifest_sha256": None}
    if CONFIG["variant"] == "E4_COPYPASTE":
        e4_payload_root, e4_payload = resolve_e4_payload()
    else:
        e4_payload_root, e4_payload = None, None
    install_dependencies()
    pipeline = Path("/kaggle/working/l1_tile_pipeline.py")
    pipeline.write_text(PIPELINE_SOURCE, encoding="utf-8")
    if CONFIG["variant"] == "E3_P2":
        data_yaml, train_images, val_images, payload_details = build_e3_runtime(config_root, manifest, image_root)
    else:
        data_yaml, train_images, val_images, payload_details = build_e4_runtime(
            config_root, manifest, image_root, e4_payload_root, e4_payload
        )
    model, initialization = build_model()
    run_root = Path("/kaggle/working/runs")
    run_dir = run_root / CONFIG["run_name"]
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {run_dir}")
    training = CONFIG["training"]
    model.train(
        data=str(data_yaml),
        epochs=training["epochs"],
        time=training["time_hours"],
        imgsz=training["imgsz"],
        batch=training["batch"],
        device="0",
        workers=8,
        project=str(run_root),
        name=CONFIG["run_name"],
        exist_ok=False,
        pretrained=True,
        optimizer="SGD",
        seed=0,
        deterministic=True,
        patience=100,
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
        copy_paste_mode="flip",
        auto_augment="randaugment",
        erasing=0.4,
        save_period=1,
        plots=True,
    )
    best, last = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
    required = (best, last, run_dir / "args.yaml", run_dir / "results.csv")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Training output is incomplete: {missing}")
    evaluation = evaluate_full_image(pipeline, config_root, image_root, baseline, best, run_dir)
    copy_evidence(run_dir, data_yaml, config_root)
    requested = {
        "experiment": CONFIG["experiment"],
        "variant": CONFIG["variant"],
        "run_name": CONFIG["run_name"],
        "development_only": True,
        "auto_only": True,
        "frozen_test_set": None,
        "fold": 1,
        "split_counts": {"train": len(train_images), "val": len(val_images), "test_frozen": None},
        "runtime_pins": CONFIG["runtime_pins"],
        "baseline_weight": {"sha256": sha256_file(baseline), "fixed_full_mAP50_95": CONFIG["baseline_full_map50_95"]},
        "training": training,
        "initialization": initialization,
        "dataset_payload": payload_details,
        "evaluation": evaluation,
        "gate": CONFIG["gate"],
    }
    write_json(run_dir / "requested_config.json", requested)
    write_json(run_dir / "environment.json", environment_payload())
    write_json(run_dir / "initialization.json", initialization)
    (run_dir / "README.md").write_text(
        f"# {CONFIG['experiment']}\n\n"
        "This is a single-seed Fold 1 development-only, auto_only screening run. "
        "The 198-image release-validation set is not mounted, read, linked, or evaluated. "
        "`time_hours` bounds training before the Kaggle 12-hour session ceiling; `save_period=1` keeps per-epoch checkpoints.\n",
        encoding="utf-8",
    )
    archive = Path(shutil.make_archive(str(Path("/kaggle/working") / f"{CONFIG['run_name']}_results"), "zip", run_root, CONFIG["run_name"]))
    print(json.dumps({"run_dir": str(run_dir), "archive": str(archive), "best_sha256": sha256_file(best)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
