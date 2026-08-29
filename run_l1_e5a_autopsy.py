"""Run the Fold 1 E5a high-confidence FP autopsy on CPU only.

The script writes a pre-registration manifest before inference and keeps
source data read-only. It stores only unmatched high-confidence candidates,
not the full raw prediction dump.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# E5a is deliberately CPU-only even when the host has a CUDA installation.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "dianjienie_ryw18_oofconsensus_v1_dev_only_r1"
DEFAULT_VAL_LIST = DEFAULT_DATA_ROOT / "splits" / "folds" / "fold_01" / "val.txt"
DEFAULT_E3_WEIGHT = (
    ROOT
    / "l1_e3_p2_v4_recovery_20260824_r2"
    / "dianjienie_medium_yolo11m_p2_640_kaggle_seed0_oofconsensusv1_fold01"
    / "weights"
    / "best.pt"
)
DEFAULT_E4_WEIGHT = (
    ROOT
    / "l1_e4_copypaste_v4_recovery_20260824_r2"
    / "dianjienie_medium_yolo11m_cp_640_kaggle_seed0_oofconsensusv1_fold01"
    / "weights"
    / "best.pt"
)
DEFAULT_OUTPUT_PARENT = ROOT
DEFAULT_SEED = 20260824
DEFAULT_CLASS_CONF_FLOOR = 0.70
DEFAULT_CLASS_CONF_QUANTILE = 0.90
DEFAULT_MATCH_IOU = 0.10
DEFAULT_PER_CLASS_CAP = 20
DEFAULT_SOURCE_QUOTA = 10
DEFAULT_INFERENCE_CONF_FLOOR = 0.001
DEFAULT_NMS_IOU = 0.70
DEFAULT_IMGSZ = 640
DEFAULT_MAX_DET = 300
DEFAULT_BATCH = 4

CSV_FIELDS = [
    "candidate_id",
    "source_model",
    "image_name",
    "image_path",
    "image_width",
    "image_height",
    "class_id",
    "class_name",
    "confidence",
    "threshold_used",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "w",
    "h",
    "same_class_max_iou",
    "all_class_max_iou",
    "matched_same_class_label_index",
    "auto_label_count",
    "candidate_rule",
    "sampled",
    "sample_rank",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E5a Fold 1 high-confidence FP autopsy (CPU-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--val-list", type=Path, default=DEFAULT_VAL_LIST)
    parser.add_argument("--e3-weight", type=Path, default=DEFAULT_E3_WEIGHT)
    parser.add_argument("--e4-weight", type=Path, default=DEFAULT_E4_WEIGHT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="A new versioned output directory; defaults to l1_e5a_fp_autopsy_<timestamp>.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--conf-floor", type=float, default=DEFAULT_CLASS_CONF_FLOOR)
    parser.add_argument("--conf-quantile", type=float, default=DEFAULT_CLASS_CONF_QUANTILE)
    parser.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--per-class-cap", type=int, default=DEFAULT_PER_CLASS_CAP)
    parser.add_argument("--source-quota", type=int, default=DEFAULT_SOURCE_QUOTA)
    parser.add_argument("--inference-conf-floor", type=float, default=DEFAULT_INFERENCE_CONF_FLOOR)
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths, Fold membership, labels, hashes and model class order without inference or output writes.",
    )
    return parser.parse_args()


def normalize_model_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(value) for value in names]
    raise TypeError(f"Unsupported model names type: {type(names)!r}")


def read_classes(data_root: Path) -> list[str]:
    path = data_root / "classes.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing classes file: {path}")
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"Empty classes file: {path}")
    return classes


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def yolo_to_xyxy(values: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, box_w, box_h = values
    x1 = max(0.0, (cx - box_w / 2.0) * width)
    y1 = max(0.0, (cy - box_h / 2.0) * height)
    x2 = min(float(width), (cx + box_w / 2.0) * width)
    y2 = min(float(height), (cy + box_h / 2.0) * height)
    return x1, y1, x2, y2


def parse_labels(path: Path, width: int, height: int, class_count: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation label: {path}")
    labels: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Expected five-column label at {path}:{line_number}")
        try:
            class_float, cx, cy, box_w, box_h = map(float, fields)
        except ValueError as error:
            raise ValueError(f"Invalid label value at {path}:{line_number}") from error
        if not class_float.is_integer() or not 0 <= int(class_float) < class_count:
            raise ValueError(f"Invalid class at {path}:{line_number}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < box_w <= 1 and 0 < box_h <= 1):
            raise ValueError(f"Invalid normalized box at {path}:{line_number}")
        labels.append(
            {
                "class_id": int(class_float),
                "xywh": [cx, cy, box_w, box_h],
                "xyxy": list(yolo_to_xyxy([cx, cy, box_w, box_h], width, height)),
                "index": len(labels),
            }
        )
    return labels


def load_validation_records(data_root: Path, val_list: Path, classes: list[str]) -> list[dict[str, Any]]:
    images_root = (data_root / "images").resolve()
    labels_root = (data_root / "labels").resolve()
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(f"Expected images/ and labels/ under {data_root}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(val_list.read_text(encoding="utf-8").splitlines(), start=1):
        value = raw.strip()
        if not value:
            continue
        candidate = Path(value)
        image = (candidate if candidate.is_absolute() else data_root / candidate).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Missing Fold 1 image at {val_list}:{line_number}: {image}")
        if "release-validation" in image.as_posix().lower():
            raise ValueError(f"Forbidden release-validation image: {image}")
        try:
            image.relative_to(images_root)
        except ValueError as error:
            raise ValueError(f"Validation image is outside data_root/images: {image}") from error
        if image.name in seen:
            raise ValueError(f"Duplicate validation image: {image.name}")
        seen.add(image.name)
        width, height = image_size(image)
        labels = parse_labels(labels_root / f"{image.stem}.txt", width, height, len(classes))
        records.append(
            {
                "image_name": image.name,
                "image_path": image,
                "width": width,
                "height": height,
                "labels": labels,
                "label_path": labels_root / f"{image.stem}.txt",
            }
        )
    if len(records) != 376:
        raise ValueError(f"E5a requires exactly 376 Fold 1 images, found {len(records)}")
    return records


def validate_config(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, list[str], list[dict[str, Any]]]:
    data_root = args.data_root.resolve()
    val_list = args.val_list.resolve()
    e3_weight = args.e3_weight.resolve()
    e4_weight = args.e4_weight.resolve()
    for path in (data_root, val_list, e3_weight, e4_weight):
        if not path.exists():
            raise FileNotFoundError(path)
    if not 0 < args.conf_floor <= 1:
        raise ValueError("--conf-floor must be in (0, 1]")
    if not 0 < args.conf_quantile < 1:
        raise ValueError("--conf-quantile must be in (0, 1)")
    if not 0 <= args.match_iou < 1:
        raise ValueError("--match-iou must be in [0, 1)")
    if args.per_class_cap <= 0 or args.source_quota <= 0:
        raise ValueError("Sampling caps must be positive")
    if args.source_quota * 2 > args.per_class_cap:
        raise ValueError("Two source quotas cannot exceed the per-class cap")
    if not 0 < args.inference_conf_floor <= 1:
        raise ValueError("--inference-conf-floor must be in (0, 1]")
    if not 0 < args.nms_iou <= 1 or args.imgsz <= 0 or args.max_det <= 0 or args.batch <= 0:
        raise ValueError("Invalid inference configuration")
    if "release-validation" in " ".join(str(p) for p in (data_root, val_list, e3_weight, e4_weight)).lower():
        raise ValueError("E5a input path contains forbidden release-validation")
    classes = read_classes(data_root)
    records = load_validation_records(data_root, val_list, classes)

    # Loading weights is cheap compared with inference and catches class-order drift during dry-run.
    from ultralytics import YOLO

    for name, weight in (("e3_p2", e3_weight), ("e4_copypaste", e4_weight)):
        model_names = normalize_model_names(YOLO(str(weight)).names)
        if model_names != classes:
            raise ValueError(f"{name} class order differs from classes.txt: {model_names!r} != {classes!r}")
    return data_root, val_list, e3_weight, e4_weight, classes, records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def box_iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def result_rows(result: Any) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    coordinates = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    return [
        {
            "class_id": int(class_id),
            "confidence": float(confidence),
            "xyxy": [float(value) for value in xyxy],
        }
        for xyxy, confidence, class_id in zip(coordinates, confidences, classes, strict=True)
    ]


def infer_model(
    model_name: str,
    weight: Path,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    classes: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model = YOLO(str(weight))
    if normalize_model_names(model.names) != classes:
        raise ValueError(f"Model class order changed before inference: {weight}")
    predictions: dict[str, list[dict[str, Any]]] = {}
    total_boxes = 0
    for start in range(0, len(records), args.batch):
        batch = records[start : start + args.batch]
        results = model.predict(
            source=[str(row["image_path"]) for row in batch],
            imgsz=args.imgsz,
            conf=args.inference_conf_floor,
            iou=args.nms_iou,
            max_det=args.max_det,
            device="cpu",
            verbose=False,
        )
        if len(results) != len(batch):
            raise RuntimeError(f"Result count mismatch for {model_name}: {len(results)} != {len(batch)}")
        for record, result in zip(batch, results, strict=True):
            rows = result_rows(result)
            predictions[record["image_name"]] = rows
            total_boxes += len(rows)
        print(f"[{model_name}] {min(start + len(batch), len(records))}/{len(records)} images", flush=True)
    return predictions, {
        "model": model_name,
        "weight": str(weight),
        "weight_sha256": sha256_file(weight),
        "images": len(records),
        "predictions_after_model_nms": total_boxes,
    }


def build_thresholds(
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    classes: list[str],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for model_name, by_image in predictions.items():
        per_class: dict[str, Any] = {}
        for class_id, class_name in enumerate(classes):
            eligible = [
                float(row["confidence"])
                for rows in by_image.values()
                for row in rows
                if row["class_id"] == class_id and row["confidence"] >= args.conf_floor
            ]
            q_value = quantile(eligible, args.conf_quantile) if eligible else args.conf_floor
            threshold = max(args.conf_floor, q_value)
            per_class[class_name] = {
                "class_id": class_id,
                "eligible_count_at_conf_floor": len(eligible),
                "quantile": args.conf_quantile,
                "quantile_value": round(q_value, 8),
                "threshold": round(threshold, 8),
            }
        output[model_name] = per_class
    return output


def max_label_iou(box: list[float], labels: list[dict[str, Any]], class_id: int | None) -> tuple[float, int | None]:
    selected = [row for row in labels if class_id is None or row["class_id"] == class_id]
    if not selected:
        return 0.0, None
    scores = [(box_iou(box, row["xyxy"]), row["index"]) for row in selected]
    return max(scores, key=lambda value: value[0])


def make_candidate_rows(
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    records_by_name: dict[str, dict[str, Any]],
    classes: list[str],
    thresholds: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, by_image in predictions.items():
        for image_name, image_predictions in by_image.items():
            record = records_by_name[image_name]
            for prediction_index, prediction in enumerate(image_predictions):
                class_id = int(prediction["class_id"])
                confidence = float(prediction["confidence"])
                threshold = float(thresholds[model_name][classes[class_id]]["threshold"])
                if confidence < threshold:
                    continue
                same_iou, same_index = max_label_iou(prediction["xyxy"], record["labels"], class_id)
                all_iou, _ = max_label_iou(prediction["xyxy"], record["labels"], None)
                if same_iou >= args.match_iou:
                    continue
                x1, y1, x2, y2 = prediction["xyxy"]
                width, height = record["width"], record["height"]
                rows.append(
                    {
                        "candidate_id": f"{model_name}__{Path(image_name).stem}__p{prediction_index:04d}",
                        "source_model": model_name,
                        "image_name": image_name,
                        "image_path": str(record["image_path"]),
                        "image_width": width,
                        "image_height": height,
                        "class_id": class_id,
                        "class_name": classes[class_id],
                        "confidence": round(confidence, 8),
                        "threshold_used": round(threshold, 8),
                        "x1": round(x1, 3),
                        "y1": round(y1, 3),
                        "x2": round(x2, 3),
                        "y2": round(y2, 3),
                        "cx": round((x1 + x2) / 2.0 / width, 8),
                        "cy": round((y1 + y2) / 2.0 / height, 8),
                        "w": round((x2 - x1) / width, 8),
                        "h": round((y2 - y1) / height, 8),
                        "same_class_max_iou": round(same_iou, 8),
                        "all_class_max_iou": round(all_iou, 8),
                        "matched_same_class_label_index": same_index if same_index is not None else "",
                        "auto_label_count": len(record["labels"]),
                        "candidate_rule": f"conf>={threshold:.8f}; same_class_iou<{args.match_iou:.2f}",
                        "sampled": False,
                        "sample_rank": "",
                    }
                )
    rows.sort(key=lambda row: (int(row["class_id"]), row["source_model"], row["image_name"], -float(row["confidence"])))
    return rows


def sample_candidates(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["class_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for class_id in sorted(by_class):
        class_rows = by_class[class_id]
        rng = random.Random(args.seed + class_id)
        chosen_ids: set[str] = set()
        for source_model in ("e3_p2", "e4_copypaste"):
            pool = [row for row in class_rows if row["source_model"] == source_model]
            if len(pool) > args.source_quota:
                pool = rng.sample(pool, args.source_quota)
            for row in pool:
                chosen_ids.add(row["candidate_id"])
        remaining = [row for row in class_rows if row["candidate_id"] not in chosen_ids]
        capacity = max(0, args.per_class_cap - len(chosen_ids))
        if len(remaining) > capacity:
            remaining = rng.sample(remaining, capacity)
        chosen_ids.update(row["candidate_id"] for row in remaining)
        class_selected = [row for row in class_rows if row["candidate_id"] in chosen_ids]
        class_selected.sort(key=lambda row: (row["source_model"], row["image_name"], -float(row["confidence"])))
        for rank, row in enumerate(class_selected, start=1):
            row["sampled"] = True
            row["sample_rank"] = rank
            selected.append(row)
    selected.sort(key=lambda row: (int(row["class_id"]), row["source_model"], row["image_name"], int(row["sample_rank"])))
    return selected


def materialize_review_dataset(
    output_dir: Path,
    sampled_rows: list[dict[str, Any]],
    classes: list[str],
) -> dict[str, Any]:
    review_root = output_dir / "review_dataset"
    images_dir = review_root / "images"
    labels_dir = review_root / "labels_with_conf"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (review_root / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    names = {row["image_name"]: row for row in sampled_rows}
    rows_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sampled_rows:
        rows_by_image[row["image_name"]].append(row)
    materialization_modes: Counter[str] = Counter()
    for image_name, image_rows in sorted(rows_by_image.items()):
        source = Path(image_rows[0]["image_path"])
        target = images_dir / image_name
        try:
            os.link(source, target)
            materialization_modes["hardlink"] += 1
        except OSError:
            shutil.copy2(source, target)
            materialization_modes["copy"] += 1
        label_path = labels_dir / f"{Path(image_name).stem}.txt"
        with label_path.open("w", encoding="utf-8") as handle:
            for row in sorted(image_rows, key=lambda item: (int(item["sample_rank"]), item["candidate_id"])):
                handle.write(
                    f"{int(row['class_id'])} {float(row['cx']):.8f} {float(row['cy']):.8f} "
                    f"{float(row['w']):.8f} {float(row['h']):.8f} {float(row['confidence']):.8f}\n"
                )
    json_dump(
        review_root / "sample_meta.json",
        {
            "images": [
                {
                    "name": image_name,
                    "candidate_count": len(rows_by_image[image_name]),
                    "candidate_ids": [row["candidate_id"] for row in rows_by_image[image_name]],
                }
                for image_name in sorted(rows_by_image)
            ]
        },
    )
    data_yaml = (
        f'path: "{review_root.as_posix()}"\n'
        'train: "images"\n'
        'val: "images"\n'
        f"nc: {len(classes)}\n"
        "names:\n"
        + "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
        + "\n"
    )
    (review_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    return {
        "review_root": str(review_root),
        "images": len(rows_by_image),
        "boxes": len(sampled_rows),
        "materialization_modes": dict(materialization_modes),
        "labels_with_conf": str(labels_dir),
    }


def write_readme(output_dir: Path, manifest: dict[str, Any], sampled_count: int) -> None:
    text = f"""# L1 E5a FP Autopsy

本目录是 Fold 1 的零 GPU、高置信未匹配候选审查产物，不是人工真值、Golden Set 或模型排名报告。

- 范围：`dianjienie_ryw18_oofconsensus_v1_dev_only_r1` 的 Fold 1 验证图 376 张
- 模型：E3 P2 回收权重、E4 Copy-Paste 回收权重
- 预注册参数：见 `pre_registration_manifest.json`
- 逐类阈值：见 `derived_class_thresholds.json`，由预注册的 q90 规则在全量推理后计算
- 全量未匹配候选：`candidate_boxes_all.csv`
- 抽样候选：`candidate_boxes_sampled.csv`（本次 {sampled_count} 个框）
- 审查视图：`review_dataset/`；审查页由现有 `generate_box_roi_review.py` 生成

审查页状态映射：`keep` = 真缺陷，`remove` = 非缺陷，`uncertain` = 存疑。E5a 闸门只在用户完成抽查并据此统计后判定。

禁止事项：不覆盖原始图片或标签，不读取 `release-validation`，不将本目录直接用于训练，不把 E5a 结果作为 E3/E4 排名证据。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_root, val_list, e3_weight, e4_weight, classes, records = validate_config(args)
    output_dir = (args.output_dir or (DEFAULT_OUTPUT_PARENT / f"l1_e5a_fp_autopsy_{datetime.now():%Y%m%d_%H%M%S}")).resolve()
    input_summary = {
        "data_root": str(data_root),
        "val_list": str(val_list),
        "val_list_sha256": sha256_file(val_list),
        "data_yaml": str(data_root / "data.yaml"),
        "data_yaml_sha256": sha256_file(data_root / "data.yaml"),
        "classes_file": str(data_root / "classes.txt"),
        "classes_sha256": sha256_file(data_root / "classes.txt"),
        "images": len(records),
        "auto_label_boxes": sum(len(record["labels"]) for record in records),
        "weights": {
            "e3_p2": {"path": str(e3_weight), "sha256": sha256_file(e3_weight)},
            "e4_copypaste": {"path": str(e4_weight), "sha256": sha256_file(e4_weight)},
        },
    }
    if args.dry_run:
        print(json.dumps({"status": "dry_run_pass", "output_dir": str(output_dir), "input": input_summary}, ensure_ascii=False, indent=2))
        return 0
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_registration = {
        "schema_version": 1,
        "status": "pre_registered_before_inference",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "L1 E5a E3/E4 high-confidence FP visual autopsy",
        "development_only": True,
        "frozen_test_set": None,
        "scope": {
            "fold": 1,
            "images": 376,
            "validation_list": str(val_list),
            "forbidden_sources": ["release-validation"],
        },
        "registered_parameters": {
            "inference": {
                "device": "cpu",
                "imgsz": args.imgsz,
                "batch": args.batch,
                "conf_floor": args.inference_conf_floor,
                "nms_iou": args.nms_iou,
                "max_det": args.max_det,
            },
            "candidate": {
                "confidence_floor": args.conf_floor,
                "confidence_quantile": args.conf_quantile,
                "match_iou": args.match_iou,
                "match_scope": "same_class_only",
            },
            "sampling": {
                "seed": args.seed,
                "per_class_cap": args.per_class_cap,
                "source_quota_per_class": {
                    "e3_p2": args.source_quota,
                    "e4_copypaste": args.source_quota,
                },
                "strategy": "per-class pooled cap; source quotas first; deterministic fill from remaining candidates",
            },
            "review_decisions": {
                "keep": "true_defect",
                "remove": "not_defect",
                "uncertain": "uncertain",
            },
        },
        "input": input_summary,
        "non_claims": [
            "not_manual_truth",
            "not_Golden_Set",
            "not_model_ranking_evidence",
            "not_training_labels",
        ],
    }
    json_dump(output_dir / "pre_registration_manifest.json", pre_registration)
    json_dump(output_dir / "validation_input_summary.json", input_summary)

    records_by_name = {record["image_name"]: record for record in records}
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    inference_summaries = []
    for model_name, weight in (("e3_p2", e3_weight), ("e4_copypaste", e4_weight)):
        by_image, summary = infer_model(model_name, weight, records, args, classes)
        predictions[model_name] = by_image
        inference_summaries.append(summary)
    json_dump(output_dir / "inference_summary.json", {"models": inference_summaries})

    thresholds = build_thresholds(predictions, classes, args)
    json_dump(
        output_dir / "derived_class_thresholds.json",
        {
            "schema_version": 1,
            "derived_after_inference_before_sampling": True,
            "rule": f"threshold=max(conf_floor={args.conf_floor}, q{args.conf_quantile:.2f} among predictions at or above conf_floor)",
            "thresholds": thresholds,
        },
    )
    all_rows = make_candidate_rows(predictions, records_by_name, classes, thresholds, args)
    sampled_rows = sample_candidates(all_rows, args)
    sampled_ids = {row["candidate_id"] for row in sampled_rows}
    for row in all_rows:
        row["sampled"] = row["candidate_id"] in sampled_ids
        if not row["sampled"]:
            row["sample_rank"] = ""
    write_csv(output_dir / "candidate_boxes_all.csv", all_rows)
    write_csv(output_dir / "candidate_boxes_sampled.csv", sampled_rows)
    review_summary = materialize_review_dataset(output_dir, sampled_rows, classes)
    stats_by_class: dict[str, Any] = {}
    for class_id, class_name in enumerate(classes):
        all_class = [row for row in all_rows if int(row["class_id"]) == class_id]
        sampled_class = [row for row in sampled_rows if int(row["class_id"]) == class_id]
        stats_by_class[class_name] = {
            "class_id": class_id,
            "all_candidates": len(all_class),
            "sampled_candidates": len(sampled_class),
            "all_by_source": dict(Counter(row["source_model"] for row in all_class)),
            "sampled_by_source": dict(Counter(row["source_model"] for row in sampled_class)),
        }
    stats = {
        "images": len(records),
        "auto_label_boxes": sum(len(record["labels"]) for record in records),
        "all_candidate_boxes": len(all_rows),
        "sampled_candidate_boxes": len(sampled_rows),
        "candidate_images": len({row["image_name"] for row in all_rows}),
        "sampled_images": len({row["image_name"] for row in sampled_rows}),
        "by_source": {
            source: {
                "all_candidates": sum(1 for row in all_rows if row["source_model"] == source),
                "sampled_candidates": sum(1 for row in sampled_rows if row["source_model"] == source),
            }
            for source in ("e3_p2", "e4_copypaste")
        },
        "by_class": stats_by_class,
        "review_dataset": review_summary,
        "gate_status": "pending_user_review",
    }
    json_dump(output_dir / "sample_statistics.json", stats)
    write_readme(output_dir, pre_registration, len(sampled_rows))
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "statistics": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
