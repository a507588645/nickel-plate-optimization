#!/usr/bin/env python3
"""
Re-extract detector-derived grade features with the frozen YOLO11m 640 ONNX
model and join the detector-independent texture columns.

The 43 detector columns deliberately reuse the historical aggregation
functions from extract_dianji_grade_features.py.  The only detector-side
change is the Ultralytics YOLO11m ONNX inference configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from extract_dianji_grade_features import (
    CLASS_NAMES,
    compute_plate_area,
    features_from_detections,
    read_image_bgr,
)
from train_dianji_grade_v3 import BASE_COLUMNS, GRADES, GRADE_TO_ID


ROOT = Path(__file__).resolve().parent
DEFAULT_VIEW_ROOT = Path(r"E:/PythonProject/archives/dianjinie_grade_classification_v1")
DEFAULT_TEXTURE_FEATURES = ROOT / "dianji_grade_v3" / "features_v3_full.csv"
DEFAULT_OUTPUT_DIR = ROOT / "dianji_grade_y11m640_v1"
DEFAULT_WEIGHTS = Path(
    r"E:/PythonProject/models/runs/"
    r"dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_finaldev_ep083/"
    r"exports/onnx/final_fixed_epoch.onnx"
)
DEFAULT_PROJECT_ROOT = Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi")
EXPECTED_WEIGHT_SHA256 = (
    "901ccf6006f44689483aaf6f0d1d1b8c18132a85c7d881be7e62fcdccc255b21"
)
STATUS = "unconfirmed_pending_enterprise"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用冻结 YOLO11m@640 重新提取电积镍等级 43 列检测特征，并复用 20 列纹理特征。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--view-root", type=Path, default=DEFAULT_VIEW_ROOT)
    parser.add_argument("--texture-features", type=Path, default=DEFAULT_TEXTURE_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--imgsz", "--img-size", dest="imgsz", type=int, default=640)
    parser.add_argument("--conf", "--conf-thres", dest="conf", type=float, default=0.25)
    parser.add_argument("--iou", "--iou-thres", dest="iou", type=float, default=0.45)
    parser.add_argument("--split", choices=["train", "valid", "test"], default=None)
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 张，0 表示全部。")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true", help="只校验路径和 schema，不加载模型或推理。")
    return parser.parse_args(argv)


def normalize_key(value: Any) -> str:
    return str(value or "").strip().lower()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def path_is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_manifest(view_root: Path) -> list[dict[str, str]]:
    manifest_path = view_root / "manifests" / "samples.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = read_csv(manifest_path)
    required = {
        "image_name",
        "image_stem",
        "grade",
        "split",
        "dup_group_id",
        "sha256",
        "link_relpath",
    }
    missing = sorted(required.difference(rows[0] if rows else {}))
    if missing:
        raise ValueError(f"manifest 缺少字段: {missing}")
    if len(rows) != 1072:
        raise ValueError(f"归档 manifest 行数不是 1072: {len(rows)}")
    if len({normalize_key(row["sha256"]) for row in rows}) != len(rows):
        raise ValueError("归档 manifest SHA-256 不唯一")
    if len({row["image_name"] for row in rows}) != len(rows):
        raise ValueError("归档 manifest image_name 不唯一")
    return rows


def select_manifest_rows(
    manifest: list[dict[str, str]],
    split: str | None,
    limit: int,
) -> list[dict[str, str]]:
    rows = [row for row in manifest if split is None or row["split"] == split]
    if limit < 0:
        raise ValueError("--limit 不能小于 0")
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("没有可处理的 manifest 行")
    return rows


def validate_image_paths(
    rows: list[dict[str, str]],
    view_root: Path,
) -> list[tuple[dict[str, str], Path]]:
    root = view_root.resolve()
    resolved: list[tuple[dict[str, str], Path]] = []
    for row in rows:
        relative = row.get("link_relpath", "").strip()
        if not relative:
            raise ValueError(f"manifest 缺少 link_relpath: {row.get('image_name')}")
        image_path = (view_root / relative).resolve()
        if not path_is_inside(image_path, root):
            raise ValueError(f"图像路径越出归档根目录: {image_path}")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        resolved.append((row, image_path))
    return resolved


def texture_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        raise ValueError("纹理源 CSV 为空")
    fields = list(rows[0])
    columns = [
        field
        for field in fields
        if field.startswith("tx__") and field != "tx__texture_ok"
    ]
    if len(columns) != 20:
        raise ValueError(f"训练用纹理列数量不是 20: {len(columns)}")
    return columns


def build_unique_index(
    rows: list[dict[str, str]],
    key: str,
    source_name: str,
) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        value = normalize_key(row.get(key, ""))
        if not value:
            raise ValueError(f"{source_name} 的 {key} 存在空值")
        if value in index:
            raise ValueError(f"{source_name} 的 {key} 不唯一: {value}")
        index[value] = row
    return index


def join_texture_columns(
    base_rows: list[dict[str, Any]],
    texture_path: Path,
    expected_full_count: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not texture_path.is_file():
        raise FileNotFoundError(texture_path)
    texture_rows = read_csv(texture_path)
    if expected_full_count and len(texture_rows) != 1072:
        raise ValueError(f"纹理源行数不是 1072: {len(texture_rows)}")
    tx_cols = texture_columns(texture_rows)
    texture_fields = set(texture_rows[0])
    if "sha256" in texture_fields and all(normalize_key(row.get("sha256")) for row in texture_rows):
        key = "sha256"
    elif "image_stem" in texture_fields:
        key = "image_stem"
    else:
        raise ValueError("纹理源没有可用的 sha256 或 image_stem 身份列")

    texture_index = build_unique_index(texture_rows, key, "纹理源")
    joined: list[dict[str, Any]] = []
    mismatches: list[dict[str, str]] = []
    for base in base_rows:
        lookup = normalize_key(base.get(key, ""))
        texture = texture_index.get(lookup)
        if texture is None:
            mismatches.append(
                {"image_name": str(base.get("image_name", "")), "reason": f"missing_{key}"}
            )
            continue
        if normalize_key(base.get("image_stem")) != normalize_key(texture.get("image_stem")):
            mismatches.append(
                {"image_name": str(base.get("image_name", "")), "reason": "image_stem_mismatch"}
            )
            continue
        if base.get("grade") != texture.get("grade") or base.get("split") != texture.get("split"):
            mismatches.append(
                {"image_name": str(base.get("image_name", "")), "reason": "grade_or_split_mismatch"}
            )
            continue
        row = dict(base)
        for column in tx_cols:
            value = texture.get(column, "")
            if value == "":
                mismatches.append(
                    {"image_name": str(base.get("image_name", "")), "reason": f"empty_{column}"}
                )
                break
            row[column] = value
        else:
            joined.append(row)

    if mismatches:
        preview = mismatches[:5]
        raise ValueError(f"纹理 join 存在 {len(mismatches)} 个不匹配: {preview}")
    if len(joined) != len(base_rows):
        raise ValueError(f"纹理 join 行数不一致: {len(joined)} != {len(base_rows)}")

    return joined, {
        "source_path": str(texture_path.resolve()),
        "join_key": key,
        "source_rows": len(texture_rows),
        "matched_rows": len(joined),
        "mismatch_rows": len(mismatches),
        "texture_columns": tx_cols,
    }


def model_names(model: Any) -> list[str]:
    raw = getattr(model, "names", None)
    if isinstance(raw, dict):
        names = [raw.get(i, raw.get(str(i), "")) for i in range(len(raw))]
    else:
        names = list(raw or [])
    return [str(name) for name in names]


def load_yolo_model(weights: Path) -> Any:
    # Prevent Ultralytics from installing packages or selecting a CUDA provider.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    model.overrides["device"] = "cpu"
    names = model_names(model)
    if names != CLASS_NAMES:
        raise ValueError(
            "ONNX 类别顺序不匹配: "
            f"expected={CLASS_NAMES!r}, actual={names!r}"
        )
    return model


def predict_detections(
    model: Any,
    image_bgr: np.ndarray,
    image_width: int,
    image_height: int,
    imgsz: int,
    conf: float,
    iou: float,
) -> list[dict[str, Any]]:
    results = model.predict(
        source=image_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device="cpu",
        verbose=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"单图推理结果数量异常: {len(results)}")
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy()
    detections: list[dict[str, Any]] = []
    for coords, confidence, class_id_float in zip(xyxy, confidences, class_ids):
        class_id = int(class_id_float)
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"ONNX 输出非法类别 id: {class_id}")
        x1, y1, x2, y2 = [float(value) for value in coords[:4]]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0 or height <= 0:
            continue
        cx = (x1 + x2) / 2.0 / image_width
        cy = (y1 + y2) / 2.0 / image_height
        detections.append(
            {
                "label": class_id,
                "class_name": CLASS_NAMES[class_id],
                "conf": float(confidence),
                "cx": cx,
                "cy": cy,
                "w": width / image_width,
                "h": height / image_height,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return detections


def base_feature_row(
    manifest_row: dict[str, str],
    image_path: Path,
    image_width: int,
    image_height: int,
    plate_area: float,
    plate_area_fallback: bool,
    detections: list[dict[str, Any]],
    weight_sha256: str,
    imgsz: int,
    conf: float,
    iou: float,
    project_root: Path,
) -> dict[str, Any]:
    features = features_from_detections(
        detections=detections,
        image_width=image_width,
        image_height=image_height,
        plate_area=plate_area,
        conf_thres=conf,
    )
    row: dict[str, Any] = {
        "image_name": manifest_row["image_name"],
        "image_stem": manifest_row["image_stem"],
        "image_name_raw": manifest_row.get("image_name_raw", ""),
        "image_path": str(image_path),
        "image_path_view": manifest_row["link_relpath"],
        "source_dataset": manifest_row.get("source_dataset", ""),
        "source_split": manifest_row.get("source_split", ""),
        "grade": manifest_row["grade"],
        "true_grade": manifest_row["grade"],
        "split": manifest_row["split"],
        "dup_group_id": manifest_row.get("dup_group_id", ""),
        "sha256": manifest_row["sha256"],
        "feature_version": "y11m640_base43",
        "infer_img_size": imgsz,
        "feature_conf_thres": conf,
        "detector_conf": conf,
        "detector_iou": iou,
        "detector_weight_sha256": weight_sha256,
        "plate_area_fallback": int(plate_area_fallback),
        "project_root_for_plate_area": str(project_root.resolve()),
        "status": STATUS,
        "image_width": image_width,
        "image_height": image_height,
        "plate_area": plate_area,
        "max_confidence": features["max_confidence"],
    }
    for column in BASE_COLUMNS:
        if column not in features:
            raise ValueError(f"历史聚合结果缺少 BASE 列: {column}")
        row[column] = features[column]
    return row


def per_box_fields() -> list[str]:
    return [
        "image_name",
        "image_stem",
        "image_path",
        "image_path_view",
        "source_dataset",
        "source_split",
        "grade",
        "split",
        "dup_group_id",
        "sha256",
        "label",
        "class_name",
        "conf",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "w",
        "h",
        "image_width",
        "image_height",
        "plate_area",
        "plate_area_fallback",
        "infer_img_size",
        "detector_conf",
        "detector_iou",
    ]


def training_feature_fields(texture_cols: list[str]) -> list[str]:
    metadata = [
        "image_name",
        "image_stem",
        "image_name_raw",
        "image_path",
        "image_path_view",
        "source_dataset",
        "source_split",
        "grade",
        "true_grade",
        "split",
        "dup_group_id",
        "sha256",
        "feature_version",
        "infer_img_size",
        "feature_conf_thres",
        "detector_conf",
        "detector_iou",
        "detector_weight_sha256",
        "plate_area_fallback",
        "image_width",
        "image_height",
        "plate_area",
        "max_confidence",
        "status",
    ]
    return metadata + list(BASE_COLUMNS) + texture_cols


def validate_inputs(
    args: argparse.Namespace,
    selected_rows: list[dict[str, str]],
) -> dict[str, Any]:
    if not args.view_root.is_dir():
        raise FileNotFoundError(args.view_root)
    if not args.texture_features.is_file():
        raise FileNotFoundError(args.texture_features)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    resolved_images = validate_image_paths(selected_rows, args.view_root)
    texture_rows = read_csv(args.texture_features)
    tx_cols = texture_columns(texture_rows)
    if args.imgsz <= 0:
        raise ValueError("--imgsz 必须为正数")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf 必须在 [0, 1] 内")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou 必须在 [0, 1] 内")
    weight_sha256 = file_sha256(args.weights)
    if weight_sha256 != EXPECTED_WEIGHT_SHA256:
        raise ValueError(
            f"权重 SHA-256 不匹配: expected={EXPECTED_WEIGHT_SHA256}, actual={weight_sha256}"
        )
    return {
        "selected_rows": len(selected_rows),
        "resolved_images": len(resolved_images),
        "texture_rows": len(texture_rows),
        "texture_columns": tx_cols,
        "weight_sha256": weight_sha256,
        "full_view": args.split is None and args.limit == 0,
    }


def run_extraction(
    args: argparse.Namespace,
    selected_rows: list[dict[str, str]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight_sha256 = validation["weight_sha256"]
    resolved_images = validate_image_paths(selected_rows, args.view_root)
    model = load_yolo_model(args.weights)

    box_path = args.output_dir / "per_box_y11m640.csv"
    base_path = args.output_dir / "features_y11m640_base43.csv"
    final_path = args.output_dir / "features_y11m640_v1.csv"
    box_fields = per_box_fields()
    base_rows: list[dict[str, Any]] = []
    total_boxes = 0
    images_with_boxes = 0
    fallback_count = 0
    class_counts: Counter[str] = Counter()
    started = time.time()

    with box_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=box_fields, extrasaction="ignore")
        writer.writeheader()
        for index, (manifest_row, image_path) in enumerate(resolved_images, start=1):
            image_bgr = read_image_bgr(image_path)
            if image_bgr is None:
                raise RuntimeError(f"图像无法读取: {image_path}")
            image_height, image_width = image_bgr.shape[:2]
            plate_area, plate_fallback = compute_plate_area(
                image_bgr,
                args.project_root,
            )
            detections = predict_detections(
                model=model,
                image_bgr=image_bgr,
                image_width=image_width,
                image_height=image_height,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
            )
            if detections:
                images_with_boxes += 1
            total_boxes += len(detections)
            fallback_count += int(plate_fallback)
            for detection in detections:
                class_counts[detection["class_name"]] += 1
                writer.writerow(
                    {
                        "image_name": manifest_row["image_name"],
                        "image_stem": manifest_row["image_stem"],
                        "image_path": str(image_path),
                        "image_path_view": manifest_row["link_relpath"],
                        "source_dataset": manifest_row.get("source_dataset", ""),
                        "source_split": manifest_row.get("source_split", ""),
                        "grade": manifest_row["grade"],
                        "split": manifest_row["split"],
                        "dup_group_id": manifest_row.get("dup_group_id", ""),
                        "sha256": manifest_row["sha256"],
                        "label": detection["label"],
                        "class_name": detection["class_name"],
                        "conf": f"{detection['conf']:.9f}",
                        "x1": f"{detection['x1']:.6f}",
                        "y1": f"{detection['y1']:.6f}",
                        "x2": f"{detection['x2']:.6f}",
                        "y2": f"{detection['y2']:.6f}",
                        "cx": f"{detection['cx']:.9f}",
                        "cy": f"{detection['cy']:.9f}",
                        "w": f"{detection['w']:.9f}",
                        "h": f"{detection['h']:.9f}",
                        "image_width": image_width,
                        "image_height": image_height,
                        "plate_area": f"{plate_area:.9f}",
                        "plate_area_fallback": int(plate_fallback),
                        "infer_img_size": args.imgsz,
                        "detector_conf": args.conf,
                        "detector_iou": args.iou,
                    }
                )
            base_rows.append(
                base_feature_row(
                    manifest_row=manifest_row,
                    image_path=image_path,
                    image_width=image_width,
                    image_height=image_height,
                    plate_area=plate_area,
                    plate_area_fallback=plate_fallback,
                    detections=detections,
                    weight_sha256=weight_sha256,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    project_root=args.project_root,
                )
            )
            if index % args.log_every == 0 or index == len(resolved_images):
                elapsed = time.time() - started
                rate = index / elapsed if elapsed > 0 else 0.0
                print(
                    f"[extract] {index}/{len(resolved_images)} "
                    f"boxes={total_boxes} rate={rate:.2f} img/s"
                )
            handle.flush()

    write_csv(base_path, base_rows, list(base_rows[0]) if base_rows else list(BASE_COLUMNS))
    joined_rows, join_summary = join_texture_columns(
        base_rows=base_rows,
        texture_path=args.texture_features,
        expected_full_count=validation["full_view"],
    )
    texture_cols = join_summary["texture_columns"]
    final_fields = training_feature_fields(texture_cols)
    write_csv(final_path, joined_rows, final_fields)

    if validation["full_view"]:
        if len(base_rows) != 1072 or len(joined_rows) != 1072:
            raise ValueError(
                f"全量特征行数不为 1072: base={len(base_rows)} joined={len(joined_rows)}"
            )
        if join_summary["matched_rows"] != 1072 or join_summary["mismatch_rows"] != 0:
            raise ValueError(f"全量纹理 join 未通过: {join_summary}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": STATUS,
        "view_root": str(args.view_root.resolve()),
        "texture_features": str(args.texture_features.resolve()),
        "weights": str(args.weights.resolve()),
        "weight_sha256": weight_sha256,
        "expected_weight_sha256": EXPECTED_WEIGHT_SHA256,
        "project_root": str(args.project_root.resolve()),
        "inference": {
            "device": "cpu",
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "class_names": CLASS_NAMES,
        },
        "manifest_rows": len(selected_rows),
        "images_processed": len(base_rows),
        "images_with_boxes": images_with_boxes,
        "total_boxes": total_boxes,
        "plate_area_fallback_images": fallback_count,
        "class_counts": dict(class_counts),
        "grade_counts": dict(Counter(row["grade"] for row in selected_rows)),
        "split_counts": dict(Counter(row["split"] for row in selected_rows)),
        "base_columns": list(BASE_COLUMNS),
        "base_column_count": len(BASE_COLUMNS),
        "texture_columns": texture_cols,
        "texture_column_count": len(texture_cols),
        "final_feature_columns": list(BASE_COLUMNS) + texture_cols,
        "final_feature_count": len(BASE_COLUMNS) + len(texture_cols),
        "join": join_summary,
        "outputs": {
            "per_box_csv": str(box_path.resolve()),
            "base_features_csv": str(base_path.resolve()),
            "features_csv": str(final_path.resolve()),
        },
        "elapsed_sec": time.time() - started,
    }
    write_json(args.output_dir / "extraction_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.view_root)
    selected_rows = select_manifest_rows(manifest, args.split, args.limit)
    validation = validate_inputs(args, selected_rows)
    print("=== 电积镍等级 YOLO11m@640 特征提取 ===")
    print(f"view_root={args.view_root.resolve()}")
    print(f"selected={len(selected_rows)} split={args.split or 'all'} limit={args.limit}")
    print(f"weight_sha256={validation['weight_sha256']}")
    print(f"base_columns={len(BASE_COLUMNS)} texture_columns={len(validation['texture_columns'])}")
    print(f"status={STATUS}")
    if args.dry_run:
        print("[dry-run] 路径、权重哈希和 schema 校验通过，跳过模型加载与推理")
        return 0

    summary = run_extraction(args, selected_rows, validation)
    print(
        f"完成: images={summary['images_processed']} boxes={summary['total_boxes']} "
        f"features={summary['outputs']['features_csv']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
