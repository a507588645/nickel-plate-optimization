#!/usr/bin/env python3
"""
Extract grade features with ryw.pt (YOLOv5, 18 classes) at imgsz=640,
conf=0.25, iou=0.45, CPU on the 1072-image dianjinie grade classification view.
Join 20 detector-independent texture columns from features_v3_full.csv.

Schema for Arm 4 (18 classes: RYW18):
- 43 base detection columns:
    A_ratio, B_ratio, C_ratio, Airhole_ratio, has_nodule,
    total_bbox_union_image_ratio, detection_count,
    count__<18 classes>, area_ratio__<18 classes>
- 20 texture columns (tx__*) joined from features_v3_full.csv
Total 63 feature columns (identical schema to Arm 1 ryw.pt@1536 and Arm 2 YOLO11m@640).
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
from train_dianji_grade_v3 import BASE_COLUMNS


ROOT = Path(__file__).resolve().parent
DEFAULT_VIEW_ROOT = Path(r"E:/PythonProject/archives/dianjinie_grade_classification_v1")
DEFAULT_TEXTURE_FEATURES = ROOT / "dianji_grade_v3" / "features_v3_full.csv"
DEFAULT_OUTPUT_DIR = ROOT / "dianji_grade_detector_compare_v1" / "arm4_ryw640"
DEFAULT_WEIGHTS = Path(r"E:/PythonProject/Tools/ModelAutoLabelTool/weights/yolov5/ryw.pt")
DEFAULT_YOLOV5_REPO = Path(r"E:/PythonProject/Tools/ModelAutoLabelTool/vendor/yolov5")
DEFAULT_PROJECT_ROOT = Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi")
EXPECTED_WEIGHT_SHA256 = "f404ceda25e083964f84fbe35fad2be9b48ebe42f0bf4b3d11813068542178eb"
STATUS = "unconfirmed_pending_enterprise"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 ryw.pt@640 提取电积镍等级 43 列检测特征，并复用 20 列纹理特征（隔离 imgsz 混淆控制臂）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--view-root", type=Path, default=DEFAULT_VIEW_ROOT)
    parser.add_argument("--texture-features", type=Path, default=DEFAULT_TEXTURE_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--yolov5-repo", type=Path, default=DEFAULT_YOLOV5_REPO)
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
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
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


def load_yolov5_model(weights: Path, yolov5_repo: Path, conf: float, iou: float) -> Any:
    import torch

    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"

    yolo_str = str(yolov5_repo.resolve())
    if yolo_str not in sys.path:
        sys.path.insert(0, yolo_str)

    try:
        import utils.general
        utils.general.check_requirements = lambda *args, **kwargs: None
    except Exception:
        pass

    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _load  # type: ignore[assignment]

    old_cwd = Path.cwd()
    os.chdir(yolo_str)
    try:
        model = torch.hub.load(
            yolo_str,
            "custom",
            path=str(weights.resolve()),
            source="local",
            device="cpu",
            autoshape=True,
            _verbose=False,
        )
    finally:
        os.chdir(old_cwd)

    model.eval()
    if hasattr(model, "conf"):
        model.conf = float(conf)
    if hasattr(model, "iou"):
        model.iou = float(iou)

    actual_names = list(model.names.values()) if hasattr(model, "names") else []
    if actual_names != CLASS_NAMES:
        raise ValueError(
            f"ryw.pt 类别列表不匹配: expected={CLASS_NAMES!r}, actual={actual_names!r}"
        )
    return model


def predict_yolov5_detections(
    model: Any,
    image_bgr: np.ndarray,
    image_width: int,
    image_height: int,
    imgsz: int,
    conf_thres: float,
) -> list[dict[str, Any]]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model(image_bgr, size=imgsz)

    xyxy_tensor = results.xyxy[0] if len(results.xyxy) else None
    if xyxy_tensor is None or len(xyxy_tensor) == 0:
        return []

    dets_arr = xyxy_tensor.cpu().numpy()
    detections: list[dict[str, Any]] = []
    for det in dets_arr:
        x1, y1, x2, y2, conf, cls = det[:6]
        cls_id = int(cls)
        conf_f = float(conf)
        if conf_f < conf_thres:
            continue
        if cls_id < 0 or cls_id >= len(CLASS_NAMES):
            raise ValueError(f"ryw.pt 输出非法类别 id: {cls_id}")
        bw = max(0.0, float(x2 - x1))
        bh = max(0.0, float(y2 - y1))
        if bw <= 0 or bh <= 0:
            continue
        cx = (float(x1) + float(x2)) / 2.0 / image_width
        cy = (float(y1) + float(y2)) / 2.0 / image_height
        detections.append(
            {
                "label": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "conf": conf_f,
                "cx": cx,
                "cy": cy,
                "w": bw / image_width,
                "h": bh / image_height,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )
    return detections


def base_feature_row_ryw(
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
        "feature_version": "ryw640_base43",
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
            raise ValueError(f"聚合结果缺少 BASE 列: {column}")
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


def run_extraction(args: argparse.Namespace) -> dict[str, Any]:
    view_root = args.view_root.resolve()
    manifest_rows = load_manifest(view_root)
    selected_rows = select_manifest_rows(manifest_rows, args.split, args.limit)
    resolved_pairs = validate_image_paths(selected_rows, view_root)

    weight_path = args.weights.resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(f"未找到权重文件: {weight_path}")
    actual_weight_sha256 = file_sha256(weight_path)
    if actual_weight_sha256 != EXPECTED_WEIGHT_SHA256:
        raise ValueError(
            f"ryw.pt SHA-256 不匹配: expected={EXPECTED_WEIGHT_SHA256}, actual={actual_weight_sha256}"
        )

    if not args.texture_features.is_file():
        raise FileNotFoundError(f"未找到纹理特征文件: {args.texture_features}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        texture_sample = read_csv(args.texture_features)
        tx_cols = texture_columns(texture_sample)
        summary = {
            "dry_run": True,
            "status": STATUS,
            "selected_rows": len(selected_rows),
            "manifest_total": len(manifest_rows),
            "weights": str(weight_path),
            "weights_sha256": actual_weight_sha256,
            "base_columns_count": len(BASE_COLUMNS),
            "texture_columns_count": len(tx_cols),
            "total_feature_count": len(BASE_COLUMNS) + len(tx_cols),
        }
        write_json(args.output_dir / "extraction_summary.json", summary)
        print(f"[dry-run] 校验通过，待处理图像: {len(selected_rows)}")
        return summary

    model = load_yolov5_model(
        weights=weight_path,
        yolov5_repo=args.yolov5_repo,
        conf=args.conf,
        iou=args.iou,
    )

    base_feature_rows: list[dict[str, Any]] = []
    per_box_rows: list[dict[str, Any]] = []
    fallback_count = 0
    start_time = time.time()

    print(f"=== 开始 ryw.pt@640 特征提取 ({len(resolved_pairs)} 张图像) ===")
    for index, (manifest_row, image_path) in enumerate(resolved_pairs, start=1):
        image_bgr = read_image_bgr(image_path)
        if image_bgr is None:
            raise RuntimeError(f"图像无法读取: {image_path}")
        image_height, image_width = image_bgr.shape[:2]
        plate_area, plate_area_fallback = compute_plate_area(image_bgr, args.project_root)
        if plate_area_fallback:
            fallback_count += 1

        detections = predict_yolov5_detections(
            model=model,
            image_bgr=image_bgr,
            image_width=image_width,
            image_height=image_height,
            imgsz=args.imgsz,
            conf_thres=args.conf,
        )

        for det in detections:
            per_box_rows.append(
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
                    "label": det["label"],
                    "class_name": det["class_name"],
                    "conf": f"{det['conf']:.6f}",
                    "x1": f"{det['x1']:.2f}",
                    "y1": f"{det['y1']:.2f}",
                    "x2": f"{det['x2']:.2f}",
                    "y2": f"{det['y2']:.2f}",
                    "cx": f"{det['cx']:.6f}",
                    "cy": f"{det['cy']:.6f}",
                    "w": f"{det['w']:.6f}",
                    "h": f"{det['h']:.6f}",
                    "image_width": image_width,
                    "image_height": image_height,
                    "plate_area": f"{plate_area:.2f}",
                    "plate_area_fallback": int(plate_area_fallback),
                    "infer_img_size": args.imgsz,
                    "detector_conf": args.conf,
                    "detector_iou": args.iou,
                }
            )

        row = base_feature_row_ryw(
            manifest_row=manifest_row,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            plate_area=plate_area,
            plate_area_fallback=plate_area_fallback,
            detections=detections,
            weight_sha256=actual_weight_sha256,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            project_root=args.project_root,
        )
        base_feature_rows.append(row)

        if index % args.log_every == 0 or index == len(resolved_pairs):
            elapsed = time.time() - start_time
            rate = index / elapsed if elapsed > 0 else 0.0
            print(f"  [{index}/{len(resolved_pairs)}] {rate:.2f} img/s - {elapsed / 60.0:.1f} min")

    joined_rows, texture_meta = join_texture_columns(
        base_rows=base_feature_rows,
        texture_path=args.texture_features.resolve(),
        expected_full_count=(args.limit == 0 and args.split is None),
    )

    tx_cols = texture_meta["texture_columns"]
    out_dir = args.output_dir.resolve()
    base_fields = training_feature_fields([])
    full_fields = training_feature_fields(tx_cols)

    base_only_rows = [
        {k: row[k] for k in base_fields if k in row}
        for row in base_feature_rows
    ]

    base_csv_path = out_dir / "features_ryw640_base43.csv"
    full_csv_path = out_dir / "features_ryw640_v1.csv"
    per_box_csv_path = out_dir / "per_box_ryw640.csv"

    write_csv(base_csv_path, base_only_rows, base_fields)
    write_csv(full_csv_path, joined_rows, full_fields)
    write_csv(per_box_csv_path, per_box_rows, per_box_fields())

    box_counter = Counter(r["class_name"] for r in per_box_rows)
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": STATUS,
        "weights": str(weight_path),
        "weights_sha256": actual_weight_sha256,
        "images_processed": len(base_feature_rows),
        "manifest_total": len(manifest_rows),
        "total_boxes_detected": len(per_box_rows),
        "detections_per_class": dict(box_counter),
        "plate_area_fallbacks": fallback_count,
        "base_columns_count": len(BASE_COLUMNS),
        "texture_columns_count": len(tx_cols),
        "total_feature_columns": len(BASE_COLUMNS) + len(tx_cols),
        "texture_join": texture_meta,
        "output_files": {
            "features_base43_csv": str(base_csv_path),
            "features_full_csv": str(full_csv_path),
            "per_box_csv": str(per_box_csv_path),
        },
    }
    write_json(out_dir / "extraction_summary.json", summary)
    print(f"=== 提取完成: {len(base_feature_rows)} 图, {len(per_box_rows)} 框, 产物写入 {out_dir} ===")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_extraction(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
