#!/usr/bin/env python3
"""Independently audit an L1 E4 Copy-Paste derived view.

This verifier is read-only by default except for its JSON report. It confirms
the source boundary, fold isolation, hard-linked originals, five-column labels,
and every recorded paste location against the target's original annotations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any

from build_l1_e4_copypaste_dataset import (
    Box,
    Rect,
    box_to_rect,
    inspect_source,
    load_classes,
    parse_label,
    read_split,
    rects_intersect,
    sha256_file,
    validate_source_boundary,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = ROOT / "dianjienie_ryw18_oofconsensus_v1_dev_only_r1"
DEFAULT_OUTPUT_ROOT = ROOT / "l1_e4_oofconsensus_v1_cp_v1_20260823"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="独立审计 L1 E4 Copy-Paste 开发集派生视图的边界、标签和粘贴记录。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, help="默认写入 output-root/manifests/verification_report.json")
    parser.add_argument("--no-write", action="store_true", help="只审计并打印 JSON，不写报告。")
    return parser.parse_args(argv)


def image_name(entry: str) -> str:
    name = PurePosixPath(entry.strip().replace("\\", "/")).name
    if not name or Path(name).suffix.lower() != ".png":
        raise ValueError(f"清单含非 PNG 路径：{entry}")
    return name


def read_list(path: Path) -> list[str]:
    rows = [image_name(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"清单为空或含重复图片：{path}")
    return rows


def box_from_mapping(raw: dict[str, Any]) -> Box:
    required = {"class_id", "x", "y", "width", "height", "line_number"}
    if set(raw) != required:
        raise ValueError("记录中的 YOLO 框字段不完整")
    return Box(
        int(raw["class_id"]),
        float(raw["x"]),
        float(raw["y"]),
        float(raw["width"]),
        float(raw["height"]),
        int(raw["line_number"]),
    )


def rect_from_mapping(raw: dict[str, Any]) -> Rect:
    required = {"left", "top", "right", "bottom"}
    if set(raw) != required:
        raise ValueError("记录中的像素矩形字段不完整")
    return Rect(*(int(raw[name]) for name in ("left", "top", "right", "bottom")))


def close(left: float, right: float, tolerance: float = 1e-7) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def assert_same_box(left: Box, right: Box, context: str) -> None:
    if left.class_id != right.class_id or not all(
        close(first, second) for first, second in zip((left.x, left.y, left.width, left.height), (right.x, right.y, right.width, right.height))
    ):
        raise ValueError(f"标签框不一致：{context}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSONL 无法解析：{path}:{line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"JSONL 行不是对象：{path}:{line_number}")
        records.append(record)
    return records


def audit_fold(
    source_root: Path,
    output_root: Path,
    fold: int,
    dimensions: dict[str, tuple[int, int]],
    source_boxes: dict[str, list[Box]],
    class_count: int,
) -> dict[str, Any]:
    source_train = read_split(source_root, fold, "train")
    source_val = read_split(source_root, fold, "val")
    source_train_set, source_val_set = set(source_train), set(source_val)
    if source_train_set & source_val_set:
        raise ValueError(f"源 Fold {fold} train/val 重叠")
    fold_root = output_root / f"fold_{fold:02d}"
    if not fold_root.is_dir():
        raise FileNotFoundError(f"缺少 Fold 目录：{fold_root}")
    output_train = read_list(fold_root / "splits" / "train.txt")
    output_val = read_list(fold_root / "splits" / "val.txt")
    if output_val != source_val:
        raise ValueError(f"Fold {fold} val 清单发生变化")
    if output_train[:len(source_train)] != source_train:
        raise ValueError(f"Fold {fold} train 原始成员顺序或内容发生变化")
    synthetic_names = output_train[len(source_train):]
    if set(output_train) & set(output_val):
        raise ValueError(f"Fold {fold} 派生 train/val 重叠")
    if any(not name.startswith(f"cp_f{fold:02d}_") for name in synthetic_names):
        raise ValueError(f"Fold {fold} train 清单含未登记的非原始图片")

    manifest_path = fold_root / "manifests" / "synthetic_samples.jsonl"
    records = read_jsonl(manifest_path)
    record_names = [image_name(str(record.get("synthetic_image", ""))) for record in records]
    if record_names != synthetic_names:
        raise ValueError(f"Fold {fold} 合成清单与训练清单不一致")
    if len(record_names) != len(set(record_names)):
        raise ValueError(f"Fold {fold} 合成清单含重名图片")

    expected_original = set(dimensions)
    actual_images = {path.name for path in (fold_root / "images").glob("*.png")}
    actual_labels = {path.with_suffix(".png").name for path in (fold_root / "labels").glob("*.txt")}
    expected_all = expected_original | set(synthetic_names)
    if actual_images != expected_all or actual_labels != expected_all:
        raise ValueError(
            f"Fold {fold} 图/标签集合不一致：images={len(actual_images)} labels={len(actual_labels)} expected={len(expected_all)}"
        )

    original_hardlinks = 0
    for name in sorted(expected_original):
        output_image = fold_root / "images" / name
        source_image = source_root / "images" / name
        if not os.path.samefile(output_image, source_image):
            raise ValueError(f"Fold {fold} 原图不是源图硬链接：{name}")
        original_hardlinks += 1
        output_boxes = parse_label(fold_root / "labels" / Path(name).with_suffix(".txt").name, class_count)
        expected_boxes = source_boxes[name]
        if len(output_boxes) != len(expected_boxes):
            raise ValueError(f"Fold {fold} 原始标签框数改变：{name}")
        for index, (output_box, source_box) in enumerate(zip(output_boxes, expected_boxes), start=1):
            assert_same_box(output_box, source_box, f"fold={fold} image={name} index={index}")

    verified_synthetic = 0
    for record in records:
        synthetic_name = image_name(str(record["synthetic_image"]))
        donor_name = image_name(str(record["source_image"]))
        target_name = image_name(str(record["target_image"]))
        if donor_name not in source_train_set or target_name not in source_train_set:
            raise ValueError(f"Fold {fold} 供体或目标不属于 train：{synthetic_name}")
        if donor_name in source_val_set or target_name in source_val_set or donor_name == target_name:
            raise ValueError(f"Fold {fold} 供体/目标验证泄漏或自粘贴：{synthetic_name}")
        donor_box = box_from_mapping(record["source_box_yolo"])
        donor_box_rect = rect_from_mapping(record["source_box_pixels"])
        donor_crop_rect = rect_from_mapping(record["source_crop_pixels"])
        source_width, source_height = dimensions[donor_name]
        if box_to_rect(donor_box, source_width, source_height) != donor_box_rect:
            raise ValueError(f"Fold {fold} 供体像素框与源标签不一致：{synthetic_name}")
        source_line = donor_box.line_number
        if source_line < 1 or source_line > len(source_boxes[donor_name]):
            raise ValueError(f"Fold {fold} 供体标签行越界：{synthetic_name}")
        assert_same_box(donor_box, source_boxes[donor_name][source_line - 1], f"donor={donor_name}:{source_line}")
        if donor_crop_rect.left > donor_box_rect.left or donor_crop_rect.top > donor_box_rect.top or donor_crop_rect.right < donor_box_rect.right or donor_crop_rect.bottom < donor_box_rect.bottom:
            raise ValueError(f"Fold {fold} 供体 crop 未包含完整框：{synthetic_name}")
        placement = record["target_placement_pixels"]
        patch_rect = rect_from_mapping(placement)
        target_width, target_height = dimensions[target_name]
        if patch_rect.width != donor_crop_rect.width or patch_rect.height != donor_crop_rect.height:
            raise ValueError(f"Fold {fold} patch 尺寸发生缩放：{synthetic_name}")
        if patch_rect.left < 0 or patch_rect.top < 0 or patch_rect.right > target_width or patch_rect.bottom > target_height:
            raise ValueError(f"Fold {fold} patch 越界：{synthetic_name}")
        target_rects = [box_to_rect(box, target_width, target_height) for box in source_boxes[target_name]]
        if any(rects_intersect(patch_rect, target_rect) for target_rect in target_rects):
            raise ValueError(f"Fold {fold} patch 与目标标签相交：{synthetic_name}")
        actual_synthetic_boxes = parse_label(fold_root / "labels" / Path(synthetic_name).with_suffix(".txt").name, class_count)
        expected_target_boxes = source_boxes[target_name]
        if len(actual_synthetic_boxes) != len(expected_target_boxes) + 1:
            raise ValueError(f"Fold {fold} 合成标签框数异常：{synthetic_name}")
        for index, (actual_box, target_box) in enumerate(zip(actual_synthetic_boxes[:-1], expected_target_boxes), start=1):
            assert_same_box(actual_box, target_box, f"synthetic={synthetic_name} target index={index}")
        pasted = box_from_mapping(record["pasted_box_yolo"])
        assert_same_box(actual_synthetic_boxes[-1], pasted, f"synthetic={synthetic_name} pasted")
        pasted_rect = box_to_rect(pasted, target_width, target_height)
        if any(rects_intersect(pasted_rect, target_rect) for target_rect in target_rects):
            raise ValueError(f"Fold {fold} 新增框与目标标签相交：{synthetic_name}")
        expected_width = donor_box_rect.width / target_width
        expected_height = donor_box_rect.height / target_height
        if not close(pasted.width, expected_width) or not close(pasted.height, expected_height):
            raise ValueError(f"Fold {fold} 新增框显示出几何缩放：{synthetic_name}")
        if not (fold_root / "images" / synthetic_name).is_file():
            raise FileNotFoundError(f"Fold {fold} 合成图缺失：{synthetic_name}")
        verified_synthetic += 1

    summary = json.loads((fold_root / "manifests" / "fold_summary.json").read_text(encoding="utf-8"))
    if summary.get("selected_synthetic_total") != verified_synthetic:
        raise ValueError(f"Fold {fold} 摘要合成数量不一致")
    return {
        "fold": fold,
        "original_hardlinks_verified": original_hardlinks,
        "synthetic_samples_verified": verified_synthetic,
        "train_images": len(output_train),
        "val_images": len(output_val),
        "train_val_overlap": 0,
        "manifest_sha256": sha256_file(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root, output_root = args.source_root.resolve(), args.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"派生视图不存在：{output_root}")
    source_manifest = validate_source_boundary(source_root)
    classes = load_classes(source_root)
    dimensions, source_boxes, source_audit = inspect_source(source_root, len(classes))
    build_manifest_path = output_root / "build_manifest.json"
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    if build_manifest.get("development_only") is not True or build_manifest.get("frozen_test_access") != "not read; not linked; not included":
        raise ValueError("派生视图未声明开发集与冻结集边界")
    if build_manifest.get("source_root") != str(source_root):
        raise ValueError("派生视图记录的 source_root 不匹配")
    if build_manifest.get("source_view_manifest_sha256") != sha256_file(source_root / "view_manifest.json"):
        raise ValueError("源 view_manifest 哈希变化或记录不匹配")
    if build_manifest.get("source_audit", {}).get("source_label_tree_sha256") != source_audit["source_label_tree_sha256"]:
        raise ValueError("源标签树哈希变化或记录不匹配")
    folds = sorted(int(value) for value in build_manifest.get("folds", {}))
    if folds != [1, 2, 3]:
        raise ValueError(f"派生视图 Fold 集合错误：{folds}")
    fold_reports = [audit_fold(source_root, output_root, fold, dimensions, source_boxes, len(classes)) for fold in folds]
    report = {
        "status": "passed",
        "stage": "E4 Copy-Paste postbuild verification",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "development_only": True,
        "frozen_test_access": "not read; not linked; not included",
        "auto_only": True,
        "source_view_name": source_manifest.get("view_name"),
        "source_label_tree_sha256": source_audit["source_label_tree_sha256"],
        "build_manifest_sha256": sha256_file(build_manifest_path),
        "folds": fold_reports,
        "totals": {
            "original_hardlinks_verified": sum(item["original_hardlinks_verified"] for item in fold_reports),
            "synthetic_samples_verified": sum(item["synthetic_samples_verified"] for item in fold_reports),
        },
        "checks": [
            "source view declares development_only and frozen test isolation",
            "all original output images are NTFS hard links to source development images",
            "all source and derived labels parse as valid five-column YOLO labels",
            "each Fold keeps its source val list unchanged and appends synthetic images only to train",
            "every donor and target is a source train member for that same Fold",
            "each recorded patch and pasted box has zero intersection with all original target labels",
            "pasted dimensions equal donor pixel-box dimensions at target normalization, proving no resize",
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.no_write:
        report_path = (args.report_path or output_root / "manifests" / "verification_report.json").resolve()
        if output_root not in report_path.parents:
            raise ValueError("--report-path 必须位于派生输出目录内")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[report] {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
