#!/usr/bin/env python3
"""Apply completed human reannotation records into a new, immutable label version.

This script accepts JSONL exported by ``build_manual_reannotation_review.py``.
It never changes the union dataset's original ``labels/`` directory. Completed
images receive five-column YOLO labels in a versioned directory below the
current optimization workspace; an explicitly completed image with no boxes
gets an empty label file, which means a human-confirmed negative sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_UNION = Path(r"E:/PythonProject/datasets/union_original_labels_v1")
DEFAULT_OUTPUT = ROOT / "manual_reannotation_v1" / "labels_human_reviewed_v1"

CLASSES = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将人工重标 JSONL 校验后导出为新 YOLO 五列标签版本（不覆盖原 labels/）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--review-jsonl", type=Path, required=True,
                        help="人工重标页导出的 manual_reannotation_result.jsonl")
    parser.add_argument("--union", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--scope-file", type=Path, default=None,
                        help="人工重标页生成的 review_scope.txt；与 --require-completed 一起确保整个批次均已完成")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="必须是优化工作区下的新版本目录；已存在且非空时会拒绝写入")
    parser.add_argument("--require-completed", action="store_true",
                        help="要求 --scope-file 中每张图均已完成；没有 scope 时会拒绝执行")
    parser.add_argument("--dry-run", action="store_true", help="验证 JSONL，不写文件")
    return parser.parse_args(argv)


def load_manifest(union: Path) -> set[str]:
    path = union / "manifests" / "samples.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["union_name"] for row in csv.DictReader(handle) if row.get("union_name")}


def validate_box(raw: object, line_number: int, image_name: str) -> tuple[int, list[float]]:
    if not isinstance(raw, dict):
        raise ValueError(f"第 {line_number} 行 {image_name}: box 不是对象")
    try:
        class_id = int(raw["class_id"])
        xywh = [float(value) for value in raw["xywh"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"第 {line_number} 行 {image_name}: box 缺少 class_id/xywh") from exc
    if len(xywh) != 4:
        raise ValueError(f"第 {line_number} 行 {image_name}: xywh 长度不是 4")
    if not 0 <= class_id < len(CLASSES):
        raise ValueError(f"第 {line_number} 行 {image_name}: 非法类别 {class_id}")
    if not all(math.isfinite(value) for value in xywh):
        raise ValueError(f"第 {line_number} 行 {image_name}: 坐标不是有限数")
    cx, cy, width, height = xywh
    if width <= 0 or height <= 0:
        raise ValueError(f"第 {line_number} 行 {image_name}: 框宽高必须大于 0")
    if (
        cx < 0 or cx > 1 or cy < 0 or cy > 1 or width > 1 or height > 1
        or cx - width / 2 < 0 or cy - height / 2 < 0
        or cx + width / 2 > 1 or cy + height / 2 > 1
    ):
        raise ValueError(f"第 {line_number} 行 {image_name}: 框越出 [0,1] 图像边界")
    return class_id, xywh


def load_scope(path: Path | None, valid_names: set[str]) -> set[str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"找不到审查范围清单: {path}")
    names = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    invalid = sorted(names - valid_names)
    if invalid:
        raise ValueError(f"审查范围含不属于 union 的图片，例如: {invalid[0]}")
    if not names:
        raise ValueError("审查范围清单为空")
    return names


def load_completed_records(
    review_jsonl: Path, valid_names: set[str], scope_names: set[str] | None, require_completed: bool,
) -> tuple[dict[str, list[tuple[int, list[float]]]], dict[str, object]]:
    if not review_jsonl.is_file():
        raise FileNotFoundError(f"找不到审查结果: {review_jsonl}")
    records: dict[str, list[tuple[int, list[float]]]] = {}
    total = completed = incomplete = empty_completed = 0
    seen_names: set[str] = set()
    with review_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是合法 JSON") from exc
            image_name = str(record.get("image_name", ""))
            if not image_name or image_name not in valid_names:
                raise ValueError(f"第 {line_number} 行包含不属于 union 的 image_name: {image_name!r}")
            if scope_names is not None and image_name not in scope_names:
                raise ValueError(f"第 {line_number} 行 {image_name} 不在指定审查范围内")
            if image_name in seen_names:
                raise ValueError(f"第 {line_number} 行 image_name 重复: {image_name}")
            seen_names.add(image_name)
            if not bool(record.get("completed")):
                incomplete += 1
                if require_completed:
                    raise ValueError(f"第 {line_number} 行 {image_name} 未完成，--require-completed 禁止导出")
                continue
            boxes = [validate_box(box, line_number, image_name) for box in record.get("boxes", [])]
            records[image_name] = boxes
            completed += 1
            if not boxes:
                empty_completed += 1
    if require_completed:
        if scope_names is None:
            raise ValueError("--require-completed 必须与 --scope-file 一起使用，才能验证整个审查批次")
        missing = sorted(scope_names - seen_names)
        if missing:
            raise ValueError(f"JSONL 缺少 {len(missing)} 张范围内图片，例如: {missing[0]}")
    return records, {
        "jsonl_records": total,
        "completed_images": completed,
        "incomplete_images": incomplete,
        "completed_negative_images": empty_completed,
        "unique_images": len(seen_names),
        "scope_images": len(scope_names) if scope_names is not None else None,
    }


def ensure_new_output(path: Path) -> None:
    resolved_root = ROOT.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"输出必须位于优化工作区内: {resolved_root}") from exc
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"输出目录已存在且非空，拒绝覆盖: {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        valid_names = load_manifest(args.union.resolve())
        scope_names = load_scope(args.scope_file, valid_names)
        records, stats = load_completed_records(args.review_jsonl, valid_names, scope_names, args.require_completed)
        if not records:
            raise ValueError("没有任何已完成图片，未生成标签版本")
        output = args.output_dir.resolve()
        ensure_new_output(output)
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        print(f"校验失败: {exc}", file=sys.stderr)
        return 2

    per_class = Counter(class_id for boxes in records.values() for class_id, _ in boxes)
    stats["boxes_total"] = sum(per_class.values())
    stats["per_class_boxes"] = {CLASSES[class_id]: per_class[class_id] for class_id in range(len(CLASSES))}
    stats["output_dir"] = str(output)
    stats["policy"] = {
        "source_labels": "union 原 labels/ 保持未改",
        "included_images": "仅 JSONL 中 completed=true 的图片",
        "negative_images": "completed=true 且 boxes=[] 时写同名空标签，代表人工确认无缺陷",
        "uncompleted_images": "不导出；不得进入冻结测试集或正式训练视图",
    }

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("[dry-run] 未写入标签")
        return 0

    output.mkdir(parents=True, exist_ok=True)
    labels_dir = output / "labels"
    labels_dir.mkdir()
    manifest_rows: list[dict[str, object]] = []
    for image_name, boxes in sorted(records.items()):
        label_path = labels_dir / f"{image_name}.txt"
        with label_path.open("w", encoding="utf-8", newline="\n") as handle:
            for class_id, xywh in boxes:
                handle.write(f"{class_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")
        manifest_rows.append({
            "union_name": image_name,
            "label_path": str(label_path),
            "box_count": len(boxes),
            "is_human_confirmed_negative": not boxes,
        })
    with (output / "completed_samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["union_name", "label_path", "box_count", "is_human_confirmed_negative"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "review_jsonl": str(args.review_jsonl.resolve()),
        "scope_file": str(args.scope_file.resolve()) if args.scope_file else None,
        **stats,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# 人工重标标签版本 v1\n\n"
        "本目录由 `apply_manual_reannotation.py` 从 completed=true 的人工重标 JSONL 导出。\n\n"
        "- `labels/`：标准 YOLO 五列标签；空文件仅表示人工确认的无缺陷图。\n"
        "- `completed_samples.csv`：导出范围。未完成图片未被写入，不能默认为负样本。\n"
        "- 原始 `E:/PythonProject/datasets/union_original_labels_v1/labels/` 未改动。\n"
        "- 该目录尚不是训练划分；后续须先做物理镍板/近重复分组、冻结测试集和独立训练视图。\n",
        encoding="utf-8",
    )
    print(f"产物: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
