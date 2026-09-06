#!/usr/bin/env python3
"""B3 前置只读审计：datasets/Dianjinie0211（电积镍 5 类）。

只读取，不修改数据集任何文件。产出：
- audit/audit_report.json  机器可读全文
- audit/audit_summary.md   人工摘要

审计项（AGENTS.md §4.4/§4.5）：
1. 图片/标签/npy 清单与配对（缺标签、孤立标签、空标签）
2. 标签合法性（5 列、类别 0-4、坐标有限、宽高 (0,1]、框不越界）
3. 类别分布
4. train/val 主名冲突
5. 同板分组启发式（文件名前缀码）跨 train/val 泄漏检查
6. 与等级视图 archives/dianjinie_grade_classification_v1 主名重叠检查
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DATASET_ROOT = Path(r"E:/PythonProject/datasets/Dianjinie0211")
GRADE_CSV = Path(r"E:/PythonProject/archives/dianjinie_grade_classification_v1/manifests/samples.csv")
EXPECTED_CLASSES = ["waterstain", "waterblock", "colorstain", "nodule", "bubble"]
SPLITS = ("train", "val")

STEM_SUFFIX_RE = re.compile(r"(?P<code>.*?)(?:top|bottom|side)?_(?P<idx>\d+)$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读审计 Dianjinie0211")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "audit")
    return parser.parse_args(argv)


def list_dir(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file())


def group_key(stem: str) -> str:
    """同板分组启发式：去掉尾部 top/bottom 与切片序号后的前缀码。"""
    match = STEM_SUFFIX_RE.match(stem)
    if not match:
        return stem
    code = match.group("code") or stem
    return code


def read_label(path: Path) -> tuple[list[list[float]], list[str]]:
    boxes: list[list[float]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{path.name}:{line_number}:字段数={len(fields)}")
            continue
        try:
            class_id = int(fields[0])
            x, y, w, h = (float(v) for v in fields[1:])
        except ValueError:
            errors.append(f"{path.name}:{line_number}:非数值")
            continue
        if not 0 <= class_id < len(EXPECTED_CLASSES):
            errors.append(f"{path.name}:{line_number}:类别越界={class_id}")
            continue
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            errors.append(f"{path.name}:{line_number}:坐标非有限数")
            continue
        if not (0 < w <= 1 and 0 < h <= 1):
            errors.append(f"{path.name}:{line_number}:宽高非法 w={w} h={h}")
            continue
        epsilon = 1e-9
        if (
            x - w / 2 < -epsilon
            or x + w / 2 > 1 + epsilon
            or y - h / 2 < -epsilon
            or y + h / 2 > 1 + epsilon
        ):
            errors.append(f"{path.name}:{line_number}:框越界")
            continue
        boxes.append([class_id, x, y, w, h])
    return boxes, errors


def grade_view_stems(csv_path: Path) -> tuple[set[str], set[str]]:
    stems: set[str] = set()
    raw_names: set[str] = set()
    if not csv_path.is_file():
        return stems, raw_names
    import csv as csv_mod

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv_mod.DictReader(handle):
            if row.get("image_stem"):
                stems.add(row["image_stem"].strip())
            if row.get("image_name_raw"):
                raw_names.add(row["image_name_raw"].strip())
    return stems, raw_names


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.dataset_root
    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "expected_classes": EXPECTED_CLASSES,
        "splits": {},
        "top_level_dirs": {},
        "warnings": [],
    }

    for name in sorted(p.name for p in root.iterdir() if p.is_dir()):
        report["top_level_dirs"][name] = "dir"
    for name in sorted(p.name for p in list_dir(root)):
        report["top_level_dirs"][name] = "file"

    all_stems: dict[str, set[str]] = {}
    all_groups: dict[str, set[str]] = {}
    grade_stems, grade_raw = grade_view_stems(GRADE_CSV)
    report["grade_view"] = {"csv": str(GRADE_CSV), "stem_count": len(grade_stems)}

    for split in SPLITS:
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        files = list_dir(images_dir)
        by_ext = Counter(p.suffix.lower() for p in files)
        pngs = [p for p in files if p.suffix.lower() == ".png"]
        label_files = [p for p in list_dir(labels_dir) if p.suffix.lower() == ".txt"]

        stems = {p.stem for p in pngs}
        label_stems = {p.stem for p in label_files}
        missing_labels = sorted(stems - label_stems)
        orphan_labels = sorted(label_stems - stems)

        empty_labels: list[str] = []
        class_counter: Counter[int] = Counter()
        box_total = 0
        label_errors: list[str] = []
        for label in label_files:
            boxes, errors = read_label(label)
            label_errors.extend(errors)
            if not boxes:
                empty_labels.append(label.name)
            for box in boxes:
                class_counter[box[0]] += 1
            box_total += len(boxes)

        groups = {stem: group_key(stem) for stem in stems}
        all_stems[split] = stems
        all_groups[split] = set(groups.values())

        report["splits"][split] = {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "files_by_extension": dict(by_ext),
            "png_count": len(pngs),
            "label_txt_count": len(label_files),
            "cache_files": sorted(
                p.name for p in list_dir(labels_dir) if p.suffix in (".cache", ".npy")
            ),
            "missing_label_count": len(missing_labels),
            "missing_labels": missing_labels,
            "orphan_label_count": len(orphan_labels),
            "orphan_labels": orphan_labels,
            "empty_label_count": len(empty_labels),
            "empty_labels": empty_labels,
            "label_format_errors": label_errors,
            "box_total": box_total,
            "class_distribution": {
                EXPECTED_CLASSES[cid]: count
                for cid, count in sorted(class_counter.items())
            },
        }

    train_stems, val_stems = all_stems["train"], all_stems["val"]
    stem_collision = sorted(train_stems & val_stems)
    group_collision = sorted(all_groups["train"] & all_groups["val"])
    report["cross_split"] = {
        "stem_collision": stem_collision,
        "stem_collision_count": len(stem_collision),
        "group_collision": group_collision,
        "group_collision_count": len(group_collision),
    }

    dataset_stems = train_stems | val_stems
    overlap_stems = sorted(dataset_stems & grade_stems)
    report["cross_split"]["grade_view_stem_overlap"] = overlap_stems
    report["cross_split"]["grade_view_stem_overlap_count"] = len(overlap_stems)

    if report["splits"]["train"]["missing_label_count"]:
        report["warnings"].append(
            f"train 有 {report['splits']['train']['missing_label_count']} 张 png 无标签文件，需人工确认是负样本还是漏标"
        )
    if report["splits"]["val"]["missing_label_count"]:
        report["warnings"].append(
            f"val 有 {report['splits']['val']['missing_label_count']} 张 png 无标签文件，需人工确认"
        )
    if report["cross_split"]["group_collision_count"]:
        report["warnings"].append(
            f"train/val 按-{report['cross_split']['group_collision_count']} 个前缀码分组重叠（继承原始划分，与 slf.pt 同条件）"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Dianjinie0211 只读审计摘要（B3 前置）",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 数据集：`{root}`",
        "",
        "| 项 | train | val |",
        "| --- | ---: | ---: |",
    ]
    for key, label in (
        ("png_count", "png 图片"),
        ("label_txt_count", "标签 txt"),
        ("missing_label_count", "缺标签图片"),
        ("orphan_label_count", "孤立标签"),
        ("empty_label_count", "空标签"),
        ("box_total", "框总数"),
    ):
        lines.append(
            f"| {label} | {report['splits']['train'][key]} | {report['splits']['val'][key]} |"
        )
    lines.append("")
    lines.append(f"- 文件类型：train {report['splits']['train']['files_by_extension']}，val {report['splits']['val']['files_by_extension']}")
    lines.append(f"- 类别分布：train {report['splits']['train']['class_distribution']}，val {report['splits']['val']['class_distribution']}")
    lines.append(f"- train/val 主名冲突：{report['cross_split']['stem_collision_count']}")
    lines.append(f"- train/val 前缀码分组重叠：{report['cross_split']['group_collision_count']}")
    lines.append(f"- 与等级视图主名重叠：{report['cross_split']['grade_view_stem_overlap_count']}")
    lines.append(f"- 标签格式错误：train {len(report['splits']['train']['label_format_errors'])}，val {len(report['splits']['val']['label_format_errors'])}")
    if report["warnings"]:
        lines.append("")
        lines.append("## 警告")
        lines.extend(f"- {w}" for w in report["warnings"])
    summary_path = args.output_dir / "audit_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "report": str(report_path),
        "summary": str(summary_path),
        "warnings": report["warnings"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
