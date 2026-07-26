#!/usr/bin/env python3
"""对 combined_all_pseudo_v1 去重并重建数据集。

流程：
1. 读取唯一性审计结果，决定每个完全重复组保留哪张图片。
2. 按物理板 group 做整体划分（同一 group 不跨 train/val/test）。
3. 生成新版本数据集目录（硬链接图片 + 标签）。
4. 从 labels_with_conf 生成 5 列训练标签 labels/。
5. 输出 data.yaml 和划分统计。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="去重并重建 combined_all_pseudo_v1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/combined_all_pseudo_v1"),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path(r"E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/uniqueness_audit"),
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/combined_all_pseudo_v1_dedup_v2"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-instead-of-link", action="store_true", help="复制而非硬链接图片")
    return parser.parse_args()


def read_exact_duplicates(audit_dir: Path) -> dict[str, list[str]]:
    """读取完全重复组，返回 md5 -> 路径列表。"""
    result: dict[str, list[str]] = {}
    csv_path = audit_dir / "exact_duplicates.csv"
    if not csv_path.exists():
        return result
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["md5"]] = row["paths"].split(";")
    return result


def read_all_images(audit_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    csv_path = audit_dir / "all_images_md5.csv"
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_physical_groups(audit_dir: Path) -> dict[str, list[str]]:
    """读取物理分组，返回 group_id -> 路径列表。"""
    result: dict[str, list[str]] = {}
    csv_path = audit_dir / "physical_groups.csv"
    if not csv_path.exists():
        return result
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["group"]] = row["paths"].split(";")
    return result


def choose_representative(paths: list[str]) -> str:
    """从重复组中选择保留的代表。"""
    path_objs = [Path(p) for p in paths]

    def score(p: Path) -> tuple[int, int, str]:
        split = p.parts[-2] if p.parts[-2] in SPLITS else "unknown"
        split_rank = {"train": 0, "val": 1, "test": 2, "unknown": 3}[split]
        name = p.name
        copy_penalty = 0
        if "副本" in name or "Copy" in name or "(1)" in name or "(2)" in name:
            copy_penalty = 1
        return (split_rank, copy_penalty, str(p))

    return str(min(path_objs, key=score))


def build_keep_set(all_images: list[dict[str, Any]], duplicates: dict[str, list[str]]) -> set[str]:
    """确定需要保留的所有图片路径。"""
    all_paths = {row["path"] for row in all_images}
    removed: set[str] = set()
    for group in duplicates.values():
        rep = choose_representative(group)
        for p in group:
            if p != rep:
                removed.add(p)
    return all_paths - removed


def group_images_by_physical(keep_paths: set[str], physical_groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """将保留的图片按物理板 group 分组；未匹配的图片各自成组。"""
    assigned: set[str] = set()
    result: dict[str, list[str]] = {}
    for group_id, paths in physical_groups.items():
        kept = [p for p in paths if p in keep_paths]
        if kept:
            result[group_id] = kept
            assigned.update(kept)
    # 未分组的图片各自成组
    for p in sorted(keep_paths):
        if p not in assigned:
            result[f"single_{Path(p).stem}"] = [p]
    return result


def assign_splits(groups: dict[str, list[str]], seed: int) -> dict[str, list[str]]:
    """按 group 整体分配 split，保持 70/15/15 比例。"""
    random.seed(seed)
    group_items = list(groups.items())
    random.shuffle(group_items)

    total = sum(len(paths) for _, paths in group_items)
    targets = {split: int(round(TARGET_RATIOS[split] * total)) for split in SPLITS}
    # 调整四舍五入误差
    diff = total - sum(targets.values())
    if diff > 0:
        targets["train"] += diff
    elif diff < 0:
        targets["train"] += diff

    split_paths: dict[str, list[str]] = {split: [] for split in SPLITS}
    counts = {split: 0 for split in SPLITS}

    for group_id, paths in group_items:
        # 选择距离目标比例最远的 split
        best_split = min(SPLITS, key=lambda s: (counts[s] + len(paths)) / total - TARGET_RATIOS[s] if total > 0 else 0)
        split_paths[best_split].extend(paths)
        counts[best_split] += len(paths)

    return split_paths


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(str(src), str(dst))
    else:
        try:
            hardlink(src, dst)
        except OSError:
            shutil.copy2(str(src), str(dst))


def hardlink(src: Path, dst: Path) -> None:
    import os
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.link(str(src), str(dst))


def copy_label(src_label: Path, dst_label: Path, drop_confidence: bool) -> None:
    """复制标签文件；可选删除第六列置信度。"""
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    if not src_label.exists():
        dst_label.write_text("", encoding="utf-8")
        return
    if not drop_confidence:
        shutil.copy2(str(src_label), str(dst_label))
        return
    lines_out = []
    for line in src_label.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) == 6:
            lines_out.append(" ".join(fields[:5]))
        elif len(fields) == 5:
            lines_out.append(line.strip())
    dst_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def count_classes(label_dir: Path, class_names: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for split in SPLITS:
        d = label_dir / split
        if not d.exists():
            continue
        for label_file in d.iterdir():
            if label_file.suffix != ".txt":
                continue
            for line in label_file.read_text(encoding="utf-8").splitlines():
                fields = line.strip().split()
                if not fields:
                    continue
                cid = int(fields[0])
                if 0 <= cid < len(class_names):
                    counter[class_names[cid]] += 1
    return counter


def main() -> None:
    args = parse_args()
    src_root = args.src_root.resolve()
    audit_dir = args.audit_dir.resolve()
    dst_root = args.dst_root.resolve()

    print("读取审计结果...")
    all_images = read_all_images(audit_dir)
    duplicates = read_exact_duplicates(audit_dir)
    physical_groups = read_physical_groups(audit_dir)

    print(f"总图片: {len(all_images)}")
    keep_paths = build_keep_set(all_images, duplicates)
    removed_count = len(all_images) - len(keep_paths)
    print(f"去重后保留: {len(keep_paths)}, 删除重复: {removed_count}")

    groups = group_images_by_physical(keep_paths, physical_groups)
    print(f"物理分组/单图组数: {len(groups)}")

    split_paths = assign_splits(groups, args.seed)
    for split in SPLITS:
        print(f"  {split}: {len(split_paths[split])} 张")

    # 读取类别名称
    classes_path = src_root / "labels_with_conf" / "classes.txt"
    class_names = (
        [line.strip() for line in classes_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if classes_path.exists()
        else []
    )

    # 创建输出目录
    if dst_root.exists():
        print(f"目标目录已存在，将覆盖: {dst_root}")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    # 复制 classes.txt
    if classes_path.exists():
        dst_conf_dir = dst_root / "labels_with_conf"
        dst_conf_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(classes_path), str(dst_conf_dir / "classes.txt"))

    # 复制图片和标签
    for split in SPLITS:
        for src_img_path in split_paths[split]:
            src_img = Path(src_img_path)
            stem = src_img.stem

            # 图片
            dst_img = dst_root / "images" / split / src_img.name
            link_or_copy(src_img, dst_img, args.copy_instead_of_link)

            # 同时链接到 images_all
            dst_img_all = dst_root / "images_all" / src_img.name
            link_or_copy(src_img, dst_img_all, args.copy_instead_of_link)

            # labels_with_conf (flat, 6 columns)
            src_conf_label = src_root / "labels_with_conf" / f"{stem}.txt"
            dst_conf_label = dst_root / "labels_with_conf" / f"{stem}.txt"
            copy_label(src_conf_label, dst_conf_label, drop_confidence=False)

            # labels (split, 5 columns)
            src_label = src_root / "labels" / split / f"{stem}.txt"
            # 优先从 labels_with_conf 生成 5 列，保证一致
            dst_label = dst_root / "labels" / split / f"{stem}.txt"
            copy_label(src_conf_label, dst_label, drop_confidence=True)

    # 生成 data.yaml
    data_yaml = {
        "path": str(dst_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    with (dst_root / "data.yaml").open("w", encoding="utf-8") as f:
        for k, v in data_yaml.items():
            if isinstance(v, list):
                f.write(f"{k}: {v}\n")
            else:
                f.write(f"{k}: {v}\n")

    # 统计框数
    label_counter = count_classes(dst_root / "labels", class_names)
    distribution = {
        "summary": {
            "total_images": len(keep_paths),
            "removed_duplicates": removed_count,
            "train": len(split_paths["train"]),
            "val": len(split_paths["val"]),
            "test": len(split_paths["test"]),
            "total_boxes": sum(label_counter.values()),
        },
        "class_distribution": dict(sorted(label_counter.items(), key=lambda x: -x[1])),
    }
    splits_dir = dst_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "distribution.json").write_text(json.dumps(distribution, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"{dst_root.name} split summary",
        "=" * 40,
        "",
        f"train: {len(split_paths['train'])} images",
        f"val:   {len(split_paths['val'])} images",
        f"test:  {len(split_paths['test'])} images",
        f"",
        f"kept boxes: {sum(label_counter.values())}",
        "",
        "Class distribution:",
    ]
    for name, count in distribution["class_distribution"].items():
        lines.append(f"  {name}: {count}")
    (splits_dir / "distribution.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n去重后数据集已生成: {dst_root}")
    print(f"  train/val/test: {len(split_paths['train'])}/{len(split_paths['val'])}/{len(split_paths['test'])}")
    print(f"  总框数: {sum(label_counter.values())}")


if __name__ == "__main__":
    main()
