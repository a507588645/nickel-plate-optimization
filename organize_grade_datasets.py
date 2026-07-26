#!/usr/bin/env python3
"""
把电解镍、电积镍的等级分类数据集统一整理到一个根目录下。

输出结构：
    E:\PythonProject\datasets\classify_grade_datasets/
    ├── dianjie/          # 电解镍多类软连接（当前 type 3/4）
    │   ├── train/
    │   │   ├── 3/
    │   │   └── 4/
    │   └── valid/
    │       ├── 3/
    │       └── 4/
    └── dianji/           # 电积镍 4 类软连接
        ├── train/
        │   ├── 3#/
        │   ├── 9950/
        │   ├── 9996/
        │   └── bad/
        └── valid/
            ├── 3#/
            ├── 9950/
            ├── 9996/
            └── bad/

不改动任何原数据，全部使用软连接。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def create_symlink(src: Path, dst: Path) -> bool:
    if dst.exists() or dst.is_symlink():
        return True
    try:
        os.symlink(src, dst)
        return True
    except OSError:
        return False


def organize_dianjie(output_root: Path) -> None:
    source_labels = Path(r"E:/PythonProject/datasets/classify_multiclass_dianjie/labels.json")
    if not source_labels.exists():
        print("警告: 未找到电解镍多类标签文件，跳过 dianjie")
        return

    with open(source_labels, "r", encoding="utf-8") as f:
        labels = json.load(f)

    ok = 0
    fail = 0
    for src_str, type_id in labels.items():
        src = Path(src_str).resolve()
        split = src.parent.parent.name  # train or valid
        dst = output_root / "dianjie" / split / str(type_id) / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if create_symlink(src, dst):
            ok += 1
        else:
            fail += 1

    print(f"电解镍: 软连接 {ok} 个，失败 {fail} 个")


def organize_dianji(output_root: Path) -> None:
    source_records = Path(r"E:/PythonProject/优化/classify_from_detection_v1/dianji/link_records.csv")
    if not source_records.exists():
        print("警告: 未找到电积镍连接记录，跳过 dianji")
        return

    ok = 0
    fail = 0
    with open(source_records, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dst_old = Path(row["link"])
            target = Path(row["target"]).resolve()
            # 从旧路径解析 split 和 class_name
            parts = dst_old.parts
            # 期望路径中最后三级为 train/3#/filename.png 或 valid/9950/...
            split = parts[-3] if parts[-3] in ("train", "valid") else None
            cls_name = parts[-2] if split else None
            if split is None or cls_name is None:
                # 兜底：从 target 路径解析
                split = target.parent.parent.name
                cls_name = target.parent.name
            dst = output_root / "dianji" / split / cls_name / target.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if create_symlink(target, dst):
                ok += 1
            else:
                fail += 1

    print(f"电积镍: 软连接 {ok} 个，失败 {fail} 个")


def main() -> None:
    parser = argparse.ArgumentParser(description="统一整理镍板等级分类数据集")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/classify_grade_datasets"),
        help="统一输出根目录",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"统一数据集根目录: {args.output_root}")

    organize_dianjie(args.output_root)
    organize_dianji(args.output_root)

    print("\n完成。结构如下:")
    for split_dir in sorted(args.output_root.rglob("*")):
        if split_dir.is_dir() and split_dir.parent.name in ("train", "valid"):
            n = len(list(split_dir.iterdir()))
            print(f"  {split_dir.relative_to(args.output_root)}: {n} 张")


if __name__ == "__main__":
    main()
