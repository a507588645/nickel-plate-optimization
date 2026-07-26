#!/usr/bin/env python3
"""
基于电积镍检测特征定义 4 级严重程度标准，生成软连接 folder 数据集。

分级标准（可调整）：
    A: total_defect_count <= 5  且 total_defect_area_ratio <= 0.05
    B: total_defect_count <= 15 且 total_defect_area_ratio <= 0.20
    C: total_defect_count <= 25 且 total_defect_area_ratio <= 0.45
    D: 其他（最严重）

输出：
    E:\PythonProject\datasets\classify_dianji_severity/
        train/
            A/ B/ C/ D/
        valid/
            A/ B/ C/ D/
        labels.json
        link_records.csv

不改动原图。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def severity_grade(count: int, ratio: float) -> str:
    if count <= 5 and ratio <= 0.05:
        return "A"
    if count <= 15 and ratio <= 0.20:
        return "B"
    if count <= 25 and ratio <= 0.45:
        return "C"
    return "D"


def create_symlink(src: Path, dst: Path, records: list[dict[str, Any]]) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
        records.append({"link": str(dst), "target": str(src), "type": "symlink"})
    except OSError:
        records.append({"link": str(dst), "target": str(src), "type": "path_record"})


def main() -> None:
    parser = argparse.ArgumentParser(description="基于检测特征生成电积镍 4 级严重程度软连接数据集")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path(r"E:/PythonProject/优化/classify_from_detection_v1/dianji/features.csv"),
        help="电积镍检测特征 CSV",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/classify_dianji_severity"),
        help="输出根目录",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    df["severity_grade"] = df.apply(
        lambda row: severity_grade(row["total_defect_count"], row["total_defect_area_ratio"]), axis=1
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    labels: dict[str, str] = {}
    link_records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        src = Path(row["image_path"]).resolve()
        grade = row["severity_grade"]
        split = row["split"]
        dst = args.output_root / split / grade / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        create_symlink(src, dst, link_records)
        labels[str(src)] = grade

    with open(args.output_root / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    with open(args.output_root / "link_records.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["link", "target", "type"])
        writer.writeheader()
        writer.writerows(link_records)

    print("=== 电积镍 4 级严重程度数据集 ===")
    print(df["severity_grade"].value_counts().sort_index())
    print("\n每级统计:")
    print(
        df.groupby("severity_grade")[["total_defect_count", "total_defect_area_ratio"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print(f"\n输出目录: {args.output_root}")
    print(f"软连接/记录数: {len(link_records)}")


if __name__ == "__main__":
    main()
