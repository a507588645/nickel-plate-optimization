#!/usr/bin/env python3
"""从等级特征 CSV 生成 LightGBM 分类数据集，并评估可行性。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


TYPE_NAMES = {
    0: "无板",
    1: "3号板",
    2: "烫洗板",
    3: "合格整版",
    4: "合格600",
    5: "合格不足600",
    6: "2号板",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 LightGBM 等级数据集")
    parser.add_argument("--grade-csv", type=Path, default=Path(r"E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v2_grade/samples.csv"))
    parser.add_argument("--combined-root", type=Path, default=Path(r"E:/PythonProject/datasets/combined_all_pseudo_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v2_grade"))
    return parser.parse_args()


def resolve_split(image_name: str, combined_root: Path) -> str:
    for split in ("train", "val", "test"):
        if (combined_root / "images" / split / image_name).exists():
            return split
    return "unknown"


def main() -> None:
    args = parse_args()
    grade_csv = args.grade_csv
    combined_root = args.combined_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with grade_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for row in reader:
            row["split"] = resolve_split(row["image_name"], combined_root)
            rows.append(row)

    # 选择特征列（排除元数据、文本字段和第二种标签）
    meta_cols = {
        "image_name",
        "image_path",
        "label_path",
        "image_width",
        "image_height",
        "findcent_fallback",
        "type_name_detection_only",
        "type_id_assume_resnet_good",
        "type_name_assume_resnet_good",
        "split",
    }
    feature_cols = [c for c in fieldnames if c not in meta_cols and c != "type_id_detection_only"]

    out_csv = output_dir / "lightgbm_grade_dataset.csv"
    out_fieldnames = ["image_name", "split", "type_id_detection_only"] + feature_cols
    with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # 统计
    overall = Counter(str(r["type_id_detection_only"]) for r in rows)
    split_counters: dict[str, Counter[str]] = {}
    for r in rows:
        split_counters.setdefault(r["split"], Counter())[str(r["type_id_detection_only"])] += 1

    def dist(counter: Counter[str]) -> dict[str, Any]:
        total = sum(counter.values())
        return {
            TYPE_NAMES.get(int(k), k): {"count": v, "rate": round(v / total, 4) if total else 0.0}
            for k, v in sorted(counter.items(), key=lambda x: int(x[0]))
        }

    summary = {
        "total": len(rows),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "overall_distribution": dist(overall),
        "split_distribution": {s: dist(c) for s, c in sorted(split_counters.items())},
    }
    summary_path = output_dir / "lightgbm_grade_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"LightGBM 数据集: {out_csv}")
    print(f"总样本: {len(rows)}，特征数: {len(feature_cols)}")
    print("\n整体等级分布（type_id_detection_only）:")
    for name, info in summary["overall_distribution"].items():
        print(f"  {name}: {info['count']} ({info['rate']*100:.2f}%)")
    print("\n各划分等级分布:")
    for split_name, d in summary["split_distribution"].items():
        print(f"  {split_name}: {sum(info['count'] for info in d.values())} 张")
        for name, info in d.items():
            print(f"    {name}: {info['count']} ({info['rate']*100:.2f}%)")

    # 可行性判断
    print("\n可行性评估:")
    max_rate = max(info["rate"] for info in summary["overall_distribution"].values())
    min_rate = min(info["rate"] for info in summary["overall_distribution"].values())
    if max_rate > 0.8:
        print(f"  - 类别严重不平衡：最大类占比 {max_rate*100:.1f}%，直接多分类会偏向多数类。")
        print("  - 建议：使用类别权重、过采样少数类、或改为二分类（合格整版 vs 非合格整版）。")
    else:
        print("  - 类别分布相对均衡，可直接多分类。")
    if min_rate < 0.01:
        rare = [n for n, i in summary["overall_distribution"].items() if i["rate"] < 0.01]
        print(f"  - 稀有类别 {rare} 样本极少，可能无法稳定学习。")
    print("  - 特征包含 A/B/C/气孔比例、各类缺陷计数与面积比，具有一定可解释性。")


if __name__ == "__main__":
    main()
