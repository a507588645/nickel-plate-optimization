#!/usr/bin/env python3
"""合并两个同特征模式的电积镍特征数据集，不修改原产物。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合并电积镍检测特征 CSV 和软连接树",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/classify_from_detection_v1"),
        help="已有特征数据集根目录",
    )
    parser.add_argument(
        "--extra-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/classify_from_detection_enterprise_0309_v1"),
        help="新增特征数据集根目录",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/classify_from_detection_combined_v2"),
        help="合并输出目录",
    )
    return parser.parse_args()


def link_tree(source_root: Path, output_root: Path) -> tuple[int, int]:
    created = 0
    existing = 0
    image_root = source_root / "dianji" / "images"
    for source_link in sorted(p for p in image_root.rglob("*") if p.is_file()):
        relative = source_link.relative_to(image_root)
        target = source_link.resolve()
        dst = output_root / "dianji" / "images" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and dst.resolve() == target:
                existing += 1
                continue
            raise FileExistsError(f"合并链接冲突: {dst}")
        os.symlink(target, dst)
        created += 1
    return created, existing


def main() -> None:
    args = parse_args()
    base_csv = args.base_root / "dianji" / "features.csv"
    extra_csv = args.extra_root / "dianji" / "features.csv"
    base = pd.read_csv(base_csv)
    extra = pd.read_csv(extra_csv)
    if list(base.columns) != list(extra.columns):
        missing_base = sorted(set(extra.columns) - set(base.columns))
        missing_extra = sorted(set(base.columns) - set(extra.columns))
        raise ValueError(f"特征列不一致: base缺少={missing_base}, extra缺少={missing_extra}")

    combined = pd.concat([base, extra], ignore_index=True)
    duplicate_paths = combined[combined.duplicated("image_path", keep=False)]["image_path"].tolist()
    if duplicate_paths:
        raise ValueError(f"存在重复图片路径，拒绝合并: {duplicate_paths[:10]}")

    output_task = args.output_root / "dianji"
    output_task.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_task / "features.csv", index=False, encoding="utf-8-sig")

    base_created, base_existing = link_tree(args.base_root, args.output_root)
    extra_created, extra_existing = link_tree(args.extra_root, args.output_root)

    metadata = {
        "version": "combined_v2",
        "sources": [str(args.base_root.resolve()), str(args.extra_root.resolve())],
        "base_samples": len(base),
        "extra_samples": len(extra),
        "total_samples": len(combined),
        "columns": list(combined.columns),
        "class_distribution": {
            f"{split}/{class_name}": int(count)
            for (split, class_name), count in combined.groupby(["split", "class_name"]).size().items()
        },
        "links": {
            "base_created": base_created,
            "base_existing": base_existing,
            "extra_created": extra_created,
            "extra_existing": extra_existing,
        },
        "tasks": {
            "dianji": {
                "label_map": {"3#": 0, "9950": 1, "9996": 2, "bad": 3},
                "feature_csv": str((output_task / "features.csv").resolve()),
            }
        },
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"基础样本: {len(base)}")
    print(f"新增样本: {len(extra)}")
    print(f"合并样本: {len(combined)}")
    print(f"输出: {args.output_root}")


if __name__ == "__main__":
    main()
