#!/usr/bin/env python3
"""
构建双轨交互审查视图（复用 generate_flat_yolo_review.py 的旧审查流程）。

- gt/：union 1455 张有标签图 + 原标签转六列（conf=1.00，state=accepted 显示）
- candidates/：union 1810 张无标签图 + pseudo_candidates 六列候选框
- classes_display.txt：18 类中英文显示名

产物：`E:\\PythonProject\\优化\\label_review_v1\\`
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UNION = Path(r"E:/PythonProject/datasets/union_original_labels_v1")
CAND = ROOT / "label_check_ryw_v1" / "pseudo_candidates" / "labels_with_conf"
OUT = ROOT / "label_review_v1"

CLASSES_DISPLAY = [
    ("wuzi", "污渍"), ("waterstain", "水渍"), ("yellowstain", "黄渍"),
    ("nodule", "边部结瘤"), ("airhole", "气孔"), ("bubble", "结粒"),
    ("bump", "疙瘩"), ("fawu", "发乌"), ("overfire", "烧板"),
    ("NiOH", "氢氧化镍"), ("greenstain", "绿色结晶"), ("suanwuran", "酸污染"),
    ("quejiao", "缺角"), ("watermark", "水印"), ("youpao", "油泡"),
    ("heibian", "黑边"), ("mijixiaokeng", "密集小坑"), ("fanxitong", "返析铜"),
]


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="双轨交互审查视图构建", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--union", type=Path, default=UNION)
    p.add_argument("--candidates", type=Path, default=CAND)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    import csv

    rows = list(csv.DictReader((args.union / "manifests" / "samples.csv").open(encoding="utf-8-sig")))
    img_index = {p_.stem: p_ for p_ in (args.union / "images").iterdir()}
    labeled = [r for r in rows if r.get("has_label") == "True"]
    unlabeled = [r for r in rows if r.get("has_label") != "True"]
    print(f"GT 轨 {len(labeled)} 张；候选轨 {len(unlabeled)} 张")
    if args.dry_run:
        return 0

    out = args.output_dir
    # GT 轨
    n = 0
    for r in labeled:
        ip = img_index.get(r["union_name"])
        if ip is None:
            continue
        link_or_copy(ip, out / "gt" / "images" / ip.name)
        src = args.union / "labels" / f"{r['union_name']}.txt"
        dst = out / "gt" / "labels_with_conf" / f"{r['union_name']}.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                lines.append(" ".join(parts[:5]) + " 1.0000")
        dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n += 1
    # 候选轨
    m = 0
    for r in unlabeled:
        ip = img_index.get(r["union_name"])
        if ip is None:
            continue
        link_or_copy(ip, out / "candidates" / "images" / ip.name)
        src = args.candidates / f"{r['union_name']}.txt"
        dst = out / "candidates" / "labels_with_conf" / f"{r['union_name']}.txt"
        if src.exists():
            link_or_copy(src, dst)
        else:  # 无框图：空标签文件，审查页才能显示该图
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("", encoding="utf-8")
        m += 1

    (out / "classes_display.txt").write_text(
        "\n".join(f"{en} {zh}" for en, zh in CLASSES_DISPLAY) + "\n", encoding="utf-8"
    )
    print(f"GT 轨写入 {n}；候选轨写入 {m}；classes_display.txt 就绪")
    print("接下来运行：")
    print(f"  python generate_flat_yolo_review.py {out / 'gt'} --absolute --classes-file {out / 'classes_display.txt'}")
    print(f"  python generate_flat_yolo_review.py {out / 'candidates'} --absolute --classes-file {out / 'classes_display.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
