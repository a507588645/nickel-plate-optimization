#!/usr/bin/env python3
"""
L1 检测线：combined_all_pseudo_v2 伪标签分布审计（只读）。

按 accepted(>=0.70) / uncertain(0.40~0.70) / rejected(0.15~0.40) 三态统计：
- 框数：类别 × 状态
- 图数：类别 × 状态（图内该类该状态 ≥1 框即计 1）——分层划分设计的直接依据
- 每图框数分布与空图清单

产物写入 E:\\PythonProject\\优化\\detection_label_distribution_v1\\，不修改数据集。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = Path(r"E:/PythonProject/datasets/combined_all_pseudo_v2")
DEFAULT_OUT = ROOT / "detection_label_distribution_v1"

CLASS_NAMES_18 = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]
STATES = ["accepted", "uncertain", "rejected"]
ACC_T, UNC_T, REJ_T = 0.70, 0.40, 0.15


def state_of(conf: float) -> str | None:
    if conf >= ACC_T:
        return "accepted"
    if conf >= UNC_T:
        return "uncertain"
    if conf >= REJ_T:
        return "rejected"
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="L1 检测线伪标签分布审计（只读，未审查前不划分）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    labels_dir = args.dataset / "labels_with_conf"
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=== L1 伪标签分布审计 ===")
    print(f"dataset: {args.dataset}")
    if args.dry_run:
        print(f"[dry-run] labels_with_conf 存在: {labels_dir.exists()}")
        return 0
    if not labels_dir.exists():
        print(f"缺少标签目录: {labels_dir}", file=sys.stderr)
        return 2

    box_counts: dict[str, dict[str, int]] = {s: defaultdict(int) for s in STATES}
    img_counts: dict[str, dict[str, int]] = {s: defaultdict(int) for s in STATES}
    per_image_boxes: list[tuple[str, int]] = []
    empty_images: list[str] = []
    below_floor = 0
    n_files = 0

    for lf in sorted(labels_dir.glob("*.txt")):
        n_files += 1
        seen: set[tuple[str, str]] = set()
        n_box = 0
        for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                cid = int(float(parts[0]))
                conf = float(parts[5])
            except ValueError:
                continue
            if not (0 <= cid < len(CLASS_NAMES_18)):
                continue
            st = state_of(conf)
            if st is None:
                below_floor += 1
                continue
            name = CLASS_NAMES_18[cid]
            box_counts[st][name] += 1
            seen.add((name, st))
            n_box += 1
        for name, st in seen:
            img_counts[st][name] += 1
        per_image_boxes.append((lf.stem, n_box))
        if n_box == 0:
            empty_images.append(lf.stem)

    rows = []
    for name in CLASS_NAMES_18:
        row = {"class": name}
        for st in STATES:
            row[f"box_{st}"] = box_counts[st].get(name, 0)
            row[f"img_{st}"] = img_counts[st].get(name, 0)
        rows.append(row)

    csv_path = out / "class_state_distribution.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    box_vals = [b for _, b in per_image_boxes]
    box_vals.sort()
    n_img = len(per_image_boxes)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "label_files": n_files,
        "thresholds": {"accepted": ACC_T, "uncertain": UNC_T, "rejected": REJ_T},
        "box_totals": {st: sum(box_counts[st].values()) for st in STATES},
        "boxes_below_0.15_ignored": below_floor,
        "empty_images_after_floor": len(empty_images),
        "per_image_box_stats": {
            "min": box_vals[0] if box_vals else 0,
            "p50": box_vals[n_img // 2] if n_img else 0,
            "p90": box_vals[int(n_img * 0.9)] if n_img else 0,
            "max": box_vals[-1] if box_vals else 0,
            "mean": round(sum(box_vals) / n_img, 2) if n_img else 0,
        },
        "per_class": rows,
        "note": "审查前审计；划分须等 labels_reviewed 生成后按本分布做分组分层",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "empty_images.txt").write_text(
        "\n".join(empty_images), encoding="utf-8"
    )

    # 简表报告
    lines = [
        "# L1 检测线伪标签分布审计（审查前）",
        "",
        f"- 数据集：`{args.dataset}`（labels_with_conf {n_files} 个）",
        f"- 三态框数：accepted={summary['box_totals']['accepted']}、uncertain={summary['box_totals']['uncertain']}、rejected={summary['box_totals']['rejected']}（<0.15 忽略 {below_floor}）",
        f"- 空图（≥0.15 无框）：{len(empty_images)}（清单 empty_images.txt）",
        "",
        "| 类别 | 框 acc | 框 unc | 框 rej | 图 acc | 图 unc | 图 rej |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['class']} | {r['box_accepted']} | {r['box_uncertain']} | {r['box_rejected']} "
            f"| {r['img_accepted']} | {r['img_uncertain']} | {r['img_rejected']} |"
        )
    lines += [
        "",
        "- 用途：确定各类审查优先级（img_accepted 过低的类优先补审）；划分须在 `labels_reviewed_*` 生成后按 img 维度分组分层。",
        "- 约束（用户 2026-07-23）：审查后标签生成前禁止划分 train/val/test；现有 `splits/` 作废待重划。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"label files: {n_files}, boxes: {summary['box_totals']}, empty: {len(empty_images)}")
    print(f"产物: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
