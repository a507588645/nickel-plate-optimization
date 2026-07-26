#!/usr/bin/env python3
"""将 generate_flat_yolo_review.py 导出的 JSONL 审查结果应用为五列 YOLO 标签。

用法示例：
    python apply_flat_review_results.py \
        --review-result "E:/PythonProject/优化/Downloads/..._review_result.jsonl" \
        --dataset-root "E:/PythonProject/优化/pseudo_label_datasets/..." \
        --output-suffix labels_reviewed_v2

说明：
- 只读取审查结果 JSONL，不修改 labels_with_conf/ 主版本。
- decision 为 keep/accepted 的框保留（按原坐标写出）；remove/rejected 的框丢弃。
- uncertain 框默认保留（用 --drop-uncertain 剔除）；pending 框始终跳过。
- 所有图片都会生成对应的 .txt，即使全部被删也生成空标签文件。
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


KEEP_DECISIONS = {"keep", "accepted"}
DROP_DECISIONS = {"remove", "rejected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="应用扁平 YOLO 伪标签审查结果")
    parser.add_argument("--review-result", type=Path, required=True, help="review.html 导出的 JSONL 文件路径")
    parser.add_argument("--dataset-root", type=Path, required=True, help="扁平数据集根目录（包含 images/ 和 labels_with_conf/）")
    parser.add_argument("--output-suffix", default="labels_reviewed_v2", help="输出标签目录名后缀")
    parser.add_argument("--drop-uncertain", action="store_true", help="不保留 uncertain 框（默认保留）")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入文件")
    return parser.parse_args()


def write_yolo_label(path: Path, boxes: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root.resolve()
    review_result: Path = args.review_result.resolve()
    output_dir = dataset_root / args.output_suffix
    keep_uncertain = not args.drop_uncertain

    if not review_result.exists():
        raise FileNotFoundError(f"审查结果文件不存在: {review_result}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集根目录不存在: {dataset_root}")

    # 清空旧输出，避免残留
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        for old in output_dir.glob("*.txt"):
            old.unlink()

    stats = {
        "review_result": str(review_result),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "drop_uncertain": args.drop_uncertain,
        "dry_run": args.dry_run,
        "total_images": 0,
        "kept_boxes": 0,
        "dropped_boxes": 0,
        "uncertain_skipped": 0,
        "pending_skipped": 0,
        "empty_labels": 0,
        "per_class_kept": defaultdict(int),
    }

    with open(review_result, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_name = rec["image_name"]
            stem = Path(image_name).stem
            kept_boxes = []

            for bd in rec.get("box_decisions", []):
                decision = bd.get("decision", "pending")
                class_id = bd["class_id"]
                xywh = bd["xywh"]

                if decision in KEEP_DECISIONS:
                    kept_boxes.append([class_id, *xywh])
                    stats["kept_boxes"] += 1
                    stats["per_class_kept"][class_id] += 1
                elif decision in DROP_DECISIONS:
                    stats["dropped_boxes"] += 1
                elif decision == "uncertain":
                    if keep_uncertain:
                        kept_boxes.append([class_id, *xywh])
                        stats["kept_boxes"] += 1
                        stats["per_class_kept"][class_id] += 1
                    else:
                        stats["uncertain_skipped"] += 1
                else:
                    stats["pending_skipped"] += 1

            if not kept_boxes:
                stats["empty_labels"] += 1

            if not args.dry_run:
                write_yolo_label(output_dir / f"{stem}.txt", kept_boxes)

            stats["total_images"] += 1

    stats["per_class_kept"] = dict(sorted(stats["per_class_kept"].items()))

    report_path = dataset_root / f"{args.output_suffix}_report.json"
    if not args.dry_run:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if not args.dry_run:
        print(f"\n已输出到: {output_dir}")
        print(f"报告已保存: {report_path}")


if __name__ == "__main__":
    main()
