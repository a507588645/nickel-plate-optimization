"""
将审查结果应用到本地数据集，生成 labels_pseudo_v1_v2/。

逻辑：
- 读取 review_result.jsonl 中的人工审查决定
- 对审查过的图片：
    * accepted 框默认保留，被标记 remove 的剔除
    * uncertain / rejected 框默认不保留，被标记 keep 的加入
- 对未审查的图片：从 labels_pseudo_v1/ 原样复制到 v2
- labels_pseudo_v1/ 本身不被修改

输出：
- {dataset}/labels_pseudo_v1_v2/{train,val}/
- {dataset}/review_v2_report.json
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_mapping(dataset_dir: Path):
    mapping_file = dataset_dir / "class_mapping_v1.yaml"
    if not mapping_file.exists():
        raise FileNotFoundError(f"未找到类别映射文件: {mapping_file}")
    data = load_yaml(mapping_file)
    names = data.get("names", {})
    # names 可能是 dict {id: name} 或 list
    if isinstance(names, list):
        return {name: idx for idx, name in enumerate(names)}
    return {name: int(idx) for idx, name in names.items()}


def load_manifest(manifest_path: Path):
    """读取 jsonl，按 image_stem 聚合所有框。"""
    groups = defaultdict(list)
    if not manifest_path.exists():
        return groups
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            groups[rec["image_stem"]].append(rec)
    return groups


def load_review_result(result_path: Path):
    """读取审查结果，按 (dataset, image_stem, status, box_index) 索引。"""
    decisions = defaultdict(lambda: defaultdict(dict))  # (dataset, stem) -> status -> {idx: decision}
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["dataset"], rec["image_stem"])
            status = rec["status"]
            idx = rec.get("box_index", 0)
            decisions[key][status][idx] = rec["decision"]
    return decisions


def read_yolo_label(label_path: Path):
    """读取 YOLO 格式标签，返回 [(class_id, x, y, w, h), ...]。"""
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            boxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return boxes


def write_yolo_label(label_path: Path, boxes: list):
    """写入 YOLO 格式标签。"""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def apply_review_to_dataset(
    dataset: str,
    dataset_root: Path,
    decisions: dict,
    output_suffix: str,
    copy_unreviewed: bool,
    dry_run: bool,
):
    """对一个数据集应用审查结果。"""
    dataset_dir = dataset_root / dataset
    class_to_id = load_class_mapping(dataset_dir)
    labels_v1_dir = dataset_dir / "labels_pseudo_v1"
    labels_v2_dir = dataset_dir / output_suffix

    if not labels_v1_dir.exists():
        print(f"[警告] {dataset} 没有 labels_pseudo_v1 目录，跳过")
        return None

    # 加载三个状态的 manifest
    manifests = {}
    for status in ["accepted", "uncertain", "rejected"]:
        manifest = dataset_dir / "manifests" / f"{status}.jsonl"
        manifests[status] = load_manifest(manifest)

    # 收集所有需要处理的 stem，并记录原始 split
    stem_to_split = {}
    for split_dir in labels_v1_dir.iterdir():
        if split_dir.is_dir():
            for txt_file in split_dir.glob("*.txt"):
                stem = txt_file.stem
                # 如果同一个 stem 出现在多个 split，优先保留现有映射，但发出警告
                if stem in stem_to_split and stem_to_split[stem] != split_dir.name:
                    print(f"  [警告] stem '{stem}' 同时存在于 {stem_to_split[stem]} 和 {split_dir.name}，保留 {stem_to_split[stem]}")
                    continue
                stem_to_split[stem] = split_dir.name

    all_stems = set(stem_to_split.keys())

    # 也加入审查结果中出现过但 v1 没有标签的 stem（理论上不应该，但以防万一）
    for (ds, stem) in decisions:
        if ds == dataset and stem not in all_stems:
            all_stems.add(stem)
            # 尝试从 manifest 的 split 字段推断
            for status in ["accepted", "uncertain", "rejected"]:
                boxes = manifests[status].get(stem, [])
                if boxes:
                    stem_to_split[stem] = boxes[0].get("split", "train")
                    break
            if stem not in stem_to_split:
                stem_to_split[stem] = "train"

    stats = {
        "dataset": dataset,
        "total_stems": len(all_stems),
        "reviewed_stems": 0,
        "unreviewed_stems": 0,
        "copied_stems": 0,
        "boxes_kept_from_accepted": 0,
        "boxes_removed_from_accepted": 0,
        "boxes_added_from_uncertain": 0,
        "boxes_added_from_rejected": 0,
        "boxes_ignored_from_uncertain": 0,
        "boxes_ignored_from_rejected": 0,
        "output_boxes": 0,
        "per_class_output": defaultdict(int),
    }

    for stem in sorted(all_stems):
        split = stem_to_split.get(stem, "train")

        key = (dataset, stem)
        stem_decisions = decisions.get(key, {})
        is_reviewed = bool(stem_decisions)

        if is_reviewed:
            stats["reviewed_stems"] += 1
            output_boxes = []

            # 处理 accepted 框：默认保留，remove 剔除
            accepted_groups = manifests["accepted"]
            for idx, box in enumerate(accepted_groups.get(stem, [])):
                decision = stem_decisions.get("accepted", {}).get(idx)
                if decision == "remove":
                    stats["boxes_removed_from_accepted"] += 1
                    continue
                # 默认 keep 或未被明确标记都保留
                cls_name = box["class_name"]
                cls_id = class_to_id.get(cls_name)
                if cls_id is None:
                    print(f"  [警告] 未知类别 {cls_name}，跳过")
                    continue
                output_boxes.append([
                    cls_id,
                    box["x_center"], box["y_center"], box["width"], box["height"]
                ])
                stats["boxes_kept_from_accepted"] += 1

            # 处理 uncertain 框：默认不保留，keep 加入
            for idx, box in enumerate(manifests["uncertain"].get(stem, [])):
                decision = stem_decisions.get("uncertain", {}).get(idx)
                if decision != "keep":
                    stats["boxes_ignored_from_uncertain"] += 1
                    continue
                cls_name = box["class_name"]
                cls_id = class_to_id.get(cls_name)
                if cls_id is None:
                    print(f"  [警告] 未知类别 {cls_name}，跳过")
                    continue
                output_boxes.append([
                    cls_id,
                    box["x_center"], box["y_center"], box["width"], box["height"]
                ])
                stats["boxes_added_from_uncertain"] += 1

            # 处理 rejected 框：默认不保留，keep 加入
            for idx, box in enumerate(manifests["rejected"].get(stem, [])):
                decision = stem_decisions.get("rejected", {}).get(idx)
                if decision != "keep":
                    stats["boxes_ignored_from_rejected"] += 1
                    continue
                cls_name = box["class_name"]
                cls_id = class_to_id.get(cls_name)
                if cls_id is None:
                    print(f"  [警告] 未知类别 {cls_name}，跳过")
                    continue
                output_boxes.append([
                    cls_id,
                    box["x_center"], box["y_center"], box["width"], box["height"]
                ])
                stats["boxes_added_from_rejected"] += 1

            if not dry_run:
                write_yolo_label(labels_v2_dir / split / f"{stem}.txt", output_boxes)

            stats["output_boxes"] += len(output_boxes)
            for box in output_boxes:
                stats["per_class_output"][box[0]] += 1

        else:
            stats["unreviewed_stems"] += 1
            if copy_unreviewed:
                stats["copied_stems"] += 1
                src = labels_v1_dir / split / f"{stem}.txt"
                dst = labels_v2_dir / split / f"{stem}.txt"
                if not dry_run and src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                # 统计 copied 的框数
                boxes = read_yolo_label(src)
                stats["output_boxes"] += len(boxes)
                for box in boxes:
                    stats["per_class_output"][box[0]] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="将审查结果应用到数据集，生成 v2 伪标签")
    parser.add_argument("--review-result", type=Path, required=True, help="审查结果 JSONL 文件路径")
    parser.add_argument("--dataset-root", type=Path, default=Path("E:/PythonProject/datasets"), help="数据集根目录")
    parser.add_argument("--datasets", nargs="+", default=["Dianjinie0211", "dianjienie"], help="数据集名称")
    parser.add_argument("--output-suffix", default="labels_pseudo_v1_v2", help="输出标签目录名后缀")
    parser.add_argument("--copy-unreviewed", action="store_true", default=True, help="将未审查图片的标签从 v1 复制到 v2（默认开启）")
    parser.add_argument("--no-copy-unreviewed", action="store_true", help="不复制未审查图片，v2 只包含审查过的图片")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入文件")
    args = parser.parse_args()

    copy_unreviewed = not args.no_copy_unreviewed

    decisions = load_review_result(args.review_result)
    print(f"已加载审查结果: {len(decisions)} 张图有审查决定")

    all_stats = []
    for dataset in args.datasets:
        print(f"\n处理数据集: {dataset}")
        stats = apply_review_to_dataset(
            dataset,
            args.dataset_root,
            decisions,
            args.output_suffix,
            copy_unreviewed,
            args.dry_run,
        )
        if stats:
            all_stats.append(stats)
            print(f"  总图片: {stats['total_stems']}")
            print(f"  已审查: {stats['reviewed_stems']}")
            print(f"  未审查并复制: {stats['copied_stems']}")
            print(f"  accepted 保留: {stats['boxes_kept_from_accepted']}, 删除: {stats['boxes_removed_from_accepted']}")
            print(f"  uncertain 加入: {stats['boxes_added_from_uncertain']}")
            print(f"  rejected 加入: {stats['boxes_added_from_rejected']}")
            print(f"  最终框数: {stats['output_boxes']}")

    # 保存报告
    report_path = args.dataset_root / "review_v2_report.json"
    if not args.dry_run:
        # 将 defaultdict 转成普通 dict 以便 JSON 序列化
        for stats in all_stats:
            stats["per_class_output"] = dict(stats["per_class_output"])
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "review_result": str(args.review_result),
                "output_suffix": args.output_suffix,
                "copy_unreviewed": copy_unreviewed,
                "datasets": all_stats,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存: {report_path}")

    if args.dry_run:
        print("\n[DRY RUN] 未实际写入文件")
    else:
        print(f"\n完成。v2 标签目录: {{dataset}}/{args.output_suffix}/")
        print("v1 标签目录保持原样未修改。")


if __name__ == "__main__":
    main()
