#!/usr/bin/env python3
"""构建电积镍 0309 企业样例数据集及包含旧数据的软连接视图。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["3#", "9950", "9996", "bad"]
SOURCE_ID = "enterprise_0309"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建独立的电积镍 0309 企业数据集，并生成包含旧数据的 v2 软连接视图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"E:/Users/23916/Pictures/label/电积镍板0309"),
        help="企业样例原始目录，只读",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/dianji_enterprise_0309_v1"),
        help="独立新数据集输出目录",
    )
    parser.add_argument(
        "--old-unified-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/classify_grade_datasets"),
        help="已有统一软连接数据集，只读",
    )
    parser.add_argument(
        "--combined-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/classify_grade_datasets_v2"),
        help="旧数据加 0309 数据的新统一软连接视图",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.2, help="企业样例验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="固定分层划分种子")
    parser.add_argument("--dry-run", action="store_true", help="只审计和生成计划，不创建链接")
    return parser.parse_args()


def safe_symlink(src: Path, dst: Path, dry_run: bool) -> str:
    if dry_run:
        return "dry_run"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return "existing"
        raise FileExistsError(f"目标已存在且来源不同: {dst}")
    os.symlink(src.resolve(), dst)
    return "created"


def read_class_assignments(source_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    classify_root = source_root / "classify" / "train"
    for class_name in CLASS_NAMES:
        class_dir = classify_root / class_name
        if not class_dir.exists():
            continue
        for image in sorted(class_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            previous = result.get(image.stem)
            if previous is not None and previous != class_name:
                raise ValueError(f"同一图片出现冲突分类: {image.stem}: {previous}/{class_name}")
            result[image.stem] = class_name
    return result


def read_business_subtypes(source_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    excluded = {"classify", "defect"}
    for folder in sorted(source_root.iterdir()):
        if not folder.is_dir() or folder.name in excluded:
            continue
        for image in sorted(folder.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            previous = result.get(image.stem)
            if previous is not None and previous != folder.name:
                raise ValueError(f"同一图片出现冲突企业子类: {image.stem}: {previous}/{folder.name}")
            result[image.stem] = folder.name
    return result


def audit_yolo_label(path: Path, num_classes: int) -> tuple[int, list[str]]:
    if not path.exists():
        return 0, ["missing_label"]
    count = 0
    errors: list[str] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"line_{line_no}:field_count={len(fields)}")
            continue
        try:
            class_id = int(fields[0])
            coords = [float(v) for v in fields[1:]]
        except ValueError:
            errors.append(f"line_{line_no}:non_numeric")
            continue
        if not 0 <= class_id < num_classes:
            errors.append(f"line_{line_no}:class={class_id}")
        if any(v < 0 or v > 1 for v in coords):
            errors.append(f"line_{line_no}:coord_out_of_range")
        if coords[2] <= 0 or coords[3] <= 0:
            errors.append(f"line_{line_no}:non_positive_size")
        count += 1
    return count, errors


def stratified_split(assignments: dict[str, str], valid_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    for stem, class_name in assignments.items():
        grouped[class_name].append(stem)

    split_map: dict[str, str] = {}
    for class_name, stems in grouped.items():
        stems = sorted(stems)
        rng.shuffle(stems)
        valid_count = max(1, round(len(stems) * valid_ratio)) if len(stems) > 1 else 0
        valid_stems = set(stems[:valid_count])
        for stem in stems:
            split_map[stem] = "valid" if stem in valid_stems else "train"
    return split_map


def copy_old_view(old_root: Path, combined_root: Path, dry_run: bool) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for task in ("dianjie", "dianji"):
        for split in ("train", "valid"):
            split_dir = old_root / task / split
            if not split_dir.exists():
                continue
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                for source_link in sorted(p for p in class_dir.iterdir() if p.is_file()):
                    target = source_link.resolve()
                    dst = combined_root / task / split / class_dir.name / source_link.name
                    status = safe_symlink(target, dst, dry_run)
                    records.append({
                        "dataset": "existing",
                        "task": task,
                        "split": split,
                        "class_name": class_dir.name,
                        "link": str(dst),
                        "target": str(target),
                        "status": status,
                    })
    return records


def main() -> None:
    args = parse_args()
    if not 0 < args.valid_ratio < 1:
        raise ValueError("--valid-ratio 必须在 0 和 1 之间")

    source_root = args.source_root.resolve()
    image_root = source_root / "defect" / "image"
    label_root = source_root / "defect" / "label"
    classes_path = label_root / "classes.txt"
    if not image_root.exists() or not classes_path.exists():
        raise FileNotFoundError("缺少 defect/image 或 defect/label/classes.txt")

    defect_classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assignments = read_class_assignments(source_root)
    subtypes = read_business_subtypes(source_root)
    images = sorted(p for p in image_root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    image_stems = {p.stem for p in images}
    if image_stems != set(assignments):
        raise ValueError(
            f"分类目录与 defect/image 不一致: images_only={sorted(image_stems-set(assignments))}, "
            f"classify_only={sorted(set(assignments)-image_stems)}"
        )

    split_map = stratified_split(assignments, args.valid_ratio, args.seed)
    manifest: list[dict[str, Any]] = []
    audit_errors: list[dict[str, Any]] = []
    link_records: list[dict[str, str]] = []
    defect_distribution: Counter[int] = Counter()

    for image in images:
        stem = image.stem
        class_name = assignments[stem]
        split = split_map[stem]
        label = label_root / f"{stem}.txt"
        box_count, errors = audit_yolo_label(label, len(defect_classes))
        if label.exists():
            for raw in label.read_text(encoding="utf-8").splitlines():
                fields = raw.split()
                if len(fields) == 5:
                    try:
                        defect_distribution[int(fields[0])] += 1
                    except ValueError:
                        pass
        if errors:
            audit_errors.append({"image": str(image), "label": str(label), "errors": errors})

        prefixed_name = f"{SOURCE_ID}__{image.name}"
        class_dst = args.dataset_root / split / class_name / prefixed_name
        status = safe_symlink(image, class_dst, args.dry_run)
        link_records.append({"view": "standalone_classification", "link": str(class_dst), "target": str(image), "status": status})

        det_image_dst = args.dataset_root / "detection" / "images" / split / prefixed_name
        status = safe_symlink(image, det_image_dst, args.dry_run)
        link_records.append({"view": "standalone_detection_image", "link": str(det_image_dst), "target": str(image), "status": status})
        if label.exists():
            det_label_dst = args.dataset_root / "detection" / "labels" / split / f"{SOURCE_ID}__{label.name}"
            status = safe_symlink(label, det_label_dst, args.dry_run)
            link_records.append({"view": "standalone_detection_label", "link": str(det_label_dst), "target": str(label), "status": status})

        manifest.append({
            "source_id": SOURCE_ID,
            "image_path": str(image),
            "label_path": str(label) if label.exists() else "",
            "image_name": image.name,
            "class_name": class_name,
            "split": split,
            "business_subtype": subtypes.get(stem, ""),
            "has_detection_label": label.exists(),
            "detection_box_count": box_count,
        })

    combined_records = copy_old_view(args.old_unified_root, args.combined_root, args.dry_run)
    for row in manifest:
        image = Path(row["image_path"])
        dst = args.combined_root / "dianji" / row["split"] / row["class_name"] / f"{SOURCE_ID}__{image.name}"
        status = safe_symlink(image, dst, args.dry_run)
        combined_records.append({
            "dataset": SOURCE_ID,
            "task": "dianji",
            "split": row["split"],
            "class_name": row["class_name"],
            "link": str(dst),
            "target": str(image),
            "status": status,
        })

    if not args.dry_run:
        args.dataset_root.mkdir(parents=True, exist_ok=True)
        (args.dataset_root / "manifests").mkdir(parents=True, exist_ok=True)
        with (args.dataset_root / "manifests" / "samples.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        with (args.dataset_root / "manifests" / "link_records.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(link_records[0]))
            writer.writeheader()
            writer.writerows(link_records)
        with (args.dataset_root / "manifests" / "audit_errors.json").open("w", encoding="utf-8") as f:
            json.dump(audit_errors, f, ensure_ascii=False, indent=2)
        with (args.combined_root / "link_records.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(combined_records[0]))
            writer.writeheader()
            writer.writerows(combined_records)

        summary = {
            "version": "v1",
            "source_id": SOURCE_ID,
            "source_root": str(source_root),
            "seed": args.seed,
            "valid_ratio": args.valid_ratio,
            "images": len(images),
            "classification_distribution": dict(Counter(row["class_name"] for row in manifest)),
            "split_distribution": {
                f"{split}/{cls}": count
                for (split, cls), count in Counter((row["split"], row["class_name"]) for row in manifest).items()
            },
            "defect_classes": defect_classes,
            "defect_box_distribution": {
                defect_classes[class_id]: count for class_id, count in sorted(defect_distribution.items())
            },
            "missing_labels": sum(not row["has_detection_label"] for row in manifest),
            "audit_error_images": len(audit_errors),
            "combined_root": str(args.combined_root),
        }
        (args.dataset_root / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.dataset_root / "detection" / "classes.txt").parent.mkdir(parents=True, exist_ok=True)
        (args.dataset_root / "detection" / "classes.txt").write_text("\n".join(defect_classes) + "\n", encoding="utf-8")
        data_yaml = (
            f"path: {args.dataset_root.as_posix()}/detection\n"
            "train: images/train\n"
            "val: images/valid\n"
            f"names: {json.dumps(dict(enumerate(defect_classes)), ensure_ascii=False)}\n"
        )
        (args.dataset_root / "detection" / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print(f"企业图片: {len(images)}")
    print(f"分类分布: {dict(Counter(assignments.values()))}")
    print(f"划分分布: {dict(Counter((row['split'], row['class_name']) for row in manifest))}")
    print(f"缺失检测标签: {sum(not row['has_detection_label'] for row in manifest)}")
    print(f"存在标签审计问题的图片: {len(audit_errors)}")
    print(f"独立数据集: {args.dataset_root}")
    print(f"组合软连接视图: {args.combined_root}")


if __name__ == "__main__":
    main()
