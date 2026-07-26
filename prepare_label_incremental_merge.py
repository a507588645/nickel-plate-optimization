#!/usr/bin/env python3
"""Move, deduplicate, and incrementally merge the Pictures/label dataset.

The workflow is intentionally split into two stages:

1. ``prepare`` moves the raw directory intact, creates an exact-deduplicated
   hard-link view, and builds a new combined dataset containing old labels plus
   a separate directory of images that still need pseudo-labeling.
2. ``finalize`` validates the newly generated six-column labels, merges them,
   and creates deterministic image splits without generating five-column
   training labels from unreviewed pseudo-labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="迁移 label 原始目录、精确去重并增量汇总伪标签数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("stage", choices=("prepare", "finalize"))
    parser.add_argument("--source-root", type=Path, default=Path(r"E:/Users/23916/Pictures/label"))
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/enterprise_label_raw_20260716"),
    )
    parser.add_argument(
        "--dedup-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/enterprise_label_dedup_v1"),
    )
    parser.add_argument(
        "--old-combined",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/combined_all_pseudo_v2"),
    )
    parser.add_argument(
        "--new-combined",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/combined_all_pseudo_v3"),
    )
    parser.add_argument(
        "--records-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/label_incremental_merge/20260716_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf-keep", type=float, default=0.70)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256(path: Path, cache: dict[tuple[int, int, int], str]) -> str:
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size)
    cached = cache.get(identity)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    cache[identity] = value
    return value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hardlink(source: Path, target: Path, dry_run: bool) -> None:
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖: {target}")
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def safe_flat_name(relative: Path) -> str:
    parts = [re.sub(r'[<>:"/\\|?*]+', "_", part) for part in relative.parts]
    return "__".join(parts)


def paired_label(image: Path) -> Path | None:
    if image.parent.name.lower() in {"image", "images"}:
        for label_dir_name in ("label", "labels"):
            candidate = image.parent.parent / label_dir_name / f"{image.stem}.txt"
            if candidate.exists():
                return candidate
    return None


def representative_score(path: Path) -> tuple[int, int, int, str]:
    text = str(path).lower()
    return (
        0 if paired_label(path) else 1,
        1 if "classify" in text else 0,
        1 if "0309-defect" in text else 0,
        str(path),
    )


def move_raw(source_root: Path, raw_root: Path, dry_run: bool) -> None:
    if source_root.exists() and raw_root.exists():
        raise FileExistsError(f"源目录和目标原始目录同时存在，拒绝猜测: {source_root}, {raw_root}")
    if raw_root.exists():
        return
    if not source_root.exists():
        raise FileNotFoundError(f"源目录不存在: {source_root}")
    if dry_run:
        return
    raw_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_root), str(raw_root))


def prepare(args: argparse.Namespace) -> None:
    if args.dedup_root.exists() or args.new_combined.exists() or args.records_root.exists():
        raise FileExistsError("输出目录已存在，拒绝覆盖；请改版本号或清理未完成输出")

    move_raw(args.source_root, args.raw_root, args.dry_run)
    raw_root = args.raw_root if args.raw_root.exists() else args.source_root
    raw_images = images(raw_root)
    if not raw_images:
        raise ValueError(f"未发现图片: {raw_root}")

    cache: dict[tuple[int, int, int], str] = {}
    digest_groups: dict[str, list[Path]] = defaultdict(list)
    for index, path in enumerate(raw_images, 1):
        digest_groups[sha256(path, cache)].append(path)
        if index % 50 == 0:
            print(f"[hash raw] {index}/{len(raw_images)}")

    representatives = {
        digest: sorted(paths, key=representative_score)[0]
        for digest, paths in digest_groups.items()
    }
    duplicate_rows: list[dict[str, object]] = []
    for digest, paths in sorted(digest_groups.items()):
        representative = representatives[digest]
        for path in paths:
            duplicate_rows.append({
                "sha256": digest,
                "representative": str(representative),
                "source_path": str(path),
                "is_representative": path == representative,
                "group_size": len(paths),
            })

    dedup_manifest: list[dict[str, object]] = []
    dedup_name_by_digest: dict[str, str] = {}
    used_names: set[str] = set()
    for digest, source in sorted(representatives.items(), key=lambda item: str(item[1])):
        flat_name = safe_flat_name(source.relative_to(raw_root))
        if flat_name in used_names:
            flat_name = f"{source.stem}__{digest[:12]}{source.suffix.lower()}"
        used_names.add(flat_name)
        dedup_name_by_digest[digest] = flat_name
        target = args.dedup_root / "images" / flat_name
        label = paired_label(source)
        hardlink(source, target, args.dry_run)
        if label is not None:
            hardlink(label, args.dedup_root / "labels_original" / f"{Path(flat_name).stem}.txt", args.dry_run)
        dedup_manifest.append({
            "output_name": flat_name,
            "sha256": digest,
            "source_path": str(source),
            "source_relative": str(source.relative_to(raw_root)),
            "original_label": str(label) if label else "",
            "duplicate_group_size": len(digest_groups[digest]),
        })

    old_images_dir = args.old_combined / "images_all"
    old_labels_dir = args.old_combined / "labels_with_conf"
    old_images = images(old_images_dir)
    if not old_images or not old_labels_dir.exists():
        raise FileNotFoundError(f"旧汇总集不完整: {args.old_combined}")

    old_digest_by_path: dict[Path, str] = {}
    old_digests: set[str] = set()
    for index, path in enumerate(old_images, 1):
        digest = sha256(path, cache)
        old_digest_by_path[path] = digest
        old_digests.add(digest)
        if index % 250 == 0:
            print(f"[hash old combined] {index}/{len(old_images)}")

    new_digests = sorted(set(representatives) - old_digests)
    overlap_digests = sorted(set(representatives) & old_digests)

    for old_image in old_images:
        target = args.new_combined / "images_all" / old_image.name
        hardlink(old_image, target, args.dry_run)
        old_label = old_labels_dir / f"{old_image.stem}.txt"
        if not old_label.exists():
            raise FileNotFoundError(f"旧图片缺少六列标签: {old_label}")
        hardlink(old_label, args.new_combined / "labels_with_conf" / old_label.name, args.dry_run)

    classes_path = old_labels_dir / "classes.txt"
    if classes_path.exists():
        hardlink(classes_path, args.new_combined / "labels_with_conf" / "classes.txt", args.dry_run)

    incremental_rows: list[dict[str, object]] = []
    for digest in new_digests:
        source = representatives[digest]
        dedup_name = dedup_name_by_digest[digest]
        combined_name = f"enterprise_label_dedup_v1_{dedup_name}"
        dedup_image = args.dedup_root / "images" / dedup_name
        hardlink(dedup_image, args.new_combined / "images_all" / combined_name, args.dry_run)
        hardlink(dedup_image, args.records_root / "incremental_images" / combined_name, args.dry_run)
        incremental_rows.append({
            "combined_name": combined_name,
            "sha256": digest,
            "dedup_name": dedup_name,
            "source_path": str(source),
        })

    summary = {
        "raw_paths": len(raw_images),
        "raw_unique_images": len(representatives),
        "raw_duplicate_paths_removed_from_view": len(raw_images) - len(representatives),
        "old_combined_images": len(old_images),
        "dedup_images_already_in_old_combined": len(overlap_digests),
        "new_unique_images": len(new_digests),
        "new_combined_images": len(old_images) + len(new_digests),
        "pseudo_label_pending": len(new_digests),
    }
    if not args.dry_run:
        write_csv(args.records_root / "raw_exact_duplicates.csv", duplicate_rows)
        write_csv(args.records_root / "dedup_manifest.csv", dedup_manifest)
        write_csv(args.records_root / "incremental_manifest.csv", incremental_rows)
        (args.records_root / "prepare_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.new_combined / "BUILD_INCOMPLETE.txt").write_text(
            "新增图片尚未生成伪标签。完成 auto_label 后运行 finalize。\n", encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_six_column_label(path: Path) -> tuple[int, int]:
    boxes = 0
    invalid = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except (ValueError, IndexError):
            invalid += 1
            continue
        if len(fields) != 6 or class_id < 0 or not all(0.0 <= value <= 1.0 for value in values):
            invalid += 1
            continue
        boxes += 1
    return boxes, invalid


def finalize(args: argparse.Namespace) -> None:
    incremental_manifest_path = args.records_root / "incremental_manifest.csv"
    generated_labels = args.records_root / "incremental_labels_with_conf"
    if not incremental_manifest_path.exists():
        raise FileNotFoundError("缺少增量清单")

    with incremental_manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if rows and not generated_labels.exists():
        raise FileNotFoundError("存在新增图片，但缺少 incremental_labels_with_conf")
    validation_rows: list[dict[str, object]] = []
    total_boxes = 0
    for row in rows:
        image_name = row["combined_name"]
        label = generated_labels / f"{Path(image_name).stem}.txt"
        if not label.exists():
            if args.dry_run:
                continue
            label.write_text("", encoding="utf-8")
        boxes, invalid = validate_six_column_label(label)
        if invalid:
            raise ValueError(f"伪标签格式非法: {label}, invalid_lines={invalid}")
        total_boxes += boxes
        hardlink(label, args.new_combined / "labels_with_conf" / label.name, args.dry_run)
        validation_rows.append({"image": image_name, "boxes": boxes, "invalid_lines": invalid})

    old_split_by_name: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for path in images(args.old_combined / "images" / split):
            old_split_by_name[path.name] = split

    new_names = sorted(row["combined_name"] for row in rows)
    rng = random.Random(args.seed)
    rng.shuffle(new_names)
    split_counts = defaultdict(int)
    for split in old_split_by_name.values():
        split_counts[split] += 1
    target_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    total_final = len(old_split_by_name) + len(new_names)
    target_counts = {split: round(total_final * ratio) for split, ratio in target_ratios.items()}
    target_counts["train"] += total_final - sum(target_counts.values())
    new_split_by_name: dict[str, str] = {}
    for name in new_names:
        split = max(target_counts, key=lambda key: target_counts[key] - split_counts[key])
        new_split_by_name[name] = split
        split_counts[split] += 1

    split_rows: list[dict[str, str]] = []
    all_split_by_name = {**old_split_by_name, **new_split_by_name}
    for name, split in sorted(all_split_by_name.items()):
        source = args.new_combined / "images_all" / name
        hardlink(source, args.new_combined / "images" / split / name, args.dry_run)
        split_rows.append({"split": split, "image": name})

    summary = {
        "total_images": len(all_split_by_name),
        "old_images": len(old_split_by_name),
        "new_images": len(new_names),
        "new_pseudo_label_boxes": total_boxes,
        "splits": dict(sorted(split_counts.items())),
        "training_label_status": "not_generated_unreviewed_six_column_labels_only",
        "confidence_threshold_for_grade": args.conf_keep,
    }
    if not args.dry_run:
        write_csv(args.records_root / "incremental_label_validation.csv", validation_rows)
        write_csv(args.new_combined / "splits" / "split_manifest.csv", split_rows)
        (args.new_combined / "splits" / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        incomplete = args.new_combined / "BUILD_INCOMPLETE.txt"
        if incomplete.exists():
            incomplete.unlink()
        (args.new_combined / "README.md").write_text(
            "# combined_all_pseudo_v2\n\n"
            "该版本由 combined_all_pseudo_v1 与 enterprise_label_dedup_v1 精确去重后增量构建。\n"
            "`labels_with_conf/` 为六列可追溯伪标签；尚未生成人工审查后的五列训练标签。\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
