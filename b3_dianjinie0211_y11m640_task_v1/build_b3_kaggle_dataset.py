#!/usr/bin/env python3
"""Build the B3 Kaggle upload view for Dianjinie0211 (route B3).

Outputs (all under this task directory, never inside the source dataset):

- ``splits/train_local.txt`` / ``val_local.txt``
  train/val lists relative to the ORIGINAL dataset layout
  (``images/train/x.png``), used for local ``--dry-run`` validation only.
- ``splits/split_summary.json``
  inclusion rules, exclusions, counts and class distributions.
- with ``--stage``: ``kaggle_dataset_upload_flat_v1/``
  flat upload view (PNG hard links, label/metadata copies, trainer script,
  ``flat_manifest.json``, ``dataset-metadata.json``), following the proven
  RYW18 flat pattern.

Inclusion rule (fixed, deterministic):
- val: every PNG in ``images/val`` that has a label file.
- train: every PNG in ``images/train`` that has a label file AND is not a
  Windows ``- 副本`` copy (duplicates whose original exists).
- Unlabeled PNGs are excluded until a human confirms they are true negatives
  (AGENTS.md §6); the exclusion lists are recorded in ``split_summary.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path(r"E:/PythonProject/datasets/Dianjinie0211")
DEFAULT_OUTPUT_ROOT = ROOT / "kaggle_dataset_upload_flat_v1"
DEFAULT_SPLITS_DIR = ROOT / "splits"
TRAINER_SOURCE = ROOT / "train_dianjinie0211_b3.py"
TRAINER_UPLOAD_NAME = "trainer__train_dianjinie0211_b3.py"
EXPECTED_CLASSES = ["waterstain", "waterblock", "colorstain", "nodule", "bubble"]
DEFAULT_OWNER_SLUG = "lly3323w"
DEFAULT_DATASET_SLUG = "dianjinie0211-b3-v1"
DEFAULT_DATASET_TITLE = "dianjinie0211 B3 v1"
COPY_SUFFIX = " - 副本"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 B3 电积镍 Kaggle 上传视图与划分清单")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--owner-slug", default=DEFAULT_OWNER_SLUG)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--dataset-title", default=DEFAULT_DATASET_TITLE)
    parser.add_argument("--stage", action="store_true", help="除清单外，同时构建扁平上传视图（本地硬链接，不上传）")
    parser.add_argument("--dry-run", action="store_true", help="只审计并打印计划，不写任何文件")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_classes(dataset_root: Path) -> list[str]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"缺少 data.yaml：{data_yaml}")
    content = data_yaml.read_text(encoding="utf-8")
    if f"nc: {len(EXPECTED_CLASSES)}" not in content:
        raise ValueError(f"data.yaml 类别数不是 {len(EXPECTED_CLASSES)}")
    names = []
    in_names = False
    for line in content.splitlines():
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            stripped = line.strip()
            if not stripped:
                continue
            if ":" in stripped:
                _, _, name = stripped.partition(":")
                names.append(name.strip())
            else:
                break
    if names != EXPECTED_CLASSES:
        raise ValueError(f"data.yaml 类别顺序与预期不符：{names}")
    return names


def validate_label_bytes(text: str, label_path: Path) -> tuple[int, dict[str, int]]:
    class_counter: dict[str, int] = {}
    box_count = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"标签字段数错误：{label_path}:{line_number}")
        try:
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"标签含非数值字段：{label_path}:{line_number}") from error
        if not 0 <= class_id < len(EXPECTED_CLASSES):
            raise ValueError(f"标签类别越界：{label_path}:{line_number}")
        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"标签宽高非法：{label_path}:{line_number}")
        epsilon = 1e-9
        if (
            x - width / 2 < -epsilon
            or x + width / 2 > 1 + epsilon
            or y - height / 2 < -epsilon
            or y + height / 2 > 1 + epsilon
        ):
            raise ValueError(f"标签框越界：{label_path}:{line_number}")
        class_counter[EXPECTED_CLASSES[class_id]] = class_counter.get(EXPECTED_CLASSES[class_id], 0) + 1
        box_count += 1
    return box_count, class_counter


def scan_split(dataset_root: Path, split: str) -> dict[str, Any]:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"缺少 images/{split} 或 labels/{split}")
    pngs = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    labeled: list[Path] = []
    unlabeled: list[str] = []
    excluded_copies: list[str] = []
    box_count = 0
    class_counter: dict[str, int] = {}
    for png in pngs:
        label = labels_dir / f"{png.stem}.txt"
        if png.stem.endswith(COPY_SUFFIX):
            excluded_copies.append(png.name)
            continue
        if not label.is_file():
            unlabeled.append(png.name)
            continue
        box_count_split, class_counter_split = validate_label_bytes(
            label.read_text(encoding="utf-8"), label
        )
        box_count += box_count_split
        for name, count in class_counter_split.items():
            class_counter[name] = class_counter.get(name, 0) + count
        labeled.append(png)
    return {
        "split": split,
        "png_total": len(pngs),
        "included": labeled,
        "unlabeled": unlabeled,
        "excluded_copies": excluded_copies,
        "box_count": box_count,
        "class_distribution": class_counter,
    }


def build_split_summary(dataset_root: Path, train: dict[str, Any], val: dict[str, Any], classes: list[str]) -> dict[str, Any]:
    train_stems = {p.stem for p in train["included"]}
    val_stems = {p.stem for p in val["included"]}
    if train_stems & val_stems:
        raise ValueError(f"train/val 主名冲突：{sorted(train_stems & val_stems)[:5]}")
    all_upload_names = [p.name for p in train["included"]] + [p.name for p in val["included"]]
    if len(set(all_upload_names)) != len(all_upload_names):
        raise ValueError("上传文件名存在冲突，拒绝扁平化")
    return {
        "schema": "b3_split_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "classes": classes,
        "inclusion_rule": "带标签 PNG；train 额外排除『 - 副本』复制件；无标签图未经人工确认一律排除",
        "train": {
            "png_total": train["png_total"],
            "included_count": len(train["included"]),
            "unlabeled_excluded": train["unlabeled"],
            "copy_excluded": train["excluded_copies"],
            "box_count": train["box_count"],
            "class_distribution": train["class_distribution"],
        },
        "val": {
            "png_total": val["png_total"],
            "included_count": len(val["included"]),
            "unlabeled_excluded": val["unlabeled"],
            "copy_excluded": val["excluded_copies"],
            "box_count": val["box_count"],
            "class_distribution": val["class_distribution"],
        },
    }


def write_splits(splits_dir: Path, train: dict[str, Any], val: dict[str, Any], summary: dict[str, Any]) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train_local.txt").write_text(
        "\n".join(f"images/train/{p.name}" for p in train["included"]) + "\n", encoding="utf-8"
    )
    (splits_dir / "val_local.txt").write_text(
        "\n".join(f"images/val/{p.name}" for p in val["included"]) + "\n", encoding="utf-8"
    )
    (splits_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def create_hard_link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if os.path.samefile(source, destination):
            return
        raise FileExistsError(f"目标文件已存在且不是预期硬链接：{destination}")
    os.link(source, destination)


def copy_small_file(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.read_bytes() == source.read_bytes():
            return
        raise FileExistsError(f"目标文件已存在且内容不同：{destination}")
    shutil.copy2(source, destination)


def stage_upload_view(
    output_root: Path,
    dataset_root: Path,
    train: dict[str, Any],
    val: dict[str, Any],
    summary: dict[str, Any],
    owner_slug: str,
    dataset_slug: str,
    dataset_title: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"上传视图已存在，拒绝覆盖（如需重建请人工删除）：{output_root}")
    if not TRAINER_SOURCE.is_file():
        raise FileNotFoundError(f"缺少训练器脚本：{TRAINER_SOURCE}")
    staging_root = output_root.parent / f"{output_root.name}.building_{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"检测到未完成的上传视图，请人工检查后删除：{staging_root}")
    staging_root.mkdir()

    try:
        image_records = []
        for scan in (train, val):
            for png in scan["included"]:
                record = {
                    "source_split": scan["split"],
                    "source_name": png.name,
                    "upload_name": png.name,
                    "relative_path": f"images/{png.name}",
                }
                image_records.append(record)
                create_hard_link(png, staging_root / record["upload_name"])
        label_records = []
        for scan in (train, val):
            for png in scan["included"]:
                label = dataset_root / "labels" / scan["split"] / f"{png.stem}.txt"
                record = {
                    "source_split": scan["split"],
                    "source_name": label.name,
                    "upload_name": label.name,
                    "relative_path": f"labels/{label.name}",
                }
                label_records.append(record)
                copy_small_file(label, staging_root / record["upload_name"])

        train_list = staging_root / "meta__train.txt"
        val_list = staging_root / "meta__val.txt"
        train_list.write_text(
            "\n".join(record["relative_path"] for record in image_records if record["source_split"] == "train") + "\n",
            encoding="utf-8",
        )
        val_list.write_text(
            "\n".join(record["relative_path"] for record in image_records if record["source_split"] == "val") + "\n",
            encoding="utf-8",
        )
        classes_txt = staging_root / "meta__classes.txt"
        classes_txt.write_text("\n".join(summary["classes"]) + "\n", encoding="utf-8")
        copy_small_file(dataset_root / "data.yaml", staging_root / "meta__source_data.yaml")
        metadata_records = [
            {"source_name": "meta__train.txt", "upload_name": "meta__train.txt", "sha256": sha256_file(train_list)},
            {"source_name": "meta__val.txt", "upload_name": "meta__val.txt", "sha256": sha256_file(val_list)},
            {"source_name": "meta__classes.txt", "upload_name": "meta__classes.txt", "sha256": sha256_file(classes_txt)},
            {"source_name": "meta__source_data.yaml", "upload_name": "meta__source_data.yaml", "sha256": sha256_file(staging_root / "meta__source_data.yaml")},
        ]

        shutil.copy2(TRAINER_SOURCE, staging_root / TRAINER_UPLOAD_NAME)
        manifest = {
            "schema_version": "b3_v1",
            "dataset_root": str(dataset_root),
            "generated_at": summary["generated_at"],
            "split_summary": summary,
            "classes": summary["classes"],
            "counts": {"images": len(image_records), "labels": len(label_records), "metadata": len(metadata_records)},
            "files": {"images": image_records, "labels": label_records, "metadata": metadata_records},
            "scripts": {"trainer": TRAINER_UPLOAD_NAME},
        }
        (staging_root / "flat_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging_root / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": dataset_title,
                    "id": f"{owner_slug}/{dataset_slug}",
                    "licenses": [{"name": "other"}],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging_root / "README_UPLOAD.md").write_text(
            "# B3 Dianjinie0211 扁平上传视图\n\n"
            "由 `build_b3_kaggle_dataset.py` 生成：PNG 为硬链接、标签与元数据为独立副本，"
            "不得在本目录直接编辑。上传：`kaggle datasets create --dir-mode skip`；"
            "Kaggle 端入口见 kernel 仓库 `b3-dianjinie0211-y11m640-seed0.py`。\n",
            encoding="utf-8",
        )
        staging_root.replace(output_root)
        return manifest
    except Exception:
        raise


def publish_view(output_root: Path) -> None:
    command = [
        "kaggle", "datasets", "create",
        "--path", str(output_root),
        "--dir-mode", "skip",
        "--keep-tabular",
    ]
    print("[publish]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    classes = load_classes(dataset_root)
    train = scan_split(dataset_root, "train")
    val = scan_split(dataset_root, "val")
    summary = build_split_summary(dataset_root, train, val, classes)

    plan = {
        "dataset_root": str(dataset_root),
        "classes": classes,
        "train_included": len(train["included"]),
        "train_unlabeled_excluded": len(train["unlabeled"]),
        "train_copy_excluded": len(train["excluded_copies"]),
        "val_included": len(val["included"]),
        "val_unlabeled_excluded": len(val["unlabeled"]),
        "stage": args.stage,
        "output_root": str(args.output_root),
        "dry_run": args.dry_run,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    write_splits(args.splits_dir, train, val, summary)
    print(f"[splits] written: {args.splits_dir}")
    if args.stage:
        manifest = stage_upload_view(
            args.output_root, dataset_root, train, val, summary,
            args.owner_slug, args.dataset_slug, args.dataset_title,
        )
        print(
            f"[stage] ready: {args.output_root} "
            f"({manifest['counts']['images']} images, {manifest['counts']['labels']} labels)"
        )
        print("[stage] upload when ready: kaggle datasets create --dir-mode skip (review dataset-metadata.json first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
