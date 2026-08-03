#!/usr/bin/env python3
"""Build a development-only hard-link view from L1 OOF consensus candidates.

The view deliberately contains only the 1,127 development images and their
candidate five-column labels.  It never reads the frozen test split or copies
any frozen-test image/label.  Source anchor files remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from workspace_config import resolve_path


ROOT = Path(__file__).resolve().parent
DEFAULT_OOF_ROOT = ROOT / "l1_oof_consensus_20260802_232333_v1"
DEFAULT_ANCHOR_ROOT = resolve_path(
    "l1_anchor_ryw18_v1", Path(r"E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1")
)
DEFAULT_OUTPUT_ROOT = ROOT / "dianjienie_ryw18_oofconsensus_v1_dev_only_r1"
EXPECTED_CLASSES = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 L1 OOF 共识候选构建只含开发集的硬链接训练视图，不读取冻结测试集。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--oof-root", type=Path, default=DEFAULT_OOF_ROOT)
    parser.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="只校验输入与构建计划，不写入文件。")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少必要文件：{path}")


def read_classes(path: Path) -> list[str]:
    require_file(path)
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if classes != EXPECTED_CLASSES:
        raise ValueError("类别顺序不是 RYW18，拒绝构建开发集视图")
    return classes


def split_image_name(value: str) -> str:
    name = PurePosixPath(value.strip().replace("\\", "/")).name
    if not name or Path(name).suffix.lower() != ".png":
        raise ValueError(f"划分清单包含非 PNG 图片路径：{value}")
    return name


def read_fold_names(anchor_root: Path, fold: int, kind: str) -> list[str]:
    path = anchor_root / "splits" / "folds" / f"fold_{fold:02d}" / f"{kind}.txt"
    require_file(path)
    names = [split_image_name(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{path} 为空或含重复图片")
    return names


def validate_label(path: Path, class_count: int) -> int:
    boxes = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"候选训练标签字段数错误：{path}:{line_number}")
        try:
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"候选训练标签含非数值字段：{path}:{line_number}") from error
        if not 0 <= class_id < class_count:
            raise ValueError(f"候选训练标签类别越界：{path}:{line_number}")
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"候选训练标签坐标非有限数：{path}:{line_number}")
        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"候选训练标签宽高非法：{path}:{line_number}")
        epsilon = 1e-9
        if x - width / 2 < -epsilon or x + width / 2 > 1 + epsilon or y - height / 2 < -epsilon or y + height / 2 > 1 + epsilon:
            raise ValueError(f"候选训练标签越过图像边界：{path}:{line_number}")
        boxes += 1
    return boxes


def read_oof_summary(oof_root: Path) -> dict[str, Any]:
    summary_path = oof_root / "summary.json"
    require_file(summary_path)
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"OOF summary 无法解析：{summary_path}") from error
    if summary.get("purpose") != "L1 OOF two-architecture consensus candidate generation":
        raise ValueError("OOF 目录用途不匹配")
    if summary.get("images") != 1127 or summary.get("class_names") != EXPECTED_CLASSES:
        raise ValueError("OOF summary 的图片数或类别顺序不匹配")
    if summary.get("frozen_test_access") != "not read; not inferred; not included":
        raise ValueError("OOF summary 未声明冻结测试集隔离，拒绝继续")
    return summary


def collect_inputs(
    oof_root: Path, anchor_root: Path
) -> tuple[dict[str, Any], dict[int, dict[str, list[str]]], list[Path], int, int]:
    summary = read_oof_summary(oof_root)
    oof_classes = read_classes(oof_root / "classes.txt")
    anchor_classes = read_classes(anchor_root / "classes.txt")
    if oof_classes != anchor_classes:
        raise ValueError("OOF 类别文件与 anchor 类别文件不一致")

    candidate_labels_root = oof_root / "labels_consensus_training_candidate_5col"
    if not candidate_labels_root.is_dir():
        raise FileNotFoundError(f"缺少 OOF 五列候选标签目录：{candidate_labels_root}")
    candidate_labels = sorted(candidate_labels_root.glob("*.txt"), key=lambda path: path.name)
    if len(candidate_labels) != 1127:
        raise ValueError(f"OOF 五列候选标签数量应为 1127，实际为 {len(candidate_labels)}")
    candidate_names = {path.with_suffix(".png").name for path in candidate_labels}
    if len(candidate_names) != len(candidate_labels):
        raise ValueError("OOF 五列候选标签文件名重复")

    folds: dict[int, dict[str, list[str]]] = {}
    val_union: set[str] = set()
    box_count = 0
    for fold in range(1, 4):
        train = read_fold_names(anchor_root, fold, "train")
        val = read_fold_names(anchor_root, fold, "val")
        if set(train) & set(val):
            raise ValueError(f"Fold {fold} train/val 存在重叠")
        if set(train) | set(val) != candidate_names:
            raise ValueError(f"Fold {fold} 的开发集图片不等于 OOF 候选标签图片集")
        folds[fold] = {"train": train, "val": val}
        val_union.update(val)
    if val_union != candidate_names:
        raise ValueError("三个 OOF val Fold 的并集不等于候选标签图片集")

    historical_box_count = 0
    for label_path in candidate_labels:
        image_path = anchor_root / "images" / label_path.with_suffix(".png").name
        if not image_path.is_file():
            raise FileNotFoundError(f"候选标签缺少对应开发集源图：{image_path}")
        box_count += validate_label(label_path, len(EXPECTED_CLASSES))
        historical_box_count += validate_label(
            anchor_root / "labels" / label_path.name, len(EXPECTED_CLASSES)
        )

    expected_boxes = historical_box_count + int(summary["counts"].get("action:add_to_training_candidate_view", -1))
    if box_count != expected_boxes:
        raise ValueError(f"候选标签框数为 {box_count}，预期为 {expected_boxes}")
    return summary, folds, candidate_labels, box_count, historical_box_count


def hard_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"构建目录中已有目标文件，拒绝覆盖：{destination}")
    os.link(source, destination)
    if not os.path.samefile(source, destination):
        raise RuntimeError(f"硬链接验证失败：{destination}")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def yaml_content(view_root: Path, train_value: str, val_value: str | None = None) -> str:
    lines = [f'path: "{view_root.as_posix()}"', f'train: "{train_value}"']
    if val_value is not None:
        lines.append(f'val: "{val_value}"')
    lines.extend(["", "nc: 18", "names:"])
    lines.extend(f"  {index}: {name}" for index, name in enumerate(EXPECTED_CLASSES))
    return "\n".join(lines) + "\n"


def source_manifest_rows(anchor_root: Path, image_names: set[str], view_root: Path) -> tuple[list[str], list[dict[str, str]]]:
    source_manifest = anchor_root / "manifests" / "source_manifest.csv"
    require_file(source_manifest)
    with source_manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("anchor source_manifest.csv 缺少表头")
        fields = list(reader.fieldnames)
        rows = list(reader)
    required = {"anchor_image", "anchor_label", "source_image"}
    if not required.issubset(fields):
        raise ValueError("anchor source_manifest.csv 缺少必要列")
    selected: list[dict[str, str]] = []
    for row in rows:
        image_name = Path(row["anchor_image"]).name
        if image_name not in image_names:
            continue
        adjusted = dict(row)
        adjusted["anchor_image"] = str(view_root / "images" / image_name)
        adjusted["anchor_label"] = str(view_root / "labels" / Path(image_name).with_suffix(".txt"))
        selected.append(adjusted)
    selected_names = {Path(row["anchor_image"]).name for row in selected}
    if selected_names != image_names or len(selected) != len(image_names):
        raise ValueError("anchor source_manifest.csv 无法完整映射开发集视图")
    return fields, sorted(selected, key=lambda row: Path(row["anchor_image"]).name)


def digest_label_tree(labels: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in labels:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_view(
    output_root: Path,
    oof_root: Path,
    anchor_root: Path,
    summary: dict[str, Any],
    folds: dict[int, dict[str, list[str]]],
    candidate_labels: list[Path],
    box_count: int,
    historical_box_count: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    image_names = {path.with_suffix(".png").name for path in candidate_labels}
    fields, manifest_rows = source_manifest_rows(anchor_root, image_names, output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f"{output_root.name}.building_{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"发现未完成构建目录，请人工检查：{staging_root}")
    staging_root.mkdir()
    try:
        (staging_root / "labels").mkdir()
        for source_label in candidate_labels:
            image_name = source_label.with_suffix(".png").name
            hard_link(anchor_root / "images" / image_name, staging_root / "images" / image_name)
            shutil.copy2(source_label, staging_root / "labels" / source_label.name)
        shutil.copy2(oof_root / "classes.txt", staging_root / "classes.txt")
        for fold, lists in folds.items():
            fold_root = staging_root / "splits" / "folds" / f"fold_{fold:02d}"
            write_text(fold_root / "train.txt", "\n".join(f"images/{name}" for name in lists["train"]) + "\n")
            write_text(fold_root / "val.txt", "\n".join(f"images/{name}" for name in lists["val"]) + "\n")
            write_text(fold_root / "data.yaml", yaml_content(output_root, "train.txt", "val.txt"))
        write_text(
            staging_root / "data.yaml",
            yaml_content(output_root, "splits/folds/fold_01/train.txt", "splits/folds/fold_01/val.txt"),
        )

        adjusted_rows: list[dict[str, str]] = []
        for row in manifest_rows:
            adjusted = dict(row)
            image_name = Path(adjusted["anchor_image"]).name
            adjusted["anchor_image"] = str(output_root / "images" / image_name)
            adjusted["anchor_label"] = str(output_root / "labels" / Path(image_name).with_suffix(".txt"))
            adjusted_rows.append(adjusted)
        manifest_path = staging_root / "manifests" / "source_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(adjusted_rows)

        manifest = {
            "schema_version": 1,
            "view_name": output_root.name,
            "mode": "development_only",
            "frozen_test_access": "not read; not linked; not included",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_anchor_root": str(anchor_root),
            "source_oof_root": str(oof_root),
            "source_oof_summary_sha256": sha256_file(oof_root / "summary.json"),
            "source_oof_decisions_sha256": sha256_file(oof_root / "decisions.jsonl"),
            "source_candidate_labels_digest": digest_label_tree(candidate_labels),
            "classes": EXPECTED_CLASSES,
            "counts": {
                "images": len(candidate_labels),
                "labels": len(candidate_labels),
                "historical_boxes": historical_box_count,
                "boxes": box_count,
                "accepted_additions": int(summary["counts"]["action:add_to_training_candidate_view"]),
            },
            "fold_sizes": {
                f"fold_{fold:02d}": {name: len(paths) for name, paths in lists.items()}
                for fold, lists in folds.items()
            },
            "image_storage": "NTFS hard links to anchor development images",
        }
        write_text(staging_root / "view_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        write_text(
            staging_root / "README.md",
            "# L1 OOF 共识开发集候选训练视图\n\n"
            "本目录仅包含 1,127 张开发集图片的 NTFS 硬链接和 OOF 双模型共识五列候选标签。"
            "不包含、未读取、未链接冻结测试集。历史锚点未被修改。\n\n"
            "该视图用于后续明确标注为 `development_only` 的实验，不是人工审查真值，"
            "不能把其结果与 anchor v1 基线混为无偏架构结论。完整来源和哈希见 `view_manifest.json`。\n",
        )
        staging_root.replace(output_root)
        return manifest
    except Exception:
        # Preserve a failed staging tree for diagnosis instead of deleting links/files.
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    oof_root = args.oof_root.resolve()
    anchor_root = args.anchor_root.resolve()
    output_root = args.output_root.resolve()
    summary, folds, candidate_labels, box_count, historical_box_count = collect_inputs(oof_root, anchor_root)
    plan = {
        "oof_root": str(oof_root),
        "anchor_root": str(anchor_root),
        "output_root": str(output_root),
        "mode": "development_only",
        "frozen_test_access": "not read; not linked; not included",
        "images": len(candidate_labels),
        "labels": len(candidate_labels),
        "historical_boxes": historical_box_count,
        "boxes": box_count,
        "fold_sizes": {f"fold_{fold:02d}": {name: len(paths) for name, paths in lists.items()} for fold, lists in folds.items()},
        "dry_run": args.dry_run,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    manifest = build_view(
        output_root,
        oof_root,
        anchor_root,
        summary,
        folds,
        candidate_labels,
        box_count,
        historical_box_count,
    )
    print(f"[view] ready: {output_root}")
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
