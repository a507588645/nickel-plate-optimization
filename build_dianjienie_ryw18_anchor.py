#!/usr/bin/env python3
"""Build a deduplicated, grouped 18-class historical anchor from dianjienie.

The source dataset is never changed.  This builder reads only the canonical
``train/images`` + ``train/labels`` and ``valid/images`` + ``valid/labels``
layouts, creates a new hard-link image view, and writes cleaned five-column
labels into the new versioned dataset.

The class ids 0-16 already match the confirmed ryw.pt order.  Only the class
names are normalized and id 17 (fanxitong) is added as an unsupported class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(r"E:/PythonProject/datasets/dianjienie")
DEFAULT_OUTPUT = Path(r"E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NEGATIVE_BUCKET = "__explicit_empty_label__"

RYW_CLASSES = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]
SOURCE_SPLITS = (
    ("train", Path("train/images"), Path("train/labels")),
    ("valid", Path("valid/images"), Path("valid/labels")),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 dianjienie 的去重、分组、18 类历史标签锚点",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="原始 dianjienie 数据集（只读）")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="新的版本化锚点目录")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="分组独立测试集比例")
    parser.add_argument("--folds", type=int, default=3, help="开发集内的分组多标签折数")
    parser.add_argument("--seed", type=int, default=20260724, help="固定划分随机种子")
    parser.add_argument("--near-hamming", type=int, default=8, help="dHash(256 bit) 近重复最大汉明距离")
    parser.add_argument("--near-correlation", type=float, default=0.995,
                        help="缩略灰度图近重复最小相关系数，越高越保守")
    parser.add_argument("--skip-near-duplicates", action="store_true", help="只做精确 SHA-256 去重")
    parser.add_argument("--workers", type=int, default=4, help="哈希和缩略图计算线程数")
    parser.add_argument("--dry-run", action="store_true", help="只审计和计算划分，不写任何产物")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(lines: list[str]) -> str:
    content = "\n".join(sorted(lines)) + ("\n" if lines else "")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_label(label_path: Path) -> dict[str, Any]:
    """Keep valid source lines verbatim; reject malformed or frame-crossing boxes."""
    raw_lines = [line.strip() for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    valid_lines: list[str] = []
    class_ids: list[int] = []
    invalid: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_lines, 1):
        fields = line.split()
        if len(fields) != 5:
            invalid.append({"line": line_number, "reason": f"field_count={len(fields)}", "content": line})
            continue
        try:
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError:
            invalid.append({"line": line_number, "reason": "non_numeric", "content": line})
            continue
        if not 0 <= class_id <= 16:
            invalid.append({"line": line_number, "reason": f"class_id={class_id}", "content": line})
            continue
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            invalid.append({"line": line_number, "reason": "non_finite_coordinate", "content": line})
            continue
        if width <= 0 or height <= 0 or width > 1 or height > 1:
            invalid.append({"line": line_number, "reason": "invalid_width_or_height", "content": line})
            continue
        # A YOLO box must remain inside the normalized image frame.  Tiny
        # floating point noise is accepted without changing source coordinates.
        epsilon = 1e-9
        if x - width / 2 < -epsilon or x + width / 2 > 1 + epsilon or y - height / 2 < -epsilon or y + height / 2 > 1 + epsilon:
            invalid.append({"line": line_number, "reason": "box_crosses_image_boundary", "content": line})
            continue
        valid_lines.append(line)
        class_ids.append(class_id)
    return {
        "raw_line_count": len(raw_lines),
        "valid_lines": valid_lines,
        "class_ids": class_ids,
        "invalid": invalid,
        "label_sha256_clean": sha256_text(valid_lines),
    }


def source_layout_audit(source: Path) -> dict[str, Any]:
    """Record the duplicate source layouts so they cannot be double-counted."""
    checks = []
    mirrors = ((Path("train/images"), Path("images/train")), (Path("valid/images"), Path("images/val")),
               (Path("train/labels"), Path("labels/train")), (Path("valid/labels"), Path("labels/val")))
    for primary_rel, mirror_rel in mirrors:
        primary, mirror = source / primary_rel, source / mirror_rel
        primary_names = {path.name for path in primary.iterdir() if path.is_file()} if primary.is_dir() else set()
        mirror_names = {path.name for path in mirror.iterdir() if path.is_file()} if mirror.is_dir() else set()
        checks.append({
            "primary": str(primary_rel).replace("\\", "/"),
            "mirror": str(mirror_rel).replace("\\", "/"),
            "primary_files": len(primary_names),
            "mirror_files": len(mirror_names),
            "name_difference_count": len(primary_names ^ mirror_names),
        })
    return {"canonical_layout": "train|valid/images + train|valid/labels", "mirror_checks": checks}


def collect_source_samples(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"splits": {}, "invalid_label_lines": []}
    source_order = {"train": 0, "valid": 1}
    for split, image_rel, label_rel in SOURCE_SPLITS:
        image_dir, label_dir = source / image_rel, source / label_rel
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"源数据缺少 {image_dir} 或 {label_dir}")
        images = {path.stem: path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
        labels = {path.stem: path for path in label_dir.glob("*.txt") if path.name.lower() != "classes.txt"}
        missing_labels = sorted(set(images) - set(labels))
        orphan_labels = sorted(set(labels) - set(images))
        audit["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": missing_labels,
            "orphan_labels": orphan_labels,
            "explicit_empty_labels": 0,
            "invalid_label_lines": 0,
            "all_boxes_invalid_images": [],
        }
        for stem in missing_labels:
            excluded.append({
                "reason": "missing_label_file_not_assumed_negative",
                "source_split": split,
                "source_image": str(images[stem]),
                "source_label": "",
                "detail": "图片没有同名标签文件，未自动当作无缺陷样本",
            })
        for stem in orphan_labels:
            excluded.append({
                "reason": "orphan_label_without_image",
                "source_split": split,
                "source_image": "",
                "source_label": str(labels[stem]),
                "detail": "标签没有同名图片",
            })
        for stem in sorted(set(images) & set(labels)):
            image_path, label_path = images[stem], labels[stem]
            parsed = read_label(label_path)
            if not parsed["raw_line_count"]:
                audit["splits"][split]["explicit_empty_labels"] += 1
            audit["splits"][split]["invalid_label_lines"] += len(parsed["invalid"])
            for problem in parsed["invalid"]:
                audit["invalid_label_lines"].append({
                    "source_split": split,
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    **problem,
                })
            if parsed["raw_line_count"] and not parsed["valid_lines"]:
                audit["splits"][split]["all_boxes_invalid_images"].append(stem)
                excluded.append({
                    "reason": "all_source_boxes_invalid",
                    "source_split": split,
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    "detail": "非空标签的所有框均不合法，未自动改为空负样本",
                })
                continue
            samples.append({
                "sample_id": f"{split}__{stem}",
                "source_split": split,
                "source_order": source_order[split],
                "source_image": image_path,
                "source_label": label_path,
                "source_stem": stem,
                "output_name": f"dianjienie__{split}__{stem}{image_path.suffix.lower()}",
                "valid_lines": parsed["valid_lines"],
                "class_ids": parsed["class_ids"],
                "raw_line_count": parsed["raw_line_count"],
                "invalid_line_count": len(parsed["invalid"]),
                "label_sha256_clean": parsed["label_sha256_clean"],
            })
    return samples, audit, excluded


def hash_samples(samples: list[dict[str, Any]], workers: int) -> None:
    """Hash only equal-size candidates; a unique byte size cannot be an exact duplicate."""
    size_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        sample["image_size_bytes"] = sample["source_image"].stat().st_size
        size_groups[sample["image_size_bytes"]].append(sample)
    candidates = [sample for group in size_groups.values() if len(group) > 1 for sample in group]
    for sample in samples:
        if sample not in candidates:
            sample["image_sha256"] = f"unique_by_size:{sample['image_size_bytes']}:{sample['sample_id']}"
    print(f"[hash] 同尺寸精确重复候选: {len(candidates)}/{len(samples)}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(sha256_file, sample["source_image"]): sample for sample in candidates}
        for future in as_completed(futures):
            futures[future]["image_sha256"] = future.result()


def choose_canonical(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return min(samples, key=lambda item: (item["source_order"], item["source_stem"], str(item["source_image"])))


def exact_deduplicate(samples: list[dict[str, Any]], excluded: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_hash[sample["image_sha256"]].append(sample)
    retained: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for image_hash, members in sorted(by_hash.items()):
        if len(members) == 1:
            member = members[0]
            member["exact_group"] = f"sha256:{image_hash}"
            retained.append(member)
            continue
        label_hashes = {member["label_sha256_clean"] for member in members}
        group = {
            "image_sha256": image_hash,
            "count": len(members),
            "members": [member["sample_id"] for member in sorted(members, key=lambda item: item["sample_id"])],
            "clean_label_hashes": sorted(label_hashes),
        }
        if len(label_hashes) != 1:
            group["resolution"] = "excluded_label_conflict"
            groups.append(group)
            for member in members:
                excluded.append({
                    "reason": "exact_duplicate_with_label_conflict",
                    "source_split": member["source_split"],
                    "source_image": str(member["source_image"]),
                    "source_label": str(member["source_label"]),
                    "detail": f"sha256={image_hash}",
                })
            continue
        canonical = choose_canonical(members)
        canonical["exact_group"] = f"sha256:{image_hash}"
        retained.append(canonical)
        group["resolution"] = "kept_canonical"
        group["canonical"] = canonical["sample_id"]
        groups.append(group)
        for member in members:
            if member is canonical:
                continue
            excluded.append({
                "reason": "exact_duplicate_of_canonical",
                "source_split": member["source_split"],
                "source_image": str(member["source_image"]),
                "source_label": str(member["source_label"]),
                "detail": f"canonical={canonical['sample_id']};sha256={image_hash}",
            })
    return retained, groups


def read_gray(path: Path) -> np.ndarray | None:
    """Read the paired source array at low resolution without decoding a 4K PNG."""
    try:
        array_path = path.with_suffix(".npy")
        if not array_path.is_file():
            return None
        array = np.load(array_path, mmap_mode="r")
        if array.ndim == 3:
            rows = np.linspace(0, array.shape[0] - 1, 64).round().astype(np.intp)
            cols = np.linspace(0, array.shape[1] - 1, 64).round().astype(np.intp)
            sampled = array[rows][:, cols, :].astype(np.float32)
            # Fixed BT.601 conversion is sufficient for duplicate grouping.
            return np.clip(0.299 * sampled[:, :, 0] + 0.587 * sampled[:, :, 1] + 0.114 * sampled[:, :, 2], 0, 255).astype(np.uint8)
        if array.ndim == 2:
            rows = np.linspace(0, array.shape[0] - 1, 64).round().astype(np.intp)
            cols = np.linspace(0, array.shape[1] - 1, 64).round().astype(np.intp)
            return array[rows][:, cols].astype(np.uint8)
        return None
    except (OSError, ValueError, TypeError):
        return None


def near_signature(path: Path) -> tuple[int, np.ndarray] | None:
    image = read_gray(path)
    if image is None:
        return None
    row_indices = np.linspace(0, image.shape[0] - 1, 16).round().astype(np.intp)
    col_indices = np.linspace(0, image.shape[1] - 1, 17).round().astype(np.intp)
    dhash_image = image[row_indices][:, col_indices]
    bits = (dhash_image[:, 1:] > dhash_image[:, :-1]).reshape(-1)
    value = int.from_bytes(np.packbits(bits).tobytes(), byteorder="big")
    row_indices = np.linspace(0, image.shape[0] - 1, 32).round().astype(np.intp)
    col_indices = np.linspace(0, image.shape[1] - 1, 32).round().astype(np.intp)
    thumb = image[row_indices][:, col_indices]
    return value, thumb


def thumbnail_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    a, b = left.astype(np.float32).reshape(-1), right.astype(np.float32).reshape(-1)
    mae = float(np.mean(np.abs(a - b)))
    a_centered, b_centered = a - float(a.mean()), b - float(b.mean())
    denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denominator <= 1e-12:
        return (1.0 if mae <= 1e-6 else 0.0), mae
    return float(np.dot(a_centered, b_centered) / denominator), mae


def find_near_duplicate_edges(
    samples: list[dict[str, Any]], hamming_threshold: int, correlation_threshold: float, workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    del workers  # Decode sequentially: industrial PNGs can exhaust memory in parallel OpenCV decoders.
    signatures: dict[str, tuple[int, np.ndarray]] = {}
    unreadable: list[dict[str, Any]] = []
    for number, sample in enumerate(samples, 1):
        result = near_signature(sample["source_image"])
        if result is None:
            unreadable.append({"sample_id": sample["sample_id"], "source_image": str(sample["source_image"])})
        else:
            signatures[sample["sample_id"]] = result
        if number % 100 == 0 or number == len(samples):
            print(f"[near] 签名进度: {number}/{len(samples)}")
    edges: list[dict[str, Any]] = []
    ordered = sorted((sample for sample in samples if sample["sample_id"] in signatures), key=lambda item: item["sample_id"])
    for left_index, left in enumerate(ordered):
        left_hash, left_thumb = signatures[left["sample_id"]]
        for right in ordered[left_index + 1:]:
            right_hash, right_thumb = signatures[right["sample_id"]]
            hamming = (left_hash ^ right_hash).bit_count()
            if hamming > hamming_threshold:
                continue
            correlation, mae = thumbnail_correlation(left_thumb, right_thumb)
            if correlation >= correlation_threshold:
                edges.append({
                    "sample_id_1": left["sample_id"],
                    "sample_id_2": right["sample_id"],
                    "hamming": hamming,
                    "correlation": round(correlation, 8),
                    "thumbnail_mae": round(mae, 6),
                })
    return edges, unreadable


class UnionFind:
    def __init__(self, keys: list[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_groups(samples: list[dict[str, Any]], near_edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {sample["sample_id"]: sample for sample in samples}
    union_find = UnionFind(list(index))
    for edge in near_edges:
        union_find.union(edge["sample_id_1"], edge["sample_id_2"])
    members_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_id, sample in index.items():
        members_by_root[union_find.find(sample_id)].append(sample)
    groups: list[dict[str, Any]] = []
    for number, members in enumerate(sorted(members_by_root.values(), key=lambda values: min(item["sample_id"] for item in values)), 1):
        group_id = f"group_{number:05d}"
        image_presence: Counter[str] = Counter()
        box_counts: Counter[int] = Counter()
        for member in members:
            class_set = sorted(set(member["class_ids"]))
            if class_set:
                image_presence.update(str(class_id) for class_id in class_set)
            else:
                image_presence[NEGATIVE_BUCKET] += 1
            box_counts.update(member["class_ids"])
            member["group_id"] = group_id
        groups.append({
            "group_id": group_id,
            "members": sorted(members, key=lambda item: item["sample_id"]),
            "size": len(members),
            "image_presence": image_presence,
            "box_counts": box_counts,
        })
    return groups


def assign_grouped_multilabel(
    groups: list[dict[str, Any]], names: list[str], fractions: list[float], seed: int
) -> dict[str, str]:
    """Deterministic linear-time greedy assignment for grouped multilabel balance."""
    if not groups:
        raise ValueError("没有可划分的分组")
    if len(names) != len(fractions) or not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("分区名称与比例不匹配")
    rng = random.Random(seed)
    total_presence = Counter()
    for group in groups:
        total_presence.update(group["image_presence"])
    tie_break = {group["group_id"]: rng.random() for group in groups}

    def rarity(group: dict[str, Any]) -> float:
        return sum(group["image_presence"][label] / max(total_presence[label], 1) for label in group["image_presence"])

    ordered = sorted(groups, key=lambda group: (-rarity(group), -group["size"], tie_break[group["group_id"]], group["group_id"]))
    assignments: dict[str, str] = {}
    current_sizes: Counter[str] = Counter()
    current_presence: dict[str, Counter[str]] = {name: Counter() for name in names}
    total_size = sum(group["size"] for group in groups)
    for group in ordered:
        best_name: str | None = None
        best_score: tuple[float, float, str] | None = None
        for name, fraction in zip(names, fractions):
            size_target = total_size * fraction
            size_cost = ((current_sizes[name] + group["size"] - size_target) ** 2 - (current_sizes[name] - size_target) ** 2) / max(size_target, 1.0)
            label_cost = 0.0
            for label, count in group["image_presence"].items():
                target = total_presence[label] * fraction
                before = current_presence[name][label]
                label_cost += ((before + count - target) ** 2 - (before - target) ** 2) / max(target, 1.0)
            # Label balance matters more than raw image count; use current
            # occupancy as a deterministic tie-breaker.
            key = (round(2.0 * label_cost + size_cost, 12), current_sizes[name] / max(size_target, 1.0), name)
            if best_score is None or key < best_score:
                best_name, best_score = name, key
        if best_name is None:
            raise RuntimeError("无法给分组分区")
        assignments[group["group_id"]] = best_name
        current_sizes[best_name] += group["size"]
        current_presence[best_name].update(group["image_presence"])
    return assignments


def records_for_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [member for group in groups for member in group["members"]]


def distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    image_presence: Counter[int] = Counter()
    box_counts: Counter[int] = Counter()
    explicit_empty = 0
    for record in records:
        classes = set(record["class_ids"])
        if not classes:
            explicit_empty += 1
        image_presence.update(classes)
        box_counts.update(record["class_ids"])
    return {
        "images": len(records),
        "explicit_empty_labels": explicit_empty,
        "class_image_counts": {RYW_CLASSES[class_id]: image_presence[class_id] for class_id in range(len(RYW_CLASSES))},
        "class_box_counts": {RYW_CLASSES[class_id]: box_counts[class_id] for class_id in range(len(RYW_CLASSES))},
    }


def make_split_plan(groups: list[dict[str, Any]], test_ratio: float, folds: int, seed: int) -> dict[str, Any]:
    if not 0 < test_ratio < 0.5:
        raise ValueError("test-ratio 必须在 (0, 0.5) 内")
    if folds < 2:
        raise ValueError("folds 至少为 2")
    test_assignment = assign_grouped_multilabel(groups, ["test", "development"], [test_ratio, 1 - test_ratio], seed)
    # A class in fewer than folds + 1 independent groups cannot be evaluated
    # reliably in every development Fold. Keep all of its groups in the
    # development set, and report the final-test metric for that class as N/A.
    support_groups: Counter[int] = Counter()
    for group in groups:
        support_groups.update({class_id for member in group["members"] for class_id in member["class_ids"]})
    protected_classes = {class_id for class_id, count in support_groups.items() if count < folds + 1}
    for group in groups:
        group_classes = {class_id for member in group["members"] for class_id in member["class_ids"]}
        if group_classes & protected_classes:
            test_assignment[group["group_id"]] = "development"
    test_groups = [group for group in groups if test_assignment[group["group_id"]] == "test"]
    development_groups = [group for group in groups if test_assignment[group["group_id"]] == "development"]
    fold_names = [f"fold_{number:02d}" for number in range(1, folds + 1)]
    fold_assignment = assign_grouped_multilabel(development_groups, fold_names, [1 / folds] * folds, seed + 1)
    folds_plan: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for fold_name in fold_names:
        validation = [group for group in development_groups if fold_assignment[group["group_id"]] == fold_name]
        training = [group for group in development_groups if fold_assignment[group["group_id"]] != fold_name]
        folds_plan[fold_name] = {"train_groups": training, "val_groups": validation}
    return {
        "test_groups": test_groups,
        "development_groups": development_groups,
        "folds": folds_plan,
        "test_assignment": test_assignment,
        "fold_assignment": fold_assignment,
    }


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def yaml_content(root: Path, train_txt: Path, val_txt: Path, test_txt: Path) -> str:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(RYW_CLASSES))
    return (
        f'path: "{root.as_posix()}"\n'
        f'train: "{train_txt.as_posix()}"\n'
        f'val: "{val_txt.as_posix()}"\n'
        f'test: "{test_txt.as_posix()}"\n\n'
        f"nc: {len(RYW_CLASSES)}\n"
        f"names:\n{names}\n"
    )


def build_output(
    output: Path,
    samples: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    plan: dict[str, Any],
    audit: dict[str, Any],
    excluded: list[dict[str, Any]],
    exact_groups: list[dict[str, Any]],
    near_edges: list[dict[str, Any]],
    unreadable: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if output.exists():
        raise FileExistsError(f"输出目录已存在，为防止覆盖已拒绝：{output}")
    staging = output.parent / f".{output.name}.building_{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"临时构建目录已存在，为防止删除未知内容已拒绝：{staging}")
    images_dir, labels_dir = staging / "images", staging / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for sample in samples:
        image_target = images_dir / sample["output_name"]
        try:
            os.link(sample["source_image"], image_target)
        except OSError as error:
            raise RuntimeError(f"硬链接失败：{sample['source_image']} -> {image_target}: {error}") from error
        label_target = labels_dir / f"{Path(sample['output_name']).stem}.txt"
        write_lines(label_target, sample["valid_lines"])
        # Files are built in a same-volume staging directory and atomically
        # renamed at the end.  Persist final paths, not temporary ones.
        sample["anchor_image"] = output / "images" / sample["output_name"]
        sample["anchor_label"] = output / "labels" / f"{Path(sample['output_name']).stem}.txt"

    write_lines(staging / "classes.txt", RYW_CLASSES)
    test_records = records_for_groups(plan["test_groups"])
    development_records = records_for_groups(plan["development_groups"])
    split_dir = staging / "splits"
    write_lines(split_dir / "test.txt", [str(record["anchor_image"]) for record in sorted(test_records, key=lambda item: item["output_name"])])
    write_lines(split_dir / "development.txt", [str(record["anchor_image"]) for record in sorted(development_records, key=lambda item: item["output_name"])])
    for fold_name, fold in plan["folds"].items():
        fold_dir = split_dir / "folds" / fold_name
        train_records = records_for_groups(fold["train_groups"])
        val_records = records_for_groups(fold["val_groups"])
        train_txt, val_txt, test_txt = fold_dir / "train.txt", fold_dir / "val.txt", split_dir / "test.txt"
        write_lines(train_txt, [str(record["anchor_image"]) for record in sorted(train_records, key=lambda item: item["output_name"])])
        write_lines(val_txt, [str(record["anchor_image"]) for record in sorted(val_records, key=lambda item: item["output_name"])])
        final_fold_dir = output / "splits" / "folds" / fold_name
        (fold_dir / "data.yaml").write_text(
            yaml_content(output, final_fold_dir / "train.txt", final_fold_dir / "val.txt", output / "splits" / "test.txt"),
            encoding="utf-8",
        )
    first_fold = next(iter(plan["folds"]))
    (staging / "data.yaml").write_text(
        yaml_content(output, output / "splits" / "folds" / first_fold / "train.txt",
                     output / "splits" / "folds" / first_fold / "val.txt", output / "splits" / "test.txt"),
        encoding="utf-8",
    )

    manifest_rows = []
    for sample in sorted(samples, key=lambda item: item["output_name"]):
        manifest_rows.append({
            "sample_id": sample["sample_id"],
            "anchor_image": str(sample["anchor_image"]),
            "anchor_label": str(sample["anchor_label"]),
            "source_split": sample["source_split"],
            "source_image": str(sample["source_image"]),
            "source_label": str(sample["source_label"]),
            "image_sha256": sample["image_sha256"],
            "label_sha256_clean": sample["label_sha256_clean"],
            "group_id": sample["group_id"],
            "exact_group": sample["exact_group"],
            "raw_label_lines": sample["raw_line_count"],
            "invalid_label_lines_removed": sample["invalid_line_count"],
            "valid_boxes": len(sample["class_ids"]),
            "class_ids": " ".join(str(class_id) for class_id in sorted(set(sample["class_ids"]))),
        })
    write_csv(
        staging / "manifests" / "source_manifest.csv", manifest_rows,
        ["sample_id", "anchor_image", "anchor_label", "source_split", "source_image", "source_label", "image_sha256",
         "label_sha256_clean", "group_id", "exact_group", "raw_label_lines", "invalid_label_lines_removed", "valid_boxes", "class_ids"],
    )
    group_rows = []
    test_ids = {group["group_id"] for group in plan["test_groups"]}
    for group in groups:
        fold = "test" if group["group_id"] in test_ids else plan["fold_assignment"][group["group_id"]]
        group_rows.append({
            "group_id": group["group_id"],
            "partition": fold,
            "images": group["size"],
            "members": "|".join(member["sample_id"] for member in group["members"]),
            "class_image_presence": "|".join(f"{key}:{value}" for key, value in sorted(group["image_presence"].items())),
        })
    write_csv(staging / "manifests" / "group_assignments.csv", group_rows,
              ["group_id", "partition", "images", "members", "class_image_presence"])
    write_csv(staging / "audit" / "excluded_samples.csv", excluded,
              ["reason", "source_split", "source_image", "source_label", "detail"])
    write_csv(staging / "audit" / "invalid_label_lines.csv", audit["invalid_label_lines"],
              ["source_split", "source_image", "source_label", "line", "reason", "content"])
    exact_rows = []
    for group in exact_groups:
        exact_rows.append({
            "image_sha256": group["image_sha256"],
            "count": group["count"],
            "resolution": group["resolution"],
            "canonical": group.get("canonical", ""),
            "members": "|".join(group["members"]),
            "clean_label_hashes": "|".join(group["clean_label_hashes"]),
        })
    write_csv(staging / "audit" / "exact_duplicate_groups.csv", exact_rows,
              ["image_sha256", "count", "resolution", "canonical", "members", "clean_label_hashes"])
    write_csv(staging / "audit" / "near_duplicate_pairs.csv", near_edges,
              ["sample_id_1", "sample_id_2", "hamming", "correlation", "thumbnail_mae"])
    write_csv(staging / "audit" / "unreadable_images.csv", unreadable, ["sample_id", "source_image"])

    fold_distributions = {}
    for fold_name, fold in plan["folds"].items():
        fold_distributions[fold_name] = {
            "train": distribution(records_for_groups(fold["train_groups"])),
            "val": distribution(records_for_groups(fold["val_groups"])),
        }
    support_groups = Counter()
    for group in groups:
        for class_id in {class_id for member in group["members"] for class_id in member["class_ids"]}:
            support_groups[class_id] += 1
    unsupported = [RYW_CLASSES[class_id] for class_id in range(len(RYW_CLASSES)) if support_groups[class_id] == 0]
    insufficient_for_folds = [RYW_CLASSES[class_id] for class_id in range(len(RYW_CLASSES)) if 0 < support_groups[class_id] < args.folds]
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(args.source.resolve()),
        "source_layout": source_layout_audit(args.source.resolve()),
        "output": str(output),
        "class_mapping": {
            "source_ids_0_to_16": "identity mapping to ryw.pt ids 0..16",
            "renamed_classes": {"2": "oilstain->yellowstain", "14": "unknown_14->youpao", "15": "unknown_15->heibian", "16": "unknown_16->mijixiaokeng"},
            "added_zero_support_class": "17 fanxitong",
        },
        "parameters": {
            "seed": args.seed,
            "test_ratio": args.test_ratio,
            "folds": args.folds,
            "near_duplicates_enabled": not args.skip_near_duplicates,
            "near_hamming": args.near_hamming,
            "near_correlation": args.near_correlation,
        },
        "source_audit": audit["splits"],
        "anchor_images": len(samples),
        "anchor_groups": len(groups),
        "excluded_samples": len(excluded),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicates_removed": sum(1 for item in excluded if item["reason"] == "exact_duplicate_of_canonical"),
        "exact_duplicate_label_conflict_images_excluded": sum(1 for item in excluded if item["reason"] == "exact_duplicate_with_label_conflict"),
        "near_duplicate_pairs": len(near_edges),
        "near_duplicate_multi_image_groups": sum(1 for group in groups if group["size"] > 1),
        "unreadable_images": len(unreadable),
        "full_anchor_distribution": distribution(samples),
        "test_distribution": distribution(test_records),
        "development_distribution": distribution(development_records),
        "fold_distributions": fold_distributions,
        "unsupported_classes": unsupported,
        "classes_with_fewer_groups_than_folds": insufficient_for_folds,
    }
    (staging / "audit" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = [
        "# dianjienie RYW18 历史标签锚点",
        "",
        f"- 源数据（只读）：`{args.source.resolve()}`",
        f"- 新锚点：`{output}`",
        "- 图片方式：仅 PNG 训练图硬链接；没有复制或修改源图片。",
        "- 标签方式：复制为新五列标签；仅移除确定性非法行，源标签未修改。",
        "- 类别：源 id `0..16` 保持不变，类名对齐 `ryw.pt`；新增 id `17 fanxitong`，当前零样本。",
        f"- 精确重复：移除 {summary['exact_duplicates_removed']} 张；标签冲突重复图排除 {summary['exact_duplicate_label_conflict_images_excluded']} 张。",
        f"- 近重复：保留图片但按 {summary['near_duplicate_pairs']} 对保守近重复关系合并 group，绝不跨测试/开发或 Fold。",
        f"- 独立测试集：{summary['test_distribution']['images']} 张；开发集：{summary['development_distribution']['images']} 张；开发集 {args.folds} 折。",
        "",
        "## 训练入口",
        "",
        "每个 Fold 的 `data.yaml` 位于 `splits/folds/fold_XX/data.yaml`。YOLO11m 和 YOLO26m 必须使用相同 Fold 文件；`splits/test.txt` 保持冻结，不参与训练、早停、模型选择或阈值选择。",
        "",
        "## 限制",
        "",
        f"- 当前零监督类别：{', '.join(summary['unsupported_classes']) or '无'}。",
        f"- 无法覆盖每个开发 Fold 的稀有类别：{', '.join(summary['classes_with_fewer_groups_than_folds']) or '无'}。",
        "- 未配对的无标签图片未当作负样本，已记录在 `audit/excluded_samples.csv`。",
        "- 近重复检测使用 dHash + 缩略灰度相关性，是保守的泄漏防护，不等同于业务物理板 ID 元数据。",
    ]
    (staging / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (split_dir / "README.md").write_text(
        "# 划分说明\n\n`test.txt` 是冻结独立测试集。`folds/fold_XX/val.txt` 是开发验证集；同目录的 `train.txt` 是其余开发样本。\n",
        encoding="utf-8",
    )
    staging.replace(output)


def dry_run_summary(
    samples: list[dict[str, Any]], groups: list[dict[str, Any]], plan: dict[str, Any], audit: dict[str, Any],
    excluded: list[dict[str, Any]], exact_groups: list[dict[str, Any]], near_edges: list[dict[str, Any]], unreadable: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    test_records = records_for_groups(plan["test_groups"])
    development_records = records_for_groups(plan["development_groups"])
    return {
        "source": str(args.source.resolve()),
        "source_audit": audit["splits"],
        "eligible_after_structural_validation": len(samples) + sum(1 for item in excluded if item["reason"] == "exact_duplicate_of_canonical"),
        "anchor_images_after_exact_dedup": len(samples),
        "groups_after_near_duplicate_grouping": len(groups),
        "excluded_samples": len(excluded),
        "exact_duplicate_groups": len(exact_groups),
        "near_duplicate_pairs": len(near_edges),
        "unreadable_images": len(unreadable),
        "test_images": len(test_records),
        "development_images": len(development_records),
        "fold_validation_images": {name: len(records_for_groups(fold["val_groups"])) for name, fold in plan["folds"].items()},
        "distribution": distribution(samples),
        "test_distribution": distribution(test_records),
        "development_distribution": distribution(development_records),
        "fold_distributions": {
            name: {
                "train": distribution(records_for_groups(fold["train_groups"])),
                "val": distribution(records_for_groups(fold["val_groups"])),
            }
            for name, fold in plan["folds"].items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.source = args.source.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"输出目录已存在，为防止覆盖已拒绝：{args.output_dir}")
    samples, audit, excluded = collect_source_samples(args.source)
    print(f"[audit] 配对且可解析样本: {len(samples)}；预排除: {len(excluded)}")
    hash_samples(samples, args.workers)
    retained, exact_groups = exact_deduplicate(samples, excluded)
    print(f"[dedup] 精确去重后: {len(retained)}；精确重复组: {len(exact_groups)}")
    if args.skip_near_duplicates:
        near_edges, unreadable = [], []
    else:
        near_edges, unreadable = find_near_duplicate_edges(retained, args.near_hamming, args.near_correlation, args.workers)
    print(f"[near] 近重复对: {len(near_edges)}；不可读: {len(unreadable)}")
    groups = build_groups(retained, near_edges)
    plan = make_split_plan(groups, args.test_ratio, args.folds, args.seed)
    preview = dry_run_summary(retained, groups, plan, audit, excluded, exact_groups, near_edges, unreadable, args)
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    build_output(args.output_dir, retained, groups, plan, audit, excluded, exact_groups, near_edges, unreadable, args)
    print(f"[done] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
