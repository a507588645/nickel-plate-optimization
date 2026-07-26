#!/usr/bin/env python3
"""L1 汇总检测集标签质量审计（只读）。

审计内容：
- 图像/标签对应关系、空标签、孤立标签、非法类别、非法/越界坐标、重复框；
- 类别与每图框数分布；
- SHA-256 完全重复和 dHash 视觉近重复候选；
- 基于结构问题、稀有类和既有 ryw.pt 分歧结果的人工审查优先级。

该脚本只读取 ``union_original_labels_v1``，所有报告写入当前优化目录。
视觉近重复仅是候选，不等同于同一物理镍板，必须人工确认后才能用于分组划分。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_UNION = Path(r"E:/PythonProject/datasets/union_original_labels_v1")
DEFAULT_OUTPUT = ROOT / "label_quality_audit_v1"
DEFAULT_WORST = ROOT / "label_check_ryw_v1" / "worst_images.csv"

CLASSES_18 = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class LabelIssue:
    image_name: str
    label_file: str
    line_number: int
    issue: str
    detail: str


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="L1 汇总检测集标签质量审计（只读，不修改原图或原标签）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--union", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worst-images", type=Path, default=DEFAULT_WORST,
                        help="旧模型分歧清单；仅与 --include-model-disagreement 一起使用")
    parser.add_argument("--include-model-disagreement", action="store_true",
                        help="将旧模型分歧作为额外抽样信号；默认关闭，人工标签重建不受旧模型影响")
    parser.add_argument("--near-duplicate-hamming", type=int, default=4,
                        help="dHash 汉明距离不超过该值时列为视觉近重复候选")
    parser.add_argument("--max-near-duplicate-pairs", type=int, default=100000,
                        help="视觉近重复候选对的处理上限，防止极相似批次耗尽内存")
    parser.add_argument("--near-duplicates", action="store_true",
                        help="显式运行 dHash 视觉近重复候选检查；3265 张高分辨率图可能耗时较长")
    parser.add_argument("--skip-near-duplicates", action="store_true",
                        help="兼容旧调用：强制跳过 dHash 视觉近重复检查")
    parser.add_argument("--dry-run", action="store_true", help="只统计并打印，不写报告")
    return parser.parse_args(argv)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_label_file(path: Path, image_name: str) -> tuple[list[list[float]], list[LabelIssue], int]:
    """Return valid YOLO boxes, structural issues, and duplicate-box count."""
    boxes: list[list[float]] = []
    issues: list[LabelIssue] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    duplicate_boxes = 0

    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            issues.append(LabelIssue(image_name, path.name, line_number, "malformed_field_count", line))
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            issues.append(LabelIssue(image_name, path.name, line_number, "non_numeric_value", line))
            continue

        if class_id < 0 or class_id >= len(CLASSES_18):
            issues.append(LabelIssue(image_name, path.name, line_number, "invalid_class_id", str(class_id)))
            continue

        cx, cy, width, height = values
        if not all(math.isfinite(value) for value in values):
            issues.append(LabelIssue(image_name, path.name, line_number, "non_finite_coordinate", line))
            continue
        if width <= 0 or height <= 0:
            issues.append(LabelIssue(image_name, path.name, line_number, "non_positive_box_size", line))
            continue
        if (
            cx < 0 or cx > 1 or cy < 0 or cy > 1 or width > 1 or height > 1
            or cx - width / 2 < 0 or cy - height / 2 < 0
            or cx + width / 2 > 1 or cy + height / 2 > 1
        ):
            issues.append(LabelIssue(image_name, path.name, line_number, "box_out_of_bounds", line))
            continue

        key = (class_id, *(round(value, 6) for value in values))
        if key in seen:
            duplicate_boxes += 1
            issues.append(LabelIssue(image_name, path.name, line_number, "duplicate_box", line))
            continue
        seen.add(key)
        boxes.append([class_id, cx, cy, width, height])

    return boxes, issues, duplicate_boxes


def dhash(path: Path) -> int:
    """Decode a Windows/Chinese path safely and calculate a 64-bit dHash."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("近重复审计需要 Pillow") from exc

    # Pillow is more robust than OpenCV for a heterogeneous PNG archive and
    # avoids native decoder failures aborting the entire audit process.
    with Image.open(path) as source:
        resized = ImageOps.grayscale(source).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(resized.get_flattened_data())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column + 1] > pixels[offset + column])
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def find_near_duplicates(
    images: dict[str, Path], threshold: int, pair_limit: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
    """Return near-duplicate edges, cluster members, and decode failures.

    Eight one-byte buckets bound candidate generation: any pair with at most four
    different hash bits shares at least one byte, so no true candidate at the
    requested default threshold is silently dropped by bucketing.
    """
    hashes: dict[str, int] = {}
    failures: list[dict[str, str]] = []
    for index, (name, path) in enumerate(sorted(images.items()), start=1):
        try:
            hashes[name] = dhash(path)
        except Exception as exc:  # decode failure should not abort the label audit
            failures.append({"image_name": name, "image_path": str(path), "error": str(exc)})
        if index % 50 == 0:
            print(f"dHash: {index}/{len(images)}", flush=True)

    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for name, value in hashes.items():
        for byte_index in range(8):
            buckets[(byte_index, (value >> (byte_index * 8)) & 0xFF)].append(name)

    checked: set[tuple[str, str]] = set()
    uf = UnionFind()
    edges: list[dict[str, object]] = []
    truncated = False
    # Very common dHash bytes can create millions of combinations in homogeneous
    # line-scan batches. A pair cap is therefore enforced before materialising
    # a Cartesian product; once hit, the output is explicitly marked incomplete.
    for names in sorted(buckets.values(), key=len):
        if len(names) < 2:
            continue
        if len(checked) >= pair_limit:
            truncated = True
            break
        for left, right in combinations(sorted(names), 2):
            pair = (left, right)
            if pair in checked:
                continue
            checked.add(pair)
            if len(checked) >= pair_limit:
                truncated = True
                break
            distance = hamming(hashes[left], hashes[right])
            if distance <= threshold:
                uf.union(left, right)
                edges.append({"image_a": left, "image_b": right, "dhash_distance": distance})
        if truncated:
            break

    # Work from edge endpoints rather than repeatedly scanning all edges for
    # every image. The prior formulation was quadratic for dense image batches.
    connected_names = {
        name
        for edge in edges
        for name in (str(edge["image_a"]), str(edge["image_b"]))
    }
    groups: dict[str, list[str]] = defaultdict(list)
    for name in connected_names:
        groups[uf.find(name)].append(name)

    members: list[dict[str, object]] = []
    for cluster_index, names in enumerate(sorted(groups.values(), key=lambda group: (-len(group), group)), start=1):
        if len(names) < 2:
            continue
        cluster_id = f"near_{cluster_index:04d}"
        for name in sorted(names):
            members.append({"cluster_id": cluster_id, "image_name": name, "cluster_size": len(names)})

    if truncated:
        failures.append({
            "image_name": "",
            "image_path": "",
            "error": f"candidate_pair_limit_reached:{pair_limit}; near-duplicate result is incomplete",
        })
    return edges, members, failures


def read_worst_scores(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    scores: dict[str, float] = {}
    for row in csv_rows(path):
        try:
            scores[row["union_name"]] = float(row.get("score", 0))
        except (KeyError, ValueError):
            continue
    return scores


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return values[index]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    union = args.union.resolve()
    images_dir = union / "images"
    labels_dir = union / "labels"
    manifest_path = union / "manifests" / "samples.csv"
    if not images_dir.is_dir() or not labels_dir.is_dir() or not manifest_path.is_file():
        print("union 目录必须包含 images/、labels/ 和 manifests/samples.csv", file=sys.stderr)
        return 2

    manifest = {row["union_name"]: row for row in csv_rows(manifest_path)}
    images = {
        path.stem: path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {path.stem: path for path in labels_dir.glob("*.txt") if path.is_file()}

    issues: list[LabelIssue] = []
    boxes_by_image: dict[str, list[list[float]]] = {}
    class_box_counts: Counter[int] = Counter()
    class_image_counts: Counter[int] = Counter()
    duplicate_box_count = 0
    empty_labels: list[str] = []

    for name, path in sorted(labels.items()):
        if name not in images:
            issues.append(LabelIssue(name, path.name, 0, "orphan_label_file", "no matching image"))
            continue
        if not path.read_text(encoding="utf-8", errors="replace").strip():
            empty_labels.append(name)
        boxes, label_issues, duplicates = parse_label_file(path, name)
        boxes_by_image[name] = boxes
        issues.extend(label_issues)
        duplicate_box_count += duplicates
        seen_classes = {int(box[0]) for box in boxes}
        class_box_counts.update(int(box[0]) for box in boxes)
        class_image_counts.update(seen_classes)

    missing_expected_labels = sorted(
        name for name, row in manifest.items()
        if row.get("has_label") == "True" and name not in labels
    )
    manifest_missing_images = sorted(name for name in manifest if name not in images)
    unlabelled_images = sorted(name for name in images if name not in labels)
    exact_sha_groups: dict[str, list[str]] = defaultdict(list)
    for name, row in manifest.items():
        if name in images and row.get("sha256"):
            exact_sha_groups[row["sha256"]].append(name)
    exact_duplicates = [names for names in exact_sha_groups.values() if len(names) > 1]

    near_edges: list[dict[str, object]] = []
    near_members: list[dict[str, object]] = []
    decode_failures: list[dict[str, str]] = []
    run_near_duplicates = args.near_duplicates and not args.skip_near_duplicates
    if run_near_duplicates:
        print(f"开始 dHash 视觉近重复候选检查：{len(images)} 图", flush=True)
        near_edges, near_members, decode_failures = find_near_duplicates(
            images, args.near_duplicate_hamming, args.max_near_duplicate_pairs,
        )

    issues_by_image: dict[str, list[str]] = defaultdict(list)
    for issue in issues:
        issues_by_image[issue.image_name].append(issue.issue)
    near_by_image = {row["image_name"] for row in near_members}
    worst_scores = read_worst_scores(args.worst_images) if args.include_model_disagreement else {}
    box_counts = [len(boxes_by_image.get(name, [])) for name in labels if name in images]
    high_density_cutoff = percentile(box_counts, 0.95)

    priority_rows: list[dict[str, object]] = []
    for name, path in images.items():
        row = manifest.get(name, {})
        boxes = boxes_by_image.get(name, [])
        class_ids = sorted({int(box[0]) for box in boxes})
        reasons: list[str] = []
        score = 0.0
        if name in unlabelled_images:
            score += 35
            reasons.append("无检测标签，需人工确认缺陷或负样本")
        if name in issues_by_image:
            score += 100
            reasons.append("结构问题:" + ",".join(sorted(set(issues_by_image[name]))))
        if high_density_cutoff and len(boxes) >= high_density_cutoff:
            score += 15
            reasons.append(f"高密度标注({len(boxes)}框，P95={high_density_cutoff})")
        rare = [class_id for class_id in class_ids if class_image_counts[class_id] <= 10]
        if rare:
            score += 30 + 5 * len(rare)
            reasons.append("含稀有类:" + ",".join(CLASSES_18[class_id] for class_id in rare))
        if name in near_by_image:
            score += 12
            reasons.append("视觉近重复候选，划分前需分组")
        disagreement = worst_scores.get(name, 0.0)
        if disagreement:
            score += min(disagreement, 200.0) / 4
            reasons.append(f"旧模型分歧分数={disagreement:.1f}（仅作审查排序）")
        priority_rows.append({
            "union_name": name,
            "image_path": str(path),
            "source": row.get("source", ""),
            "source_split": row.get("source_split", ""),
            "has_original_label": str(name in labels),
            "valid_box_count": len(boxes),
            "classes": ";".join(CLASSES_18[class_id] for class_id in class_ids),
            "priority_score": round(score, 3),
            "reasons": " | ".join(reasons) or "常规人工复核",
        })
    priority_rows.sort(key=lambda row: (-float(row["priority_score"]), row["union_name"]))
    for rank, row in enumerate(priority_rows, start=1):
        row["priority_rank"] = rank

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "union": str(union),
        "images": len(images),
        "manifest_rows": len(manifest),
        "labels": len(labels),
        "unlabelled_images": len(unlabelled_images),
        "missing_expected_labels": len(missing_expected_labels),
        "orphan_labels": sum(issue.issue == "orphan_label_file" for issue in issues),
        "empty_labels": len(empty_labels),
        "invalid_label_issues": len(issues) - sum(issue.issue == "orphan_label_file" for issue in issues),
        "duplicate_boxes": duplicate_box_count,
        "manifest_missing_images": len(manifest_missing_images),
        "exact_duplicate_groups": len(exact_duplicates),
        "near_duplicate_edges": len(near_edges),
        "near_duplicate_members": len(near_members),
        "image_decode_failures": len(decode_failures),
        "near_duplicate_hamming": args.near_duplicate_hamming,
        "near_duplicate_skipped": not run_near_duplicates,
        "model_disagreement_used_for_priority": args.include_model_disagreement,
        "class_box_counts": {CLASSES_18[class_id]: class_box_counts[class_id] for class_id in range(len(CLASSES_18))},
        "class_image_counts": {CLASSES_18[class_id]: class_image_counts[class_id] for class_id in range(len(CLASSES_18))},
        "note": "近重复仅为 dHash 视觉候选，不足以证明同一物理镍板；默认人工重标排序不使用旧模型分歧。若显式启用，该分歧只用于抽样而非标签裁决。",
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("[dry-run] 未写入报告")
        return 0

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(
        output / "label_issues.csv",
        [issue.__dict__ for issue in issues],
        ["image_name", "label_file", "line_number", "issue", "detail"],
    )
    write_csv(
        output / "review_priority.csv",
        priority_rows,
        ["priority_rank", "union_name", "image_path", "source", "source_split", "has_original_label",
         "valid_box_count", "classes", "priority_score", "reasons"],
    )
    write_csv(output / "near_duplicate_edges.csv", near_edges, ["image_a", "image_b", "dhash_distance"])
    write_csv(output / "near_duplicate_clusters.csv", near_members, ["cluster_id", "image_name", "cluster_size"])
    write_csv(output / "image_decode_failures.csv", decode_failures, ["image_name", "image_path", "error"])
    (output / "missing_expected_labels.txt").write_text("\n".join(missing_expected_labels), encoding="utf-8")
    (output / "unlabelled_images.txt").write_text("\n".join(unlabelled_images), encoding="utf-8")
    (output / "empty_labels.txt").write_text("\n".join(empty_labels), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# L1 原始标签质量审计",
        "",
        f"- 图像：{summary['images']}；原始标签文件：{summary['labels']}；无检测标签图：{summary['unlabelled_images']}。",
        f"- 缺失预期标签：{summary['missing_expected_labels']}；孤立标签：{summary['orphan_labels']}；空标签：{summary['empty_labels']}。",
        f"- 非法/异常标签项：{summary['invalid_label_issues']}；重复框：{summary['duplicate_boxes']}。",
        f"- 完全重复组：{summary['exact_duplicate_groups']}；dHash 近重复边：{summary['near_duplicate_edges']}；候选成员：{summary['near_duplicate_members']}。",
        "",
        "## 使用边界",
        "",
        "- `review_priority.csv` 只用于确定人工复核顺序；默认不使用旧模型分歧。显式启用时，该分歧也不是标签正确性的证据。",
        "- `near_duplicate_clusters.csv` 是视觉候选，需结合物理镍板/采集批次人工确认，不能直接当作分组真值。未显式传入 `--near-duplicates` 时，本次未运行此项。",
        "- 无标签图不等于无缺陷；只有人工逐图确认后才可以导出为空标签。",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"产物: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
