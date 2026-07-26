#!/usr/bin/env python3
"""Analyze folder-grade datasets using cached ryw.pt detections.

The script is read-only for source datasets. It joins source paths to the
versioned flat pseudo-label manifest, calculates interpretable bounding-box
coverage, and reproduces the project's electrolytic-nickel hardcoded grading
ratios without rerunning model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

A_LABELS = {0, 1}
B_LABELS = {2, 4, 5, 6, 10, 12, 13}
C_LABELS = {7, 8, 11}
AIRHOLE_LABELS = {4}
NODULE_LABEL = 3

TYPE_NAMES = {
    0: "无板",
    1: "3号板",
    2: "烫洗板",
    3: "合格整版",
    4: "合格600",
    5: "合格不足600",
    6: "2号板",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用已缓存的 ryw.pt 输出分析电积镍分类等级特征和电解镍硬编码规则结果",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cached-root",
        type=Path,
        default=Path(
            r"E:/PythonProject/优化/pseudo_label_datasets/"
            "dianjinie_20260713_131840_v1_dianjienie+classify_4types_dianji+enterprise_dianji_0309"
        ),
        help="包含 source_manifest.csv、images/ 和 labels_with_conf/ 的 ryw.pt 伪标签集",
    )
    parser.add_argument(
        "--classify-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/classify_4types_dianji"),
    )
    parser.add_argument(
        "--enterprise-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/enterprise_dianji_0309_classification"),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi"),
        help="用于调用原项目 preprocess.findcent",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/grade_feature_analysis"),
    )
    parser.add_argument("--conf-thres", type=float, default=0.25, help="主分析和硬编码规则使用的置信度阈值")
    parser.add_argument("--high-conf-thres", type=float, default=0.70, help="高置信度敏感性统计阈值")
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def union_area(rectangles: Iterable[tuple[float, float, float, float]]) -> float:
    rects = [(x1, y1, x2, y2) for x1, y1, x2, y2 in rectangles if x2 > x1 and y2 > y1]
    if not rects:
        return 0.0
    xs = sorted({value for rect in rects for value in (rect[0], rect[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (y1, y2) for x1, y1, x2, y2 in rects if x1 < right and x2 > left
        )
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                covered += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        covered += end - start
        total += (right - left) * covered
    return total


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_findcent(project_root: Path) -> Any:
    project_root = project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from preprocess import findcent

    return findcent


def load_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def resolve_manifest_outputs(rows: list[dict[str, str]]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for row in rows:
        if row["output_image"]:
            outputs[row["sha256"]] = Path(row["output_image"])
    return outputs


def extract_grade(relative_path: Path, dataset_name: str) -> tuple[str, str]:
    parts = relative_path.parts
    if dataset_name == "classify_4types_dianji":
        if len(parts) < 3 or parts[0] not in {"train", "valid"}:
            raise ValueError(f"无法解析原四分类路径: {relative_path}")
        return parts[0], parts[1]
    if len(parts) < 3 or parts[0] != "images":
        raise ValueError(f"无法解析企业分类路径: {relative_path}")
    return "all", parts[1]


def read_detections(label_path: Path, conf_thres: float) -> list[dict[str, float | int]]:
    detections: list[dict[str, float | int]] = []
    if not label_path.exists():
        return detections
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 6:
            raise ValueError(f"{label_path}:{line_number} 应为六列，实际 {len(fields)} 列")
        class_id = int(fields[0])
        cx, cy, width, height, confidence = map(float, fields[1:])
        if confidence < conf_thres:
            continue
        detections.append(
            {
                "class_id": class_id,
                "cx": cx,
                "cy": cy,
                "w": width,
                "h": height,
                "confidence": confidence,
            }
        )
    return detections


def normalized_rect(det: dict[str, float | int]) -> tuple[float, float, float, float]:
    cx = float(det["cx"])
    cy = float(det["cy"])
    width = float(det["w"])
    height = float(det["h"])
    return (
        max(0.0, cx - width / 2),
        max(0.0, cy - height / 2),
        min(1.0, cx + width / 2),
        min(1.0, cy + height / 2),
    )


def hardcoded_rect(
    det: dict[str, float | int], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    cx = float(det["cx"]) * image_width
    cy = float(det["cy"]) * image_height
    width = float(det["w"]) * image_width
    height = float(det["h"]) * image_height
    x1 = max(0, int((cx - width / 2) / 4))
    y1 = max(0, int((cy - height / 2) / 4))
    x2 = min(4069, int((cx + width / 2) / 4))
    y2 = min(3554, int((cy + height / 2) / 4))
    return x1, y1, x2, y2


def hardcoded_features(
    detections: list[dict[str, float | int]], image_width: int, image_height: int, plate_area: float
) -> dict[str, Any]:
    filtered: list[dict[str, float | int]] = []
    has_nodule = False
    for det in detections:
        class_id = int(det["class_id"])
        if class_id == NODULE_LABEL:
            width = float(det["w"]) * image_width
            height = float(det["h"]) * image_height
            aspect = max(width / height, height / width) if min(width, height) > 0 else 1.0
            if width * height < 13000.0 and aspect < 3.0:
                continue
            has_nodule = True
        filtered.append(det)

    def group_ratio(labels: set[int]) -> float:
        rectangles = [
            hardcoded_rect(det, image_width, image_height)
            for det in filtered
            if int(det["class_id"]) in labels
        ]
        return union_area(rectangles) / plate_area if plate_area > 0 else 0.0

    ratios = {
        "A_ratio": group_ratio(A_LABELS),
        "B_ratio": group_ratio(B_LABELS),
        "C_ratio": group_ratio(C_LABELS),
        "Airhole_ratio": group_ratio(AIRHOLE_LABELS),
    }
    return {"has_nodule": has_nodule, **ratios}


def classify_hardcoded(features: dict[str, Any], assume_resnet_good: bool) -> int:
    has_nodule = bool(features["has_nodule"])
    a_ratio = float(features["A_ratio"])
    b_ratio = float(features["B_ratio"])
    c_ratio = float(features["C_ratio"])
    airhole_ratio = float(features["Airhole_ratio"])
    if not has_nodule and a_ratio < 0.2 and b_ratio < 0.1 and airhole_ratio < 0.01 and c_ratio < 0.03:
        return 3
    if has_nodule and assume_resnet_good:
        return 1
    if not has_nodule and a_ratio >= 0.2 and b_ratio < 0.05 and c_ratio < 0.05:
        return 2
    if c_ratio > 0.5:
        return 1
    if b_ratio > 0.5 and c_ratio < 0.5:
        return 6
    return 4


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        path.write_text("", encoding="utf-8-sig")
        return
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_grades(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[(sample["dataset"], sample["original_grade"])].append(sample)
    rows: list[dict[str, Any]] = []
    for (dataset, grade), group in sorted(groups.items()):
        counts = [float(row["detection_count"]) for row in group]
        coverages = [float(row["total_bbox_union_image_ratio"]) for row in group]
        rows.append(
            {
                "dataset": dataset,
                "original_grade": grade,
                "images": len(group),
                "unique_sha256": len({row["sha256"] for row in group}),
                "empty_detection_images": sum(int(row["detection_count"]) == 0 for row in group),
                "mean_boxes": mean(counts),
                "median_boxes": median(counts),
                "p90_boxes": percentile(counts, 90),
                "mean_total_bbox_union_image_ratio": mean(coverages),
                "median_total_bbox_union_image_ratio": median(coverages),
                "p90_total_bbox_union_image_ratio": percentile(coverages, 90),
                "mean_A_ratio_hardcoded": mean([float(row["A_ratio"]) for row in group]),
                "mean_B_ratio_hardcoded": mean([float(row["B_ratio"]) for row in group]),
                "mean_C_ratio_hardcoded": mean([float(row["C_ratio"]) for row in group]),
                "mean_Airhole_ratio_hardcoded": mean([float(row["Airhole_ratio"]) for row in group]),
                "nodule_images": sum(bool(row["has_nodule"]) for row in group),
            }
        )
    return rows


def summarize_classes(samples: list[dict[str, Any]], class_names: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[(sample["dataset"], sample["original_grade"])].append(sample)
    rows: list[dict[str, Any]] = []
    for (dataset, grade), group in sorted(groups.items()):
        for class_id, class_name in enumerate(class_names):
            counts = [int(sample[f"count__{class_name}"]) for sample in group]
            areas = [float(sample[f"area_ratio__{class_name}"]) for sample in group]
            detected_areas = [area for count, area in zip(counts, areas) if count > 0]
            detected_images = sum(count > 0 for count in counts)
            rows.append(
                {
                    "dataset": dataset,
                    "original_grade": grade,
                    "class_id": class_id,
                    "class_name": class_name,
                    "images": len(group),
                    "detected_images": detected_images,
                    "image_prevalence": detected_images / len(group),
                    "boxes": sum(counts),
                    "mean_boxes_per_image": mean([float(value) for value in counts]),
                    "mean_union_image_ratio_all": mean(areas),
                    "mean_union_image_ratio_detected": mean(detected_areas),
                    "median_union_image_ratio_detected": median(detected_areas),
                    "p90_union_image_ratio_detected": percentile(detected_areas, 90),
                    "images_area_ge_1pct": sum(area >= 0.01 for area in areas),
                    "images_area_ge_5pct": sum(area >= 0.05 for area in areas),
                    "images_area_ge_10pct": sum(area >= 0.10 for area in areas),
                    "images_area_ge_20pct": sum(area >= 0.20 for area in areas),
                    "images_area_ge_50pct": sum(area >= 0.50 for area in areas),
                }
            )
    return rows


def crosstab(samples: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for sample in samples:
        groups[(sample["dataset"], sample["original_grade"])][str(sample[field])] += 1
    type_columns = [TYPE_NAMES[type_id] for type_id in sorted(TYPE_NAMES)]
    rows: list[dict[str, Any]] = []
    for (dataset, grade), counter in sorted(groups.items()):
        total = sum(counter.values())
        row: dict[str, Any] = {"dataset": dataset, "original_grade": grade, "images": total}
        for type_name in type_columns:
            row[type_name] = counter[type_name]
            row[f"{type_name}_rate"] = counter[type_name] / total if total else 0.0
        rows.append(row)
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    result.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(result)


def build_report(
    samples: list[dict[str, Any]],
    grade_rows: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    hardcoded_rows: list[dict[str, Any]],
    hardcoded_good_rows: list[dict[str, Any]],
    audit: dict[str, Any],
    conf_thres: float,
    high_conf_thres: float,
) -> str:
    enterprise_samples = [
        sample for sample in samples if sample["dataset"] == "enterprise_dianji_0309_classification"
    ]

    def enterprise_rule_count(grade: str, predicate: Any) -> tuple[int, int]:
        group = [sample for sample in enterprise_samples if sample["original_grade"] == grade]
        return sum(bool(predicate(sample)) for sample in group), len(group)

    field_rules = [
        ("检出 overfire", lambda row: int(row["count__overfire"]) > 0),
        ("检出 suanwuran", lambda row: int(row["count__suanwuran"]) > 0),
        ("airhole 框并集 >= 15%", lambda row: float(row["area_ratio__airhole"]) >= 0.15),
        ("bubble 框并集 >= 10%", lambda row: float(row["area_ratio__bubble"]) >= 0.10),
        (
            "overfire/酸污染/大气孔任一",
            lambda row: int(row["count__overfire"]) > 0
            or int(row["count__suanwuran"]) > 0
            or float(row["area_ratio__airhole"]) >= 0.15,
        ),
    ]
    lines = [
        "# 电积镍等级特征与电解镍硬编码规则分析",
        "",
        "## 口径",
        "",
        f"- 检测来源：版本化缓存 `ryw.pt` 六列标签；主阈值 `conf >= {conf_thres:.2f}`，高置信敏感性阈值 `{high_conf_thres:.2f}`。",
        "- `检测框面积占比` 是同类框并集 / 整图面积，只代表检测框覆盖，不等于分割后的真实缺陷面积。",
        "- `A/B/C/Airhole_ratio` 严格复现项目代码：原图检测坐标除以 4 后求并集，再除以 `findcent` 得到的 1024x888 板面面积。",
        "- 纯检测硬编码不使用 `ryw.pth`；另给出“假设 ResNet=good”时结瘤分支的敏感性结果。",
        "",
        "## 数据审计",
        "",
        markdown_table(
            ["数据集", "路径数", "唯一内容", "跨等级重复内容", "损坏/不可读", "标签缺失"],
            [
                [
                    name,
                    str(info["image_paths"]),
                    str(info["unique_sha256"]),
                    str(info["cross_grade_duplicate_sha256"]),
                    str(info["unreadable_images"]),
                    str(info["missing_cached_labels"]),
                ]
                for name, info in audit["datasets"].items()
            ],
        ),
        "",
        "## 总体特征",
        "",
        markdown_table(
            ["数据集", "原等级", "图片", "平均框数", "框并集均值", "框并集P90", "A均值", "B均值", "C均值", "气孔均值"],
            [
                [
                    row["dataset"],
                    row["original_grade"],
                    str(row["images"]),
                    f"{row['mean_boxes']:.2f}",
                    format_pct(row["mean_total_bbox_union_image_ratio"]),
                    format_pct(row["p90_total_bbox_union_image_ratio"]),
                    format_pct(row["mean_A_ratio_hardcoded"]),
                    format_pct(row["mean_B_ratio_hardcoded"]),
                    format_pct(row["mean_C_ratio_hardcoded"]),
                    format_pct(row["mean_Airhole_ratio_hardcoded"]),
                ]
                for row in grade_rows
            ],
        ),
        "",
        "## 重点缺陷",
        "",
    ]
    focus_classes = {"overfire", "airhole", "suanwuran", "nodule", "wuzi", "waterstain", "watermark"}
    focus_rows = [row for row in class_rows if row["class_name"] in focus_classes]
    lines.append(
        markdown_table(
            ["数据集", "原等级", "缺陷", "出现率", "检出时平均框面积占图", ">=5%图片", ">=20%图片"],
            [
                [
                    row["dataset"],
                    row["original_grade"],
                    row["class_name"],
                    format_pct(row["image_prevalence"]),
                    format_pct(row["mean_union_image_ratio_detected"]),
                    str(row["images_area_ge_5pct"]),
                    str(row["images_area_ge_20pct"]),
                ]
                for row in focus_rows
                if row["detected_images"] > 0
            ],
        )
    )
    lines.extend(["", "## 纯检测硬编码结果", ""])
    lines.append(
        markdown_table(
            ["数据集", "原等级", "图片", "合格整版", "合格600", "烫洗板", "3号板", "2号板"],
            [
                [
                    row["dataset"],
                    row["original_grade"],
                    str(row["images"]),
                    str(row["合格整版"]),
                    str(row["合格600"]),
                    str(row["烫洗板"]),
                    str(row["3号板"]),
                    str(row["2号板"]),
                ]
                for row in hardcoded_rows
            ],
        )
    )
    lines.extend(["", "## 假设 ResNet=good 的结瘤敏感性", ""])
    lines.append(
        markdown_table(
            ["数据集", "原等级", "图片", "合格整版", "合格600", "3号板"],
            [
                [
                    row["dataset"],
                    row["original_grade"],
                    str(row["images"]),
                    str(row["合格整版"]),
                    str(row["合格600"]),
                    str(row["3号板"]),
                ]
                for row in hardcoded_good_rows
            ],
        )
    )
    lines.extend(["", "## 现场候选规则对照", ""])
    lines.append(
        markdown_table(
            ["候选触发条件", "企业 3# 命中", "企业 9950 命中", "结论"],
            [
                [
                    rule_name,
                    f"{enterprise_rule_count('3#', predicate)[0]}/{enterprise_rule_count('3#', predicate)[1]}",
                    f"{enterprise_rule_count('9950', predicate)[0]}/{enterprise_rule_count('9950', predicate)[1]}",
                    "不能单独区分" if rule_name != "bubble 框并集 >= 10%" else "可作严重度辅助，不能当牌号规则",
                ]
                for rule_name, predicate in field_rules
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 判定建议",
            "",
            "- 若现场正式确认“检出烧板即最高只能到 3#”，应新增一个优先级高于原七状态规则的业务覆盖项；当前七状态代码没有这条规则。",
            "- 该覆盖项上线前必须人工复核被 ryw.pt 判为 overfire 的 9950 图片，因为本批企业 9950 的 overfire 检出率不低于 3#，直接使用会产生明显等级冲突。",
            "- 9950/9996 应继续由 LIMS/MES/批次牌号给出；图像端建议只输出 appearance_status、缺陷清单、面积和 manual_review，不应由外观反推化学牌号。",
            "- 企业 3# 的主要可见严重度信号是大面积 bubble、较高气孔出现率和较高 B_ratio，可用于人工复核排序，但不同来源数据的分布差异很大，暂不能固定为统一阈值。",
            "",
            "## 规则边界",
            "",
            "- C 组为 `fawu/overfire/suanwuran`，项目只有 C 组面积比严格大于 50% 才直接输出 3号板；“检测到烧板即 3#”并不是当前硬编码。",
            "- `NiOH/youpao/heibian/mijixiaokeng/fanxitong` 未进入 A/B/C 面积规则；`nodule` 仅通过独立结瘤分支参与判定。",
            "- `9950/9996` 是原目录业务/化学牌号，七状态硬编码不能直接还原这两个牌号。",
            "- 面积来自矩形框，不是缺陷掩膜；大面积气孔或酸污染的真实面积需实例分割或人工标定复核。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cached_root = args.cached_root.resolve()
    metadata = json.loads((cached_root / "dataset_metadata.json").read_text(encoding="utf-8-sig"))
    if metadata.get("teacher_model") != "ryw.pt":
        raise ValueError(f"缓存教师模型不是 ryw.pt: {metadata.get('teacher_model')}")
    cached_candidate_threshold = float(metadata.get("confidence_candidate_threshold", 1.0))
    if args.conf_thres < cached_candidate_threshold:
        raise ValueError(
            f"conf-thres={args.conf_thres} 低于缓存候选阈值 {cached_candidate_threshold}，无法完整分析"
        )

    class_names = load_classes(cached_root / "classes.txt")
    manifest_rows = load_manifest(cached_root / "source_manifest.csv")
    sha_to_output = resolve_manifest_outputs(manifest_rows)
    findcent = load_findcent(args.project_root)

    dataset_specs = [
        ("classify_4types_dianji", args.classify_root.resolve()),
        ("enterprise_dianji_0309_classification", args.enterprise_root.resolve()),
    ]
    root_by_name = dict(dataset_specs)
    path_to_manifest: dict[str, dict[str, str]] = {
        str(Path(row["source_path"]).resolve()).casefold(): row for row in manifest_rows
    }

    source_samples: list[tuple[str, Path, str, str, dict[str, str]]] = []
    for dataset_name, root in dataset_specs:
        if not root.exists():
            raise FileNotFoundError(root)
        for image_path in sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
            relative = image_path.relative_to(root)
            split, grade = extract_grade(relative, dataset_name)
            manifest_row = path_to_manifest.get(str(image_path.resolve()).casefold())
            if manifest_row is None:
                manifest_row = {
                    "source_path": str(image_path),
                    "sha256": "",
                    "status": "missing_manifest",
                    "output_image": "",
                }
            source_samples.append((dataset_name, image_path, split, grade, manifest_row))

    sha_grade_map: dict[str, set[tuple[str, str]]] = defaultdict(set)
    sha_counts: Counter[str] = Counter()
    for dataset_name, _, _, grade, manifest_row in source_samples:
        digest = manifest_row.get("sha256", "")
        if digest:
            sha_grade_map[digest].add((dataset_name, grade))
            sha_counts[digest] += 1

    samples: list[dict[str, Any]] = []
    audit_by_dataset: dict[str, dict[str, Any]] = {
        name: {
            "root": str(root),
            "image_paths": 0,
            "unique_sha256": 0,
            "duplicate_paths": 0,
            "cross_grade_duplicate_sha256": 0,
            "unreadable_images": 0,
            "missing_manifest_rows": 0,
            "missing_cached_labels": 0,
            "plate_area_fallbacks": 0,
            "grade_distribution": Counter(),
        }
        for name, root in dataset_specs
    }

    for index, (dataset_name, image_path, split, grade, manifest_row) in enumerate(source_samples, 1):
        audit = audit_by_dataset[dataset_name]
        audit["image_paths"] += 1
        audit["grade_distribution"][grade] += 1
        digest = manifest_row.get("sha256", "")
        if not digest:
            audit["missing_manifest_rows"] += 1
        output_image = sha_to_output.get(digest)
        label_path = cached_root / "labels_with_conf" / f"{output_image.stem}.txt" if output_image else None
        if label_path is None or not label_path.exists():
            audit["missing_cached_labels"] += 1

        image = read_image(image_path)
        if image is None:
            audit["unreadable_images"] += 1
            image_height, image_width = 888, 1024
            plate_area = float(image_width * image_height)
            plate_fallback = True
        else:
            image_height, image_width = image.shape[:2]
            resized = cv2.resize(image, (1024, 888))
            try:
                _, plate_width, plate_height, _ = findcent(resized)
                plate_area = float(plate_width * plate_height)
                if plate_area <= 0:
                    raise ValueError("plate_area <= 0")
                plate_fallback = False
            except Exception:
                plate_area = float(1024 * 888)
                plate_fallback = True
                audit["plate_area_fallbacks"] += 1

        detections = read_detections(label_path, args.conf_thres) if label_path else []
        high_detections = read_detections(label_path, args.high_conf_thres) if label_path else []
        per_class_rects: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
        per_class_counts: Counter[int] = Counter()
        per_class_max_conf: defaultdict[int, float] = defaultdict(float)
        all_rects: list[tuple[float, float, float, float]] = []
        sum_box_area_ratio = 0.0
        for det in detections:
            class_id = int(det["class_id"])
            rect = normalized_rect(det)
            per_class_rects[class_id].append(rect)
            per_class_counts[class_id] += 1
            per_class_max_conf[class_id] = max(per_class_max_conf[class_id], float(det["confidence"]))
            all_rects.append(rect)
            sum_box_area_ratio += max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])

        hardcoded = hardcoded_features(detections, image_width, image_height, plate_area)
        type_detection_only = classify_hardcoded(hardcoded, assume_resnet_good=False)
        type_assume_good = classify_hardcoded(hardcoded, assume_resnet_good=True)
        row: dict[str, Any] = {
            "dataset": dataset_name,
            "split": split,
            "original_grade": grade,
            "source_path": str(image_path),
            "sha256": digest,
            "same_content_path_count": sha_counts[digest] if digest else 0,
            "cross_grade_duplicate": len(sha_grade_map[digest]) > 1 if digest else False,
            "cached_image": str(output_image) if output_image else "",
            "cached_label": str(label_path) if label_path else "",
            "image_width": image_width,
            "image_height": image_height,
            "plate_area_1024x888": plate_area,
            "plate_area_fallback": plate_fallback,
            "detection_count": len(detections),
            "high_conf_detection_count": len(high_detections),
            "max_confidence": max((float(det["confidence"]) for det in detections), default=0.0),
            "sum_bbox_area_ratio": sum_box_area_ratio,
            "total_bbox_union_image_ratio": union_area(all_rects),
            **hardcoded,
            "hardcoded_type_detection_only": type_detection_only,
            "hardcoded_name_detection_only": TYPE_NAMES[type_detection_only],
            "hardcoded_type_assume_resnet_good": type_assume_good,
            "hardcoded_name_assume_resnet_good": TYPE_NAMES[type_assume_good],
        }
        for class_id, class_name in enumerate(class_names):
            row[f"count__{class_name}"] = per_class_counts[class_id]
            row[f"area_ratio__{class_name}"] = union_area(per_class_rects[class_id])
            row[f"max_conf__{class_name}"] = per_class_max_conf[class_id]
        samples.append(row)
        if index % 100 == 0 or index == len(source_samples):
            print(f"已分析 {index}/{len(source_samples)}")

    for dataset_name, _ in dataset_specs:
        dataset_samples = [sample for sample in samples if sample["dataset"] == dataset_name]
        audit = audit_by_dataset[dataset_name]
        digests = {sample["sha256"] for sample in dataset_samples if sample["sha256"]}
        audit["unique_sha256"] = len(digests)
        audit["duplicate_paths"] = len(dataset_samples) - len(digests)
        audit["cross_grade_duplicate_sha256"] = sum(
            len({grade for ds, grade in sha_grade_map[digest] if ds == dataset_name}) > 1 for digest in digests
        )
        audit["grade_distribution"] = dict(audit["grade_distribution"])

    grade_rows = summarize_grades(samples)
    class_rows = summarize_classes(samples, class_names)
    hardcoded_rows = crosstab(samples, "hardcoded_name_detection_only")
    hardcoded_good_rows = crosstab(samples, "hardcoded_name_assume_resnet_good")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root.resolve() / f"dianji_ryw_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "samples.csv", samples)
    write_csv(output_dir / "grade_summary.csv", grade_rows)
    write_csv(output_dir / "class_features.csv", class_rows)
    write_csv(output_dir / "hardcoded_crosstab_detection_only.csv", hardcoded_rows)
    write_csv(output_dir / "hardcoded_crosstab_assume_resnet_good.csv", hardcoded_good_rows)

    ignored_by_area_rule = [
        {"class_id": class_id, "class_name": name}
        for class_id, name in enumerate(class_names)
        if class_id not in A_LABELS | B_LABELS | C_LABELS | {NODULE_LABEL}
    ]
    audit_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cached_root": str(cached_root),
        "teacher_model": metadata.get("teacher_model"),
        "cached_candidate_threshold": cached_candidate_threshold,
        "analysis_conf_threshold": args.conf_thres,
        "high_conf_threshold": args.high_conf_thres,
        "datasets": audit_by_dataset,
        "ignored_by_hardcoded_area_rule": ignored_by_area_rule,
    }
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "output_dir": str(output_dir),
        "samples": len(samples),
        "grade_summary": grade_rows,
        "hardcoded_detection_only": hardcoded_rows,
        "hardcoded_assume_resnet_good": hardcoded_good_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(
        samples,
        grade_rows,
        class_rows,
        hardcoded_rows,
        hardcoded_good_rows,
        audit_payload,
        args.conf_thres,
        args.high_conf_thres,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
