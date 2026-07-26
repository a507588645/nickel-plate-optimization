#!/usr/bin/env python3
"""
电积镍等级 v2：合并已核实标签数据集、去重、v2a 特征 join、v2b 高精度重推理。

产物仅写入 dianji_grade_v2/，不修改原始数据集与 dianji_grade_rule_v1/。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CLASSIFY = Path(r"E:/PythonProject/datasets/classify_4types_dianji")
DEFAULT_ENTERPRISE = Path(r"E:/PythonProject/datasets/enterprise_dianji_0309_classification")
DEFAULT_SAMPLES_CSV = ROOT / "grade_feature_analysis" / "combined_all_pseudo_v2_grade" / "samples.csv"
DEFAULT_OUT = ROOT / "dianji_grade_v2"
DEFAULT_WEIGHTS = Path(r"E:/PythonProject/Tools/ModelAutoLabelTool/weights/yolov5/ryw.pt")
DEFAULT_YOLOV5_REPO = Path(r"E:/PythonProject/Tools/ModelAutoLabelTool/vendor/yolov5")
DEFAULT_PROJECT_ROOT = Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

CLASS_NAMES: list[str] = [
    "wuzi",
    "waterstain",
    "yellowstain",
    "nodule",
    "airhole",
    "bubble",
    "bump",
    "fawu",
    "overfire",
    "NiOH",
    "greenstain",
    "suanwuran",
    "quejiao",
    "watermark",
    "youpao",
    "heibian",
    "mijixiaokeng",
    "fanxitong",
]

A_LABELS = {0, 1}
B_LABELS = {2, 4, 5, 6, 10, 12, 13}
C_LABELS = {7, 8, 11}
AIRHOLE_LABELS = {4}
NODULE_LABEL = 3

FEATURE_COLUMNS: list[str] = (
    [
        "A_ratio",
        "B_ratio",
        "C_ratio",
        "Airhole_ratio",
        "has_nodule",
        "total_bbox_union_image_ratio",
        "detection_count",
    ]
    + [f"count__{n}" for n in CLASS_NAMES]
    + [f"area_ratio__{n}" for n in CLASS_NAMES]
)

GRADES = ("3#", "9950", "9996", "bad")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="构建电积镍等级 v2 合并清单、v2a/v2b 特征（未经企业确认）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--classify-root", type=Path, default=DEFAULT_CLASSIFY)
    p.add_argument("--enterprise-root", type=Path, default=DEFAULT_ENTERPRISE)
    p.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--yolov5-repo", type=Path, default=DEFAULT_YOLOV5_REPO)
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--img-size", type=int, default=1536, help="v2b 推理尺寸")
    p.add_argument("--conf-thres", type=float, default=0.25, help="检测器内部 conf")
    p.add_argument("--iou-thres", type=float, default=0.45)
    p.add_argument(
        "--feature-conf-thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.40],
        help="从逐框结果计算特征时的 conf 阈值列表",
    )
    p.add_argument(
        "--phash-threshold",
        type=int,
        default=10,
        help="近重复 dHash hamming 距离阈值（含）；默认 10/256bit≈3.9%%",
    )
    p.add_argument(
        "--phash-size",
        type=int,
        default=16,
        help="dHash 高度；宽度=size+1，bit 数=size*size（默认 256）",
    )
    p.add_argument(
        "--near-dup-same-grade-only",
        action="store_true",
        default=True,
        help="近重复仅在同 grade 内比较（默认开启，降低跨等级误并）",
    )
    p.add_argument(
        "--near-dup-all-grades",
        action="store_true",
        help="近重复跨 grade 全比较（覆盖 --near-dup-same-grade-only）",
    )
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|0|1|...")
    p.add_argument("--batch-log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="只扫描与去重，不写特征/不推理")
    p.add_argument("--skip-v2a", action="store_true", help="跳过 samples.csv join")
    p.add_argument("--skip-v2b", action="store_true", help="跳过 1536 重推理")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前 N 张（调试）；0 表示全部",
    )
    p.add_argument(
        "--resume-v2b",
        action="store_true",
        help="v2b 时跳过已有逐框 CSV 的图片",
    )
    return p.parse_args(argv)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_image_bgr(path: Path) -> np.ndarray | None:
    import cv2

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def difference_hash_bits(path: Path, hash_size: int = 16) -> int | None:
    """
    纯 PIL/numpy 差分哈希（dHash）。
    比 aHash 对工业板面整体亮度更不敏感；hash_size=16 → 16*15=240 bit。
    """
    from PIL import Image

    try:
        with Image.open(path) as im:
            # 宽=hash_size+1，高=hash_size，比较相邻列
            im = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.int16)
    except OSError:
        return None
    # 差分：pixel[x] > pixel[x+1]
    diff = arr[:, 1:] > arr[:, :-1]
    # pack bits
    flat = diff.reshape(-1)
    value = 0
    for i, bit in enumerate(flat.tolist()):
        if bit:
            value |= 1 << i
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def union_area(rectangles: Iterable[tuple[float, float, float, float]]) -> float:
    rects = [(x1, y1, x2, y2) for x1, y1, x2, y2 in rectangles if x2 > x1 and y2 > y1]
    if not rects:
        return 0.0
    xs = sorted({v for r in rects for v in (r[0], r[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted((y1, y2) for x1, y1, x2, y2 in rects if x1 < right and x2 > left)
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for ns, ne in intervals[1:]:
            if ns > end:
                covered += end - start
                start, end = ns, ne
            else:
                end = max(end, ne)
        covered += end - start
        total += (right - left) * covered
    return total


def normalized_rect(det: dict[str, Any]) -> tuple[float, float, float, float]:
    cx, cy, w, h = float(det["cx"]), float(det["cy"]), float(det["w"]), float(det["h"])
    return (
        max(0.0, cx - w / 2),
        max(0.0, cy - h / 2),
        min(1.0, cx + w / 2),
        min(1.0, cy + h / 2),
    )


def hardcoded_rect(
    det: dict[str, Any], image_width: int, image_height: int
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


def compute_plate_area(image_bgr: np.ndarray, project_root: Path) -> tuple[float, bool]:
    """返回 (plate_area, is_fallback)。优先 findcent(1024x888)。"""
    import cv2

    h, w = image_bgr.shape[:2]
    fallback = float(w * h)
    try:
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from preprocess import findcent

        resized = cv2.resize(image_bgr, (1024, 888))
        _, pw, ph, _ = findcent(resized)
        area = float(pw * ph)
        if area <= 0:
            return fallback, True
        return area, False
    except Exception:
        return fallback, True


def features_from_detections(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    plate_area: float,
    conf_thres: float,
) -> dict[str, Any]:
    """与历史 samples.csv 同构的 43 列核心特征 + 元数据。"""
    kept = [d for d in detections if float(d.get("conf", d.get("confidence", 0))) >= conf_thres]

    # has_nodule：沿用电解镍硬编码过滤
    has_nodule = False
    for det in kept:
        if int(det["label"]) != NODULE_LABEL:
            continue
        width = float(det["w"]) * image_width
        height = float(det["h"]) * image_height
        aspect = max(width / height, height / width) if min(width, height) > 0 else 1.0
        if width * height >= 13000.0 or aspect >= 3.0:
            has_nodule = True

    def group_ratio(labels: set[int]) -> float:
        rects = [
            hardcoded_rect(det, image_width, image_height)
            for det in kept
            if int(det["label"]) in labels
        ]
        # 与 analyze_dianji 一致：并集面积 / plate_area（1024x888 坐标系缩放后）
        # hardcoded_rect 已 /4 到 4069x3554 画布尺度；原项目 scalculate 在该画布上计像素
        if plate_area <= 0:
            return 0.0
        # 将 /4 后的整数框转回与 plate_area 同尺度：plate_area 是 1024x888 下的宽*高
        # 原项目：canvas 在 4069x3554，sum 后 / plate_area；等价于 (union of /4 rects) / plate_area
        area = 0.0
        # 用 numpy 画布近似（小图）会 OOM；改用 union_area 浮点
        float_rects = [(float(a), float(b), float(c), float(d)) for a, b, c, d in rects]
        area = union_area(float_rects)
        return float(area / plate_area) if plate_area > 0 else 0.0

    all_norm = [normalized_rect(d) for d in kept]
    per_class_rects: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    per_class_counts: Counter[int] = Counter()
    for det in kept:
        cid = int(det["label"])
        per_class_counts[cid] += 1
        per_class_rects[cid].append(normalized_rect(det))

    out: dict[str, Any] = {
        "image_width": image_width,
        "image_height": image_height,
        "plate_area": plate_area,
        "detection_count": len(kept),
        "max_confidence": max((float(d.get("conf", d.get("confidence", 0))) for d in kept), default=0.0),
        "total_bbox_union_image_ratio": union_area(all_norm),
        "A_ratio": group_ratio(A_LABELS),
        "B_ratio": group_ratio(B_LABELS),
        "C_ratio": group_ratio(C_LABELS),
        "Airhole_ratio": group_ratio(AIRHOLE_LABELS),
        "has_nodule": int(has_nodule),
    }
    for i, name in enumerate(CLASS_NAMES):
        out[f"count__{name}"] = int(per_class_counts.get(i, 0))
        out[f"area_ratio__{name}"] = union_area(per_class_rects.get(i, []))
    return out


def scan_classify(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        raise FileNotFoundError(root)
    for split in ("train", "valid", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        split_name = "valid" if split == "val" else split
        for grade_dir in sorted(split_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            grade = grade_dir.name
            if grade not in GRADES:
                continue
            for path in sorted(grade_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    rows.append(
                        {
                            "image_name": path.name,
                            "image_stem": path.stem,
                            "image_path": str(path.resolve()),
                            "source": "classify_4types_dianji",
                            "grade": grade,
                            "split": split_name,
                        }
                    )
    return rows


def scan_enterprise(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    images_root = root / "images"
    if not images_root.exists():
        raise FileNotFoundError(images_root)
    for path in sorted(images_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        grade = path.parent.name
        rows.append(
            {
                "image_name": path.name,
                "image_stem": path.stem,
                "image_path": str(path.resolve()),
                "source": "enterprise_dianji_0309_classification",
                "grade": grade,
                "split": "enterprise_all",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys = fieldnames or []
    if not keys:
        keys = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def merge_and_dedup(
    classify_rows: list[dict[str, Any]],
    enterprise_rows: list[dict[str, Any]],
    phash_threshold: int,
    phash_size: int,
    same_grade_only: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows = classify_rows + enterprise_rows
    print(f"[扫描] classify={len(classify_rows)} enterprise={len(enterprise_rows)} total={len(all_rows)}")

    # 1) 完整哈希
    hash_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(all_rows):
        digest = sha256_file(Path(row["image_path"]))
        row["sha256"] = digest
        hash_to_indices[digest].append(i)

    exact_dup_groups = {h: idxs for h, idxs in hash_to_indices.items() if len(idxs) > 1}
    exact_dup_images = sum(len(v) for v in exact_dup_groups.values())
    # 每个完整重复组只保留一份：优先 classify 源，其次 train split
    keep_mask = [True] * len(all_rows)
    exact_removed = 0
    for h, idxs in exact_dup_groups.items():
        ranked = sorted(
            idxs,
            key=lambda i: (
                0 if all_rows[i]["source"] == "classify_4types_dianji" else 1,
                0 if all_rows[i]["split"] == "train" else 1,
                all_rows[i]["image_path"],
            ),
        )
        for j in ranked[1:]:
            keep_mask[j] = False
            exact_removed += 1

    kept = [r for r, k in zip(all_rows, keep_mask) if k]
    print(f"[完整重复] 组数={len(exact_dup_groups)} 涉及图片={exact_dup_images} 移除={exact_removed} 保留={len(kept)}")

    # 2) 近重复 dHash（在保留集上）
    n_bits = phash_size * phash_size  # dHash: H x Wdiff = size x size
    phashes: list[int | None] = []
    for i, row in enumerate(kept):
        ph = difference_hash_bits(Path(row["image_path"]), hash_size=phash_size)
        phashes.append(ph)
        if (i + 1) % 200 == 0:
            print(f"  dhash {i+1}/{len(kept)}")

    parent = list(range(len(kept)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    near_pairs = 0
    # 按 grade 分桶，避免不同等级因整体外观相似被错误连通
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(kept):
        if phashes[i] is None:
            continue
        key = row["grade"] if same_grade_only else "__all__"
        buckets[key].append(i)

    for _bucket, idxs in buckets.items():
        for a_i in range(len(idxs)):
            i = idxs[a_i]
            for b_i in range(a_i + 1, len(idxs)):
                j = idxs[b_i]
                dist = hamming(phashes[i], phashes[j])  # type: ignore[arg-type]
                if dist <= phash_threshold:
                    union(i, j)
                    near_pairs += 1

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(kept)):
        groups[find(i)].append(i)

    multi_groups = {k: v for k, v in groups.items() if len(v) > 1}
    root_to_gid: dict[int, str] = {}
    gid_counter = 0
    for root_id, members in sorted(groups.items(), key=lambda kv: min(kv[1])):
        if len(members) > 1:
            gid = f"near_g{gid_counter:04d}"
            gid_counter += 1
        else:
            gid = f"single_{members[0]:04d}"
        root_to_gid[root_id] = gid

    for i, row in enumerate(kept):
        row["dup_group_id"] = root_to_gid[find(i)]
        row["phash"] = f"{phashes[i]:0{max(1, (n_bits + 3) // 4)}x}" if phashes[i] is not None else ""
        row["kept_after_exact_dedup"] = 1

    multi_sizes = sorted((len(v) for v in multi_groups.values()), reverse=True)
    meta = {
        "classify_count": len(classify_rows),
        "enterprise_count": len(enterprise_rows),
        "raw_total": len(all_rows),
        "exact_duplicate_groups": len(exact_dup_groups),
        "exact_duplicate_images_involved": exact_dup_images,
        "exact_removed": exact_removed,
        "kept_after_exact_dedup": len(kept),
        "hash_method": "dHash",
        "phash_size": phash_size,
        "phash_bits": n_bits,
        "phash_threshold": phash_threshold,
        "near_dup_same_grade_only": same_grade_only,
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_groups": len(multi_groups),
        "near_duplicate_images_in_multi_groups": sum(len(v) for v in multi_groups.values()),
        "near_duplicate_group_size_top10": multi_sizes[:10],
        "grade_counts_kept": dict(Counter(r["grade"] for r in kept)),
        "source_counts_kept": dict(Counter(r["source"] for r in kept)),
        "split_counts_kept": dict(Counter(r["split"] for r in kept)),
        "note": (
            "完整重复(SHA-256)只留一份；近重复(dHash)不删，仅同一 dup_group_id 防泄漏。"
            f"默认同 grade 内比较，阈值 hamming≤{phash_threshold}/{n_bits}bit。"
        ),
    }
    print(
        f"[近重复] dHash 阈值<= {phash_threshold}/{n_bits}bit 对={near_pairs} "
        f"多成员组={len(multi_groups)} 涉及图={meta['near_duplicate_images_in_multi_groups']} "
        f"top sizes={multi_sizes[:5]}"
    )
    return kept, meta


def join_v2a(
    manifest: list[dict[str, Any]],
    samples_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    if not samples_csv.exists():
        raise FileNotFoundError(samples_csv)

    samples_by_name: dict[str, dict[str, str]] = {}
    samples_by_stem: dict[str, list[dict[str, str]]] = defaultdict(list)
    with samples_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("image_name", "")
            samples_by_name[name] = row
            samples_by_stem[Path(name).stem].append(row)

    joined: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for m in manifest:
        stem = m["image_stem"]
        grade = m["grade"]
        split = m["split"]
        source = m["source"]
        name = m["image_name"]
        hit: dict[str, str] | None = None

        if source == "classify_4types_dianji":
            # 历史命名：classify_4types_dianji_{split}_{grade}_{filename}
            candidates = [
                f"classify_4types_dianji_{split}_{grade}_{name}",
                f"classify_4types_dianji_train_{grade}_{name}",
                f"classify_4types_dianji_valid_{grade}_{name}",
            ]
            for c in candidates:
                if c in samples_by_name:
                    hit = samples_by_name[c]
                    break
            if hit is None:
                # 宽松：stem 出现在 combined 名中且含 classify_4types 与 grade
                for full_stem, cands in samples_by_stem.items():
                    if stem in full_stem and "classify_4types_dianji" in full_stem and grade in full_stem:
                        hit = cands[0]
                        break
        else:
            token = f"images_{grade}_{stem}"
            key_candidates = [
                f"enterprise_dianji_0309_classification_images_{grade}_{name}",
                f"enterprise_dianji_0309_classification_images_{grade}_{stem}.png",
            ]
            for c in key_candidates:
                if c in samples_by_name:
                    hit = samples_by_name[c]
                    break
            if hit is None:
                for full_stem, cands in samples_by_stem.items():
                    if token in full_stem or (full_stem.endswith(f"_{stem}") and "0309" in full_stem and grade in full_stem):
                        hit = cands[0]
                        break

        if hit is None:
            missing.append(
                {
                    "image_name": name,
                    "source": source,
                    "grade": grade,
                    "split": split,
                    "image_path": m["image_path"],
                }
            )
            continue

        out: dict[str, Any] = {
            "image_name": name,
            "image_stem": stem,
            "image_path": m["image_path"],
            "source": source,
            "grade": grade,
            "true_grade": grade,
            "split": split,
            "dup_group_id": m["dup_group_id"],
            "sha256": m.get("sha256", ""),
            "feature_version": "v2a_join_samples_csv",
            "joined_image_name": hit.get("image_name", ""),
            "joined_label_path": hit.get("label_path", ""),
        }
        for col in (
            "image_width",
            "image_height",
            "plate_area",
            "detection_count",
            "max_confidence",
            "total_bbox_union_image_ratio",
            "A_ratio",
            "B_ratio",
            "C_ratio",
            "Airhole_ratio",
            "has_nodule",
            "type_id_detection_only",
            "type_name_detection_only",
        ):
            out[col] = hit.get(col, "")
        for n in CLASS_NAMES:
            out[f"count__{n}"] = hit.get(f"count__{n}", "0")
            out[f"area_ratio__{n}"] = hit.get(f"area_ratio__{n}", "0")
        joined.append(out)

    meta = {
        "manifest_total": len(manifest),
        "joined": len(joined),
        "missing": len(missing),
        "coverage": len(joined) / len(manifest) if manifest else 0.0,
        "grade_counts_joined": dict(Counter(r["grade"] for r in joined)),
        "grade_counts_missing": dict(Counter(r["grade"] for r in missing)),
        "source_counts_joined": dict(Counter(r["source"] for r in joined)),
        "source_counts_missing": dict(Counter(r["source"] for r in missing)),
        "samples_csv": str(samples_csv.resolve()),
        "detection_params_note": "历史口径 ryw.pt conf=0.25 iou=0.45 imgsz=1024（combined_all_pseudo_v2）",
        "status": "unconfirmed_pending_enterprise",
    }
    return joined, meta, missing


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return requested


def load_yolov5(weights: Path, yolov5_repo: Path, device: str, conf: float, iou: float, imgsz: int):
    import torch

    # torch>=2.6 默认 weights_only=True，YOLOv5 权重需兼容
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _load  # type: ignore[assignment]

    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not (yolov5_repo / "hubconf.py").is_file():
        raise FileNotFoundError(yolov5_repo)

    yolo_str = str(yolov5_repo.resolve())
    if yolo_str not in sys.path:
        sys.path.insert(0, yolo_str)
    old = Path.cwd()
    os.chdir(yolo_str)
    try:
        model = torch.hub.load(
            yolo_str,
            "custom",
            path=str(weights),
            source="local",
            device=device,
            autoshape=True,
            _verbose=False,
        )
    finally:
        os.chdir(old)

    model.eval()
    if hasattr(model, "conf"):
        model.conf = float(conf)
    if hasattr(model, "iou"):
        model.iou = float(iou)
    model.imgsz = imgsz
    return model


def run_v2b_inference(
    manifest: list[dict[str, Any]],
    output_dir: Path,
    weights: Path,
    yolov5_repo: Path,
    project_root: Path,
    img_size: int,
    conf_thres: float,
    iou_thres: float,
    feature_conf_thresholds: list[float],
    device_req: str,
    batch_log_every: int,
    resume: bool,
    limit: int,
) -> dict[str, Any]:
    import cv2
    import torch

    device = resolve_device(device_req)
    det_dir = output_dir / "detections_1536"
    det_dir.mkdir(parents=True, exist_ok=True)
    boxes_csv = det_dir / "all_boxes.csv"
    missing_path = det_dir / "missing_images.csv"

    print(f"[v2b] device={device} cuda={torch.cuda.is_available()} imgsz={img_size}")
    print(f"[v2b] weights={weights}")
    model = load_yolov5(weights, yolov5_repo, device, conf_thres, iou_thres, img_size)
    names = []
    try:
        names = list(model.model.names.values()) if hasattr(model.model, "names") else list(model.names.values())
    except Exception:
        names = CLASS_NAMES

    rows = manifest[: limit] if limit and limit > 0 else list(manifest)

    # 断点：已完成 stem
    done_stems: set[str] = set()
    if resume and boxes_csv.exists():
        with boxes_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                done_stems.add(r.get("image_stem", ""))
        print(f"[v2b] resume: 已有 {len(done_stems)} 个 stem 的框记录")

    # 逐框 + 已处理图片清单
    write_header = not (resume and boxes_csv.exists())
    box_fields = [
        "image_name",
        "image_stem",
        "source",
        "grade",
        "split",
        "dup_group_id",
        "label",
        "class_name",
        "conf",
        "cx",
        "cy",
        "w",
        "h",
        "image_width",
        "image_height",
        "infer_img_size",
    ]
    box_f = boxes_csv.open("a" if resume and boxes_csv.exists() else "w", encoding="utf-8-sig", newline="")
    box_writer = csv.DictWriter(box_f, fieldnames=box_fields)
    if write_header:
        box_writer.writeheader()

    processed_csv = det_dir / "processed_images.csv"
    proc_fields = [
        "image_name",
        "image_stem",
        "source",
        "grade",
        "split",
        "dup_group_id",
        "image_path",
        "image_width",
        "image_height",
        "plate_area",
        "plate_area_fallback",
        "n_boxes",
        "sha256",
    ]
    proc_write_header = not (resume and processed_csv.exists())
    proc_f = processed_csv.open(
        "a" if resume and processed_csv.exists() else "w", encoding="utf-8-sig", newline=""
    )
    proc_writer = csv.DictWriter(proc_f, fieldnames=proc_fields)
    if proc_write_header:
        proc_writer.writeheader()

    done_keys: set[str] = set()
    if resume and processed_csv.exists():
        with processed_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                done_keys.add(f"{r['source']}::{r['image_stem']}::{r['grade']}")
        print(f"[v2b] resume: processed_images 已有 {len(done_keys)} 张")

    per_image_dets: dict[str, list[dict[str, Any]]] = {}
    per_image_meta: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    t0 = time.time()
    processed = 0

    try:
        for i, m in enumerate(rows):
            stem = m["image_stem"]
            key = f"{m['source']}::{stem}::{m['grade']}"
            if resume and key in done_keys:
                continue

            path = Path(m["image_path"])
            img = read_image_bgr(path)
            if img is None:
                missing.append({**m, "reason": "unreadable"})
                continue

            h0, w0 = img.shape[:2]
            plate_area, plate_fallback = compute_plate_area(img, project_root)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model(img, size=img_size)

            dets_xyxy = result.xyxy[0].cpu().numpy() if len(result.xyxy) else np.empty((0, 6))
            dets: list[dict[str, Any]] = []
            for det in dets_xyxy:
                x1, y1, x2, y2, conf, cls = det[:6]
                cls_id = int(cls)
                bw = max(0.0, float(x2 - x1))
                bh = max(0.0, float(y2 - y1))
                if bw <= 0 or bh <= 0:
                    continue
                cx = (float(x1) + float(x2)) / 2.0 / w0
                cy = (float(y1) + float(y2)) / 2.0 / h0
                wn = bw / w0
                hn = bh / h0
                d = {
                    "label": cls_id,
                    "conf": float(conf),
                    "cx": cx,
                    "cy": cy,
                    "w": wn,
                    "h": hn,
                }
                dets.append(d)
                box_writer.writerow(
                    {
                        "image_name": m["image_name"],
                        "image_stem": stem,
                        "source": m["source"],
                        "grade": m["grade"],
                        "split": m["split"],
                        "dup_group_id": m["dup_group_id"],
                        "label": cls_id,
                        "class_name": names[cls_id] if 0 <= cls_id < len(names) else str(cls_id),
                        "conf": f"{float(conf):.6f}",
                        "cx": f"{cx:.6f}",
                        "cy": f"{cy:.6f}",
                        "w": f"{wn:.6f}",
                        "h": f"{hn:.6f}",
                        "image_width": w0,
                        "image_height": h0,
                        "infer_img_size": img_size,
                    }
                )

            side = det_dir / "per_image" / m["source"] / m["grade"] / f"{stem}.txt"
            side.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                f"{d['label']} {d['cx']:.6f} {d['cy']:.6f} {d['w']:.6f} {d['h']:.6f} {d['conf']:.6f}"
                for d in dets
            ]
            side.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            per_image_dets[key] = dets
            per_image_meta[key] = {
                "image_width": w0,
                "image_height": h0,
                "plate_area": plate_area,
                "plate_area_fallback": int(plate_fallback),
                **m,
            }
            proc_writer.writerow(
                {
                    "image_name": m["image_name"],
                    "image_stem": stem,
                    "source": m["source"],
                    "grade": m["grade"],
                    "split": m["split"],
                    "dup_group_id": m["dup_group_id"],
                    "image_path": m["image_path"],
                    "image_width": w0,
                    "image_height": h0,
                    "plate_area": plate_area,
                    "plate_area_fallback": int(plate_fallback),
                    "n_boxes": len(dets),
                    "sha256": m.get("sha256", ""),
                }
            )
            processed += 1
            if processed % batch_log_every == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(rows) - i - 1) / rate / 60 if rate > 0 else -1
                print(
                    f"  [v2b] {processed} new / idx {i+1}/{len(rows)}  "
                    f"avg={rate:.2f} img/s  elapsed={elapsed/60:.1f}min  eta~{eta:.1f}min"
                )
            box_f.flush()
            proc_f.flush()
    finally:
        box_f.close()
        proc_f.close()

    # 从 processed + boxes 重建全量（含 resume 旧结果）
    per_image_dets = {}
    per_image_meta = {}
    if processed_csv.exists():
        with processed_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                key = f"{r['source']}::{r['image_stem']}::{r['grade']}"
                per_image_meta[key] = {
                    "image_width": int(float(r["image_width"])),
                    "image_height": int(float(r["image_height"])),
                    "plate_area": float(r["plate_area"]),
                    "plate_area_fallback": int(float(r.get("plate_area_fallback", 0) or 0)),
                    "image_name": r["image_name"],
                    "image_stem": r["image_stem"],
                    "source": r["source"],
                    "grade": r["grade"],
                    "split": r["split"],
                    "dup_group_id": r["dup_group_id"],
                    "image_path": r.get("image_path", ""),
                    "sha256": r.get("sha256", ""),
                }
                per_image_dets[key] = []
    if boxes_csv.exists():
        with boxes_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                key = f"{r['source']}::{r['image_stem']}::{r['grade']}"
                per_image_dets.setdefault(key, []).append(
                    {
                        "label": int(float(r["label"])),
                        "conf": float(r["conf"]),
                        "cx": float(r["cx"]),
                        "cy": float(r["cy"]),
                        "w": float(r["w"]),
                        "h": float(r["h"]),
                    }
                )
                if key not in per_image_meta:
                    per_image_meta[key] = {
                        "image_width": int(float(r["image_width"])),
                        "image_height": int(float(r["image_height"])),
                        "plate_area": float(int(float(r["image_width"])) * int(float(r["image_height"]))),
                        "plate_area_fallback": 1,
                        "image_name": r["image_name"],
                        "image_stem": r["image_stem"],
                        "source": r["source"],
                        "grade": r["grade"],
                        "split": r["split"],
                        "dup_group_id": r["dup_group_id"],
                        "image_path": "",
                    }

    # 对清单中仍缺失的图补读
    for m in rows:
        key = f"{m['source']}::{m['image_stem']}::{m['grade']}"
        if key in per_image_meta:
            meta = per_image_meta[key]
            for k in ("image_path", "source", "grade", "split", "dup_group_id", "image_name", "image_stem", "sha256"):
                if k in m and m[k]:
                    meta[k] = m[k]
            continue
        img = read_image_bgr(Path(m["image_path"]))
        if img is None:
            missing.append({**m, "reason": "unreadable_final"})
            continue
        h0, w0 = img.shape[:2]
        plate_area, plate_fallback = compute_plate_area(img, project_root)
        per_image_dets[key] = []
        per_image_meta[key] = {
            "image_width": w0,
            "image_height": h0,
            "plate_area": plate_area,
            "plate_area_fallback": int(plate_fallback),
            **m,
        }

    # 计算特征两套阈值
    feature_paths: dict[str, str] = {}
    for thr in feature_conf_thresholds:
        feat_rows: list[dict[str, Any]] = []
        for m in rows:
            key = f"{m['source']}::{m['image_stem']}::{m['grade']}"
            if key not in per_image_meta:
                continue
            meta = per_image_meta[key]
            dets = per_image_dets.get(key, [])
            feats = features_from_detections(
                dets,
                int(meta["image_width"]),
                int(meta["image_height"]),
                float(meta["plate_area"]),
                conf_thres=float(thr),
            )
            row = {
                "image_name": m["image_name"],
                "image_stem": m["image_stem"],
                "image_path": m["image_path"],
                "source": m["source"],
                "grade": m["grade"],
                "true_grade": m["grade"],
                "split": m["split"],
                "dup_group_id": m["dup_group_id"],
                "sha256": m.get("sha256", ""),
                "feature_version": f"v2b_imgsz{img_size}_conf{thr:.2f}",
                "infer_img_size": img_size,
                "feature_conf_thres": thr,
                "detector_conf": conf_thres,
                "detector_iou": iou_thres,
                "plate_area_fallback": meta.get("plate_area_fallback", 0),
                "status": "unconfirmed_pending_enterprise",
                **feats,
            }
            feat_rows.append(row)
        thr_tag = f"{thr:.2f}".replace(".", "p")
        out_csv = output_dir / f"features_v2b_imgsz{img_size}_conf{thr_tag}.csv"
        write_csv(out_csv, feat_rows)
        feature_paths[f"conf_{thr_tag}"] = str(out_csv)
        print(f"[v2b] 特征 conf>={thr}: {len(feat_rows)} -> {out_csv.name}")

    write_csv(missing_path, missing)
    covered = len(per_image_meta)
    summary = {
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "weights": str(weights.resolve()),
        "img_size": img_size,
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
        "manifest_requested": len(rows),
        "images_with_meta": covered,
        "coverage": covered / len(rows) if rows else 0.0,
        "missing_count": len(missing),
        "missing_path": str(missing_path),
        "boxes_csv": str(boxes_csv),
        "feature_csvs": feature_paths,
        "class_names": names,
        "elapsed_sec": time.time() - t0,
        "status": "unconfirmed_pending_enterprise",
    }
    (det_dir / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=== 电积镍等级 v2 数据集构建 ===")
    print(f"输出: {out}")
    print(f"时间: {datetime.now().isoformat(timespec='seconds')}")
    print("状态标记: unconfirmed_pending_enterprise（未经企业确认）")

    classify_rows = scan_classify(args.classify_root)
    enterprise_rows = scan_enterprise(args.enterprise_root)
    same_grade_only = not bool(args.near_dup_all_grades)
    manifest, dedup_meta = merge_and_dedup(
        classify_rows,
        enterprise_rows,
        phash_threshold=args.phash_threshold,
        phash_size=args.phash_size,
        same_grade_only=same_grade_only,
    )

    manifest_fields = [
        "image_name",
        "image_stem",
        "image_path",
        "source",
        "grade",
        "split",
        "dup_group_id",
        "sha256",
        "phash",
        "kept_after_exact_dedup",
    ]
    write_csv(out / "merged_manifest.csv", manifest, manifest_fields)
    (out / "dedup_summary.json").write_text(
        json.dumps(dedup_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[清单] merged_manifest.csv 行数={len(manifest)}")

    if args.dry_run:
        print("[dry-run] 跳过 v2a/v2b")
        return 0

    build_summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "unconfirmed_pending_enterprise",
        "dedup": dedup_meta,
        "manifest_rows": len(manifest),
        "small_sample_warning": {
            "3#_total": dedup_meta["grade_counts_kept"].get("3#", 0),
            "bad_total": dedup_meta["grade_counts_kept"].get("bad", 0),
            "message": "3# 与 bad 仍为小样本，指标方差大，禁止过拟合后宣称可用",
        },
    }

    if not args.skip_v2a:
        joined, v2a_meta, missing = join_v2a(manifest, args.samples_csv)
        write_csv(out / "features_v2a.csv", joined)
        write_csv(out / "features_v2a_missing.csv", missing)
        (out / "features_v2a_summary.json").write_text(
            json.dumps(v2a_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        build_summary["v2a"] = v2a_meta
        print(
            f"[v2a] join {v2a_meta['joined']}/{v2a_meta['manifest_total']} "
            f"coverage={v2a_meta['coverage']:.2%} missing={v2a_meta['missing']}"
        )

    if not args.skip_v2b:
        v2b_summary = run_v2b_inference(
            manifest=manifest,
            output_dir=out,
            weights=args.weights,
            yolov5_repo=args.yolov5_repo,
            project_root=args.project_root,
            img_size=args.img_size,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            feature_conf_thresholds=list(args.feature_conf_thresholds),
            device_req=args.device,
            batch_log_every=args.batch_log_every,
            resume=args.resume_v2b,
            limit=args.limit,
        )
        build_summary["v2b"] = v2b_summary
        print(
            f"[v2b] coverage={v2b_summary['coverage']:.2%} "
            f"missing={v2b_summary['missing_count']} device={v2b_summary['device']}"
        )

    (out / "build_summary.json").write_text(
        json.dumps(build_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("完成。产物见", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
