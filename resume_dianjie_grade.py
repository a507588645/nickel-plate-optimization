#!/usr/bin/env python3
"""续跑 apply_dianjie_grade_to_combined_sequential.py，处理剩余图片并更新摘要。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
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
    parser = argparse.ArgumentParser(description="续跑等级划分")
    parser.add_argument("--combined-root", type=Path, default=Path(r"E:/PythonProject/datasets/combined_all_pseudo_v2"))
    parser.add_argument("--project-root", type=Path, default=Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v2_grade"))
    parser.add_argument("--conf-thres", type=float, default=0.70)
    parser.add_argument("--plate-area-fallback", type=float, default=1024.0 * 888.0)
    return parser.parse_args()


args = parse_args()
COMBINED_ROOT = args.combined_root
PROJECT_ROOT = args.project_root
OUTPUT_DIR = args.output_dir
CONF_THRES = args.conf_thres
PLATE_AREA_FALLBACK = args.plate_area_fallback


def load_findcent() -> Any:
    p = PROJECT_ROOT.resolve()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    from preprocess import findcent

    return findcent


def load_rule() -> Any:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from dianjie_grade_rule import DianjieGradeRule

    return DianjieGradeRule()


def read_image(path: Path) -> np.ndarray | None:
    """Read Windows paths containing non-ASCII characters safely."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def read_detections(label_path: Path, conf_thres: float) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if not label_path.exists():
        return detections
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 6:
            continue
        class_id = int(fields[0])
        cx, cy, w, h, confidence = map(float, fields[1:])
        if confidence < conf_thres:
            continue
        detections.append(
            {
                "label": class_id,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "confidence": confidence,
            }
        )
    return detections


def normalized_rect(det: dict[str, Any]) -> tuple[float, float, float, float]:
    cx = float(det["cx"])
    cy = float(det["cy"])
    w = float(det["w"])
    h = float(det["h"])
    return (
        max(0.0, cx - w / 2),
        max(0.0, cy - h / 2),
        min(1.0, cx + w / 2),
        min(1.0, cy + h / 2),
    )


def union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
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


def to_pixel_dets(
    detections: list[dict[str, Any]], image_width: int, image_height: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for det in detections:
        out.append(
            {
                "label": int(det["label"]),
                "cx": float(det["cx"]) * image_width,
                "cy": float(det["cy"]) * image_height,
                "w": float(det["w"]) * image_width,
                "h": float(det["h"]) * image_height,
            }
        )
    return out


def process_image(image_path: Path, labels_dir: Path, findcent: Any, rule: Any, class_names: list[str]) -> dict[str, Any]:
    label_path = labels_dir / f"{image_path.stem}.txt"
    image = read_image(image_path)
    findcent_fallback = False
    if image is None:
        image_height, image_width = 3554, 4096
        plate_area = PLATE_AREA_FALLBACK
        findcent_fallback = True
    else:
        image_height, image_width = image.shape[:2]
        resized = cv2.resize(image, (1024, 888))
        try:
            _, plate_width, plate_height, _ = findcent(resized)
            plate_area = float(plate_width * plate_height)
            if plate_area <= 0:
                raise ValueError("plate_area <= 0")
        except Exception:
            plate_area = PLATE_AREA_FALLBACK
            findcent_fallback = True

    detections = read_detections(label_path, CONF_THRES) if label_path.exists() else []
    pixel_dets = to_pixel_dets(detections, image_width, image_height)
    result_none = rule.classify(pixel_dets, plate_area, predicted_idx=None)
    result_good = rule.classify(pixel_dets, plate_area, predicted_idx=1)

    all_rects = [normalized_rect(det) for det in detections]
    total_bbox_union_ratio = union_area(all_rects)

    per_class_counts: dict[int, int] = defaultdict(int)
    per_class_areas: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for det in detections:
        cid = int(det["label"])
        per_class_counts[cid] += 1
        per_class_areas[cid].append(normalized_rect(det))

    row: dict[str, Any] = {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "label_path": str(label_path) if label_path.exists() else "",
        "image_width": image_width,
        "image_height": image_height,
        "plate_area": plate_area,
        "findcent_fallback": findcent_fallback,
        "detection_count": len(detections),
        "max_confidence": max((float(det["confidence"]) for det in detections), default=0.0),
        "total_bbox_union_image_ratio": total_bbox_union_ratio,
        "A_ratio": result_none.ratios["A_ratio"],
        "B_ratio": result_none.ratios["B_ratio"],
        "C_ratio": result_none.ratios["C_ratio"],
        "Airhole_ratio": result_none.ratios["Airhole_ratio"],
        "has_nodule": int(result_none.has_nodule),
        "nodule_status": result_none.nodule_status,
        "type_id_detection_only": result_none.type_id,
        "type_name_detection_only": result_none.type_name,
        "type_id_assume_resnet_good": result_good.type_id,
        "type_name_assume_resnet_good": result_good.type_name,
    }
    for class_id, class_name in enumerate(class_names):
        row[f"count__{class_name}"] = per_class_counts[class_id]
        row[f"area_ratio__{class_name}"] = union_area(per_class_areas[class_id])
    return row


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    findcent = load_findcent()
    rule = load_rule()
    images_dir = COMBINED_ROOT / "images_all"
    labels_dir = COMBINED_ROOT / "labels_with_conf"

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )

    classes_path = labels_dir / "classes.txt"
    class_names = (
        [line.strip() for line in classes_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if classes_path.exists()
        else []
    )

    csv_path = OUTPUT_DIR / "samples.csv"
    processed: set[str] = set()
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] | None = None
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            for row in reader:
                processed.add(row["image_name"])
                rows.append(row)

    remaining = [p for p in image_paths if p.name not in processed]
    print(f"已处理 {len(processed)} 张，剩余 {len(remaining)} 张")

    with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [], extrasaction="ignore")
        for i, image_path in enumerate(remaining, 1):
            row = process_image(image_path, labels_dir, findcent, rule, class_names)
            writer.writerow(row)
            rows.append(row)
            if i % 100 == 0 or i == len(remaining):
                print(f"  已补充 {i}/{len(remaining)}", flush=True)

    # 重新统计
    def distribution(key: str) -> dict[str, Any]:
        counter = Counter(str(row[key]) for row in rows)
        return {TYPE_NAMES.get(int(k), k): v for k, v in sorted(counter.items(), key=lambda x: int(x[0]))}

    audit = {
        "total_images": len(image_paths),
        "missing_labels": sum(1 for r in rows if not r.get("label_path")),
        "unreadable_images": sum(
            1 for r in rows if int(r["image_width"]) == 4096 and int(r["image_height"]) == 3554 and r.get("findcent_fallback") == "True"
        ),
        "findcent_fallbacks": sum(1 for r in rows if r.get("findcent_fallback") == "True"),
        "empty_after_filter": sum(1 for r in rows if int(r["detection_count"]) == 0),
    }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "combined_root": str(COMBINED_ROOT),
        "conf_threshold": CONF_THRES,
        "total_images": len(rows),
        "audit": audit,
        "grade_distribution_detection_only": distribution("type_id_detection_only"),
        "grade_distribution_assume_resnet_good": distribution("type_id_assume_resnet_good"),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"总图片: {len(rows)}")
    print(f"审计: {audit}")
    print("\n等级分布（detection_only）:")
    for name, count in summary["grade_distribution_detection_only"].items():
        print(f"  {name}: {count} ({count / len(rows) * 100:.2f}%)")
    print("\n等级分布（assume_resnet_good）:")
    for name, count in summary["grade_distribution_assume_resnet_good"].items():
        print(f"  {name}: {count} ({count / len(rows) * 100:.2f}%)")


if __name__ == "__main__":
    main()
