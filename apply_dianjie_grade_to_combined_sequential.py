#!/usr/bin/env python3
"""对 combined_all_pseudo_v1 汇总数据集使用原项目电解镍等级规则做等级划分（顺序单进程，避免多进程潜在死锁）。"""

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
    parser = argparse.ArgumentParser(
        description="对 combined_all_pseudo_v1 用原项目电解镍等级规则做等级划分（顺序版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--combined-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/combined_all_pseudo_v2"),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v2_grade"),
    )
    parser.add_argument("--conf-thres", type=float, default=0.70)
    parser.add_argument(
        "--plate-area-fallback",
        type=float,
        default=1024.0 * 888.0,
    )
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def load_findcent(project_root: Path) -> Any:
    project_root = project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
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


def main() -> None:
    args = parse_args()
    combined_root = args.combined_root.resolve()
    images_dir = combined_root / "images_all"
    labels_dir = combined_root / "labels_with_conf"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    findcent = load_findcent(args.project_root)
    rule = load_rule()

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

    audit = {
        "total_images": len(image_paths),
        "missing_labels": 0,
        "unreadable_images": 0,
        "findcent_fallbacks": 0,
        "empty_after_filter": 0,
    }

    csv_path = output_dir / "samples.csv"
    fieldnames: list[str] | None = None
    samples_count = 0

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer: csv.DictWriter | None = None

        for index, image_path in enumerate(image_paths, 1):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                audit["missing_labels"] += 1

            image = read_image(image_path)
            findcent_fallback = False
            if image is None:
                audit["unreadable_images"] += 1
                image_height, image_width = 3554, 4096
                plate_area = args.plate_area_fallback
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
                    plate_area = args.plate_area_fallback
                    findcent_fallback = True
                    audit["findcent_fallbacks"] += 1

            detections = read_detections(label_path, args.conf_thres) if label_path.exists() else []
            if not detections:
                audit["empty_after_filter"] += 1

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

            if writer is None:
                fieldnames = list(row)
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
            writer.writerow(row)
            samples_count += 1

            if index % args.progress_every == 0 or index == len(image_paths):
                print(f"已处理 {index}/{len(image_paths)}", flush=True)

    # 汇总分布
    samples: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(row)

    def distribution(key: str) -> dict[str, Any]:
        counter = Counter(str(sample[key]) for sample in samples)
        return {TYPE_NAMES.get(int(k), k): v for k, v in sorted(counter.items(), key=lambda x: int(x[0]))}

    dist_detection = distribution("type_id_detection_only")
    dist_good = distribution("type_id_assume_resnet_good")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "combined_root": str(combined_root),
        "conf_threshold": args.conf_thres,
        "total_images": len(samples),
        "audit": audit,
        "grade_distribution_detection_only": dist_detection,
        "grade_distribution_assume_resnet_good": dist_good,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n输出目录: {output_dir}")
    print(f"总图片: {len(samples)}")
    print(f"审计: {audit}")
    print("\n等级分布（detection_only）:")
    for name, count in dist_detection.items():
        print(f"  {name}: {count} ({count / len(samples) * 100:.2f}%)")
    print("\n等级分布（assume_resnet_good）:")
    for name, count in dist_good.items():
        print(f"  {name}: {count} ({count / len(samples) * 100:.2f}%)")


if __name__ == "__main__":
    main()
