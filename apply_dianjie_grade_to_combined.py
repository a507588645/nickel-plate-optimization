#!/usr/bin/env python3
"""对 combined_all_pseudo_v1 汇总数据集使用原项目电解镍等级规则做等级划分（多进程加速）。

- 读取 images_all/ 下的软连接图片（扁平汇总）。
- 使用原项目 preprocess.findcent 计算镍板面积（1024x888 坐标系）。
- 读取 labels_with_conf/ 六列伪标签，按置信度阈值过滤。
- 调用 dianjie_grade_rule.DianjieGradeRule 计算等级与 A/B/C/气孔比例。
- 输出 CSV 与统计摘要，供 LightGBM 等级分类可行性判断。
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
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

# 每个工作进程的全局对象，在 initializer 中设置
_worker_globals: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 combined_all_pseudo_v1 用原项目电解镍等级规则做等级划分",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--combined-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/combined_all_pseudo_v1"),
        help="汇总数据集根目录，需包含 images_all/ 和 labels_with_conf/",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi"),
        help="原项目根目录，用于导入 preprocess.findcent",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade"),
        help="输出目录",
    )
    parser.add_argument("--conf-thres", type=float, default=0.70, help="伪标签置信度阈值")
    parser.add_argument(
        "--plate-area-fallback",
        type=float,
        default=1024.0 * 888.0,
        help="findcent 失败时使用的默认镍板面积",
    )
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1), help="并行进程数")
    return parser.parse_args()


def _worker_init(project_root_str: str, script_dir_str: str) -> None:
    project_root = Path(project_root_str).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from preprocess import findcent

    script_dir = Path(script_dir_str).resolve()
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from dianjie_grade_rule import DianjieGradeRule

    _worker_globals["findcent"] = findcent
    _worker_globals["rule"] = DianjieGradeRule()


def _read_detections(label_path: Path, conf_thres: float) -> list[dict[str, Any]]:
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


def _normalized_rect(det: dict[str, Any]) -> tuple[float, float, float, float]:
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


def _union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
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


def _to_pixel_dets(
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


def _process_image(
    task: tuple[str, str, float, float],
) -> dict[str, Any] | None:
    image_path_str, labels_dir_str, conf_thres, plate_area_fallback = task
    image_path = Path(image_path_str)
    labels_dir = Path(labels_dir_str)
    label_path = labels_dir / f"{image_path.stem}.txt"
    findcent = _worker_globals["findcent"]
    rule = _worker_globals["rule"]

    image = cv2.imread(str(image_path))
    findcent_fallback = False
    if image is None:
        image_height, image_width = 3554, 4096
        plate_area = plate_area_fallback
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
            plate_area = plate_area_fallback
            findcent_fallback = True

    detections = _read_detections(label_path, conf_thres) if label_path.exists() else []
    pixel_dets = _to_pixel_dets(detections, image_width, image_height)

    result_none = rule.classify(pixel_dets, plate_area, predicted_idx=None)
    result_good = rule.classify(pixel_dets, plate_area, predicted_idx=1)

    all_rects = [_normalized_rect(det) for det in detections]
    total_bbox_union_ratio = _union_area(all_rects)

    per_class_counts: dict[int, int] = defaultdict(int)
    per_class_areas: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for det in detections:
        cid = int(det["label"])
        per_class_counts[cid] += 1
        per_class_areas[cid].append(_normalized_rect(det))

    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "label_exists": label_path.exists(),
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
        "per_class_counts": dict(per_class_counts),
        "per_class_areas": {k: _union_area(v) for k, v in per_class_areas.items()},
    }


def main() -> None:
    args = parse_args()
    combined_root = args.combined_root.resolve()
    images_dir = combined_root / "images_all"
    labels_dir = combined_root / "labels_with_conf"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"未找到汇总图片目录: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"未找到伪标签目录: {labels_dir}")

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )

    # 加载类别名称（如果存在）
    classes_path = labels_dir / "classes.txt"
    class_names = (
        [line.strip() for line in classes_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if classes_path.exists()
        else []
    )

    tasks = [
        (str(p), str(labels_dir), args.conf_thres, args.plate_area_fallback)
        for p in image_paths
    ]

    print(f"开始处理 {len(tasks)} 张图片，使用 {args.workers} 个进程...")
    results: list[dict[str, Any] | None] = []
    with mp.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(str(args.project_root.resolve()), str(Path(__file__).resolve().parent)),
    ) as pool:
        for i, result in enumerate(pool.imap(_process_image, tasks), 1):
            results.append(result)
            if i % 500 == 0 or i == len(tasks):
                print(f"  已处理 {i}/{len(tasks)}")

    # 组装最终样本行
    samples: list[dict[str, Any]] = []
    audit = {
        "total_images": len(image_paths),
        "missing_labels": 0,
        "unreadable_images": 0,
        "findcent_fallbacks": 0,
        "empty_after_filter": 0,
    }
    for raw in results:
        if raw is None:
            continue
        if not raw["label_exists"]:
            audit["missing_labels"] += 1
        if raw["image_width"] == 4096 and raw["image_height"] == 3554 and raw["findcent_fallback"]:
            # 不可读时我们默认设为该尺寸，这里也统计为不可读
            audit["unreadable_images"] += 1
        if raw["findcent_fallback"]:
            audit["findcent_fallbacks"] += 1
        if raw["detection_count"] == 0:
            audit["empty_after_filter"] += 1

        row: dict[str, Any] = {
            "image_name": raw["image_name"],
            "image_path": raw["image_path"],
            "label_path": str(labels_dir / f"{Path(raw['image_path']).stem}.txt"),
            "image_width": raw["image_width"],
            "image_height": raw["image_height"],
            "plate_area": raw["plate_area"],
            "findcent_fallback": raw["findcent_fallback"],
            "detection_count": raw["detection_count"],
            "max_confidence": raw["max_confidence"],
            "total_bbox_union_image_ratio": raw["total_bbox_union_image_ratio"],
            "A_ratio": raw["A_ratio"],
            "B_ratio": raw["B_ratio"],
            "C_ratio": raw["C_ratio"],
            "Airhole_ratio": raw["Airhole_ratio"],
            "has_nodule": raw["has_nodule"],
            "nodule_status": raw["nodule_status"],
            "type_id_detection_only": raw["type_id_detection_only"],
            "type_name_detection_only": raw["type_name_detection_only"],
            "type_id_assume_resnet_good": raw["type_id_assume_resnet_good"],
            "type_name_assume_resnet_good": raw["type_name_assume_resnet_good"],
        }
        for class_id, class_name in enumerate(class_names):
            row[f"count__{class_name}"] = raw["per_class_counts"].get(class_id, 0)
            row[f"area_ratio__{class_name}"] = raw["per_class_areas"].get(class_id, 0.0)
        samples.append(row)

    # 输出 CSV
    csv_path = output_dir / "samples.csv"
    fieldnames = list(samples[0]) if samples else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)

    # 等级分布
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

    # 控制台摘要
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
