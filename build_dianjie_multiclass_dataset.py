#!/usr/bin/env python3
"""
基于 308 分支电解镍检测模型 + 硬编码规则，
把二分类 folder 数据集 classify_2types 扩展为多类软连接 folder 数据集。

输出类别（与原项目 start.py 一致）：
    0 无板
    1 3号板
    2 烫洗板
    3 合格整版
    4 合格600
    5 合格不足600
    6 2号板

产物结构：
    classify_multiclass_dianjie/
        train/
            0/
            1/
            ...
        valid/
            0/
            ...
        link_records.csv
        labels.json   # 每张原图 -> type 的映射

要求：以软连接方式引用原图，不复制、不改动原 classify_2types。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

# 允许加载原项目 YOLOv5 加密权重
_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args: Any, **kwargs: Any) -> Any:
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

_ORIGINAL_PROJECT = Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi")
_SOURCE_DATASET = Path(r"E:/PythonProject/datasets/classify_2types")


def _ensure_original_project_in_path() -> None:
    if str(_ORIGINAL_PROJECT) not in sys.path:
        sys.path.insert(0, str(_ORIGINAL_PROJECT))
    os.chdir(_ORIGINAL_PROJECT)


def create_symlink_or_record(src: Path, dst: Path, records: list[dict[str, Any]]) -> None:
    """尝试创建软连接，失败则记录路径映射。"""
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
        records.append({"link": str(dst), "target": str(src), "type": "symlink"})
    except OSError:
        records.append({"link": str(dst), "target": str(src), "type": "path_record"})


def load_detector(weights_dir: Path, device: str) -> Any:
    _ensure_original_project_in_path()
    from model_decryptor import load_yolov5_with_encryption
    model = load_yolov5_with_encryption(str(weights_dir / "ryw.pt"), device=device, autoshape=True)
    model.conf = 0.25
    model.iou = 0.45
    return model


def detect_one(model: Any, image_path: Path, resize_wh: tuple[int, int] = (1024, 888)) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        return np.empty((0, 6))
    resized = cv2.resize(img, resize_wh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model(resized)
    if len(results.xyxy) == 0 or results.xyxy[0] is None:
        return np.empty((0, 6))
    return results.xyxy[0].cpu().numpy()


def compute_plate_area(image_path: Path, resize_wh: tuple[int, int] = (1024, 888)) -> float:
    _ensure_original_project_in_path()
    from preprocess import findcent
    img = cv2.imread(str(image_path))
    if img is None:
        return float(resize_wh[0] * resize_wh[1])
    resized = cv2.resize(img, resize_wh)
    try:
        _, w, h, _ = findcent(resized)
        return float(w * h)
    except Exception:
        return float(resize_wh[0] * resize_wh[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="构建电解镍多类软连接 folder 数据集")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_SOURCE_DATASET,
        help="原二分类 folder 数据集根目录",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/classify_multiclass_dianjie"),
        help="输出多类 folder 数据集根目录",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=_ORIGINAL_PROJECT,
        help="原项目权重目录（含 ryw.pt）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="检测模型设备",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出统计信息，不创建软连接",
    )
    args = parser.parse_args()

    from dianjie_grade_rule import dianjie_grade_from_xyxy

    print("加载电解镍检测模型 ryw.pt...")
    detector = load_detector(args.weights_dir, args.device)

    args.output_root.mkdir(parents=True, exist_ok=True)
    link_records: list[dict[str, Any]] = []
    labels: dict[str, int] = {}
    type_counts: dict[str, dict[int, int]] = {"train": {}, "valid": {}}

    for split in ["train", "valid"]:
        split_dir = args.source_root / split
        if not split_dir.exists():
            continue

        # 收集该 split 下所有图片（不论原 good/bad 子文件夹）
        samples: list[Path] = []
        for cls_dir in split_dir.iterdir():
            if cls_dir.is_dir():
                samples.extend(cls_dir.glob("*.*"))

        print(f"\n处理 {split}: {len(samples)} 张")

        for idx, src_path in enumerate(samples, 1):
            rel_name = src_path.name
            plate_area = compute_plate_area(src_path)
            detections = detect_one(detector, src_path)
            grade = dianjie_grade_from_xyxy(detections, plate_area)
            type_id = grade.type_id

            labels[str(src_path)] = type_id
            type_counts[split][type_id] = type_counts[split].get(type_id, 0) + 1

            if not args.dry_run:
                type_dir = args.output_root / split / str(type_id)
                type_dir.mkdir(parents=True, exist_ok=True)
                dst_path = type_dir / rel_name
                create_symlink_or_record(src_path.resolve(), dst_path, link_records)

            if idx % 50 == 0:
                print(f"  已处理 {idx}/{len(samples)}，当前 {split} 分布: {type_counts[split]}")

    # 保存标签映射和链接记录
    if not args.dry_run:
        with open(args.output_root / "labels.json", "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

        with open(args.output_root / "link_records.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["link", "target", "type"])
            writer.writeheader()
            writer.writerows(link_records)

    print("\n=== 完成 ===")
    print("各 split 类别分布:")
    for split, counts in type_counts.items():
        print(f"  {split}: {dict(sorted(counts.items()))}")
    if not args.dry_run:
        print(f"输出目录: {args.output_root}")
        print(f"软连接成功/记录数: {len(link_records)}")


if __name__ == "__main__":
    main()
