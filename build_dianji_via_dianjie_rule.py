#!/usr/bin/env python3
"""
用电解镍的 7 类规则给电积镍图片打伪标签，然后归纳电积镍 4 类标准。

流程：
1. 电积镍图片 -> slf.pt 检测
2. slf.pt 标签按 start.py 映射表转成电解镍标签
3. 应用 dianjie_grade_rule 得到 type=0~6
4. 统计每类 type 的特征/缺陷分布
5. 根据分布相似性把 7 类归并为 4 类
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
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args: Any, **kwargs: Any) -> Any:
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false

_ORIGINAL_PROJECT = Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi")
_DIANJI_ROOT = Path(r"E:/PythonProject/datasets/classify_4types_dianji")

# start.py 中电积镍 -> 电解镍的缺陷标签映射
SLF_TO_DIANJIE_LABEL = {
    0: 1,  # waterstain水渍 -> 水渍
    1: 0,  # waterblock深色水块 -> 污渍
    2: 2,  # colorstain带颜色的 -> 黄渍
    3: 3,  # nodule边部结瘤 -> 边部结瘤
    4: 5,  # bubble结粒 -> 结粒
}


def _ensure_original_project_in_path() -> None:
    if str(_ORIGINAL_PROJECT) not in sys.path:
        sys.path.insert(0, str(_ORIGINAL_PROJECT))
    os.chdir(_ORIGINAL_PROJECT)


def load_slf_detector(weights_dir: Path, device: str) -> Any:
    _ensure_original_project_in_path()
    from model_decryptor import load_yolov5_with_encryption
    model = load_yolov5_with_encryption(str(weights_dir / "slf.pt"), device=device, autoshape=True)
    model.conf = 0.25
    model.iou = 0.45
    return model


def detect_and_map(image_path: Path, detector: Any, resize_wh: tuple[int, int] = (1024, 888)) -> tuple[np.ndarray, list[dict[str, Any]]]:
    img = cv2.imread(str(image_path))
    if img is None:
        return np.empty((0, 6)), []
    resized = cv2.resize(img, resize_wh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = detector(resized)
    dets = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else np.empty((0, 6))

    output = []
    mapped_dets = []
    for det in dets:
        x1, y1, x2, y2, conf, cls = det
        slf_cls = int(cls)
        dianjie_cls = SLF_TO_DIANJIE_LABEL.get(slf_cls, slf_cls)
        output.append(
            {
                "label": dianjie_cls,
                "cx": float((x1 + x2) / 2),
                "cy": float((y1 + y2) / 2),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
            }
        )
        mapped_dets.append([x1, y1, x2, y2, conf, dianjie_cls])

    return np.array(mapped_dets), output


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
    parser = argparse.ArgumentParser(description="用电解镍规则给电积镍打伪标签")
    parser.add_argument(
        "--dianji-root",
        type=Path,
        default=_DIANJI_ROOT,
        help="电积镍 folder 数据集根目录",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/classify_dianji_via_dianjie_rule"),
        help="输出根目录",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=_ORIGINAL_PROJECT,
        help="原项目权重目录（含 slf.pt）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="检测设备",
    )
    args = parser.parse_args()

    from dianjie_grade_rule import DianjieGradeRule

    print("加载电积镍检测模型 slf.pt...")
    detector = load_slf_detector(args.weights_dir, args.device)
    rule = DianjieGradeRule()

    args.output_root.mkdir(parents=True, exist_ok=True)

    # 收集所有电积镍图片
    samples: list[Path] = []
    for split in ["train", "valid"]:
        split_dir = args.dianji_root / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if cls_dir.is_dir():
                samples.extend(cls_dir.glob("*.*"))

    print(f"共 {len(samples)} 张电积镍图片")

    records: list[dict[str, Any]] = []
    type_counts: dict[int, int] = {}
    labels: dict[str, int] = {}
    link_records: list[dict[str, Any]] = []

    for idx, src_path in enumerate(samples, 1):
        plate_area = compute_plate_area(src_path)
        mapped_dets, output = detect_and_map(src_path, detector)
        grade = rule.classify(output, plate_area)
        type_id = grade.type_id

        type_counts[type_id] = type_counts.get(type_id, 0) + 1
        labels[str(src_path)] = type_id

        split = src_path.parent.parent.name
        dst = args.output_root / split / str(type_id) / src_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not dst.exists() and not dst.is_symlink():
                os.symlink(src_path.resolve(), dst)
            link_records.append({"link": str(dst), "target": str(src_path), "type": "symlink"})
        except OSError:
            link_records.append({"link": str(dst), "target": str(src_path), "type": "path_record"})

        records.append(
            {
                "image_path": str(src_path),
                "split": split,
                "original_class": src_path.parent.name,
                "dianjie_type": type_id,
                "dianjie_type_name": grade.type_name,
                "plate_area": plate_area,
                **grade.ratios,
            }
        )

        if idx % 100 == 0:
            print(f"  已处理 {idx}/{len(samples)}，当前分布: {type_counts}")

    # 保存结果
    df = pd.DataFrame(records)
    df.to_csv(args.output_root / "pseudo_labels.csv", index=False, encoding="utf-8-sig")

    with open(args.output_root / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    with open(args.output_root / "link_records.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["link", "target", "type"])
        writer.writeheader()
        writer.writerows(link_records)

    print("\n=== 电解镍规则作用在电积镍上的分布 ===")
    print(f"type_counts: {dict(sorted(type_counts.items()))}")

    print("\n=== 每类的缺陷比例特征 ===")
    print(df.groupby("dianjie_type")[["A_ratio", "B_ratio", "C_ratio", "Airhole_ratio"]].mean().round(4))

    print(f"\n输出目录: {args.output_root}")


if __name__ == "__main__":
    main()
