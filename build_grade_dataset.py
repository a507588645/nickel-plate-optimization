#!/usr/bin/env python3
"""
构建基于检测特征的等级分类数据集。

不改动原始分类数据集，通过软连接建立新的版本化目录，
并使用原项目加密/非加密 YOLOv5 权重提取缺陷统计特征。
"""

from __future__ import annotations

import argparse
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
from tqdm import tqdm

# 抑制 torch.cuda.amp.autocast 的 FutureWarning，避免 CPU 推理时刷屏
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 允许在本机加载原项目 YOLOv5 权重（PyTorch 2.6+ 默认 weights_only=True）
_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args: Any, **kwargs: Any) -> Any:
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false

# 将原项目目录加入路径以便导入 model_decryptor 和 preprocess
_ORIGINAL_PROJECT = Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi")
sys.path.insert(0, str(_ORIGINAL_PROJECT))
os.chdir(_ORIGINAL_PROJECT)

from model_decryptor import load_yolov5_with_encryption  # noqa: E402
from preprocess import findcent  # noqa: E402


TASK_CONFIG = {
    "dianjie": {
        "name": "电解镍整板等级",
        "weights": "ryw.pt",
        "class_folders": ["bad", "good"],
        "label_map": {"bad": 0, "good": 1},
    },
    "dianji": {
        "name": "电积镍整板等级",
        "weights": "slf.pt",
        "class_folders": ["3#", "9950", "9996", "bad"],
        "label_map": {"3#": 0, "9950": 1, "9996": 2, "bad": 3},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建基于原项目检测输出的等级分类数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dianjie-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/classify_2types"),
        help="电解镍分类数据集根目录",
    )
    parser.add_argument(
        "--dianji-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/classify_4types_dianji"),
        help="电积镍分类数据集根目录",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/classify_from_detection_v1"),
        help="输出版本化数据集根目录",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=_ORIGINAL_PROJECT,
        help="原项目权重所在目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="推理设备，如 cpu / cuda:0",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="检测置信度阈值",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="检测 NMS IoU 阈值",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[1024, 888],
        help="推理前 resize 尺寸 (宽 高)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成清单，不创建软连接、不运行推理",
    )
    parser.add_argument(
        "--skip-symlinks",
        action="store_true",
        help="不创建软连接，仅记录原始路径到清单",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=["dianjie", "dianji"],
        default=["dianjie", "dianji"],
        help="要处理的任务",
    )
    return parser.parse_args()


def create_symlink_or_record(src: Path, dst: Path, records: list[dict[str, Any]]) -> None:
    """优先创建软连接；失败则记录原始路径，不复制原图。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
        records.append({"link": str(dst), "target": str(src), "type": "symlink"})
    except OSError:
        # Windows 未开启 Developer Mode 或权限不足时，仅记录路径
        records.append({"link": str(dst), "target": str(src), "type": "path_record"})


def collect_samples(task: str, root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """收集一个任务下的所有图片样本，保持原始 train/valid 划分。"""
    samples = []
    for split in ["train", "valid"]:
        split_dir = root / split
        if not split_dir.exists():
            warnings.warn(f"划分目录不存在: {split_dir}", stacklevel=2)
            continue
        for class_name in cfg["class_folders"]:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    continue
                samples.append({
                    "image_path": str(img_path.resolve()),
                    "split": split,
                    "class_name": class_name,
                    "label": cfg["label_map"][class_name],
                    "task": task,
                    "image_name": img_path.name,
                })
    return samples


def build_symlink_tree(
    task: str,
    samples: list[dict[str, Any]],
    output_root: Path,
    skip_symlinks: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """在新版本目录下建立软连接树，返回链接记录。"""
    link_records: list[dict[str, Any]] = []
    if skip_symlinks or dry_run:
        for s in samples:
            link_records.append({
                "link": None,
                "target": s["image_path"],
                "type": "dry_run" if dry_run else "path_record",
            })
        return link_records

    for s in samples:
        src = Path(s["image_path"])
        dst = output_root / task / "images" / s["split"] / s["class_name"] / s["image_name"]
        create_symlink_or_record(src, dst, link_records)
    return link_records


def load_detector(weights_path: Path, device: str) -> Any:
    """加载原项目 YOLOv5 检测器。"""
    if not weights_path.exists():
        raise FileNotFoundError(f"权重不存在: {weights_path}")
    model = load_yolov5_with_encryption(str(weights_path), device=device, autoshape=True)
    model.conf = 0.25  # 默认置信度
    model.iou = 0.45   # 默认 NMS IoU
    return model


def extract_image_features(
    image_path: str,
    model: Any,
    resize_wh: tuple[int, int],
    conf_thres: float,
    iou_thres: float,
    class_names: list[str],
) -> dict[str, Any]:
    """对单张图提取缺陷统计特征。"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    resized = cv2.resize(img, tuple(resize_wh))
    h, w = resized.shape[:2]

    # 计算镍板区域（与原项目 start.py 一致）
    try:
        _, plate_w, plate_h, _ = findcent(resized)
        plate_area = float(plate_w * plate_h)
    except Exception:
        plate_area = float(w * h)

    # 运行检测
    model.conf = conf_thres
    model.iou = iou_thres
    results = model(resized)
    detections = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else np.empty((0, 6))

    feature: dict[str, Any] = {
        "image_path": image_path,
        "img_w": w,
        "img_h": h,
        "plate_area": plate_area,
        "total_defect_count": len(detections),
    }

    total_defect_area = 0.0
    per_class = {name: [] for name in class_names}

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        area = float((x2 - x1) * (y2 - y1))
        total_defect_area += area
        cls_id = int(cls)
        if 0 <= cls_id < len(class_names):
            per_class[class_names[cls_id]].append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": float(conf),
                "area": area,
                "w": float(x2 - x1),
                "h": float(y2 - y1),
            })

    feature["total_defect_area"] = total_defect_area
    feature["total_defect_area_ratio"] = total_defect_area / plate_area if plate_area > 0 else 0.0

    for name in class_names:
        dets = per_class[name]
        feature[f"{name}_count"] = len(dets)
        feature[f"{name}_total_area"] = sum(d["area"] for d in dets)
        feature[f"{name}_max_area"] = max((d["area"] for d in dets), default=0.0)
        feature[f"{name}_max_width"] = max((d["w"] for d in dets), default=0.0)
        feature[f"{name}_max_height"] = max((d["h"] for d in dets), default=0.0)
        feature[f"{name}_area_ratio"] = feature[f"{name}_total_area"] / plate_area if plate_area > 0 else 0.0
        feature[f"{name}_max_conf"] = max((d["conf"] for d in dets), default=0.0)
        feature[f"{name}_mean_conf"] = (sum(d["conf"] for d in dets) / len(dets)) if dets else 0.0

    return feature


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "version": "v1",
        "seed": args.seed,
        "device": args.device,
        "conf_thres": args.conf_thres,
        "iou_thres": args.iou_thres,
        "resize_wh": args.resize,
        "tasks": {},
    }

    for task in args.tasks:
        cfg = TASK_CONFIG[task]
        src_root = args.dianjie_root if task == "dianjie" else args.dianji_root
        print(f"\n=== 处理任务: {cfg['name']} ({task}) ===")
        print(f"源数据: {src_root}")

        samples = collect_samples(task, src_root, cfg)
        print(f"共收集 {len(samples)} 张图片")
        if not samples:
            continue

        # 1) 建立软连接树
        link_records = build_symlink_tree(task, samples, output_root, args.skip_symlinks, args.dry_run)
        link_df = pd.DataFrame(link_records)
        link_csv = output_root / task / "link_records.csv"
        link_csv.parent.mkdir(parents=True, exist_ok=True)
        link_df.to_csv(link_csv, index=False, encoding="utf-8-sig")
        print(f"链接记录已保存: {link_csv}")

        if args.dry_run:
            print("--dry-run 模式：跳过推理与特征提取")
            metadata["tasks"][task] = {
                "num_samples": len(samples),
                "num_links": len(link_records),
                "weights": cfg["weights"],
                "class_folders": cfg["class_folders"],
                "label_map": cfg["label_map"],
            }
            continue

        # 2) 加载检测器
        weights_path = args.weights_dir / cfg["weights"]
        print(f"加载检测模型: {weights_path}")
        detector = load_detector(weights_path, args.device)
        class_names = list(detector.model.names.values())
        print(f"检测类别数: {len(class_names)} -> {class_names}")

        # 3) 提取特征
        features = []
        for s in tqdm(samples, desc=f"{task} 特征提取"):
            try:
                feat = extract_image_features(
                    s["image_path"],
                    detector,
                    tuple(args.resize),
                    args.conf_thres,
                    args.iou_thres,
                    class_names,
                )
                feat.update(s)
                features.append(feat)
            except Exception as e:
                warnings.warn(f"处理 {s['image_path']} 失败: {e}", stacklevel=2)

        feature_df = pd.DataFrame(features)
        feat_csv = output_root / task / "features.csv"
        feature_df.to_csv(feat_csv, index=False, encoding="utf-8-sig")
        print(f"特征矩阵已保存: {feat_csv}，形状={feature_df.shape}")

        metadata["tasks"][task] = {
            "num_samples": len(samples),
            "num_success": len(features),
            "num_links": len(link_records),
            "weights": cfg["weights"],
            "class_names": class_names,
            "class_folders": cfg["class_folders"],
            "label_map": cfg["label_map"],
            "feature_csv": str(feat_csv),
            "link_csv": str(link_csv),
        }

    # 保存元数据
    meta_path = output_root / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\n元数据已保存: {meta_path}")


if __name__ == "__main__":
    main()
