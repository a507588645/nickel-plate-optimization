#!/usr/bin/env python3
"""
等级分类器推理接口。

输入：原图路径 + 检测模型输出（或调用检测模型）
输出：等级分类结果
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

# 同 build_grade_dataset.py：允许加载原项目 YOLOv5 权重
_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args: Any, **kwargs: Any) -> Any:
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

_ORIGINAL_PROJECT = Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi")
sys_path_inserted = False


def _ensure_original_project_in_path() -> None:
    global sys_path_inserted
    if not sys_path_inserted:
        sys.path.insert(0, str(_ORIGINAL_PROJECT))
        sys_path_inserted = True


class GradeClassifier:
    """加载 LightGBM 等级分类器，支持从检测框直接推理。"""

    def __init__(self, model_path: str | Path):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self.label_map = artifact["label_map"]
        self.n_classes = artifact["n_classes"]
        self.class_distribution = artifact.get("class_distribution", {})

        # 建立 id -> class_name 反向映射
        self.id_to_name = {v: k for k, v in self.label_map.items()}

    def _build_feature_vector(
        self,
        detections: np.ndarray,
        plate_area: float,
        img_w: int,
        img_h: int,
        class_names: list[str],
    ) -> pd.DataFrame:
        """从检测框构建与训练时一致的特征向量。"""
        total_defect_area = 0.0
        per_class = {name: [] for name in class_names}

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            area = float((x2 - x1) * (y2 - y1))
            total_defect_area += area
            cls_id = int(cls)
            if 0 <= cls_id < len(class_names):
                per_class[class_names[cls_id]].append({
                    "area": area,
                    "w": float(x2 - x1),
                    "h": float(y2 - y1),
                    "conf": float(conf),
                })

        feature: dict[str, Any] = {
            "img_w": img_w,
            "img_h": img_h,
            "plate_area": plate_area,
            "total_defect_count": len(detections),
            "total_defect_area": total_defect_area,
            "total_defect_area_ratio": total_defect_area / plate_area if plate_area > 0 else 0.0,
        }

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

        df = pd.DataFrame([feature])
        # 保证列顺序与训练一致，缺失列补 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.feature_names]

    def predict_from_detections(
        self,
        detections: np.ndarray,
        plate_area: float,
        img_w: int,
        img_h: int,
        class_names: list[str],
    ) -> dict[str, Any]:
        """给定检测框，直接输出分类结果。"""
        X = self._build_feature_vector(detections, plate_area, img_w, img_h, class_names)
        pred_id = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]
        return {
            "label_id": pred_id,
            "label_name": self.id_to_name.get(pred_id, str(pred_id)),
            "confidence": float(np.max(proba)),
            "probabilities": {
                self.id_to_name.get(i, str(i)): float(p)
                for i, p in enumerate(proba)
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="等级分类器推理示例",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="训练好的等级分类器路径（.pkl）",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="待推理图片路径",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=_ORIGINAL_PROJECT / "ryw.pt",
        help="原项目 YOLOv5 权重路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="推理设备",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[1024, 888],
        help="推理前 resize 尺寸 (宽 高)",
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
    return parser.parse_args()


def load_yolo(weights: Path, device: str) -> Any:
    _ensure_original_project_in_path()
    import os
    os.chdir(_ORIGINAL_PROJECT)
    from model_decryptor import load_yolov5_with_encryption
    model = load_yolov5_with_encryption(str(weights), device=device, autoshape=True)
    model.conf = 0.25
    model.iou = 0.45
    return model


def main() -> None:
    _ensure_original_project_in_path()

    args = parse_args()
    clf = GradeClassifier(args.model)
    print(f"加载分类器: {args.model}")
    print(f"标签映射: {clf.label_map}")

    # 加载检测模型
    detector = load_yolo(args.weights, args.device)
    class_names = list(detector.model.names.values())

    # 读取并 resize
    img = cv2.imread(str(args.image))
    if img is None:
        raise ValueError(f"无法读取图片: {args.image}")
    resized = cv2.resize(img, tuple(args.resize))
    h, w = resized.shape[:2]

    # 计算镍板面积
    import os
    os.chdir(_ORIGINAL_PROJECT)
    from preprocess import findcent
    try:
        _, plate_w, plate_h, _ = findcent(resized)
        plate_area = float(plate_w * plate_h)
    except Exception:
        plate_area = float(w * h)

    # 检测
    detector.conf = args.conf_thres
    detector.iou = args.iou_thres
    results = detector(resized)
    detections = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else np.empty((0, 6))

    result = clf.predict_from_detections(detections, plate_area, w, h, class_names)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
