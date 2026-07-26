#!/usr/bin/env python3
"""
等级分类器 API 封装。

用法示例（替换 start.py 中的硬编码逻辑）：
    from grade_classifier_api import GradeClassifierAPI
    grade_api = GradeClassifierAPI(model_root=r"E:/PythonProject/优化/classify_from_detection_v1/models")

    # 电解镍：检测模型输出 -> 等级分类器 -> 最终类型
    result = grade_api.predict("dianjie", image_path, detections=dets, plate_area=plate_area)
    # result = {"label_id": int, "label_name": str, "confidence": float, "probabilities": dict}
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

# 允许加载原项目 YOLOv5 权重
_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args: Any, **kwargs: Any) -> Any:
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

_ORIGINAL_PROJECT = Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi")


def _ensure_original_project_in_path() -> None:
    if str(_ORIGINAL_PROJECT) not in sys.path:
        sys.path.insert(0, str(_ORIGINAL_PROJECT))
    os.chdir(_ORIGINAL_PROJECT)


class GradeClassifierAPI:
    """同时管理检测模型与等级分类器，提供端到端的板级等级预测。"""

    def __init__(
        self,
        model_root: str | Path = r"E:/PythonProject/优化/classify_from_detection_v1/models",
        weights_dir: str | Path = _ORIGINAL_PROJECT,
        device: str = "cpu",
        resize_wh: tuple[int, int] = (1024, 888),
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ):
        self.model_root = Path(model_root)
        self.weights_dir = Path(weights_dir)
        self.device = device
        self.resize_wh = resize_wh
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self._clf_cache: dict[str, Any] = {}
        self._det_cache: dict[str, Any] = {}

    def _load_classifier(self, task: str) -> Any:
        if task not in self._clf_cache:
            path = self.model_root / f"{task}_lgbm.pkl"
            if not path.exists():
                raise FileNotFoundError(f"分类器不存在: {path}")
            self._clf_cache[task] = joblib.load(path)
        return self._clf_cache[task]

    def _load_detector(self, task: str) -> Any:
        if task not in self._det_cache:
            _ensure_original_project_in_path()
            from model_decryptor import load_yolov5_with_encryption
            weights = self.weights_dir / ("ryw.pt" if task == "dianjie" else "slf.pt")
            model = load_yolov5_with_encryption(str(weights), device=self.device, autoshape=True)
            model.conf = self.conf_thres
            model.iou = self.iou_thres
            self._det_cache[task] = model
        return self._det_cache[task]

    def _build_feature_vector(
        self,
        detections: np.ndarray,
        plate_area: float,
        img_w: int,
        img_h: int,
        class_names: list[str],
        feature_names: list[str],
    ) -> pd.DataFrame:
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

        base_df = pd.DataFrame([feature])
        missing = {col: 0.0 for col in feature_names if col not in base_df.columns}
        if missing:
            base_df = pd.concat([base_df, pd.DataFrame([missing])], axis=1)
        return base_df[feature_names]

    def predict(
        self,
        task: str,
        image_path: str | Path,
        detections: np.ndarray | None = None,
        plate_area: float | None = None,
    ) -> dict[str, Any]:
        """
        对单张图片预测板级等级。

        Args:
            task: "dianjie" 或 "dianji"
            image_path: 图片路径
            detections: 可选，N×6 检测框 [x1,y1,x2,y2,conf,cls]；不提供则内部调用检测模型
            plate_area: 可选，镍板像素面积；不提供则内部用 findcent 计算
        """
        _ensure_original_project_in_path()
        from preprocess import findcent

        artifact = self._load_classifier(task)
        clf = artifact["model"]
        feature_names = artifact["feature_names"]
        label_map = artifact["label_map"]
        id_to_name = {v: k for k, v in label_map.items()}

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        resized = cv2.resize(img, self.resize_wh)
        h, w = resized.shape[:2]

        if plate_area is None:
            try:
                _, pw, ph, _ = findcent(resized)
                plate_area = float(pw * ph)
            except Exception:
                plate_area = float(w * h)

        if detections is None:
            detector = self._load_detector(task)
            detector.conf = self.conf_thres
            detector.iou = self.iou_thres
            results = detector(resized)
            detections = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else np.empty((0, 6))

        class_names = list(self._load_detector(task).model.names.values())
        X = self._build_feature_vector(detections, plate_area, w, h, class_names, feature_names)
        pred_id = int(clf.predict(X)[0])
        proba = clf.predict_proba(X)[0]

        return {
            "label_id": pred_id,
            "label_name": id_to_name.get(pred_id, str(pred_id)),
            "confidence": float(np.max(proba)),
            "probabilities": {id_to_name.get(i, str(i)): float(p) for i, p in enumerate(proba)},
            "detections": detections.tolist(),
            "plate_area": plate_area,
        }


if __name__ == "__main__":
    import glob

    api = GradeClassifierAPI()
    sample = glob.glob(r"E:/PythonProject/datasets/classify_2types/valid/good/*.png")[0]
    print("样本:", sample)
    print(api.predict("dianjie", sample))
