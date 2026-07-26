#!/usr/bin/env python3
"""
统一等级判定 API（v2）：
- 电解镍：沿用原项目硬编码的多类规则（已提取到 dianjie_grade_rule.py）。
- 电积镍：基于检测特征 + ResNet18 整板图像特征的 LightGBM 4 类分类器。

设计目标：
1. 不改原数据集；
2. 电解镍保持企业认可的多类输出（type=0~6）；
3. 电积镍稳定输出 4 类（3#/9950/9996/bad）。

用法（替换 start.py 中的硬编码分支）：
    from grade_classifier_unified_api_v2 import UnifiedGradeAPI
    api = UnifiedGradeAPI()

    # 电解镍：返回 type_id=0~6 的多类结果
    result = api.predict_dianjie(image_path)

    # 电积镍：返回 type_id 映射后的 4 类结果
    result = api.predict_dianji(image_path)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from dianjie_grade_rule import DianjieGradeRule, dianjie_grade_from_xyxy

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

_ORIGINAL_PROJECT = Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi")
_MODEL_ROOT = Path(r"E:/PythonProject/优化/classify_from_detection_v1/models")

# 允许加载原项目 YOLOv5 加密权重
_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args: Any, **kwargs: Any) -> Any:
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false


def _ensure_original_project_in_path() -> None:
    import os
    import sys
    if str(_ORIGINAL_PROJECT) not in sys.path:
        sys.path.insert(0, str(_ORIGINAL_PROJECT))
    os.chdir(_ORIGINAL_PROJECT)


class ImageFeatureExtractor:
    """用 ResNet18 提取整板图像特征（512 维）。"""

    def __init__(self, device: str = "cpu", img_resize: tuple[int, int] = (224, 224)):
        from torchvision.models import resnet18, ResNet18_Weights

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.img_resize = img_resize
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        model = model.to(self.device)
        model.eval()
        self.model = model
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(img_resize),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def extract(self, image_path: str | Path) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            return np.zeros(512, dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor).squeeze().cpu().numpy()
        return feat


class DianjiImgFeatClassifier:
    """加载基于检测+图像特征的 LightGBM 分类器。"""

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = device
        self.artifact = joblib.load(self.model_path)
        self.model = self.artifact["model"]
        self.feature_names = self.artifact["feature_names"]
        self.label_map = self.artifact["label_map"]
        self.id_to_name = {v: k for k, v in self.label_map.items()}
        self.img_extractor = ImageFeatureExtractor(device=device)

    def predict(self, image_path: str | Path, feature_vector: pd.DataFrame) -> dict[str, Any]:
        img_feat = self.img_extractor.extract(image_path)
        # 特征名形如 img_feat_0 ... img_feat_511
        for i, val in enumerate(img_feat):
            col = f"img_feat_{i}"
            if col in self.feature_names:
                feature_vector[col] = float(val)

        X = feature_vector[self.feature_names]
        pred_id = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]
        return {
            "label_id": pred_id,
            "label_name": self.id_to_name.get(pred_id, str(pred_id)),
            "confidence": float(np.max(proba)),
            "probabilities": {self.id_to_name.get(i, str(i)): float(p) for i, p in enumerate(proba)},
        }


class UnifiedGradeAPI:
    """统一等级判定入口。"""

    # 电积镍输出映射：ResNet 类别名 -> 原项目 type_id
    DIANJI_TYPE_MAP = {
        "3#": 4,
        "9950": 3,
        "9996": 2,
        "bad": 1,
    }

    def __init__(
        self,
        dianji_model_path: str | Path = _MODEL_ROOT / "dianji_lgbm_imgfeat.pkl",
        weights_dir: str | Path = _ORIGINAL_PROJECT,
        device: str = "cpu",
        resize_wh: tuple[int, int] = (1024, 888),
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ):
        self.weights_dir = Path(weights_dir)
        self.device = device
        self.resize_wh = resize_wh
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.dianji_classifier = DianjiImgFeatClassifier(dianji_model_path, device=device)
        self.dianjie_rule = DianjieGradeRule()
        self._det_cache: dict[str, Any] = {}

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

    def _read_image(self, image_path: str | Path) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        return cv2.resize(img, self.resize_wh)

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
                per_class[class_names[cls_id]].append(
                    {
                        "area": area,
                        "w": float(x2 - x1),
                        "h": float(y2 - y1),
                        "conf": float(conf),
                    }
                )

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

    def predict_dianjie(self, image_path: str | Path) -> dict[str, Any]:
        """电解镍：检测 + 硬编码多类规则。"""
        _ensure_original_project_in_path()
        from preprocess import findcent

        resized = self._read_image(image_path)
        try:
            _, pw, ph, _ = findcent(resized)
            plate_area = float(pw * ph)
        except Exception:
            plate_area = float(self.resize_wh[0] * self.resize_wh[1])

        detector = self._load_detector("dianjie")
        results = detector(resized)
        detections = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else np.empty((0, 6))

        grade = dianjie_grade_from_xyxy(detections, plate_area, rule=self.dianjie_rule)

        output = []
        for det in detections:
            x1, y1, x2, y2, _, cls = det
            output.append(
                {
                    "label": int(cls),
                    "cx": float((x1 + x2) / 2),
                    "cy": float((y1 + y2) / 2),
                    "w": float(x2 - x1),
                    "h": float(y2 - y1),
                }
            )

        return {
            "code": 1,
            "msg": "success",
            "data": {
                "type": grade.type_id,
                "type_name": grade.type_name,
                "nodule": grade.nodule_status,
                "has_nodule": grade.has_nodule,
                "ratios": grade.ratios,
                "defects": output,
            },
        }

    def predict_dianji(self, image_path: str | Path) -> dict[str, Any]:
        """电积镍：检测特征 + ResNet18 图像特征 -> LightGBM 4 类。"""
        _ensure_original_project_in_path()
        from preprocess import findcent

        resized = self._read_image(image_path)
        try:
            _, pw, ph, _ = findcent(resized)
            plate_area = float(pw * ph)
        except Exception:
            plate_area = float(self.resize_wh[0] * self.resize_wh[1])

        detector = self._load_detector("dianji")
        results = detector(resized)
        detections = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else np.empty((0, 6))

        class_names = list(detector.model.names.values())
        feature_names = self.dianji_classifier.feature_names
        feature_df = self._build_feature_vector(detections, plate_area, resized.shape[1], resized.shape[0], class_names, feature_names)

        cls_result = self.dianji_classifier.predict(image_path, feature_df)
        type_id = self.DIANJI_TYPE_MAP.get(cls_result["label_name"], 1)

        # 电积镍 -> 电解镍显示映射（与原项目一致）
        mapping = {0: 1, 1: 0, 2: 2, 3: 3, 4: 5}
        output = []
        for det in detections:
            x1, y1, x2, y2, _, cls = det
            output.append(
                {
                    "label": mapping.get(int(cls), int(cls)),
                    "cx": float((x1 + x2) / 2),
                    "cy": float((y1 + y2) / 2),
                    "w": float(x2 - x1),
                    "h": float(y2 - y1),
                }
            )

        return {
            "code": 1,
            "msg": "success",
            "data": {
                "type": type_id,
                "type_name": cls_result["label_name"],
                "confidence": cls_result["confidence"],
                "probabilities": cls_result["probabilities"],
                "defects": output,
            },
        }


if __name__ == "__main__":
    import glob

    api = UnifiedGradeAPI()

    # 电积镍测试
    samples = glob.glob(r"E:/PythonProject/datasets/classify_4types_dianji/valid/*/*.png")
    if samples:
        sample = samples[0]
        print("电积镍样本:", sample)
        print(api.predict_dianji(sample))

    # 电解镍测试
    dianjie_samples = glob.glob(r"E:/PythonProject/datasets/classify_2types/valid/good/*.png")
    if dianjie_samples:
        sample = dianjie_samples[0]
        print("\n电解镍样本:", sample)
        print(api.predict_dianjie(sample))
