#!/usr/bin/env python3
"""
统一等级判定 API：
- 电解镍：沿用原项目硬编码的多类规则（已提取到 dianjie_grade_rule.py）。
- 电积镍：使用在 folder 4 类数据上微调的 ResNet18。

设计目标：
1. 不改原数据集；
2. 电解镍保持企业认可的多类输出；
3. 电积镍稳定输出 4 类（3#/9950/9996/bad）。

用法（替换 start.py 中的硬编码分支）：
    from grade_classifier_unified_api import UnifiedGradeAPI
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
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from dianjie_grade_rule import DianjieGradeRule, dianjie_grade_from_xyxy

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

_ORIGINAL_PROJECT = Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi")
_RESNET_MODEL_DIR = Path(r"E:/PythonProject/优化/classify_from_detection_v1/models/resnet_dianji_weighted_os")

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


class ResNetFolderClassifier:
    """加载 ImageFolder 上微调的 ResNet18 模型。"""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cpu",
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self._artifact: dict[str, Any] | None = None
        self._model: torch.nn.Module | None = None
        self.class_names: list[str] = []
        self.label_map: dict[str, int] = {}
        self.img_size: int = 224
        self._load()

    def _load(self) -> None:
        artifact_path = self.model_dir / "best_model.pth"
        if not artifact_path.exists():
            raise FileNotFoundError(f"ResNet 模型不存在: {artifact_path}")
        self._artifact = torch.load(artifact_path, map_location=self.device, weights_only=False)
        from torchvision.models import resnet18

        self.class_names = self._artifact["class_names"]
        self.label_map = self._artifact["label_map"]
        self.img_size = self._artifact.get("img_size", 224)
        num_classes = self._artifact["num_classes"]

        model = resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(self._artifact["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        self._model = model

    @property
    def transform(self) -> T.Compose:
        return T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        return {
            "label_id": pred_id,
            "label_name": self.class_names[pred_id],
            "confidence": float(probs[pred_id]),
            "probabilities": {name: float(probs[i]) for i, name in enumerate(self.class_names)},
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
        dianji_model_dir: str | Path = _RESNET_MODEL_DIR,
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
        self.dianji_classifier = ResNetFolderClassifier(dianji_model_dir, device=device)
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

        # 兼容原项目输出字段
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
        """电积镍：ResNet folder 分类器 + 检测框输出。"""
        resized = self._read_image(image_path)
        detector = self._load_detector("dianji")
        results = detector(resized)
        detections = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else np.empty((0, 6))

        cls_result = self.dianji_classifier.predict(image_path)
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
