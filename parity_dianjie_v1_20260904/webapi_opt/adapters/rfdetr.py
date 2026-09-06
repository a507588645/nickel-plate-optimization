"""RF-DETR 检测器适配器：把 RF-DETR 输出适配成 YOLOv5 results.xyxy[0] 形状。

每行 [x1, y1, x2, y2, confidence, class_id]，torch.Tensor，像素坐标在原图坐标系。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .base import BaseDetector, register


def _class_names(model):
    names = getattr(model, "class_names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "class_names", None)
    return list(names or [])


@register("rfdetr")
class RFDETRDetector(BaseDetector):
    def __init__(self, weights, device="cpu", threshold=0.25, shape=None, **_):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"找不到 RF-DETR 权重文件: {weights}")
        # 惰性 import：未安装 rfdetr 时不影响导入本模块
        from rfdetr import RFDETR
        self.model = RFDETR.from_checkpoint(str(weights))
        self.threshold = threshold
        # shape=None 时保持原图分辨率，输出像素坐标在原图坐标系（业务判级需要全分辨率坐标）
        self.shape = (shape, shape) if shape is not None else None
        self.names = {i: n for i, n in enumerate(_class_names(self.model))}

    def __call__(self, image_path):
        det = self.model.predict(
            str(image_path),
            threshold=self.threshold,
            shape=self.shape,
            include_source_image=False,
        )
        xyxy = np.asarray(det.xyxy, np.float32).reshape(-1, 4)
        conf = (np.ones((xyxy.shape[0], 1), np.float32) if det.confidence is None
                else np.asarray(det.confidence, np.float32).reshape(-1, 1))
        cls = (np.zeros((xyxy.shape[0], 1), np.float32) if det.class_id is None
               else np.asarray(det.class_id, np.float32).reshape(-1, 1))
        rows = np.concatenate([xyxy, conf, cls], 1) if xyxy.size else np.zeros((0, 6), np.float32)
        return SimpleNamespace(xyxy=[torch.from_numpy(rows)], names=self.names)
