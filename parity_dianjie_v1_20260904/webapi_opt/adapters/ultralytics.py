"""Ultralytics YOLO 检测器适配器（覆盖 YOLOv8 / v9 / v10 / v11 / YOLO26）。

ultralytics 包用同一个 YOLO() 类加载这些所有版本，所以一个适配器即可。
以后换版本或换权重，只需在 models.yaml 把 weights 指向新的 .pt 或 .onnx。
配置可使用通用 kind: ultralytics，也可使用 YOLO26 的显式别名 kind: yolo26 / yolov26。

依赖：uv add ultralytics（引擎在 .venv 的 site-packages，不在项目目录）。
输出契约：results.xyxy[0] 为 torch.Tensor，每行 [x1, y1, x2, y2, confidence, class_id]，
像素坐标在原图坐标系——与 YOLOv5/RF-DETR 适配器一致。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .base import BaseDetector, register


@register("ultralytics")
@register("yolo26")
@register("yolov26")
class UltralyticsDetector(BaseDetector):
    def __init__(self, weights, device="cpu", threshold=0.25, imgsz=None, iou=0.7, **_):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"找不到权重文件: {weights}")
        # 惰性导入：未安装 ultralytics 时不影响导入本模块
        from ultralytics import YOLO
        self.model = YOLO(str(weights), task="detect")
        self.device = device
        self.threshold = threshold
        self.imgsz = imgsz
        self.iou = iou
        # ultralytics 的类别名为 {idx: name}
        self.names = dict(getattr(self.model, "names", {}) or {})

    def __call__(self, image_path):
        kwargs = {"conf": self.threshold, "iou": self.iou,
                  "device": self.device, "verbose": False}
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        # 保留 ndarray 的 BGR 数据，避免业务层的内存直通路径退化为字符串。
        results = self.model(image_path, **kwargs)
        r = results[0]  # 单张图
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            rows = np.zeros((0, 6), dtype=np.float32)
        else:
            # 优先使用稳定属性，避免不同 Ultralytics/YOLO 版本调整 boxes.data 列布局。
            xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
            conf = boxes.conf.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
            cls = boxes.cls.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
            rows = np.concatenate([xyxy, conf, cls], axis=1)
        return SimpleNamespace(xyxy=[torch.from_numpy(rows)], names=self.names)
