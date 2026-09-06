"""YOLOv5 检测器适配器。

包装项目原有的加密加载 + AutoShape。AutoShape 本身就返回 results.xyxy[0]，
所以 __call__ 直接委托给它即可。
"""
from __future__ import annotations

from .base import BaseDetector, register


@register("yolov5")
class YoloV5Detector(BaseDetector):
    def __init__(self, weights, device="cpu", threshold=None, autoshape=True, **_):
        # 惰性 import：model_decryptor 依赖 pycryptodome，仅在真正加载 YOLOv5 时才需要
        from model_decryptor import load_yolov5_with_encryption
        self.model = load_yolov5_with_encryption(weights, device=device, autoshape=autoshape)
        self.model.eval()
        if device != "cpu":
            self.model.to(device)
        # YOLOv5 的置信度阈值由 AutoShape.conf 控制
        if threshold is not None:
            try:
                self.model.conf = float(threshold)
            except Exception:
                pass

    def __call__(self, image_path, size=None):
        if size is None:
            return self.model(image_path)
        return self.model(image_path, size=size)
