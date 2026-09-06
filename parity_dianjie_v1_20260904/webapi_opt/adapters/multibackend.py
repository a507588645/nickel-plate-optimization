"""YOLOv5 DetectMultiBackend 适配器。

支持 *.pt / *.onnx / *.engine 等权重格式，方便后续切到 ONNX Runtime
或 TensorRT，而不用改业务判级代码。
"""
from __future__ import annotations

from pathlib import Path

import torch

from .base import BaseDetector, register


@register("multibackend")
@register("onnxruntime")
@register("tensorrt")
class MultiBackendDetector(BaseDetector):
    def __init__(self, weights, device="cpu", threshold=0.25, iou=0.45,
                 autoshape=True, dnn=False, fp16=False, data=None, **_):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"找不到多后端权重文件: {weights}")

        from models.common import AutoShape, DetectMultiBackend

        backend = DetectMultiBackend(
            str(weights),
            device=torch.device(device),
            dnn=dnn,
            data=data,
            fp16=fp16,
        )
        if autoshape:
            backend = AutoShape(backend)
            backend.conf = float(threshold)
            backend.iou = float(iou)

        self.model = backend

    def __call__(self, image_path, size=None):
        if size is None:
            return self.model(image_path)
        return self.model(image_path, size=size)
