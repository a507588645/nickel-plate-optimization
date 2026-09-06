"""推理业务逻辑包。

将原 start.py 中的模型加载与业务逻辑抽离到本包，供 FastAPI 路由层调用。
"""
from __future__ import annotations

from inference.abdetect import ab_detect, ab_detect_nd
from inference.cutting import huaxian
from inference.detectors import dianjie_detect, dianjie_detect_nd, dianji_detect, dianji_detect_nd
from inference.models import (
    available_detectors,
    load_all_models,
    load_dianji_detector,
    load_dianjie_detector,
    load_resnet,
    load_slf_resnet,
    load_yolov5seg,
    unload_models,
    use_gpu,
)
from inference.preprocess import preprocess_image, preprocess_nd
from inference.pipeline import detect_plate
from inference.segment import is_nieban, nietiao_segment

__all__ = [
    "ab_detect",
    "ab_detect_nd",
    "huaxian",
    "dianjie_detect",
    "dianjie_detect_nd",
    "dianji_detect",
    "dianji_detect_nd",
    "preprocess_image",
    "preprocess_nd",
    "detect_plate",
    "is_nieban",
    "nietiao_segment",
    "load_all_models",
    "load_dianjie_detector",
    "load_dianji_detector",
    "load_resnet",
    "load_slf_resnet",
    "load_yolov5seg",
    "unload_models",
    "available_detectors",
    "use_gpu",
]
