"""检测器适配器包。

- 换模型：改 models.yaml 里某任务的 kind / weights。
- 加模型：在本包内新建适配器文件，继承 BaseDetector 并 @register("名字")，
  再在下方 import 一行触发注册即可。

各适配器的重依赖（rfdetr、YOLOv5 加密加载等）都在适配器内部惰性 import，
因此导入本包本身不会因为某个框架没装而失败。
"""
from .base import BaseDetector, register, create_detector, available_detectors

# 导入具体适配器以触发注册
from . import yolov5        # noqa: F401,E402
from . import rfdetr        # noqa: F401,E402
from . import ultralytics   # noqa: F401,E402  # YOLOv8/v9/v10/v11/YOLO26
from . import multibackend  # noqa: F401,E402  # YOLOv5 DetectMultiBackend / ONNX / TensorRT

__all__ = ["BaseDetector", "register", "create_detector", "available_detectors"]
