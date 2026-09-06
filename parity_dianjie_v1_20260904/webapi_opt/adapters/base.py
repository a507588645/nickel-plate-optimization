"""检测器适配器：统一接口 + 注册表 + 工厂。

统一调用契约（所有检测器适配器都满足）：
    detector(image_path) -> results
其中 results.xyxy[0] 是 torch.Tensor，每行为 [x1, y1, x2, y2, confidence, class_id]
（像素坐标，原图坐标系）。这样下游业务代码可统一写：
    results = detector(image_path)
    inference_result = results.xyxy[0]

新增模型支持：在本包内新建一个适配器文件，继承 BaseDetector，用 @register("名字") 注册，
然后在 models.yaml 里把某个任务的 kind 指向这个名字即可，业务代码无需改动。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

_REGISTRY = {}


def register(name):
    """类装饰器：把一个检测器适配器按名字注册到工厂。"""
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def create_detector(kind, **kwargs):
    """按 kind 创建检测器实例。kwargs 透传给适配器构造函数（weights/device/threshold 等）。"""
    if kind not in _REGISTRY:
        raise KeyError(f"未注册的检测器类型: {kind!r}，已注册: {sorted(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)


def available_detectors():
    return sorted(_REGISTRY)


class BaseDetector(ABC):
    """检测器适配器基类。子类必须实现 __call__(image_path) 并返回带 .xyxy[0] 的对象。"""

    @abstractmethod
    def __call__(self, image_path):
        raise NotImplementedError
