"""模型加载与全局状态管理。

从原 start.py 中抽离，负责：
- 检测器/分类器/分割模型加载与卸载
- 暴露全局模型变量给业务逻辑模块
"""
from __future__ import annotations

import json
import logging
import os
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import yaml
from torchvision import transforms

from adapters import available_detectors, create_detector
from app_paths import resolve_project_path
from app_settings import get_app_settings
from model_decryptor import load_encrypted_torch
from segment.segpredict import loadmodel


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
if PARENT_ROOT not in sys.path:
    sys.path.append(PARENT_ROOT)


logger = logging.getLogger("jinchuan.inference")
APP_SETTINGS = get_app_settings()

# 全局模型实例
yolo = None
slf_yolo = None
resnet = None
slf_resnet = None
model = None
stride = None
pt = None
names = None
dianji_grade_model = None
dianji_grade_contract = None
dianji_grade_feature_names = None
dianji_grade_threshold = None
dianji_grade_best_iteration = None

# 自动检测 GPU 可用性
use_gpu = torch.cuda.is_available()
if use_gpu:
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    logger.info(
        "gpu_detected",
        extra={"event": "gpu_detected", "device_name": device_name, "device_count": device_count},
    )
else:
    logger.warning("gpu_not_available", extra={"event": "gpu_not_available"})


def _load_models_config() -> dict:
    """读取 models.yaml：定义每个检测任务用哪个模型/权重/阈值。"""
    cfg_path = resolve_project_path("models.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolved_path(path_value: str | None) -> str | None:
    path = resolve_project_path(path_value)
    return str(path) if path is not None else None


def _detector_kwargs(cfg: dict) -> dict:
    return dict(cfg.get("options") or {})


MODELS_CONFIG = _load_models_config()


class DianjiGradeContractError(RuntimeError):
    """电积镍等级模型与发布合同不一致。"""


def _clear_dianji_grade_model() -> None:
    global dianji_grade_model, dianji_grade_contract, dianji_grade_feature_names
    global dianji_grade_threshold, dianji_grade_best_iteration
    dianji_grade_model = None
    dianji_grade_contract = None
    dianji_grade_feature_names = None
    dianji_grade_threshold = None
    dianji_grade_best_iteration = None


def _read_dianji_grade_contract(contract_path: str) -> tuple[dict, tuple[str, ...], float, int]:
    with open(contract_path, "r", encoding="utf-8") as handle:
        contract = json.load(handle)

    feature_names = contract.get("feature_names")
    if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
        raise DianjiGradeContractError("电积镍等级合同缺少有效的 feature_names")
    if len(feature_names) != 63 or len(set(feature_names)) != 63:
        raise DianjiGradeContractError(f"电积镍等级合同特征数量无效: {len(feature_names)}")
    if contract.get("feature_count") != len(feature_names):
        raise DianjiGradeContractError("电积镍等级合同 feature_count 与 feature_names 不一致")

    model_contract = contract.get("model")
    if not isinstance(model_contract, dict):
        raise DianjiGradeContractError("电积镍等级合同缺少 model 配置")
    if model_contract.get("feature_names_in_model") != feature_names:
        raise DianjiGradeContractError("电积镍等级合同中的模型特征顺序不一致")
    if model_contract.get("num_class") != 4:
        raise DianjiGradeContractError("电积镍等级合同 num_class 必须为 4")

    grade_to_id = contract.get("grade_to_id")
    id_to_grade = contract.get("id_to_grade")
    grades = contract.get("grades")
    if (
        not isinstance(grade_to_id, dict)
        or not isinstance(id_to_grade, dict)
        or not isinstance(grades, list)
        or {str(grade_to_id.get(grade)) for grade in grades} != {"0", "1", "2", "3"}
        or any(id_to_grade.get(str(index)) != grade for index, grade in enumerate(grades))
    ):
        raise DianjiGradeContractError("电积镍等级合同类别映射无效")

    rejection = contract.get("rejection")
    if not isinstance(rejection, dict):
        raise DianjiGradeContractError("电积镍等级合同缺少 rejection 配置")
    threshold = float(rejection["threshold"])
    best_iteration = int(model_contract["best_iteration"])
    return contract, tuple(feature_names), threshold, best_iteration


def load_dianji_grade_model() -> None:
    """加载 LightGBM 等级模型，并拒绝与部署合同不匹配的模型文件。"""
    global dianji_grade_model, dianji_grade_contract, dianji_grade_feature_names
    global dianji_grade_threshold, dianji_grade_best_iteration
    _clear_dianji_grade_model()
    cfg = (MODELS_CONFIG.get("grade_models") or {}).get("dianji_lgbm")
    if not cfg or not cfg.get("enabled", True):
        logger.warning("dianji_grade_model_disabled", extra={"event": "dianji_grade_model_disabled"})
        return

    try:
        contract_path = _resolved_path(cfg["contract"])
        weight_path = _resolved_path(cfg["weights"])
        if contract_path is None or weight_path is None:
            raise FileNotFoundError("电积镍等级模型缺少权重或合同路径")
        contract, feature_names, contract_threshold, best_iteration = _read_dianji_grade_contract(contract_path)
        configured_threshold = float(cfg["threshold"])
        if configured_threshold != contract_threshold:
            raise DianjiGradeContractError(
                f"电积镍等级阈值与合同不一致: config={configured_threshold}, contract={contract_threshold}"
            )

        import lightgbm as lgb

        logger.info(
            "load_dianji_grade_model",
            extra={"event": "load_grade_model", "model_name": "dianji_grade_lgbm"},
        )
        booster = lgb.Booster(model_file=weight_path)
        model_feature_names = tuple(booster.feature_name())
        if model_feature_names != feature_names:
            raise DianjiGradeContractError(
                "电积镍 LightGBM 特征顺序不匹配: "
                f"expected={list(feature_names)!r}, actual={list(model_feature_names)!r}"
            )
        if booster.num_feature() != len(feature_names):
            raise DianjiGradeContractError(
                f"电积镍 LightGBM 特征数量不匹配: {booster.num_feature()} != {len(feature_names)}"
            )
        if booster.num_model_per_iteration() != contract["model"]["num_class"]:
            raise DianjiGradeContractError("电积镍 LightGBM 类别数量与合同不匹配")

        dianji_grade_model = booster
        dianji_grade_contract = contract
        dianji_grade_feature_names = feature_names
        dianji_grade_threshold = contract_threshold
        dianji_grade_best_iteration = best_iteration
    except DianjiGradeContractError:
        _clear_dianji_grade_model()
        logger.exception("dianji_grade_model_contract_invalid")
        raise
    except Exception:
        _clear_dianji_grade_model()
        logger.exception("dianji_grade_model_load_failed")


def load_dianjie_detector() -> None:
    """电解镍检测器：按 models.yaml 的 detectors.dianjie 配置创建。"""
    global yolo
    cfg = MODELS_CONFIG["detectors"]["dianjie"]
    device_str = "cuda" if use_gpu else "cpu"
    logger.info("load_dianjie_detector", extra={"event": "load_detector", "model_name": "dianjie_detector"})
    yolo = create_detector(
        cfg["kind"],
        weights=_resolved_path(cfg["weights"]),
        device=device_str,
        threshold=cfg.get("threshold", 0.25),
        **_detector_kwargs(cfg),
    )


def load_dianji_detector() -> None:
    """电积镍检测器：按 models.yaml 的 detectors.dianji 配置创建。"""
    global slf_yolo
    cfg = MODELS_CONFIG["detectors"]["dianji"]
    device_str = "cuda" if use_gpu else "cpu"
    logger.info("load_dianji_detector", extra={"event": "load_detector", "model_name": "dianji_detector"})
    slf_yolo = create_detector(
        cfg["kind"],
        weights=_resolved_path(cfg["weights"]),
        device=device_str,
        threshold=cfg.get("threshold", 0.25),
        **_detector_kwargs(cfg),
    )


def _validate_classifier_head(model, weight_path: str | None, expected_num_classes: int) -> None:
    """确认加载的分类器保留训练权重中的 fc 分类头。"""
    if not hasattr(model, "fc") or model.fc is None:
        raise RuntimeError(f"分类器权重 {weight_path} 缺少 fc 分类头")

    actual_num_classes = getattr(model.fc, "out_features", None)
    if actual_num_classes is None:
        raise RuntimeError(f"分类器权重 {weight_path} 的 fc 分类头缺少 out_features")

    if actual_num_classes != expected_num_classes:
        raise RuntimeError(
            f"分类器权重 {weight_path} 的 fc.out_features={actual_num_classes}，"
            f"但 models.yaml num_classes={expected_num_classes}"
        )


def load_resnet() -> None:
    global resnet
    cfg = MODELS_CONFIG["classifiers"]["dianjie_resnet"]
    logger.info("load_resnet", extra={"event": "load_classifier", "model_name": "dianjie_resnet"})
    weight_path = _resolved_path(cfg["weights"])
    resnet = load_encrypted_torch(weight_path, device="cuda" if use_gpu else "cpu")
    _validate_classifier_head(resnet, weight_path, cfg.get("num_classes", 2))

    resnet.eval()

    if use_gpu:
        resnet.cuda()


def load_slf_resnet() -> None:
    global slf_resnet
    cfg = MODELS_CONFIG["classifiers"]["dianji_resnet"]
    logger.info("load_slf_resnet", extra={"event": "load_classifier", "model_name": "dianji_resnet"})
    weight_path = _resolved_path(cfg["weights"])
    slf_resnet = load_encrypted_torch(weight_path, device="cuda" if use_gpu else "cpu")
    _validate_classifier_head(slf_resnet, weight_path, cfg.get("num_classes", 4))

    slf_resnet.eval()

    if use_gpu:
        slf_resnet.cuda()


def load_yolov5seg() -> None:
    global model, stride, pt, names
    cfg = MODELS_CONFIG["segments"]["nietiao"]
    logger.info("load_yolov5seg", extra={"event": "load_segment", "model_name": "nietiao_segment"})
    model, stride, pt, names = loadmodel(weights=_resolved_path(cfg["weights"]))


def load_all_models() -> None:
    load_dianjie_detector()
    load_dianji_grade_model()
    load_resnet()
    load_slf_resnet()
    load_dianji_detector()
    load_yolov5seg()


def unload_models() -> None:
    global yolo, slf_yolo, resnet, slf_resnet, model, stride, pt, names
    yolo = None
    slf_yolo = None
    resnet = None
    slf_resnet = None
    model = None
    stride = None
    pt = None
    names = None
    _clear_dianji_grade_model()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def classification_transform(image_path: str | np.ndarray):
    """分类模型通用预处理：读取、resize、normalize、unsqueeze、转 device。"""
    if isinstance(image_path, np.ndarray):
        if image_path.ndim == 2:
            pil_image = Image.fromarray(image_path).convert("RGB")
        elif image_path.ndim == 3 and image_path.shape[2] == 4:
            pil_image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGRA2RGB))
        elif image_path.ndim == 3 and image_path.shape[2] == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("image must be a grayscale, BGR, or BGRA ndarray")
    else:
        pil_image = Image.open(image_path)

    image = transforms.ToTensor()(pil_image)
    image = transforms.Resize((224, 224))(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image.unsqueeze(0))
    return image.to("cuda" if use_gpu else "cpu")
