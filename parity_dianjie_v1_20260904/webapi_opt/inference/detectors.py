"""电解镍 / 电积镍检测业务逻辑。"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

from app_settings import get_app_settings
from inference.grade_features import extract_grade_features
from inference import models as model_state
from inference.models import classification_transform
from preprocess import (
    read_A,
    read_B,
    read_C,
    read_airhole,
    readimg,
    scalculate,
    findcent,
)


logger = logging.getLogger("jinchuan.inference")
BUSINESS_RULES = get_app_settings().business_rules
DIANJI_GRADE_TO_TYPE = {"3#": 4, "9950": 3, "9996": 2, "bad": 1}
DIANJI_YOLO_CLASS_COUNT = 18


def _readimg(image: str | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return cv2.resize(image, (1024, 888))
    return readimg(image)


def dianjie_detect(image_path: str | np.ndarray) -> dict:
    """电解镍检测（原 /dianjiEclass）。"""
    return _dianjie_detect(image_path)


def dianjie_detect_nd(image: np.ndarray) -> dict:
    """电解镍检测，直接接收已解码的 BGR ndarray。"""
    return _dianjie_detect(image)


def _dianjie_detect(image: str | np.ndarray) -> dict:
    image1 = _readimg(image)  # (4096,3554) 转为 (1024,888)

    image_tensor = classification_transform(image)

    try:
        with torch.no_grad():
            outputs = model_state.resnet(image_tensor)
            _, predicted_idx = torch.max(outputs, 1)

        # 使用 models.yaml 配置的检测器进行推理
        results = model_state.yolo(image)
        inference_result = results.xyxy[0]

        output = []
        flag_wuzi_watersatin = True
        flag_nodule = False
        for detection in inference_result:
            label = int(detection[5])

            # 判断是否有边部结瘤，若有 nodule 标志为负，意味着直接归为废板
            if label == 3:
                nodule_width = detection[2] - detection[0]
                nodule_height = detection[3] - detection[1]
                ratio = (
                    nodule_height / nodule_width
                    if nodule_height > nodule_width
                    else nodule_width / nodule_height
                )
                if (
                    nodule_width.item() * nodule_height.item() < BUSINESS_RULES.dianjie.nodule.ignore_area_below
                    and ratio < BUSINESS_RULES.dianjie.nodule.ignore_aspect_ratio_below
                ):
                    continue
                else:
                    flag_nodule = True

            # 判断有没有除了水印、污渍和边部结瘤以外的缺陷
            if flag_wuzi_watersatin:
                if label in BUSINESS_RULES.dianjie.non_water_labels:
                    flag_wuzi_watersatin = False

            cx = (detection[0] + detection[2]) / 2
            cy = (detection[1] + detection[3]) / 2
            w = detection[2] - detection[0]
            h = detection[3] - detection[1]

            label_dict = {
                "label": label,
                "cx": cx.item(),
                "cy": cy.item(),
                "w": w.item(),
                "h": h.item(),
            }
            output.append(label_dict)

        result = findcent(image1)
        width = result[1]
        height = result[2]
        nieban_area = width * height

        A_area = read_A(output)
        B_area = read_B(output)
        C_area = read_C(output)
        Airhole_area = read_airhole(output)

        A_area = scalculate(A_area)
        B_area = scalculate(B_area)
        C_area = scalculate(C_area)
        Airhole_area = scalculate(Airhole_area)

        A_ratio = A_area / nieban_area
        B_ratio = B_area / nieban_area
        C_ratio = C_area / nieban_area
        Airhole_ratio = Airhole_area / nieban_area

        logger.debug("dianjie_flag_nodule", extra={"event": "dianjie_flag_nodule", "flag_nodule": flag_nodule})

        # 若无边部结瘤且 A 类缺陷占比 20% 以下，B 类 C 类占比 5% 以下，则返回合格板
        if (
            not flag_nodule
            and A_ratio < BUSINESS_RULES.dianjie.good_board.a_ratio_max
            and B_ratio < BUSINESS_RULES.dianjie.good_board.b_ratio_max
            and Airhole_ratio < BUSINESS_RULES.dianjie.good_board.airhole_ratio_max
            and C_ratio < BUSINESS_RULES.dianjie.good_board.c_ratio_max
        ):
            return {"code": 1, "msg": "success", "data": {"type": 3, "nodule": 0, "defects": output}}

        # 若有边部结瘤且分类结果为合格，返回 3 号板
        if flag_nodule and predicted_idx == 1:
            return {"code": 1, "msg": "success", "data": {"type": 1, "nodule": 1, "defects": output}}

        # 以下为分类为坏板的部分，若没有除水印和污渍以外的缺陷，则判断是否需要烫洗
        if (
            not flag_nodule
            and A_ratio >= BUSINESS_RULES.dianjie.hot_wash.a_ratio_min
            and B_ratio < BUSINESS_RULES.dianjie.hot_wash.b_ratio_max
            and C_ratio < BUSINESS_RULES.dianjie.hot_wash.c_ratio_max
        ):
            return {"code": 1, "msg": "success", "data": {"type": 2, "nodule": 2, "defects": output}}

        # 若 C 类缺陷占比大于 50%，则为 3 号板
        if C_ratio > BUSINESS_RULES.dianjie.c_ratio_bad_threshold:
            return {"code": 1, "msg": "success", "data": {"type": 1, "nodule": 3, "defects": output}}

        # 若 B 类缺陷占比大于 50%，C 类占比小于 50%，则为 2 号板
        if (
            B_ratio > BUSINESS_RULES.dianjie.b_ratio_bad_threshold
            and C_ratio < BUSINESS_RULES.dianjie.c_ratio_bad_threshold
        ):
            return {"code": 1, "msg": "success", "data": {"type": 6, "nodule": 3, "defects": output}}

        # 其他情况，则输出合格 600，等待最终的处理方案
        return {"code": 1, "msg": "success", "data": {"type": 4, "nodule": 3, "defects": output}}

    except Exception:
        logger.exception("dianjie_detection_failed")
        return {"code": 0, "msg": "dianjie detection failed", "data": {}}


def dianji_detect(image_path: str | np.ndarray) -> dict:
    """电积镍检测（原 /dianjiclass）。"""
    return _dianji_detect(image_path)


def dianji_detect_nd(image: np.ndarray) -> dict:
    """电积镍检测，直接接收已解码的 BGR ndarray。"""
    return _dianji_detect(image)


def _read_dianji_grade_image(image: str | np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("电积镍等级输入必须是 BGR 三通道 ndarray")
        return image, None

    encoded = np.fromfile(image, dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"无法读取电积镍图像: {image}")
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"无法解码电积镍图像: {image}")
    image_gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise ValueError(f"无法解码电积镍灰度图像: {image}")
    return image_bgr, image_gray


def _legacy_dianji_type(image_bgr: np.ndarray) -> int:
    """保留历史 ResNet 分支，供等级模型不可用时降级。"""
    image_tensor = classification_transform(image_bgr)
    with torch.no_grad():
        # 注意：原代码此处使用的是 resnet，而非 slf_resnet
        outputs = model_state.resnet(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)

    predicted = int(predicted_idx.item())
    if predicted == 2:
        return 2
    if predicted == 1:
        return 3
    if predicted == 0:
        return 4
    return 1


def _grade_detector_names(detector) -> list[str]:
    raw_names = getattr(detector, "names", None)
    if raw_names is None:
        raw_names = getattr(getattr(detector, "model", None), "names", None)
    if isinstance(raw_names, dict):
        return [str(raw_names.get(index, raw_names.get(str(index), ""))) for index in range(len(raw_names))]
    return [str(name) for name in raw_names] if raw_names is not None else []


def _dianji_yolo_extraction() -> dict:
    contract = model_state.dianji_grade_contract
    if isinstance(contract, dict) and isinstance(contract.get("extraction"), dict):
        return contract["extraction"]

    # 等级模型不可用时仍使用部署检测器配置完成前端显示推理。
    detector_config = (model_state.MODELS_CONFIG.get("detectors") or {}).get("dianji") or {}
    options = detector_config.get("options") or {}
    return {
        "imgsz": options.get("imgsz", 640),
        "conf": detector_config.get("threshold", 0.25),
        "iou": 0.45,
    }


def _predict_dianji_yolo_raw(image_bgr: np.ndarray) -> list[dict]:
    detector = model_state.slf_yolo
    if detector is None:
        raise RuntimeError("电积镍 YOLO11m 检测器未加载")

    extraction = _dianji_yolo_extraction()
    yolo_model = getattr(detector, "model", None)
    if yolo_model is None or not hasattr(yolo_model, "predict"):
        raise RuntimeError("已加载的 YOLO11m 检测器不支持原始 predict 接口")
    results = yolo_model.predict(
        source=image_bgr,
        imgsz=int(extraction["imgsz"]),
        conf=float(extraction["conf"]),
        iou=float(extraction["iou"]),
        device="cpu",
        verbose=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"YOLO11m 单图推理结果数量异常: {len(results)}")
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy()
    detections = []
    for coordinates, confidence, raw_class_id in zip(xyxy, confidences, class_ids):
        class_id = int(raw_class_id)
        if class_id < 0 or class_id >= DIANJI_YOLO_CLASS_COUNT:
            raise ValueError(f"YOLO11m 输出非法类别 id: {class_id}")
        x1, y1, x2, y2 = (float(value) for value in coordinates[:4])
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0 or height <= 0:
            continue
        detections.append(
            {
                "label": class_id,
                "conf": float(confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return detections


def _predict_dianji_grade_detections(
    image_bgr: np.ndarray,
    raw_detections: list[dict] | None = None,
) -> list[dict]:
    contract = model_state.dianji_grade_contract
    detector = model_state.slf_yolo
    if contract is None or detector is None:
        raise RuntimeError("电积镍等级模型或 YOLO11m 检测器未加载")
    extraction = contract["extraction"]
    expected_names = [str(name) for name in extraction["class_names"]]
    actual_names = _grade_detector_names(detector)
    if actual_names and actual_names != expected_names:
        raise RuntimeError(f"YOLO11m 类别顺序不匹配: expected={expected_names!r}, actual={actual_names!r}")

    if raw_detections is None:
        raw_detections = _predict_dianji_yolo_raw(image_bgr)

    image_height, image_width = image_bgr.shape[:2]
    detections = []
    for raw_detection in raw_detections:
        class_id = int(raw_detection["label"])
        if class_id < 0 or class_id >= len(expected_names):
            raise ValueError(f"YOLO11m 输出非法类别 id: {class_id}")
        x1 = float(raw_detection["x1"])
        y1 = float(raw_detection["y1"])
        x2 = float(raw_detection["x2"])
        y2 = float(raw_detection["y2"])
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0 or height <= 0:
            continue
        detections.append(
            {
                "label": class_id,
                "conf": float(raw_detection["conf"]),
                "cx": (x1 + x2) / 2.0 / image_width,
                "cy": (y1 + y2) / 2.0 / image_height,
                "w": width / image_width,
                "h": height / image_height,
            }
        )
    return detections


def _dianji_display_detections(raw_detections: list[dict]) -> list[dict]:
    # YOLO11m 的 18 类 id 直接透传，前端使用 RYW 类别空间。
    output = []
    for raw_detection in raw_detections:
        x1 = float(raw_detection["x1"])
        y1 = float(raw_detection["y1"])
        x2 = float(raw_detection["x2"])
        y2 = float(raw_detection["y2"])
        output.append(
            {
                "label": int(raw_detection["label"]),
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "w": x2 - x1,
                "h": y2 - y1,
            }
        )
    return output


def _dianji_lgbm_grade(
    image_bgr: np.ndarray,
    image_gray: np.ndarray | None = None,
    grade_detections: list[dict] | None = None,
) -> dict:
    model = model_state.dianji_grade_model
    contract = model_state.dianji_grade_contract
    feature_names = model_state.dianji_grade_feature_names
    threshold = model_state.dianji_grade_threshold
    best_iteration = model_state.dianji_grade_best_iteration
    if any(value is None for value in (model, contract, feature_names, threshold, best_iteration)):
        raise RuntimeError("电积镍 LightGBM 等级模型未加载")

    detections = (
        grade_detections
        if grade_detections is not None
        else _predict_dianji_grade_detections(image_bgr)
    )
    features = extract_grade_features(image_bgr, detections, feature_names, image_gray=image_gray)
    feature_row = np.asarray([[features[name] for name in feature_names]], dtype=np.float64)
    probabilities = np.asarray(model.predict(feature_row, num_iteration=int(best_iteration)), dtype=np.float64).reshape(-1)
    if probabilities.size != 4 or not np.isfinite(probabilities).all():
        raise RuntimeError(f"电积镍 LightGBM 概率输出异常: {probabilities!r}")

    grade_id = int(np.argmax(probabilities))
    grade = str(contract["id_to_grade"].get(str(grade_id), ""))
    if grade not in DIANJI_GRADE_TO_TYPE:
        raise RuntimeError(f"电积镍 LightGBM 输出未知等级: {grade_id}")
    confidence = float(probabilities[grade_id])
    return {
        "type": DIANJI_GRADE_TO_TYPE[grade],
        "grade": grade,
        "confidence": confidence,
        "manual_review": confidence < float(threshold),
        "grade_status": str(contract["status"]),
        "grade_source": "lgbm_y11m640_v1",
    }


def _fallback_dianji_grade(image_bgr: np.ndarray) -> dict:
    try:
        type_ = _legacy_dianji_type(image_bgr)
    except Exception:
        logger.exception("dianji_resnet_fallback_failed")
        type_ = 1
    return {
        "type": type_,
        "grade": "unknown",
        "confidence": 0.0,
        "manual_review": True,
        "grade_status": "fallback",
        "grade_source": "resnet_fallback",
    }


def _dianji_detect(image: str | np.ndarray) -> dict:
    try:
        image_bgr, image_gray = _read_dianji_grade_image(image)
        raw_detections = _predict_dianji_yolo_raw(image_bgr)
        output = _dianji_display_detections(raw_detections)
        try:
            grade_detections = _predict_dianji_grade_detections(image_bgr, raw_detections)
            grade_data = _dianji_lgbm_grade(image_bgr, image_gray, grade_detections)
        except Exception:
            logger.exception("dianji_lgbm_grade_failed")
            grade_data = _fallback_dianji_grade(image_bgr)

        return {
            "code": 1,
            "msg": "success",
            "data": {"type": grade_data["type"], "defects": output, **grade_data},
        }

    except Exception:
        logger.exception("dianji_detection_failed")
        return {"code": 0, "msg": "dianji detection failed", "data": {}}
