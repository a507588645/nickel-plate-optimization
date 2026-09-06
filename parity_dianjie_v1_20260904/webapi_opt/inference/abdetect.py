"""异常检测业务逻辑。"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from abdetect import is_dark_light_clear


logger = logging.getLogger("jinchuan.inference")


def _is_dark_light_clear_nd(
    image: np.ndarray,
    value1: int,
    value2: int,
    value3: int,
) -> int:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("image must be a grayscale, BGR, or BGRA ndarray")

    average_brightness = float(cv2.mean(gray)[0])
    image_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if image_var <= value3:
        return 3
    if average_brightness < value1:
        return 2
    if average_brightness > value2:
        return 1
    return 0


def ab_detect_nd(image: np.ndarray, value1: int, value2: int, value3: int) -> dict:
    """Run AB detection directly on an already decoded image."""
    try:
        type_ = _is_dark_light_clear_nd(image, value1, value2, value3)
        return {"code": 1, "msg": "success", "data": {"type": type_}}
    except Exception:
        logger.exception("abdetect_nd_failed")
        return {"code": 0, "msg": "abdetect failed", "data": {}}


def ab_detect(image_path: str | np.ndarray, value1: int, value2: int, value3: int) -> dict:
    """原 /abdetect 业务逻辑。"""
    if isinstance(image_path, np.ndarray):
        return ab_detect_nd(image_path, value1, value2, value3)

    try:
        type_ = is_dark_light_clear(image_path, value1, value2, value3)
        data = {"type": type_}
        return {"code": 1, "msg": "success", "data": data}
    except Exception:
        logger.exception("abdetect_failed")
        return {"code": 0, "msg": "abdetect failed", "data": {}}
