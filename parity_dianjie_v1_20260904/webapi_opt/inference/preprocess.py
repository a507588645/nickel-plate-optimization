"""图像预处理业务逻辑。"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import cv2
import numpy as np

from preprocess import findcent, xuanZhuan, xuanZhuan_and_flip


logger = logging.getLogger("jinchuan.inference")


class _PreprocessResponse(dict):
    """Dict-shaped response carrying the ndarray for the in-process pipeline."""

    def __init__(self, payload: dict, processed_image: np.ndarray):
        super().__init__(payload)
        self.processed_image = processed_image


def _should_flip(image_name: str, surface: str | None) -> bool:
    normalized_surface = str(surface or "").strip().lower()
    if normalized_surface and normalized_surface != "auto":
        return normalized_surface in {"bottom", "bottom_0", "back", "reverse", "1"}
    return "bottom_0" in image_name


def _as_bgr_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty ndarray")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("image must be a 2D or 3D ndarray")
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError("image must have 3 or 4 channels")


def _preprocess_nd_image(
    image: np.ndarray,
    out_imagedirpath: str,
    surface: str | None = None,
) -> tuple[dict, np.ndarray]:
    image = _as_bgr_image(image)
    output_dir = Path(out_imagedirpath)
    output_dir.mkdir(parents=True, exist_ok=True)

    resized = cv2.resize(image, (1024, 888))
    initial = findcent(resized)
    theta = initial[0]

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), theta, 1)
    rotated = cv2.warpAffine(image, matrix, (width, height))
    rotated = cv2.resize(rotated, (1024, 888))

    after = findcent(rotated)
    dx = 1024 / 2 - after[3][0]
    dy = 888 / 2 - after[3][1]
    move_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    centered = cv2.warpAffine(rotated, move_matrix, (1024, 888))
    if _should_flip("", surface):
        centered = cv2.flip(centered, 1)

    processed_image = cv2.resize(centered, (4096, 3554))
    output_path = output_dir / f"image_{uuid.uuid4().hex}.png"
    encoded_ok, encoded = cv2.imencode(".png", processed_image)
    if not encoded_ok:
        raise ValueError("failed to encode preprocessed image")
    encoded.tofile(output_path)

    data = {
        "theta": theta,
        "width": after[1],
        "height": after[2],
        "center": after[3],
        "out_image_path": str(output_path),
    }
    return data, processed_image


def preprocess_nd(
    image: np.ndarray,
    out_imagedirpath: str,
    surface: str | None = None,
) -> dict:
    """Preprocess an already decoded image and retain its ndarray in-process."""
    try:
        data, processed_image = _preprocess_nd_image(image, out_imagedirpath, surface)
        return _PreprocessResponse({"code": 1, "msg": "success", "data": data}, processed_image)
    except Exception:
        logger.exception("preprocess_nd_failed")
        return {"code": 0, "msg": "preprocess failed", "data": {}}


def preprocess_image(
    image_path: str | np.ndarray,
    out_imagedirpath: str,
    surface: str | None = None,
) -> dict:
    """原 /preprocess 业务逻辑。"""
    if isinstance(image_path, np.ndarray):
        return preprocess_nd(image_path, out_imagedirpath, surface)

    try:
        # 返回：[旋转角度，宽度，高度，镍板中心点(x,y)]
        if _should_flip(image_path, surface):
            result = xuanZhuan_and_flip(image_path, out_imagedirpath)
        else:
            result = xuanZhuan(image_path, out_imagedirpath)

        theta = result[0]
        width = result[1]
        height = result[2]
        center = result[3]
        save_path = os.path.join(out_imagedirpath, os.path.basename(image_path))

        data = {
            "theta": theta,
            "width": width,
            "height": height,
            "center": center,
            "out_image_path": save_path,
        }
        return {"code": 1, "msg": "success", "data": data}
    except Exception:
        logger.exception("preprocess_failed")
        return {"code": 0, "msg": "preprocess failed", "data": {}}
