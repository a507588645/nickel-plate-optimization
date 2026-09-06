"""组合式镍板检测流程。"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from app_paths import runtime_dir
from app_settings import get_app_settings
from inference.abdetect import ab_detect, ab_detect_nd
from inference.detectors import (
    dianjie_detect,
    dianjie_detect_nd,
    dianji_detect,
    dianji_detect_nd,
)
from inference.preprocess import preprocess_image, preprocess_nd


logger = logging.getLogger("jinchuan.inference")


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("image body is empty")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("image body is not a supported image")
    return image


def _decode_image_path(image_path: str) -> np.ndarray:
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"image path does not exist: {path}")

    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"image path is not a supported image: {path}")
    return image


@contextmanager
def _materialize_input(
    image_bytes: bytes | None,
    image_path: str | None,
) -> Iterator[str | np.ndarray]:
    if (image_bytes is None) == (image_path is None):
        raise ValueError("provide exactly one of image_bytes or image_path")

    if image_path is not None:
        path = Path(image_path).expanduser()
        _decode_image_path(str(path))
        yield str(path)
        return

    yield _decode_image_bytes(image_bytes or b"")


def _prepare_output_dir(out_dir: str | None) -> str:
    if out_dir and str(out_dir).strip():
        output_dir = Path(out_dir).expanduser()
    else:
        output_dir = runtime_dir() / "detect_plate"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def _stage_data(result: object) -> dict:
    if not isinstance(result, dict):
        return {}
    data = result.get("data")
    return data if isinstance(data, dict) else {}


def _finish(
    started_at: float,
    timings: dict[str, float],
    code: int,
    msg: str,
    data: dict,
) -> dict:
    timings["total"] = round((time.perf_counter() - started_at) * 1000, 3)
    data["elapsed_ms"] = {name: round(value, 3) for name, value in timings.items()}
    return {"code": code, "msg": msg, "data": data}


def _failed_stage(result: object) -> bool:
    return not isinstance(result, dict) or result.get("code") != 1


def detect_plate(
    image_bytes: bytes | None,
    image_path: str | None,
    surface: str,
    plate_type: str,
    out_dir: str | None,
) -> dict:
    """Run AB detection, preprocessing, and plate classification in order."""
    started_at = time.perf_counter()
    timings = {
        "total": 0.0,
        "decode": 0.0,
        "abdetect": 0.0,
        "preprocess": 0.0,
        "classify": 0.0,
    }
    data = {"abdetect": {}, "preprocess": {}, "classify": {}}

    try:
        normalized_plate_type = str(plate_type or "").strip().lower()
        if normalized_plate_type == "dianjie":
            path_classifier = dianjie_detect
            ndarray_classifier = dianjie_detect_nd
        elif normalized_plate_type == "dianji":
            path_classifier = dianji_detect
            ndarray_classifier = dianji_detect_nd
        else:
            raise ValueError("plate_type must be dianji or dianjie")

        normalized_surface = str(surface or "auto").strip().lower()
        settings = get_app_settings()
        abdetect_rules = settings.business_rules.abdetect
        output_dir = _prepare_output_dir(out_dir)

        decode_started_at = time.perf_counter()
        with _materialize_input(image_bytes, image_path) as source:
            timings["decode"] = (time.perf_counter() - decode_started_at) * 1000
            is_ndarray = isinstance(source, np.ndarray)

            stage_started_at = time.perf_counter()
            abdetect_result = (ab_detect_nd if is_ndarray else ab_detect)(
                source,
                abdetect_rules.value1,
                abdetect_rules.value2,
                abdetect_rules.value3,
            )
            timings["abdetect"] = (time.perf_counter() - stage_started_at) * 1000
            data["abdetect"] = _stage_data(abdetect_result)
            if _failed_stage(abdetect_result):
                msg = abdetect_result.get("msg", "abdetect failed") if isinstance(abdetect_result, dict) else "abdetect failed"
                return _finish(started_at, timings, 0, str(msg), data)

            stage_started_at = time.perf_counter()
            preprocess_result = (
                preprocess_nd(source, output_dir, normalized_surface)
                if is_ndarray
                else preprocess_image(source, output_dir, normalized_surface)
            )
            timings["preprocess"] = (time.perf_counter() - stage_started_at) * 1000
            data["preprocess"] = _stage_data(preprocess_result)
            if _failed_stage(preprocess_result):
                msg = preprocess_result.get("msg", "preprocess failed") if isinstance(preprocess_result, dict) else "preprocess failed"
                return _finish(started_at, timings, 0, str(msg), data)

            stage_started_at = time.perf_counter()
            if is_ndarray:
                processed_image = getattr(preprocess_result, "processed_image", None)
                if processed_image is None and isinstance(preprocess_result, dict):
                    processed_image = preprocess_result.get("_image")
                if not isinstance(processed_image, np.ndarray):
                    raise ValueError("preprocess did not return a processed ndarray")
                classify_result = ndarray_classifier(processed_image)
            else:
                processed_path = data["preprocess"].get("out_image_path")
                if not processed_path:
                    raise ValueError("preprocess did not return out_image_path")
                classify_result = path_classifier(str(processed_path))
            timings["classify"] = (time.perf_counter() - stage_started_at) * 1000
            data["classify"] = _stage_data(classify_result)
            if _failed_stage(classify_result):
                msg = classify_result.get("msg", "classify failed") if isinstance(classify_result, dict) else "classify failed"
                return _finish(started_at, timings, 0, str(msg), data)

        return _finish(started_at, timings, 1, "success", data)
    except Exception as exc:
        logger.exception("plate_detection_pipeline_failed")
        return _finish(started_at, timings, 0, str(exc), data)
