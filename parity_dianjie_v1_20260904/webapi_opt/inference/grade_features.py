"""电积镍 LightGBM 等级模型的特征提取。"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from preprocess import findcent


CLASS_NAMES = (
    "wuzi",
    "waterstain",
    "yellowstain",
    "nodule",
    "airhole",
    "bubble",
    "bump",
    "fawu",
    "overfire",
    "NiOH",
    "greenstain",
    "suanwuran",
    "quejiao",
    "watermark",
    "youpao",
    "heibian",
    "mijixiaokeng",
    "fanxitong",
)
A_LABELS = {0, 1}
B_LABELS = {2, 4, 5, 6, 10, 12, 13}
C_LABELS = {7, 8, 11}
AIRHOLE_LABELS = {4}
NODULE_LABEL = 3
DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "weights" / "dianji_grade_lgbm_v1_contract.json"


@lru_cache(maxsize=1)
def default_feature_names() -> tuple[str, ...]:
    """从随模型发布的合同取得唯一有效的特征顺序。"""
    with DEFAULT_CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    feature_names = contract.get("feature_names")
    if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
        raise ValueError(f"等级模型合同缺少有效 feature_names: {DEFAULT_CONTRACT_PATH}")
    return tuple(feature_names)


def union_area(rectangles: Iterable[tuple[float, float, float, float]]) -> float:
    rects = [(x1, y1, x2, y2) for x1, y1, x2, y2 in rectangles if x2 > x1 and y2 > y1]
    if not rects:
        return 0.0

    xs = sorted({value for rect in rects for value in (rect[0], rect[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted((y1, y2) for x1, y1, x2, y2 in rects if x1 < right and x2 > left)
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                covered += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        covered += end - start
        total += (right - left) * covered
    return float(total)


def normalized_rect(detection: dict[str, Any]) -> tuple[float, float, float, float]:
    cx = float(detection["cx"])
    cy = float(detection["cy"])
    width = float(detection["w"])
    height = float(detection["h"])
    return (
        max(0.0, cx - width / 2),
        max(0.0, cy - height / 2),
        min(1.0, cx + width / 2),
        min(1.0, cy + height / 2),
    )


def hardcoded_rect(
    detection: dict[str, Any], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    cx = float(detection["cx"]) * image_width
    cy = float(detection["cy"]) * image_height
    width = float(detection["w"]) * image_width
    height = float(detection["h"]) * image_height
    return (
        max(0, int((cx - width / 2) / 4)),
        max(0, int((cy - height / 2) / 4)),
        min(4069, int((cx + width / 2) / 4)),
        min(3554, int((cy + height / 2) / 4)),
    )


def compute_plate_area(image_bgr: np.ndarray) -> float:
    height, width = image_bgr.shape[:2]
    fallback = float(width * height)
    try:
        resized = cv2.resize(image_bgr, (1024, 888))
        _, plate_width, plate_height, _ = findcent(resized)
        area = float(plate_width * plate_height)
        return area if area > 0 else fallback
    except Exception:
        return fallback


def _base_features(
    detections: Sequence[dict[str, Any]], image_width: int, image_height: int, plate_area: float
) -> dict[str, float]:
    kept = [detection for detection in detections if float(detection.get("conf", 0.0)) >= 0.25]
    has_nodule = False
    for detection in kept:
        if int(detection["label"]) != NODULE_LABEL:
            continue
        width = float(detection["w"]) * image_width
        height = float(detection["h"]) * image_height
        aspect = max(width / height, height / width) if min(width, height) > 0 else 1.0
        if width * height >= 13000.0 or aspect >= 3.0:
            has_nodule = True

    def group_ratio(labels: set[int]) -> float:
        rects = [
            hardcoded_rect(detection, image_width, image_height)
            for detection in kept
            if int(detection["label"]) in labels
        ]
        if plate_area <= 0:
            return 0.0
        area = union_area((float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in rects)
        return float(area / plate_area)

    all_normalized = [normalized_rect(detection) for detection in kept]
    per_class_rects: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    per_class_counts: Counter[int] = Counter()
    for detection in kept:
        class_id = int(detection["label"])
        per_class_counts[class_id] += 1
        per_class_rects[class_id].append(normalized_rect(detection))

    features: dict[str, float] = {
        "A_ratio": group_ratio(A_LABELS),
        "B_ratio": group_ratio(B_LABELS),
        "C_ratio": group_ratio(C_LABELS),
        "Airhole_ratio": group_ratio(AIRHOLE_LABELS),
        "has_nodule": float(int(has_nodule)),
        "total_bbox_union_image_ratio": union_area(all_normalized),
        "detection_count": float(len(kept)),
    }
    for class_id, class_name in enumerate(CLASS_NAMES):
        features[f"count__{class_name}"] = float(per_class_counts.get(class_id, 0))
        features[f"area_ratio__{class_name}"] = union_area(per_class_rects.get(class_id, []))
    return features


def resize_short_side(gray: np.ndarray, short_side: int = 1024) -> np.ndarray:
    height, width = gray.shape[:2]
    current_short_side = min(height, width)
    if current_short_side <= short_side:
        return gray
    scale = short_side / float(current_short_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(gray, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def lbp_hist_10bin(gray: np.ndarray) -> np.ndarray:
    gray_int = gray.astype(np.int16)
    height, width = gray_int.shape
    if height < 3 or width < 3:
        return np.zeros(10, dtype=np.float64)
    center = gray_int[1:-1, 1:-1]
    codes = np.zeros_like(center, dtype=np.uint8)
    neighbors = (
        (gray_int[0:-2, 0:-2], 0),
        (gray_int[0:-2, 1:-1], 1),
        (gray_int[0:-2, 2:], 2),
        (gray_int[1:-1, 2:], 3),
        (gray_int[2:, 2:], 4),
        (gray_int[2:, 1:-1], 5),
        (gray_int[2:, 0:-2], 6),
        (gray_int[1:-1, 0:-2], 7),
    )
    for neighbor, bit in neighbors:
        codes |= (neighbor >= center).astype(np.uint8) << bit
    histogram, _ = np.histogram(codes.ravel(), bins=10, range=(0, 256), density=False)
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    if total > 0:
        histogram /= total
    return histogram


def glcm_props(gray: np.ndarray, distances: Sequence[int] = (1, 2)) -> dict[str, float]:
    levels = 16
    quantized = (gray.astype(np.float64) / 255.0 * (levels - 1)).astype(np.int32)
    quantized = np.clip(quantized, 0, levels - 1)
    height, width = quantized.shape
    angles = ((1, 0), (1, 1), (0, 1), (-1, 1))
    properties = {
        "contrast": [],
        "dissimilarity": [],
        "homogeneity": [],
        "energy": [],
        "correlation": [],
    }
    i_index = np.arange(levels).reshape(-1, 1)
    j_index = np.arange(levels).reshape(1, -1)
    difference = i_index - j_index

    for distance in distances:
        for dx, dy in angles:
            x0 = max(0, -dx * distance)
            x1 = min(width, width - dx * distance)
            y0 = max(0, -dy * distance)
            y1 = min(height, height - dy * distance)
            if x1 <= x0 or y1 <= y0:
                continue
            source = quantized[y0:y1, x0:x1]
            target = quantized[
                y0 + dy * distance : y1 + dy * distance,
                x0 + dx * distance : x1 + dx * distance,
            ]
            if source.size == 0 or source.shape != target.shape:
                continue
            matrix = np.zeros((levels, levels), dtype=np.float64)
            np.add.at(matrix, (source.ravel(), target.ravel()), 1)
            matrix = matrix + matrix.T
            total = matrix.sum()
            if total <= 0:
                continue
            probability = matrix / total
            properties["contrast"].append(float((probability * (difference**2)).sum()))
            properties["dissimilarity"].append(float((probability * np.abs(difference)).sum()))
            properties["homogeneity"].append(float((probability / (1.0 + np.abs(difference))).sum()))
            properties["energy"].append(float((probability**2).sum()))
            mean_i = float((probability.sum(axis=1) * np.arange(levels)).sum())
            mean_j = float((probability.sum(axis=0) * np.arange(levels)).sum())
            std_i = float(
                np.sqrt(max((probability.sum(axis=1) * (np.arange(levels) - mean_i) ** 2).sum(), 1e-12))
            )
            std_j = float(
                np.sqrt(max((probability.sum(axis=0) * (np.arange(levels) - mean_j) ** 2).sum(), 1e-12))
            )
            correlation = float(
                ((i_index - mean_i) * (j_index - mean_j) * probability).sum() / (std_i * std_j + 1e-12)
            )
            properties["correlation"].append(correlation)

    return {
        f"glcm_{name}": float(np.mean(values)) if values else 0.0
        for name, values in properties.items()
    }


def _texture_features(gray: np.ndarray) -> dict[str, float]:
    resized = resize_short_side(gray)
    gray_float = resized.astype(np.float64)
    features: dict[str, float] = {
        "tx__gray_mean": float(gray_float.mean() / 255.0),
        "tx__gray_var": float(gray_float.var() / (255.0**2)),
    }
    gradient_x = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    features["tx__grad_mean"] = float(magnitude.mean() / 255.0)
    features["tx__grad_var"] = float(magnitude.var() / (255.0**2))
    features["tx__grad_p90"] = float(np.percentile(magnitude, 90) / 255.0)
    for index, value in enumerate(lbp_hist_10bin(resized)):
        features[f"tx__lbp_bin{index:02d}"] = float(value)
    for name, value in glcm_props(resized).items():
        features[f"tx__{name}"] = value
    return features


def extract_grade_features(
    image_bgr: np.ndarray,
    detections: Sequence[dict[str, Any]],
    feature_names: Sequence[str] | None = None,
    image_gray: np.ndarray | None = None,
) -> dict[str, float]:
    """按部署合同的特征顺序返回 43 个检测特征和 20 个纹理特征。"""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("等级特征输入必须是 BGR 三通道 ndarray")
    if image_bgr.shape[0] <= 0 or image_bgr.shape[1] <= 0:
        raise ValueError("等级特征输入图像不能为空")

    image_height, image_width = image_bgr.shape[:2]
    if image_gray is not None and (image_gray.ndim != 2 or image_gray.shape != image_bgr.shape[:2]):
        raise ValueError("等级纹理输入必须是与 BGR 图像同尺寸的灰度 ndarray")
    gray = image_gray if image_gray is not None else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    values = _base_features(detections, image_width, image_height, compute_plate_area(image_bgr))
    values.update(_texture_features(gray))
    ordered_names = tuple(feature_names) if feature_names is not None else default_feature_names()
    if len(ordered_names) != 63:
        raise ValueError(f"等级模型合同特征数量必须为 63，实际为 {len(ordered_names)}")
    missing = [name for name in ordered_names if name not in values]
    extra = [name for name in values if name not in ordered_names]
    if missing or extra:
        raise ValueError(f"等级特征与合同不一致: missing={missing}, extra={extra}")
    return {name: float(values[name]) for name in ordered_names}
