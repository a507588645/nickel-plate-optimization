"""校验 WebAPI 的等级特征与已验证的 y11m640 特征表一致。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


GRADE_NAMES = ("3#", "9950", "9996", "bad")
DEFAULT_CSV = Path(r"E:\PythonProject\优化\dianji_grade_y11m640_v1\features_y11m640_v1.csv")
DEFAULT_REFERENCE_MODEL = Path(r"E:\PythonProject\优化\dianji_grade_y11m640_v1\lgbm_model.txt")


def _read_images(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.fromfile(image_path, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    image_gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image_bgr is None or image_gray is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return image_bgr, image_gray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_rows(csv_path: Path, per_grade: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("grade") in GRADE_NAMES and Path(row.get("image_path", "")).is_file():
                grouped[row["grade"]].append(row)
    selected: list[dict[str, str]] = []
    for grade in GRADE_NAMES:
        if len(grouped[grade]) < per_grade:
            raise ValueError(f"特征表中可用 {grade} 样本不足 {per_grade} 张")
        selected.extend(grouped[grade][:per_grade])
    return selected


def _predict_id(booster, values: np.ndarray, best_iteration: int) -> int:
    probabilities = np.asarray(booster.predict(values, num_iteration=best_iteration), dtype=np.float64).reshape(-1)
    if probabilities.size != 4:
        raise RuntimeError(f"LightGBM 概率输出异常: {probabilities!r}")
    return int(np.argmax(probabilities))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--reference-model", type=Path, default=DEFAULT_REFERENCE_MODEL)
    parser.add_argument("--per-grade", type=int, default=3, help="每个等级抽取的图像数，默认共 12 张")
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    reference_model_path = args.reference_model.expanduser().resolve()
    if args.per_grade <= 0:
        raise ValueError("--per-grade 必须大于 0")
    rows = _select_rows(csv_path, args.per_grade)

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from inference import models as model_state
    from inference.detectors import _predict_dianji_grade_detections
    from inference.grade_features import extract_grade_features
    import lightgbm as lgb

    model_state.load_dianji_detector()
    model_state.load_dianji_grade_model()
    if model_state.dianji_grade_model is None:
        raise RuntimeError("WebAPI LightGBM 等级模型未加载")
    feature_names = tuple(model_state.dianji_grade_feature_names)
    best_iteration = int(model_state.dianji_grade_best_iteration)
    max_abs_feature_diff = 0.0
    grade_id_matches = 0
    try:
        # LightGBM 的 Windows 原生接口不能直接打开中文路径，临时副本仅用于只读参考模型加载。
        with tempfile.TemporaryDirectory(prefix="dianji_lgbm_") as temporary_dir:
            temporary_model_path = Path(temporary_dir) / "lgbm_model.txt"
            shutil.copy2(reference_model_path, temporary_model_path)
            reference_booster = lgb.Booster(model_file=str(temporary_model_path))
            if tuple(reference_booster.feature_name()) != feature_names:
                raise AssertionError("参考 LightGBM 模型与部署合同特征顺序不一致")

            for row in rows:
                image_path = Path(row["image_path"])
                actual_sha256 = _sha256(image_path)
                if actual_sha256 != row["sha256"]:
                    raise AssertionError(f"样本 sha256 不匹配: {image_path}")
                image_bgr, image_gray = _read_images(image_path)
                detections = _predict_dianji_grade_detections(image_bgr)
                actual_features = extract_grade_features(image_bgr, detections, feature_names, image_gray=image_gray)
                actual_values = np.asarray([[actual_features[name] for name in feature_names]], dtype=np.float64)
                expected_values = np.asarray([[float(row[name]) for name in feature_names]], dtype=np.float64)
                feature_diff = float(np.max(np.abs(actual_values - expected_values)))
                max_abs_feature_diff = max(max_abs_feature_diff, feature_diff)
                if feature_diff > 1e-6:
                    raise AssertionError(f"特征差异超过 1e-6: {image_path}, diff={feature_diff}")

                actual_grade_id = _predict_id(model_state.dianji_grade_model, actual_values, best_iteration)
                reference_grade_id = _predict_id(reference_booster, expected_values, best_iteration)
                if actual_grade_id != reference_grade_id:
                    raise AssertionError(
                        f"等级 id 不一致: {image_path}, actual={actual_grade_id}, reference={reference_grade_id}"
                    )
                grade_id_matches += 1
                print(
                    json.dumps(
                        {
                            "image": str(image_path),
                            "grade": row["grade"],
                            "feature_max_abs_diff": feature_diff,
                            "grade_id": actual_grade_id,
                        },
                        ensure_ascii=False,
                    )
                )
    finally:
        model_state.unload_models()

    print(
        json.dumps(
            {
                "images_checked": len(rows),
                "max_abs_feature_diff": max_abs_feature_diff,
                "grade_id_matches": grade_id_matches,
                "status": "PASS",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
