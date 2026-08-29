# -*- coding: utf-8 -*-
"""对照生产 ndarray 路径（cvtColor 灰度）与归档 parity 路径（imdecode 灰度）的等级特征差异。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(r"E:\PythonProject\ImetricCodes\JinchuanPythonWebapi")
CSV_PATH = Path(r"E:\PythonProject\优化\dianji_grade_y11m640_v1\features_y11m640_v1.csv")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from inference import models as model_state
from inference.detectors import _predict_dianji_grade_detections
from inference.grade_features import extract_grade_features

model_state.load_dianjie_detector()
model_state.load_dianji_grade_model()
feature_names = tuple(model_state.dianji_grade_feature_names)
best_iteration = int(model_state.dianji_grade_best_iteration)

rows = []
with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        p = row.get("image_path", "")
        if p and Path(p).is_file():
            rows.append(row)
        if len(rows) >= 6:
            break

report = []
try:
    for row in rows:
        data = np.fromfile(row["image_path"], dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        gray_decode = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        gray_cvt = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_pixel_diff = int(np.abs(gray_decode.astype(np.int16) - gray_cvt.astype(np.int16)).max())

        detections = _predict_dianji_grade_detections(bgr)
        feat_decode = extract_grade_features(bgr, detections, feature_names, image_gray=gray_decode)
        feat_cvt = extract_grade_features(bgr, detections, feature_names, image_gray=gray_cvt)
        v1 = np.asarray([[feat_decode[n] for n in feature_names]])
        v2 = np.asarray([[feat_cvt[n] for n in feature_names]])
        feat_diff = float(np.max(np.abs(v1 - v2)))
        p1 = np.asarray(model_state.dianji_grade_model.predict(v1, num_iteration=best_iteration)).reshape(-1)
        p2 = np.asarray(model_state.dianji_grade_model.predict(v2, num_iteration=best_iteration)).reshape(-1)
        report.append({
            "image": Path(row["image_path"]).name,
            "gray_pixel_maxdiff": gray_pixel_diff,
            "feature_max_abs_diff": feat_diff,
            "grade_id_decode": int(np.argmax(p1)),
            "grade_id_cvt": int(np.argmax(p2)),
            "confidence_delta": float(abs(p1.max() - p2.max())),
        })
finally:
    model_state.unload_models()

for item in report:
    print(json.dumps(item, ensure_ascii=False))
flip = sum(1 for item in report if item["grade_id_decode"] != item["grade_id_cvt"])
print(json.dumps({"images": len(report), "grade_flips": flip,
                  "max_feature_diff": max(r["feature_max_abs_diff"] for r in report),
                  "max_confidence_delta": max(r["confidence_delta"] for r in report)}, ensure_ascii=False))
