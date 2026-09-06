# -*- coding: utf-8 -*-
"""B3-clean 判级集成零容差复核（流程同 A3）。

对比两臂（161 张 grade-test 图，CPU、部署 ONNX 同权重）：
- 参考臂（独立实现）：ultralytics 直接加载部署 ONNX predict +
  训练侧 features_from_slf_detections 聚合代码副本 + 部署 LGBM booster；
- 服务臂（集成管线）：worktree detectors.dianji_detect 完整路径（显示 18 类 +
  判级专用 5 类检测器 + 33 列特征 + LGBM）。

期望：两臂特征逐列一致（<=1e-9）、概率一致（<=1e-12）、等级/置信度/manual_review
逐图一致。任何不一致即说明集成与合同存在实现差异，切换不得放行。
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

WORKTREE = Path("C:/Users/23916/AppData/Local/Temp/jcw_b3clean_wt")
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
MANIFEST = Path(r"E:/PythonProject/archives/dianjinie_grade_classification_v1/manifests/samples.csv")
VIEW_ROOT = Path(r"E:/PythonProject/archives/dianjinie_grade_classification_v1")
SLF_CLASS_NAMES = ["waterstain", "waterblock", "colorstain", "nodule", "bubble"]
SLF_NODULE_LABEL = 3
FEATURE_DIFF_TOL = 1e-9
PROB_DIFF_TOL = 1e-12
CONF_DIFF_TOL = 1e-9


def features_from_slf_detections(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    conf_thres: float,
) -> dict[str, Any]:
    """训练侧聚合语义副本（来源 extract_dianji_grade_b3_y11m640_features.py，独立于 WebAPI 实现）。"""
    kept = [d for d in detections if float(d.get("conf", 0)) >= conf_thres]
    has_nodule = int(any(int(d["label"]) == SLF_NODULE_LABEL for d in kept))

    def normalized_rect(det: dict[str, Any]) -> tuple[float, float, float, float]:
        cx, cy, w, h = float(det["cx"]), float(det["cy"]), float(det["w"]), float(det["h"])
        return (
            max(0.0, cx - w / 2),
            max(0.0, cy - h / 2),
            min(1.0, cx + w / 2),
            min(1.0, cy + h / 2),
        )

    def union_area(rectangles) -> float:
        rects = [(x1, y1, x2, y2) for x1, y1, x2, y2 in rectangles if x2 > x1 and y2 > y1]
        if not rects:
            return 0.0
        xs = sorted({v for r in rects for v in (r[0], r[2])})
        total = 0.0
        for left, right in zip(xs, xs[1:]):
            if right <= left:
                continue
            intervals = sorted((y1, y2) for x1, y1, x2, y2 in rects if x1 < right and x2 > left)
            if not intervals:
                continue
            covered = 0.0
            start, end = intervals[0]
            for ns, ne in intervals[1:]:
                if ns > end:
                    covered += end - start
                    start, end = ns, ne
                else:
                    end = max(end, ne)
            covered += end - start
            total += (right - left) * covered
        return total

    all_norm = [normalized_rect(d) for d in kept]
    per_class_rects: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    per_class_counts: dict[int, int] = {}
    for det in kept:
        cid = int(det["label"])
        per_class_counts[cid] = per_class_counts.get(cid, 0) + 1
        per_class_rects[cid].append(normalized_rect(det))

    out: dict[str, Any] = {
        "detection_count": float(len(kept)),
        "total_bbox_union_image_ratio": union_area(all_norm),
        "has_nodule": float(has_nodule),
    }
    for i, name in enumerate(SLF_CLASS_NAMES):
        out[f"count__{name}"] = float(per_class_counts.get(i, 0))
        out[f"area_ratio__{name}"] = union_area(per_class_rects.get(i, []))
    return out


def read_image_pair(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    image_gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image_bgr is None or image_gray is None:
        raise ValueError(f"无法解码图像: {image_path}")
    return image_bgr, image_gray


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(WORKTREE))
    os.chdir(WORKTREE)

    import lightgbm as lgb
    from ultralytics import YOLO

    from inference import models as model_state
    from inference.detectors import (
        _dianji_grade_schema,
        _predict_dianji_grade_detections,
        dianji_detect,
    )
    from inference.grade_features import extract_slf5_grade_features, _texture_features

    model_state.load_dianji_grade_model()
    model_state.load_dianji_grade_detector()
    model_state.load_dianji_detector()
    model_state.load_resnet()
    assert model_state.dianji_grade_model is not None
    assert _dianji_grade_schema() == "slf5"
    contract = model_state.dianji_grade_contract
    feature_names = list(model_state.dianji_grade_feature_names)
    best_iteration = int(model_state.dianji_grade_best_iteration)
    threshold = float(model_state.dianji_grade_threshold)
    extraction = contract["extraction"]
    conf = float(extraction["conf"])
    imgsz = int(extraction["imgsz"])
    iou = float(extraction["iou"])
    grade_onnx = WORKTREE / "weights" / "dianji0211_y11m640_b3clean.onnx"
    print(f"contract={contract['contract_version']} features={len(feature_names)} "
          f"threshold={threshold} imgsz={imgsz} conf={conf} iou={iou}", flush=True)

    booster = model_state.dianji_grade_model

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "test"]
    rows.sort(key=lambda r: r["image_name"])
    assert len(rows) == 161, f"test 行数 {len(rows)} != 161"

    reference_model = YOLO(str(grade_onnx), task="detect")
    reference_model.overrides["device"] = "cpu"

    per_image: list[dict[str, Any]] = []
    n_feature_ok = n_prob_ok = n_grade_ok = n_end2end_ok = 0
    max_feature_diff = 0.0
    max_prob_diff = 0.0
    max_conf_diff = 0.0
    t0 = time.perf_counter()

    for index, row in enumerate(rows, 1):
        image_path = VIEW_ROOT / row["link_relpath"]
        image_bgr, image_gray = read_image_pair(image_path)
        image_height, image_width = image_bgr.shape[:2]

        # —— 参考臂：独立 predict + 训练侧聚合 ——
        results = reference_model.predict(
            source=image_bgr, imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False
        )
        assert len(results) == 1
        boxes = results[0].boxes
        ref_detections: list[dict[str, Any]] = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy()
            for coords, cf, cid in zip(xyxy, confidences, class_ids):
                x1, y1, x2, y2 = (float(v) for v in coords[:4])
                w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                ref_detections.append({
                    "label": int(cid), "conf": float(cf),
                    "cx": (x1 + x2) / 2.0 / image_width, "cy": (y1 + y2) / 2.0 / image_height,
                    "w": w / image_width, "h": h / image_height,
                })
        ref_base = features_from_slf_detections(ref_detections, image_width, image_height, conf)
        ref_features = dict(ref_base)
        ref_features.update(_texture_features(image_gray))
        ref_row = np.asarray([[ref_features[name] for name in feature_names]], dtype=np.float64)
        ref_probs = np.asarray(
            booster.predict(ref_row, num_iteration=best_iteration), dtype=np.float64
        ).reshape(-1)
        ref_grade_id = int(np.argmax(ref_probs))
        ref_grade = str(contract["id_to_grade"][str(ref_grade_id)])
        ref_conf = float(ref_probs[ref_grade_id])

        # —— 服务臂：集成管线内部特征路径 ——
        grade_detections = _predict_dianji_grade_detections(image_bgr)
        serve_features = extract_slf5_grade_features(
            image_bgr, grade_detections, feature_names, image_gray=image_gray, conf_thres=conf
        )
        serve_row = np.asarray([[serve_features[name] for name in feature_names]], dtype=np.float64)
        serve_probs = np.asarray(
            booster.predict(serve_row, num_iteration=best_iteration), dtype=np.float64
        ).reshape(-1)
        serve_grade_id = int(np.argmax(serve_probs))
        serve_grade = str(contract["id_to_grade"][str(serve_grade_id)])
        serve_conf_internal = float(serve_probs[serve_grade_id])

        # —— 服务臂：端到端入口（含显示路径，与 HTTP 同一函数） ——
        response = dianji_detect(str(image_path))
        data = response.get("data") or {}
        e2e_grade = data.get("grade")
        e2e_conf = data.get("confidence")
        e2e_manual_review = data.get("manual_review")

        feature_diff = float(np.max(np.abs(ref_row - serve_row)))
        prob_diff = float(np.max(np.abs(ref_probs - serve_probs)))
        conf_diff = abs(ref_conf - float(e2e_conf))
        feature_ok = feature_diff <= FEATURE_DIFF_TOL
        prob_ok = prob_diff <= PROB_DIFF_TOL
        grade_ok = (ref_grade == serve_grade == e2e_grade)
        end2end_ok = (
            response.get("code") == 1
            and grade_ok
            and conf_diff <= CONF_DIFF_TOL
            and e2e_manual_review == (float(e2e_conf) < threshold)
            and data.get("grade_source") == "lgbm_b3clean_v1"
        )
        n_feature_ok += int(feature_ok)
        n_prob_ok += int(prob_ok)
        n_grade_ok += int(grade_ok)
        n_end2end_ok += int(end2end_ok)
        max_feature_diff = max(max_feature_diff, feature_diff)
        max_prob_diff = max(max_prob_diff, prob_diff)
        max_conf_diff = max(max_conf_diff, conf_diff)
        per_image.append({
            "image_name": row["image_name"],
            "true_grade": row["grade"],
            "ref_grade": ref_grade,
            "serve_grade": serve_grade,
            "e2e_grade": e2e_grade,
            "ref_confidence": ref_conf,
            "e2e_confidence": float(e2e_conf),
            "feature_max_abs_diff": feature_diff,
            "prob_max_abs_diff": prob_diff,
            "conf_abs_diff": conf_diff,
            "feature_ok": feature_ok,
            "prob_ok": prob_ok,
            "grade_ok": grade_ok,
            "end2end_ok": end2end_ok,
        })
        if index == 1 or index % 20 == 0 or index == len(rows):
            rate = index / (time.perf_counter() - t0)
            print(f"  [{index}/{len(rows)}] {rate:.2f} img/s e2e_ok={n_end2end_ok}", flush=True)

    all_pass = (
        n_feature_ok == len(rows)
        and n_prob_ok == len(rows)
        and n_grade_ok == len(rows)
        and n_end2end_ok == len(rows)
    )
    summary = {
        "contract_version": contract["contract_version"],
        "grade_onnx_sha256_prefix": __import__("hashlib").sha256(grade_onnx.read_bytes()).hexdigest()[:16],
        "images": len(rows),
        "feature_ok": n_feature_ok,
        "prob_ok": n_prob_ok,
        "grade_ok": n_grade_ok,
        "end2end_ok": n_end2end_ok,
        "max_feature_abs_diff": max_feature_diff,
        "max_prob_abs_diff": max_prob_diff,
        "max_conf_abs_diff": max_conf_diff,
        "tolerances": {
            "feature": FEATURE_DIFF_TOL,
            "prob": PROB_DIFF_TOL,
            "confidence": CONF_DIFF_TOL,
        },
        "status": "PASS" if all_pass else "FAIL",
        "wall_seconds": time.perf_counter() - t0,
    }
    (OUT_DIR / "a3_zero_tolerance_recheck_b3clean.json").write_text(
        json.dumps({"summary": summary, "per_image": per_image}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (OUT_DIR / "a3_zero_tolerance_recheck_b3clean.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
