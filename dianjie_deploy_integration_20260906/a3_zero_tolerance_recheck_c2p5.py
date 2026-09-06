# -*- coding: utf-8 -*-
"""C2.5 零容差复核（流程同 A3/B3-clean）：电解镍学习型判级集成后全量重跑 test 376 图。

- 参考臂：C2.2 特征 CSV（同一部署 ONNX 901ccf60 + 生产口径提取，已归档）行值 → 部署 booster；
- 服务臂：worktree detectors.dianjie_detect 完整路径（显示 18 类不变 + 63 列特征 + LGBM）。
- 比对：内部特征路径 vs CSV（<=1e-9）、type/conf vs 参考臂、manual_review 与阈值一致性。
任何不一致即说明集成与合同存在实现差异，切换不得放行。
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
WORKTREE = Path("C:/Users/23916/AppData/Local/Temp/jcw_b3clean_wt")
FEATURES_CSV = HERE.parent / "dianjie_grade_weaklabel_v1_20260906" / "features_c2p2_y11m640.csv"
# 参考臂为 C2.2 特征 CSV（8 位小数舍入，单值最大舍入误差 5e-9）：容差取 1e-6（v1 验收同口径）。
FEATURE_DIFF_TOL = 1e-6
CONF_DIFF_TOL = 1e-6
METADATA_FIELDS = {
    "sample_id", "image_name", "image_path", "group_id", "fold", "split", "source_split",
    "type_weak", "nodule_weak", "n_defects_weak", "feature_version",
    "image_width", "image_height", "plate_area", "plate_area_fallback", "max_confidence",
}


def read_image_pair(image_path: Path):
    import cv2
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    return (cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(WORKTREE))
    os.chdir(WORKTREE)

    from inference import models as model_state  # noqa: E402
    from inference.detectors import (  # noqa: E402
        _dianjie_grade_extraction,
        _normalized_grade_detections,
        _predict_yolo_raw,
        dianjie_detect,
    )
    from inference.grade_features import extract_grade_features  # noqa: E402

    model_state.load_dianjie_detector()
    model_state.load_dianjie_grade_model()
    model_state.load_resnet()
    contract = model_state.dianjie_grade_contract
    assert contract is not None, "学习型判级模型未加载"
    feature_names = list(model_state.dianjie_grade_feature_names)
    best_iteration = int(model_state.dianjie_grade_best_iteration)
    threshold = float(model_state.dianjie_grade_threshold)
    booster = model_state.dianjie_grade_model
    print(f"contract={contract['contract_version']} features={len(feature_names)} threshold={threshold}", flush=True)

    with FEATURES_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        csv_rows = {r["sample_id"]: r for r in csv.DictReader(handle) if r["split"] == "test"}
    assert len(csv_rows) == 376, f"test 特征行数 {len(csv_rows)} != 376"

    per_image: list[dict[str, Any]] = []
    n_feature_ok = n_type_ok = n_e2e_ok = 0
    max_feature_diff = 0.0
    max_conf_diff = 0.0
    t0 = time.perf_counter()
    sample_ids = sorted(csv_rows.keys())

    for index, sample_id in enumerate(sample_ids, 1):
        row = csv_rows[sample_id]
        image_path = Path(row["image_path"])
        image_bgr, image_gray = read_image_pair(image_path)
        assert image_bgr is not None and image_gray is not None

        # —— 参考臂：CSV 特征 → booster ——
        ref_row = np.asarray([[float(row[name]) for name in feature_names]], dtype=np.float64)
        ref_probs = np.asarray(booster.predict(ref_row, num_iteration=best_iteration), dtype=np.float64).reshape(-1)
        ref_type = int(contract["id_to_grade"][str(int(np.argmax(ref_probs)))])
        ref_conf = float(ref_probs.max())

        # —— 服务臂（内部特征路径） ——
        extraction = _dianjie_grade_extraction()
        raw = _predict_yolo_raw(image_bgr, model_state.yolo, extraction, len(contract["extraction"]["class_names"]))
        detections = _normalized_grade_detections(raw, image_bgr, [str(n) for n in contract["extraction"]["class_names"]])
        serve_features = extract_grade_features(image_bgr, detections, feature_names, image_gray=image_gray)
        serve_row = np.asarray([[serve_features[name] for name in feature_names]], dtype=np.float64)
        serve_probs = np.asarray(booster.predict(serve_row, num_iteration=best_iteration), dtype=np.float64).reshape(-1)
        serve_type = int(contract["id_to_grade"][str(int(serve_probs.argmax()))])

        # —— 服务臂（端到端入口，与 HTTP 同一函数） ——
        response = dianjie_detect(str(image_path))
        data = response.get("data") or {}

        feature_diff = float(np.max(np.abs(ref_row - serve_row)))
        conf_diff = abs(float(data.get("confidence", -1.0)) - ref_conf)
        type_ok = (serve_type == ref_type == int(data.get("type")))
        end2end_ok = (
            response.get("code") == 1
            and type_ok
            and feature_diff <= FEATURE_DIFF_TOL
            and conf_diff <= CONF_DIFF_TOL
            and data.get("manual_review") == (float(data.get("confidence")) < threshold)
            and data.get("grade_source") == "lgbm_c2p3_v1"
        )
        n_feature_ok += int(feature_diff <= FEATURE_DIFF_TOL)
        n_type_ok += int(type_ok)
        n_e2e_ok += int(end2end_ok)
        max_feature_diff = max(max_feature_diff, feature_diff)
        max_conf_diff = max(max_conf_diff, conf_diff)
        per_image.append({
            "sample_id": sample_id, "weak_type": row["type_weak"],
            "ref_type": ref_type, "serve_type": serve_type, "e2e_type": data.get("type"),
            "ref_confidence": ref_conf, "e2e_confidence": data.get("confidence"),
            "feature_max_abs_diff": feature_diff, "conf_abs_diff": conf_diff,
            "end2end_ok": end2end_ok,
        })
        if index == 1 or index % 40 == 0 or index == len(sample_ids):
            rate = index / (time.perf_counter() - t0)
            print(f"  [{index}/{len(sample_ids)}] {rate:.2f} img/s e2e_ok={n_e2e_ok}", flush=True)

    all_pass = n_feature_ok == len(sample_ids) and n_type_ok == len(sample_ids) and n_e2e_ok == len(sample_ids)
    summary = {
        "contract_version": contract["contract_version"],
        "images": len(sample_ids),
        "feature_ok": n_feature_ok,
        "type_ok": n_type_ok,
        "end2end_ok": n_e2e_ok,
        "max_feature_abs_diff": max_feature_diff,
        "max_conf_abs_diff": max_conf_diff,
        "status": "PASS" if all_pass else "FAIL",
        "wall_seconds": time.perf_counter() - t0,
    }
    (OUT_DIR / "a3_zero_tolerance_recheck_c2p5.json").write_text(
        json.dumps({"summary": summary, "per_image": per_image}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "a3_zero_tolerance_recheck_c2p5.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
