# -*- coding: utf-8 -*-
"""A 方案关卡③（部署件打包）：B3-clean 33 列 LightGBM 部署产物生成。

- 复用 `train_dianji_grade_v3.py` 的 split/训练函数（完全同协议：seed42、类平衡权重、
  early_stopping 40），在 B3-clean 33 列特征（13 BASE + 20 纹理）上重训并保存 booster；
- 拒判阈值按 v1 同法标定：valid 置信度分位数，目标覆盖率 0.70；
- 产出 deploy contract JSON（结构镜像 v1 契约，detectors.py 加载器 `_read_dianji_grade_contract`
  的全部必填字段齐备）+ booster txt，并做回读验证。
- 本脚本只产部署候选件到本目录，不改 WebAPI 仓库、不切换任何生产配置。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402

import train_dianji_grade_v3 as tv  # noqa: E402

FEATURES_CSV = ROOT / "dianji_grade_b3clean_y11m640_v1" / "features_b3y11m640_v1.csv"
SUMMARY_JSON = ROOT / "dianji_grade_b3clean_y11m640_v1" / "extraction_summary.json"
SEED = 42
TARGET_COVERAGE = 0.70


def main() -> None:
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    # 33 列特征（顺序 = features CSV 列序；基列 13 + 纹理 20），已对照 CSV 表头核验
    columns = [
        "detection_count", "total_bbox_union_image_ratio", "has_nodule",
        "count__waterstain", "count__waterblock", "count__colorstain", "count__nodule", "count__bubble",
        "area_ratio__waterstain", "area_ratio__waterblock", "area_ratio__colorstain",
        "area_ratio__nodule", "area_ratio__bubble",
        "tx__gray_mean", "tx__gray_var", "tx__grad_mean", "tx__grad_var", "tx__grad_p90",
        "tx__lbp_bin00", "tx__lbp_bin01", "tx__lbp_bin02", "tx__lbp_bin03", "tx__lbp_bin04",
        "tx__lbp_bin05", "tx__lbp_bin06", "tx__lbp_bin07", "tx__lbp_bin08", "tx__lbp_bin09",
        "tx__glcm_contrast", "tx__glcm_dissimilarity", "tx__glcm_homogeneity", "tx__glcm_energy",
        "tx__glcm_correlation",
    ]
    assert len(columns) == 33, f"预期 33 列，实际 {len(columns)}"

    rows = tv.read_csv(FEATURES_CSV)
    missing = [c for c in columns if c not in rows[0]]
    assert not missing, f"特征 CSV 缺列: {missing}"
    r0 = rows[0]
    onnx_path = HERE / "dianji0211_y11m640_b3clean.onnx"
    h = __import__("hashlib").sha256()
    with open(onnx_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    onnx_sha = h.hexdigest()
    print(f"deploy ONNX sha256={onnx_sha}；特征行 detector_conf={r0['detector_conf']} "
          f"detector_iou={r0['detector_iou']} infer_img_size={r0['infer_img_size']} "
          f"pt_sha={r0['detector_weight_sha256'][:12]}…", flush=True)
    splits = tv.split_rows(rows)
    X_tr, y_tr, _ = tv.matrix_from_rows(splits["train"], columns)
    X_va, y_va, _ = tv.matrix_from_rows(splits["valid"], columns)
    X_te, y_te, _ = tv.matrix_from_rows(splits["test"], columns)
    print(f"split: train={len(y_tr)} valid={len(y_va)} test={len(y_te)}", flush=True)

    bundle = tv.train_lgbm(X_tr, y_tr, X_va, y_va, seed=SEED, num_leaves=8, max_depth=4,
                           min_data=10, columns=columns)
    booster = bundle["model"]
    best_iter = int(bundle["best_iteration"])
    proba_va = booster.predict(X_va, num_iteration=best_iter)
    proba_te = booster.predict(X_te, num_iteration=best_iter)

    # 拒判阈值：valid 置信度分位数（与 v1 同法，仅用 valid，test 只报告一次）
    conf_va = proba_va.max(axis=1)
    threshold = float(np.quantile(conf_va, 1.0 - TARGET_COVERAGE))
    cov_va = float((conf_va >= threshold).mean())
    conf_te = proba_te.max(axis=1)
    pred_te = proba_te.argmax(axis=1)
    acc_all = float((pred_te == y_te).mean())
    acc_acc = float((pred_te[conf_te >= threshold] == y_te[conf_te >= threshold]).mean())
    cov_te = float((conf_te >= threshold).mean())
    rep = tv.metrics_report(y_te, pred_te)
    print(f"best_iter={best_iter} threshold={threshold:.4f} cov_va={cov_va:.4f} "
          f"test acc={acc_all:.5f} macroF1={rep['macro_f1']:.5f} cov_te={cov_te:.4f} "
          f"acc@accepted={acc_acc:.5f}", flush=True)

    model_txt = HERE / "dianji_grade_lgbm_b3clean_v1.txt"
    # LightGBM 的 C 库对中文路径写/读都可能失败（与 cv2.imwrite 同类问题）：先存 ASCII 临时路径再移动
    import shutil
    import tempfile
    tmp_txt = Path(tempfile.gettempdir()) / "dianji_grade_lgbm_b3clean_v1.txt"
    booster.save_model(str(tmp_txt))
    rb = lgb.Booster(model_file=str(tmp_txt))
    pred_rb = rb.predict(X_te, num_iteration=best_iter).argmax(axis=1)
    round_trip_ok = bool((pred_rb == pred_te).all())
    print(f"round_trip identical={round_trip_ok}", flush=True)
    shutil.move(str(tmp_txt), str(model_txt))

    grades = list(tv.GRADES)
    grade_to_id = {g: i for i, g in enumerate(grades)}
    contract = {
        "contract_version": "dianji_grade_b3clean_v1",
        "status": "labels_enterprise_confirmed_20260904_deployment_pending_field_trial",
        "feature_names": columns,
        "feature_count": len(columns),
        "grades": grades,
        "grade_to_id": grade_to_id,
        "id_to_grade": {str(i): g for g, i in grade_to_id.items()},
        "model": {
            "file": model_txt.name,
            "format": "lightgbm_booster_text",
            "best_iteration": best_iter,
            "num_class": 4,
            "feature_names_in_model": list(rb.feature_name()),
        },
        "rejection": {
            "selected_target_coverage": TARGET_COVERAGE,
            "threshold": round(threshold, 4),
            "valid_coverage": round(cov_va, 4),
            "test_coverage_reference": round(cov_te, 4),
            "test_n_accepted_reference": int((conf_te >= threshold).sum()),
            "source": "valid confidence quantile only; test evaluated once for reporting",
            "manual_review_when_confidence_below_threshold": True,
        },
        "extraction": {
            "weight_path": str(onnx_path),
            "weight_sha256": onnx_sha,
            "imgsz": int(float(r0["infer_img_size"])),
            "conf": float(r0["detector_conf"]),
            "iou": float(r0["detector_iou"]),
            "device": "cpu",
            "class_names": ["waterstain", "waterblock", "colorstain", "nodule", "bubble"],
            "aggregation_reference": "B3clean 提取包装器（slf 5 类 13 BASE + 20 纹理）",
            "source_checkpoint": {
                "path": summary.get("weights"),
                "sha256": r0["detector_weight_sha256"],
                "note": "部署 ONNX 由该 best.pt 导出（PT↔ONNX 平均匹配率 0.979@IoU0.5，见 export_parity_report.json）",
            },
        },
        "training": {
            "seed": SEED, "protocol": "train_dianji_grade_v3 defaults (num_leaves=8, max_depth=4, "
                                      "min_data=10, lr=0.05, ff=0.9, bf=0.9, early_stop=40, 类平衡权重)",
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        },
        "data": {
            "features_csv": str(FEATURES_CSV),
            "view": summary.get("view_root"),
            "splits": {"train": len(y_tr), "valid": len(y_va), "test": len(y_te)},
            "leakage": "zero-leakage：B3clean 训练集 653 图不含等级视图 valid/test 339 图",
        },
        "metrics": {
            "test_accuracy": round(acc_all, 5),
            "test_macro_f1": round(rep["macro_f1"], 5),
            "test_per_class": rep.get("per_class"),
        },
        "round_trip": {"verified": round_trip_ok, "predict_identical": round_trip_ok},
    }
    (HERE / "dianji_grade_lgbm_b3clean_v1_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE -> {model_txt.name} + contract json", flush=True)


if __name__ == "__main__":
    main()
