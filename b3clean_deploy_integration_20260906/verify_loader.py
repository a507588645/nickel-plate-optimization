# -*- coding: utf-8 -*-
"""集成最小验证：加载器 + schema 分派 + 单图直连判级 sanity（只读，不改任何配置）。"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

WORKTREE = Path("C:/Users/23916/AppData/Local/Temp/jcw_b3clean_wt")
sys.path.insert(0, str(WORKTREE))
os.chdir(WORKTREE)

from inference import models as model_state  # noqa: E402
from inference.detectors import _dianji_grade_schema, _grade_detector_names, dianji_detect  # noqa: E402

model_state.load_dianji_grade_model()
model_state.load_dianji_grade_detector()
model_state.load_dianji_detector()

assert model_state.dianji_grade_model is not None, "LGBM 未加载"
feature_names = model_state.dianji_grade_feature_names
assert len(feature_names) == 33, f"特征列数 {len(feature_names)} != 33"
assert abs(model_state.dianji_grade_threshold - 0.7819) < 1e-12
assert model_state.dianji_grade_yolo is not None, "判级专用检测器未加载"
assert model_state.slf_yolo is not None

info = {
    "contract_version": model_state.dianji_grade_contract["contract_version"],
    "feature_count": len(feature_names),
    "threshold": model_state.dianji_grade_threshold,
    "schema": _dianji_grade_schema(),
    "grade_detector_names": _grade_detector_names(model_state.dianji_grade_yolo),
    "display_detector_class_count": len(_grade_detector_names(model_state.slf_yolo)),
}
print(json.dumps(info, ensure_ascii=False))
assert info["grade_detector_names"] == ["waterstain", "waterblock", "colorstain", "nodule", "bubble"]
assert info["display_detector_class_count"] == 18

manifest = Path(r"E:/PythonProject/archives/dianjinie_grade_classification_v1/manifests/samples.csv")
import csv
with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = [r for r in csv.DictReader(handle) if r["split"] == "test"]
image_path = Path(r"E:/PythonProject/archives/dianjinie_grade_classification_v1") / rows[0]["link_relpath"]
result = dianji_detect(str(image_path))
data = result.get("data") or {}
print(json.dumps({
    "image": rows[0]["image_name"],
    "true_grade": rows[0]["grade"],
    "code": result.get("code"),
    "type": data.get("type"),
    "grade": data.get("grade"),
    "confidence": data.get("confidence"),
    "manual_review": data.get("manual_review"),
    "grade_source": data.get("grade_source"),
    "grade_status": data.get("grade_status"),
    "defect_count": len(data.get("defects") or []),
}, ensure_ascii=False))
assert result.get("code") == 1
assert data.get("grade_source") == "lgbm_b3clean_v1", f"grade_source 错误: {data.get('grade_source')}"
assert data.get("grade_status") == "labels_enterprise_confirmed_20260904_deployment_pending_field_trial"
assert data.get("manual_review") == (data.get("confidence", 1.0) < 0.7819)
print("VERIFY LOADER PASS")
