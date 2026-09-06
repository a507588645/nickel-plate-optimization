# -*- coding: utf-8 -*-
"""C2.5 集成最小验证：dianjie 判级模型加载 + 直连单图 sanity + 规则树兜底检查。"""
import csv, json, os, sys
from pathlib import Path
from pathlib import Path
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
WORKTREE = Path("C:/Users/23916/AppData/Local/Temp/jcw_b3clean_wt")
sys.path.insert(0, str(WORKTREE)); os.chdir(WORKTREE)
from inference import models as model_state
from inference.detectors import dianjie_detect

model_state.load_dianjie_detector()
model_state.load_dianjie_grade_model()
model_state.load_resnet()
assert model_state.dianjie_grade_model is not None
c = model_state.dianjie_grade_contract
info = {"version": c["contract_version"], "features": len(model_state.dianjie_grade_feature_names),
        "threshold": model_state.dianjie_grade_threshold, "num_class": c["model"]["num_class"],
        "type_to_nodule": c["type_to_nodule"], "best_iter": model_state.dianjie_grade_best_iteration}
print(json.dumps(info, ensure_ascii=False))

img = r"E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1/images"
import csv
rows = [r for r in csv.DictReader(open(r"E:/PythonProject/优化/dianjie_grade_weaklabel_v1_20260906/weaklabels_summary.csv", encoding="utf-8-sig")) if r["split"]=="test"][:3]
for r in rows:
    res = dianjie_detect(r["image_path"])
    d = res.get("data") or {}
    print(json.dumps({"image": Path(r["image_path"]).name, "weak_type": r["type"], "code": res.get("code"),
        "type": d.get("type"), "nodule": d.get("nodule"), "confidence": d.get("confidence"),
        "manual_review": d.get("manual_review"), "grade_source": d.get("grade_source"),
        "defect_count": len(d.get("defects") or [])}, ensure_ascii=False))
    assert res.get("code") == 1 and d.get("grade_source") == "lgbm_c2p3_v1"
    assert d.get("type") in (1,2,3,4,6) and d.get("nodule") in (0,1,2,3)
    assert d.get("manual_review") == (d.get("confidence", 1.0) < 0.8525)
print("VERIFY PASS")
