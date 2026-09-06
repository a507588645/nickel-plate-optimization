# -*- coding: utf-8 -*-
import json, os, sys, traceback
from pathlib import Path
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
WORKTREE = Path("C:/Users/23916/AppData/Local/Temp/jcw_b3clean_wt")
sys.path.insert(0, str(WORKTREE)); os.chdir(WORKTREE)
from inference import models as model_state
from inference.detectors import _dianjie_lgbm_grade_for_dianjie, _readimg
model_state.load_dianjie_detector()
model_state.load_dianjie_grade_model()
img = r"E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1/images/dianjienie__train__1.png"
try:
    r = _dianjie_lgbm_grade_for_dianjie(img)
    print("GRADE OK", json.dumps(r, ensure_ascii=False))
except Exception:
    traceback.print_exc()
