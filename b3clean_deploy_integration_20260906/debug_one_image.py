# -*- coding: utf-8 -*-
import os, sys
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
WORKTREE = "C:/Users/23916/AppData/Local/Temp/jcw_b3clean_wt"
sys.path.insert(0, WORKTREE)
os.chdir(WORKTREE)
import json
import cv2
import numpy as np
import lightgbm as lgb
from inference import models as model_state
from inference.detectors import _predict_dianji_yolo_raw, dianji_detect

model_state.load_dianji_grade_model(); model_state.load_dianji_grade_detector()
model_state.load_dianji_detector(); model_state.load_resnet()
old = lgb.Booster(model_file=WORKTREE + "/weights/dianji_grade_lgbm_v1.txt")
print("boosters loaded", flush=True)

img = r"E:/PythonProject/archives/dianjinie_grade_classification_v1/images/test/3#/c4_train_3sharp_20230509165356877_0_2.png"
bgr = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
raw = _predict_dianji_yolo_raw(bgr)
print("display raw boxes:", len(raw), flush=True)
res = dianji_detect(img)
print("new arm:", json.dumps({k: res["data"].get(k) for k in ("grade","confidence","type")}, ensure_ascii=False), flush=True)
print("ONE IMAGE OK", flush=True)
