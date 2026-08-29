# Role

You are the implementation executor. Kimi Code owns planning and final acceptance. Work autonomously only in the exact working directory supplied by the launcher (`E:\PythonProject\ImetricCodes\JinchuanPythonWebapi`, branch must be `优化分支`; NOTE: console may mangle the Chinese branch name to `????` — that is an encoding artifact, the branch IS correct; do not abort on it, verify via `git rev-parse --abbrev-ref HEAD` written to a UTF-8 file if needed).

# Objective

User decision (2026-08-28): **电解镍 (dianjie) reverts to the original ryw.pt stack; 电积镍 (dianji) switches BOTH detection display and grade classification to the YOLO11m ONNX + LightGBM stack**, with 18-class identity label passthrough to the frontend. This also resolves the D4 gate as "switch approved".

# Current Evidence (all verified by Kimi)

- Current HEAD: `ac2d38a` (LightGBM integration). Previous: `8c436ff` (D1 .pth head-validation fix + D3 dianjie→ONNX swap). `02fb9ca` is the last commit where dianjie used ryw.pt.
- Original dianjie detector config (from `git show 8c436ff~1:models.yaml`):
  ```yaml
  dianjie:
    kind: yolov5
    weights: 'weights/ryw.pt'
    threshold: 0.25
  ```
- Current `models.yaml`: `detectors.dianjie` = ultralytics ONNX (imgsz 640, threshold 0.25); `detectors.dianji` = yolov5 `weights/slf.pt` 0.25; `grade_models.dianji_lgbm` = LGBM + contract (threshold 0.7362).
- `inference/detectors.py` current structure (post-ac2d38a):
  - `_dianjie_detect` (lines ~46-163): resnet (ryw.pth 2-class) + `model_state.yolo(image)` with `.xyxy[0]` + hardcoded A/B/C/Airhole rules → type/nodule/defects. **Unchanged by this task except that `model_state.yolo` becomes ryw.pt again via config.**
  - `_predict_dianji_grade_detections(image_bgr)`: uses `model_state.yolo`, validates class names vs contract, calls `detector.model.predict(source, imgsz=640, conf=0.25, iou=0.45, device="cpu")`, returns NORMALIZED detection dicts `{label, conf, cx, cy, w, h}`.
  - `_dianji_detect`: `_read_dianji_grade_image` → `_dianji_lgbm_grade` (fallback `_fallback_dianji_grade` = legacy resnet type) → display path `model_state.slf_yolo(image)` with 5-class mapping `{0:1, 1:0, 2:2, 3:3, 4:5}` → pixel cx/cy/w/h defect dicts.
  - Response: `data = {"type", "defects", "grade", "confidence", "manual_review", "grade_status", "grade_source"}`.
- Frontend display label space IS the 18-class RYW id space (dianjie endpoint emits raw 0-17 labels; old dianji mapping target ids {0,1,2,3,5} are a subset). YOLO11m ONNX class order == contract `extraction.class_names` (18 classes, validated at runtime by `_grade_detector_names`). **Therefore the new dianji display mapping is identity: label = int(class_id), 0-17, no remap.**
- The ultralytics adapter (`adapters/ultralytics.py`) wraps predict into a `.xyxy[0]`-style result; raw model accessible via `.model` (as the grade path already does).
- Grade feature contract: `weights/dianji_grade_lgbm_v1_contract.json` — extraction imgsz 640 / conf 0.25 / iou 0.45, normalized coords.
- Pre-existing untracked file `tools/d4_dianji_business_compare.py`: preserve untouched.

# Implementation Plan

1. `git status`; confirm no unexpected changes; branch `优化分支` (see encoding note above).
2. `models.yaml`:
   - `detectors.dianjie` → back to `kind: yolov5 / weights: 'weights/ryw.pt' / threshold: 0.25` (drop the ultralytics/imgsz options; keep/update comments to state the revert decision).
   - `detectors.dianji` → `kind: ultralytics / weights: 'weights/dianjienie_yolo11m_ryw18_640.onnx' / threshold: 0.25 / options.imgsz: 640` (mirror the structure D3 used for dianjie, including the SHA-256 comment `901ccf6006f44689483aaf6f0d1d1b8c18132a85c7d881be7e62fcdccc255b21`).
3. `inference/detectors.py`:
   - Rewire `_predict_dianji_grade_detections` (and `_grade_detector_names` usage) from `model_state.yolo` to `model_state.slf_yolo` (the dianji detector, now the YOLO11m ONNX). Class-name contract validation unchanged.
   - `_dianji_detect` display path: replace the slf_yolo + 5-class mapping with boxes from the SAME YOLO11m inference. Preferred: run the raw predict ONCE on `image_bgr` and derive both the normalized grade detections and the pixel-space display defects (`cx=(x1+x2)/2`, `cy=(y1+y2)/2`, `w=x2-x1`, `h=y2-y1` in original image pixels, `label=int(class_id)` identity 0-17). Refactor so grade and display share one inference (e.g. one helper returning raw xyxy/conf/cls, with two converters). Keep behavior correct when zero boxes (empty defects list, grade features all-zero detection side).
   - Keep `_legacy_dianji_type`, `_fallback_dianji_grade`, response shape, and all new fields unchanged.
   - Failure semantics: grade-model failure → resnet fallback for grade fields while display still comes from YOLO11m; dianji detector failure → `code 0` as before.
4. `tools/parity_dianji_grade.py`: it currently calls `model_state.load_dianjie_detector()` to obtain the YOLO11m instance for `_predict_dianji_grade_detections` — after rewiring, it must load `load_dianji_detector()` instead. Update accordingly.
5. `tools/smoke_dianji_grade.py`: extend `_validate_api_result` to assert every defect label is an int in 0-17 (identity 18-class passthrough).
6. Update stale comments/docstrings touched by this change (e.g. the old "映射表…" comment in the dianji display path, models.yaml comments).

# Allowed Scope

- May change: `models.yaml`, `inference/detectors.py`, `tools/parity_dianji_grade.py`, `tools/smoke_dianji_grade.py`.
- Do NOT touch: `inference/models.py` (`_validate_classifier_head` / D1 fix stays), `inference/grade_features.py`, `runtime_state.py`, weights files, `tools/d4_dianji_business_compare.py`, anything outside the repo.
- Do not commit, push, reset, rebase. Do not overwrite pre-existing user changes.

# Constraints

- Keep the LightGBM grade path semantics byte-identical (same contract params, same normalized detection dict format, same fallback).
- Identity passthrough means NO filtering of classes: all 18 ids may appear in dianji defects.
- Match surrounding code style; comments in Chinese.
- Keep single-pass inference (do not run YOLO11m twice per request).

# Verification

- `.venv/Scripts/python.exe tools/parity_dianji_grade.py --per-grade 3` → must still print `"max_abs_feature_diff": 0.0` and `"grade_id_matches": 12`, `"status": "PASS"`.
- `.venv/Scripts/python.exe tools/smoke_dianji_grade.py` → PASS, defect labels 0-17.
- `.venv/Scripts/python.exe tools/smoke_dianjie.py "E:\PythonProject\优化\dianjienie_ryw18_oofconsensus_v1_dev_only_r1\images" --limit 3` → PASS (box counts will differ from the ONNX era — ryw.pt is back; internal API-vs-direct consistency must hold).
- Fallback fault injection (temporarily point grade weights to a missing path in a throwaway check, not committed config): endpoint still returns code 1 with `grade_source: "resnet_fallback"`.
- `git diff --stat` — only the 4 allowed files.

# Acceptance Criteria

- dianjie endpoint serves ryw.pt detections again; dianji endpoint serves YOLO11m detections with identity 0-17 labels + LGBM grade fields.
- Parity still 0.0 / 12-of-12 (proves grade features unchanged by the rewiring).
- All three smoke/fault-injection checks PASS.
- Single YOLO11m inference per dianji request.

# Required Final Report

## Context Gathered
## Changes Made
## Verification Results
## Remaining Issues
