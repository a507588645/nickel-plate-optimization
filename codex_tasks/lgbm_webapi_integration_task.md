# Role

You are the implementation executor. Kimi Code owns planning and final acceptance. Work autonomously only in the exact working directory supplied by the launcher (`E:\PythonProject\ImetricCodes\JinchuanPythonWebapi`, git branch must be `优化分支` — verify with `git branch --show-current` before touching anything; if it is not `优化分支`, STOP and report).

# Objective

Replace the legacy ResNet grade branch inside `_dianji_detect`（电积镍 `/dianjiclass`）with the validated LightGBM grade model (`dianji_grade_y11m640_v1`), porting the 63-column feature extraction (43 detector-base + 20 texture) into the WebAPI repo, while keeping the defect-display path (slf_yolo) and the frontend response contract intact.

# Current Evidence

## Target code (all verified by Kimi on 2026-08-28)

- `inference/detectors.py:174-225` — `_dianji_detect(image: str | np.ndarray)`:
  - Grade branch: `model_state.resnet(image_tensor)` → `predicted_idx` → type mapping `2→type 2, 1→type 3, 0→type 4, else→type 1`. NOTE: it currently uses `model_state.resnet`（电解镍 2 类），so `predicted_idx==2` never fires — historical quirk, do not preserve it.
  - Defect path: `model_state.slf_yolo(image)` → 5-class → mapping `{0:1, 1:0, 2:2, 3:3, 4:5}` → `data.defects` list of `{label, cx, cy, w, h}`. **This path must stay byte-for-byte behavior-identical.**
  - Response shape: `{"code": 1, "msg": "success", "data": {"type": int, "defects": [...]}}`; failure: `{"code": 0, "msg": "dianji detection failed", "data": {}}`.
- `inference/models.py` — global model state + loaders; `load_dianjie_detector()` already creates `yolo` = ultralytics adapter on `weights/dianjienie_yolo11m_ryw18_640.onnx` (imgsz 640, threshold 0.25). `load_all_models()` calls all loaders. `preprocess.findcent` already exists in this repo (used by the historical extraction via project_root import).
- `models.yaml` — `detectors.dianjie` is the YOLO11m ONNX; `detectors.dianji` is still `slf.pt` (DO NOT change; D4 business gate pending user adjudication).
- `.venv`: onnxruntime 1.29.0 present; **lightgbm is NOT installed** — install `lightgbm==4.6.0` into `.venv` (training version per contract) and update the repo dependency manifest if one exists (check pyproject.toml / requirements*.txt).

## Grade model contract (single source of truth)

`E:\PythonProject\优化\dianji_grade_y11m640_v1\deploy_contract.json`:
- 63 features in exact order: 4 ratios (`A_ratio,B_ratio,C_ratio,Airhole_ratio`) + `has_nodule,total_bbox_union_image_ratio,detection_count` + 18×`count__*` + 18×`area_ratio__*` + 20×`tx__*`. Read the JSON for the authoritative ordered list.
- `grade_to_id`: `{"3#":0, "9950":1, "9996":2, "bad":3}`; LightGBM `num_class=4`, `best_iteration=186`, model file `lgbm_model.txt` (booster text format).
- Rejection: threshold `0.7362` on max class probability (valid-set 0.70 coverage quantile); below threshold → `manual_review=true`.
- Extraction config: YOLO11m ONNX sha256 `901ccf60...`, imgsz 640, conf 0.25, iou 0.45, cpu.

## Feature extraction reference implementations (READ-ONLY, in `E:\PythonProject\优化\`)

- 43 base: `extract_dianji_grade_features.py` — reuse semantics of `CLASS_NAMES` (18), label-group constants (`A_LABELS/B_LABELS/C_LABELS/AIRHOLE_LABELS/NODULE_LABEL`), `hardcoded_rect` (÷4, clamp x≤4069 y≤3554), `union_area`, `normalized_rect`, `features_from_detections(detections, W, H, plate_area, conf_thres=0.25)`, `compute_plate_area(image_bgr, project_root)` (resizes to (1024,888), calls `preprocess.findcent` → `pw*ph`, fallback = full image W×H). Detections fed to it are dicts `{label:int, conf:float, cx,cy,w,h}` with **normalized** cx/cy/w/h.
- 20 texture: `extract_dianji_grade_v3_features.py` — `texture_features(gray, short_side=1024)` plus helpers `resize_short_side`, `lbp_hist_10bin`, `glcm_props`. Pure numpy/cv2, no skimage. Input is the ORIGINAL full-resolution grayscale; function internally shrinks short side to ≤1024.
- Detection idiom (from `extract_dianji_grade_y11m640_features.py:285-355`): ultralytics `model.predict(source=bgr, imgsz=640, conf=0.25, iou=0.45)` → `boxes.xyxy/conf/cls` → normalize by image W/H into the dict format above. In WebAPI, reuse the already-loaded `model_state.yolo` instance — check `adapters/ultralytics.py` for how to get raw boxes or call the underlying model; ensure iou=0.45 is used for this path even if the adapter default differs.
- Grade→type mapping for the new branch (matches legacy semantics): `3#→4, 9950→3, 9996→2, bad→1`. `data.type` must ALWAYS be an int in {1,2,3,4} — frontend contract must not break.

# Implementation Plan

1. `git status` + confirm branch `优化分支`; preserve all pre-existing changes.
2. Install `lightgbm==4.6.0` into `.venv`; update dependency manifest if tracked.
3. Copy `E:\PythonProject\优化\dianji_grade_y11m640_v1\lgbm_model.txt` → `weights/dianji_grade_lgbm_v1.txt` and `deploy_contract.json` → `weights/dianji_grade_lgbm_v1_contract.json`.
4. New module `inference/grade_features.py`: port the 43-base + 20-texture extraction (numpy/cv2 only), exposing one function like `extract_grade_features(image_bgr, detections) -> dict[str, float]` returning all 63 contract-ordered features. Import `findcent` from the repo's own `preprocess`.
5. `models.yaml`: add a `grade_models.dianji_lgbm` section (weights, contract, threshold 0.7362, enabled: true). `models.py`: add `load_dianji_grade_model()` (load booster, validate `booster.feature_name()` equals the contract's 63-name order, fail loudly at startup if not), call it from `load_all_models()`, extend `unload_models()`.
6. `inference/detectors.py` `_dianji_detect`: replace the resnet grade block with: run YOLO11m (reuse `model_state.yolo`, conf 0.25 / iou 0.45 / imgsz 640) on the original-resolution BGR image → detections → `extract_grade_features` → LGBM predict (4-class probs) → grade/confidence → `type` via the mapping above. Response `data` gains: `grade` (str), `confidence` (float), `manual_review` (bool, confidence < 0.7362), `grade_status: "unconfirmed_pending_enterprise"`, `grade_source: "lgbm_y11m640_v1"`. Keep `defects` from the slf_yolo path unchanged. Image input: if `image` is a path, read with `np.fromfile`+`cv2.imdecode` (Chinese-path safe); if ndarray, use as-is (must be BGR 3-channel).
7. Graceful degradation: if the LGBM model failed to load or the grade pipeline raises, log the exception, fall back to the legacy resnet branch for `type`, and set `grade_source: "resnet_fallback"` (+ `grade_status: "fallback"`); the endpoint must not 500.
8. New `tools/smoke_dianji_grade.py`, mirroring `tools/smoke_dianjie.py` style: boot the app in-process, call `/dianjiclass` on a few sample images, assert `code==1`, `type ∈ {1,2,3,4}`, new fields present, `manual_review == (confidence < 0.7362)`.
9. Parity harness (script under `tools/` or a temp file, NOT committed if throwaway): for ≥10 images spanning all 4 grades from the read-only view `E:\PythonProject\archives\dianjinie_grade_classification_v1`, compute sha256 of file bytes, run the in-repo extraction + LGBM, and compare against `E:\PythonProject\优化\dianji_grade_y11m640_v1\features_y11m640_v1.csv` (join on sha256): max abs feature diff ≤ 1e-6 and identical predicted grade id for every image.

# Allowed Scope

- May change inside the WebAPI repo: `inference/detectors.py`, `inference/models.py`, new `inference/grade_features.py`, `models.yaml`, `weights/` (add the 2 new files only), `tools/smoke_dianji_grade.py` (new), dependency manifest, `.venv` package install.
- May read (never write): `E:\PythonProject\优化\` reference scripts/contract/CSV/model, `E:\PythonProject\archives\dianjinie_grade_classification_v1` images.
- Do NOT touch: `detectors.dianji` config (slf.pt stays), the dianjie endpoint behavior, `ryw.pt`/`slf.pt` weights, any file in `E:\PythonProject\优化`.
- Do not commit, push, reset, rebase, or rewrite history. Do not overwrite pre-existing user changes.

# Constraints

- Match existing code style (this repo uses mixed CRLF/LF; follow each file's dominant style; comments in Chinese like surrounding code).
- No new heavyweight dependencies: feature code must be numpy/cv2 only (no skimage, no scipy unless already present).
- Reuse `model_state.yolo`; do not load a second copy of the 18-class ONNX.
- Do not "fix" unrelated legacy quirks beyond the specified grade branch replacement.
- The 63-feature order must come from the bundled contract JSON at runtime (load and validate), not a hand-copied list that can drift.

# Verification

- `git diff --stat` for unintended changes.
- Run parity harness → report max abs feature diff and grade-id match count.
- Run `tools/smoke_dianji_grade.py` → PASS.
- Run existing `tools/smoke_dianjie.py` → PASS (no dianjie regression).
- Import check: `.venv/Scripts/python.exe -c "import inference.detectors"` (or equivalent) with no import errors.

# Acceptance Criteria

- Parity: max abs diff ≤ 1e-6 and 100% grade-id agreement on ≥10 view images vs `features_y11m640_v1.csv`.
- Smoke tests pass; response keeps `type`/`defects` and adds the 5 new fields.
- LGBM load-time validation rejects a mismatched feature order.
- Fallback path works (simulate by temporarily pointing weights at a missing file in a throwaway test, not in committed config).

# Required Final Report

## Context Gathered
## Changes Made
## Verification Results
## Remaining Issues
