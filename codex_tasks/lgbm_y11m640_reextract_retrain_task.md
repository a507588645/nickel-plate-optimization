# Role

You are the implementation executor. Kimi Code owns planning and final acceptance. Work autonomously only in the exact working directory supplied by the launcher (`E:\PythonProject\优化`).

# Objective

Re-extract the 43 detector-derived BASE grade features on the 1072-image 电积镍 grade view using the frozen YOLO11m 640 weight (replacing the historical ryw.pt@1536 extraction), join with the existing 20 detector-independent texture (`tx__`) columns, retrain the LightGBM `v3_texture_only`-equivalent configuration (43 new + 20 tx = 63 columns), serialize a deployable model + contract JSON, and document the new metrics. This is the first half of the LightGBM WebAPI integration; WebAPI code changes are NOT in scope.

# Current Evidence

- **Grade view (READ-ONLY)**: `E:\PythonProject\archives\dianjinie_grade_classification_v1` — 1072 images, classes `3#`(78)/`9950`(564)/`9996`(384)/`bad`(46) split train/valid/test = 750/161/161 (seed 42, dup-group aware). Structure: `images/{train,valid,test}/{3#,9950,9996,bad}/`, `splits/{train,valid,test}.txt`, `manifests/samples.csv`, `dataset_summary.json`. NOTE: it was MOVED from `E:\PythonProject\datasets\...`; older CSVs may embed the old path — do not follow stale paths, use the archives location.
- **Existing features (REUSE texture columns)**: `E:\PythonProject\优化\dianji_grade_v3\features_v3_full.csv` (118 columns: 18 metadata + 43 BASE detection + ~30 `sp__` spatial + 20 `tx__` texture). The 20 `tx__` columns are pure image statistics (Sobel/LBP/GLCM on grayscale) computed on these same 1072 images — reuse them as-is by joining on image identity (verify with sha256 column if present, else image stem). Do NOT reuse any of the 43 BASE or `sp__` columns.
- **BASE feature aggregation reference**: `E:\PythonProject\优化\extract_dianji_grade_features.py` shows how the historical pipeline turned per-box detections into the 43 BASE columns: `A_ratio/B_ratio/C_ratio/Airhole_ratio/has_nodule/total_bbox_union_image_ratio/detection_count` + `count__<cls>` and `area_ratio__<cls>` for the 18 RYW18 classes. Read it and reproduce the aggregation EXACTLY (same definitions, same column names, same plate-area normalization), changing only the detector (YOLO11m ONNX, imgsz 640, conf 0.25, iou 0.45) instead of ryw.pt@1536. If A/B/C class-group definitions live elsewhere (e.g. a rules module), locate them by reading the script's imports.
- **New detector weight (READ-ONLY)**: `E:\PythonProject\models\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_finaldev_ep083\exports\onnx\final_fixed_epoch.onnx`, SHA-256 `901ccf6006f44689483aaf6f0d1d1b8c18132a85c7d881be7e62fcdccc255b21` (an identical copy exists at `E:\PythonProject\ImetricCodes\JinchuanPythonWebapi\weights\dianjienie_yolo11m_ryw18_640.onnx`). 18-class RYW18, order: 0=wuzi, 1=waterstain, 2=yellowstain, 3=nodule, 4=airhole, 5=bubble, 6=bump, 7=fawu, 8=overfire, 9=NiOH, 10=greenstain, 11=suanwuran, 12=quejiao, 13=watermark, 14=youpao, 15=heibian, 16=mijixiaokeng, 17=fanxitong.
- **Training entry**: `E:\PythonProject\优化\train_dianji_grade_v3.py` — reads a features CSV (`--features` arg or DEFAULT_FEATURES), config `v3_texture_only` = BASE_COLUMNS(43) + `tx__`(20), LightGBM deterministic seed 42, valid-based early stop/model selection, `reject_curve` thresholds from VALID confidence quantiles only, test evaluated once. Check its CLI args before running.
- **Environments**: training historically ran in `E:\PythonProject\优化\.venv_classify` (has lightgbm); YOLO11m ONNX inference needs `ultralytics==8.4.66` + onnxruntime — available in `E:\PythonProject\models\.venv` (Ultralytics 8.4.66, Torch 2.6.0) and in the WebAPI uv env. Verify with `--version`-style checks before use; do not install anything globally.
- **Historical reference metrics** (to be cited as history only, NOT a target): v3_texture_only test acc=0.7764 / macroF1=0.6631; rejection at valid-quantile coverage 0.70 gave threshold 0.7397, test coverage 0.714, accepted acc=0.8696.
- Grade→id mapping used by the training script: `GRADES = ["3#", "9950", "9996", "bad"]` (ids 0-3).

# Implementation Plan

1. Inspect `extract_dianji_grade_features.py` (and anything it imports for A/B/C grouping and plate-area normalization) and `train_dianji_grade_v3.py` CLI. Report what you found before coding.
2. Write a new script `extract_dianji_grade_y11m640_features.py` in `E:\PythonProject\优化`: run the YOLO11m ONNX (imgsz 640, conf 0.25, iou 0.45, CPU) over all 1072 grade-view images (from the archives path), dump per-box rows, and aggregate the 43 BASE columns with EXACTLY the historical definitions. Save per-box audit CSV as well.
3. Build the new training features file: metadata (image identity, split, grade) + 43 new BASE columns + 20 `tx__` columns reused from `features_v3_full.csv` (join by image identity; assert 1072/1072 matched, zero mismatches). Write to the new output dir (step 5).
4. Retrain with the SAME protocol as `train_dianji_grade_v3.py` (seed 42, deterministic, valid for early stop + rejection threshold, test once) on the 63-column config. Prefer invoking the existing script with `--features`/`--output-dir` args; only patch the script if the CLI genuinely cannot express this, and keep any patch backward-compatible.
5. Output dir: `E:\PythonProject\优化\dianji_grade_y11m640_v1\` containing: per-box CSV, features CSV, training metrics/report, `reject_curve.csv`, the serialized model (`lgbm_model.txt` via `Booster.save_model`, plus a joblib/pickle copy if convenient), and `deploy_contract.json` with: ordered 63 feature names, grade list and grade→id mapping, rejection threshold + its valid-coverage, extraction config (weight SHA-256, imgsz, conf, iou), and `status: "unconfirmed_pending_enterprise"`.
6. Fresh-process round-trip: load `lgbm_model.txt` in a new process, rebuild features for 3 test-split images from scratch via your new extraction script, predict, and confirm the predicted ids match the training-time predictions for those images.

# Allowed Scope

- May create: the new extraction script and `dianji_grade_y11m640_v1\` output dir in `E:\PythonProject\优化`.
- May patch `train_dianji_grade_v3.py` ONLY if its CLI cannot point at a different features CSV/output dir; keep backward compatibility and show the diff.
- Must NOT modify: anything under `E:\PythonProject\archives`, `E:\PythonProject\datasets`, `E:\PythonProject\models`, `E:\PythonProject\ImetricCodes`; existing `dianji_grade_v2`/`dianji_grade_v3` artifacts.
- Do not commit, push, or perform any git mutation.

# Constraints

- Windows; use the venvs named above, no global installs.
- CPU only; no GPU assumption; no detector training.
- Chinese for report text; code identifiers English; new script supports `--help` and a `--dry-run`/small-sample mode (e.g. `--limit 8`) which you must run before the full run.
- Windows中文路径读取沿用 `np.fromfile + cv2.imdecode` 惯例。
- Do not access `E:\PythonProject\优化\l1_final_release_yolo11m_oofconsensusv1_ep083_20260815_2158` (locked release-validation).

# Verification

- Feature join: 1072/1072 rows matched, no stale-path reads (assert every image path resolved under the archives root).
- Schema: new features CSV's 43 BASE + 20 tx__ column names identical to `features_v3_full.csv`'s corresponding names.
- Training outputs: valid/test acc + macroF1, per-grade report, reject_curve.csv; quote them in your report alongside the historical reference values labeled as history.
- Round-trip check from plan step 6 passes.
- `git -C E:\PythonProject\优化 status --short` shows only your new files.

# Acceptance Criteria

- 1072 images re-extracted with YOLO11m@640; features joined and trained; model serialized; contract JSON complete and self-consistent.
- New metrics documented honestly (they may differ from historical 0.7764/0.6631 — that is expected and acceptable; report, don't tune).
- Rejection threshold derived from valid only; test used exactly once.

# Required Final Report

Return these headings:

## Context Gathered
## Changes Made
## Verification Results
## Remaining Issues
