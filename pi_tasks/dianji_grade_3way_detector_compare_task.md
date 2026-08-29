# Role

You are the implementation executor. Kimi Code owns planning and final acceptance. Work autonomously only in the exact working directory supplied by the launcher (`E:\PythonProject\优化`).

# Objective

Run a fair four-arm comparison of detector weights as feature extractors for 电积镍 (dianjinie) grade classification with LightGBM, on the SAME 1072-image grade view, SAME train/valid/test split (750/161/161, seed 42), SAME training protocol. Arms: ① ryw.pt@1536 (historical, reuse), ② YOLO11m@640 (already produced by a prior executor, reuse), ③ slf.pt@640 (NEW extraction), ④ ryw.pt@640 (NEW extraction, control arm to isolate the imgsz confound). Deliver a side-by-side comparison report.

# Current Evidence

- **Grade view (READ-ONLY)**: `E:\PythonProject\archives\dianjinie_grade_classification_v1` — 1072 images, classes `3#`/`9950`/`9996`/`bad`, split 750/161/161. It was MOVED from `E:\PythonProject\datasets\...`; older CSVs may embed stale paths — always resolve under the archives root.
- **Existing v3 features**: `E:\PythonProject\优化\dianji_grade_v3\features_v3_full.csv` — arm ① source (43 BASE detection cols from ryw.pt@1536 + 20 `tx__` texture cols). Historical metrics with `train_dianji_grade_v3.py` config `v3_texture_only` (63 cols): test acc=0.7764 / macroF1=0.6631; reject at valid-quantile coverage 0.70 → threshold 0.7397, test coverage 0.714, accepted acc=0.8696. Also historical base-only (43 cols, no texture): acc=0.6398 / macroF1=0.4131.
- **Arm ② inputs**: `E:\PythonProject\优化\dianji_grade_y11m640_v1\` (produced by a prior Codex task): YOLO11m ONNX (imgsz 640, conf 0.25, iou 0.45) re-extracted 43 BASE cols + reused 20 tx__ cols, features CSV + training metrics + `deploy_contract.json`. Read its report for exact filenames; if anything is missing or malformed, STOP and report instead of regenerating it yourself.
- **Training entry**: `E:\PythonProject\优化\train_dianji_grade_v3.py` — CLI has `--features`, `--output-dir`, `--seed` (default 42). Config definitions live near line 294; `BASE_COLUMNS` (43 cols) at lines 40-45; `GRADES = ["3#","9950","9996","bad"]`. The script's feature-set configs are keyed by name; you may need to add config entries for the new arms — keep patches minimal and backward-compatible.
- **Feature aggregation reference**: `E:\PythonProject\优化\extract_dianji_grade_features.py` — historical ryw.pt@1536 extraction; read it (and its imports for A/B/C class-group definitions and plate-area normalization) before writing any new extraction. Also read `E:\PythonProject\优化\extract_dianji_grade_y11m640_features.py` (prior task's script) — it already reproduces the 43-col aggregation for a new detector at 640; adapt it rather than starting from scratch.
- **Arm ③ weight**: `E:\PythonProject\ImetricCodes\JinchuanPythonWebapi\weights\slf.pt` (READ-ONLY), SHA-256 `d15831042281747965d0a4f4eb2dd85c95b121e0112fbd90ea2e200cd8dbeb40`, YOLOv5 architecture, 5 classes in id order: 0=waterstain, 1=waterblock, 2=colorstain, 3=nodule, 4=bubble. Production runs it at 640 (YOLOv5 default), threshold 0.25.
- **Arm ④ weight**: `E:\PythonProject\Tools\ModelAutoLabelTool\weights\yolov5\ryw.pt` (READ-ONLY; identical copy in the WebAPI repo `weights/ryw.pt`), YOLOv5, 18-class RYW18 (order: 0=wuzi, 1=waterstain, 2=yellowstain, 3=nodule, 4=airhole, 5=bubble, 6=bump, 7=fawu, 8=overfire, 9=NiOH, 10=greenstain, 11=suanwuran, 12=quejiao, 13=watermark, 14=youpao, 15=heibian, 16=mijixiaokeng, 17=fanxitong).
- **Environments**: LightGBM training in `E:\PythonProject\优化\.venv_classify`; YOLOv5 (.pt) inference environment — check how `extract_dianji_grade_features.py` loaded ryw.pt (torch hub / local yolov5 repo) and reuse exactly that mechanism; YOLO11m ONNX envs exist in `E:\PythonProject\models\.venv` and the WebAPI uv env if needed. Verify environments before long runs; no global installs.
- Aggregation definitions for the 43 BASE columns: `A_ratio/B_ratio/C_ratio/Airhole_ratio/has_nodule/total_bbox_union_image_ratio/detection_count` + `count__<cls>` + `area_ratio__<cls>` ×18. The A/B/C class groups are defined on RYW18 classes — find the exact grouping in the historical script's imports.

# Implementation Plan

1. Read the reference scripts and the prior task's outputs; report what exists and what's missing before coding.
2. **Arm ③ extraction** — new script `extract_dianji_grade_slf640_features.py`: run slf.pt (YOLOv5, imgsz 640, conf 0.25, iou 0.45, CPU) over all 1072 grade-view images; aggregate an ADAPTED base feature set: `detection_count`, `total_bbox_union_image_ratio`, `has_nodule` (class 3 count>0), `plate_area` if the historical recipe uses it, plus `count__<cls>` and `area_ratio__<cls>` for the 5 slf classes (13 detection cols). Do NOT fabricate A/B/C/Airhole groupings for the 5-class system — document their absence. Save per-box audit CSV. Join the 20 `tx__` columns from `features_v3_full.csv` by image identity (assert 1072/1072 matched).
3. **Arm ④ extraction** — extend or parameterize the existing YOLO11m extraction approach for ryw.pt at imgsz 640, conf 0.25, iou 0.45, producing the SAME 43-col BASE schema (18 classes), plus reused tx__ join. (Alternatively parameterize `extract_dianji_grade_features.py` for imgsz 640 if cleaner — your call, minimal patch.)
4. **Training**: for each new arm, run the same protocol (seed 42, deterministic, valid-based selection + rejection threshold, test once) on (a) detection-cols-only and (b) detection+tx, so the report can show texture contribution per arm. Reuse `train_dianji_grade_v3.py` via CLI; add config entries only if needed.
5. **Output**: everything under `E:\PythonProject\优化\dianji_grade_detector_compare_v1\` — per-arm feature CSVs, per-box CSVs, per-arm metrics JSONs, and a top-level `report.md` with the master comparison table: arm × (feature dims, imgsz, valid acc/macroF1, test acc/macroF1, per-grade F1, reject@0.70-coverage threshold + accepted acc). Include arm ① and ② numbers (clearly labeled as reused from prior runs with their source paths).
6. The report must contain a `## 局限` section: arm ③ has a smaller feature schema by construction (5-class); arms differ in imgsz (①1536 vs ②③④640) — arm ④ exists to isolate this; all results remain `unconfirmed_pending_enterprise` and grade↔batch confounding is unresolved.

# Allowed Scope

- May create: new extraction script(s) and `dianji_grade_detector_compare_v1\` in `E:\PythonProject\优化`.
- May patch `train_dianji_grade_v3.py` minimally for new feature-set configs (backward-compatible only); show the diff.
- Must NOT modify: anything under `E:\PythonProject\archives`, `E:\PythonProject\datasets`, `E:\PythonProject\models`, `E:\PythonProject\Tools`, `E:\PythonProject\ImetricCodes`; existing `dianji_grade_v2`/`dianji_grade_v3`/`dianji_grade_y11m640_v1` artifacts.
- Do not commit, push, or perform any git mutation.

# Constraints

- Windows; use the named venvs; CPU only; no GPU assumption; no detector training/fine-tuning.
- Chinese report text; English identifiers; new scripts support `--help` and a small-sample mode (`--limit 8`) which must be run before full runs.
- Windows 中文路径读取沿用 `np.fromfile + cv2.imdecode` 惯例。
- Do not access `E:\PythonProject\优化\l1_final_release_yolo11m_oofconsensusv1_ep083_20260815_2158` (locked release-validation).
- Do not tune hyperparameters to chase the historical 0.7764 — run the fixed protocol and report honestly.

# Verification

- Join integrity: 1072/1072 per arm, all image paths under the archives root.
- Fresh-process spot check: re-run extraction for 3 images (first/middle/last of sorted list) and confirm identical aggregates.
- Training runs: quote each arm's valid/test acc + macroF1 and reject threshold; `git -C E:\PythonProject\优化 status --short` shows only your new files.

# Acceptance Criteria

- Four arms comparable in one report.md with the master table and 局限 section.
- New extractions audited (per-box CSVs exist); reused numbers carry source-path labels.
- No scope violations; no git mutations.

# Required Final Report

Return these headings:

## Context Gathered
## Changes Made
## Verification Results
## Remaining Issues
