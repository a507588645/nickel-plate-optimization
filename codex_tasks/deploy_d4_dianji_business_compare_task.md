# Role

You are the implementation executor. Kimi Code owns planning and final acceptance. Work autonomously only in the exact working directory supplied by the launcher (`E:\PythonProject\ImetricCodes\JinchuanPythonWebapi`, git branch `优化分支`).

# Objective

D4 cross-domain gate, phase 1 (analysis only, NO config switch): compare the current 电积镍 (dianji) detector `slf.pt` against the frozen YOLO11m 640 ONNX at the BUSINESS OUTPUT level on 电积镍 validation images, and produce a quantitative report plus a side-by-side review page of the largest disagreements for user adjudication. Do not change any production config.

# Current Evidence

- Current dianji detector: `weights/slf.pt` (YOLOv5 architecture, 5 classes, id order: 0=waterstain, 1=waterblock, 2=colorstain, 3=nodule, 4=bubble), production threshold 0.25 (see `models.yaml` `detectors.dianji`). Loaded via the repo's yolov5 adapter.
- Candidate: `weights/dianjienie_yolo11m_ryw18_640.onnx` (already in repo, committed in 8c436ff), 18-class RYW18, `kind: ultralytics`, imgsz 640, threshold 0.25. RYW18 order: 0=wuzi, 1=waterstain, 2=yellowstain, 3=nodule, 4=airhole, 5=bubble, 6=bump, 7=fawu, 8=overfire, 9=NiOH, 10=greenstain, 11=suanwuran, 12=quejiao, 13=watermark, 14=youpao, 15=heibian, 16=mijixiaokeng, 17=fanxitong.
- Same-name classes usable for cross-system comparison: waterstain (slf id 0 ↔ RYW18 id 1), nodule (3 ↔ 3), bubble (4 ↔ 5). `waterblock` (slf 1) and `colorstain` (slf 2) have NO confirmed mapping into RYW18 — do NOT invent or assume one; report them as slf-only classes. RYW18-only classes are candidate-only detections.
- Evaluation data (READ-ONLY, do not modify anything under it): `E:\PythonProject\datasets\Dianjinie0211`. The directory reportedly has duplicate layouts (`images/train` + `labels/train` and `train/images` + `train/labels`, similarly for valid); verify the actual layout first and use the valid split images (expected 432; record the real count). About half the images have no label file — irrelevant for this comparison (we compare detector outputs, not ground truth), but note it in the report.
- Production dianji behavior (`inference/detectors.py:174-225`): the grade `type` comes from the resnet classifier and is IDENTICAL for both detectors, so exclude it; compare only the defects list. The detector receives the ORIGINAL image (path or BGR ndarray), not the resized (1024,888) copy. Feed both detectors the original image.
- Both detectors are runnable in this repo's uv environment: yolov5 adapter for slf.pt, ultralytics adapter for the ONNX (both already registered). CPU only.
- Project rule: derived artifacts must be written under `E:\PythonProject\优化` in a clearly named new directory, never into dataset directories.

# Implementation Plan

1. Verify the Dianjinie0211 layout; enumerate the valid-split image list deterministically (sorted by filename). Record the count and layout in the report.
2. Write `tools/d4_dianji_business_compare.py`: for each valid image, run slf.pt (conf 0.25) and the YOLO11m ONNX (conf 0.25, imgsz 640) on the original image; record per-image: defect count, summed box area / image area, per-class counts for slf classes 0-4 and for RYW18 same-name ids (1, 3, 5), plus count of candidate-only detections (all other RYW18 ids). Save raw per-box rows (model, image, class id, conf, xywh) to support auditing.
3. Aggregate: distribution stats of per-image count delta and area-ratio delta (median/mean/p90/max), image-level detection rate for each model and its delta, same-name class total counts, and a table of the top 20 divergence images ranked by |Δcount| + |Δarea_ratio|.
4. Render the top 20 divergence images side-by-side (slf.pt annotations left, YOLO11m right, boxes + class id + conf drawn) and build a simple self-contained `review.html` (relative image paths or embedded base64, your choice, but the HTML must open standalone).
5. Write all outputs to a new timestamped directory `E:\PythonProject\优化\d4_dianji_business_compare_<YYYYMMDD_HHMMSS>\`: `report.md` (method, counts, stats, top-20 table, limitations), `per_image.csv`, `per_box.csv`, `review.html`, and the rendered images in a subdir.
6. The report must end with a `## 局限` section noting: no ground-truth comparison, no confirmed mapping for waterblock/colorstain, single confidence threshold 0.25, CPU inference, and that the pass/fail gate is the user's adjudication of the review page.

# Allowed Scope

- May create: `tools/d4_dianji_business_compare.py` in the repo; the output directory under `E:\PythonProject\优化` as specified.
- Must NOT change: `models.yaml`, any file under `inference/`, `adapters/`, or any existing repo file; nothing under `E:\PythonProject\datasets` or `E:\PythonProject\models`.
- Do not commit, push, reset, rebase, or rewrite history. Leave changes uncommitted for Kimi's review.
- Preserve all unrelated and pre-existing user changes.

# Constraints

- Windows, Git Bash available; run Python via `uv run --no-sync python ...` in the repo.
- CPU only; no GPU assumption; no training.
- Chinese for report text; code identifiers in English; match the repo's comment style.
- If the yolov5 adapter's output contract differs from what you expect, read `adapters/yolov5.py` and `adapters/base.py` before coding; do not guess.
- If runtime exceeds ~40 minutes for the full valid set, you may use a deterministic subsample (fixed seed 0, sorted list, every Nth image) of at least 150 images, and state this in the report.

# Verification

- Run the full comparison and attach: total images evaluated, wall time, per-model mean defect count, detection rates, and the top-5 rows of the divergence table.
- Spot-check: re-run 3 images (first, middle, last of the sorted list) in a fresh process and confirm identical per-image counts.
- Confirm `review.html` exists and references/contains exactly 20 rendered comparisons (or fewer if fewer divergent images exist).
- Confirm nothing outside the allowed scope was modified: `git status --short` must show only the new `tools/d4_dianji_business_compare.py` (untracked).

# Acceptance Criteria

- All valid-split images (or documented subsample) evaluated with both detectors; per-image and per-box CSVs complete.
- Quantitative report with the stats listed above and the 局限 section.
- `review.html` opens standalone with side-by-side annotated comparisons for top divergences.
- No production config touched; no dataset writes.

# Required Final Report

Return these headings:

## Context Gathered
## Changes Made
## Verification Results
## Remaining Issues
