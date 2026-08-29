# Correction Round 2 — Fix smoke_dianjie.py direct baseline after dianjie revert

## What happened

Your delivery is correct, but one verification gap remains: `tools/smoke_dianjie.py:93-106` hardcodes `weights/dianjienie_yolo11m_ryw18_640.onnx` + `UltralyticsDetector` for the direct-inference baseline. Now that `detectors.dianjie` is back to `yolov5 / weights/ryw.pt`, the script compares API (ryw.pt) against a direct run of a DIFFERENT model (ONNX), which is meaningless and fails the ±2 check.

## What to do now

Update `tools/smoke_dianjie.py` so the direct baseline is built from the LIVE config instead of a hardcoded model:

- Read `MODELS_CONFIG["detectors"]["dianjie"]` (or re-read models.yaml) and construct the direct detector via the same `create_detector(cfg["kind"], weights=..., device=..., threshold=..., **options)` path that `inference/models.py load_dianjie_detector()` uses (mirror `_resolved_path` and `_detector_kwargs` behavior).
- Keep the rest of the script (API boot, field validation, ±2 comparison, JSON summary) unchanged.
- The script must work for ANY configured dianjie detector kind (yolov5 today, ultralytics if it changes back later) without further edits.

`tools/smoke_dianjie.py` is now explicitly in allowed scope. Everything else from the original task remains unchanged; do not touch other files.

## Verification

- `.venv/Scripts/python.exe tools/smoke_dianjie.py "E:\PythonProject\优化\dianjienie_ryw18_oofconsensus_v1_dev_only_r1\images" --limit 3` → PASS, with API vs direct count delta within ±2 (same model on both sides now).
- Re-run `.venv/Scripts/python.exe tools/smoke_dianji_grade.py` → PASS (regression check).

# Required Final Report

## Changes Made
## Verification Results
## Remaining Issues
