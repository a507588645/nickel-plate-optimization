# Role

You are the implementation executor. Kimi Code owns planning and final acceptance. Work autonomously only in the exact working directory supplied by the launcher (`E:\PythonProject\ImetricCodes\JinchuanPythonWebapi`, git branch `优化分支`).

# Objective

Prepare the field deployment of the frozen L1 detection candidate in the WebAPI repo: (D1) fix the `.pth` classifier-head overwrite bug, (D2) switch the 电解镍 (dianjie) detector from `ryw.pt` (YOLOv5) to the frozen YOLO11m 640 ONNX weight, (D3) run an end-to-end smoke test through the FastAPI chain. Do NOT switch the 电积镍 (dianji) detector in this batch.

# Current Evidence

- Repo `E:\PythonProject\ImetricCodes\JinchuanPythonWebapi` is on branch `优化分支` with a clean tree. Branch `308` is the production mainline — do not touch it.
- **D1 bug**: `inference/models.py` lines 118-119 (`load_resnet`) and 133-134 (`load_slf_resnet`) overwrite the trained `fc` head with a fresh `nn.Linear` AFTER loading the weight file, with no `load_state_dict` afterwards. Result: classifier outputs are randomly initialized and change across process restarts. Measured ground truth: `weights/ryw.pth` is a ResNet18 with `fc.out_features == 2`; `weights/slf.pth` is a ResNet18 with `fc.out_features == 4`. `models.yaml` `classifiers` already declares `num_classes: 2` / `4`.
  - Impact in calling code: `inference/detectors.py:51` uses `resnet` in `_dianjie_detect` (only affects the nodule+good branch at line 133); `inference/detectors.py:180` uses `resnet` (NOT `slf_resnet`) in `_dianji_detect` where `predicted_idx` fully decides the grade `type_` — so the dianji grade output is currently effectively random.
- **D2 weight source (read-only, do not modify)**: deployment ONNX at `E:\PythonProject\models\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_finaldev_ep083\exports\onnx\final_fixed_epoch.onnx`, SHA-256 = `901ccf6006f44689483aaf6f0d1d1b8c18132a85c7d881be7e62fcdccc255b21`, input size 640, FP32, opset 19, static batch=1, exported from Ultralytics 8.4.66. The sibling `best.pt` (SHA-256 `bb501e224b48e610aac5c05293d0b3c63ef9eccad792712466a25803bd4de521`) may be READ for class-name comparison only.
- **Class-order constraint**: `inference/detectors.py` business rules reference NUMERIC label ids (`label == 3` = nodule, `BUSINESS_RULES.dianjie.non_water_labels`, mapping table at lines 198-204). The new weight is 18-class RYW18 and was designed to match `ryw.pt`'s class order, but this MUST be verified empirically before switching: load `weights/ryw.pt` and the new weight and compare the full ordered `names` list. If they differ, STOP, do not switch, and report the mismatch.
- **Adapter risk**: `adapters/ultralytics.py` was added as a placeholder (git log `267c355`); verify whether it is functional. YOLO11 ONNX raw output is `(1, 4+nc, 8400)` with NO objectness column — the YOLOv5 `DetectMultiBackend` NMS path (`adapters/multibackend.py`) expects an objectness column and is likely INCOMPATIBLE. Preferred path: `kind: ultralytics` adapter, loading the `.onnx` through Ultralytics' own runtime (handles decode+NMS natively). If the ultralytics adapter is a non-functional stub, implement the minimal adapter functionality needed; falling back to the `.pt` weight via the ultralytics adapter is acceptable if ONNX via ultralytics fails.
- **Interface contract**: `detectors.py:55-56` calls `model_state.yolo(image)` and consumes `results.xyxy[0]` as rows `[x1, y1, x2, y2, conf, cls]`. The adapter layer (`adapters/base.py`) is expected to normalize this; keep that contract.
- Dependency note: the repo uses `uv` (`uv.lock` present). `models.yaml` comments already document `uv add ultralytics` as the procedure. If you add it, pin `ultralytics==8.4.66` (the training/export version) to avoid decode behavior drift. CPU-only environment is fine; do not assume CUDA.
- `weights/` currently holds `ryw.pt`, `slf.pt`, `ryw.pth`, `slf.pth`, `best.pt` etc. Keep `ryw.pt` in place — the switch must be config-level and reversible.
- Sample images for smoke test (READ-ONLY): `E:\PythonProject\优化\dianjienie_ryw18_oofconsensus_v1_dev_only_r1\images` (pick any 3-5 `.jpg`). Do not modify anything under `E:\PythonProject\优化` or `E:\PythonProject\datasets`.

# Implementation Plan

1. **D1** — In `inference/models.py`, fix `load_resnet` and `load_slf_resnet`: after `load_encrypted_torch(...)`, keep the weight file's own `fc`; assert `model.fc.out_features == cfg.get("num_classes", ...)` and raise a clear `RuntimeError` naming the weight file and both dimensions on mismatch. Remove the `nn.Linear` overwrite. Keep `eval()` / `cuda()` behavior. If the saved model lacks `fc` entirely, fail loudly with a clear message.
2. **D1 verify** — In two separate fresh Python processes, load `ryw.pth` via the fixed loader and run `classification_transform` + forward on one fixed sample image; confirm identical `predicted_idx` across processes. Also confirm `slf.pth` loads with `fc.out_features == 4`.
3. **D2 verify class order** — Script (temporary, not committed): load `weights/ryw.pt` (YOLOv5 hub or torch load — use whatever the repo already provides) and the new ONNX/pt; print both ordered class-name lists side by side; they must be identical 18-class lists. Attach the output to your report. If mismatch: stop here.
4. **D2 switch** — Copy the ONNX to `weights/dianjienie_yolo11m_ryw18_640.onnx`, verify SHA-256 matches `901ccf60...` after copy. Update `models.yaml` `detectors.dianjie` to `kind: ultralytics` + the new weights path, `threshold: 0.25` unchanged, with a short Chinese comment noting the source run directory and SHA-256. Implement/fix the ultralytics adapter as needed to satisfy the `results.xyxy[0]` contract (image input may be a path string or BGR ndarray; `_dianjie_detect` passes the original image). If the ultralytics-ONNX path cannot be made to work, fall back to copying `best.pt` as `weights/dianjienie_yolo11m_ryw18_640.pt` and document why.
5. **D3 smoke** — Using FastAPI `TestClient` (or a live `uvicorn` instance if simpler), call the dianjie detection endpoint with 3-5 sample images from the dev-set images dir; assert `code == 1`, defect label ids within the valid 18-class range, and sane box coordinates. Additionally run the new weight directly (same conf=0.25) on one sample image outside the API and confirm the box count via the API is within ±2 of the direct run. Record outputs.
6. Keep code style consistent with the file (Chinese comments, existing logging patterns). Update stale comments in `models.yaml` / `inference/models.py` that describe the old behavior.

# Allowed Scope

- May change: `inference/models.py`, `adapters/ultralytics.py` (and `adapters/base.py` ONLY if strictly required), `models.yaml`, `pyproject.toml`/`uv.lock` (only for the pinned ultralytics dependency), new weight file(s) under `weights/`, a new smoke-test script under the repo (e.g. `tools/` or repo root, your choice, committed as a file).
- Must NOT change: `inference/detectors.py` business rules, `detectors.dianji` config (stays `slf.pt`), threshold values, anything on branch `308`, anything outside the working directory except READING the two weight files and the sample images listed above.
- Preserve all unrelated and pre-existing user changes.
- Do not commit, push, reset, rebase, or rewrite history. Leave the working tree dirty with your changes for Kimi's review.

# Constraints

- Windows environment; Git Bash available; use `uv run` or the repo's existing environment mechanism for Python.
- No GPU assumption; CPU inference only for verification.
- Do not retrain, fine-tune, or export any model.
- Do not access `E:\PythonProject\优化\l1_final_release_yolo11m_oofconsensusv1_ep083_20260815_2158` (frozen release-validation, access is locked).
- Search the repository before editing; do not guess APIs. If `encrypt_models.py`/`.enc` machinery interacts with loading, keep it working.

# Verification

- Run: the D1 two-process determinism check (step 2), the class-order comparison (step 3), and the D3 smoke test (step 5). Include exact commands and outputs.
- Run: `git diff --stat` and inspect the full diff for unintended changes; include it in the report.
- If the repo has an existing test suite or `python -m compileall` style sanity check, run it.

# Acceptance Criteria

- `ryw.pth` / `slf.pth` load with their trained heads intact; dimension assertions pass; deterministic outputs across two fresh processes.
- Class-order verification output shows the new weight's 18 classes exactly match `ryw.pt` order (or a documented STOP with the mismatch).
- `models.yaml` `detectors.dianjie` points at the new frozen weight; `detectors.dianji` untouched; thresholds unchanged.
- Smoke test: dianjie endpoint returns `code == 1` for all sample images with valid label ids; API box count ≈ direct-run box count (±2) on the compared image.
- SHA-256 of the copied weight file matches the source hash.

# Required Final Report

Return these headings:

## Context Gathered
## Changes Made
## Verification Results
## Remaining Issues
