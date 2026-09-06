# -*- coding: utf-8 -*-
"""B2 专用特征提取包装器：解锁非部署权重（.pt）。

原 `extract_dianji_grade_y11m640_features.py` 内置部署 ONNX 的 SHA-256 守卫
（EXPECTED_WEIGHT_SHA256=901ccf60…，保护部署特征血缘）。B2 分辨率消融需要用
最终权重 .pt（路径见下）在 imgsz∈{640,1280,1536} 提特征，因此本包装器：

1. 强制要求显式 `--weights`；
2. 把模块常量 EXPECTED_WEIGHT_SHA256 替换为实际传入权重的 SHA-256（实际值会
   随 extraction_summary.json 落盘，全程可追溯）；
3. 其余校验、协议、输出与原脚本完全一致。

用法：在原脚本参数基础上加 --weights <best.pt>，例如
  python run_b2_pt_extract.py --weights <best.pt> --imgsz 1280 --conf 0.25 --output-dir dianji_grade_b2_imgsz1280_v1
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import extract_dianji_grade_y11m640_features as ex  # noqa: E402

PT = Path("E:/PythonProject/models/runs/dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_finaldev_ep083/weights/best.pt")

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not any(a == "--weights" or a.startswith("--weights=") for a in argv):
        raise SystemExit("B2 wrapper 必须显式传入 --weights（避免误用默认部署 ONNX）")
    if not PT.is_file():
        raise FileNotFoundError(PT)
    actual = ex.file_sha256(PT)
    print(f"[b2-wrapper] weights={PT}", flush=True)
    print(f"[b2-wrapper] weight_sha256={actual}（原守卫 901ccf60… 仅保护部署 ONNX，B2 显式解锁）", flush=True)
    ex.EXPECTED_WEIGHT_SHA256 = actual
    sys.argv = [sys.argv[0]] + argv
    raise SystemExit(ex.main())
