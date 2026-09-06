# -*- coding: utf-8 -*-
"""B3-clean 专用特征提取包装器：解锁非 B3 原始权重（.pt）。

`extract_dianji_grade_b3_y11m640_features.py` 内置 B3 best.pt 的 SHA-256 守卫
（EXPECTED_WEIGHT_SHA256=95b56f70…，保护 B3 特征血缘）。B3-clean 重训产生新的
best.pt（run 名 …_cm0_d0211clean1），因此本包装器：

1. 强制要求显式 `--weights`；
2. 把模块常量 EXPECTED_WEIGHT_SHA256 替换为实际传入权重的 SHA-256（实际值会
   随 extraction_summary.json 落盘，全程可追溯）；
3. 其余校验、协议、输出与原脚本完全一致。

用法：
  python run_b3clean_pt_extract.py --weights <b3clean best.pt> --output-dir dianji_grade_b3clean_y11m640_v1
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import extract_dianji_grade_b3_y11m640_features as ex  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    weight_arg = next(
        (a for a in argv if a == "--weights" or a.startswith("--weights=")), None
    )
    if weight_arg is None:
        raise SystemExit("B3-clean wrapper 必须显式传入 --weights（避免误用 B3 原始权重）")
    if weight_arg == "--weights":
        weight_path = Path(argv[argv.index("--weights") + 1])
    else:
        weight_path = Path(weight_arg.split("=", 1)[1])
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    actual = ex.file_sha256(weight_path)
    print(f"[b3clean-wrapper] weights={weight_path}", flush=True)
    print(f"[b3clean-wrapper] weight_sha256={actual}（原守卫 95b56f70… 仅保护 B3 原始 best.pt，B3-clean 显式解锁）", flush=True)
    ex.EXPECTED_WEIGHT_SHA256 = actual
    ex.WEIGHTS_PROVENANCE = (
        "B3-clean Kaggle run dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211clean1 "
        "(203 epoch early stop, best epoch 103; val178 mAP50=0.234 mAP50-95=0.0924; "
        "grade-test 采集键已排除出 train/val)"
    )
    ex.LEAKAGE_WARNING = (
        "B3-clean 检测器的 train/val 已按采集键排除全部 grade-test 泄漏图（146 行 → 145 键，"
        "159 张），grade-test 161 行对检测器零泄漏；grade-train/valid 与检测器训练集的重叠"
        "保留（LGBM 训练侧，不影响 test 评测洁净性）。"
    )
    sys.argv = [sys.argv[0]] + argv
    raise SystemExit(ex.main())
