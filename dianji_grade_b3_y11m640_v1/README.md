# dianji_grade_v3 产物说明

- 状态：**unconfirmed_pending_enterprise**，企业确认前不得用于生产。
- 内容：方案 4（空间/纹理特征）与方案 5（拒判分析）对 v2 基线的对照。
- 文件：
  - `features_v3_full.csv` / `features_spatial.csv` / `features_texture.csv`：特征（提取脚本 `extract_dianji_grade_v3_features.py`）
  - `model_comparison.csv`：三模型 × 四特征集 × valid/test
  - `metrics_*.json`：各特征集详细指标与混淆矩阵
  - `reject_curve.csv`：拒判曲线（阈值仅来自 valid）
  - `report.md`：结论报告
- 基线对齐：slf_base13 重训 test LGBM acc=0.6708 / macroF1=0.4744（03:00 条目 0.640/0.413）
- 生成时间：2026-09-05T13:48:26
