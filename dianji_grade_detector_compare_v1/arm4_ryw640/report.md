# 电积镍等级 v3：空间/纹理特征 + 拒判分析报告

> ## ⚠️ 未经企业确认 / UNCONFIRMED
> 全部结果 `unconfirmed_pending_enterprise`，企业书面标准确认前不得上线。
> 小样本警告：3#=46、bad=28（test 各 7/4 张），少数类指标不稳。

- 生成时间：`2026-08-28T15:35:03`
- 特征来源：`dianji_grade_v3\features_v3_full.csv`（v2b conf0.25 43 列 + 空间 + 纹理）
- 协议：train 训练、valid 选模/早停/定拒判阈值、test 仅最终一次；seed=42
- v2 基线重训行（v2b_base43）test LGBM：acc=0.6025 / macroF1=0.3900
  （03:00 条目参考值 0.640/0.413，漂移 -0.0375，±0.01 内视为自洽）

## 1. 方案 4：特征增量对照（LightGBM, test）

| 特征集 | 列数 | valid macroF1 | test acc | test macroF1 | F1_3# | F1_9996 |
| --- | --- | --- | --- | --- | --- | --- |
| v2b_base43 | 43 | 0.4219 | 0.6025 | 0.3900 | 0.308 | 0.614 |
| v3_spatial_only | 43 | 0.4219 | 0.6025 | 0.3900 | 0.308 | 0.614 |
| v3_texture_only | 63 | 0.6856 | 0.7578 | 0.6628 | 0.545 | 0.739 |
| v3_full | 63 | 0.6856 | 0.7578 | 0.6628 | 0.545 | 0.739 |

- 最优特征集：**v3_texture_only**；相对 v2 基线重训 Δacc=+0.1553、ΔmacroF1=+0.2728。
- 全模型对照见 `model_comparison.csv`（majority/decision_tree/lightgbm × valid/test）。
- 最优特征集 LightGBM 特征重要性 top20：

- `tx__lbp_bin00`: 3652.4789
- `tx__glcm_correlation`: 2976.4487
- `tx__lbp_bin06`: 1436.3911
- `tx__glcm_energy`: 1292.3995
- `tx__gray_mean`: 1289.8200
- `tx__gray_var`: 1114.3840
- `tx__lbp_bin07`: 841.9671
- `total_bbox_union_image_ratio`: 596.9004
- `tx__lbp_bin03`: 596.1136
- `tx__grad_var`: 532.5439
- `tx__lbp_bin08`: 514.0974
- `B_ratio`: 499.6622
- `detection_count`: 447.4620
- `tx__lbp_bin02`: 444.5421
- `tx__lbp_bin01`: 383.5002
- `tx__lbp_bin05`: 377.5960
- `tx__grad_mean`: 367.7612
- `area_ratio__watermark`: 298.1473
- `area_ratio__wuzi`: 289.9652
- `tx__lbp_bin09`: 256.4455

## 2. 方案 5：拒判分析（最优配置 = v3_texture_only + LightGBM）

阈值仅由 valid 置信度分位确定；test 一次性报告。拒判样本业务上记 `manual_review`。

| 目标覆盖率 | 阈值(valid) | test 实际覆盖 | test acc | test macroF1 |
| --- | --- | --- | --- | --- |
| 1.00 | 0.0000 | 1.000 | 0.7578 | 0.6628 |
| 0.90 | 0.5762 | 0.894 | 0.7708 | 0.6462 |
| 0.80 | 0.6596 | 0.820 | 0.7879 | 0.6677 |
| 0.70 | 0.7288 | 0.739 | 0.8067 | 0.7011 |
| 0.60 | 0.8348 | 0.627 | 0.8614 | 0.7053 |

- acc≥0.85 的覆盖点：达到（覆盖率≈63%）。

## 3. 结论与局限

- 空间/纹理特征对 test 的增量：Δacc=+0.1553、ΔmacroF1=+0.2728（<0.01 视为无实质提升）。
- 3#/bad 小样本约束不变；9950↔9996 混对是否缓解见 `metrics_v3_full.json` 混淆矩阵。
- 拒判可作为部署保底策略，但阈值与覆盖率均为未确认候选，需企业确认业务可接受精度。

## 4. 复现

```text
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe extract_dianji_grade_v3_features.py
.\.venv_classify\Scripts\python.exe train_dianji_grade_v3.py
```
