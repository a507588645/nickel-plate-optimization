# 电积镍等级 v3：空间/纹理特征 + 拒判分析报告

> ## ⚠️ 未经企业确认 / UNCONFIRMED
> 全部结果 `unconfirmed_pending_enterprise`，企业书面标准确认前不得上线。
> 小样本警告：3#=46、bad=28（test 各 7/4 张），少数类指标不稳。

- 生成时间：`2026-09-05T01:54:21`
- 特征来源：`dianji_grade_v3\features_v3_full.csv`（v2b conf0.25 43 列 + 空间 + 纹理）
- 协议：train 训练、valid 选模/早停/定拒判阈值、test 仅最终一次；seed=42
- v2 基线重训行（v2b_base43）test LGBM：acc=0.6149 / macroF1=0.4309
  （03:00 条目参考值 0.640/0.413，漂移 -0.0251，±0.01 内视为自洽）

## 1. 方案 4：特征增量对照（LightGBM, test）

| 特征集 | 列数 | valid macroF1 | test acc | test macroF1 | F1_3# | F1_9996 |
| --- | --- | --- | --- | --- | --- | --- |
| v2b_base43 | 43 | 0.5420 | 0.6149 | 0.4309 | 0.154 | 0.662 |
| v3_spatial_only | 43 | 0.5420 | 0.6149 | 0.4309 | 0.154 | 0.662 |
| v3_texture_only | 63 | 0.6697 | 0.7081 | 0.6208 | 0.545 | 0.703 |
| v3_full | 63 | 0.6697 | 0.7081 | 0.6208 | 0.545 | 0.703 |

- 最优特征集：**v3_texture_only**；相对 v2 基线重训 Δacc=+0.0932、ΔmacroF1=+0.1899。
- 全模型对照见 `model_comparison.csv`（majority/decision_tree/lightgbm × valid/test）。
- 最优特征集 LightGBM 特征重要性 top20：

- `tx__lbp_bin00`: 4105.7093
- `tx__glcm_correlation`: 2537.1585
- `tx__lbp_bin06`: 1055.2578
- `tx__gray_mean`: 1044.9054
- `tx__glcm_energy`: 897.1670
- `area_ratio__waterstain`: 852.9192
- `tx__gray_var`: 617.9045
- `area_ratio__overfire`: 522.6271
- `total_bbox_union_image_ratio`: 504.2952
- `tx__lbp_bin07`: 484.6404
- `C_ratio`: 470.3404
- `tx__lbp_bin05`: 413.8762
- `tx__lbp_bin03`: 376.2841
- `A_ratio`: 346.3346
- `count__waterstain`: 338.9418
- `tx__grad_var`: 333.5798
- `tx__lbp_bin01`: 299.0495
- `B_ratio`: 298.4647
- `tx__lbp_bin02`: 278.3435
- `tx__lbp_bin08`: 270.8770

## 2. 方案 5：拒判分析（最优配置 = v3_texture_only + LightGBM）

阈值仅由 valid 置信度分位确定；test 一次性报告。拒判样本业务上记 `manual_review`。

| 目标覆盖率 | 阈值(valid) | test 实际覆盖 | test acc | test macroF1 |
| --- | --- | --- | --- | --- |
| 1.00 | 0.0000 | 1.000 | 0.7081 | 0.6208 |
| 0.90 | 0.5917 | 0.839 | 0.7704 | 0.6755 |
| 0.80 | 0.6505 | 0.758 | 0.7869 | 0.6426 |
| 0.70 | 0.7478 | 0.640 | 0.8350 | 0.7120 |
| 0.60 | 0.8137 | 0.584 | 0.8511 | 0.7195 |

- acc≥0.85 的覆盖点：达到（覆盖率≈58%）。

## 3. 结论与局限

- 空间/纹理特征对 test 的增量：Δacc=+0.0932、ΔmacroF1=+0.1899（<0.01 视为无实质提升）。
- 3#/bad 小样本约束不变；9950↔9996 混对是否缓解见 `metrics_v3_full.json` 混淆矩阵。
- 拒判可作为部署保底策略，但阈值与覆盖率均为未确认候选，需企业确认业务可接受精度。

## 4. 复现

```text
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe extract_dianji_grade_v3_features.py
.\.venv_classify\Scripts\python.exe train_dianji_grade_v3.py
```
