# 电积镍等级 v3：空间/纹理特征 + 拒判分析报告

> ## ⚠️ 未经企业确认 / UNCONFIRMED
> 全部结果 `unconfirmed_pending_enterprise`，企业书面标准确认前不得上线。
> 小样本警告：3#=46、bad=28（test 各 7/4 张），少数类指标不稳。

- 生成时间：`2026-07-23T04:43:46`
- 特征来源：`dianji_grade_v3\features_v3_full.csv`（v2b conf0.25 43 列 + 空间 + 纹理）
- 协议：train 训练、valid 选模/早停/定拒判阈值、test 仅最终一次；seed=42
- v2 基线重训行（v2b_base43）test LGBM：acc=0.6233 / macroF1=0.4181
  （03:00 条目参考值 0.640/0.413，漂移 -0.0167，±0.01 内视为自洽）

## 1. 方案 4：特征增量对照（LightGBM, test）

| 特征集 | 列数 | valid macroF1 | test acc | test macroF1 | F1_3# | F1_9996 |
| --- | --- | --- | --- | --- | --- | --- |
| v2b_base43 | 43 | 0.3746 | 0.6233 | 0.4181 | 0.182 | 0.652 |
| v3_spatial_only | 73 | 0.4440 | 0.6507 | 0.4386 | 0.200 | 0.667 |
| v3_texture_only | 63 | 0.6118 | 0.7055 | 0.6074 | 0.545 | 0.702 |
| v3_full | 93 | 0.5095 | 0.7397 | 0.5623 | 0.222 | 0.750 |

- 最优特征集：**v3_texture_only**；相对 v2 基线重训 Δacc=+0.0822、ΔmacroF1=+0.1893。
- 全模型对照见 `model_comparison.csv`（majority/decision_tree/lightgbm × valid/test）。
- 最优特征集 LightGBM 特征重要性 top20：

- `tx__lbp_bin00`: 2849.9290
- `tx__glcm_correlation`: 2115.1449
- `tx__lbp_bin06`: 1250.4206
- `tx__lbp_bin03`: 1122.1156
- `tx__gray_mean`: 1093.7700
- `tx__grad_var`: 1076.3393
- `tx__gray_var`: 952.8212
- `B_ratio`: 529.1863
- `tx__glcm_energy`: 513.5664
- `total_bbox_union_image_ratio`: 511.8993
- `area_ratio__bubble`: 457.7837
- `tx__grad_mean`: 379.4730
- `area_ratio__watermark`: 372.8810
- `tx__lbp_bin07`: 362.0586
- `tx__lbp_bin01`: 334.6731
- `tx__lbp_bin05`: 326.1173
- `tx__lbp_bin02`: 324.2120
- `tx__lbp_bin08`: 276.0018
- `area_ratio__bump`: 272.3571
- `count__bump`: 210.6587

## 2. 方案 5：拒判分析（最优配置 = v3_texture_only + LightGBM）

阈值仅由 valid 置信度分位确定；test 一次性报告。拒判样本业务上记 `manual_review`。

| 目标覆盖率 | 阈值(valid) | test 实际覆盖 | test acc | test macroF1 |
| --- | --- | --- | --- | --- |
| 1.00 | 0.0000 | 1.000 | 0.7055 | 0.6074 |
| 0.90 | 0.5587 | 0.918 | 0.7388 | 0.6387 |
| 0.80 | 0.6276 | 0.815 | 0.7983 | 0.6880 |
| 0.70 | 0.6936 | 0.712 | 0.8365 | 0.6569 |
| 0.60 | 0.7677 | 0.596 | 0.8736 | 0.7369 |

- acc≥0.85 的覆盖点：达到（覆盖率≈60%）。

## 3. 结论与局限

- 空间/纹理特征对 test 的增量：Δacc=+0.0822、ΔmacroF1=+0.1893（<0.01 视为无实质提升）。
- 3#/bad 小样本约束不变；9950↔9996 混对是否缓解见 `metrics_v3_full.json` 混淆矩阵。
- 拒判可作为部署保底策略，但阈值与覆盖率均为未确认候选，需企业确认业务可接受精度。

## 4. 复现

```text
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe extract_dianji_grade_v3_features.py
.\.venv_classify\Scripts\python.exe train_dianji_grade_v3.py
```
