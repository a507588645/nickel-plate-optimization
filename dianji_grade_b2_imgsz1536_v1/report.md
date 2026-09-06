# 电积镍等级 v3：空间/纹理特征 + 拒判分析报告

> ## ⚠️ 未经企业确认 / UNCONFIRMED
> 全部结果 `unconfirmed_pending_enterprise`，企业书面标准确认前不得上线。
> 小样本警告：3#=46、bad=28（test 各 7/4 张），少数类指标不稳。

- 生成时间：`2026-09-05T01:54:24`
- 特征来源：`dianji_grade_v3\features_v3_full.csv`（v2b conf0.25 43 列 + 空间 + 纹理）
- 协议：train 训练、valid 选模/早停/定拒判阈值、test 仅最终一次；seed=42
- v2 基线重训行（v2b_base43）test LGBM：acc=0.6398 / macroF1=0.4500
  （03:00 条目参考值 0.640/0.413，漂移 -0.0002，±0.01 内视为自洽）

## 1. 方案 4：特征增量对照（LightGBM, test）

| 特征集 | 列数 | valid macroF1 | test acc | test macroF1 | F1_3# | F1_9996 |
| --- | --- | --- | --- | --- | --- | --- |
| v2b_base43 | 43 | 0.4739 | 0.6398 | 0.4500 | 0.250 | 0.697 |
| v3_spatial_only | 43 | 0.4739 | 0.6398 | 0.4500 | 0.250 | 0.697 |
| v3_texture_only | 63 | 0.6478 | 0.7516 | 0.6605 | 0.615 | 0.752 |
| v3_full | 63 | 0.6478 | 0.7516 | 0.6605 | 0.615 | 0.752 |

- 最优特征集：**v3_texture_only**；相对 v2 基线重训 Δacc=+0.1118、ΔmacroF1=+0.2105。
- 全模型对照见 `model_comparison.csv`（majority/decision_tree/lightgbm × valid/test）。
- 最优特征集 LightGBM 特征重要性 top20：

- `tx__lbp_bin00`: 4263.0211
- `tx__glcm_correlation`: 2411.0679
- `tx__gray_mean`: 1206.7674
- `tx__glcm_energy`: 842.9180
- `tx__lbp_bin06`: 747.2019
- `tx__gray_var`: 579.3264
- `A_ratio`: 570.7520
- `total_bbox_union_image_ratio`: 569.7536
- `area_ratio__waterstain`: 568.5156
- `count__waterstain`: 525.6335
- `tx__lbp_bin03`: 522.7415
- `C_ratio`: 511.7798
- `tx__lbp_bin07`: 411.2190
- `B_ratio`: 368.5677
- `tx__lbp_bin01`: 322.0185
- `tx__lbp_bin08`: 302.9442
- `tx__grad_p90`: 302.2267
- `tx__grad_mean`: 287.9994
- `tx__grad_var`: 268.5602
- `tx__lbp_bin02`: 266.5548

## 2. 方案 5：拒判分析（最优配置 = v3_texture_only + LightGBM）

阈值仅由 valid 置信度分位确定；test 一次性报告。拒判样本业务上记 `manual_review`。

| 目标覆盖率 | 阈值(valid) | test 实际覆盖 | test acc | test macroF1 |
| --- | --- | --- | --- | --- |
| 1.00 | 0.0000 | 1.000 | 0.7516 | 0.6605 |
| 0.90 | 0.5623 | 0.857 | 0.7681 | 0.6826 |
| 0.80 | 0.6298 | 0.752 | 0.8182 | 0.7250 |
| 0.70 | 0.7115 | 0.677 | 0.8257 | 0.7824 |
| 0.60 | 0.7622 | 0.596 | 0.8438 | 0.7691 |

- acc≥0.85 的覆盖点：未达到。

## 3. 结论与局限

- 空间/纹理特征对 test 的增量：Δacc=+0.1118、ΔmacroF1=+0.2105（<0.01 视为无实质提升）。
- 3#/bad 小样本约束不变；9950↔9996 混对是否缓解见 `metrics_v3_full.json` 混淆矩阵。
- 拒判可作为部署保底策略，但阈值与覆盖率均为未确认候选，需企业确认业务可接受精度。

## 4. 复现

```text
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe extract_dianji_grade_v3_features.py
.\.venv_classify\Scripts\python.exe train_dianji_grade_v3.py
```
