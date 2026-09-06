# 电积镍等级 v3：空间/纹理特征 + 拒判分析报告

> ## ⚠️ 未经企业确认 / UNCONFIRMED
> 全部结果 `unconfirmed_pending_enterprise`，企业书面标准确认前不得上线。
> 小样本警告：3#=46、bad=28（test 各 7/4 张），少数类指标不稳。

- 生成时间：`2026-09-05T01:54:18`
- 特征来源：`dianji_grade_v3\features_v3_full.csv`（v2b conf0.25 43 列 + 空间 + 纹理）
- 协议：train 训练、valid 选模/早停/定拒判阈值、test 仅最终一次；seed=42
- v2 基线重训行（v2b_base43）test LGBM：acc=0.6025 / macroF1=0.4799
  （03:00 条目参考值 0.640/0.413，漂移 -0.0375，±0.01 内视为自洽）

## 1. 方案 4：特征增量对照（LightGBM, test）

| 特征集 | 列数 | valid macroF1 | test acc | test macroF1 | F1_3# | F1_9996 |
| --- | --- | --- | --- | --- | --- | --- |
| v2b_base43 | 43 | 0.5026 | 0.6025 | 0.4799 | 0.421 | 0.633 |
| v3_spatial_only | 43 | 0.5026 | 0.6025 | 0.4799 | 0.421 | 0.633 |
| v3_texture_only | 63 | 0.6905 | 0.7453 | 0.6566 | 0.545 | 0.734 |
| v3_full | 63 | 0.6905 | 0.7453 | 0.6566 | 0.545 | 0.734 |

- 最优特征集：**v3_texture_only**；相对 v2 基线重训 Δacc=+0.1429、ΔmacroF1=+0.1767。
- 全模型对照见 `model_comparison.csv`（majority/decision_tree/lightgbm × valid/test）。
- 最优特征集 LightGBM 特征重要性 top20：

- `tx__lbp_bin00`: 3962.3731
- `tx__glcm_correlation`: 2921.3021
- `tx__gray_mean`: 1195.7817
- `tx__glcm_energy`: 1063.7882
- `B_ratio`: 886.0967
- `tx__lbp_bin06`: 824.7070
- `tx__gray_var`: 671.0887
- `tx__lbp_bin07`: 587.0123
- `area_ratio__waterstain`: 575.1500
- `area_ratio__wuzi`: 558.6889
- `total_bbox_union_image_ratio`: 487.8767
- `tx__lbp_bin03`: 468.8429
- `area_ratio__greenstain`: 398.4279
- `tx__lbp_bin01`: 377.2967
- `tx__lbp_bin09`: 337.4989
- `tx__lbp_bin05`: 329.8508
- `tx__grad_var`: 316.2087
- `count__waterstain`: 292.7335
- `A_ratio`: 271.4725
- `tx__grad_mean`: 249.3617

## 2. 方案 5：拒判分析（最优配置 = v3_texture_only + LightGBM）

阈值仅由 valid 置信度分位确定；test 一次性报告。拒判样本业务上记 `manual_review`。

| 目标覆盖率 | 阈值(valid) | test 实际覆盖 | test acc | test macroF1 |
| --- | --- | --- | --- | --- |
| 1.00 | 0.0000 | 1.000 | 0.7453 | 0.6566 |
| 0.90 | 0.5699 | 0.944 | 0.7632 | 0.7011 |
| 0.80 | 0.6768 | 0.839 | 0.8000 | 0.7200 |
| 0.70 | 0.7497 | 0.689 | 0.8468 | 0.7778 |
| 0.60 | 0.7883 | 0.634 | 0.8627 | 0.7859 |

- acc≥0.85 的覆盖点：达到（覆盖率≈63%）。

## 3. 结论与局限

- 空间/纹理特征对 test 的增量：Δacc=+0.1429、ΔmacroF1=+0.1767（<0.01 视为无实质提升）。
- 3#/bad 小样本约束不变；9950↔9996 混对是否缓解见 `metrics_v3_full.json` 混淆矩阵。
- 拒判可作为部署保底策略，但阈值与覆盖率均为未确认候选，需企业确认业务可接受精度。

## 4. 复现

```text
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe extract_dianji_grade_v3_features.py
.\.venv_classify\Scripts\python.exe train_dianji_grade_v3.py
```
