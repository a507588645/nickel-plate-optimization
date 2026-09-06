# 电积镍等级 v3：空间/纹理特征 + 拒判分析报告

> ## ⚠️ 未经企业确认 / UNCONFIRMED
> 全部结果 `unconfirmed_pending_enterprise`，企业书面标准确认前不得上线。
> 小样本警告：3#=46、bad=28（test 各 7/4 张），少数类指标不稳。

- 生成时间：`2026-09-05T00:18:41`
- 特征来源：`dianji_grade_v3\features_v3_full.csv`（v2b conf0.25 43 列 + 空间 + 纹理）
- 协议：train 训练、valid 选模/早停/定拒判阈值、test 仅最终一次；seed=42
- v2 基线重训行（v2b_base43）test LGBM：acc=0.6087 / macroF1=0.5017
  （03:00 条目参考值 0.640/0.413，漂移 -0.0313，±0.01 内视为自洽）

## 1. 方案 4：特征增量对照（LightGBM, test）

| 特征集 | 列数 | valid macroF1 | test acc | test macroF1 | F1_3# | F1_9996 |
| --- | --- | --- | --- | --- | --- | --- |
| v2b_base43 | 43 | 0.4871 | 0.6087 | 0.5017 | 0.556 | 0.615 |
| v3_spatial_only | 43 | 0.4871 | 0.6087 | 0.5017 | 0.556 | 0.615 |
| v3_texture_only | 63 | 0.7616 | 0.7578 | 0.6862 | 0.545 | 0.750 |
| v3_full | 63 | 0.7616 | 0.7578 | 0.6862 | 0.545 | 0.750 |

- 最优特征集：**v3_texture_only**；相对 v2 基线重训 Δacc=+0.1491、ΔmacroF1=+0.1845。
- 全模型对照见 `model_comparison.csv`（majority/decision_tree/lightgbm × valid/test）。
- 最优特征集 LightGBM 特征重要性 top20：

- `tx__lbp_bin00`: 4014.4097
- `tx__glcm_correlation`: 2610.8341
- `tx__glcm_energy`: 1034.2837
- `tx__gray_mean`: 967.0330
- `tx__gray_var`: 883.8843
- `total_bbox_union_image_ratio`: 814.8671
- `B_ratio`: 750.4166
- `tx__lbp_bin06`: 695.8222
- `count__waterstain`: 592.6976
- `tx__lbp_bin07`: 511.3901
- `area_ratio__wuzi`: 505.7724
- `tx__lbp_bin03`: 451.2819
- `area_ratio__waterstain`: 420.6423
- `tx__lbp_bin01`: 344.0922
- `area_ratio__greenstain`: 337.0116
- `tx__lbp_bin09`: 311.5807
- `tx__lbp_bin05`: 305.8613
- `tx__lbp_bin08`: 275.3234
- `tx__grad_mean`: 250.5447
- `area_ratio__airhole`: 231.4175

## 2. 方案 5：拒判分析（最优配置 = v3_texture_only + LightGBM）

阈值仅由 valid 置信度分位确定；test 一次性报告。拒判样本业务上记 `manual_review`。

| 目标覆盖率 | 阈值(valid) | test 实际覆盖 | test acc | test macroF1 |
| --- | --- | --- | --- | --- |
| 1.00 | 0.0000 | 1.000 | 0.7578 | 0.6862 |
| 0.90 | 0.5489 | 0.957 | 0.7662 | 0.7029 |
| 0.80 | 0.6313 | 0.851 | 0.8029 | 0.7687 |
| 0.70 | 0.7133 | 0.689 | 0.8468 | 0.7917 |
| 0.60 | 0.7614 | 0.596 | 0.8854 | 0.7719 |

- acc≥0.85 的覆盖点：达到（覆盖率≈60%）。

## 3. 结论与局限

- 空间/纹理特征对 test 的增量：Δacc=+0.1491、ΔmacroF1=+0.1845（<0.01 视为无实质提升）。
- 3#/bad 小样本约束不变；9950↔9996 混对是否缓解见 `metrics_v3_full.json` 混淆矩阵。
- 拒判可作为部署保底策略，但阈值与覆盖率均为未确认候选，需企业确认业务可接受精度。

## 4. 复现

```text
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe extract_dianji_grade_v3_features.py
.\.venv_classify\Scripts\python.exe train_dianji_grade_v3.py
```
