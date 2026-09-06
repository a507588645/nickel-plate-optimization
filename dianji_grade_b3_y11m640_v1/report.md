# 电积镍等级 v3：空间/纹理特征 + 拒判分析报告

> ## ⚠️ 未经企业确认 / UNCONFIRMED
> 全部结果 `unconfirmed_pending_enterprise`，企业书面标准确认前不得上线。
> 小样本警告：3#=46、bad=28（test 各 7/4 张），少数类指标不稳。

- 生成时间：`2026-09-05T13:48:26`
- 特征来源：`dianji_grade_v3\features_v3_full.csv`（v2b conf0.25 43 列 + 空间 + 纹理）
- 协议：train 训练、valid 选模/早停/定拒判阈值、test 仅最终一次；seed=42
- v2 基线重训行（slf_base13）test LGBM：acc=0.6708 / macroF1=0.4744
  （03:00 条目参考值 0.640/0.413，漂移 +0.0308，±0.01 内视为自洽）

## 1. 方案 4：特征增量对照（LightGBM, test）

| 特征集 | 列数 | valid macroF1 | test acc | test macroF1 | F1_3# | F1_9996 |
| --- | --- | --- | --- | --- | --- | --- |
| slf_base13 | 13 | 0.6236 | 0.6708 | 0.4744 | 0.267 | 0.724 |
| slf_texture33 | 33 | 0.7925 | 0.8385 | 0.8180 | 0.933 | 0.824 |

- 最优特征集：**slf_texture33**；相对 v2 基线重训 Δacc=+0.1677、ΔmacroF1=+0.3436。
- 全模型对照见 `model_comparison.csv`（majority/decision_tree/lightgbm × valid/test）。
- 最优特征集 LightGBM 特征重要性 top20：

- `tx__lbp_bin00`: 4288.6166
- `tx__glcm_correlation`: 2012.5921
- `area_ratio__colorstain`: 1414.4696
- `area_ratio__bubble`: 1268.4840
- `tx__gray_mean`: 1219.6225
- `total_bbox_union_image_ratio`: 799.9184
- `area_ratio__waterstain`: 794.6070
- `tx__glcm_energy`: 736.9273
- `area_ratio__waterblock`: 566.4423
- `tx__lbp_bin09`: 551.4861
- `tx__gray_var`: 519.2511
- `tx__lbp_bin07`: 500.9286
- `tx__lbp_bin01`: 425.3308
- `count__waterblock`: 403.4649
- `tx__lbp_bin06`: 385.4579
- `tx__lbp_bin08`: 333.3864
- `tx__lbp_bin04`: 307.2514
- `tx__grad_mean`: 284.8520
- `detection_count`: 282.2294
- `tx__lbp_bin03`: 277.6900

## 2. 方案 5：拒判分析（最优配置 = slf_texture33 + LightGBM）

阈值仅由 valid 置信度分位确定；test 一次性报告。拒判样本业务上记 `manual_review`。

| 目标覆盖率 | 阈值(valid) | test 实际覆盖 | test acc | test macroF1 |
| --- | --- | --- | --- | --- |
| 1.00 | 0.0000 | 1.000 | 0.8385 | 0.8180 |
| 0.90 | 0.5794 | 0.882 | 0.8521 | 0.8233 |
| 0.80 | 0.7077 | 0.783 | 0.8651 | 0.8275 |
| 0.70 | 0.8134 | 0.596 | 0.9062 | 0.8501 |
| 0.60 | 0.8632 | 0.509 | 0.9268 | 0.8863 |

- acc≥0.85 的覆盖点：达到（覆盖率≈88%）。

## 3. 结论与局限

- 空间/纹理特征对 test 的增量：Δacc=+0.1677、ΔmacroF1=+0.3436（<0.01 视为无实质提升）。
- 3#/bad 小样本约束不变；9950↔9996 混对是否缓解见 `metrics_v3_full.json` 混淆矩阵。
- 拒判可作为部署保底策略，但阈值与覆盖率均为未确认候选，需企业确认业务可接受精度。

## 4. 复现

```text
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe extract_dianji_grade_v3_features.py
.\.venv_classify\Scripts\python.exe train_dianji_grade_v3.py
```
