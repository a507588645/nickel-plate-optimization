# 电积镍等级 v3：空间/纹理特征 + 拒判分析报告

> ## ⚠️ 未经企业确认 / UNCONFIRMED
> 全部结果 `unconfirmed_pending_enterprise`，企业书面标准确认前不得上线。
> 小样本警告：3#=46、bad=28（test 各 7/4 张），少数类指标不稳。

- 生成时间：`2026-09-05T23:06:19`
- 特征来源：`dianji_grade_v3\features_v3_full.csv`（v2b conf0.25 43 列 + 空间 + 纹理）
- 协议：train 训练、valid 选模/早停/定拒判阈值、test 仅最终一次；seed=42
- v2 基线重训行（slf_base13）test LGBM：acc=0.6460 / macroF1=0.5708
  （03:00 条目参考值 0.640/0.413，漂移 +0.0060，±0.01 内视为自洽）

## 1. 方案 4：特征增量对照（LightGBM, test）

| 特征集 | 列数 | valid macroF1 | test acc | test macroF1 | F1_3# | F1_9996 |
| --- | --- | --- | --- | --- | --- | --- |
| slf_base13 | 13 | 0.6032 | 0.6460 | 0.5708 | 0.455 | 0.671 |
| slf_texture33 | 33 | 0.7422 | 0.7578 | 0.7225 | 0.800 | 0.748 |

- 最优特征集：**slf_texture33**；相对 v2 基线重训 Δacc=+0.1118、ΔmacroF1=+0.1516。
- 全模型对照见 `model_comparison.csv`（majority/decision_tree/lightgbm × valid/test）。
- 最优特征集 LightGBM 特征重要性 top20：

- `tx__lbp_bin00`: 4346.8953
- `total_bbox_union_image_ratio`: 2426.4888
- `tx__glcm_correlation`: 1287.9437
- `tx__gray_mean`: 1168.3249
- `area_ratio__waterblock`: 1109.2422
- `area_ratio__bubble`: 997.0179
- `tx__glcm_energy`: 748.7675
- `area_ratio__waterstain`: 656.0960
- `tx__gray_var`: 535.5479
- `area_ratio__colorstain`: 442.3907
- `tx__lbp_bin04`: 410.0677
- `tx__lbp_bin07`: 357.5537
- `tx__lbp_bin08`: 343.5588
- `tx__lbp_bin01`: 335.8365
- `tx__lbp_bin06`: 328.3373
- `count__waterblock`: 323.8164
- `tx__lbp_bin02`: 306.1208
- `count__waterstain`: 283.0430
- `count__bubble`: 280.9471
- `tx__lbp_bin03`: 266.9490

## 2. 方案 5：拒判分析（最优配置 = slf_texture33 + LightGBM）

阈值仅由 valid 置信度分位确定；test 一次性报告。拒判样本业务上记 `manual_review`。

| 目标覆盖率 | 阈值(valid) | test 实际覆盖 | test acc | test macroF1 |
| --- | --- | --- | --- | --- |
| 1.00 | 0.0000 | 1.000 | 0.7578 | 0.7225 |
| 0.90 | 0.6414 | 0.826 | 0.8195 | 0.7859 |
| 0.80 | 0.7058 | 0.745 | 0.8417 | 0.8150 |
| 0.70 | 0.7819 | 0.671 | 0.8611 | 0.8256 |
| 0.60 | 0.8351 | 0.602 | 0.8660 | 0.8238 |

- acc≥0.85 的覆盖点：达到（覆盖率≈67%）。

## 3. 结论与局限

- 空间/纹理特征对 test 的增量：Δacc=+0.1118、ΔmacroF1=+0.1516（<0.01 视为无实质提升）。
- 3#/bad 小样本约束不变；9950↔9996 混对是否缓解见 `metrics_v3_full.json` 混淆矩阵。
- 拒判可作为部署保底策略，但阈值与覆盖率均为未确认候选，需企业确认业务可接受精度。

## 4. 复现

```text
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe extract_dianji_grade_v3_features.py
.\.venv_classify\Scripts\python.exe train_dianji_grade_v3.py
```
