# 电积镍等级 v3：空间/纹理特征 + 拒判分析报告

> ## ⚠️ 未经企业确认 / UNCONFIRMED
> 全部结果 `unconfirmed_pending_enterprise`，企业书面标准确认前不得上线。
> 小样本警告：3#=46、bad=28（test 各 7/4 张），少数类指标不稳。

- 生成时间：`2026-08-28T15:34:44`
- 特征来源：`dianji_grade_v3\features_v3_full.csv`（v2b conf0.25 43 列 + 空间 + 纹理）
- 协议：train 训练、valid 选模/早停/定拒判阈值、test 仅最终一次；seed=42
- v2 基线重训行（slf_base13）test LGBM：acc=0.6460 / macroF1=0.4278
  （03:00 条目参考值 0.640/0.413，漂移 +0.0060，±0.01 内视为自洽）

## 1. 方案 4：特征增量对照（LightGBM, test）

| 特征集 | 列数 | valid macroF1 | test acc | test macroF1 | F1_3# | F1_9996 |
| --- | --- | --- | --- | --- | --- | --- |
| slf_base13 | 13 | 0.5143 | 0.6460 | 0.4278 | 0.333 | 0.681 |
| slf_texture33 | 33 | 0.7074 | 0.7702 | 0.6528 | 0.545 | 0.754 |

- 最优特征集：**slf_texture33**；相对 v2 基线重训 Δacc=+0.1242、ΔmacroF1=+0.2250。
- 全模型对照见 `model_comparison.csv`（majority/decision_tree/lightgbm × valid/test）。
- 最优特征集 LightGBM 特征重要性 top20：

- `tx__lbp_bin00`: 4044.6246
- `tx__glcm_correlation`: 2496.9291
- `tx__gray_mean`: 1343.7160
- `tx__lbp_bin06`: 1180.3775
- `tx__glcm_energy`: 1119.7464
- `area_ratio__waterstain`: 1068.5710
- `total_bbox_union_image_ratio`: 947.2302
- `tx__gray_var`: 672.6377
- `tx__lbp_bin07`: 634.5892
- `count__bubble`: 519.6655
- `tx__lbp_bin03`: 519.2363
- `tx__lbp_bin01`: 473.3888
- `tx__lbp_bin05`: 460.3923
- `tx__lbp_bin02`: 451.8462
- `area_ratio__waterblock`: 427.6753
- `tx__grad_var`: 426.9681
- `tx__lbp_bin08`: 393.9161
- `tx__grad_mean`: 354.3209
- `area_ratio__bubble`: 353.9147
- `count__waterstain`: 317.4154

## 2. 方案 5：拒判分析（最优配置 = slf_texture33 + LightGBM）

阈值仅由 valid 置信度分位确定；test 一次性报告。拒判样本业务上记 `manual_review`。

| 目标覆盖率 | 阈值(valid) | test 实际覆盖 | test acc | test macroF1 |
| --- | --- | --- | --- | --- |
| 1.00 | 0.0000 | 1.000 | 0.7702 | 0.6528 |
| 0.90 | 0.5866 | 0.882 | 0.8028 | 0.6986 |
| 0.80 | 0.7117 | 0.776 | 0.8160 | 0.6839 |
| 0.70 | 0.7872 | 0.658 | 0.8491 | 0.7183 |
| 0.60 | 0.8485 | 0.565 | 0.8901 | 0.7427 |

- acc≥0.85 的覆盖点：达到（覆盖率≈57%）。

## 3. 结论与局限

- 空间/纹理特征对 test 的增量：Δacc=+0.1242、ΔmacroF1=+0.2250（<0.01 视为无实质提升）。
- 3#/bad 小样本约束不变；9950↔9996 混对是否缓解见 `metrics_v3_full.json` 混淆矩阵。
- 拒判可作为部署保底策略，但阈值与覆盖率均为未确认候选，需企业确认业务可接受精度。

## 4. 复现

```text
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe extract_dianji_grade_v3_features.py
.\.venv_classify\Scripts\python.exe train_dianji_grade_v3.py
```
