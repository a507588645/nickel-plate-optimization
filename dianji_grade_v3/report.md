# 电积镍等级 v3：空间/纹理特征 + 拒判分析报告

> ## ⚠️ 未经企业确认 / UNCONFIRMED
> 全部结果 `unconfirmed_pending_enterprise`，企业书面标准确认前不得上线。
> 小样本警告：3#=46、bad=28（test 各 7/4 张），少数类指标不稳。

- 生成时间：`2026-07-23T04:42:17`
- 特征来源：`dianji_grade_v3\features_v3_full.csv`（v2b conf0.25 43 列 + 空间 + 纹理）
- 协议：train 训练、valid 选模/早停/定拒判阈值、test 仅最终一次；seed=42
- v2 基线重训行（v2b_base43）test LGBM：acc=0.6398 / macroF1=0.4131
  （03:00 条目参考值 0.640/0.413，漂移 -0.0002，±0.01 内视为自洽）

## 1. 方案 4：特征增量对照（LightGBM, test）

| 特征集 | 列数 | valid macroF1 | test acc | test macroF1 | F1_3# | F1_9996 |
| --- | --- | --- | --- | --- | --- | --- |
| v2b_base43 | 43 | 0.3562 | 0.6398 | 0.4131 | 0.133 | 0.667 |
| v3_spatial_only | 73 | 0.4855 | 0.6894 | 0.4508 | 0.167 | 0.700 |
| v3_texture_only | 63 | 0.6086 | 0.7764 | 0.6631 | 0.500 | 0.774 |
| v3_full | 93 | 0.5662 | 0.7764 | 0.5787 | 0.222 | 0.767 |

- 最优特征集：**v3_texture_only**；相对 v2 基线重训 Δacc=+0.1366、ΔmacroF1=+0.2500。
- 全模型对照见 `model_comparison.csv`（majority/decision_tree/lightgbm × valid/test）。
- 最优特征集 LightGBM 特征重要性 top20：

- `tx__lbp_bin00`: 3048.6224
- `tx__glcm_correlation`: 2646.7594
- `tx__lbp_bin06`: 1440.8829
- `tx__gray_mean`: 1323.5818
- `tx__gray_var`: 976.6024
- `tx__glcm_energy`: 949.2187
- `tx__lbp_bin03`: 812.5977
- `tx__lbp_bin07`: 729.9727
- `area_ratio__bubble`: 577.3957
- `B_ratio`: 534.5625
- `tx__lbp_bin05`: 531.9986
- `tx__grad_var`: 531.3830
- `tx__lbp_bin02`: 381.6542
- `area_ratio__watermark`: 352.4105
- `tx__grad_mean`: 347.2271
- `area_ratio__bump`: 346.9410
- `tx__lbp_bin08`: 331.9581
- `tx__lbp_bin01`: 309.7325
- `total_bbox_union_image_ratio`: 277.7437
- `count__wuzi`: 248.8900

## 2. 方案 5：拒判分析（最优配置 = v3_texture_only + LightGBM）

阈值仅由 valid 置信度分位确定；test 一次性报告。拒判样本业务上记 `manual_review`。

| 目标覆盖率 | 阈值(valid) | test 实际覆盖 | test acc | test macroF1 |
| --- | --- | --- | --- | --- |
| 1.00 | 0.0000 | 1.000 | 0.7764 | 0.6631 |
| 0.90 | 0.5613 | 0.888 | 0.8042 | 0.6254 |
| 0.80 | 0.6593 | 0.801 | 0.8217 | 0.6892 |
| 0.70 | 0.7397 | 0.714 | 0.8696 | 0.7254 |
| 0.60 | 0.7991 | 0.665 | 0.8972 | 0.7408 |

- acc≥0.85 的覆盖点：达到（覆盖率≈71%）。

## 3. 结论与局限

- 空间/纹理特征对 test 的增量：Δacc=+0.1366、ΔmacroF1=+0.2500（<0.01 视为无实质提升）。
- 3#/bad 小样本约束不变；9950↔9996 混对是否缓解见 `metrics_v3_full.json` 混淆矩阵。
- 拒判可作为部署保底策略，但阈值与覆盖率均为未确认候选，需企业确认业务可接受精度。

## 3.1 稳健性检查：纹理增益不是「两个来源数据集」混淆（Kimi 追加，2026-07-23）

- 动机：9996/bad 只存在于 4types，0309 只有 9950/3#；若两来源成像条件不同，纹理可能是在学「哪批照片」而非等级（工作日志 2026-07-14 早有警告：四分类数据可能拟合「拍摄背景、批次纹理等伪相关特征」）。
- 方法：仅在 `classify_4types_dianji` 单源 986 张上重跑同协议（`robustness_4types_only\`）。
- 结果：基线 0.6233/0.4181 → texture_only **0.7055/0.6074（Δ+0.082/+0.189）**，增益在单源内保持 → 「两个来源」混淆不成立。
- **仍未排除**：4types 内部按等级分批拍摄的可能性（等级↔拍摄批次↔光照/背景相关）。本数据无批次元数据，无法用现有信息证伪；**定论需要用新拍摄会话的图像做外部队时验证**（建议列为企业考察项）。
- 物理解释的可信面：3# 含烧板/缺角、bad 为整板严重缺陷、9996 vs 9950 表面光洁度差异——这些本身就是亮度/纹理现象，纹理特征有正当物理意义；重要性 top 特征（lbp_bin00、glcm_correlation、gray_mean）与之一致。

## 4. 复现

```text
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe extract_dianji_grade_v3_features.py
.\.venv_classify\Scripts\python.exe train_dianji_grade_v3.py
# 稳健性（单源）：
.\.venv_classify\Scripts\python.exe train_dianji_grade_v3.py --features dianji_grade_v3\features_v3_full_4types_only.csv --output-dir dianji_grade_v3\robustness_4types_only
```
