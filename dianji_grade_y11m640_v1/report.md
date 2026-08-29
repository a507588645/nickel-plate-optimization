# 电积镍等级 YOLO11m@640：检测特征 + 纹理特征 + 拒判分析报告

> ## 未经企业确认 / UNCONFIRMED
> 全部结果 `unconfirmed_pending_enterprise`，企业书面标准确认前不得上线。
> 小样本警告：3#=46、bad=28（test 各 7/4 张），少数类指标不稳。

- 生成时间：`2026-08-28T14:28:01`
- 特征来源：`features_y11m640_v1.csv`（YOLO11m@640 新提取 43 列 BASE + 复用 20 列 `tx__`）
- BASE schema：与历史 `features_v3_full.csv` 的对应 43 个列名集合完全一致（43/43）；历史 CSV 的物理顺序为 count/area 交错，本产物按 `train_dianji_grade_v3.py` 的 `BASE_COLUMNS`（先 count 后 area）生成，部署契约固定该训练顺序。
- 检测权重：冻结 YOLO11m ONNX，SHA-256=`901ccf6006f44689483aaf6f0d1d1b8c18132a85c7d881be7e62fcdccc255b21`
- 检测参数：CPU，`imgsz=640`，`conf=0.25`，`iou=0.45`；图像全部从 `archives/dianjinie_grade_classification_v1` 的 `link_relpath` 解析
- 协议：train 训练、valid 选模/早停/定拒判阈值、test 仅最终一次；seed=42
- v2 基线重训行（v2b_base43）test LGBM：acc=0.6149 / macroF1=0.5192
  （03:00 条目参考值 0.640/0.413，漂移 -0.0251，±0.01 内视为自洽）
- 历史参考（仅历史，不是本次目标）：历史 `ryw.pt@1536` 的 `v3_texture_only` test acc=0.7764 / macroF1=0.6631；其 valid-only 目标覆盖率 0.70 阈值=0.7397、test coverage=0.714、accepted acc=0.8696。本次换检测器后指标不应与其直接当作同一模型结果。

## 1. 特征增量对照（LightGBM, test）

| 特征集 | 列数 | valid macroF1 | test acc | test macroF1 | F1_3# | F1_9996 |
| --- | --- | --- | --- | --- | --- | --- |
| v2b_base43 | 43 | 0.4648 | 0.6149 | 0.5192 | 0.600 | 0.601 |
| v3_spatial_only | 43 | 0.4648 | 0.6149 | 0.5192 | 0.600 | 0.601 |
| v3_texture_only | 63 | 0.7058 | 0.7453 | 0.6797 | 0.545 | 0.741 |
| v3_full | 63 | 0.7058 | 0.7453 | 0.6797 | 0.545 | 0.741 |

- 最优特征集：**v3_texture_only**；相对本次 v2 基线重训 Δacc=+0.1304、ΔmacroF1=+0.1605。
- 全模型对照见 `model_comparison.csv`（majority/decision_tree/lightgbm × valid/test）。
- 本次输入没有 `sp__` 列，因此 `v3_spatial_only` 与 `v2b_base43` 相同，`v3_full` 与 `v3_texture_only` 相同；目标部署模型仍固定为 63 列 `v3_texture_only` 等价配置。
- 目标配置详细指标：valid acc=0.7453 / macroF1=0.7058；test acc=0.7453 / macroF1=0.6797。逐等级 Precision/Recall/F1 见 `metrics_v3_texture_only.json`。

### 目标配置逐等级指标

| split | grade | Precision | Recall | F1 | support |
| --- | --- | ---: | ---: | ---: | ---: |
| valid | 3# | 1.0000 | 0.5714 | 0.7273 | 7 |
| valid | 9950 | 0.7805 | 0.7805 | 0.7805 | 82 |
| valid | 9996 | 0.7101 | 0.7206 | 0.7153 | 68 |
| valid | bad | 0.5000 | 0.7500 | 0.6000 | 4 |
| test | 3# | 0.7500 | 0.4286 | 0.5455 | 7 |
| test | 9950 | 0.7750 | 0.7561 | 0.7654 | 82 |
| test | 9996 | 0.7067 | 0.7794 | 0.7413 | 68 |
| test | bad | 1.0000 | 0.5000 | 0.6667 | 4 |

## 2. 拒判分析（v3_texture_only + LightGBM）

阈值仅由 valid 置信度分位确定；test 一次性报告。拒判样本业务上记 `manual_review`。

| 目标覆盖率 | 阈值(valid) | test 实际覆盖 | test acc | test macroF1 |
| --- | --- | --- | --- | --- |
| 1.00 | 0.0000 | 1.000 | 0.7453 | 0.6797 |
| 0.90 | 0.5783 | 0.919 | 0.7838 | 0.7475 |
| 0.80 | 0.6625 | 0.807 | 0.8154 | 0.7637 |
| 0.70 | 0.7362 | 0.739 | 0.8487 | 0.7820 |
| 0.60 | 0.7871 | 0.652 | 0.8667 | 0.7708 |

- 部署契约选定的候选点是目标覆盖率 0.70：阈值仅由 valid 得到 `0.7362`，valid coverage=`0.7019`；test 一次性报告 coverage=`0.7391`、acc=`0.8487`、macroF1=`0.7820`。该阈值仍未获企业确认。

## 3. 结论与局限

- 相对本次同一 YOLO11m@640 BASE 43 重训，加入 20 列 `tx__` 后 test 增量为 Δacc=+0.1304、ΔmacroF1=+0.1605；这不是与历史 `ryw.pt@1536` 结果的直接排名。
- 3#/bad 小样本约束不变；9950↔9996 混对见 `metrics_v3_full.json` 混淆矩阵。
- 拒判可作为部署保底策略，但阈值与覆盖率均为未确认候选，需企业确认业务可接受精度。
- `lgbm_model.txt` 已序列化，`deploy_contract.json` 固定 63 列顺序、等级 id 映射和提取参数；fresh-process 3 图 round-trip 通过，证据见 `roundtrip_summary.json`。
- 本次全量提取为 1072/1072 图像、9672 个框；面积回退 0 张，特征 join 1072/1072。

## 4. 复现

```text
cd /d E:/PythonProject/优化
E:/PythonProject/ImetricCodes/JinchuanPythonWebapi/.venv/Scripts/python.exe extract_dianji_grade_y11m640_features.py
E:/PythonProject/优化/.venv_classify/Scripts/python.exe train_dianji_grade_v3.py --features dianji_grade_y11m640_v1/features_y11m640_v1.csv --output-dir dianji_grade_y11m640_v1
```
