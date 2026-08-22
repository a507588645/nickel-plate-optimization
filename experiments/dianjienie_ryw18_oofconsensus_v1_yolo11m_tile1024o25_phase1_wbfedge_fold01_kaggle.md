# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo11m_tile1024o25_phase1_wbfedge_fold01_kaggle` |
| 所属主线 | L1 小目标研究 / Phase 1 后处理修复验证 |
| 日期 | 2026-08-22 |
| 执行人 | Codex |
| 实验状态 | `complete` |
| 证据等级 | `valid`，仅用于当前单 Fold `auto_only` 切片方向的停止闸门；不构成最终模型或人工真值结论 |
| 父计划 / 决策闸道 | `L1_小目标优化分阶段计划.md` Phase 1 |
| 研究假设 | G4b 的切片大幅回退主要来自跨切片重复、低置信噪声和边缘截断；修复后处理可恢复至固定整图基线。 |
| 只改变的主要变量 | 同一回收候选检查点的切片推理后处理：`legacy`、`perclass_nms`、`wbf`、`wbf_edge`。不训练、不续训。 |
| 停止条件 | `tile_model_tiled - baseline_full < -0.01`，或重点小目标类 AP/Recall 任一下降。 |
| 运行环境 | Kaggle 私有评测 Kernel `lly3323w/l1-ryw18-y11m-oof-v1-tile-eval-wbf-f01` version 1 |
| GPU / Python / Torch / Ultralytics | Tesla P100 16 GB / 3.12.13 / 2.6.0+cu124 / 8.4.66 |

## 2. 数据与边界

| 项目 | 内容 |
| --- | --- |
| 数据集 | `dianjienie` RYW18 OOF 自动共识候选开发集 |
| 数据版本 | `oofconsensusv1`，`auto_only` 五列标签 |
| Fold | 固定 Fold 1：train 751 图、val 376 图 |
| 切片推理 | 原图 1024 px、overlap 25%，val 共 9,400 个推理切片 |
| 固定整图基线 | Fold 1 640 YOLO11m，SHA-256=`b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce` |
| 固定切片候选 | G4b 回收 `best.pt`，SHA-256=`c4a903e8e8438798d44fc0548d666923a8728509f012d124101a8b980d7d875e` |
| 冻结测试集 | `development_only=true`、`frozen_test_set=null`；未挂载、未读取、未评估历史 198 张 `release-validation` |
| 阈值边界 | Fold 1 的逐类阈值仅由 Fold 2/3 原始预测校准；默认 0.25，阈值文件位于 `l1_phase0_diagnostics_20260822_1455/fold01_holdout_thresholds/` |

本次只读取 Kaggle 完成归档。ZIP SHA-256=`252bcb40bc39b6de1b5cf3484ce1e2109e38b13d532fda7fc17d6a4337a4d4af`，已通过 CRC、路径安全、配置、权重、4 个 profile、两份 376 图 gzip 逐框明细及聚合计数交叉校验。

## 3. 评测配置

| 参数 | 值 |
| --- | --- |
| model / 训练 | 无训练；仅评测已固定的基线与回收检查点 |
| 原图评测器 | `l1_tile_pipeline.py evaluate`；预测回映射至原图后按类别评测 |
| AP 口径 | IoU 0.50:0.95，101 点精度包络 |
| P/R 口径 | confidence 0.25、IoU 0.50 |
| `legacy` | 原有类别感知全局 NMS，IoU 0.70 |
| `perclass_nms` | Fold 2/3 留出校准的逐类 confidence + 原有 NMS |
| `wbf` | 逐类 confidence + WBF，IoU 0.25 |
| `wbf_edge` | `wbf` 加 16 px 内部切片边缘过滤；指定主 profile |

## 4. 结果

固定整图基线 `baseline_full` 的 mAP50-95 为 `0.168001`。四种切片后处理均未恢复到该水平。

| Profile | baseline tiled mAP50-95 | tile model tiled mAP50-95 | 相对固定整图基线 |
| --- | ---: | ---: | ---: |
| `legacy` | 0.023219 | 0.063427 | -0.104574 |
| `perclass_nms` | 0.021883 | 0.050408 | -0.117593 |
| `wbf` | 0.019617 | 0.050063 | -0.117939 |
| `wbf_edge` | 0.022955 | 0.052221 | -0.115781 |

`legacy` 数值复现了先前 G4b 独立评测，说明新增 profile 未改变旧路径。指定主 profile `wbf_edge` 的 P/R/mAP50/mAP50-95 为 `0.490958 / 0.336918 / 0.105026 / 0.052221`；候选权重整图推理为 `0.156492` mAP50-95，也低于固定基线 `0.011510`。

| 重点类别 | AP50-95 变化（wbf_edge - baseline_full） | Recall@0.25 变化 |
| --- | ---: | ---: |
| `wuzi` | +0.006584 | +0.047953 |
| `waterstain` | -0.032347 | -0.039567 |
| `airhole` | -0.036854 | -0.113861 |
| `bubble` | -0.077809 | -0.239437 |
| `greenstain` | +0.144997 | +0.303030 |
| `watermark` | -0.116453 | -0.214580 |

## 5. 逐框归因

| 模型 | 原始 crop 预测 | conf < 0.25 | 触及内部切片边缘 | `legacy` 输出 | `wbf_edge` 输出 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定基线 | 1,309,610 | 1,260,957（96.28%） | 689,069（52.62%） | 1,084,406 | 21,789 |
| 切片候选 | 470,692 | 459,709（97.67%） | 326,895（69.45%） | 375,469 | 3,458 |

低置信预测和边缘预测的占比确实极高，`wbf_edge` 也显著减少了输出框数量；但模型精度没有恢复。因此“后处理失效是全部回退原因”的假设不成立，不能把框数清理本身视为切片训练有效。

## 6. 决策

- 闸门结果：`fail_stop_tile_direction`。
- 不启动 Phase 2 的 512 切片重训，不恢复或续训 G4b `last.pt`，不扩展 Fold 2/3，也不启动 Phase 3 的 P2 检测头。
- 保持既有 640 `auto_only` 候选和历史交付不变。
- 若未来继续小目标研究，应作为新的、隔离的数据增强或采样实验重新立项；可评估 Phase 4 Copy-Paste，但不能复用本次切片结论宣称收益。

## 7. 文件位置

| 文件 | 路径 |
| --- | --- |
| Kaggle 任务 | `https://www.kaggle.com/code/lly3323w/l1-ryw18-y11m-oof-v1-tile-eval-wbf-f01` |
| 已下载归档 | `E:/PythonProject/优化/kaggle_l1_ryw18_y11m_oof_v1_tile_eval_wbf_f01_20260822/remote_output_v1/dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_tile1024o25_focus6_mix_fold01_evaluation_results.zip` |
| 校验/解压目录 | `E:/PythonProject/优化/kaggle_l1_ryw18_y11m_oof_v1_tile_eval_wbf_f01_20260822/remote_output_v1_extracted/` |
| 可读校验报告 | `E:/PythonProject/优化/kaggle_l1_ryw18_y11m_oof_v1_tile_eval_wbf_f01_20260822/remote_output_v1_extracted/phase1_verification_report.md` |
| 归档校验脚本 | `E:/PythonProject/优化/verify_l1_phase1_tile_evaluation.py` |
| 逐框归因 | `E:/PythonProject/优化/kaggle_l1_ryw18_y11m_oof_v1_tile_eval_wbf_f01_20260822/remote_output_v1_extracted/tile_evaluation/tiled_attribution_*.json` |
