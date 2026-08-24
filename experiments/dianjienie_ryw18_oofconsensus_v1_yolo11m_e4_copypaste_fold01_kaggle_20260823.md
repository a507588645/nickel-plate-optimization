# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo11m_e4_copypaste_fold01_kaggle_20260823` |
| 所属主线 | L1 小目标研究 / E4 稀有类 Copy-Paste |
| 日期 | 2026-08-23 |
| 执行人 | Codex |
| 实验状态 | `not_passed_do_not_expand`；version 4 已产生训练期 `results.csv` 和最佳检查点，但远端在训练后的评估配置读取处报错，最终 Kernel 状态为 `ERROR`。 |
| 证据等级 | 仅开发集、单 Fold、单 seed、`auto_only`。训练阶段的权重已回收并完成独立固定原图级评测；该结果不构成多 Fold 稳定性结论。 |
| 父计划 / 决策闸道 | `L1_小目标优化分阶段计划.md` §5 E4 |
| 研究假设 | 在不改变验证集、原始训练图和历史标签的前提下，少量真实小缺陷 Copy-Paste 可改善稀有/微小类检测。 |
| 唯一主要变量 | Fold 1 训练清单中增加 36 张冻结几何记录生成的合成图；其余训练协议与固定 Fold 1 基线一致。 |
| 自动停止条件 | 完成归档后，若固定原图级 Fold 1 mAP50-95 相对基线未达到 `+0.020000`，或稀有类 AP/Recall 无改善或有实质回退，即 `not_passed_do_not_expand`。 |

## 2. 数据与边界

| 项目 | 内容 |
| --- | --- |
| 基础数据版本 | `dianjienie_ryw18_oofconsensus_v1_dev_only_r1`，18 类、`auto_only` 五列标签。 |
| E4 派生视图 | `l1_e4_oofconsensus_v1_cp_v1_20260823`；仅提供 Fold 1 所需标签、清单、36 条冻结合成记录与验证证据。 |
| Fold | 原始 train 751 图加 36 张合成图，共 787 图；原始 val 保持 376 图，零交集。 |
| 合成边界 | 供体与目标均来自同一 Fold 的原训练集，补丁无缩放、与已有标签零相交；源标签树未改写。 |
| 冻结测试集 | 不挂载、不读取、不评估；`development_only=true`、`frozen_test_set=null`。 |
| 初始化和比较锚 | 使用固定 Fold 1 基线权重 SHA-256=`b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce`；固定原图级基线 mAP50-95=`0.16800147577890728`。 |

## 3. 固定训练协议

| 参数 | 值 |
| --- | --- |
| GPU Kernel | [`lly3323w/l1-ryw18-y11m-cp-e4-fold01`](https://www.kaggle.com/code/lly3323w/l1-ryw18-y11m-cp-e4-fold01) |
| 训练数据挂载 | `lly3323w/dianjienie`、OOF 开发配置、固定权重与私有 `lly3323w/dianjienie-ryw18-oof-cp-v1-fold01-r2` payload。 |
| epochs / time / plots | 300 上限 / 不传 Ultralytics `time` 小时上限 / version 4 为 `false` |
| imgsz / batch | 640 / 8 |
| optimizer / seed / deterministic | SGD / 0 / true |
| resume / patience / save_period | false / 100 / 1 |
| 运行时 | Torch 2.6.0+cu124、Torchvision 0.21.0、NumPy 1.26.4、Ultralytics 8.4.66 |

Kaggle 运行时按已审计的供体、目标、裁剪和位置记录重建 36 张合成图，然后以原始 376 张验证图执行固定原图级评测；不执行切片训练或切片推理。

## 4. 提交与验收

- version 1 在模型和依赖加载前 `ERROR`：脚本从重叠的 Kaggle 挂载根递归发现同一份 `payload_manifest.json` 多次，并错误地拒绝它为非唯一输入。未开始训练、未写模型权重或指标。
- version 2 本地任务包：`E:/PythonProject/优化/l1_e3_e4_parallel_kaggle_20260823_r7/kernels/e4_copypaste/`；代码 SHA-256=`139003279d304e9287a878820e5cd4b5db95fa9ee655826b314e6a9b251314c0`。它按绝对路径去重再执行唯一 payload 检查，已用重叠根模拟测试验证。
- version 2 已成功通过 payload 去重、运行时视图构建和 train/val 扫描，并进入首个正式 batch；随后也报 `Segmentation fault (core dumped)`。这证明 version 1 的 payload 问题已修复，但没有产生模型或指标。
- 对比已完成的同运行时 640 基线，version 2 的正式调用共同额外传入 `time=9.0`，而成功基线为 `time: null`。version 3 删除该参数后仍在同一位置崩溃，故该假设已被否决。
- 固定 Ultralytics 8.4.66 源码显示，`plots=True` 会在首个 batch 完成反向/进度输出后调用 `plot_training_samples`。E4 version 3 正式训练恰在该位置崩溃；该候选根因由 E3 的同数据 `plots=False` 一轮预检通过进一步支持，但尚未独立证明。
- version 4 只关闭正式训练的 `plots`，保留 E4 的 36 张合成图、数据边界、640、batch 8、SGD、seed、增强、无 `time` 和每 epoch checkpoint。任务包：`E:/PythonProject/优化/l1_e3_e4_parallel_kaggle_20260823_r9/kernels/e4_copypaste/`；代码 SHA-256=`606b96b9cb69093a867a56cc899e347556cea53bf5150d79c379ce9ea357b78d`。2026-08-23 16:34 首次查询为 `RUNNING`。
- E4 payload 的 1,172 个必要文件已在提交前复核；独立派生视图验证报告为 `l1_e4_oofconsensus_v1_cp_v1_20260823/manifests/verification_report.json`。version 1 错误摘要见 `E:/PythonProject/优化/l1_e3_e4_kaggle_error_evidence_20260823/failure_summary.json`，version 2/3 共同崩溃证据见 `E:/PythonProject/优化/l1_e3_e4_kaggle_error_evidence_20260823/failure_summary_v2.json` / `failure_summary_v3.json`，version 4 提交记录见 `E:/PythonProject/优化/l1_e3_e4_parallel_kaggle_20260823_r9/submission_record.json`。
- version 4 写入 234 行训练记录（epoch `0..234`），最佳训练期为 epoch 134（训练期 mAP50-95=`0.16505`）。随后远端评估配置缺少 `evaluation_predict_batch` 并报 `KeyError`，故 Kaggle 状态为 `ERROR`；这是训练后收尾错误，不把它误记为完整远端归档。
- 已回收的 `best.pt` SHA-256=`d75500d532ee7ca3984ed3000882193a356b5f130f4a0829cf2c9d71ef61b00b`，其回收清单为 `l1_e4_copypaste_v4_recovery_20260824_r2/recovery_manifest.json`。本地评测使用同一固定开发集 Fold 1 的 376 张原图和基线，无重建训练视图、无切片训练或推理。
- 固定原图级结果为：基线 mAP50-95=`0.16800132865049708`，Copy-Paste 候选=`0.15976185710342633`，差值=`-0.00823947154707075`；mAP50 差值=`-0.01678289874826322`。评测 ZIP SHA-256=`1756a30ab6bbfbf588133a75a3035cc67cac761c15e3fc14ef1ef8e4350607cc`，路径为 `l1_e4_copypaste_fold01_original_eval_local_20260824/`；配置明确 `development_only=true`、`frozen_test_set=null`。

## 5. 当前结果与限制

Copy-Paste 候选未达到 `+0.020000` mAP50-95 闸门，实际差值为负。`yellowstain`、`quejiao` 等个别稀有类有改善，但 `nodule`、`heibian` 明显退化；微平均 Recall 由 `0.40972` 升至 `0.43082`，同时 Precision 由 `0.46491` 降至 `0.44178`，不能支持总体收益结论。

结论为 `not_passed_do_not_expand`：不启动 E4 Fold 2/3、不以任何检查点续训、不替换既有 640 `auto_only` 候选，也不访问 `release-validation`。Copy-Paste 视图仍只证明供体/目标同 Fold、无缩放和与已有标签零相交；它不能证明目标背景不存在漏标，故不得把本次自动合成标签当作人工真值。
