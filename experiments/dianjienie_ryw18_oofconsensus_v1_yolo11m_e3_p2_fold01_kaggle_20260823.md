# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo11m_e3_p2_fold01_kaggle_20260823` |
| 所属主线 | L1 小目标研究 / E3 原生 P2 检测头 |
| 日期 | 2026-08-23 |
| 执行人 | Codex |
| 实验状态 | `not_passed_do_not_expand`；version 4 的远端状态为 `CANCEL_ACKNOWLEDGED`，已回收其最佳检查点并完成固定原图级 Fold 1 诊断评测。 |
| 证据等级 | 仅开发集、单 Fold、单 seed、`auto_only`。因训练在第 94/300 轮附近取消，权重只能作为中断检查点诊断证据，不能视为完整正式训练归档。 |
| 父计划 / 决策闸道 | `L1_小目标优化分阶段计划.md` §5 E3 |
| 研究假设 | 在 640 输入下加入原生 P2（stride 4）检测头，可改善微小缺陷检测而不发生重点类回退。 |
| 唯一主要变量 | YOLO11m 检测头结构：标准 P3/P4/P5/P6 路径改为含 P2/P3/P4/P5 的原生模块 YAML。 |
| 自动停止条件 | 完成归档后，若固定原图级 Fold 1 mAP50-95 相对基线未达到 `+0.020000`，或重点小目标类 AP/Recall 未同向改善，即 `not_passed_do_not_expand`。 |

## 2. 数据与边界

| 项目 | 内容 |
| --- | --- |
| 数据版本 | `dianjienie_ryw18_oofconsensus_v1_dev_only_r1`，18 类、`auto_only` 五列标签。 |
| Fold | 固定 Fold 1：train 751 图，val 376 图。 |
| 冻结测试集 | 不挂载、不读取、不评估；`development_only=true`、`frozen_test_set=null`。 |
| 初始化 | 官方 `yolo11m.pt` 载入 P2 YAML 的可匹配预训练参数；不跨架构加载 `ryw.pt`。 |
| 固定比较锚 | 原图级 `l1_tile_pipeline.py --mode evaluate`；固定基线权重 SHA-256=`b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce`，mAP50-95=`0.16800147577890728`。 |

## 3. 固定训练协议

| 参数 | 值 |
| --- | --- |
| GPU Kernel | [`lly3323w/l1-ryw18-y11m-p2-e3-fold01`](https://www.kaggle.com/code/lly3323w/l1-ryw18-y11m-p2-e3-fold01) |
| epochs / time / plots | 300 上限 / 不传 Ultralytics `time` 小时上限 / version 4 为 `false` |
| imgsz / batch | 640 / 8 |
| workers | version 1 为 8（Kaggle 实际启动 4 个 worker）；version 2/3 为 0，以排除首批次底层进程崩溃。 |
| optimizer / seed / deterministic | SGD / 0 / true |
| resume / patience / save_period | false / 100 / 1 |
| 运行时 | Torch 2.6.0+cu124、Torchvision 0.21.0、NumPy 1.26.4、Ultralytics 8.4.66 |
| 本地结构前置 | P2/P3/P4/P5 步长 `4/8/16/32`，3 个合成 batch 前反向通过；P2=`21,811,608` 参数、`119.3830912` GFLOPs@640。 |

训练按 300 epoch 上限和早停执行，完成后再运行固定原图级评测。`save_period=1` 仅用于中断后回收证据，不以 `last.pt` 恢复续训或改变本次实验协议。

## 4. 提交与验收

- version 1 在完成运行时安装、P2 模型加载、train/val 标签扫描后，于第一个训练批次发生 `Segmentation fault (core dumped)`，Kaggle 状态为 `ERROR`。没有可用 `best.pt`、结果 CSV 或原图级评测，不能形成指标结论。
- version 2 本地任务包：`E:/PythonProject/优化/l1_e3_e4_parallel_kaggle_20260823_r7/kernels/e3_p2/`；代码 SHA-256=`533012c999d6439df7f6b591c27f5a224b48603cfc032216447e31bdc4c64107`。它先运行一次完整 P2、同数据、同 batch 的 1 epoch GPU 预检，再启动新的受限正式训练；预检和正式训练均使用 `workers=0`。
- version 2 的 1 epoch 预检完整通过并满足其 `results.csv` 守卫；新的正式 300 epoch 调用仍在首个 batch 报 `Segmentation fault (core dumped)`。因此预检只能证明 P2 路径可执行一次，不能作为训练结果。
- 对比已完成的同运行时 640 基线，version 2 的正式调用共同额外传入 `time=9.0`，而成功基线为 `time: null`。version 3 删除该参数后仍在同一位置崩溃，故该假设已被否决。
- 固定 Ultralytics 8.4.66 源码显示，`plots=True` 会在首个 batch 完成反向/进度输出后调用 `plot_training_samples`。version 3 的 E3 正式训练恰在该位置崩溃，而 E3 的同数据 `plots=False` 1 epoch 预检完整通过；这是有机制支持但尚未独立证明的候选根因。
- version 4 只关闭正式训练的 `plots`，保留 P2、数据、640、batch 8、SGD、seed、增强、`workers=0`、无 `time` 和每 epoch checkpoint。任务包：`E:/PythonProject/优化/l1_e3_e4_parallel_kaggle_20260823_r9/kernels/e3_p2/`；代码 SHA-256=`d29bd8ef20f2b56c6a7fd75dcb6a22ed06e648796b426cded50c9dc7eadbc504`。Kaggle 实际 Kernel 保持 `fold01`，2026-08-23 16:34 首次查询为 `RUNNING`。
- version 1 错误摘要见 `E:/PythonProject/优化/l1_e3_e4_kaggle_error_evidence_20260823/failure_summary.json`，version 2/3 共同崩溃证据见 `E:/PythonProject/优化/l1_e3_e4_kaggle_error_evidence_20260823/failure_summary_v2.json` / `failure_summary_v3.json`，version 4 提交记录见 `E:/PythonProject/优化/l1_e3_e4_parallel_kaggle_20260823_r9/submission_record.json`。
- version 4 实际写入 `results.csv` 的 epoch 为 `0..93`，最佳训练期为 epoch 86（训练期 mAP50-95=`0.13636`）；远端最终状态为 `CANCEL_ACKNOWLEDGED`，不是完整结果 ZIP。回收的 `best.pt` SHA-256=`4206ee85d9887a07f199fd7a046f1bc3ac22813ab9b672b6021b4ccd552e81c2`，回收清单为 `l1_e3_p2_v4_recovery_20260824_r2/recovery_manifest.json`。
- 已在固定开发集 Fold 1 的 376 张原图上完成同口径评测：基线 mAP50-95=`0.16800132865049708`，P2 候选=`0.13864613633181390`，差值=`-0.02935519231868318`；mAP50 差值=`-0.04013997240538970`。评测 ZIP SHA-256=`c80f5f499a29dfbafa5d5a88f5289466d3f77939ca7160e9f3c3be49e0f40fa7`，路径为 `l1_e3_p2_fold01_original_eval_local_20260824/`。
- 基线以 4096 像素单块覆盖全部验证原图，整图与等价单块切片 mAP50-95 完全一致；评测配置明确 `development_only=true`、`frozen_test_set=null`。因此该差值可直接用于本实验的扩展闸门，但不把中断权重当成最终模型。

## 5. 当前结果与限制

P2 候选未达到 `+0.020000` mAP50-95 闸门，且六个重点类并非全部 AP/Recall 同向改善：`greenstain` 有改善，但 `waterstain`、`airhole`、`bubble`、`watermark` 中至少一项 AP 或 Recall 退化。微平均 Recall 虽由 `0.40972` 升至 `0.43557`，但 Precision 由 `0.46491` 降至 `0.37953`，不能抵消总体 AP 明显下降。

结论为 `not_passed_do_not_expand`：不启动 E3 Fold 2/3、不以 `last.pt` 续训、不替换既有 640 `auto_only` 候选，也不访问 `release-validation`。关闭训练 batch JPG 的假设仅解释 version 1--3 的首批次崩溃，不构成 P2 精度收益证据。
