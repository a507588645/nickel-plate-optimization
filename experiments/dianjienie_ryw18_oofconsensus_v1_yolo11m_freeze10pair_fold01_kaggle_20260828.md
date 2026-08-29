# L1 RYW18 YOLO11m Freeze10 成对消融（Fold 1）

| 字段 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo11m_freeze10pair_fold01_kaggle_20260828` |
| 主线 | L1 电解镍缺陷检测 |
| 状态 | `completed_not_passed_do_not_train_finaldev`；v2 归档、统一复评和预注册成对闸门均已完成 |
| 证据等级 | `auto_only`、单 Fold/单 seed 的开发集筛选证据；不能表述为人工真值或稳定优势 |
| 用户授权 | 2026-08-28，用户对独立冻结层 GPU 消融回复“允许” |

## 假设与唯一变量

- 假设：从 `yolo11m.pt` 迁移到当前 RYW18 OOF 候选开发视图时，先冻结前 10 个模块 20 epoch、再完整解冻 63 epoch，可能优于相同两阶段训练流程中第一阶段不冻结的对照。
- 唯一变量：第一阶段的 `freeze`，对照为 `0`，候选为 `10`。
- 两个分支均从公开 `yolo11m.pt` 启动，均在第 20 epoch 后从其自身 `last.pt` 新开第二阶段，`resume=False`、`freeze=0`；因此优化器重置不是分支变量。
- 固定总预算：`20 + 63 = 83` epoch，`imgsz=640`、`batch=8`、`SGD`、`seed=0`、`mosaic=1.0`。为规避已知 Kaggle 首批训练图渲染回调崩溃，两个分支都固定 `plots=False`。

## 数据与边界

- 数据版本：`oofconsensusv1` 开发视图，只包含 `1,127` 张图和 `15,671` 个五列候选框。
- 固定 Fold 1：训练 `751` 张、验证 `376` 张；训练与验证不重叠。
- Kaggle 仅挂载 `lly3323w/dianjienie` 图片集和 `lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config` 配置集。
- 不挂载、不读取、不评估冻结测试集或历史 `release-validation`。

## 任务与产物

| 角色 | Kaggle Kernel | 预期 run 名 | 第一阶段 |
| --- | --- | --- | --- |
| 两阶段对照 | `lly3323w/l1-ryw18-y11m-control-s1e20-s2e63-fold-01` | `dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_control_s1e020_s2e063_fold01` | `freeze=0` |
| Freeze10 候选 | `lly3323w/l1-ryw18-y11m-freeze10-s1e20-s2e63-fold-01` | `dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_freeze10_s1e020_s2e063_fold01` | `freeze=10` |

- 任务包根目录：`E:\PythonProject\优化\l1_freeze_pair_20260828_r1\`。
- 训练入口：`train_l1_freeze_two_stage.py`；任务包生成器：`prepare_l1_freeze_kaggle_kernel.py`。
- 两个完成 ZIP 归档后，分别用 `collect_l1_kaggle_run.py --development-only --install` 安装到 `E:\PythonProject\models\runs\`。
- 统一本机复评入口：`reeval_l1_yolo_run.py --development-only --fold 1`。

## 提交证据

- v1 提交时间：`2026-08-28 14:44`；两个任务都在训练前 `ERROR`。远端日志表明开发视图恢复和依赖安装成功，随后训练器对 Kaggle 图片符号链接调用 `resolve()`，将锚点名还原为源名 `1.png`，错误查找了不存在的 `labels/1.txt`。两条任务均未进入 Stage 1、未产生权重或指标，不能用于比较。
- 修复：标签审计和全开发集清单比较改为保留恢复视图中的锚点路径，不解析图片符号链接；在本机调用同一配置恢复器后，重新审计到 `1,127` 图、`15,671` 框和 Fold 1 `751/376`，通过。
- v2 提交时间：`2026-08-28 15:01`；两个任务均为 Kaggle version `2`，随后均 `COMPLETE`。
- 提交包记录：`E:\PythonProject\优化\l1_freeze_pair_20260828_r1\submission_record.json`。
- v1 训练器 SHA-256：`b2c228be3b1e8bf70d754f38d118815e46682e02a35c0c04d18e1ffe25c02bef`；v2 修复训练器 SHA-256：`5fcc0306839dba93477e1f9a3f344599da5dbf307cad2f38052ef00efdf310c4`。
- v2 Kernel 标题与任务标识已一致；两条 v2 脚本在忽略 run 名、变体名和 `freeze` 值后完全一致。

## 预注册闸门

仅在两个 ZIP 的运行时、参数、开发集边界和统一复评全部校验通过后，执行 `verify_l1_freeze_pair.py`。

1. `Freeze10 mAP50-95 - 对照 mAP50-95 >= +0.010000`。
2. `wuzi`、`waterstain`、`airhole`、`bubble`、`greenstain`、`watermark` 的 Recall 均不得低于对照。
3. 任何一项失败：状态为 `not_passed_do_not_train_finaldev`，不训练全开发集、不导出新的部署 ONNX。
4. 全部通过：状态为 `pass_finaldev_eligible`，仅表示可申请下一步全开发集固定 epoch 训练；单 Fold、单 seed 结果不得表述为稳定提升。

## v2 归档、复评与闸门结果

- 两个结果 ZIP 均经 CRC、16 个文件归档、必需配置/权重和开发集边界校验后安装。对照 ZIP SHA-256 为 `4064cb1f1fda7a84ec1c8f97fdd9f2892bc26e94c1fc031bd2ea004e2338089e`，Freeze10 ZIP SHA-256 为 `73732e615378734276c2106358c326eb45dadc9f40f8bcf1e78dca89cf78ae0c`。
- 阶段参数核验：对照 Stage 1 为 `freeze=0`，Freeze10 Stage 1 为 `freeze=10`；两组 Stage 2 均为 `freeze=0`，其余已核验训练参数和 Kaggle 运行时一致。
- 统一本机复评固定为同一 OOF 开发集 Fold 1 的 `376` 张验证图，`imgsz=640`、`batch=4`、`seed=0`、Ultralytics `8.4.66`；未读取冻结测试集或历史 `release-validation`。

| 指标 | 两阶段对照 | Freeze10 | Freeze10 - 对照 |
| --- | ---: | ---: | ---: |
| Precision | 0.587103 | 0.432034 | -0.155069 |
| Recall | 0.310667 | 0.309154 | -0.001513 |
| mAP50 | 0.315673 | 0.303669 | -0.012005 |
| mAP50-95 | 0.162499 | 0.155994 | -0.006505 |

| 重点类别 Recall | Freeze10 - 对照 |
| --- | ---: |
| `wuzi` | -0.017544 |
| `waterstain` | +0.000713 |
| `airhole` | -0.007044 |
| `bubble` | +0.021127 |
| `greenstain` | +0.005057 |
| `watermark` | -0.008253 |

- 预注册闸门两项均失败：mAP50-95 未达到 `+0.010000` 增益，且 `wuzi`、`airhole`、`watermark` Recall 下降。
- 最终状态：`not_passed_do_not_train_finaldev`。不训练 Freeze10 全开发集版本，不导出新部署 ONNX，现有已发布 ONNX 保持不变。
- 结构化验收产物：`E:\PythonProject\优化\l1_freeze_pair_verification_20260828_v2\l1_freeze_pair_gate_decision.json` 和 `E:\PythonProject\优化\l1_freeze_pair_verification_20260828_v2\l1_freeze_pair_gate_report.md`。

## 未决风险与下一步

- 风险：候选标签是 `auto_only` 五列视图，不能把该视图上的单 Fold 指标当成人工真值性能。
- 风险：冻结只可能改变迁移初期的收敛与过拟合行为，不能替代标签审计、极小目标输入尺度和新三光域验证。
- 下一步：本轮 Freeze10 路线到此停止；保留归档和审计证据，后续 L1 优化不得把本次单 Fold 结果写成稳定结论，也不得因此替换现有部署 ONNX。
