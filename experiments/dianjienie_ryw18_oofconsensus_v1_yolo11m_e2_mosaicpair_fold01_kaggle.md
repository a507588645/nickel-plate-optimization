# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo11m_e2_mosaicpair_fold01_kaggle` |
| 所属主线 | L1 小目标研究 / E2 成对 Mosaic 消融 |
| 日期 | 2026-08-22 至 2026-08-23 |
| 执行人 | Codex |
| 实验状态 | `complete`；两个 Kaggle 私有 GPU Kernel 均为 `COMPLETE`，成对归档验收结果为 `not_passed_do_not_expand`。 |
| 证据等级 | `provisional`；成对合同、权重、评测和自动闸门均完整，但仅限单 Fold、单 seed、`auto_only` 开发证据。 |
| 父计划 / 决策闸道 | `L1_小目标优化分阶段计划.md` §5 E2 |
| 研究假设 | 在 4096x3554 原图经 640 输入训练时，持续 Mosaic 会把微小缺陷缩小到近乎不可见；关闭 Mosaic 可提升小目标检测。 |
| 唯一主要变量 | `mosaic=1.0` 对照与 `mosaic=0.0` 候选。 |
| 停止条件 | 任一归档不完整、契约不一致、候选相对对照 mAP50-95 < `+0.01`、任一焦点类 Recall 下降，或候选相对固定基线 < `+0.01`。 |
| 运行环境 | Kaggle 私有 GPU Kernel；本机仅完成 staging、提交和后续归档校验，不运行常驻监控器。 |

## 2. 数据与边界

| 项目 | 内容 |
| --- | --- |
| 数据集 | `dianjienie` RYW18 OOF 自动共识候选开发集 |
| 数据版本 | `oofconsensusv1`，`auto_only` 五列标签 |
| Fold | 固定 Fold 1：train 751 图、val 376 图 |
| 类别 | 固定 RYW18 18 类顺序 |
| 共享初始化权重 | `dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold01/weights/best.pt` |
| 初始化权重 SHA-256 | `b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce` |
| 共享权重 Dataset | 私有 `lly3323w/dianjienie-ryw18-oof-yolo11m-fold01-640-weight`；其 `baseline_manifest.json` 已核验同一 SHA-256，两个 Kernel 共同挂载，不创建重复 Dataset 版本。 |
| 冻结测试集 | 不挂载、不读取、不评估；`development_only=true`、`frozen_test_set=null` |
| 固定整图评测锚 | `l1_tile_pipeline.py --mode evaluate`，Fold 1 `baseline_full` mAP50-95=`0.16800147577890728` |

## 3. 成对训练协议

两项任务从相同的共享检查点启动新的训练任务，均为 `resume=false`。成对设计消除“30 epoch 重新微调本身带来变化”的混杂因素；不得用旧的单侧 `Mosaic=0` 任务包做结论或提交。

| 参数 | 两侧固定值 |
| --- | --- |
| model / base checkpoint | YOLO11m / 共享 Fold 1 `best.pt` |
| epochs / imgsz / batch | 30 / 640 / 8 |
| optimizer / seed / deterministic | SGD / 0 / true |
| resume / close_mosaic / patience | false / 0 / 100 |
| lr0 / lrf / momentum / weight_decay | 0.01 / 0.01 / 0.937 / 0.0005 |
| warmup_epochs / save_period | 3.0 / 1 |
| 其余增广 | hsv、translate、scale、fliplr、RandAugment、erasing 与归档 Fold 1 一致 |
| 运行时 | Torch 2.6.0+cu124、Torchvision 0.21.0、NumPy 1.26.4、Ultralytics 8.4.66 |

| 角色 | Kaggle Kernel | run_name | Mosaic |
| --- | --- | --- | ---: |
| 对照 | `lly3323w/l1-ryw18-y11m-oof-v1-mosaic1-ft30-f01` | `dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_mosaic1_ft30_fold01` | 1.0 |
| 候选 | `lly3323w/l1-ryw18-y11m-oof-v1-mosaic0-ft30-f01` | `dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_mosaic0_ft30_fold01` | 0.0 |

评测在每项训练结束后使用同一原图级评测器执行。`tile_size=4096`、`overlap=0.0` 仅确保每张原图有一个完整 crop，不进行切片训练、切片推理或跨切片 WBF。

## 4. 自动验收闸门

仅在两个完成 ZIP 均通过 CRC、路径安全、共享契约哈希、基础权重哈希、运行时版本、751/376 不重叠清单、实际 `args.yaml`、评测脚本哈希和原图级评测证据的成对校验后，才评估以下闸门：

| 检查 | 通过条件 |
| --- | --- |
| 因果主比较 | `Mosaic=0 mAP50-95 - Mosaic=1 mAP50-95 >= +0.010000` |
| 焦点类 Recall | `wuzi/waterstain/airhole/bubble/greenstain/watermark` 的 Recall@conf0.25/IoU0.5 均不低于对照 |
| 实用性锚 | `Mosaic=0 mAP50-95 - fixed baseline_full >= +0.010000` |

三项全部通过才记录为 `pass_expand_eligible`。这仍只是单 Fold、单 seed、`auto_only` 的“是否值得另行授权扩展”证据，不代表最终模型或人工真值。任一失败即 `not_passed_do_not_expand`，不启动 Fold 2/3。

## 5. 已核验结果与决定

两个任务均已完成，且使用同一开发集 Fold 1 的 376 张原图评测。完整 Kaggle 输出 ZIP 各约 2.32 GB，包含无关的大型临时评测工作区；为避免把原始图片和临时数据混入本地实验产物，使用只读范围提取器从远端 ZIP 的中央目录提取 14 项必需证据，逐项核验源 ZIP CRC 后构建最小证据包，再由成对校验器复核合同和闸门。

| 指标 | Mosaic=1 对照 | Mosaic=0 候选 | 候选差值 |
| --- | ---: | ---: | ---: |
| 原图级 mAP50-95 | 0.162465 | 0.155948 | -0.006518 |
| 相对固定 `baseline_full=0.168001` | -0.005536 | -0.012054 | 候选未达 `+0.010000` |

| 焦点类 | Mosaic=0 − Mosaic=1 Recall@conf0.25/IoU0.5 |
| --- | ---: |
| wuzi | -0.031579 |
| waterstain | +0.000471 |
| airhole | -0.016502 |
| bubble | -0.021127 |
| greenstain | +0.020202 |
| watermark | +0.050894 |

验收器确认两侧共享同一基础权重 SHA-256、同一 751/376 无交集清单、同一非 Mosaic 训练参数、固定运行时和同一原图级评测器；两个 `best.pt` 均与各自评测报告哈希一致。因主比较、实用性锚和焦点类 Recall 三项均未通过，最终状态为 `not_passed_do_not_expand`：**不启动 E2 Fold 2/3，不使用 Mosaic=0 权重替换既有 640 `auto_only` 候选，也不访问 `release-validation`。**

- 对照：[`lly3323w/l1-ryw18-y11m-oof-v1-mosaic1-ft30-f01`](https://www.kaggle.com/code/lly3323w/l1-ryw18-y11m-oof-v1-mosaic1-ft30-f01)。
- 候选：[`lly3323w/l1-ryw18-y11m-oof-v1-mosaic0-ft30-f01`](https://www.kaggle.com/code/lly3323w/l1-ryw18-y11m-oof-v1-mosaic0-ft30-f01)。

## 6. 文件位置

| 文件 | 路径 |
| --- | --- |
| 成对任务生成器 | `E:/PythonProject/优化/prepare_l1_e2_mosaic_pair_kaggle_kernels.py` |
| 已提交本地 staging | `E:/PythonProject/优化/l1_e2_mosaic_pair_20260822_r5/` |
| 共享检查点审计镜像 | `E:/PythonProject/优化/l1_e2_mosaic_pair_20260822_r5/checkpoint_dataset/` |
| 对照 Kernel staging | `E:/PythonProject/优化/l1_e2_mosaic_pair_20260822_r5/control_kernel/` |
| 候选 Kernel staging | `E:/PythonProject/优化/l1_e2_mosaic_pair_20260822_r5/candidate_kernel/` |
| 远端共享权重 Dataset | `lly3323w/dianjienie-ryw18-oof-yolo11m-fold01-640-weight` |
| 结果成对校验器 | `E:/PythonProject/优化/verify_l1_e2_mosaic_pair.py` |
| 校验器单元测试 | `E:/PythonProject/优化/test_verify_l1_e2_mosaic_pair.py` |
| 最小证据范围提取器 | `E:/PythonProject/优化/collect_l1_e2_minimal_evidence.py` |
| 对照/候选最小证据与来源清单 | `E:/PythonProject/优化/l1_e2_mosaic_pair_minimal_evidence_20260823/` |
| 独立评测 JSON 镜像 | `E:/PythonProject/优化/l1_e2_mosaic_pair_remote_evidence_20260823/` |
| 成对验收报告、决定与安全解压证据 | `E:/PythonProject/优化/l1_e2_mosaic_pair_verification_20260823/` |
| 旧单侧包（仅追溯，禁止提交） | `E:/PythonProject/优化/kaggle_l1_ryw18_y11m_oof_v1_nomosaic_ft30_f01_20260822/` |
