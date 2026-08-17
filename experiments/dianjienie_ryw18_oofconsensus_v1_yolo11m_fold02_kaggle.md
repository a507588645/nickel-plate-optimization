# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo11m_fold02_kaggle_seed0` |
| 日期 | 2026-08-08 |
| 执行人 | Codex |
| 运行环境 | Kaggle 私有 Script Kernel |
| Kernel | `lly3323w/l1-ryw18-yolo11m-oof-consensus-v1-fold02` |
| 当前状态 | 已完成：结果 ZIP 已校验并归档，2026-08-09 已完成固定开发集 Fold 2 统一复评 |
| 证据等级 | `provisional`：归档和统一复评完整；候选标签仍未经过固定人工评测，不能用于最终模型选择 |

## 2. 假设与比较边界

- 目标：在与 Fold 1 相同的候选开发集、模型、输入尺寸和训练配置下，补齐 YOLO11m 的 Fold 2 OOF 证据。
- 唯一变化：固定 OOF 三折中的验证折从 Fold 1 切换到 Fold 2；不与 `anchor_v1` 的旧标签指标直接排名。
- 数据质量结论仍依赖已冻结的 `l1_oof_evaluation_contract_v1` 人工审查结果；本次训练指标不能自证候选标签质量。

## 3. 数据信息

| 项目 | 内容 |
| --- | --- |
| 数据集 | `dianjienie` RYW18 OOF 自动共识候选开发集 |
| 数据版本 | `oofconsensusv1` / `dianjienie_ryw18_oofconsensus_v1_dev_only_r1` |
| Kaggle 图片源 | `lly3323w/dianjienie`，只读图片仓库 |
| Kaggle 配置源 | `lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config` |
| train / val 图片数 | 752 / 375 |
| 冻结 test / release-validation | 不包含、不读取、不评估 |
| 类别数 | 18 类 RYW18 |
| 标签版本 | 15,426 个历史框 + 245 个 OOF accepted 新框 = 15,671 框；非人工真值 |
| 固定评测契约 | `E:/PythonProject/优化/l1_oof_evaluation_contract_v1`，仅开发集、未填写人工决定 |

## 4. 训练配置

| 参数 | 值 |
| --- | --- |
| model / pretrained weights | `yolo11m.pt` / 官方预训练，不使用 `ryw.pt` 初始化 |
| epochs / imgsz / batch | 300 / 640 / 8 |
| optimizer | SGD |
| lr0 / lrf | 0.01 / 0.01 |
| momentum / weight_decay | 0.937 / 0.0005 |
| warmup_epochs / patience | 3.0 / 100 |
| seed / deterministic | 0 / true |
| close_mosaic / resume | 0 / false |

## 5. 提交前验证

- Kaggle 账户设置页提交前显示 GPU `00:00 / 30 hrs`。
- 本地 Kernel 脚本与 `prepare_l1_oofconsensus_kaggle_kernel.py` 的 Fold 2 渲染结果逐字节一致，脚本 SHA-256=`5199876e9e514121ae46b04848049559c17d4a15bdac564d6645856ea641b150`。
- Kernel 元数据与生成器一致；两个私有 Dataset 引用保持不变，GPU 已启用。
- `kaggle kernels push` 成功返回 version 1；随后 `kaggle kernels status` 返回 `RUNNING`。

## 6. 结果与后续核验

- 结果 ZIP 已通过收集器校验并归档：29 个文件、CRC、必需文件、`best.pt`、`last.pt`、`requested_config.json` 和仅开发集 `752/375` 清单均通过；ZIP SHA-256=`c8dc6d28ac551cf3ec4df54210e9b883e6d3b445218c3126f0f9f34f9edf48fb`。
- 归档路径：`E:/PythonProject/models/runs/dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold02`。
- 训练期 `results.csv` 的最大 mAP50-95 行为 epoch 83，P/R/mAP50/mAP50-95=`0.43059/0.22185/0.27236/0.17269`。这是候选标签训练期指标，不能与 anchor 指标直接排名，也不能用于选择最终模型。
- 2026-08-09 使用与 Fold 1 相同的 `E:/PythonProject/models/.venv` 完成 development-only 固定 Fold 2 复评：P/R/mAP50/mAP50-95=`0.42342/0.23208/0.26962/0.17194`。入口为 `reeval_l1_yolo_run.py --development-only`，仅创建开发集验证视图，不读取 `release-validation`。
- 复评证据：`E:/PythonProject/优化/l1_reeval_yolo11m_oofconsensusv1_fold02/metrics_per_class_reeval.json`；环境为 Python 3.10.11、Ultralytics 8.4.66、Torch 2.6.0+cu124、`imgsz=640`、`batch=4`、`conf=0.001`、`iou=0.7`、`max_det=300`。

## 7. 风险与决策

- 单个 Fold / 单个 seed 不能选择最终架构。
- 该 Fold 已纳入候选版三折统一汇总；同标签候选内的架构差异仍只有单 seed 证据。
- 固定人工评测契约尚未填写，候选标签自评不能证明其标签质量优于历史标签锚点。
- `youpao`、`fanxitong` 无监督实例；`heibian`、`mijixiaokeng` 等稀有类可能不可评。
