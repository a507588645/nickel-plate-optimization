# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo11m_fold03_kaggle_seed0` |
| 日期 | 2026-08-08 |
| 执行人 | Codex |
| 运行环境 | Kaggle 私有 Script Kernel |
| Kernel | `lly3323w/l1-ryw18-yolo11m-oof-consensus-v1-fold03` |
| 当前状态 | 已完成：结果 ZIP 已校验、归档并完成固定开发集 Fold 3 统一复评 |
| 证据等级 | `provisional`：归档和统一复评完整；固定人工评测未完成，不能用于最终模型选择 |

## 2. 假设与比较边界

- 目标：在与候选版 YOLO11m Fold 1/2 相同的数据版本、训练配置下补齐 Fold 3 OOF 证据。
- 提交前提：YOLO11m Fold 2 已通过结构归档校验；本次只复用固定配置，不基于 Fold 2 训练期指标改变模型、阈值或数据。
- 本次与仍在运行的 YOLO26m Fold 2 占用两个独立 GPU 槽。并行只缩短日历时间，不能增加总配额，也不能替代统一复评或人工评测。

## 3. 数据信息

| 项目 | 内容 |
| --- | --- |
| 数据集 | `dianjienie` RYW18 OOF 自动共识候选开发集 |
| 数据版本 | `oofconsensusv1` / `dianjienie_ryw18_oofconsensus_v1_dev_only_r1` |
| Kaggle 图片源 | `lly3323w/dianjienie`，只读图片仓库 |
| Kaggle 配置源 | `lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config` |
| train / val 图片数 | 751 / 376 |
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

- 提交前 Kaggle 账户设置页显示 GPU `13:23 / 30 hrs`。
- 本地 Kernel 脚本与 `prepare_l1_oofconsensus_kaggle_kernel.py` 的 Fold 3 渲染结果逐字节一致，脚本 SHA-256=`f3fee5475e22ba4fef606bd510d78fdb9b2a144e06b29def06da54acfe1d40a7`。
- Kernel 元数据与生成器一致；私有 Dataset 引用、GPU 启用状态和代码语法均已校验。
- `kaggle kernels push` 成功返回 version 1；随后状态为 `RUNNING`。

## 6. 结果与后续核验

- 2026-08-09 已通过 `collect_l1_kaggle_run.py --development-only --install` 归档并安装：ZIP SHA-256=`199c86ef7572e76e7409b566cbb3deec186cbad0212fd4ac0672164bebba5c58`，29 个文件条目、CRC、必需产物和 `best.pt`/`last.pt` 均通过。
- `requested_config.json` 为 `development_only=true`；归档 `train/val=751/376` 清单按图片名与本地固定 Fold 3 清单完全一致，未包含或读取 `release-validation`。
- 训练期最大 mAP50-95 为 epoch 136 的 `0.21700`，仅留作训练过程证据。
- 2026-08-09 在与 Fold 1 相同的本机 `E:/PythonProject/models/.venv` 中完成 development-only 复评：P/R/mAP50/mAP50-95=`0.49413/0.39202/0.34760/0.21910`。证据：`E:/PythonProject/优化/l1_reeval_yolo11m_oofconsensusv1_fold03/metrics_per_class_reeval.json`。

## 7. 风险与决策

- 三折候选指标已汇总，但候选标签仍须在固定人工评测契约中与历史锚点比较。
- 单 seed、候选标签自评和稀有类别缺失均不支持最终架构或训练视图选择。
- `youpao`、`fanxitong` 无监督实例；`heibian`、`mijixiaokeng` 等稀有类可能不可评。
