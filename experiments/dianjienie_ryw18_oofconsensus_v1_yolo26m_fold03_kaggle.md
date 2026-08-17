# 模型优化实验记录

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo26m_fold03_kaggle_seed0` |
| 日期 | 2026-08-09 |
| 执行人 | Codex |
| Kernel | `lly3323w/l1-ryw18-yolo26m-oof-consensus-v1-fold03` |
| 当前状态 | 已完成：结果 ZIP 已校验、归档并完成固定开发集 Fold 3 统一复评 |
| 证据等级 | `provisional`：归档和统一复评完整；固定人工评测未完成，不能用于最终模型选择 |

## 数据与配置

- 数据视图：`dianjienie_ryw18_oofconsensus_v1_dev_only_r1`，仅开发集，18 类、15,671 框；`release-validation` 不读、不链接、不评估。
- 固定 Fold 3：train/val=`751/376`；官方 `yolo26m.pt`，300 epochs、640、batch 8、SGD、seed 0、patience 100、`resume=false`。

## 提交前验证与决定

- 生成器渲染、`kernel-metadata.json` 与提交脚本一致，语法编译通过；脚本 SHA-256=`acb1cd4757ca6d034eaee46781414a4c10c9eb0a37c9c5f310d81d54daa3b3d1`。
- 2026-08-09 GPU 为 `19:58 / 30 hrs`，YOLO11m Fold 3 仍为 `RUNNING`；YOLO26m Fold 2 实际累计约 9 小时 26 分。
- 用户明确确认 Kaggle 会自行完成，要求直接增加本任务；`kaggle kernels push` 已成功返回 version 1，随后查询为 `RUNNING`。未启动本机监控器。

## 归档与统一复评

- 2026-08-09 已通过 `collect_l1_kaggle_run.py --development-only --install` 归档并安装：ZIP SHA-256=`3fbeadb462ecfd0c7955de9ae0603c37bfb953eec9d3dfad6e31fdbe97a0d8d4`，29 个文件条目、CRC、必需产物和 `best.pt`/`last.pt` 均通过。
- `requested_config.json` 为 `development_only=true`；归档 `train/val=751/376` 清单按图片名与本地固定 Fold 3 清单完全一致，未包含或读取 `release-validation`。
- 训练期最大 mAP50-95 为 epoch 153 的 `0.21891`，仅留作训练过程证据。
- 2026-08-09 在与 Fold 1 相同的本机 `E:/PythonProject/models/.venv` 中完成 development-only 复评：P/R/mAP50/mAP50-95=`0.54795/0.27801/0.35582/0.22548`。证据：`E:/PythonProject/优化/l1_reeval_yolo26m_oofconsensusv1_fold03/metrics_per_class_reeval.json`。

## 风险

- 三折候选指标已汇总，但候选标签仍须在固定人工评测契约中与历史锚点比较。
- 单 seed、候选标签自评和稀有类别缺失均不支持最终架构或训练视图选择。
