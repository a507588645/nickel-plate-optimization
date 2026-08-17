# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo26m_finaldev_kaggle_seed0` |
| 所属主线 | L1 |
| 日期 | 2026-08-15 |
| 执行人 | Codex |
| 实验状态 | complete |
| 证据等级 | provisional |
| 父计划 / 决策闸道 | `当前工作计划.md` G3；`l1_auto_gate_oofconsensusv1_20260809/report.md` |
| 研究假设 | 已通过自动非回退闸门的 YOLO26m 可作为独立全开发集辅助候选完成可追溯训练。 |
| 只改变的主要变量 | 架构为 `yolo26m.pt`；该架构独立三折曲线冻结 epoch=174。 |
| 运行环境 | Kaggle 私有 Script Kernel，Tesla P100-PCIE-16GB，Python 3.12.13，Torch 2.6.0+cu124，Ultralytics 8.4.66。 |

## 2. 数据信息

| 项目 | 内容 |
| --- | --- |
| 数据集 / 版本 | `dianjienie` RYW18 OOF 自动共识候选开发集 / `oofconsensusv1` |
| train 图片数 / 框数 | 1,127 / 15,671 |
| val 图片数 | 1,127，仅供解析；训练配置 `val=false` |
| 类别 | 18 类 RYW18，归档 `data.yaml` 与 `classes.txt` 已校验 |
| 标签边界 | 15,426 历史框 + 245 OOF accepted 新框；自动候选并非人工真值 |
| 冻结测试集访问 | 训练、归档和预检均未读取；为避免后验选模，本 run 不做 release-validation |

## 3. 训练配置

| 参数 | 值 |
| --- | --- |
| model / pretrained | `yolo26m.pt` / 官方预训练，不使用 `ryw.pt` 初始化 |
| epochs / completed epochs | 174 / 174 |
| imgsz / batch | 640 / 8 |
| optimizer | SGD |
| lr0 / lrf | 0.01 / 0.01 |
| momentum / weight_decay | 0.937 / 0.0005 |
| warmup_epochs / patience | 3.0 / 0 |
| seed / deterministic | 0 / true |
| box / cls / dfl | 7.5 / 0.5 / 1.5 |
| mosaic / close_mosaic | 1.0 / 0 |
| 其他增强 | hsv=0.015/0.7/0.4，translate=0.1，scale=0.5，fliplr=0.5，erasing=0.4，randaugment |
| checkpoint 规则 | `final_fixed_epoch.pt` 为 `last.pt` 的固定 epoch 副本，SHA-256 已核验 |

## 4. 结果

| 指标 | 数值 |
| --- | ---: |
| 训练期最终行 P / R / mAP50 / mAP50-95 | 0.90350 / 0.81563 / 0.87638 / 0.67478 |
| 固定发布权重 epoch | 174 |
| 固定权重 SHA-256 | `d0cea7c8ff52534734ddae5ebb54473f3f6d70d4c1a369296e05fe8a709dd227` |
| Kaggle ZIP SHA-256 | `a4c6100d11bfc8dd29039d59164b9a61daa0e06a75603e597753f95d455ae719` |
| 产物完整性 | 完整 |

训练期指标来自全开发集日志，不能作为发布性能或与 YOLO11m 的直接排名。归档后无冻结图像预检通过；本辅助候选不执行 release-validation 和 ONNX 部署导出。

## 5. 结论

- YOLO26m 全开发集固定 epoch 训练和完整归档已经完成，保留为辅助候选和可回溯证据。
- 自动闸门已经在本次发布确认前选择 YOLO11m；不读取同一冻结集是防止测试泄漏的必要边界。
- 无人工 OOF 闸门不等于逐框人工真值，稀有类与无监督类限制仍存在。

## 6. 文件位置

| 文件 | 路径 |
| --- | --- |
| run | `E:/PythonProject/models/runs/dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_oofconsensusv1_finaldev_ep174/` |
| 固定 epoch 选择 | `E:/PythonProject/优化/l1_final_epoch_selection_oofconsensusv1_yolo26m_20260815/` |
| Kaggle Kernel | `https://www.kaggle.com/code/lly3323w/l1-ryw18-yolo26m-oof-consensus-v1-finaldev` |
