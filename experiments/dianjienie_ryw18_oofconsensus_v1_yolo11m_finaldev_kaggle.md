# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo11m_finaldev_kaggle_seed0` |
| 所属主线 | L1 |
| 日期 | 2026-08-15 |
| 执行人 | Codex |
| 实验状态 | complete |
| 证据等级 | provisional |
| 父计划 / 决策闸道 | `当前工作计划.md` G3/G5；`l1_auto_gate_oofconsensusv1_20260809/report.md` |
| 研究假设 | 自动闸门选择的 YOLO11m 可在冻结的 OOF 候选开发视图上完成固定 epoch 训练、一次发布确认与 ONNX 导出。 |
| 只改变的主要变量 | 训练对象为自动主候选 `yolo11m.pt`；epoch 由三折曲线冻结为 83。 |
| 运行环境 | Kaggle 私有 Script Kernel；后续确认在本机 Ultralytics 8.4.66 / Torch 2.6.0+cu124。 |
| GPU | Kaggle Tesla P100-PCIE-16GB；发布确认使用本机 GTX 1660 Ti。 |

## 2. 数据信息

| 项目 | 内容 |
| --- | --- |
| 数据集 | `dianjienie` RYW18 OOF 自动共识候选开发集 |
| 数据版本 | `oofconsensusv1` |
| train 图片数 / 框数 | 1,127 / 15,671 |
| val 图片数 | 1,127，仅供 Ultralytics 解析；训练配置 `val=false` |
| 类别数与顺序 | 18 类 RYW18，由 `classes.txt` 和归档 `data.yaml` 校验一致 |
| 数据与标签边界 | 保留 15,426 个历史框和 245 个 OOF accepted 新框；不覆盖历史标签；自动候选不是人工真值 |
| 冻结测试集访问 | 训练未读；选模后执行且仅执行一次 198 图历史 release-validation 自动确认 |

## 3. 训练配置

| 参数 | 值 |
| --- | --- |
| model / pretrained | `yolo11m.pt` / 官方预训练，不使用 `ryw.pt` 初始化 |
| epochs / completed epochs | 83 / 83 |
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
| 训练期最终行 P / R / mAP50 / mAP50-95 | 0.81674 / 0.68790 / 0.76772 / 0.54049 |
| 历史 release-validation P / R / mAP50 / mAP50-95 | 0.42582 / 0.38473 / 0.35959 / 0.23147 |
| release-validation 图片 / 历史框 | 198 / 2,806 |
| ONNX 一致性 | 通过；8 张开发图同类 IoU>=0.90 全局匹配率 0.96617，单图检测数最大差 1 |
| 产物完整性 | 完整 |

训练期指标来自全开发集日志，不能作为发布性能。发布确认只验证已经预选的 YOLO11m，不能用于与 YOLO26m 后验比较。逐类 release 指标见 `l1_final_release_yolo11m_oofconsensusv1_ep083_20260815_2158/metrics_per_class_release.json`；`NiOH`、`youpao`、`heibian`、`mijixiaokeng`、`fanxitong` 在该冻结集无真值，不能评价。

## 5. 结论

- Kaggle 训练、归档、固定权重哈希、唯一发布确认和 ONNX 导出均完成；YOLO11m 形成 L1 的 `auto_only` 候选交付。
- 结果不证明标签已经人工正确，也不证明 YOLO11m 具有多 seed 稳定架构优势。
- YOLO26m 保持完整归档辅助候选，但不访问冻结历史集，避免反向选模。
- ONNX 对照中的 `conf=0.25` 只用于技术兼容性检查；生产后处理阈值仍需在固定人工评测集上单独标定。

## 6. 文件位置

| 文件 | 路径 |
| --- | --- |
| run | `E:/PythonProject/models/runs/dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_finaldev_ep083/` |
| 固定权重 | `.../weights/final_fixed_epoch.pt` |
| release-validation | `E:/PythonProject/优化/l1_final_release_yolo11m_oofconsensusv1_ep083_20260815_2158/` |
| ONNX / 一致性报告 | `E:/PythonProject/优化/l1_onnx_consistency_yolo11m_oofconsensusv1_ep083_20260815_2230_semantic_v1/` |
| 部署 ONNX 副本 | `.../exports/onnx/final_fixed_epoch.onnx` |
| Kaggle Kernel | `https://www.kaggle.com/code/lly3323w/l1-ryw18-yolo11m-oof-consensus-v1-finaldev` |
