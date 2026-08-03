# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo11m_fold01_kaggle_seed0` |
| 日期 | 2026-08-03 |
| 执行人 | Codex |
| 运行环境 | Kaggle 私有 Script Kernel |
| Kernel | `lly3323w/l1-ryw18-yolo11m-oof-consensus-v1-fold01` |
| GPU / Python / Torch / Ultralytics | Tesla P100-PCIE-16GB / Python 3.12.13 / Torch 2.6.0+cu124 / Ultralytics 8.4.66 |
| 当前状态 | Kaggle version 1 `COMPLETE`；结果 ZIP 已校验、归档并完成本机开发集复评 |

## 2. 数据信息

| 项目 | 内容 |
| --- | --- |
| 数据集 | `dianjienie` RYW18 OOF 自动共识候选开发集 |
| 数据版本 | `oofconsensusv1` / `dianjienie_ryw18_oofconsensus_v1_dev_only_r1` |
| Kaggle 图片源 | `lly3323w/dianjienie`，只读图片仓库 |
| Kaggle 配置源 | `lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config` |
| train / val 图片数 | 751 / 376 |
| 冻结 test | 不包含、不读取、不评估 |
| 类别数 | 18 类 RYW18 |
| 标签版本 | 15,426 个历史框 + 245 个 OOF accepted 新框 = 15,671 框；非人工真值 |
| 远端预检 | `COMPLETE`，确认 1,127 图、1,127 标签、`development_only=true` |

## 3. 训练配置

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

## 4. 结果

- 结果 ZIP 通过收集器校验：29 个文件、CRC/必需文件/权重/`requested_config.json`/仅开发集 `train.txt` 与 `val.txt` 均通过；SHA-256=`0c0db6afcb9ef8503b2d2f35cb4139ba111fda55638803c414d8a54903bec430`。
- 训练期 `results.csv` 的 mAP50-95 最大行：epoch 190，P/R/mAP50/mAP50-95=`0.51399/0.29538/0.30617/0.16382`。
- 本机开发集复评：376 张 Fold 1 val 图、4,835 个实例，P/R/mAP50/mAP50-95=`0.49024/0.30787/0.31292/0.16513`。
- 复评记录显式为 `development_only=true`；`frozen_test_list_not_evaluated` 与 `frozen_test_images_not_evaluated` 均为 `null`。`youpao`、`mijixiaokeng`、`fanxitong` 在该验证折无真值，不可评。
- 此结果仅是候选标签版本的单 Fold/单 seed 证据，不能与历史 anchor 指标直接排名，也不能据此选择最终模型。

## 5. 文件位置

| 文件 | 路径 |
| --- | --- |
| 预期 run | `E:/PythonProject/models/runs/dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold01` |
| 预期归档 | `dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold01_results.zip` |
| Kaggle URL | `https://www.kaggle.com/code/lly3323w/l1-ryw18-yolo11m-oof-consensus-v1-fold01` |
| 完成后归档 | `collect_l1_kaggle_run.py --development-only --kernel lly3323w/l1-ryw18-yolo11m-oof-consensus-v1-fold01 --run-name dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold01 --install` |
| 归档校验 | `E:/PythonProject/models/runs/dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold01/kaggle_collection.json` |
| 本机复评 | `E:/PythonProject/优化/l1_reeval_yolo11m_oofconsensusv1_fold01/metrics_per_class_reeval.json` |
