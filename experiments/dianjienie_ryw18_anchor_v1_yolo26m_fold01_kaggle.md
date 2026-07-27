# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_anchor_v1_yolo26m_fold01_kaggle_seed0` |
| 日期 | 2026-07-26 提交，2026-07-27 归档与复评 |
| 执行人 | Codex（/root） |
| 运行环境 | Kaggle 私有 Script Kernel |
| Kernel | `lly3323w/l1-ryw18-yolo26m-anchorv1-fold01` |
| GPU | Tesla P100-PCIE-16GB |
| Python | 3.12.13 |
| Torch | 2.6.0+cu124，CUDA 12.4 |
| Ultralytics | 8.4.66 |
| 当前状态 | Kaggle version 1 已完成，结果已通过校验归档，并完成本机固定开发集复评 |

## 2. 数据信息

| 项目 | 内容 |
| --- | --- |
| 数据集 | `dianjienie` RYW18 历史标签锚点 |
| 数据版本 | `dianjienie_ryw18_anchor_v1` |
| Kaggle 图片源 | `lly3323w/dianjienie`，只读图片仓库 |
| Kaggle 配置源 | `lly3323w/dianjienie-ryw18-anchor-v1-config` |
| train 图片数 | 751 |
| val 图片数 | 376 |
| test 图片数 | 198，冻结且未参与训练、早停或调参 |
| 类别数与顺序 | 18 类 RYW18，见锚点 `classes.txt` |
| 是否清理重复图 | 是；固定锚点已完成精确重复排除与近重复分组 |
| 是否通过标签审计 | 是；本机 dry-run 与 Kaggle 预检均通过 |

## 3. 训练配置

| 参数 | 值 |
| --- | --- |
| model | `yolo26m.pt` |
| pretrained weights | 官方 `yolo26m.pt`，不使用 `ryw.pt` 初始化 |
| epochs | 300 |
| imgsz | 640 |
| batch | 8 |
| optimizer | SGD |
| lr0 / lrf | 0.01 / 0.01 |
| momentum / weight_decay | 0.937 / 0.0005 |
| warmup_epochs | 3.0 |
| patience | 100 |
| seed | 0 |
| deterministic | true |
| close_mosaic | 0 |
| resume | false |
| Kaggle 请求时长 | `43200` 秒；服务端 12 小时上限优先 |

除 `model=yolo26m.pt` 外，该配置与 YOLO11m Fold 1 重试任务完全一致。该任务仅用于开发集 Fold 1 比较，完成后仍不可据此选择最终模型或报告冻结测试集指标。

## 4. 结果

Kaggle Kernel version 1 已完成。结果 ZIP 位于 `E:\PythonProject\models\kaggle_outputs\lly3323w__l1-ryw18-yolo26m-anchorv1-fold01\dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_anchorv1_fold01_results.zip`，SHA-256=`2e01f70a615e4a4a472b487e2c7e9ae06db86582b20e3206d14299703c8fab48`。CRC、路径安全、30 个归档文件、必需文件、权重和归档内 train/val/test 清单均已校验；清单与锚点固定 Fold 1/冻结 test 逐项一致，为 `751/376/198`。

| 口径 | checkpoint / epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Kaggle 训练期 | `best.pt` / 244 | 0.54941 | 0.29688 | 0.30990 | 0.15720 |
| Kaggle 训练期 | `last.pt` / 300 | 0.50531 | 0.26874 | 0.27614 | 0.14763 |
| 本机统一复评 | `best.pt` / Fold 1 val | 0.48451 | 0.29343 | 0.30540 | 0.15454 |

权重已归档至 `E:\PythonProject\models\runs\dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_anchorv1_fold01`；`best.pt` SHA-256=`C64CC0774CD3BA4E2EBEE832BA54C00203E918A665CEA1AA89B0AE7FEC1D36DC`，`last.pt` SHA-256=`3C5A49320956E59A2294786C60B28967A3331CC302E4AC4AA437870C76C02BD1`。本机复评固定为 Ultralytics 8.4.66、Torch 2.6.0+cu124、640、batch 4、FP32、NMS IoU=0.7、max_det=300，且只读取 376 张开发 val；逐类指标和派生硬链接视图见 `E:\PythonProject\优化\l1_reeval_yolo26m_fold01\metrics_per_class_reeval.json`。198 张冻结 test 未读取。

在同一套本机复评口径下，YOLO26m 的 mAP50/mAP50-95 比 YOLO11m Fold 1 的 `0.27848/0.14971` 高，但 Precision 较低（0.48451 vs 0.57977）。这只是一个开发 Fold 和一个随机种子，不能作为最终架构选择。

## 5. Kaggle 入口

- Kernel：`https://www.kaggle.com/code/lly3323w/l1-ryw18-yolo26m-anchorv1-fold01`
- 预检：`https://www.kaggle.com/code/lly3323w/l1-ryw18-anchor-v1-preflight`
- 已使用归档入口：`collect_l1_kaggle_run.py --kernel lly3323w/l1-ryw18-yolo26m-anchorv1-fold01 --run-name dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_anchorv1_fold01 --install`
