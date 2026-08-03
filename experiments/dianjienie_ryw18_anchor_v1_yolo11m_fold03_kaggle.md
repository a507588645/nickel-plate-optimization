# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_anchor_v1_yolo11m_fold03_kaggle_seed0` |
| 日期 | 2026-08-02 |
| 执行人 | Codex（/root） |
| 运行环境 | Kaggle 私有 Script Kernel |
| Kernel | `lly3323w/l1-ryw18-yolo11m-anchorv1-fold03` |
| GPU | 待远端环境记录回传 |
| Python | 目标 3.12.13 |
| Torch | 目标 2.6.0+cu124，CUDA 12.4 |
| Ultralytics | 目标 8.4.66 |
| 当前状态 | 已完成 Kaggle 归档校验和固定开发集本机复评；冻结 test 未访问 |

## 2. 数据信息

| 项目 | 内容 |
| --- | --- |
| 数据集 | `dianjienie` RYW18 历史标签锚点 |
| 数据版本 | `dianjienie_ryw18_anchor_v1` |
| Kaggle 图片源 | `lly3323w/dianjienie`，只读图片仓库 |
| Kaggle 配置源 | `lly3323w/dianjienie-ryw18-anchor-v1-config` |
| train 图片数 | 751 |
| val 图片数 | 376 |
| test 图片数 | 198，冻结且不参与训练、早停或调参 |
| 类别数与顺序 | 18 类 RYW18，见锚点 `classes.txt` |
| 是否清理重复图 | 是；固定锚点已完成精确重复排除与近重复分组 |
| 是否通过标签审计 | 是；本机 Fold 3 dry-run 通过，远端运行器会再次复核 |

## 3. 训练配置

| 参数 | 值 |
| --- | --- |
| model | `yolo11m.pt` |
| pretrained weights | 官方 `yolo11m.pt`，不使用 `ryw.pt` 初始化 |
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

除固定分组开发 Fold 从 2 切换为 3 外，该配置与已归档 YOLO11m Fold 1/2 和并行的 YOLO26m Fold 3 一致。任务只用于三折开发验证；完成三个 Fold 前不得据此选择最终模型，冻结测试集不得访问。

## 4. 结果

本机 `--dry-run` 已验证固定清单 `train/val/test=751/376/198`，三集合两两无交集、总计 1325 图；标签和 RYW18 类别顺序由训练入口重新校验。Kaggle version 1 的结果 ZIP 已通过 CRC、30 个归档文件、必需文件、权重 SHA-256 及归档内三份固定清单校验，并安装至 `E:/PythonProject/models/runs/dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold03`。ZIP SHA-256 为 `5b2c0002b2e28695a517efc13eb725e5cffeef16f162145a007bc2dfb6cdf491`。

训练期以 Ultralytics fitness（本版本仅采用 mAP50-95）最大的 `results.csv` 行选择 `best.pt`：epoch 97，P/R/mAP50/mAP50-95=`0.56571/0.32812/0.32951/0.20305`；最后记录为 epoch 197，mAP50-95=`0.14355`。权重 SHA-256：`best.pt=D2BFF2119D0886440E2C7589A34E576081921C3E0ED6555E42810C022223E187`，`last.pt=1FC87AEB8868F486E0A08033752D6F1DE8FB5AD9ED74D7901C1A4046AEE386E0`。

使用同一 `best.pt`、本机 Ultralytics `8.4.66`、GTX 1660 Ti，对固定 Fold 3 的 376 张开发 val 重新评估得到 P/R/mAP50/mAP50-95=`0.56997/0.33965/0.33382/0.20488`。证据为 `E:/PythonProject/优化/l1_reeval_yolo11m_fold03/metrics_per_class_reeval.json`；复评清单与锚点 Fold 3 val 完全一致，198 张冻结 test 仅计数记录、未读取或评估。跨 Fold 和跨架构比较一律采用本机复评口径。

## 5. Kaggle 入口

- Kernel：`https://www.kaggle.com/code/lly3323w/l1-ryw18-yolo11m-anchorv1-fold03`
- 本地预检：`python .\train_dianjienie_ryw18_anchor.py --model yolo11m.pt --fold 3 --dry-run`
- 完成后归档：`collect_l1_kaggle_run.py --kernel lly3323w/l1-ryw18-yolo11m-anchorv1-fold03 --run-name dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold03 --install`
