# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_anchor_v1_yolo26m_fold02_kaggle_seed0` |
| 日期 | 2026-08-01 |
| 执行人 | Codex（/root） |
| 运行环境 | Kaggle 私有 Script Kernel |
| Kernel | `lly3323w/l1-ryw18-yolo26m-anchorv1-fold02` |
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
| train 图片数 | 752 |
| val 图片数 | 375 |
| test 图片数 | 198，冻结且不参与训练、早停或调参 |
| 类别数与顺序 | 18 类 RYW18，见锚点 `classes.txt` |
| 是否清理重复图 | 是；固定锚点已完成精确重复排除与近重复分组 |
| 是否通过标签审计 | 是；本机 Fold 2 dry-run 通过，远端运行器会再次复核 |

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

除模型从 `yolo11m.pt` 切换为 `yolo26m.pt` 外，该配置与已归档的 YOLO11m Fold 2 一致。任务只用于三折开发验证；完成后仍不可访问冻结测试集或作出最终架构选择。

## 4. 结果

本机 `--dry-run` 已验证固定清单 `train/val/test=752/375/198`，三集合两两无交集、总计 1325 图；标签和 RYW18 类别顺序由训练入口重新校验。Kaggle version 1 的结果 ZIP 已通过 CRC、30 个归档文件、必需文件、权重 SHA-256 及归档内三份固定清单校验，并安装至 `E:\PythonProject\models\runs\dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_anchorv1_fold02`。ZIP SHA-256 为 `bd4e6038d39d5b989749fce24aca0f000f7ca5d76de4374847cf0e473548e283`。

训练期以 Ultralytics fitness（本版本仅采用 mAP50-95）最大的 `results.csv` 行选择 `best.pt`：epoch 58，P/R/mAP50/mAP50-95=`0.56126/0.19116/0.25637/0.16023`；最后记录为 epoch 158，mAP50-95=`0.12299`。权重 SHA-256：`best.pt=A08151DC9FDC68005E46E4FC8C2189A7316E1F2A4DA1D0BF036954E377593F51`，`last.pt=CB808A450199F7A66A7DAA5E4B929A9F8B6BBF0464CB75838B2181A7498240F8`。

使用同一 `best.pt`、本机 Ultralytics `8.4.66`、GTX 1660 Ti，对固定 Fold 2 的 375 张开发 val 重新评估得到 P/R/mAP50/mAP50-95=`0.56254/0.21363/0.20602/0.11342`。证据为 `E:\PythonProject\优化\l1_reeval_yolo26m_fold02\metrics_per_class_reeval.json`；复评清单与锚点 Fold 2 val 完全一致，198 张冻结 test 仅计数记录、未读取或评估。跨 Fold 和跨架构比较一律采用该本机复评口径；尚缺两个 Fold 3，禁止提前选择架构。

## 5. Kaggle 入口

- Kernel：`https://www.kaggle.com/code/lly3323w/l1-ryw18-yolo26m-anchorv1-fold02`
- 本地预检：`python .\train_dianjienie_ryw18_anchor.py --model yolo26m.pt --fold 2 --dry-run`
- 完成后归档：`collect_l1_kaggle_run.py --kernel lly3323w/l1-ryw18-yolo26m-anchorv1-fold02 --run-name dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_anchorv1_fold02 --install`
