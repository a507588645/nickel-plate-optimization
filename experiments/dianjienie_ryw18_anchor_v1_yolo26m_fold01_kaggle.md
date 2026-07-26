# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_anchor_v1_yolo26m_fold01_kaggle_seed0` |
| 日期 | 2026-07-26 |
| 执行人 | Codex（/root） |
| 运行环境 | Kaggle 私有 Script Kernel |
| Kernel | `lly3323w/l1-ryw18-yolo26m-anchorv1-fold01` |
| GPU | 待远端环境记录回传 |
| Python | 待远端环境记录回传 |
| Torch | 目标 `2.6.0` + CUDA 12.4 |
| Ultralytics | 目标 `8.4.66` |
| 当前状态 | Kaggle version 1 已于 2026-07-26 提交；首次状态为 `RUNNING` |

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

当前 Kernel 已确认进入 `RUNNING`。完成后才填写远端 GPU/软件版本、`best.pt`、`last.pt`、`args.yaml`、`results.csv`、曲线、混淆矩阵和逐类指标。只有完整归档通过 CRC、路径安全和必需文件校验后，才把结果写入 `E:\PythonProject\models\runs`。

## 5. Kaggle 入口

- Kernel：`https://www.kaggle.com/code/lly3323w/l1-ryw18-yolo26m-anchorv1-fold01`
- 预检：`https://www.kaggle.com/code/lly3323w/l1-ryw18-anchor-v1-preflight`
- 完成后归档：`collect_l1_kaggle_run.py --kernel lly3323w/l1-ryw18-yolo26m-anchorv1-fold01 --run-name dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_anchorv1_fold01 --install`
