# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_anchor_v1_yolo11m_fold02_kaggle_seed0` |
| 日期 | 2026-07-27 |
| 执行人 | Codex（/root） |
| 运行环境 | Kaggle 私有 Script Kernel |
| Kernel | `lly3323w/l1-ryw18-yolo11m-anchorv1-fold02` |
| GPU | 待远端环境记录回传 |
| Python | 目标 3.12.13 |
| Torch | 目标 2.6.0+cu124，CUDA 12.4 |
| Ultralytics | 目标 8.4.66 |
| 当前状态 | Kaggle version 1 已于 2026-07-27 13:55 成功提交并确认 `RUNNING` |

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
| 是否通过标签审计 | 本机 Fold 2 dry-run 通过；远端恢复后仍须由运行器复核 |

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

除固定分组开发 Fold 从 1 切换为 2 外，该配置与已归档的 YOLO11m/YOLO26m Fold 1 基线一致。任务只用于三折开发验证，完成后仍不可访问冻结测试集或作出最终架构选择。

## 4. 结果

本地 `--dry-run` 已验证固定清单 `train/val/test=752/375/198`，三集合两两无交集、总计 1325 图。Kaggle CLI 已返回 version 1 推送成功，随后状态查询为 `RUNNING`。提交、远端完成、ZIP CRC/权重/固定清单校验和同口径本机复评后再填写最终指标。结果必须归档到 `E:\PythonProject\models\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold02`；不得覆盖 Fold 1 run。

## 5. Kaggle 入口

- Kernel：`https://www.kaggle.com/code/lly3323w/l1-ryw18-yolo11m-anchorv1-fold02`
- 本地预检：`python .\train_dianjienie_ryw18_anchor.py --model yolo11m.pt --fold 2 --dry-run`
- 完成后归档：`collect_l1_kaggle_run.py --kernel lly3323w/l1-ryw18-yolo11m-anchorv1-fold02 --run-name dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold02 --install`
