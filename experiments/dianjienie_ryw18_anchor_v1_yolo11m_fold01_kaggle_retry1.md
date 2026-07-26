# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_anchor_v1_yolo11m_fold01_kaggle_seed0_retry1` |
| 日期 | 2026-07-26 |
| 执行人 | Codex（/root） |
| 运行环境 | Kaggle 私有 Script Kernel |
| Kernel | `lly3323w/l1-ryw18-yolo11m-anchorv1-fold01-retry1` |
| GPU | Tesla P100-PCIE-16GB |
| Python | `3.12.13` |
| Torch | `2.6.0+cu124`，CUDA `12.4` |
| Ultralytics | `8.4.66` |
| 与原尝试关系 | 原任务在 epoch 250 后取消且无可用 checkpoint；本任务是独立新 run，不续训 |

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

## 4. 结果与证据

Kaggle 状态为 `COMPLETE`。结果 ZIP `dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold01_results.zip` 已下载、CRC 校验并完成必需文件与路径安全校验：共 30 个归档文件，SHA-256 为 `a6b4e4cf656cb057276af879a825eae7562013d93ee8c09666ea5575c6c918e7`。已安装至 `E:\PythonProject\models\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold01`，其中 `best.pt` SHA-256 为 `66A2CFAF90716D7770F3F3A80C2150A91F725120E9F26DA7AB13248526EB6D5E`。

训练期开发验证共记录 277 个 epoch。`best.pt` 按 Ultralytics fitness 选择 epoch 177，Precision=`0.54481`、Recall=`0.32513`、mAP50=`0.29306`、mAP50-95=`0.15646`。最后一行 epoch 277 的指标为 Precision=`0.52872`、Recall=`0.27825`、mAP50=`0.26412`、mAP50-95=`0.14432`，不得误作 `best.pt` 指标。

为统一后续权重比较，已在本机对归档 `best.pt` 做一次独立固定 Fold 1 开发验证：376 张 val 图，Ultralytics `8.4.66`、Torch `2.6.0+cu124`、640、batch 4、FP32、NMS IoU=`0.7`、max_det=`300`。结果为 Precision=`0.57977`、Recall=`0.29190`、mAP50=`0.27848`、mAP50-95=`0.14971`；逐类指标与审计清单位于 `E:\PythonProject\优化\l1_reeval_yolo11m_fold01_retry2\metrics_per_class_reeval.json`。本机复评与 Kaggle 训练期验证的数值不同，后续 YOLO11m/YOLO26m 比较必须复用同一本机复评入口，不可混用两种口径。

归档 `train.txt`、`val.txt`、`test.txt` 已分别与锚点 Fold 1 / 冻结测试清单逐项一致（751/376/198）。本实验未读取或评估 198 张冻结测试图，当前仅是单个开发 Fold，不构成模型选择结论。

## 5. Kaggle 入口

- Kernel：`https://www.kaggle.com/code/lly3323w/l1-ryw18-yolo11m-anchorv1-fold01-retry1`
- 预检：`https://www.kaggle.com/code/lly3323w/l1-ryw18-anchor-v1-preflight`
- 完成后归档：`collect_l1_kaggle_run.py --kernel lly3323w/l1-ryw18-yolo11m-anchorv1-fold01-retry1 --run-name dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold01 --install`
