# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_anchor_v1_yolo11m_fold02_kaggle_seed0` |
| 日期 | 2026-07-27 提交，2026-07-28 归档与复评 |
| 执行人 | Codex（/root） |
| 运行环境 | Kaggle 私有 Script Kernel |
| Kernel | `lly3323w/l1-ryw18-yolo11m-anchorv1-fold02` |
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
| train 图片数 | 752 |
| val 图片数 | 375 |
| test 图片数 | 198，冻结且未参与训练、早停、调参或本机复评 |
| 类别数与顺序 | 18 类 RYW18，见锚点 `classes.txt` |
| 是否清理重复图 | 是；固定锚点已完成精确重复排除与近重复分组 |
| 是否通过标签审计 | 是；本机 dry-run 与 Kaggle 运行器复核均通过 |

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

除固定分组开发 Fold 从 1 切换为 2 外，该配置与已归档的 YOLO11m/YOLO26m Fold 1 基线一致。该任务只用于三折开发验证；完成三个 Fold 前不得据此选择最终模型，冻结测试集不得访问。

## 4. 结果

Kaggle Kernel version 1 已完成。结果 ZIP 位于 `E:\PythonProject\models\kaggle_outputs\lly3323w__l1-ryw18-yolo11m-anchorv1-fold02\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold02_results.zip`，SHA-256=`a0a8ea9793ebc06b1a2cd95789e397fb393d60117dfba6192cb889d5c11eccfb`。CRC、路径安全、30 个归档文件、必需文件、权重和归档内 train/val/test 清单均已校验；清单与锚点固定 Fold 2/冻结 test 逐项一致，为 `752/375/198`。

| 口径 | checkpoint / epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Kaggle 训练期 | `best.pt` 对应的最高 fitness 记录 / 146 | 0.36560 | 0.36931 | 0.29995 | 0.18877 |
| Kaggle 训练期 | `last.pt` / 最后记录 epoch 246 | 0.31911 | 0.26493 | 0.21850 | 0.11915 |
| 本机统一复评 | `best.pt` / Fold 2 val | 0.28965 | 0.37252 | 0.30358 | 0.19036 |

权重已归档至 `E:\PythonProject\models\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold02`；`best.pt` SHA-256=`9FB90C95E7016AAB020CBB6B1268E28CD29CDFA9606B4F087E999B52DF826CF0`，`last.pt` SHA-256=`CC6DF4DD988F1F64C186B5A44C1E64D587D8F0FE0F0F31D9D0418FC08DA00E23`。本机复评固定为 Ultralytics 8.4.66、Torch 2.6.0+cu124、640、batch 4、FP32、NMS IoU=0.7、max_det=300，只读取 375 张开发 val；逐类指标和派生硬链接视图见 `E:\PythonProject\优化\l1_reeval_yolo11m_fold02\metrics_per_class_reeval.json`。派生视图的图像、标签与锚点 Fold 2 val 清单均为 375，文件名逐项一致；198 张冻结 test 未读取。

用户已确认 Kaggle GPU 配额耗尽。因此本次未提交 YOLO26m Fold 2 或任何 Fold 3 任务。当前只有 YOLO11m 的两个开发 Fold 和 YOLO26m 的一个开发 Fold，不能选择最终架构，也不得把不同 Fold 的结果直接排名。

## 5. Kaggle 入口

- Kernel：`https://www.kaggle.com/code/lly3323w/l1-ryw18-yolo11m-anchorv1-fold02`
- 本地预检：`python .\train_dianjienie_ryw18_anchor.py --model yolo11m.pt --fold 2 --dry-run`
- 已使用归档入口：`collect_l1_kaggle_run.py --kernel lly3323w/l1-ryw18-yolo11m-anchorv1-fold02 --run-name dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold02 --install`
