# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_anchor_v1_yolo11m_fold01_kaggle_seed0` |
| 日期 | 2026-07-25 |
| 执行人 | Codex（/root） |
| 运行环境 | Kaggle 私有 Script Kernel |
| GPU | 运行中，待任务环境记录回传 |
| Python | 待任务环境记录回传 |
| Torch | 目标 `2.6.0` + CUDA 12.4 |
| Ultralytics | 目标 `8.4.66` |

## 2. 数据信息

| 项目 | 内容 |
| --- | --- |
| 数据集 | `dianjienie` RYW18 历史标签锚点 |
| 数据版本 | `dianjienie_ryw18_anchor_v1` |
| Kaggle 图片源 | `lly3323w/dianjienie`，只读图片仓库 |
| Kaggle 配置源 | `lly3323w/dianjienie-ryw18-anchor-v1-config` |
| data.yaml | 运行时生成于 `/kaggle/working/training_runtime_configs/` |
| train 图片数 | 751 |
| val 图片数 | 376 |
| test 图片数 | 198，冻结且未参与训练/早停/调参 |
| 类别数与顺序 | 18 类 RYW18，见锚点 `classes.txt` |
| 是否清理重复图 | 是；固定锚点已完成精确重复排除与近重复分组 |
| 是否通过标签审计 | 是；Kaggle 预检已通过 |
| 数据变更说明 | 旧 Kaggle `dianjienie` 的标签与旧 `train/valid` 划分不使用；仅复用已核验图像字节 |

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

## 4. 结果

**状态：无效的中断 run，不得用于模型比较、选择或测试集评估。**

- Kaggle 最终状态为 `CANCEL_ACKNOWLEDGED`，无失败信息。该任务最初请求的上限为 `32400` 秒；日志在约 `32413` 秒进入第 251 个 epoch 后中断，与该上限相符，但 Kaggle 未提供可归因的失败消息。
- `results.csv` 仅含完成的 epoch 1--250。过程记录中的最高开发验证 `mAP50=0.29743`（epoch 222），最高 `mAP50-95=0.15646`（epoch 177）；二者均为中断前的过程观测，不能代替有效 `best.pt` 或正式实验结果。
- Kaggle 残留输出中 `weights/best.pt` 为 0 字节，`weights/last.pt`、`environment.json`、`README.md`、`dataset_config/` 和结果 ZIP 均缺失，故无法续训、归档或安装到 `E:\PythonProject\models\runs`。
- 已将远端残留输出和原始日志保存在 `E:\PythonProject\优化\l1_kaggle_chain_monitor\failed_run_probe`，未写入 `models/runs`。

将以完全相同的 L1 基线参数重新启动独立 `retry1` Kernel，并把请求时长提高到 Kaggle 的 12 小时上限；不使用 `resume=True`，不使用中断 run 的零字节权重。

## 5. Kaggle 入口

- Kernel：`https://www.kaggle.com/code/lly3323w/l1-ryw18-yolo11m-anchorv1-fold01`
- 预检：`https://www.kaggle.com/code/lly3323w/l1-ryw18-anchor-v1-preflight`
- 预检结论：远端自动定位图片与配置 payload；恢复 1325 图/1325 标签，Fold 1 为 751/376/198。

## 6. 中断证据

| 文件 | 路径 |
| --- | --- |
| Kaggle 原始流日志 | `E:\PythonProject\优化\l1_kaggle_chain_monitor\failed_run_probe\kaggle_live_log.json` |
| 远端残留输出 | `E:\PythonProject\优化\l1_kaggle_chain_monitor\failed_run_probe\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold01` |
| 后续重试实验记录 | `E:\PythonProject\优化\experiments\dianjienie_ryw18_anchor_v1_yolo11m_fold01_kaggle_retry1.md` |
