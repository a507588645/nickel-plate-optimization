# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo11m_hr1280_fold01_kaggle_seed0` |
| 所属主线 | L1 |
| 日期 | 2026-08-16 |
| 执行人 | Codex |
| 实验状态 | completed_no_go |
| 证据等级 | 单 Fold 完成；不通过扩展闸门 |
| 父计划 / 决策闸道 | G4 L1-hr Fold 1 |
| 研究假设 | 将输入尺寸从 640 提升到 1280 可改善可评小目标的检测指标。 |
| 只改变的主要变量 | `imgsz=640 -> 1280`；物理 batch 由 8 调整为 4 以适应 Kaggle P100 显存。 |
| 停止条件 | `mAP50-95` 绝对提升至少 0.02，且关键小目标类 AP/Recall 同向改善，才考虑 Fold 2/3。 |
| 运行环境 | Kaggle 私有 Script Kernel |
| GPU | Tesla P100-PCIE-16GB |
| Python / Torch / Ultralytics | Python 3.12.13 / Torch 2.6.0+cu124 / Ultralytics 8.4.66 |

## 2. 数据信息

| 项目 | 内容 |
| --- | --- |
| 数据集 | `dianjienie` RYW18 OOF 自动共识候选开发集 |
| 数据版本 | `oofconsensusv1` / `dianjienie_ryw18_oofconsensus_v1_dev_only_r1` |
| Kaggle 图片源 | `lly3323w/dianjienie`，只读图片仓库 |
| Kaggle 配置源 | `lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config` |
| train / val 图片数 | 751 / 376 |
| test 图片数 | 不包含、不读取、不评估 |
| 类别数与顺序 | 18 类 RYW18，顺序由配置包 manifest 和运行器校验 |
| 是否清理重复图 | 是，继承锚点的分组去重和固定 Fold |
| 是否通过标签审计 | 是，预检已完成；五列训练标签共 15,671 框，仍为 `auto_only` 候选标签 |
| 冻结测试集访问状态 | 不读取 |
| 数据变更说明 | 与 640 OOF Fold 1 使用同一开发集和同一 Fold，只改变高分辨率研究配置。 |

## 3. 训练配置

| 参数 | 值 |
| --- | --- |
| model / pretrained weights | `yolo11m.pt` / 官方预训练，不使用 `ryw.pt` 初始化 |
| epochs / imgsz / batch | 300 / 1280 / 4 |
| optimizer | SGD |
| lr0 / lrf | 0.01 / 0.01 |
| momentum / weight_decay | 0.937 / 0.0005 |
| warmup_epochs / patience | 3.0 / 100 |
| seed / deterministic | 0 / true |
| close_mosaic / resume | 0 / false |
| 资源约束或中断原因 | 未 OOM、无中断证据；实际记录 183 个 epoch，最佳 epoch 为 83，符合 `patience=100` 的正常早停。 |

## 4. 基线与结果

主比较使用两次 Kaggle 训练期的最佳验证指标：两者的 `train.txt` 与 `val.txt` 哈希完全相同，均为 751/376 张、无交集、共 1,127 张开发图；均无 `test.txt`。两次运行环境相同，训练配置除运行路径外只存在 `imgsz=640 -> 1280` 和显存导致的物理 `batch=8 -> 4` 两项差异。

| 指标 | 1280 Fold 1 最佳（epoch 83） | 640 Fold 1 最佳（epoch 190） | 1280 - 640 |
| --- | ---: | ---: | ---: |
| Precision | 0.58596 | 0.51399 | +0.07197 |
| Recall | 0.30003 | 0.29538 | +0.00465 |
| mAP50 | 0.30224 | 0.30617 | -0.00393 |
| mAP50-95 | 0.16264 | 0.16382 | -0.00118 |

`mAP50-95` 未达到预先规定的绝对提升 `>=0.02`，反而低 0.00118；因此无需再为了逐类 AP 额外消耗 GPU，直接判定本研究不扩展到 Fold 2/3。已有本机 640 统一复评基线 P/R/mAP50/mAP50-95=`0.49024/0.30787/0.31292/0.16513` 仅作历史参照，不能与本次未做统一复评的 1280 数字直接排名。

该结论只说明当前 OOF 自动候选训练视图上，1280 整图输入不值得继续投入；物理 batch 变化使其不是严格单变量消融，不用于否定 YOLO11m、OOF 标签路线或最终 640 候选模型。未重新访问历史 release-validation，也不构成最终模型选型。

## 5. 文件位置

| 文件 | 路径 |
| --- | --- |
| Kaggle 预检 | `lly3323w/l1-ryw18-yolo11m-oof-consensus-v1-hr1280-b4-fold01`，version 1 `COMPLETE` |
| Kaggle GPU 任务 | `https://www.kaggle.com/code/lly3323w/l1-ryw18-yolo11m-oof-v1-hr1280-b4-e300-fold01`，version 1 `COMPLETE` |
| Kernel 定义 | `E:/PythonProject/优化/kaggle_l1_ryw18_yolo11m_oof_v1_hr1280_b4_e300_f01/` |
| 生成器 | `E:/PythonProject/优化/prepare_l1_hr_kaggle_kernel.py` |
| 已归档 run | `E:/PythonProject/models/runs/dianjienie_medium_yolo11m_1280_kaggle_seed0_cm0_oofconsensusv1_fold01/` |
| 结果 ZIP | `E:/PythonProject/models/kaggle_outputs/lly3323w__l1-ryw18-yolo11m-oof-v1-hr1280-b4-e300-fold01/dianjienie_medium_yolo11m_1280_kaggle_seed0_cm0_oofconsensusv1_fold01_results.zip`，SHA-256=`711054844c577edead03b44c6c031564ffd486c5e0e5f9f17a5dc5d0c7b304e1` |
| 固定权重 | `weights/best.pt`，SHA-256=`5d68707e2148c5d433658dfe21dfbf35b72d0b22e5762a0b5f9844d1de9870ee` |
| 归档入口 | `collect_l1_kaggle_run.py --development-only --kernel lly3323w/l1-ryw18-yolo11m-oof-v1-hr1280-b4-e300-fold01 --run-name dianjienie_medium_yolo11m_1280_kaggle_seed0_cm0_oofconsensusv1_fold01 --install` |
