## 2026-07-20 13:37 | 准备 RF-DETR Small 640 Kaggle 对比训练

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex |
| 任务目标 | 按用户要求在 YOLO11s、YOLO26s、项目 YOLOv5-s 之后增加 RF-DETR Small 640 Kaggle 实验，并形成四模型正式比较 |
| 状态 | 进行中 |

### 新增文件

- `E:/PythonProject/models/kaggle_rfdetr_kernel_dianjinie_small_640_project_baseline/kernel-metadata.json`
- `E:/PythonProject/models/kaggle_rfdetr_kernel_dianjinie_small_640_project_baseline/run_rfdetr_kaggle.py`

### 已核查配置与源码

- 官方仓库 tag `1.8.3` 对应 commit `3bd6bffbcb13cac3a5b1c37da5a0fd5453b50c86`；PyPI 最新可安装版本同为 `rfdetr==1.8.3`。
- 官方源码说明 `checkpoint_best_total.pth` 是在 regular/EMA 中按验证 mAP50-95 选出的最终推理与基准权重；新内核将明确使用该文件评估，不再依赖通配符顺序。
- 训练配置：`RFDETRSmall`、resolution=640、epochs=100、batch=4、grad_accum=4（有效 batch=16）、lr=1e-4、seed=0、eval_interval=5、EMA、GPU augmentation、无早停。
- 与现有 576 成功实验相比，核心训练变量只改变 resolution 576→640、seed 42→0；checkpoint 间隔改为 100 仅用于减少 Kaggle 输出体积，不改变优化过程。
- 新内核会输出 `requested_config.json`、`runtime_environment.json`、`dataset_audit.json`、`metrics*.json`、`run_summary.json`，并将验证集零真值类别写为 `null/不可评`。

### 验证与下一步

- `run_rfdetr_kaggle.py` 已通过 `py_compile`，metadata JSON 已解析并核对 kernel ID。
- 下一步提交 `lly3323w/rf-detr-small-dianjinie-640-project-baseline`，训练完成后下载并归档到 `E:/PythonProject/models/runs/dianjinie_small_rfdetr_640_kaggle`，再生成四模型报告。
