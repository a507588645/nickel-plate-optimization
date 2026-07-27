## 2026-07-21 01:40 | 完成 RF-DETR Small 640 Kaggle 训练、归档与四模型报告

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex |
| 任务目标 | 在 Kaggle 训练官方 Roboflow RF-DETR Small，并与现有 YOLO11s、YOLO26s、项目 YOLOv5-s Small 640 结果比较 |
| 状态 | 已完成 |

### Kaggle 训练

- Kernel：`lly3323w/rf-detr-small-dianjinie-640-project-baseline`，version 2，状态 `COMPLETE`。
- 模型/版本：`RFDETRSmall`，`roboflow/rf-detr==1.8.3`；官方 tag `1.8.3` / commit `3bd6bffbcb13cac3a5b1c37da5a0fd5453b50c86` 已本地核对。
- 环境：Kaggle Tesla P100-PCIE-16GB、Python 3.12.13、Torch 2.6.0+cu124、RF-DETR 1.8.3。
- 配置：resolution=640、epochs=100、batch=4、grad_accum=4（有效 batch=16）、lr=1e-4、seed=0、EMA、GPU augmentation、多尺度、eval_interval=5、无早停；实际时长 7.42187 h。
- version 1 在 epoch 开始前被标签审计主动中止：验证标签目录混入 `classes.txt` 类别元数据。version 2 仅忽略该严格文件名且写入审计记录，未修改数据集原件；其他非法 YOLO 标签仍会中止。

### 数据与结果

- 数据审计：train=836 张/17,655 框，valid=216 张/4,337 框；无空标签、无非法 YOLO 行。valid 的 `nodule=0`，所有模型该类均为 `null/不可评`。
- 最终权重：RF-DETR 官方 `checkpoint_best_total.pth`，从 EMA/regular 最优权重按验证 mAP50-95 选择；`last.ckpt` 仅用于续训。
- RF-DETR 标准 COCO 重评：mAP50=`0.27502`、mAP50-95=`0.10587`、mAP75=`0.06532`。
- 单 seed Small 640 排名（mAP50-95）：RF-DETR Small `0.10587` > YOLO26s `0.09363` > YOLO11s `0.09110` > 项目 YOLOv5-s `0.06233`。
- RF-DETR 相对项目 YOLOv5-s 的 mAP50-95 增量 `0.04354`（相对 +69.9%）；相对 YOLO26s 增量 `0.01224`。
- 逐类 mAP50-95：YOLO26s 在 waterstain / waterblock 最高；RF-DETR 在 colorstain / bubble 最高，且整体 mAP50-95 最高。

### 产物与核验

- runs：`E:/PythonProject/models/runs/dianjinie_small_rfdetr_640_kaggle/rfdetr_small_dianjinie_640_project_baseline`；含最终/EMA/regular 权重、last.ckpt、metrics、完整训练配置、数据审计、环境、日志和 TensorBoard events。
- 最终权重 SHA-256：`2C0618E563D7254B7346011BF3F5E3BC7A211B6A0C3AB8D8F51C3C3A519F2A8C`。
- Kaggle 1,302,413,334 字节结果 ZIP 使用 78 个 verified Range 分块重组，ZIP `testzip()` 全文件 CRC 通过（15 项，bad=None）。此前两份不完整 `.corrupt.zip` 仅保留在 `models/kaggle_outputs/...` 追溯，未进入 runs。
- 对比数据：`E:/PythonProject/models/results/dianjinie_small_640_four_model_comparison.{csv,json}`。
- 正式实验记录：`E:/PythonProject/models/results/dianjinie_small_640_rfdetr_vs_yolo_实验记录.md`。
- 报告：`E:/PythonProject/models/results/电积镍Small640四模型训练结果对比.docx`、`.pdf`。
- Word 已用本机 Word 导出 6 页 PDF，并用 pypdfium2 渲染逐页视觉检查；修正了 P/R 表格换行及逐类领先结论后复检通过，无截断/溢出/错误类别表述。

### 更新文件

- `models/kaggle_rfdetr_kernel_dianjinie_small_640_project_baseline/{kernel-metadata.json,run_rfdetr_kaggle.py}`
- `models/tmp/download_kaggle_kernel_outputs.py`
- `models/tmp/build_dianjinie_small_640_four_model_report.py`
- `models/runs/dianjinie_small_rfdetr_640_kaggle/README.md`
- `models/runs/README.md`
- `models/results/dianjinie_small_640_rfdetr_vs_yolo_实验记录.md`
- `models/results/电积镍Small640四模型训练结果对比.{docx,pdf}`
- `models/results/dianjinie_small_640_four_model_comparison.{csv,json}`
- `CLAUDE.md`、`工作日志.md`

### 风险与下一步

- 只有 seed 0，且 RF-DETR 与 YOLO 的 optimizer、loss、EMA、有效 batch、epochs、多尺度策略不同；本条只说明现有方案效果，不能声称稳定架构优势或严格同预算胜出。
- 后续补 RF-DETR Small、YOLO11s、YOLO26s 的 seed 1、2；补 `nodule` 验证标签；进行同图人工审查和 ONNX 部署验证。
