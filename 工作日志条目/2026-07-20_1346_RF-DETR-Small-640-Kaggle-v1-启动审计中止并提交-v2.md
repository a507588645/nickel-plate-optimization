## 2026-07-20 13:46 | RF-DETR Small 640 Kaggle v1 启动审计中止并提交 v2

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex |
| 任务目标 | 处理 RF-DETR Small 640 kernel v1 启动失败并恢复训练 |
| 状态 | 进行中 |

### 结果与修复

- v1 在依赖安装、CUDA 初始化和数据挂载后由主动标签审计中止，未进入 epoch 训练。
- 根因：`Dianjinie0211/labels/val/classes.txt` 位于标签目录，内容是 5 行类别名称，不是 YOLO 图像标签；旧训练模板会静默忽略，新审计最初将其报为 5 行非法标签。
- 已修改 `run_rfdetr_kaggle.py`：仅忽略文件名严格为 `classes.txt` 的元数据，并写入 `dataset_audit.json: ignored_metadata_files`；其他非法标签仍会中止训练。
- 未修改或删除本地/Kaggle 数据集原件；本地核对确认 `labels/val/classes.txt` 为 52 字节，`data.yaml` 仍为 5 类。
- 修复后脚本再次通过 `py_compile`，Kaggle version 2 已提交并处于 `RUNNING`。

### Kaggle 运行环境（v1 已确认）

- Python 3.12.13、Torch 2.6.0+cu124、RF-DETR 1.8.3、Tesla P100-PCIE-16GB、CUDA capability 6.0。

### 下一步

- 监控 v2 是否通过数据审计并进入训练；完成后下载和核验所有产物。
