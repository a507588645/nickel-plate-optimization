## 2026-07-22 16:00 | 启动 Kaggle 串行 YOLO11m/YOLO26m/YOLOv8m/YOLOv8s 640 对比

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex |
| 任务目标 | 按用户要求增加 YOLO11m、YOLO26m、YOLOv8m、YOLOv8s Kaggle 训练，并严格一次只运行一个以节省 GPU 配额 |
| 状态 | 进行中 |

### 已核对事实

- 历史 `dianjinie_medium_yolo11_640_kaggle` / `...yolo26...` 使用 seed=42、close_mosaic=10，与当前 Small 640 基线（seed=0、close_mosaic=0）不同；新运行必须独立建目录，不能覆盖历史 Medium 结果。
- 新队列统一配置：Dianjinie0211 5 类、imgsz=640、epochs=300、batch=8、patience=100、seed=0、SGD、lr0=0.01、mosaic=1.0、close_mosaic=0、NMS IoU=0.7、max_det=300。
- 验证集固定为 216 张/4337 框，`nodule=0`；所有后续报告必须写 `null/不可评`。

### 首个内核

- 新增 `E:/PythonProject/models/kaggle_yolo11m_dianjinie_640_project_baseline/kernel-metadata.json`。
- 新增 `E:/PythonProject/models/kaggle_yolo11m_dianjinie_640_project_baseline/run_yolo11m_kaggle.py`。
- Kernel ID：`lly3323w/yolo11m-dianjinie-640-project-baseline`。
- 输出目录计划：`E:/PythonProject/models/runs/dianjinie_medium_yolo11_640_kaggle_projectbaseline_seed0`。
- 脚本 `py_compile` 与 metadata JSON 已通过；目标 Kaggle kernel 尚不存在。

### 串行顺序与下一步

1. 提交并完成 YOLO11m，下载/校验/归档后再提交 YOLO26m。
2. YOLO26m 完成后再依次提交 YOLOv8m、YOLOv8s。
3. 所有四项完成后更新统一比较 CSV/JSON/Word/PDF 和 runs 索引。

### 风险

- 每项可能因早停而实际 epochs 少于 300；以 `best.pt` 标准评测和 `args.yaml` 为准。
- 四项均为单 seed，且 Medium/Small 模型规模不同；结果不能替代多 seed 的稳定性结论。
