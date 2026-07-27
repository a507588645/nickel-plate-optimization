## 2026-07-19 23:47 | 核验 runs 内 Dianjinie0211 YOLOv5 运行产物的文档性

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex |
| 任务目标 | 判断 `runs/train/exp_dianjinie_yolov5m*` 下配置、结果文件是否可正确描述新训练 |
| 状态 | 已完成 |

### 结论

1. `runs` 下不存在独立 Markdown 实验报告；可用记录为 `opt.yaml`、`hyp.yaml`、`results.csv`、TensorBoard event、图和权重。
2. `exp_dianjinie_yolov5m/opt.yaml` 是正确的运行参数记录：初始权重为 `yolov5m.pt`、`cfg=models/yolov5m.yaml`、数据为 `Dianjinie0211`、未 resume。结合 `train.py` 的 YAML 优先级，它不支持“未改官方 m 架构”的解读。
3. `exp_dianjinie_yolov5m2/opt.yaml` 正确记录了续训状态（`resume: true`、`cfg: ''`），但它不能单独说明最初架构；必须追溯首次运行和最终 `best.pt` 内嵌 YAML。
4. `hyp.yaml` 是本次使用的 scratch-low 参数记录。`results.csv` 的数值与曲线/权重选择链路一致，但列名 `metrics/mAP_0.5`、`metrics/mAP_0.5:0.95` 沿用了项目 `val.py` 的错误标签，实际是宽松 IoU `0.1` 与 `0.1:0.55`，不能当标准 mAP。
5. 因此：**运行产物本身可用，但目录名 `yolov5m`、结果列名和缺少正式实验报告会误导。**后续应把该实验称为“Dianjinie0211、本地 s 规模、由 yolov5m.pt 部分迁移初始化、宽松验证口径”。

### 改动文件

- `工作日志.md`：本条目。未修改训练源码、模型权重、数据集或 `runs` 产物。

### 下一步

- 新建一次真正 m 规模、标准 IoU 的独立实验，并保存包含模型参数量、架构 YAML 哈希和标准逐类指标的正式实验记录。
