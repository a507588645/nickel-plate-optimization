## 2026-07-20 00:43 | 更新 dianjinie_medium_yolov5_640_projcode 文档与重评元数据

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex |
| 任务目标 | 按用户要求更新 `E:/PythonProject/models/runs/dianjinie_medium_yolov5_640_projcode` 文档，使模型身份、初始化链路和指标口径与源码/权重一致 |
| 状态 | 已完成 |

### 改动文件

- `E:/PythonProject/models/runs/dianjinie_medium_yolov5_640_projcode/README.md`
- `E:/PythonProject/models/runs/dianjinie_medium_yolov5_640_projcode/exp_dianjinie_yolov5m2/metrics_standard_iou.json`
- `E:/PythonProject/models/tmp/reeval_yolov5m_standard.py`
- `E:/PythonProject/models/AGENTS.md`
- `CLAUDE.md`
- `工作日志.md`

### 核验与修正

1. 目标 `best.pt` 与训练工程原产物 SHA-256 一致：`D25ADCB715B6A700A57D55A7857749F702E28A93CF28677E83E6302EF8CDFD15`。
2. README 明确：官方 `yolov5m.pt` 本身是真 m，但首次运行同时指定本地 `cfg=models/yolov5m.yaml`；`train.py` 优先使用非空 cfg，实际模型为 0.33/0.50、7,033,114 参数的 s 规模，仅迁移 57/349 项。
3. 目录名 `medium/yolov5m` 保留作历史标识；正式表格应称“YOLOv5 s 规模（项目代码）”，不能作为官方 YOLOv5m 基线。
4. 训练日志宽松指标仍为 mAP@0.1=0.32188、mAP@0.1:0.55=0.25949；标准 FP32 重评总体指标保持 mAP50=0.1808583592、mAP50-95=0.0623260134。
5. 验证集无 `nodule` 标注。旧 JSON 因 legacy `val.py` 对缺失类填总体均值而把该类 mAP50-95 写成 0.062326；现改为 `null` + `not_evaluable_no_validation_targets`，README 同步写“不可评”。
6. 三模型对比保留现有标准指标排名，但补充规模边界：该表是既有权重效果对比，不是三种 medium 模型的严格同规模对比。

### 验证

```text
E:/PythonProject/models/.venv/Scripts/python.exe -m json.tool .../metrics_standard_iou.json
E:/PythonProject/models/.venv/Scripts/python.exe -m py_compile E:/PythonProject/models/tmp/reeval_yolov5m_standard.py
```

JSON 解析通过，脚本编译通过，README 关键表述搜索通过。未重新运行模型验证，未修改权重、训练结果 CSV、图片或数据集。

### 风险与下一步

- README 中逐类 mAP50 来自标准重评 verbose 输出的历史记录；当前 JSON 只保存逐类 mAP50-95。若以后要求完全机器可复现的逐类双指标，应扩展重评脚本导出逐类 mAP50 并重新跑一次 FP32 验证。
- 真正的 YOLOv5m 公平基线仍需使用 0.67/0.75 架构新建独立实验，不得 resume 此 s 规模权重。
