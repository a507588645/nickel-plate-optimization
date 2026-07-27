## 2026-07-23 16:45 | 按用户要求将 Medium YOLO 与 RF-DETR 纳入同一七模型对比

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex |
| 任务目标 | 统一比较所有已归档的 640 模型，不再按 Small/Medium 分开；将参数量约 32.1M 的 RF-DETR Small 与 Medium YOLO 直接对照 |
| 状态 | 已完成 |

### 改动文件

- `E:\PythonProject\models\tmp\reeval_ultralytics_best_per_class.py`：新增可复现的单权重标准逐类验证入口。
- `E:\PythonProject\models\runs\dianjinie_medium_yolo11m_640_kaggle_seed0_cm0_standard\metrics_per_class_reeval_ultralytics8490.json`：新增 YOLO11m FP32 逐类标准指标。
- `E:\PythonProject\models\runs\dianjinie_medium_yolo26m_640_kaggle_seed0_cm0_standard\metrics_per_class_reeval_ultralytics8490.json`：新增 YOLO26m FP32 逐类标准指标。
- `E:\PythonProject\models\tmp\build_dianjinie_640_seven_model_comparison_report.py`：新增七模型统一报告生成器。
- `E:\PythonProject\models\results\电积镍Small640四模型对比实验报告.docx`、`.pdf`：已删除，改由七模型统一报告替代。
- `E:\PythonProject\优化\CLAUDE.md`、`E:\PythonProject\models\runs\RUN_IDENTITY_CATALOG.md`、`E:\PythonProject\models\runs\dianjinie_small_yolov5s_640_projcode\README.md`、`E:\PythonProject\优化\镍板缺陷模型优化操作手册.md`：更新为七模型报告索引及 Medium 逐类指标位置。

### 产物路径

- `E:\PythonProject\models\results\电积镍640七模型统一对比实验报告.docx`
- `E:\PythonProject\models\results\电积镍640七模型统一对比实验报告.pdf`

### 关键命令或入口

- `E:\PythonProject\ImetricCodes\JinchuanYolov5sTrain\venv\Scripts\python.exe E:\PythonProject\models\tmp\reeval_ultralytics_best_per_class.py --weights <best.pt> --data E:\PythonProject\datasets\Dianjinie0211\data.yaml --output <metrics_per_class_reeval_ultralytics8490.json> --model-name <YOLO11m|YOLO26m> --batch 4 --device 0`

### 结论

- 七模型按 mAP50-95：RF-DETR Small=`0.10587`、YOLO11m=`0.10495`、YOLO26m=`0.10347`、YOLOv8m=`0.09686`、YOLO26s=`0.09363`、YOLO11s=`0.09110`、项目 YOLOv5-s=`0.06233`。
- RF-DETR Small（约32.1M）与 YOLO11m/YOLO26m/YOLOv8m（20.03M/20.35M/25.86M）处于更接近的参数量范围；统一报告没有按规模拆表。
- 逐缺陷最佳：waterstain 为 YOLO11m（mAP50/mAP50-95=`0.3442/0.1476`）；waterblock 为 YOLO26m（`0.4436/0.1734`）；colorstain 与 bubble 为 RF-DETR（`0.1445/0.0521`、`0.2279/0.0975`）。`nodule` 无 val 真值框，不可评。
- Medium 两项复验使用本机 GTX 1660 Ti、Ultralytics 8.4.90、FP32、640、NMS IoU=0.7、max_det=300；其余模型采用各自已归档标准结果。Word 经 Word COM 导出 PDF，并由 PDFium 渲染 6 页逐页检查，无截断或跨页拆表。

### 未完成项

- [ ] YOLOv8s ZIP 下载、完整性校验和归档完成后，补齐其逐缺陷 AP，再决定是否新增八模型版本。

### 下一步

1. 当前模型选择以七模型统一报告为效果参考；若需要稳定架构结论，先补 nodule 标签、固定无泄漏测试集并进行至少 3 个种子复验。

### 风险与待确认

- 风险：不同模型的训练框架、平台、有效 batch、epoch 和优化器不完全相同，单 seed 的数值不能证明严格同预算优势。
- 待确认：YOLOv8s 完成归档后是否纳入同一报告。
