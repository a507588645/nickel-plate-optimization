## 2026-07-20 13:25 | 完成 YOLO11s/YOLO26s Small 640 Kaggle 训练、runs 归档与 Word 报告

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex |
| 任务目标 | 在 Kaggle 用统一 640 项目基线配置训练 YOLO11s、YOLO26s，结果归档到 `models/runs`，并与项目实际 YOLOv5-s 形成正式报告 |
| 状态 | 已完成 |

### Kaggle 训练

- YOLO11s kernel：`lly3323w/yolo11s-dianjinie-640-project-baseline`，version 1，状态 `COMPLETE`。
- YOLO26s kernel：`lly3323w/yolo26s-dianjinie-640-project-baseline`，version 1，状态 `COMPLETE`。
- 统一配置：`Dianjinie0211` 5 类、imgsz=640、batch=8、epochs=300、patience=100、seed=0、SGD、mosaic=1.0、close_mosaic=0、NMS IoU=0.7、max_det=300。
- Kaggle 环境：Tesla P100-PCIE-16GB、Python 3.12.13、Torch 2.6.0+cu124、Ultralytics 8.4.66。

### 最终结果

| 排名 | 模型 | best epoch / 完成 epoch | Precision | Recall | mAP50 | mAP50-95 | 耗时 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | YOLO26s | 63 / 163 | 0.32950 | 0.28182 | 0.22970 | 0.09363 | 5.562 h |
| 2 | YOLO11s | 113 / 213 | 0.33068 | 0.29123 | 0.23344 | 0.09110 | 6.856 h |
| 3 | YOLOv5-s（项目代码） | 114 / 215 | 0.27548 | 0.21990 | 0.18086 | 0.06233 | 约 4.004 h |

- 按 mAP50-95：YOLO26s > YOLO11s > 项目 YOLOv5-s。
- YOLO26s 相比项目 YOLOv5-s 提高 0.03130（相对 +50.2%）；YOLO11s 提高 0.02877（相对 +46.2%）。
- YOLO26s 与 YOLO11s 仅差 0.00253，只有 seed 0，不能宣称稳定架构优势。
- 验证集 `nodule` 框数为 0，三模型该类均不可评。

### runs 产物

- `E:/PythonProject/models/runs/dianjinie_small_yolo11_640_kaggle`
- `E:/PythonProject/models/runs/dianjinie_small_yolo26_640_kaggle`
- 两个目录均包含：`README.md`、`metrics.json`、`args.yaml`、`requested_config.json`、`results.csv`、图表、`kaggle.log`、kernel metadata/脚本、`weights/best.pt` 与 `weights/last.pt`。
- 原始 Kaggle 下载保留在 `E:/PythonProject/models/kaggle_outputs/yolo11s-dianjinie-640-project-baseline` 和 `.../yolo26s-dianjinie-640-project-baseline`。

### 报告与记录

- Word：`E:/PythonProject/models/results/电积镍Small640三模型训练结果对比.docx`
- CSV/JSON：`E:/PythonProject/models/results/dianjinie_small_640_three_model_comparison.{csv,json}`
- 实验记录：`E:/PythonProject/models/results/dianjinie_small_640_yolo11s_yolo26s_vs_projcode_yolov5s_实验记录.md`
- 生成器：`E:/PythonProject/models/tmp/build_dianjinie_small_640_report.py`
- 已更新：两个 run README、`models/runs/README.md`、`models/runs/CLAUDE.md`、本目录 `CLAUDE.md`。

### 下载修复

- Kaggle CLI 在 Windows 遇到 GBK Unicode 日志错误和网络 `IncompleteRead`；关键输出最终通过断点续传恢复。
- 修复 `models/tmp/download_kaggle_kernel_outputs.py`：Range 请求返回 416 时识别已完整文件或重置错误部分，避免卡在 requested config，YOLO11s 结果 zip 最终恢复为 41,408,721 字节。

### 验证

- YOLO11s/YOLO26s 的 `best.pt`、`last.pt` 均由 Ultralytics 成功加载，参数量分别为 9,429,727 和 9,951,734。
- 两个 `metrics.json` 与比较 JSON 均通过解析；runs 必需文件逐项存在性检查通过。
- Word 生成器使用 bundled Python 编译执行通过；`render_docx.py` 因本机无 LibreOffice 失败，改用本机 Word COM 导出 PDF，并用 `pypdfium2` 渲染为 6 页 PNG逐页检查。
- Word 6 页无截断、重叠、缺失表头或空白尾页；两张图视觉检查通过；9 张表的 `tblW=9360`、`tblInd=120`、固定布局和列宽总和均通过结构审计。

### 风险与下一步

- 本轮只有 seed 0；下一轮补 seed 1、2 后报告均值和标准差。
- 项目 YOLOv5-s 与 Ultralytics 的 loss/检测头不同，且项目权重只从 `yolov5m.pt` 迁移 57/349 项；本结论是现有方案/权重效果比较，不是只改变架构的严格实验。
- 先补 nodule 验证标注、审计 colorstain 与极小框，再开展同图人工误检/漏检比较和 ONNX 部署测试。
