# dianji_grade_v2

> ⚠️ **未经企业确认（unconfirmed_pending_enterprise）**
>
> 本目录全部模型、规则与准确率均为实验对照，**不得**作为企业正式判级标准或上线依据。

> ⚠️ **小样本警告**：`3#`=46（4types 34 + 0309 的 12）、`bad`=28。指标方差大，禁止过拟合后宣称可用。

## 产物清单

| 文件/目录 | 说明 |
| --- | --- |
| `merged_manifest.csv` | 合并去重清单（1072；含官方 70/15/15 split） |
| `features_v2a.csv` | 历史 samples.csv join（仅 156/1072，对照用） |
| `features_v2b_imgsz1536_conf0p25.csv` | 主特征：ryw.pt imgsz=1536 conf≥0.25 |
| `features_v2b_imgsz1536_conf0p40.csv` | 敏感性：conf≥0.40 |
| `detections_1536/` | 逐框输出 `all_boxes.csv` + per_image |
| `model_comparison.csv` | 四模型 × 特征 × valid/test |
| `metrics_*.json` | 机器可读指标 |
| `disagreements.csv` | 主推特征 test 分歧（与 cm off-diag 一致） |
| `report.md` | 完整分析报告 |
| `view_dataset_summary.json` | 硬链接视图划分摘要副本 |

## 硬链接视图

- `E:\PythonProject\datasets\dianjinie_grade_classification_v1`（train/valid/test=750/161/161，seed=42，按 dup_group）

## 主结论（test，v2b conf0.25，未经确认）

- LightGBM acc≈0.640，macro-F1≈0.413
- 多数类 0.509 / 决策树 0.478 / DianjieGradeRule≈0.025（仅映射 2号/3号板）
- 主要混淆：9950↔9996

## 复现

见 `report.md` 第 8 节。入口脚本：

```text
build_dianjinie_grade_view.py
extract_dianji_grade_features.py   # 或 build_dianji_grade_v2_dataset.py
train_dianji_grade_lgbm.py
```

`dianji_grade_rule_v1/` 未修改。
