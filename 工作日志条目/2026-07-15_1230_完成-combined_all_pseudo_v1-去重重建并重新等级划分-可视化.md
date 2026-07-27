## 2026-07-15 12:30 | 完成 combined_all_pseudo_v1 去重重建并重新等级划分/可视化

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi Code CLI |
| 任务目标 | 按用户要求对汇总数据集去重、重新划分、重新生成等级划分与可视化，删除旧的等级产物 |
| 状态 | 已完成 |
| 关联用户请求 | 需要，并且重新进行伪标签以及等级划分，之前的删除了 |

### 改动文件

- 新增：`E:/PythonProject/优化/deduplicate_and_rebuild_combined.py`
- 修改：`E:/PythonProject/优化/resume_dianjie_grade.py`（增加命令行参数）
- 修改：`E:/PythonProject/优化/build_lightgbm_grade_dataset.py`（增加命令行参数）
- 删除：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/` 下的旧 `samples.csv`、`summary.json`、`lightgbm_grade_dataset.csv`、`lightgbm_grade_summary.json`、`visualizations/`

### 产物路径

- 去重数据集：`E:/PythonProject/datasets/combined_all_pseudo_v1_dedup_v2/`
- 等级特征 CSV：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/samples.csv`
- 等级摘要：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/summary.json`
- LightGBM 数据集：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/lightgbm_grade_dataset.csv`
- 可视化目录：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/visualizations/`
- 唯一性审计仍保留：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/uniqueness_audit/`

### 关键命令

```text
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/deduplicate_and_rebuild_combined.py
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/apply_dianjie_grade_to_combined_sequential.py --combined-root E:/PythonProject/datasets/combined_all_pseudo_v1_dedup_v2 --output-dir E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade --conf-thres 0.70
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/resume_dianjie_grade.py --combined-root E:/PythonProject/datasets/combined_all_pseudo_v1_dedup_v2 --output-dir E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade --conf-thres 0.70
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/build_lightgbm_grade_dataset.py --combined-root E:/PythonProject/datasets/combined_all_pseudo_v1_dedup_v2
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/visualize_dianjie_grade_features.py
```

### 去重结果

- 原数据集：`E:/PythonProject/datasets/combined_all_pseudo_v1`，7396 张
- 去重后：`E:/PythonProject/datasets/combined_all_pseudo_v1_dedup_v2`，3264 张
- 删除重复：4132 张
- 划分：train 2285 / val 490 / test 489
- 按物理板 group 整体划分，同一 group 不跨 split

### 重新等级划分结果（detection_only，conf≥0.70）

- 总样本：3264
- 合格整版：2813（86.18%）
- 合格600：433（13.27%）
- 3号板：7（0.21%）
- 2号板：8（0.25%）
- 烫洗板：3（0.09%）

### LightGBM 可行性

- 类别仍严重不平衡，合格整版占 86.2%。
- 稀有类别（3号板/2号板/烫洗板）样本极少。
- 建议先尝试二分类（合格整版 vs 非合格整版），或使用类别权重/SMOTE。

### 伪标签说明

- 去重数据集直接复用了原 `combined_all_pseudo_v1` 的 `labels_with_conf/` 伪标签（来自 `ryw.pt`，conf≥0.70），未重新跑检测模型。
- 如需为去重数据集重新生成伪标签审查页 `review.html`，可再执行一次 review 生成脚本。

### 下一步

- 用户决定：是否训练 LightGBM，或重新生成伪标签审查页，或进一步调整类别/阈值。
