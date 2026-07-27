## 2026-07-15 20:45 | 重建 combined_all_pseudo_v1 审查页、等级划分与可视化

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi Code CLI |
| 任务目标 | 按用户要求：1) 删除旧 dedup_v2 产物；2) 保留原名 combined_all_pseudo_v1；3) 重新生成伪标签审查页、等级划分、LightGBM 数据集与可视化；4) 说明唯一性审计 |
| 状态 | 已完成 |
| 关联用户请求 | 删除旧汇总数据及产物；重新伪标签/等级划分/可视化；解释唯一性审计 |

### 关键发现

- `combined_all_pseudo_v1/images_all` 原为一堆断链软链接，导致审查页生成 0 张图、等级划分 cv2 报错。
- 已新建 `rebuild_images_all.py` 按原 `build_combined_pseudo_dataset.py` 的扁平命名规则，从源数据集硬链接/复制重建 `images_all`，3264/3264 张全部恢复。
- 原项目硬编码规则 `dianjie_grade_rule.py` 的 `classify()` 方法**从未返回 type 5（合格不足600）**：默认 4（合格600），仅在有条件时改为 3/2/1/6。因此该等级数量为 0 是规则逻辑所致，不是数据缺失。
- 无板（type 0）为 0 是因为汇总数据集本身不含“无板”样本。

### 改动文件

- 新增：`E:/PythonProject/优化/rebuild_images_all.py`
- 删除：`E:/PythonProject/优化/pseudo_label_reviews/combined_all_pseudo_v1_dedup_v2/`
- 删除：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/` 下除 `uniqueness_audit/` 外的旧产物
- 重建：`combined_all_pseudo_v1/images_all/`

### 产物路径

- 伪标签审查页：`E:/PythonProject/优化/pseudo_label_reviews/combined_all_pseudo_v1/review.html`（3264 张图，74331 个框）
- 等级特征 CSV：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/samples.csv`
- 等级汇总 JSON：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/summary.json`
- LightGBM 数据集：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/lightgbm_grade_dataset.csv`
- 可视化图表：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/visualizations/`
- 唯一性审计（保留）：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/uniqueness_audit/`

### 关键命令

```text
# 重建 images_all
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/rebuild_images_all.py

# 生成伪标签审查页
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/generate_flat_yolo_review.py E:/PythonProject/datasets/combined_all_pseudo_v1 --output-root E:/PythonProject/优化/pseudo_label_reviews --absolute

# 等级划分（先顺序跑，超时后续跑）
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/apply_dianjie_grade_to_combined_sequential.py --combined-root E:/PythonProject/datasets/combined_all_pseudo_v1 --output-dir E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/resume_dianjie_grade.py --combined-root E:/PythonProject/datasets/combined_all_pseudo_v1 --output-dir E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade

# LightGBM 数据集
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/build_lightgbm_grade_dataset.py --grade-csv E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/samples.csv --combined-root E:/PythonProject/datasets/combined_all_pseudo_v1 --output-dir E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade

# 可视化
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/visualize_dianjie_grade_features.py --samples-csv E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/samples.csv --summary-json E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/summary.json --output-dir E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/visualizations
```

### 等级分布（detection_only，conf>=0.70）

| 等级 | 数量 | 占比 |
| --- | --- | --- |
| 3号板 | 7 | 0.21% |
| 烫洗板 | 3 | 0.09% |
| 合格整版 | 2813 | 86.18% |
| 合格600 | 433 | 13.27% |
| 2号板 | 8 | 0.25% |
| 合格不足600 | 0 | 0.00% |
| 无板 | 0 | 0.00% |

### 唯一性审计说明

唯一性审计就是“查重/跨 split 泄漏检查”。`grade_feature_analysis/combined_all_pseudo_v1_grade/uniqueness_audit/` 中保留了之前的审计证据：
- `all_images_md5.csv`：每张图的 MD5、物理分组、路径
- `exact_duplicates.csv`：完全重复组（同图多份拷贝）
- `physical_groups.csv`：同一物理镍板分组
- `report.md` / `summary.json`：审计报告与统计

去重后当前 `combined_all_pseudo_v1` 共 3264 张（train 2285 / val 490 / test 489）。

### 下一步

1. 打开 `pseudo_label_reviews/combined_all_pseudo_v1/review.html` 人工审查伪标签。
2. 确认 `dianjie_grade_rule.py` 是否需要补全“合格不足600”分支，或该等级在业务上本就由其他规则判定。
3. 若需训练 LightGBM，注意类别极不平衡（合格整版占 86.2%）。

---
