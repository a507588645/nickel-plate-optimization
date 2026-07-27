## 2026-07-15 12:40 | 删除原汇总数据集 combined_all_pseudo_v1 及相关旧产物

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi Code CLI |
| 任务目标 | 按用户要求删除原汇总数据集 `combined_all_pseudo_v1` 及相关的旧构建日志，仅保留去重版 |
| 状态 | 已完成 |
| 关联用户请求 | 为什么还保留原本的汇总数据，删除，之前的审查之类的也删了 |

### 删除清单

- `E:/PythonProject/datasets/combined_all_pseudo_v1/`（原汇总数据集，images_all 内为指向原始来源的软链接，删除不影响原始数据集）
- `E:/PythonProject/优化/build_combined_pseudo_dataset.log`
- `E:/PythonProject/优化/build_combined_pseudo_dataset_gpu1024.log`

### 保留产物

- 去重数据集：`E:/PythonProject/datasets/combined_all_pseudo_v1_dedup_v2/`
- 等级特征/可视化：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/`
- 唯一性审计：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/uniqueness_audit/`
- 伪标签审查页：`E:/PythonProject/优化/pseudo_label_reviews/combined_all_pseudo_v1_dedup_v2/`

### 说明

- `combined_all_pseudo_v1/images_all/` 内为软链接，指向 `Dianjinie0211`、`dianjienie`、`classify_*` 等原始数据集；删除汇总目录不会删除原始数据。
- 基于 `combined_all_pseudo_v1` 的审查页此前已删除；当前仅保留基于 `dedup_v2` 的审查页。

### 下一步

- 用户决定：是否继续训练 LightGBM，或开始伪标签人工审查。
