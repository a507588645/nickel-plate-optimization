## 2026-07-16 01:23 | 删除旧汇总版本，仅保留 combined_all_pseudo_v2

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex |
| 任务目标 | 清理重复命名的汇总数据集和关联产物，仅保留最新版本并确保数据集位于 datasets |
| 状态 | 已完成 |
| 保留数据集 | `E:/PythonProject/datasets/combined_all_pseudo_v2` |
| 保留产物 | `E:/PythonProject/优化/pseudo_label_reviews/combined_all_pseudo_v2`、`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v2_grade`、`E:/PythonProject/优化/label_incremental_merge/20260716_v1` |

### 删除清单

- `E:/PythonProject/datasets/combined_all_pseudo_v1`
- `E:/PythonProject/优化/pseudo_label_reviews/combined_all_pseudo_v1`
- `E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade`
- `combined_all_pseudo_v1_dedup_v2` 及对应审查目录此前已不存在，无需重复删除。

### 同步调整

- `prepare_label_incremental_merge.py` 的默认基准版本改为 v2，下一增量输出默认改为 v3。
- 等级顺序脚本、续跑脚本和 LightGBM 构建脚本的默认路径统一改为 v2，避免误建旧 v1 目录。
- 企业原始归档 `enterprise_label_raw_20260716` 和去重来源视图 `enterprise_label_dedup_v1` 属于来源数据，不是旧汇总版本，继续保留。

### 验证

- `E:/PythonProject/datasets` 下仅保留一个 `combined*` 数据集：`combined_all_pseudo_v2`。
- 最新数据集仍为 3264 张，六列伪标签 3264 个，审查、等级、LightGBM 和可视化产物完整。

---
