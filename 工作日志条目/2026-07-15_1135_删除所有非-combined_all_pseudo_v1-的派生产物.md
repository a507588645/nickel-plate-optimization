## 2026-07-15 11:35 | 删除所有非 combined_all_pseudo_v1 的派生产物

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi Code CLI |
| 任务目标 | 按用户要求，仅保留汇总数据集 `combined_all_pseudo_v1` 的伪标签、审查、等级划分和可视化，删除其余旧数据产物 |
| 状态 | 已完成 |
| 关联用户请求 | 只保留汇总数据的伪标签还有审查以及等级划分包括可视化，其他数据的删掉 |

### 改动文件

- 无新增/修改脚本，仅删除派生产物目录与文件。

### 删除清单

- `E:/PythonProject/优化/grade_feature_analysis/dianji_ryw_20260713_223649/`
- `E:/PythonProject/优化/grade_feature_analysis/dianji_ryw_20260713_223701/`
- `E:/PythonProject/优化/pseudo_label_datasets/dianjienie_20260713_131840_v1_Dianjinie0211+classify_2types+enterprise_dianjie_0306/`
- `E:/PythonProject/优化/pseudo_label_datasets/dianjinie_20260713_131840_v1_dianjienie+classify_4types_dianji+enterprise_dianji_0309/`
- `E:/PythonProject/优化/pseudo_label_reviews/dianjie_n80_20260713_203049/`
- `E:/PythonProject/优化/pseudo_label_reviews/dianjienie_20260713_131840_v1_Dianjinie0211+classify_2types+enterprise_dianjie_0306/`
- `E:/PythonProject/优化/pseudo_label_reviews/dianjienie_20260713_131840_v1_dianjienie+classify_4types_dianji+enterprise_dianji_0309/`
- `E:/PythonProject/优化/review_enhance_samples/dianjie_n80_20260713_203049/`
- `E:/PythonProject/优化/box_roi_reviews/dianjie_n80_20260713_203049_boxroi/`
- `E:/PythonProject/优化/enterprise_dataset_records/`
- `E:/PythonProject/优化/enterprise_duplicate_audit/`

### 保留产物

- 伪标签审查：`E:/PythonProject/优化/pseudo_label_reviews/combined_all_pseudo_v1/`
- 等级特征与可视化：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/`
- 汇总数据集本身：`E:/PythonProject/datasets/combined_all_pseudo_v1/`（未改动）

### 关键命令

```text
rm -rf grade_feature_analysis/dianji_ryw_20260713_223649 grade_feature_analysis/dianji_ryw_20260713_223701 pseudo_label_datasets/dianjienie_20260713_131840_v1_* pseudo_label_reviews/dianjie_n80_20260713_203049 pseudo_label_reviews/dianjienie_20260713_131840_v1_* review_enhance_samples/dianjie_n80_20260713_203049 box_roi_reviews/dianjie_n80_20260713_203049_boxroi enterprise_dataset_records enterprise_duplicate_audit
```

### 结论

- 所有非 `combined_all_pseudo_v1` 的派生伪标签、审查、ROI、增强样本、企业记录与重复审计目录均已删除。
- 空目录 `pseudo_label_datasets`、`review_enhance_samples`、`box_roi_reviews` 已清理。
- 原始汇总数据集 `E:/PythonProject/datasets/combined_all_pseudo_v1/` 未删除。

### 下一步

- 继续基于 `combined_all_pseudo_v1` 做训练、调参或进一步可视化。
