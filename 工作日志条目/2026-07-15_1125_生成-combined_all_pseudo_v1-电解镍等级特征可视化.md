## 2026-07-15 11:25 | 生成 combined_all_pseudo_v1 电解镍等级特征可视化

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi Code CLI |
| 任务目标 | 对汇总数据集 `combined_all_pseudo_v1` 的电解镍等级特征与分类结果做可视化，验证是否符合 `dianjie_grade_rule.py` 硬编码规则 |
| 状态 | 已完成 |
| 关联用户请求 | 对于汇总数据集的特征提取还有使用项目电解镍分类结果，能不能可视化，我需要看一下是不是真符合项目电解镍的硬编码 |

### 改动文件

- `E:/PythonProject/优化/visualize_dianjie_grade_features.py`（新增）

### 产物路径

- 可视化目录：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/visualizations/`
- 等级分布：`01_grade_distribution.png`
- A/B/C/Airhole 比例箱线图：`02_ratio_boxplots.png`
- A/B/C/Airhole 比例小提琴图：`03_ratio_violins.png`
- 硬编码阈值散点图：`04_scatter_thresholds.png`
- plate_area 与检测框并集面积比：`05_plate_and_union_area.png`
- 各类缺陷面积比：`06_defect_area_ratios.png`
- 特征相关性热力图：`07_correlation_heatmap.png`
- 汇总报告：`summary_report.md`

### 关键命令或入口

```text
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/visualize_dianjie_grade_features.py
```

### 结论

- 共 7396 张图片，置信度阈值 0.70，使用 `DianjieGradeRule` 默认阈值。
- 等级分布：合格整版 6321（85.5%）、合格600 1027（13.9%）、3号板 21、2号板 20、烫洗板 7，类别严重不平衡。
- 硬编码规则在样本上表现符合预期：
  - 3号板样本集中在 C_ratio > 0.5 区域；
  - 2号板样本集中在 B_ratio > 0.5 且 C_ratio < 0.5 区域；
  - 烫洗板样本 A_ratio 明显高于合格整版阈值；
  - 合格整版与合格600 在 A/B/C/Airhole 比例上整体低于合格整版阈值，但合格600 的 B_ratio 和 Airhole_ratio 明显高于合格整版。
- 110 张图片不可读 / `findcent` 回退，使用默认 plate_area，比例计算可能失真。
- 147 张图片在 CSV 中被标记为 `findcent_fallback=True`（含不可读与 findcent 异常）。

### 未完成项

- [ ] 实际训练 LightGBM 并验证指标
- [ ] 根据可视化结果决定是否调整等级阈值或合并稀有类别

### 下一步

- 用户决定：是否继续训练 LightGBM，或先调整类别/阈值/特征。

### 风险与待确认

- 该汇总数据集以电积镍为主，套用原项目电解镍硬编码规则属于跨体系迁移，结果仅作参考，不宜直接作为电积镍真值。
- 3号板/2号板/烫洗板样本极少，散点图和箱线图可能被多数类淹没。
