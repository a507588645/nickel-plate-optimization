## 2026-07-14 21:01 | 对 combined_all_pseudo_v1（以电积镍为主）应用原项目电解镍等级规则并生成 LightGBM 数据集

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi Code CLI |
| 任务目标 | 用原项目电解镍等级规则/API 对汇总数据集做等级划分，评估是否可用于 LightGBM 分类 |
| 状态 | 已完成 |
| 关联用户请求 | 用原项目电解镍等级分类 API 对 combined_all_pseudo_v1 做等级划分，看结果并判断是否可训练 LightGBM |

### 改动文件

- `E:/PythonProject/优化/apply_dianjie_grade_to_combined.py`
- `E:/PythonProject/优化/apply_dianjie_grade_to_combined_sequential.py`
- `E:/PythonProject/优化/resume_dianjie_grade.py`
- `E:/PythonProject/优化/build_lightgbm_grade_dataset.py`

### 产物路径

- 等级特征 CSV：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/samples.csv`
- 统计摘要：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/summary.json`
- LightGBM 数据集：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/lightgbm_grade_dataset.csv`
- LightGBM 摘要：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/lightgbm_grade_summary.json`

### 关键命令或入口

```text
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/apply_dianjie_grade_to_combined_sequential.py --conf-thres 0.70
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/resume_dianjie_grade.py
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/build_lightgbm_grade_dataset.py
```

### 结论

- 共处理 7396 张汇总图片（来自 `images_all/` 软连接）。
- **注意**：该汇总数据集以 `Dianjinie0211`、`dianjienie`（电积镍）为主，本次应用的是原项目 `DianjieGradeRule`（电解镍）规则；两者类别体系/等级标准可能不完全对应，结果仅用于观察规则迁移效果，不宜直接作为电积镍真值。
- 置信度阈值 `conf ≥ 0.70`，使用原项目 `preprocess.findcent` 计算镍板面积，调用 `DianjieGradeRule` 判定等级。
- 110 张图片无法读取（软连接目标缺失或路径含特殊字符），使用默认 plate_area 回退。
- 972 张图片过滤后无检测框。
- 等级分布（detection_only）：
  - 合格整版：6321（85.47%）
  - 合格600：1027（13.89%）
  - 3号板：21（0.28%）
  - 2号板：20（0.27%）
  - 烫洗板：7（0.09%）
- 等级分布（assume_resnet_good）与上类似，3号板升至 95（1.28%），合格600 降至 953。
- 训练/验证/测试划分分布与整体一致。

### LightGBM 可行性

- 可直接用于训练，但类别严重不平衡（合格整版占 85.5%）。
- 3号板、烫洗板、2号板样本极少，直接多分类难以稳定学习这些等级。
- 建议方案：
  1. 先尝试二分类：合格整版 vs 非合格整版，基线更容易稳定。
  2. 若要多分类，使用 LightGBM `class_weight`、SMOTE 等过采样、或 focal loss。
  3. 合并稀有类别（如将 3号板/烫洗板/2号板合并为"不合格"）后再分类。

### 未完成项

- [ ] 实际训练 LightGBM 并验证指标
- [ ] 根据结果调整特征或标签策略

### 下一步

- 用户决定：是否用 `lightgbm_grade_dataset.csv` 开始训练 LightGBM，或先调整类别/阈值。
