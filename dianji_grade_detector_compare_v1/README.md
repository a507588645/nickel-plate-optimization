# 电积镍等级分类检测器特征提取四臂对比产物说明 (dianji_grade_detector_compare_v1)

- **状态**：`unconfirmed_pending_enterprise`
- **评测数据集**：`archives/dianjinie_grade_classification_v1`（1072 张图像，750 train / 161 valid / 161 test）
- **分类算法**：LightGBM (num_leaves=8, max_depth=4, min_data=10, seed=42)

## 目录结构

- `report.md`：四臂完整对比分析与局限报告（包含 Master Table 与 Rejection Curves）
- `four_arms_summary.json`：四臂对比所有关键指标与参数汇总 JSON
- `arm3_slf640/`：臂 ③（slf.pt@640, 5类, 13+20=33维）完整产物
  - `features_slf640_v1.csv`：1072 图全量特征（13 base + 20 tx__）
  - `features_slf640_base13.csv`：1072 图纯检测 13 维特征
  - `per_box_slf640.csv`：13792 个检出框明细
  - `extraction_summary.json`：提取元数据与统计
  - `metrics_slf_base13.json`：纯检测 LightGBM 指标
  - `metrics_slf_texture33.json`：全特征 LightGBM 指标
  - `reject_curve.csv`：拒判曲线
  - `model_comparison.csv`：各模型验证与测试对比
  - `report.md`：单臂报告
- `arm4_ryw640/`：臂 ④（ryw.pt@640, 18类, 43+20=63维控制臂）完整产物
  - `features_ryw640_v1.csv`：1072 图全量特征（43 base + 20 tx__）
  - `features_ryw640_base43.csv`：1072 图纯检测 43 维特征
  - `per_box_ryw640.csv`：7182 个检出框明细
  - `extraction_summary.json`：提取元数据与统计
  - `metrics_v2b_base43.json`：纯检测 LightGBM 指标
  - `metrics_v3_texture_only.json`：全特征 LightGBM 指标
  - `reject_curve.csv`：拒判曲线
  - `model_comparison.csv`：各模型验证与测试对比
  - `report.md`：单臂报告

## 复现入口

```text
# 1. 提取 slf.pt@640 特征 (Arm 3)
E:/PythonProject/ImetricCodes/JinchuanPythonWebapi/.venv/Scripts/python.exe extract_dianji_grade_slf640_features.py --output-dir dianji_grade_detector_compare_v1/arm3_slf640

# 2. 提取 ryw.pt@640 特征 (Arm 4 控制臂)
E:/PythonProject/ImetricCodes/JinchuanPythonWebapi/.venv/Scripts/python.exe extract_dianji_grade_ryw640_features.py --output-dir dianji_grade_detector_compare_v1/arm4_ryw640

# 3. 训练 Arm 3 LightGBM
E:/PythonProject/优化/.venv_classify/Scripts/python.exe train_dianji_grade_v3.py --features dianji_grade_detector_compare_v1/arm3_slf640/features_slf640_v1.csv --output-dir dianji_grade_detector_compare_v1/arm3_slf640 --seed 42

# 4. 训练 Arm 4 LightGBM
E:/PythonProject/优化/.venv_classify/Scripts/python.exe train_dianji_grade_v3.py --features dianji_grade_detector_compare_v1/arm4_ryw640/features_ryw640_v1.csv --output-dir dianji_grade_detector_compare_v1/arm4_ryw640 --seed 42
```
