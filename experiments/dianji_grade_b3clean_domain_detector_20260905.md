# B3-clean 泄漏治理重训特征实验记录（dianji_grade_b3clean，2026-09-05）

| 项 | 内容 |
| --- | --- |
| 实验 | 路线 B 步骤 B3 的洁净重训：Dianjinie0211 域内 YOLO11m@640 检测器（grade-test 采集键整体排除后 Kaggle 重训）→ slf 5 类特征 schema → LGBM 重训；**首次得到 test 侧零泄漏的域内检测器等级指标** |
| 执行者 | Kimi Code |
| 状态 | **完成；结论 pass（零泄漏口径 ΔmacroF1 vs B0 = +0.0428 ≥ +0.01 闸门）** |
| 授权 | 用户 2026-09-05 裁决选项 A（授权 B3-clean 重训，红线 5 Kaggle GPU 单独授权已满足） |
| 数据 | `archives\dianjinie_grade_classification_v1` 1072 图（企业真值），750/161/161 seed42 |
| 协议 | 提取 `run_b3clean_pt_extract.py`（B2 式包装器解锁 SHA 守卫，内部走 `extract_dianji_grade_b3_y11m640_features.py`：slf 5 类 schema + ultralytics 加载，CPU，imgsz=640 / conf=0.25 / iou=0.45）+ `train_dianji_grade_v3.py --seed 42`（slf_base13 / slf_texture33 两组） |
| 权重 | B3-clean run `dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211clean1` best.pt，SHA-256=`849fe3c9467bfabc4113fd442723fbb65482ed2e5a1475ba9342f52011144459`（203 epoch 早停，best epoch 103；val178 mAP50=0.234 / mAP50-95=0.0924；last.pt=`e3c37d69…`；训练 6.29h，Tesla P100，torch 2.6.0+cu124 / ultralytics 8.4.66） |

## 泄漏治理（与 B3 的唯一变量）

- 排除规则：grade-test 161 行中 146 行命中 B3 990 图池（sha ∪ stem ∪ dup_group）→ 提取 **145 个采集键**（B 族 17 位时间戳 / A 族 6 位前缀 / OTHER 全 stem）→ 命中键的 B3 图整体排除。
- 结果：train 774→**653**（−121 图 / −2,506 框），val 216→**178**（−38 图 / −764 框）；**grade-test 161 行对 B3-clean 检测器零泄漏**。
- grade-train/valid 与检测器训练集的重叠保留（LGBM 训练侧重叠不影响 test 评测洁净性；全排会毁掉检测器 764/774）。
- 审计轨迹：`b3clean_dianjinie0211_y11m640_task_v1\splits\`（split_summary.json / excluded_images.csv，构建器内置探针数字断言 146/145/121/38/2506）。
- nodule：train 框 5→4，val 本就 0 框，该类依旧不可评（与 B3 相同）。

## 提取产物（`优化\dianji_grade_b3clean_y11m640_v1\`）

- 1072/1072 图全量处理，**22,109 框**（waterblock 10,000 / waterstain 9,154 / bubble 2,302 / colorstain 651 / nodule 2）——比 B3 同视图多 33%（16,605 框），nodule 有 2 框残留表达（B3 为 0）。
- 纹理 join 1072/0 不匹配；`features_b3y11m640_v1.csv` = 1072 行 × 57 列；`plate_area_fallbacks=0`。
- CPU 提取 1.32 img/s，13.6 分钟。
- 注：本目录 `report.md`/`README.md` 为训练脚本模板自动生成，其中「特征来源 dianji_grade_v3\features_v3_full.csv」一行为模板固有文案（B3 产物同病），真实特征来源以本记录与 `extraction_summary.json` 为准。

## 结果（test=161，LightGBM，seed=42）

| 特征集 | 列数 | valid acc | valid macroF1 | test acc | test macroF1 | F1_3# | F1_9950 | F1_9996 | F1_bad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slf_base13 | 13 | 0.7516 | 0.6032 | 0.6460 | 0.5708 | 0.455 | 0.658 | 0.671 | 0.500 |
| **slf_texture33** | 33 | **0.8075** | **0.7422** | **0.7578** | **0.7225** | 0.800 | 0.770 | 0.748 | 0.571 |

test 混淆矩阵（texture33）：3# 6/7（1→9950）；9950 62/82（19→9996）；9996 52/68（15→9950，9950↔9996 互混仍是主残差）；bad 2/4。

## 与锚点对比（test macroF1，同 seed42 同协议）

| 血缘 | 列数 | test acc | test macroF1 | ΔmacroF1 vs B0 | 泄漏状态 |
| --- | --- | --- | --- | --- | --- |
| **B0** y11m 18 类 ONNX@640（跨域） | 63 | 0.7453 | 0.6797 | — | 无（跨域权重，构造上无泄漏） |
| arm3 slf.pt@640（YOLOv5 域内） | 33 | 0.7702 | 0.6528 | −0.0269 | test 146/161 泄漏（同 B3 性质） |
| B3-leaked（YOLO11m 域内） | 33 | 0.8385 | 0.8180 | +0.1383 | test 146/161 泄漏 |
| **B3-clean（本实验）** | 33 | **0.7578** | **0.7225** | **+0.0428** | **test 零泄漏（161/161 未见）** |

## 结论

1. **闸门判定：pass**。零泄漏口径下 B3-clean vs B0：Δacc=+0.0125、**ΔmacroF1=+0.0428 ≥ +0.01**。域内检测器血缘对等级特征的真实增益首次在无泄漏前提下成立。
2. **泄漏量化**：B3-leaked（0.8180）− B3-clean（0.7225）= **0.0955 macroF1** 为「泄漏记忆 + 训练集少 159 图」的混合上限；B3 表观增益 +0.1383 中约 2/3 来自泄漏，真实域内收益的保守估计为 +0.0428（把全部差值归给泄漏，故为下界）。
3. **同 schema 参照**：B3-clean（0.7225，零泄漏）仍高于 arm3 slf.pt@640（0.6528，带泄漏）——干净 YOLO11m 域内特征优于泄漏的 YOLOv5 域内特征，检测器质量差异独立于泄漏存在。
4. 拒判曲线（阈值仅由 valid 定）：0.70 目标覆盖 → 阈值 0.7819，test 实际覆盖 67.1%，acc=0.8611 / macroF1=0.8256；0.60 档 acc=0.8660。
5. 特征重要性 top5：tx__lbp_bin00、total_bbox_union_image_ratio、tx__glcm_correlation、tx__gray_mean、area_ratio__waterblock——纹理仍居首，域内检测器的面积/计数特征占 5 席中的 1 席（top20 中 8 席）。

## 边界与限制

- 单 seed（42）单划分；3#=7、bad=4 的 test 支持度使小类 F1 波动大。
- B3-clean 检测器训练集更小（653 vs 774），其检测指标（val178 mAP50=0.234）与 B3（val216 mAP50=0.243）**不同 val 不排名**（AGENTS §8）。
- 检测器框数增多（+33%）与等级性能的关系未单独消融；纹理特征 join 自同一图像，不受检测器影响。
- 全部结果仍为 `unconfirmed_pending_enterprise`；本实验不产出部署候选，不导出、不替换部署 ONNX。
- B3-leaked 的 +0.1383 自本记录起按「含约 2/3 泄漏成分」解读，不再作为域内收益点估计引用。

## 复现

```text
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe run_b3clean_pt_extract.py --weights E:\PythonProject\models\runs\dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211clean1\weights\best.pt --output-dir dianji_grade_b3clean_y11m640_v1
.\.venv_classify\Scripts\python.exe train_dianji_grade_v3.py --features dianji_grade_b3clean_y11m640_v1\features_b3y11m640_v1.csv --output-dir dianji_grade_b3clean_y11m640_v1 --seed 42
```
