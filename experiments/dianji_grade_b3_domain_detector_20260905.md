# B3 域内检测器特征实验记录（dianji_grade_b3，2026-09-05）

| 项 | 内容 |
| --- | --- |
| 实验 | 路线 B 步骤 B3：Dianjinie0211 域内 YOLO11m@640 检测器（Kaggle 训练）→ slf 5 类特征 schema → LGBM 重训对比 |
| 执行者 | Kimi Code |
| 状态 | **完成；结论 pass（ΔmacroF1 vs B0 = +0.1383 ≥ +0.01 闸门），但解读必须带泄漏前提（见下）** |
| 数据 | `archives\dianjinie_grade_classification_v1` 1072 图（企业真值），750/161/161 seed42 |
| 协议 | 提取 `extract_dianji_grade_b3_y11m640_features.py`（slf640 脚本 5 类 schema + y11m640 脚本 ultralytics 加载方式，CPU，imgsz=640 / conf=0.25 / iou=0.45）+ `train_dianji_grade_v3.py --seed 42`（自动识别 SLF_BASE_COLUMNS → slf_base13 / slf_texture33 两组） |
| 权重 | B3 run `dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211v1` best.pt，SHA-256=`95b56f70aaefe56e85440c15e00bc12be14f4caf81fbbc2adf84625cb538b606`（139 epoch 早停，best epoch 39；val mAP50=0.243 / mAP50-95=0.0997） |

## 提取产物（`优化\dianji_grade_b3_y11m640_v1\`）

- 1072/1072 图全量处理，**16,605 框**（waterstain 6,618 / waterblock 6,946 / bubble 2,494 / colorstain 547 / **nodule 0**——B3 训练集 nodule 仅 5 框，检测器从不预测该类，`count__nodule` 与 `has_nodule` 在本特征集中为全零常数列）。
- 纹理 join 1072/0 不匹配；`features_b3y11m640_v1.csv` = 1072 行 × 57 列（24 元数据 + 13 BASE + 20 纹理）；`plate_area_fallbacks=0`。
- CPU 提取速率 1.32 img/s，全程 13.6 分钟；日志 `优化\b3_extract_run.log`；smoke（8 图）先行通过。

## 结果（test=161，LightGBM，seed=42）

| 特征集 | 列数 | valid acc | valid macroF1 | test acc | test macroF1 | F1_3# | F1_9950 | F1_9996 | F1_bad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slf_base13 | 13 | 0.7391 | 0.6236 | 0.6708 | 0.4744 | 0.2667 | 0.6849 | 0.7237 | 0.2222 |
| **slf_texture33** | 33 | **0.8447** | **0.7925** | **0.8385** | **0.8180** | 0.9333 | 0.8485 | 0.8235 | 0.6667 |

test 混淆矩阵（texture33）：3# 7/7 全对；9950↔9996 互混各 12 张（主残差）；bad 2/4。

## 与锚点对比（test macroF1）

| 配置 | test acc | test macroF1 | ΔmacroF1 vs B0 | 备注 |
| --- | --- | --- | --- | --- |
| **B0** y11m 18 类 ONNX@640（跨域，无泄漏） | 0.7453 | 0.6797 | — | 现役部署血缘 |
| arm3 slf.pt@640（5 类域内，同泄漏性质） | 0.7702 | 0.6528 | −0.0269 | 四臂对比历史值 |
| **B3 slf_texture33（本实验）** | **0.8385** | **0.8180** | **+0.1383** | **≥+0.01 闸门通过** |

- 同 schema 同泄漏前提的 detector 质量隔离对比（B3 vs arm3 slf.pt）：Δacc=+0.0683 / ΔmacroF1=**+0.1652**；纯检测列对比（base13 vs arm3 base13）0.4744 vs 0.4278（+0.0466）。
- 对照臂噪声参考 ±0.02 macroF1（B2 PT@640 对照结论），本效应（+0.138）远超噪声带。
- 拒判曲线（阈值仅由 valid 定）：0.7 目标覆盖 → 阈值 0.8134，test 实际覆盖 59.6%，接收 acc=0.9062 / macroF1=0.8501；0.6 覆盖点 acc=0.9268 / macroF1=0.8863。
- 特征重要性 top5：tx__lbp_bin00、tx__glcm_correlation、area_ratio__colorstain、area_ratio__bubble、tx__gray_mean——域内检测器的色斑/气泡/水渍面积特征直接进入第一梯队。

## ⚠️ 泄漏前提（解读必读）

- 等级视图 1072 行中 **980 行**在 B3 train/val 清单内（行级 stem 匹配；审计口径全数据集 stem 级为 986）。按等级 split：**grade-test 161 张中 146 张（90.7%）被 B3 检测器在训练/验证期见过**（110 在 B3 train、36 在 B3 val）；grade-valid 152/161、grade-train 682/750。
- 因此「B3 > B0」不能干净归因：B0 为电解镍跨域权重无此泄漏，B3 数字含检测器记忆成分。**同性质参照系是 arm3 slf.pt@640**（slf.pt 与 B3 同数据集同原始划分训练），该对比下 B3 仍 +0.1652 macroF1，说明在同等泄漏前提下域内 YOLO11m 特征质量确实优于旧 YOLOv5 slf.pt。
- 全部结果仍为 `unconfirmed_pending_enterprise`；本实验不产出部署候选，不导出、不替换部署 ONNX。

## 边界与限制

- 单 seed（42）单划分；3#=7、bad=4 的 test 支持度使小类 F1 波动大（3# 0.9333 建立在 7 张全对上）。
- nodule 特征列全零（见上），5 类体系对结瘤类缺陷无表达。
- 若未来要把 B3 血缘推进部署，需要：① 企业确认等级标准；② 解决检测器训练集与等级评测集的泄漏（如按板重划分重训或取得全新批次图）。

## 复现

```text
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe extract_dianji_grade_b3_y11m640_features.py --output-dir dianji_grade_b3_y11m640_v1
.\.venv_classify\Scripts\python.exe train_dianji_grade_v3.py --features dianji_grade_b3_y11m640_v1\features_b3y11m640_v1.csv --output-dir dianji_grade_b3_y11m640_v1 --seed 42
```
