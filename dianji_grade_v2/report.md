# 电积镍等级分类 v2 报告

> ## ⚠️ 未经企业确认
>
> 本报告全部模型、规则与阈值均为实验性产物，**不得**作为企业正式判级标准或上线依据。
> 准确率如实呈现；禁止调参过拟合后宣称可用。

> ## ⚠️ 小样本警告
>
> `3#` 合计 46 张（含 0309 的 12 张）、`bad` 合计 28 张，类别极不平衡，少数类指标方差大。
> 用户任务中 4types 口径 3#=34、bad=28；合并 0309 后 3#=46。

- 生成时间：2026-07-23T02:52:22
- 产物目录：`E:\PythonProject\优化\dianji_grade_v2`
- 状态：`unconfirmed_pending_enterprise`
- 协议：train 训练 / valid 选模与早停 / **test 仅最终一次**

## 1. 数据与去重 / 硬链接视图

- 视图：`E:\PythonProject\datasets\dianjinie_grade_classification_v1`
- 原始：4types=986 + 0309=86
- 完整重复组：0，移除 0
- 近重复：方法=dHash thr≤5，多成员组=16，涉及图=33
- 去重后：1072；等级：`{'3#': 46, '9950': 546, '9996': 452, 'bad': 28}`
- 划分 seed=42，图片比例：`{'train': 0.6996268656716418, 'valid': 0.1501865671641791, 'test': 0.1501865671641791}`
- 各 split×等级：`{'train': {'3#': 32, '9950': 382, '9996': 316, 'bad': 20}, 'valid': {'3#': 7, '9950': 82, '9996': 68, 'bad': 4}, 'test': {'3#': 7, '9950': 82, '9996': 68, 'bad': 4}}`
- 链接：`{'hardlink': 1072}`（硬链接优先）
- **不沿用** 4types 原 train/valid；原 split 记在 manifest `source_split`

## 2. 特征版本

- **v2a** join：156/1072 = 14.55%；缺失 916（`features_v2a_missing.csv`）。 口径：历史口径 ryw.pt conf=0.25 iou=0.45 imgsz=1024
  - 说明：combined 伪标签集仅覆盖部分 4types 样本，故 v2a 仅作对照，**主结论以 v2b 为准**。
- **v2b** 主特征：覆盖 1072/1072 = 100.00%；device=0 cuda=True imgsz=1536 conf=0.25 iou=0.45
  - 逐框：`E:\PythonProject\优化\dianji_grade_v2\detections_1536\all_boxes.csv`；特征：`{'conf_0p25': 'E:\\PythonProject\\优化\\dianji_grade_v2\\features_v2b_imgsz1536_conf0p25.csv', 'conf_0p40': 'E:\\PythonProject\\优化\\dianji_grade_v2\\features_v2b_imgsz1536_conf0p40.csv'}`

## 3. 模型对比

四行基线：majority / DianjieGradeRule / decision_tree / lightgbm。
valid 用于选模；test 表为最终对照（**未参与选模**）。

### 特征 `v2a_join_samples_csv`

- n: train=114 valid=18 test=24；分布 train=`{'3#': 13, '9950': 75, '9996': 25, 'bad': 1}`

**Valid（选模）**

| 模型 | Acc | Macro-F1 | F1_3# | F1_9950 | F1_9996 | F1_bad | other_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority | 0.6667 | 0.2667 | 0.0000 | 0.8000 | 0.0000 | 0.0000 | - |
| dianjie_grade_rule | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 |
| decision_tree | 0.5556 | 0.4993 | 0.2857 | 0.6667 | 0.5455 | 0.0000 | - |
| lightgbm | 0.6667 | 0.4333 | 0.0000 | 0.8000 | 0.5000 | 0.0000 | - |

**Test（最终，未参与调参）**

| 模型 | Acc | Macro-F1 | F1_3# | F1_9950 | F1_9996 | F1_bad | other_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority | 0.7917 | 0.2946 | 0.0000 | 0.8837 | 0.0000 | 0.0000 | - |
| dianjie_grade_rule | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 |
| decision_tree | 0.2917 | 0.3162 | 0.1429 | 0.1905 | 0.6154 | 0.0000 | - |
| lightgbm | 0.7500 | 0.5000 | 0.0000 | 0.8333 | 0.6667 | 0.0000 | - |

- valid 最优：`decision_tree` macro_f1=0.49927849927849927
- 选定模型 test：`decision_tree` acc=0.2916666666666667 macro_f1=0.3162393162393162

LightGBM 特征重要性 top15（gain）：

- `area_ratio__wuzi`: 534.3875
- `total_bbox_union_image_ratio`: 337.7131
- `A_ratio`: 273.4286
- `B_ratio`: 252.9734
- `Airhole_ratio`: 226.8677
- `detection_count`: 216.8879
- `count__wuzi`: 209.5420
- `area_ratio__watermark`: 153.9108
- `count__watermark`: 121.9925
- `area_ratio__waterstain`: 107.9037
- `area_ratio__airhole`: 25.4801
- `count__waterstain`: 13.8195
- `area_ratio__bump`: 3.8303
- `count__bump`: 1.2308
- `count__airhole`: 0.0097

LightGBM 混淆矩阵（优先 test；行=真，列=预测，3#/9950/9996/bad）：

```
[[ 0  1  0  0]
 [ 2 15  2  0]
 [ 0  1  3  0]
 [ 0  0  0  0]]
```
- 9950 vs 9996：`{'n': 23, 'accuracy_on_subset': 0.782608695652174, '9950_as_9996': 2, '9996_as_9950': 1, '9950_support': 19, '9996_support': 4}`
- 分歧：17；与 cm off-diag：`{'disagreements': 17, 'cm_offdiag': 17, 'match': True}`

### 特征 `v2b_imgsz1536_conf0.25`

- n: train=750 valid=161 test=161；分布 train=`{'3#': 32, '9950': 382, '9996': 316, 'bad': 20}`

**Valid（选模）**

| 模型 | Acc | Macro-F1 | F1_3# | F1_9950 | F1_9996 | F1_bad | other_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority | 0.5093 | 0.1687 | 0.0000 | 0.6749 | 0.0000 | 0.0000 | - |
| dianjie_grade_rule | 0.0373 | 0.0300 | 0.0000 | 0.1200 | 0.0000 | 0.0000 | 0.870 |
| decision_tree | 0.4534 | 0.2877 | 0.0000 | 0.2737 | 0.6162 | 0.2609 | - |
| lightgbm | 0.6025 | 0.3562 | 0.0000 | 0.6792 | 0.5915 | 0.1538 | - |

**Test（最终，未参与调参）**

| 模型 | Acc | Macro-F1 | F1_3# | F1_9950 | F1_9996 | F1_bad | other_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority | 0.5093 | 0.1687 | 0.0000 | 0.6749 | 0.0000 | 0.0000 | - |
| dianjie_grade_rule | 0.0248 | 0.0505 | 0.1600 | 0.0421 | 0.0000 | 0.0000 | 0.839 |
| decision_tree | 0.4783 | 0.3210 | 0.2400 | 0.2737 | 0.6452 | 0.1250 | - |
| lightgbm | 0.6398 | 0.4131 | 0.1333 | 0.7097 | 0.6667 | 0.1429 | - |

- valid 最优：`lightgbm` macro_f1=0.3561601831599174
- 选定模型 test：`lightgbm` acc=0.639751552795031 macro_f1=0.4131336405529954

LightGBM 特征重要性 top15（gain）：

- `B_ratio`: 2872.8394
- `area_ratio__bubble`: 1577.2306
- `total_bbox_union_image_ratio`: 1535.2838
- `A_ratio`: 1292.1740
- `area_ratio__watermark`: 1132.3399
- `area_ratio__bump`: 1054.6820
- `detection_count`: 986.4169
- `area_ratio__wuzi`: 872.8463
- `count__bump`: 842.0079
- `count__waterstain`: 816.1599
- `count__wuzi`: 781.6366
- `area_ratio__suanwuran`: 542.0438
- `area_ratio__greenstain`: 534.5260
- `area_ratio__waterstain`: 444.1771
- `C_ratio`: 413.0410

LightGBM 混淆矩阵（优先 test；行=真，列=预测，3#/9950/9996/bad）：

```
[[ 1  5  1  0]
 [ 4 55 21  2]
 [ 2 13 46  7]
 [ 1  0  2  1]]
```
- 9950 vs 9996：`{'n': 150, 'accuracy_on_subset': 0.6733333333333333, '9950_as_9996': 21, '9996_as_9950': 13, '9950_support': 82, '9996_support': 68}`
- 分歧：58；与 cm off-diag：`{'disagreements': 58, 'cm_offdiag': 58, 'match': True}`

### 特征 `v2b_imgsz1536_conf0.40`

- n: train=750 valid=161 test=161；分布 train=`{'3#': 32, '9950': 382, '9996': 316, 'bad': 20}`

**Valid（选模）**

| 模型 | Acc | Macro-F1 | F1_3# | F1_9950 | F1_9996 | F1_bad | other_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority | 0.5093 | 0.1687 | 0.0000 | 0.6749 | 0.0000 | 0.0000 | - |
| dianjie_grade_rule | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.994 |
| decision_tree | 0.4845 | 0.2632 | 0.0000 | 0.6952 | 0.2667 | 0.0909 | - |
| lightgbm | 0.5714 | 0.3369 | 0.0000 | 0.6905 | 0.5238 | 0.1333 | - |

**Test（最终，未参与调参）**

| 模型 | Acc | Macro-F1 | F1_3# | F1_9950 | F1_9996 | F1_bad | other_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority | 0.5093 | 0.1687 | 0.0000 | 0.6749 | 0.0000 | 0.0000 | - |
| dianjie_grade_rule | 0.0062 | 0.0357 | 0.1429 | 0.0000 | 0.0000 | 0.0000 | 0.969 |
| decision_tree | 0.4037 | 0.2309 | 0.0667 | 0.5568 | 0.3000 | 0.0000 | - |
| lightgbm | 0.6211 | 0.3694 | 0.1429 | 0.7081 | 0.6269 | 0.0000 | - |

- valid 最优：`lightgbm` macro_f1=0.33690476190476193
- 选定模型 test：`lightgbm` acc=0.6211180124223602 macro_f1=0.36944933716510614

LightGBM 特征重要性 top15（gain）：

- `total_bbox_union_image_ratio`: 2800.1315
- `B_ratio`: 2054.6247
- `A_ratio`: 1783.6746
- `area_ratio__watermark`: 1650.9243
- `detection_count`: 1556.5033
- `area_ratio__wuzi`: 1344.6397
- `count__bump`: 935.2506
- `area_ratio__bump`: 838.0357
- `area_ratio__waterstain`: 671.5031
- `count__waterstain`: 589.5496
- `area_ratio__bubble`: 387.1870
- `C_ratio`: 375.8634
- `count__wuzi`: 323.0767
- `area_ratio__greenstain`: 259.8878
- `count__watermark`: 217.1664

LightGBM 混淆矩阵（优先 test；行=真，列=预测，3#/9950/9996/bad）：

```
[[ 1  5  1  0]
 [ 2 57 20  3]
 [ 3 17 42  6]
 [ 1  0  3  0]]
```
- 9950 vs 9996：`{'n': 150, 'accuracy_on_subset': 0.66, '9950_as_9996': 20, '9996_as_9950': 17, '9950_support': 82, '9996_support': 68}`
- 分歧：61；与 cm off-diag：`{'disagreements': 61, 'cm_offdiag': 61, 'match': True}`

## 4. 一致性与分歧

- 主推特征（完整覆盖优先 + valid LGBM macro_f1）：`v2b_imgsz1536_conf0.25`
- 分歧清单：`disagreements.csv`（该特征 test 上选定模型）

### 9950 vs 9996 可分性

若企业语义上两者主要差在化验成分而非外观，则图像缺陷面积特征**原则上难以稳定区分**。
本次：`{'n': 150, 'accuracy_on_subset': 0.6733333333333333, '9950_as_9996': 21, '9996_as_9950': 13, '9950_support': 82, '9996_support': 68}`。
若互相混淆高，应视为特征上限，而非继续堆模型。

主要分歧对（test）：

- `9950->9996`: 21
- `9996->9950`: 13
- `9996->bad`: 7
- `3#->9950`: 5
- `9950->3#`: 4
- `9950->bad`: 2
- `9996->3#`: 2
- `bad->9996`: 2
- `bad->3#`: 1
- `3#->9996`: 1

## 5. 与侍掠飞论文三分类的定性对照

论文 taxonomy：**合格 / 烫洗 / 不合格**（`师兄论文/2021141128_cglw_侍掠飞.txt` 约 819–820、944–946 行）。
本实验四类：**3# / 9950 / 9996 / bad**，**无「烫洗」对应类**，禁止强行映射报准确率。

特征画像（未验证假设）：

- 「烫洗」≈ 高 `waterstain`/`watermark`/`A_ratio`，且 overfire/suanwuran/C 组不突出的样本。
- 「不合格」≈ `bad` 与部分 `3#`（严重外观/C 组或大面积 B 组）。
- 「合格」≈ 部分 `9996`/`9950`；但牌号化验差无法从图像复现。

本实验回答的是「企业四类目录标签 vs 检测框面积特征」一致性，**不是**论文三类复现。

## 6. 三个业务问题

### ① LightGBM 四分类表现与最优特征版本

- 主推特征：`v2b_imgsz1536_conf0.25`
- Valid LGBM：acc=0.6024844720496895 macro_f1=0.3561601831599174 per-class={'3#': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 7}, '9950': {'precision': 0.7012987012987013, 'recall': 0.6585365853658537, 'f1': 0.6792452830188679, 'support': 82}, '9996': {'precision': 0.5675675675675675, 'recall': 0.6176470588235294, 'f1': 0.5915492957746479, 'support': 68}, 'bad': {'precision': 0.1111111111111111, 'recall': 0.25, 'f1': 0.15384615384615385, 'support': 4}}
- Test LGBM：acc=0.639751552795031 macro_f1=0.4131336405529954 per-class={'3#': {'precision': 0.125, 'recall': 0.14285714285714285, 'f1': 0.13333333333333333, 'support': 7}, '9950': {'precision': 0.7534246575342466, 'recall': 0.6707317073170732, 'f1': 0.7096774193548387, 'support': 82}, '9996': {'precision': 0.6571428571428571, 'recall': 0.6764705882352942, 'f1': 0.6666666666666666, 'support': 68}, 'bad': {'precision': 0.1, 'recall': 0.25, 'f1': 0.14285714285714285, 'support': 4}}
- backend=lightgbm best_iteration=128

### ② 与原标签一致率及分歧模式

- 见表中 Acc/混淆矩阵与 `disagreements.csv`。
- 重点：9950↔9996；3#/bad 少数类召回不稳定。

### ③ 与论文标准的定性符合度

- taxonomy 不同，仅特征画像对照，无准确率映射。见第 5 节。

## 7. 若准确率不足：备选方案

1. 更丰富空间/纹理特征（分区覆盖、LBP/GLCM）。
2. 检测层提精度（更高 imgsz、电积镍专用检测权重、人工审查伪标签）。
3. 端到端图像分类（ResNet/ViT），注意小样本与不平衡。
4. 双阶段：先合格/不合格（或论文三类）再细分牌号；牌号融合化验。
5. 补充 3# / bad / 9950–9996 边界样本。
6. `manual_review` 拒判：低置信度送人工。

## 8. 复现命令

```text
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe -m py_compile build_dianji_grade_v2_dataset.py train_dianji_grade_lgbm.py extract_dianji_grade_features.py build_dianjinie_grade_view.py
.\.venv_classify\Scripts\python.exe build_dianjinie_grade_view.py --force
# v2b 高精度推理需 CUDA torch（示例：JinchuanYolov5sTrain venv）
E:\PythonProject\ImetricCodes\JinchuanYolov5sTrain\venv\Scripts\python.exe extract_dianji_grade_features.py
# 或：同目录 build_dianji_grade_v2_dataset.py --skip-v2a --device 0
.\.venv_classify\Scripts\python.exe train_dianji_grade_lgbm.py
```

说明：特征 CSV 的 `split` 必须与视图 `dianjinie_grade_classification_v1` 的 70/15/15 对齐（本产物已对齐）。

## 9. 先前任务取舍

- 流水线入口：`build_dianjinie_grade_view.py` → `extract_dianji_grade_features.py` / `build_dianji_grade_v2_dataset.py` → `train_dianji_grade_lgbm.py`。
- 硬链接视图 + 按 `dup_group` 的 70/15/15 划分（seed=42）；**不沿用** 4types 原 train/valid 作为正式评测划分。
- `dianji_grade_rule_v1/` 与 `dianji_grade_rule.py` / `mine_dianji_grade_rules.py` / `dianjie_grade_rule.py` **未改动**。

## 10. 近重复说明

- 初版 aHash thr=6 在镍板大图上假近重复极多；已改为 **dHash + 同等级才并组**。
- 本目录 `dedup_summary.json`：完整重复组 **0**、近重复多成员组 **18**（阈值 hamming≤10/256bit，同 grade）。
- 视图侧摘要见 `view_dataset_summary.json`（thr=5，多成员组 16）。
- 完整重复复核：0 组（与 2026-07-13 审计一致：两集合完全重复图片=0）。
