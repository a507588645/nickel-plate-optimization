# B3 泄漏隔离评估（grade-test 洁净子集）

> 全部结果 `unconfirmed_pending_enterprise`；洁净子集样本量极小，仅作方向性证据。

## 泄漏复核（SHA-256 ∪ stem ∪ dup_group 传染，对 B3 990 图）

| grade split | total | sha 命中 | stem 命中 | 仅组传染 | 并集泄漏 | 洁净 |
| --- | --- | --- | --- | --- | --- | --- |
| train | 750 | 682 | 682 | 4 | 686 | 64 |
| valid | 161 | 151 | 152 | 0 | 152 | 9 |
| test | 161 | 146 | 146 | 0 | 146 | 15 |

grade-test 洁净子集 **15 张**：

- `ent0309_3sharp_40.png` grade=3# src=enterprise_dianji_0309_classification
- `ent0309_9950_111.png` grade=9950 src=enterprise_dianji_0309_classification
- `ent0309_9950_3108.png` grade=9950 src=enterprise_dianji_0309_classification
- `ent0309_9950_3853.png` grade=9950 src=enterprise_dianji_0309_classification
- `ent0309_9950_4178.png` grade=9950 src=enterprise_dianji_0309_classification
- `ent0309_9950_4318.png` grade=9950 src=enterprise_dianji_0309_classification
- `ent0309_9950_4423.png` grade=9950 src=enterprise_dianji_0309_classification
- `ent0309_9950_4427.png` grade=9950 src=enterprise_dianji_0309_classification
- `ent0309_9950_4531.png` grade=9950 src=enterprise_dianji_0309_classification
- `ent0309_9950_4623.png` grade=9950 src=enterprise_dianji_0309_classification
- `ent0309_9950_5684.png` grade=9950 src=enterprise_dianji_0309_classification
- `ent0309_9950_5767.png` grade=9950 src=enterprise_dianji_0309_classification
- `ent0309_9950_5898.png` grade=9950 src=enterprise_dianji_0309_classification
- `ent0309_9950_6405.png` grade=9950 src=enterprise_dianji_0309_classification
- `ent0309_9950_884.png` grade=9950 src=enterprise_dianji_0309_classification

## LGBM 对比（seed=42，test161 全集 vs 洁净子集）

| 血缘 | 列数 | test161 acc | test161 macroF1 | 洁净 n | 洁净 acc | 洁净 macroF1 | 洁净等级分布 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b0_y11m18_onnx640 | 63 | 0.7453 | 0.6797 | 15 | 0.9333 | 0.4828 | {'3#': 1, '9950': 14} |
| b3_d0211y11m640 | 33 | 0.8385 | 0.8180 | 15 | 1.0000 | 1.0000 | {'3#': 1, '9950': 14} |
| arm3_slf640 | 33 | 0.7702 | 0.6528 | 15 | 1.0000 | 1.0000 | {'3#': 1, '9950': 14} |

解读约束：洁净子集 macroF1 的单样本波动可达 ±0.1 以上（bad/3# 支持度个位数），
只能判断「B3 在未见图上是否崩塌」，不能给出精确排名。
