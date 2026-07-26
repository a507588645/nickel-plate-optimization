# 电积镍 0309 浅层候选规则挖掘报告

> ## ⚠️ 未经企业确认 / UNCONFIRMED
>
> 本报告全部阈值与规则均来自 **86 张企业 0309 已判级样本的统计挖掘**，
> **不是**企业书面标准，**不得**直接上线生产。
> 规则表 `status` 固定为 `unconfirmed_pending_enterprise`。

> ## ⚠️ 小样本警告
>
> - 总样本：86 张
> - 3#：12 张（极少）
> - 9950：74 张
> - 无 9996 / bad 样本 → **禁止**为这两类编造规则；未命中 → `manual_review`
> - 交叉验证分数波动大，仅作候选线索，需企业现场确认后才能固化。

## 1. 数据与 join

- 生成时间：`2026-07-23T01:49:17`
- 企业目录：`E:/PythonProject/datasets/enterprise_dianji_0309_classification`
- 特征来源：`E:\PythonProject\优化\grade_feature_analysis\combined_all_pseudo_v2_grade\samples.csv`
- Join 方法：image_name prefix join with combined_all_pseudo_v2 samples.csv (no re-inference)
- Join 覆盖：`86/86` = **100.0%**
- 缺失：0 张
- 检测权重（历史）：`E:\PythonProject\Tools\ModelAutoLabelTool\weights\yolov5\ryw.pt`
- 检测参数说明：特征来自既有 combined_all_pseudo_v2 / ryw.pt 伪标签派生；历史口径 conf=0.25,iou=0.45,img-size=1024

企业等级分布（join 后）：

- `3#`: 12
- `9950`: 74

## 2. 挖掘方法

- 方法：`sklearn.DecisionTreeClassifier`
- 随机种子：`42`
- 特征维数：`43`
- 可用特征列：`A_ratio, B_ratio, C_ratio, Airhole_ratio, has_nodule, total_bbox_union_image_ratio, detection_count, count__wuzi, count__waterstain, count__yellowstain, count__nodule, count__airhole...`

### 交叉验证

- 策略：StratifiedKFold(n_splits=5, shuffle=True, seed=42)
- 准确率：`0.6279`
- 说明：分层 5 折交叉验证

混淆矩阵（行=真值，列=预测）：

|  | 3# | 9950 |
| --- | --- | --- |
| 3# | 6 | 6 |
| 9950 | 26 | 48 |

## 3. 候选规则（有序，首条命中）

### 规则 1: `tree_leaf_3sharp_1` → **3#**

- 条件：`detection_count` <= 5.5 AND `area_ratio__airhole` <= 0.00349968 AND `total_bbox_union_image_ratio` > 0.000189745
- 训练叶节点支持：38 （3#=12, 9950=26）
- 来源：DecisionTreeClassifier
- **状态：未经企业确认**

### 规则 2: `tree_leaf_9950_3` → **9950**

- 条件：`detection_count` > 5.5
- 训练叶节点支持：28 （3#=0, 9950=28）
- 来源：DecisionTreeClassifier
- **状态：未经企业确认**

### 规则 3: `tree_leaf_9950_2` → **9950**

- 条件：`detection_count` <= 5.5 AND `area_ratio__airhole` > 0.00349968
- 训练叶节点支持：11 （3#=0, 9950=11）
- 来源：DecisionTreeClassifier
- **状态：未经企业确认**

### 规则 4: `tree_leaf_9950_0` → **9950**

- 条件：`detection_count` <= 5.5 AND `area_ratio__airhole` <= 0.00349968 AND `total_bbox_union_image_ratio` <= 0.000189745
- 训练叶节点支持：9 （3#=0, 9950=9）
- 来源：DecisionTreeClassifier
- **状态：未经企业确认**

## 4. 训练集上回判（非独立测试，仅描述拟合）

- 准确率：`0.6977`
- manual_review 数：0

逐规则：

- `tree_leaf_3sharp_1`: n=38, correct=12, acc=0.316
- `tree_leaf_9950_3`: n=28, correct=28, acc=1.000
- `tree_leaf_9950_0`: n=9, correct=9, acc=1.000
- `tree_leaf_9950_2`: n=11, correct=11, acc=1.000

回判混淆矩阵：

|  | 3# | 9950 |
| --- | --- | --- |
| 3# | 12 | 0 |
| 9950 | 26 | 48 |

## 5. 规则含义（业务解读草稿，非确认）

- 规则基于 18 类 `count__*` / `area_ratio__*` 与 A/B/C/Airhole 组面积比；
  A/B/C 分组沿用电解镍初始假设，**不是**企业电积镍正式分组。
- 决策树/网格搜索找到的阈值反映的是本批 0309 目录中 3# 与 9950 的可分性，
  可能混入批次、光照、伪标签噪声，不能外推。
- 上线前必须：① 企业书面标准对照；② 独立批次复核；③ 人工审查伪标签框质量。

## 6. 复现命令

```bash
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe mine_dianji_grade_rules.py
.\.venv_classify\Scripts\python.exe dianji_grade_rule.py --features dianji_grade_rule_v1\grade_features_0309.csv --rules dianji_grade_rule_v1\candidate_rules.json --output dianji_grade_rule_v1\grade_results_0309.csv
```

## 附录：决策树文本

```
|--- detection_count <= 5.50
|   |--- area_ratio__airhole <= 0.00
|   |   |--- total_bbox_union_image_ratio <= 0.00
|   |   |   |--- weights: [5.23, 0.00] class: 0
|   |   |--- total_bbox_union_image_ratio >  0.00
|   |   |   |--- weights: [15.11, 43.00] class: 1
|   |--- area_ratio__airhole >  0.00
|   |   |--- weights: [6.39, 0.00] class: 0
|--- detection_count >  5.50
|   |--- weights: [16.27, 0.00] class: 0

```
