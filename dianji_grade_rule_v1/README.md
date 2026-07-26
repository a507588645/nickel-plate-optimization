# dianji_grade_rule_v1 — 电积镍规则判级第一版骨架产物

> **status: `unconfirmed_pending_enterprise`**
>
> 全部候选规则与阈值来自 0309 小样本统计挖掘，**未经企业确认**，禁止当作生产标准。

## 生成时间

`2026-07-23T01:49:17`

## 文件清单

| 文件 | 说明 |
| --- | --- |
| `grade_features_0309.csv` | 86 张企业等级样本的 18 类特征 + 真值标签 |
| `candidate_rules.json` | 有序候选规则表（引擎直接加载） |
| `report.md` | 挖掘报告（含小样本警告与未确认横幅） |
| `grade_results_0309.csv` | 引擎对 0309 特征的逐图回判 |
| `README.md` | 本说明 |

## 数据来源

- 等级真值：`E:/PythonProject/datasets/enterprise_dianji_0309_classification/images/{3#,9950}/`
- 特征：`grade_feature_analysis/combined_all_pseudo_v2_grade/samples.csv`
- Join 覆盖：86/86 = 100.0%
- 检测权重（历史伪标签）：`E:/PythonProject/Tools/ModelAutoLabelTool/weights/yolov5/ryw.pt`
- 检测参数说明：特征来自既有 combined_all_pseudo_v2 / ryw.pt 伪标签派生；历史口径 conf=0.25,iou=0.45,img-size=1024

## 复现

```bash
cd /d E:\PythonProject\优化
.\.venv_classify\Scripts\python.exe mine_dianji_grade_rules.py
.\.venv_classify\Scripts\python.exe dianji_grade_rule.py --self-test
.\.venv_classify\Scripts\python.exe dianji_grade_rule.py --features dianji_grade_rule_v1\grade_features_0309.csv --rules dianji_grade_rule_v1\candidate_rules.json --output dianji_grade_rule_v1\grade_results_0309.csv
.\.venv_classify\Scripts\python.exe dianji_grade_rule.py --features dianji_grade_rule_v1\grade_features_0309.csv --dry-run --output dianji_grade_rule_v1\grade_features_dryrun.csv
```

## 边界

- 仅覆盖 `3#` / `9950`；`9996` / `bad` 无样本 → 引擎输出 `manual_review`。
- 规则表顶层固定 `status=unconfirmed_pending_enterprise`。
- 派生产物仅在本目录；未改动任何数据集原件。
