# L1 OOF 无人工自动闸门报告

## 结论

- 自动闸门结果：`auto_continue_oof_candidate`。两个架构均满足本次固定的非回退安全条件。
- 自动候选主模型：`yolo11m`。它在候选标签自评和未改动历史标签复评两种自动口径下的平均 mAP50-95 都略高；差距很小，不能表述为稳定架构优势。
- 你不需要做人工标注、逐图审核或填写审查表。下一步是按该自动候选版本完成全开发集训练，再自动做一次历史 release-validation 和 ONNX 一致性验证。
- 这个闸门是安全筛选，不是人工真值证明：它不能确认每个新增框都正确，只能证明该保守候选没有在旧标签口径上造成超出阈值的总体退化。

## 自动规则

- 历史标签不删除、不改类；OOF 候选只自动新增 245 个双架构同类共识、置信度不低于 0.70、且不重叠历史框的目标。
- 每张开发图的候选框只来自未训练过该图的 YOLO11m 和 YOLO26m Fold 权重，直接同类 IoU >= 0.55 才进入候选池。
- 非回退条件：每个架构候选权重在原始历史标签上重新评估后，三折平均 mAP50-95 下降不得超过 `0.01`，且至少 `2` 个 Fold 不低于对应锚点权重。
- 全部评估范围均为 1,127 张开发图的对应验证 Fold；本报告未读取或评估 198 张 release-validation。

## 汇总

| 模型 | 历史锚点 mAP50-95 | 候选训练后，按历史标签复评 | 变化 | 候选标签自评 mAP50-95 | 自动闸门 |
| --- | ---: | ---: | ---: | ---: | --- |
| YOLO11M | 0.18165 | 0.18135 | -0.00030 | 0.18539 | 通过 |
| YOLO26M | 0.15778 | 0.17734 | +0.01956 | 0.18166 | 通过 |

## 各 Fold 非回退核验

| 模型 | Fold | 历史锚点 mAP50-95 | 候选权重按历史标签复评 | 变化 |
| --- | ---: | ---: | ---: | ---: |
| YOLO11M | 1 | 0.14971 | 0.15926 | +0.00955 |
| YOLO11M | 2 | 0.19036 | 0.17030 | -0.02006 |
| YOLO11M | 3 | 0.20488 | 0.21449 | +0.00961 |
| YOLO26M | 1 | 0.15454 | 0.15411 | -0.00043 |
| YOLO26M | 2 | 0.11342 | 0.15725 | +0.04383 |
| YOLO26M | 3 | 0.20539 | 0.22067 | +0.01528 |

## 自动决策边界

- `0.01` 平均容忍度和“至少 2 个 Fold 非回退”是本次三折证据归档后固定的工程安全规则，不是预注册的统计显著性检验。
- 选择 YOLO11m 仅用于本次无人工 L1 候选训练，固定数据版本为 `dianjienie_ryw18_oofconsensus_v1_dev_only_r1`。
- 不能把本报告描述为“标签已经全部正确”或“新框逐个得到验证”；无人工路线只能提供可追溯的置信度、双模型 OOF 证据和自动非回退证据。
- 极稀有或零监督类别仍不能由自动闭环解决。后续若获得业务真值或新独立数据，应优先用于外部验证，而不是要求你回头做大规模人工审查。

## 证据文件

- `E:\PythonProject\优化\l1_reeval_yolo11m_fold01_retry2\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_reeval_yolo11m_fold02\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_reeval_yolo11m_fold03\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_reeval_yolo11m_oofconsensusv1_fold01\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_reeval_yolo11m_oofconsensusv1_fold02\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_reeval_yolo11m_oofconsensusv1_fold03\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_auto_anchor_nonregression_yolo11m_oofconsensusv1_fold01\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_auto_anchor_nonregression_yolo11m_oofconsensusv1_fold02\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_auto_anchor_nonregression_yolo11m_oofconsensusv1_fold03\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_reeval_yolo26m_fold01\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_reeval_yolo26m_fold02\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_reeval_yolo26m_fold03\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_reeval_yolo26m_oofconsensusv1_fold01\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_reeval_yolo26m_oofconsensusv1_fold02\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_reeval_yolo26m_oofconsensusv1_fold03\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_auto_anchor_nonregression_yolo26m_oofconsensusv1_fold01\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_auto_anchor_nonregression_yolo26m_oofconsensusv1_fold02\metrics_per_class_reeval.json`
- `E:\PythonProject\优化\l1_auto_anchor_nonregression_yolo26m_oofconsensusv1_fold03\metrics_per_class_reeval.json`
