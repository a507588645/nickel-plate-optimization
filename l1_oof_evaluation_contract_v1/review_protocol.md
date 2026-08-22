# L1 OOF 固定人工评测契约 v1

## 用途与边界

- 目的：在固定的开发集样本上核验 OOF 自动共识候选的新增框、类别冲突与正常对照，避免仅用候选五列标签自身的验证指标作结论。
- 数据范围：仅 `dianjienie_ryw18_oofconsensus_v1_dev_only_r1` 的开发集。冻结的 198 张 `release-validation` 候选不在本目录、不读取、不推理、不作为本次抽样来源。
- 这是一份审查任务契约和空白记录表，不是人工真值，也不会修改原始标签、候选标签或数据集。
- 本次冻结了 200 个物理分组、292 个框级审查单元。抽样种子为 `20260808`。

## 审查顺序

1. 先使用 `review_queue_blind.csv` 审查，避免看到模型置信度、历史标签关系或自动动作。
2. 每一行只判断该候选框：是否为真实缺陷、类别是否正确、框是否可用。填写 `human_decision`、`reviewer_id`、`reviewed_at` 和 `notes`。
3. 合法的 `human_decision`：`keep`、`delete_false_positive`、`change_class`、`box_adjust_needed`、`uncertain`、`not_reviewable`。
4. 完成首轮后，才可用 `review_queue_audit.csv` / `review_units_audit.jsonl` 查看置信度、历史关系和双模型证据，核对是否存在系统性偏差。
5. 审查结果必须导出到新的版本化结果目录；不得覆盖本契约、`labels_with_conf/`、历史人工标签或 OOF 候选五列标签。

## 通过规则

- `accepted_addition` 是主判定样本：重点统计真实新增框比例、错误类别比例与不可判比例。
- `class_conflict` 是风险样本：禁止依据本次审查把历史标签静默改类；任何改类建议都应独立记录并等待业务确认。
- `supported_control` 只用于检查双模型一致且历史支持的情形，不能替代新增框的人工判断。
- 该样本不含所有稀有类或所有候选框。若关键类别缺少可审实例，必须把该类标记为“不可评”，不能用总体比例代替逐类结论。
- 在未填写审查结果前，不得把本目录当作“已完成人工评测”的证据，也不得据此选择最终模型或最终训练视图。

## 可追溯性

- 源 OOF `summary.json` SHA-256：`7ca70e8fb7bb9850a0622cd7e7a4b080ce396fd873a382165930a12f87877716`
- 源 OOF `decisions.jsonl` SHA-256：`68fab242054460097bc81d626e0e96497cd291fb8ef6049e8578f4aa14527e2c`
- 候选开发集 `view_manifest.json` SHA-256：`712def337be52a7ee326a389bb3a6c52e81ad7d37202173836ef66a9f7861a34`
- 开发集 `source_manifest.csv` SHA-256：`9509633bff75facdeabe38a6bbeaa2ee65b5226c7c05010fbf3bf36dc67658bc`
