# L1 E5a FP Autopsy

本目录是 Fold 1 的零 GPU、高置信未匹配候选审查产物，不是人工真值、Golden Set 或模型排名报告。

- 范围：`dianjienie_ryw18_oofconsensus_v1_dev_only_r1` 的 Fold 1 验证图 376 张
- 模型：E3 P2 回收权重、E4 Copy-Paste 回收权重
- 预注册参数：见 `pre_registration_manifest.json`
- 逐类阈值：见 `derived_class_thresholds.json`，由预注册的 q90 规则在全量推理后计算
- 全量未匹配候选：`candidate_boxes_all.csv`
- 抽样候选：`candidate_boxes_sampled.csv`（本次 13 个框）
- 审查视图：`review_dataset/`；审查页由现有 `generate_box_roi_review.py` 生成

审查页状态映射：`keep` = 真缺陷，`remove` = 非缺陷，`uncertain` = 存疑。E5a 闸门只在用户完成抽查并据此统计后判定。

禁止事项：不覆盖原始图片或标签，不读取 `release-validation`，不将本目录直接用于训练，不把 E5a 结果作为 E3/E4 排名证据。
