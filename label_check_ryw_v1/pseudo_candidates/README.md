# pseudo_candidates（ryw.pt 候选伪标签，非真值）

- 生成：2026-07-23 20:16，ryw.pt imgsz=1024 conf≥0.15
- `labels_with_conf/` 六列（class cx cy w h confidence）
- 三态：accepted≥0.70 / uncertain 0.40~0.70 / rejected 0.15~0.40，见 summary.json
- **禁止直接用于训练**；人工审查后另行生成五列标签（AGENTS.md §6）
- 盲区类（yellowstain/fawu/NiOH/heibian/mijixiaokeng/overfire/greenstain）需人工标注
