## 2026-09-04 04:50 | parity 用户裁决回填与 overfire 追因（阈值不可修，根因为历史过标注）

| 字段 | 内容 |
| --- | --- |
| 执行者 | ZCode (GLM) |
| 任务目标 | 回填用户对 parity review.html top20 的裁决；追查用户唯一不认可的 overfire 类的根因并给出处置选项 |
| 状态 | 已完成分析；A3 切换三选一待用户裁决 |

### 用户裁决（parity review.html top20）

- #16 / #20：更认可 ONNX；#10：现役 ryw.pt 的 airhole 误检；#7：印象中 ONNX 的 overfire 不对。
- 总结论（用户原话大意）：其他大多数 ONNX 都认可，**就是 overfire 不行**。

### 追因结果（`analyze_overfire_threshold.py` → `outputs/overfire_threshold_analysis.json`）

1. **#7/#10 实际无 overfire 框**：两图业务输出 overfire=0；#7 的差异是候选 waterstain 81 vs 现役 52、wuzi +1；#10 为候选 wuzi +4/waterstain +4/bump +1（且用户已判定现役的 airhole 误检）。渲染器缺陷（overfire 与 wuzi 同为蓝色）已修复为 18 类黄金比例色相。
2. **overfire 阈值扫描不可修**：候选臂业务 overfire 121 框，按 anchor GT 判 TP 107 / FP 14；用户否定框 conf **0.70~0.88** 且多数命中 GT。扫描（0.25→0.60）：杀 85.7% GT-FP 需损 21/107 TP，且被否定框存活——逐类 conf 阈值不是正确杠杆。
3. **根因判定**：历史 18 类标签对 overfire 过度标注（#20 `dianjienie__valid__929.png`：现役 2 框 vs 候选 30 框，17 个 overfire 中 15 个"命中"GT 但用户不认可）；候选权重在开发集上训练，记忆放大该习惯。属**数据语义问题**，已立为路线 C0（overfire 标签审计，最高优先）。

### 改动文件

- `优化\parity_dianjie_v1_20260904\analyze_overfire_threshold.py`（新增）+ `outputs\overfire_threshold_analysis.json`
- `parity_dianjie_business_compare.py` 渲染色修复（18 类异色）
- 路线文档：§2 A2 追加裁决与追因结论；§4 新增 C0 overfire 审计任务

### 风险与待确认

- 「overfire 命中 GT 但用户不认可」说明 anchor/OOF 标签的 overfire 语义与业务语义不一致——此前所有以这些标签评测的 overfire 指标（含同口径复评中候选 overfire AP=0.72）都需按此重新解读。
- A3 三选一待用户：①直接切 + 登记 overfire 已知问题（C0 治理）；②切 + 业务护栏（需用户定义 overfire 规则，可先离线评 parity）；③暂缓切换先治理。
- 若走②，护栏实现位置（adapter 逐类过滤 vs detectors.py 后过滤）与对电积镍端点的同名影响需一并确认（同权重两共用）。

### 下一步

1. 用户三选一后执行 A3 或 C0。
2. 路线 B CPU 实验（B1 conf 消融）可并行，另立条目。
