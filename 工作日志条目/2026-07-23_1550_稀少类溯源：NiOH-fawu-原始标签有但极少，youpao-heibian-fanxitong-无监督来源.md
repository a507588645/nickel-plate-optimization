## 2026-07-23 15:50 | 稀少类溯源：NiOH/fawu 原始标签有但极少，youpao/heibian/fanxitong 无监督来源

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi |
| 任务目标 | 核实原始标注数据集与 0309 待标注集中稀少类的真实存在情况 |
| 状态 | 已完成 |

- 原始 `dianjienie`（17 类人工标签）实测：NiOH=14 框/6 图、fawu=12 框/8 图、unknown_15=12 框/2 图、unknown_16=66 框/4 图、unknown_14=0；主力类 waterstain 14858 框/1526 图。**稀少类监督存在于原始标签，审查时应携带原始人工标签而非只依赖伪标签**。
- `Dianjinie0211`（5 类）：体系不同，不可能含 NiOH/heibian；nodule 仅 10 框/8 图。
- `enterprise_dianji_0309_detection_unlabeled/images` 子目录：缺角 12、烧板 12、电击板 7、边部结瘤 4、**氢氧化镍(NiOH) 2**；**无黑边(heibian) 目录**。
- **youpao / heibian / fanxitong 三个类名仅存在于 `ryw.pt` 训练集（nieban1100_xiangmu 18 类）**，当前所有本地数据集与 0309 均无监督来源、伪标签零输出——L1 划分前须决定：剔除/并入近类/定向采集。
- **类名冲突**：`dianjienie` id2=`oilstain` vs `ryw.pt` 18 类 id2=`yellowstain`；`dianjienie` id14-16 为 unknown_14/15/16（与 18 类的 youpao/heibian/mijixiaokeng 是否对应未确认）。合并原始标签前必须先做经业务确认的类别映射（AGENTS.md §6）。
- 教师盲区实锤：dianjienie 原有 NiOH 6 图，ryw.pt 伪标签 accepted 仅 1 框/1 图；fawu 原 8 图、伪 accepted 0——稀少类伪标签不可靠，不能只审伪标签。
