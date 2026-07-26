# 类别映射表：dianjienie 17 类 ↔ ryw.pt 18 类（用户 2026-07-23 确认）

- 确认方式：用户口头确认「oilstain 和 yellowstain 一样」「三类直接就可以用 ryw.pt 的按顺序就行，一致的」。
- 结论：两套类别**按 id 顺序一一对应**，dianjienie 的 17 类 = ryw.pt 18 类去掉 id 17 `fanxitong`。
- 效力：满足 AGENTS.md §6「未经明确类别映射不得合并训练」的前置条件；dianjienie 原始人工标签可按本表映射后并入 L1 汇总检测线。

| id | dianjienie 17 类名 | ryw.pt 18 类名 | 说明 |
| ---: | --- | --- | --- |
| 0 | wuzi | wuzi | 一致 |
| 1 | waterstain | waterstain | 一致 |
| 2 | oilstain | yellowstain | **用户确认同一缺陷（油渍=黄渍）** |
| 3 | nodule | nodule | 一致 |
| 4 | airhole | airhole | 一致 |
| 5 | bubble | bubble | 一致 |
| 6 | bump | bump | 一致 |
| 7 | fawu | fawu | 一致 |
| 8 | overfire | overfire | 一致 |
| 9 | NiOH | NiOH | 一致（氢氧化镍） |
| 10 | greenstain | greenstain | 一致 |
| 11 | suanwuran | suanwuran | 一致 |
| 12 | quejiao | quejiao | 一致（缺角） |
| 13 | watermark | watermark | 一致 |
| 14 | unknown_14 | youpao | 用户确认按顺序对应（油泡）；全库零实例，用户按排除法确认 14=youpao |
| 15 | unknown_15 | heibian | 用户确认按顺序对应（黑边）；**已目测 valid/881、882 实物确认** |
| 16 | unknown_16 | mijixiaokeng | 用户确认按顺序对应（密集小坑）；**已目测 train/1、302 实物确认** |
| 17 | —（不存在） | fanxitong | dianjienie 无此类（反洗铜？），无监督来源 |

## 映射后的稀少类监督量（dianjienie 原始人工标签）

| 类 | 原始框/图 | 备注 |
| --- | --- | --- |
| youpao | 0 框 / 0 图 | 无监督；另伪标签零输出 |
| heibian | 12 框 / 2 图 | 极少但存在；伪标签零输出（教师盲区） |
| mijixiaokeng | 33 框 / 2 图 | 少但存在；伪标签仅 conf<0.70 |
| fanxitong | 不存在 | 无监督来源 |
| NiOH | 7 框 / 3 图（+0309 待标注 2 图） | 伪 accepted 仅 1 框 |
| fawu | 6 框 / 4 图 | 伪 accepted 0 |

> 统计口径：单一标签布局（`train/labels` + `valid/labels`）。`dianjienie` 存在重复布局（`labels/train` ≡ `train/labels`、`labels/val` ≡ `valid/labels`），跨布局求和会重复计数（2026-07-23 已更正，首版数字作废）。

## 待决（L1 划分前）

- youpao、fanxitong 零监督：剔除出 L1 类别表 / 保留但声明不可评 / 定向采集——划分前由用户最终决定。

## youpao 零检出调查结论（2026-07-23）

1. **人工标注层**：dianjienie 标注工程从未标过一个 id14 框（约 1347 图）；yaml 中 `unknown_14` 连名字都丢失，说明该类在标注端就无实例。Dianjinie0211 为 5 类体系无法表达。
2. **教师层**：`ryw.pt` 在所有本地数据上从未输出 youpao（含 combined 3264 与早前 dianjienie 重标注，工作日志 964 行有记录）。其训练集 `nieban1100_xiangmu` 不在本地（服务器路径），无法验证该类在训练时是否有实例。
3. **两种不可区分的假设**：H1 油泡是电解镍特定/极罕见缺陷，电积镍批次中没有；H2 教师对 youpao 识别能力为零（与 NiOH 6 图→1 框、fawu 8 图→0 框的稀少类退化一致）。区分方法：拿到一张企业确认的油泡图喂 `ryw.pt` 看是否检出。
4. **语义吞并可能**：油泡与 bubble（气泡/结粒）、oilstain/yellowstain（油渍=黄渍）语义相邻，标注员可能把疑似油泡标成 bubble 或 oilstain——bubble 现有 86 张 accepted 图中可能混有真油泡。
5. **历史先例**：2026-07-14/15 的 14 类方案已因「过滤后无样本」删除过 youpao/heibian/mijixiaokeng/fanxitong 4 类（工作日志 2406/2462 行），本次为第二次遇到；建议沿用同口径：youpao/fanxitong 标为「体系保留、当前不可评」，企业考察时确认油泡/返析铜是否存在于电积镍产线。
- Dianjinie0211 为 5 类体系（waterstain/waterblock/colorstain/nodule/bubble），与本 18 类体系是另一套，未经业务确认不得并入本表。
