# L1 检测线伪标签分布审计（审查前）

- 数据集：`E:\PythonProject\datasets\combined_all_pseudo_v2`（labels_with_conf 3265 个）
- 三态框数：accepted=18028、uncertain=23602、rejected=32701（<0.15 忽略 0）
- 空图（≥0.15 无框）：27（清单 empty_images.txt）

| 类别 | 框 acc | 框 unc | 框 rej | 图 acc | 图 unc | 图 rej |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wuzi | 2246 | 5213 | 7010 | 984 | 1786 | 2062 |
| waterstain | 10829 | 11372 | 14048 | 1917 | 2131 | 2300 |
| yellowstain | 24 | 70 | 88 | 10 | 24 | 33 |
| nodule | 103 | 134 | 183 | 80 | 114 | 146 |
| airhole | 882 | 1479 | 3086 | 392 | 517 | 602 |
| bubble | 99 | 441 | 1716 | 86 | 239 | 368 |
| bump | 661 | 1620 | 2322 | 425 | 882 | 1154 |
| fawu | 0 | 2 | 7 | 0 | 2 | 7 |
| overfire | 145 | 190 | 286 | 103 | 139 | 215 |
| NiOH | 1 | 4 | 9 | 1 | 1 | 8 |
| greenstain | 62 | 205 | 470 | 32 | 75 | 163 |
| suanwuran | 31 | 59 | 68 | 19 | 45 | 58 |
| quejiao | 29 | 68 | 64 | 26 | 54 | 56 |
| watermark | 2916 | 2741 | 3214 | 1101 | 1215 | 1354 |
| youpao | 0 | 0 | 0 | 0 | 0 | 0 |
| heibian | 0 | 0 | 0 | 0 | 0 | 0 |
| mijixiaokeng | 0 | 4 | 130 | 0 | 1 | 2 |
| fanxitong | 0 | 0 | 0 | 0 | 0 | 0 |

- 用途：确定各类审查优先级（img_accepted 过低的类优先补审）；划分须在 `labels_reviewed_*` 生成后按 img 维度分组分层。
- 约束（用户 2026-07-23）：审查后标签生成前禁止划分 train/val/test；现有 `splits/` 作废待重划。
