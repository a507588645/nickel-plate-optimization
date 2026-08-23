# L1 E2 成对 Mosaic 校验报告

- 配对 ID：`l1_e2_mosaicpair_f01_ft30_v1`
- 状态：`not_passed_do_not_expand`
- 范围：仅开发集 Fold 1、`auto_only`；未使用冻结验证集。
- 对照 Mosaic=1 mAP50-95：`0.162465`
- 候选 Mosaic=0 mAP50-95：`0.155948`
- 候选 - 对照：`-0.006518`（门槛 `+0.010000`，未通过）
- 候选 - 固定基线：`-0.012054`（门槛 `+0.010000`，未通过）
- 焦点类 Recall 不劣化：未通过。

| 类别 | Mosaic=0 − Mosaic=1 Recall@0.25/IoU0.5 |
| --- | ---: |
| wuzi | -0.031579 |
| waterstain | +0.000471 |
| airhole | -0.016502 |
| bubble | -0.021127 |
| greenstain | +0.020202 |
| watermark | +0.050894 |

结论：Do not expand E2 to Fold 2/3; retain the result as a single-Fold auto_only ablation.
