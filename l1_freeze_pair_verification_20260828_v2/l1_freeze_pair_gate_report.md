# L1 YOLO11m Freeze10 成对验收

- 范围：仅 OOF 开发集 Fold 1；未读取冻结测试集或 `release-validation`。
- 对照 mAP50-95：`0.162499`。
- Freeze10 mAP50-95：`0.155994`。
- Freeze10 - 对照：`-0.006505`。
- 闸门状态：`not_passed_do_not_train_finaldev`。

| 检查 | 结果 |
| --- | --- |
| mAP50-95 增益至少 +0.010000 | 未通过 |
| 六个重点类别 Recall 均不下降 | 未通过 |

| 类别 | Freeze10 - 对照 Recall |
| --- | ---: |
| wuzi | -0.017544 |
| waterstain | +0.000713 |
| airhole | -0.007044 |
| bubble | +0.021127 |
| greenstain | +0.005057 |
| watermark | -0.008253 |

结论：不训练 Freeze10 全开发集版本，不导出新的部署 ONNX；保留现有已发布 ONNX。
