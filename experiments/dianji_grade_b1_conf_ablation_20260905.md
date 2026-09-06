# B1 检测置信度消融实验记录（dianji_grade_b1，2026-09-05）

| 项 | 内容 |
| --- | --- |
| 实验 | 路线 B 步骤 B1：检测置信度 conf 0.25 → {0.15, 0.10} 单变量消融 |
| 执行者 | ZCode (GLM) |
| 状态 | **完成；结论 not_passed（两档均未过闸门），停在该变量** |
| 数据 | `archives\dianjinie_grade_classification_v1` 1072 图（企业真值，用户 2026-09-04 确认），750/161/161 seed42 |
| 协议 | 与 B0/四臂完全一致：`extract_dianji_grade_y11m640_features.py` + `train_dianji_grade_v3.py --seed 42`，v3_texture_only（43 BASE + 20 纹理 → LightGBM） |
| 权重 | 同一冻结权重本地导出 ONNX（SHA-256=`901ccf60…`，与部署一致，extraction_summary 双向核验）；单变量=conf，iou 0.45 / imgsz 640 不变 |
| B0 锚点 | test acc=`0.74534`，macroF1=`0.67970`（`优化\dianji_grade_y11m640_v1`） |

## 结果（test=161）

| 配置 | 提取框数 | test acc | test macroF1 | ΔmacroF1 vs B0 | 闸门 ≥+0.01 |
| --- | --- | --- | --- | --- | --- |
| B0 conf=0.25（锚点） | — | 0.74534 | 0.67970 | — | — |
| **B1 conf=0.15** | 14,403 | 0.75776 | 0.68618 | **+0.00648** | **未过** |
| **B1 conf=0.10** | 18,845 | 0.73292 | 0.59910 | **−0.08060** | 未过（明确回退） |

产物目录：`优化\dianji_grade_b1_conf015_v1\`、`优化\dianji_grade_b1_conf010_v1\`（features/per_box/metrics_v3_texture_only.json/reject_curve.csv/extraction_summary.json 全套）；运行日志 `优化\b1_run.log`。

## 解读

1. **conf=0.15 方向为正但未达闸门**：macroF1 +0.0065、acc +0.0124（161 张 test 上 acc +1.2pp≈2 张图，噪声量级）。按预注册闸门记 not_passed，不保留、不改变部署（部署特征提取仍 conf=0.25）。
2. **conf=0.10 明确有害**：macroF1 −0.081，低置信框噪声占优。
3. 结论：检测置信度不是当前等级准确率的有效杠杆，**B1 关闭，转 B2（分辨率消融）**。
4. 佐证背景：标签经 C0 审计确认与企业语义一致（2026-09-04），本实验的对比基础可信。

## 边界与限制

- 单 seed（42）单划分；如未来需要宣称稳定提升须多 seed。
- conf 降低会同时改变特征分布与特征数量，属"特征信息量"维度的单变量；本实验证明该维度已饱和。
- 提取 join 完整率 1072/1072，`images_with_boxes` 1065，无 plate_area fallback。
