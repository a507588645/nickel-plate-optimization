# B2 分辨率消融实验记录（dianji_grade_b2，2026-09-05）

| 项 | 内容 |
| --- | --- |
| 实验 | 路线 B 步骤 B2：推理分辨率 imgsz ∈ {640(对照), 1280, 1536} 单变量消融（conf 固定 0.25） |
| 执行者 | ZCode (GLM) |
| 状态 | **完成；结论 not_passed（1280 回退、1536 未达闸门），停在该变量** |
| 数据 | `archives\dianjinie_grade_classification_v1` 1072 图（企业真值），750/161/161 seed42 |
| 协议 | 三臂同后端：`run_b2_pt_extract.py`（显式解锁 SHA 守卫，实际 SHA 落盘）+ `train_dianji_grade_v3.py --seed 42`，v3_texture_only |
| 权重 | 最终权重 `.pt`（`…finaldev_ep083\weights\best.pt`，SHA-256=`c257578b…`；与部署 ONNX `901ccf60…` 同一训练产物，ONNX 为其导出） |
| 对照设计 | **PT@640 对照臂**隔离 ONNX→PT 后端切换；imgsz 为唯一变量（B1 已证 conf 非杠杆） |

## 结果（test=161，LightGBM v3_texture_only）

| 配置 | 提取框数 | test acc | test macroF1 | ΔmacroF1 vs 对照 | 闸门 ≥+0.01 |
| --- | --- | --- | --- | --- | --- |
| B0 ONNX@640（历史锚点） | — | 0.74534 | 0.67970 | — | — |
| **PT@640 对照** | 9,730 | 0.7453 | 0.6566 | —（基准） | — |
| **PT@1280** | 20,987 | 0.7081 | 0.6208 | **−0.0358** | 未过（回退） |
| **PT@1536** | 26,258 | 0.7516 | 0.6605 | **+0.0039** | 未过 |

产物：`优化\dianji_grade_b2_pt640_ctrl_v1\`、`dianji_grade_b2_imgsz1280_v1\`、`dianji_grade_b2_imgsz1536_v1\`（全套 features/metrics/reject_curve/extraction_summary）；日志 `优化\b2_run.log`。

## 解读

1. **后端等价性**：PT@640 对照与 B0 的 test acc 完全一致（0.7453≈0.74534），macroF1 −0.0231 集中在小类（3#=46、bad=28 主导 macroF1）——ONNX→PT 特征差异仅为小类级抖动，对照臂噪声量级约 ±0.02 macroF1。
2. **分辨率不是冻结 y11m 的杠杆**：1280 明确回退；1536 方向为正但 +0.0039 远低于闸门，且与 B0 几乎持平（+0.0002）。**ryw.pt 的 1536 增益（四臂 0.7764 vs 0.7578）不迁移到 y11m**——机制解释：y11m 在 640 训练，高分辨率推理属尺度外分布，框数 9,730→26,258（×2.7）涌入大量低质量检测，反而污染面积/计数特征。
3. 结论：**B2 关闭**。B 线本机特征工程维度（conf、imgsz）均已证饱和；**剩余杠杆为 B3（域内检测器重训，任务包已就绪待 Kaggle 额度）与 C3（追加企业判级数据）**。B4（拒判阈值/class weights）与 B5（特征子集）预期收益有限——texture vs base 对比已内置于每次训练输出（metrics_v2b_base43 vs metrics_v3_texture_only），拒判曲线已有 reject_curve.csv——如需再做须先给出明确预期。

## 边界与限制

- 单 seed 单划分；对照臂 macroF1 噪声 ±0.02 使小幅效应（<0.02）在本设计下不可分辨。
- 高分辨率未测试「配合高分辨率重训」的组合（那属 B3 类训练实验，本实验只测推理端分辨率）。
