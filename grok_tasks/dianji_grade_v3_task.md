# Role

你是实现执行者（同一会话的延续任务）。Kimi 负责规划与最终验收。工作目录 `E:\PythonProject\优化`。全程中文注释与报告。

# Objective

在 v2 基础上做**方案 4 + 方案 5**：为 1072 张等级样本派生空间分布与纹理特征，同协议重训 LightGBM 对比 v2 基线；并对最优模型输出 accuracy-coverage 拒判分析。产物写入新目录 `E:\PythonProject\优化\dianji_grade_v3\`，不得改动 v1/v2 产物与数据集视图。

# Current Evidence

- 用户决策（2026-07-23，已落盘工作日志 03:15 条目）：方案 4（空间/纹理特征）优先；方案 5（高置信 + manual_review 拒判）同批评估；端到端图像分类已被否决，不得投入。
- 视图（只读）：`E:\PythonProject\datasets\dianjinie_grade_classification_v1\`，1072 张，官方 split train/valid/test=750/161/161，`manifests\samples.csv` 含 grade、split、dup_group_id。
- v2 产物（只读）：`优化\dianji_grade_v2\`：
  - 逐框检测 `detections_1536\all_boxes.csv`（image,label,conf,cx,cy,w,h，imgsz=1536 坐标系，ryw.pt 18 类，conf≥0.25 已存，空间特征可直接从此派生，**无需重新推理**）；
  - 主特征 `features_v2b_imgsz1536_conf0p25.csv`（43 列，test LGBM acc=0.640/macroF1=0.413 为基线）；
  - 训练入口 `train_dianji_grade_lgbm.py`（train/valid/test 协议，test 只评一次）。
- 03:00 条目数字为准：LGBM test 0.640/0.413；主混对 9950↔9996；3#=46、bad=28 小样本。
- 项目 venv：`.venv_classify`（sklearn/lightgbm 已可用）；图像读取可用硬链接视图路径。

# Implementation Plan

1. **空间特征**（从 `all_boxes.csv` 派生，逐图）：3×3 分区各自的缺陷并集面积比与框数；边缘带（图边 10% 宽度环带）面积比与框数；最大单框面积比；top3 框面积和比；框面积均值/方差；最近邻框心距离均值（聚簇度）；各类在边缘带的出现标记（重点 quejiao/nodule/overfire）。坐标用图幅归一化。
2. **纹理特征**（从视图图像计算，整板灰度）：LBP 直方图（半径 1、8 邻域，归一化 10 bin 或等价）、GLCM 属性（contrast/dissimilarity/homogeneity/energy/correlation，1~2 个距离、4 方向取均值）、灰度均值/方差/梯度幅值统计。若无 skimage 则用 numpy 手写 LBP/GLCM，禁止全局 pip install。大图先缩到 ≤1024 短边再算纹理，保持全图一致口径并在报告注明。
3. **训练对比**：新脚本（如 `train_dianji_grade_v3.py`）合并 v2b conf0.25 43 列 + 空间 + 纹理特征，同 v2 协议（train 训、valid 选、test 一次）：majority / 决策树 / LightGBM，另跑「v2b 43 列不变」的重训作严格对照行，确保增量可归因于新特征。报 per-class P/R/F1、混淆矩阵、LightGBM 特征重要性 top20。
4. **拒判分析（方案 5）**：对 valid 选定模型取 predict_proba 最大类概率为置信度，在 valid 上选 3~4 个阈值使覆盖率≈90/80/70/60%，然后在 test 上一次性报告各覆盖点的准确率/macroF1（拒判样本记 manual_review）。输出 accuracy-coverage 曲线数据与结论：达到业务可用精度（如 acc≥0.85）时覆盖率多少。
5. **报告** `dianji_grade_v3\report.md`：特征清单与口径、对照表（v2 基线 vs v3 全特征 vs 仅空间 vs 仅纹理，如时间允许）、拒判曲线、结论与局限；`README.md` 写未确认状态与复现命令。所有输出带 `unconfirmed_pending_enterprise`。

# Allowed Scope

- 可新建：`E:\PythonProject\优化\dianji_grade_v3\`、新脚本 1~2 个（命名同 v2 风格）。
- 只读：数据集视图、`dianji_grade_v2\`、`dianji_grade_rule_v1\`、全部既有脚本。
- 不执行 git commit / push / reset / rebase。

# Constraints

- 遵循 AGENTS.md：--help、seed=42、dry-run、pathlib、数据只读。
- test 集只在最终选定模型评估一次；拒判阈值只能用 valid 选。
- 小样本警告（3#=46、bad=28）必须保留在报告；禁止把提升 <0.01 的数字包装成结论。

# Verification

- 新脚本 `py_compile`、`--help` 通过；端到端跑通。
- 特征行数=1072，无 NaN/inf（或说明处理）。
- 对照表含 v2 基线重训行，数字与 03:00 条目自洽（±0.01 内，超出需解释）。
- 拒判曲线数据完整（至少 4 个覆盖点），test 只在最终模型上使用。
- 无 Allowed Scope 之外改动。

# Acceptance Criteria

- `dianji_grade_v3\` 含：特征 CSV、report.md、README.md、拒判曲线数据（csv 或图）。
- 报告明确回答：① 空间/纹理特征是否带来 test 提升（对比 0.640/0.413 基线）；② 哪个特征组贡献最大（重要性/消融）；③ 拒判到多少覆盖率时 acc≥0.85（若可达）。

# Required Final Report

## Context Gathered
## Changes Made
## Verification Results
## Remaining Issues
