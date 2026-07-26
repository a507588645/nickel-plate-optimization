# Role

你是实现执行者（同一会话的扩展任务）。Kimi 负责规划与最终验收。工作目录 `E:\PythonProject\优化`。全程中文注释与报告。你此前已交付 `dianji_grade_rule.py`、`mine_dianji_grade_rules.py` 与 `dianji_grade_rule_v1\`（86 张 0309 基线），本次在此基础上扩展，**不得破坏 v1 产物**。

# Objective

把已验证等级标签的两个电积镍数据集合并去重后，用 `ryw.pt`（18 类）做**高精度**缺陷/面积特征提取，训练并横向比较等级分类模型（含 LightGBM），输出一致性与备选方案分析。全部产物写入新目录 `E:\PythonProject\优化\dianji_grade_v2\`。

# Current Evidence

## 等级真值（均由用户 2026-07-22/23 口头确认为企业/人工核实标签，可信）

- `E:\PythonProject\datasets\classify_4types_dianji\{train,valid}\{3#,9950,9996,bad}\`：train 3#=24、9950=372、9996=342、bad=21；valid 3#=10、9950=100、9996=110、bad=7。共 986 张，四类齐全。
- `E:\PythonProject\datasets\enterprise_dianji_0309_classification\images\{3#,9950}\`：3#=12、9950=74，共 86 张，无 9996/bad。语义：9950=2号板、3#=3号板。
- 历史审计（工作日志 2026-07-13）：两集合**完全重复图片 = 0**；需你复核并补做近重复检查。

## 已有资产（你上一任务已熟悉）

- 特征模式（43 列）：`grade_feature_analysis\combined_all_pseudo_v2_grade\samples.csv`；0309 的 86 张已 100% prefix-join 命中，说明 classify 系图片已进入 combined_all_pseudo_v2。**先尝试对 986 张做同样的 join**，缺失的再重新推理。
- 历史提取口径：`ryw.pt`（`E:\PythonProject\Tools\ModelAutoLabelTool\weights\yolov5\ryw.pt`），conf=0.25、iou=0.45、img-size=1024。相关脚本：`relabel_dianji_enterprise_with_ryw.py`、`analyze_dianji_grade_features_with_ryw.py`（只读参考）。
- 规则引擎：`dianji_grade_rule.py`（可 import，也可命令行）；`.venv_classify` 可用（先检查是否已装 lightgbm / sklearn）。
- 电解镍规则基线：`dianjie_grade_rule.py` 的 `DianjieGradeRule`，输出含 2号板(type 6)/3号板(type 1) 等，可直接套 18 类特征做迁移基线（用户已确认电积镍 9950=2号板、3#=3号板 与电解镍命名同名）。

## 边界

- 数据集目录只读；产物只允许写入 `dianji_grade_v2\`。
- 所有模型/规则输出必须带「未经企业确认」标记；报告如实呈现准确率，禁止调参到过拟合后宣称可用。
- 禁止全局 pip install；如 `.venv_classify` 缺 lightgbm，允许 **仅向该 venv** 安装 lightgbm（pip install lightgbm），装不了就退用 sklearn HistGradientBoosting。

# Implementation Plan

1. **合并与去重**：盘点两集合文件；完整重复（字节/哈希）+ 近重复（phash，阈值自定并在报告说明）双向检查；生成 `dianji_grade_v2\merged_manifest.csv`（image_name、source、grade、split、dup_group_id）。完整重复只留一份；近重复不删，但作为同一 group 参与划分，防止泄漏。
2. **特征提取（两版）**：
   - v2a 快版：从既有 `samples.csv` join（与 v1 同口径 1024/conf0.25），报告覆盖率。
   - v2b 精确版：对合并清单重新跑 `ryw.pt`：img-size 1536、conf 0.25、iou 0.45；先检测 GPU（torch.cuda.is_available()），有 GPU 用 GPU，没有则 CPU 分批。保存原始逐框输出（image, label, conf, cx, cy, w, h）到 `dianji_grade_v2\detections_1536\`，再从逐框结果计算 43 列特征（conf≥0.25 与 conf≥0.40 两套阈值各算一份，用于敏感性对比）。
3. **模型对比**（均以合并集为单位，划分尊重 dup_group 与原始 train/valid）：
   - 基线：多数类；`DianjieGradeRule` 直接套用（track C，注意它输出 7 态，只保留 2号板/3号板 映射，其余记为 other）；
   - 浅层决策树（沿用 `mine_dianji_grade_rules.py` 的参数纪律：max_depth≤3、min_samples_leaf≥5、seed=42）；
   - **LightGBM** 四分类（3#/9950/9996/bad）：`objective=multiclass, num_leaves≤8, max_depth≤4, min_data_in_leaf≥10, class_weight=balanced, seed=42`；分层 K 折（K=5，少数类不足则减小 K 并注明）报 per-class P/R/F1、混淆矩阵、特征重要性 top15；另用「4types train 训练 → 4types valid 独立验证」做一次 holdout 对照。
   - 每模型分别在 v2a / v2b（两个 conf 阈值）特征上各跑一遍，对比特征精度对结果的影响。
4. **一致性分析**：
   - 最优模型的预测 vs 原始标签：逐图分歧清单 `disagreements.csv`（含关键特征值），按等级对聚合。
   - 与侍掠飞论文三分类标准（合格/烫洗/不合格，`师兄论文/2021141128_cglw_侍掠飞.txt` 819-820、944-946 行）的**定性**对照：明确说明 taxonomy 不同（企业四类 vs 论文三类，无「烫洗」对应类），只做特征画像层面的对照（如哪个等级水印/水渍面积主导≈论文「烫洗」描述），禁止强行映射后报准确率。
5. **报告** `dianji_grade_v2\report.md`：数据与去重结果、特征版本、模型对比表、分歧分析、论文对照、以及「若准确率不足」的备选方案小节（更丰富空间/纹理特征、检测层提精度、端到端图像分类、双阶段特征、补充判级样本、manual_review 拒判策略），末尾附复现命令。`dianji_grade_v2\README.md` 写清未确认状态。

# Allowed Scope

- 可新建：`E:\PythonProject\优化\dianji_grade_v2\` 及其中全部内容；可新建脚本 `E:\PythonProject\优化\build_dianji_grade_v2_dataset.py`、`train_dianji_grade_lgbm.py`（或你判断更合理的命名）。
- 只读/禁改：两个数据集目录、`dianji_grade_rule_v1\`、`dianji_grade_rule.py`、`mine_dianji_grade_rules.py`、`dianjie_grade_rule.py`、其余现有脚本与文档。
- 不执行 git commit / push / reset / rebase。

# Constraints

- 遵循 `AGENTS.md`：--help、固定种子、dry-run、pathlib、数据只读、派生产物不进数据集目录。
- 重推理时不修改 `labels_with_conf\` 等既有伪标签；v2b 逐框输出只写到 `dianji_grade_v2\detections_1536\`。
- 如实报告小数：3# 合计 34 张、bad 合计 28 张，仍是小样本，报告必须保留这一警告。
- 图片与特征不得外传网络。

# Verification

- 所有新脚本 `py_compile` 与 `--help` 通过。
- merged_manifest 行数 = 去重后总数；dup 检查有明确数字（完整重复 X、近重复 Y 组）。
- v2b 逐框输出图片覆盖率 = 100%（或报告缺失清单）。
- LightGBM 训练在 v2a 与 v2b 特征上均端到端跑通，报告含 per-class 指标、混淆矩阵、特征重要性。
- 对照表含：多数类 / DianjieGradeRule / 决策树 / LightGBM 四行。
- disagreements.csv 生成且行数与混淆矩阵一致。
- 检查无 Allowed Scope 之外改动。

# Acceptance Criteria

- `dianji_grade_v2\` 含：merged_manifest.csv、特征 CSV（v2a、v2b×2 conf）、detections_1536\、模型对比与一致性 report.md、disagreements.csv、README.md。
- 回答三个业务问题的数据都在报告中：① LightGBM 在该特征上的四分类表现；② 特征分类与原标签的一致率与分歧模式；③ 与论文标准的定性符合度。
- 报告含小样本警告与未确认横幅。

# Required Final Report

## Context Gathered
## Changes Made
## Verification Results
## Remaining Issues
