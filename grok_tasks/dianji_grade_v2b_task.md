# Role

你是实现执行者（同一会话的计划修订）。Kimi 负责规划与最终验收。工作目录 `E:\PythonProject\优化`。全程中文注释与报告。

**修订说明（替代上一版任务 `dianji_grade_v2_task.md` 的执行顺序）**：用户明确要求先把两个等级数据集用**硬链接**合并为一个新的电积镍样本集视图并**重新划分 train/valid/test**，再在该视图上做特征提取与建模。你此前的 v1 交付（`dianji_grade_rule.py`、`mine_dianji_grade_rules.py`、`dianji_grade_rule_v1\`）保持不变。若你在被停止的上一个任务中已写出任何 `dianji_grade_v2\` 内容，先盘点并在报告中说明取舍。

# Objective

1. 在 `E:\PythonProject\datasets\` 新建电积镍等级分类硬链接视图 `dianjinie_grade_classification_v1\`（合并 `classify_4types_dianji` 与 `enterprise_dianji_0309_classification`，去重后按 70/15/15 分组分层重划分）。
2. 在该视图上用 `ryw.pt`（18 类）做高精度特征提取，训练并对比等级分类模型（含 LightGBM 四分类），输出一致性分析。派生特征与报告写入 `E:\PythonProject\优化\dianji_grade_v2\`（**不得**写进数据集目录）。

# Current Evidence

## 等级真值（用户已口头确认可信）

- `E:\PythonProject\datasets\classify_4types_dianji\{train,valid}\{3#,9950,9996,bad}\`：train 3#=24、9950=372、9996=342、bad=21；valid 3#=10、9950=100、9996=110、bad=7。共 986 张。
- `E:\PythonProject\datasets\enterprise_dianji_0309_classification\images\{3#,9950}\`：3#=12、9950=74，共 86 张。语义：9950=2号板、3#=3号板。
- 历史审计：两集合完全重复 = 0（2026-07-13，需复核）；本次必须补近重复分组。

## 项目规范要点（`E:\PythonProject\优化\AGENTS.md`）

- 数据集「只增不改」：新视图用硬链接组装（同 E: 盘，无需软链接权限；0309 企业集已有硬链接先例）；禁止移动/覆盖原图。
- 派生产物（特征、逐框检测、模型、报告）统一放 `E:\PythonProject\优化\dianji_grade_v2\`。
- 划分必须落盘为可追踪文件；**test 集只用于最终确认**，不参与调参、早停、阈值选择。
- 新脚本支持 `--help`、固定种子、dry-run；pathlib；禁止全局 pip install（允许仅向 `.venv_classify` 装 lightgbm，失败则退 sklearn HistGradientBoosting）。

## 已有资产

- 特征模式 43 列参照 `grade_feature_analysis\combined_all_pseudo_v2_grade\samples.csv`；classify 系图片多数已在其中（0309 的 86 张 100% 命中），可先 join 作为 v2a 快版对照。
- `ryw.pt`：`E:\PythonProject\Tools\ModelAutoLabelTool\weights\yolov5\ryw.pt`；历史口径 conf=0.25、iou=0.45、img-size=1024；只读参考脚本 `relabel_dianji_enterprise_with_ryw.py`、`analyze_dianji_grade_features_with_ryw.py`。
- 规则引擎 `dianji_grade_rule.py`（可 import）；电解镍规则 `dianjie_grade_rule.py` 的 `DianjieGradeRule`（7 态输出，2号板=type 6、3号板=type 1，其余记 other）做迁移基线。
- 18 类类别表见 v1 任务文件或 `dianji_grade_rule.py` 源码。

# Implementation Plan

1. **去重**：两集合盘点；完整重复（哈希）+ 近重复（phash，阈值自定并说明）；输出 dup_group_id。完整重复只保留一份（优先保留 4types 来源），近重复不删但同组。
2. **建视图** `E:\PythonProject\datasets\dianjinie_grade_classification_v1\`：
   - `images\{train,valid,test}\{3#,9950,9996,bad}\` 硬链接；
   - 划分：按 dup_group 分组 + 按等级分层，70/15/15，seed=42；**不沿用 4types 原 train/valid**（防止原划分存在近重复泄漏；原 split 记入 manifest 供对照）；
   - `manifests\samples.csv`（image_name、source_dataset、source_split、grade、split、dup_group_id、link_relpath）、`splits\{train,valid,test}.txt`、`dataset_summary.json`、`README.md`（来源、硬链接方式、划分参数、未确认声明）。
3. **特征提取**：
   - v2a：既有 `samples.csv` join（1024/conf0.25 口径），报告覆盖率；
   - v2b（主）：对视图全部图片重跑 `ryw.pt`，img-size 1536、conf 0.25、iou 0.45；先测 `torch.cuda.is_available()` 决定 GPU/CPU；逐框原始输出存 `优化\dianji_grade_v2\detections_1536\`（image,label,conf,cx,cy,w,h）；据此计算 43 列特征，conf≥0.25 与 conf≥0.40 两套。
4. **模型对比**（训练只用 train(+valid 做早停/选择)，test 只在最终模型定死后评估一次）：
   - 多数类；`DianjieGradeRule` 迁移基线；浅层决策树（max_depth≤3、min_samples_leaf≥5、seed=42）；**LightGBM** 四分类（num_leaves≤8、max_depth≤4、min_data_in_leaf≥10、class_weight=balanced、seed=42，valid 早停）。
   - 每模型在 v2a 与 v2b(两 conf) 特征上各跑一遍，对比特征精度影响；报 per-class P/R/F1、混淆矩阵、LightGBM 特征重要性 top15。
5. **一致性分析**：最优模型分歧清单 `disagreements.csv`（重点看 9950 vs 9996 可分性——若两者仅化验差异，图像特征原则上分不开，需如实呈现）；与侍掠飞论文三分类（合格/烫洗/不合格，taxonomy 不同）只做定性特征画像对照，禁止强行映射报准确率。
6. **报告与索引**：`优化\dianji_grade_v2\report.md`（含小样本警告：3# 34、bad 28；未确认横幅；「若准确率不足」备选方案小节；复现命令）、`README.md`；并在 `E:\PythonProject\优化\README.md` 数据集索引追加新视图一行。

# Allowed Scope

- 可新建：`E:\PythonProject\datasets\dianjinie_grade_classification_v1\`、`E:\PythonProject\优化\dianji_grade_v2\`、新脚本（如 `build_dianjinie_grade_view.py`、`extract_dianji_grade_features.py`、`train_dianji_grade_lgbm.py`）；可编辑 `E:\PythonProject\优化\README.md`（仅追加索引行）。
- 禁改：两个源数据集、`dianji_grade_rule_v1\`、`dianji_grade_rule.py`、`mine_dianji_grade_rules.py`、`dianjie_grade_rule.py`、其余文件。
- 不执行 git commit / push / reset / rebase。

# Constraints

- 硬链接失败（跨盘/权限）时退为复制并在 README 注明；禁止移动源文件。
- 所有模型/规则输出带「未经企业确认」标记；test 集只在报告最终一次性使用。
- 图片与特征不得外传网络。

# Verification

- 新脚本 `py_compile`、`--help` 通过。
- 视图图片数 = 去重后总数；四类在各 split 的数量与分层比例列入 `dataset_summary.json`；硬链接均指向源 inode（或注明复制）。
- v2b 逐框覆盖率 100%（或缺失清单）。
- 模型对比表含四行（多数类/DianjieGradeRule/决策树/LightGBM）× 特征版本；test 集仅最终模型一行。
- disagreements.csv 与混淆矩阵数字一致。
- 无 Allowed Scope 之外改动。

# Acceptance Criteria

- `dianjinie_grade_classification_v1\` 视图完整（images/manifests/splits/README/summary）。
- `dianji_grade_v2\` 含特征 CSV（v2a、v2b×2）、detections_1536\、report.md、disagreements.csv、README.md。
- 报告用数据回答：① LightGBM 四分类表现与最优特征版本；② 与原标签一致率及分歧模式（含 9950/9996 可分性结论）；③ 与论文标准的定性符合度。

# Required Final Report

## Context Gathered
## Changes Made
## Verification Results
## Remaining Issues
