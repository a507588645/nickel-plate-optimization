# Role

你是实现执行者。Kimi 负责规划与最终验收。在启动器指定的工作目录 `E:\PythonProject\优化` 中自主完成搜索、编辑、测试，并输出执行报告。全程使用中文注释与中文报告。

# Objective

交付电积镍「18 类缺陷种类 + 面积」规则判级的第一版骨架与候选规则产物，共三个部分：

1. 新脚本 `E:\PythonProject\优化\dianji_grade_rule.py`：配置驱动的规则判级引擎（架构仿照同目录 `dianjie_grade_rule.py`）。
2. 新脚本 `E:\PythonProject\优化\mine_dianji_grade_rules.py`：在企业 0309 已判级样本上挖掘浅层候选规则，导出规则表与报告。
3. 产物目录 `E:\PythonProject\优化\dianji_grade_rule_v1\`：候选规则表 JSON、挖掘报告、逐图判定结果。

# Current Evidence

以下事实已由 Kimi 核实，可直接引用；标注"待你核实"的请先搜索确认再动手。

## 业务背景与边界（用户 2026-07-22 口头确认，已落盘工作日志）

- 电积镍判级方向 = **按缺陷种类和面积分类**，与电解镍 `ryw.pt` 检测 + 面积硬编码（`dianjie_grade_rule.py`）同构。
- 规则基于 **18 类缺陷体系**（用户明确要求精细度）。
- 规则阈值来源：企业书面标准暂无（待实地考察），当前只能从已判级样本挖掘候选阈值，**全部标记为未经企业确认**。
- `企业0309电积镍等级规则总结.md` 已被用户声明为未验证个人推断：只可当挖掘线索，禁止当作真值引用其语义/阈值结论。

## 18 类类别表（YOLO 标签 id → 名称）

0 wuzi, 1 waterstain, 2 yellowstain, 3 nodule, 4 airhole, 5 bubble, 6 bump, 7 fawu, 8 overfire, 9 NiOH, 10 greenstain, 11 suanwuran, 12 quejiao, 13 watermark, 14 youpao, 15 heibian, 16 mijixiaokeng, 17 fanxitong

参考分组（来自电解镍规则，仅作初始假设）：A={0,1}, B={2,4,5,6,10,12,13}, C={7,8,11}, AIRHOLE={4}, NODULE=3。

## 模板文件

- `E:\PythonProject\优化\dianjie_grade_rule.py`（只读）：电解镍规则封装。架构要点：`@dataclass` 结果、规则类构造参数即阈值、`_scalculate` 用画布近似积分算并集面积、`classify()` 有序分支、`dianjie_grade_from_xyxy` 适配器、`__main__` 回归自测。新引擎应复用这些模式，但**阈值不得硬编码进类构造默认值冒充已确认**，而是来自外部 JSON 规则表。

## 等级标签数据（真值）

- 目录：`E:\PythonProject\datasets\enterprise_dianji_0309_classification\images\{3#,9950}\*.png`，目录名即等级标签。
- 历史审计数字：3# = 12 张、9950 = 74 张（共 86 张），**待你核实**实际文件数。
- 没有 9996 / bad 样本：禁止为这两类编造规则；引擎对无规则命中的样本必须输出 `manual_review`。

## 18 类检测特征来源

- 特征模式参照：`E:\PythonProject\优化\grade_feature_analysis\combined_all_pseudo_v2_grade\samples.csv`。列为：`image_name,image_path,label_path,image_width,image_height,plate_area,...,detection_count,max_confidence,total_bbox_union_image_ratio,A_ratio,B_ratio,C_ratio,Airhole_ratio,...,count__<类名>,area_ratio__<类名>（18 类各两列）`。该 CSV 覆盖 `combined_all_pseudo_v2` 的 3264 张图。
- 同目录 `lightgbm_grade_dataset.csv` 多了 `split` 等列。
- **待你核实**：0309 的 86 张图是否已按 `image_name` 进入上述 CSV（企业图文件名形如 `2911.png`，也可能带前缀）。若已覆盖，直接 join；若未覆盖，用现有提取脚本对 `enterprise_dianji_0309_classification\images` 重新提取：
  - 候选脚本（同目录，先读再用）：`analyze_dianji_grade_features_with_ryw.py`、`relabel_dianji_enterprise_with_ryw.py`、`build_lightgbm_grade_dataset.py`。
  - 权重：`E:\PythonProject\Tools\ModelAutoLabelTool\weights\yolov5\ryw.pt`（YOLOv5，约 7M 参数，CPU 可跑 86 张）。
  - 检测参数与既往一致：`conf=0.25, iou=0.45, img-size=1024`。
  - 若现有脚本依赖已失效（旧输出目录曾被移动），允许在产物目录内新写最小提取逻辑，但不许改动现有脚本。

# Implementation Plan

1. 阅读 `dianjie_grade_rule.py`、`samples.csv` 表头、0309 图片目录，核实等级样本数量与特征覆盖情况。
2. 生成/组装 86 张等级样本的 18 类特征表，按 `image_name` join 等级标签（3#/9950），输出到 `dianji_grade_rule_v1\grade_features_0309.csv`，并报告 join 覆盖率。
3. 实现 `dianji_grade_rule.py`：
   - 输入：单张图的检测框列表（label/cx/cy/w/h）或特征 dict；`plate_area`。
   - 规则表：外部 JSON（默认加载 `dianji_grade_rule_v1\candidate_rules.json`）。规则为有序列表，每条含 `name`、若干条件（特征名 + op + 阈值，op ∈ `<, <=, >, >=, ==`）、`grade`（∈ `3#, 9950, 9996, bad, manual_review`）。首条全满足即返回；无命中 → `manual_review`。
   - 输出 dataclass：grade、命中规则名、全部特征值（各类 count/area_ratio、A/B/C/Airhole 组比、并集面积比、has_nodule 等）。
   - JSON 规则表顶层必须含 `"status": "unconfirmed_pending_enterprise"`、`generated_at`、`sample_count`。
   - `--help`、`--dry-run`（只输出特征不判级）、`--rules` 指定规则表、`--features` 对特征 CSV 批量判级、`--labels` 对 YOLO 标签目录批量判级。`__main__` 放最小自测。
4. 实现 `mine_dianji_grade_rules.py`：
   - 输入 `grade_features_0309.csv`；标签 3#/9950 二分类。
   - 优先用现有环境（先查 `.venv_classify` 或其他可用 venv 是否已装 sklearn）的 `DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42, class_weight="balanced")`；若无 sklearn 且不能轻易复用，则用纯 numpy 实现「单特征阈值 + 至多两条 AND 组合」的网格搜索替代。**禁止全局 pip install**。
   - 用分层 K 折（K=5，若少数类不足则退化为 LeaveOneOut 并在报告注明）报交叉验证准确率/混淆矩阵，报告顶部必须有小样本（86 张、3# 仅 12 张）警告与「未经企业确认」横幅。
   - 把树/搜索结果导出为与引擎同 schema 的 `candidate_rules.json` + 人读 `report.md`（含每条规则的含义、支持样本数、逐规则准确率）。
5. 用引擎加载 `candidate_rules.json` 对 `grade_features_0309.csv` 批量判级，输出 `dianji_grade_rule_v1\grade_results_0309.csv`（逐图：真值、判定、命中规则、关键特征）。
6. 全部产物写入 `E:\PythonProject\优化\dianji_grade_rule_v1\`，目录内放 `README.md` 说明来源、未确认状态与复现命令。

# Allowed Scope

- 可新建：`E:\PythonProject\优化\dianji_grade_rule.py`、`E:\PythonProject\优化\mine_dianji_grade_rules.py`、`E:\PythonProject\优化\dianji_grade_rule_v1\` 及其内容。
- 只读：`dianjie_grade_rule.py`、`analyze_dianji_grade_features_with_ryw.py`、`relabel_dianji_enterprise_with_ryw.py`、`build_lightgbm_grade_dataset.py`、所有数据集目录、`企业0309电积镍等级规则总结.md`。
- 保留用户已有改动；不执行 git commit / push / reset / rebase。

# Constraints

- 遵循 `E:\PythonProject\优化\AGENTS.md`：数据只读；派生产物不得写入数据集根目录；新脚本支持 `--help`、固定随机种子、dry-run；路径用 `pathlib`；Windows 路径用原始字符串或正斜杠。
- 禁止把挖掘出的阈值表述为"企业标准"；所有输出文件带未确认标记。
- 禁止全局安装 Python 包；如需解释器，优先 `.venv_classify` 或系统 python 现有库。
- 图片/标签内容不得外传到任何网络服务。

# Verification

- `python -m py_compile dianji_grade_rule.py mine_dianji_grade_rules.py`
- 两个脚本 `--help` 正常退出。
- `python mine_dianji_grade_rules.py --help` 后端到端跑通挖掘，生成 `candidate_rules.json` 与 `report.md`。
- 引擎对 `grade_features_0309.csv` dry-run 与批量判级跑通，生成 `grade_results_0309.csv`。
- 引擎边界自测：空检测输入 → `manual_review`；构造一个必然命中首条规则的合成输入 → 返回该规则等级。
- `git status --short` 确认改动仅限 Allowed Scope。

# Acceptance Criteria

- 两个脚本编译通过、帮助可用、端到端跑通。
- `dianji_grade_rule_v1\` 内含：`grade_features_0309.csv`、`candidate_rules.json`（含 unconfirmed 标记）、`report.md`（含小样本警告与未确认横幅）、`grade_results_0309.csv`、`README.md`。
- 86 张等级样本 join 覆盖率在报告中明示；若特征需重新提取，报告提取参数与权重路径。
- 无 Allowed Scope 之外的文件改动。

# Required Final Report

返回以下标题：

## Context Gathered
## Changes Made
## Verification Results
## Remaining Issues
