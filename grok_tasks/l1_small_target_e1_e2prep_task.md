# Role

You are the implementation executor. Kimi owns planning and final acceptance. Work autonomously in the exact working directory supplied by the launcher (`E:\PythonProject\优化`).

**警告**：上一个执行者对同一任务提交过完全伪造的报告（声称完成的目录、指标和文件在磁盘上均不存在）。Kimi 会逐项对照文件系统核验你的报告：你声称创建的每个文件、每个指标数字都必须真实存在于工作区并可复算。任何无法在磁盘上核实的声明都视为任务失败。

# Objective

执行 `L1_小目标优化分阶段计划.md` 第 5 节新实验序列中的 **E1（双架构 OOF 整图 WBF 融合，本机 CPU 零训练）**，并完成 **E2（Mosaic=0 微调）的本地任务包准备（不提交 Kaggle）**。E3/E4 本轮不做。

# Current Evidence

必读文件（先读再动手）：

- `E:\PythonProject\优化\AGENTS.md`（项目操作规范，§5 日志、§6 数据安全、§8 公平训练规则为强制）
- `E:\PythonProject\优化\L1_小目标优化分阶段计划.md`（本任务的母计划；§1 硬约束、§5 E1/E2 细节、§7 交接清单）
- `E:\PythonProject\优化\工作日志.md` 摘要区 + 条目 `工作日志条目/2026-08-22_1638_L1小目标Phase1归档验证并停止切片方向.md`、`2026-08-22_1540_L1小目标Phase0完成与Phase1评测包准备.md`
- `E:\PythonProject\优化\l1_phase0_diagnostics_20260822_1455/report.md`

已验证事实：

- Phase 0 已在 CPU 上持久化 YOLO11m 三折 OOF 原始预测共 149,826 框，位于 `l1_phase0_diagnostics_20260822_1455/raw_predictions/`；生成入口 `run_l1_phase0_diagnostics.py`。
- YOLO26m 三折 OOF 权重归档于 `E:\PythonProject\models\runs\dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_oofconsensusv1_fold01/02/03`（确切目录名请先搜索确认）；本机推理环境用 `E:\PythonProject\models\.venv\Scripts\python.exe`（Python 3.10.11、Ultralytics 8.4.66、Torch 2.6.0）。
- 开发集视图：`E:\PythonProject\优化\dianjienie_ryw18_oofconsensus_v1_dev_only_r1`（1127 图/15671 框，auto_only 五列标签）；Fold 划分文件在其 `splits/` 下。
- 固定评测锚：YOLO11m Fold 1 基线 `baseline_full` mAP50-95=`0.168001`；Phase 0 逐类阈值表在 `l1_phase0_diagnostics_20260822_1455/fold01_holdout_thresholds/`。
- Phase 1 已证伪切片后处理假说，切片方向正式关闭，任何切片相关改动不在本轮范围。
- OOF Fold 1 `best.pt` SHA-256=`b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce`（E2 微调起点，归档于 `E:\PythonProject\models\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold01`）。
- 既有 Kaggle 检查点打包器：`prepare_l1_tile_interrupted_checkpoints.py`；kernel 生成器：`prepare_l1_tile_kaggle_kernel.py`（tile 专用，E2 是整图微调，请评估复用或新建生成器并记录选择理由）。

未知项（请自行搜索核验，不要猜）：YOLO26m 三折权重的确切路径与哈希、`run_l1_phase0_diagnostics.py` 的预测 JSONL schema（E1 融合须复用同一 schema）、本机 `.venv` 是否已装 `ensemble-boxes`。

# Implementation Plan

## E1 双架构 OOF 整图 WBF 融合（本机 CPU，零训练）

1. 核验 Phase 0 原始预测 schema 与三折清单对应关系；确认每张验证图与其所属 Fold 的映射。
2. 用同一环境、同一 schema 生成 YOLO26m 三折 OOF 原始预测（每图只用其未参与训练的 Fold 权重），落盘到本轮产物目录；376+375+376=1127 图全覆盖，记录每张图预测数与权重 SHA-256。
3. 实现按图 OOF 配对 WBF 融合：Fold k 的验证图只融合 `YOLO11m_foldK + YOLO26m_foldK` 两路预测，**严禁混入其他折权重**（训练泄漏）。若 `.venv` 无 `ensemble-boxes`，直接内联实现 WBF（约 50 行），不得往共享 venv 装新包。
4. WBF IoU 在 {0.25, 0.35, 0.45, 0.55} 扫描；对每个 IoU 输出整体与逐类 P/R/F1（IoU=0.50）及 mAP50-95（如 Phase 0 管线支持）；与单模型 OOF（YOLO11m 单独、YOLO26m 单独）同口径对照。
5. 产物目录 `l1_oof_dualmodel_wbf_<YYYYMMDD_HHMM>/`：融合预测、指标表、报告 `report.md`（含闸门判定：融合相对单模型 mAP50-95 或焦点类 F1 是否 ≥ +0.01）、环境清单。
6. 全程 CPU；不读取、不挂载、不评估冻结 198 张 `release-validation`。

## E2 Mosaic=0 微调任务包准备（只准备，不提交）

1. 评估复用 `prepare_l1_tile_kaggle_kernel.py` 还是新建生成器（E2 为整图 30 epoch 微调，与 tile 管线差异大），记录选择。
2. 生成 Kaggle kernel 包（slug 建议 `l1-ryw18-y11m-oof-v1-nomosaic-ft30-f01`）：固定运行时安装段（torch==2.6.0 / torchvision==0.21.0 / numpy==1.26.4 / ultralytics==8.4.66）→ 加载 OOF Fold 1 `best.pt`（先 SHA-256 校验）→ 新训练任务 `resume=false`、mosaic=0.0、30 epochs、seed 0、SGD、batch 8、imgsz 640、Fold 1 开发集；`best.pt`/`last.pt` 每轮可持续回收；训练结束后复用 `l1_tile_pipeline.py` 的 `evaluate` 模式输出四组同口径报告（整图即可，无切片 profile）。
3. 用检查点打包器模式准备含 Fold 1 `best.pt` 的私有 Dataset 本地 staging 目录（含 `dataset-metadata.json` 与哈希清单），**不上传**。
4. 本地验证：kernel 脚本语法检查、元数据校验、配置 dry-run（不联网、不训练）。

# Allowed Scope

- 可新建：本轮产物目录 `l1_oof_dualmodel_wbf_*/`、E2 kernel 包目录 `kaggle_l1_ryw18_y11m_oof_v1_nomosaic_ft30_f01*/`、必要的新脚本（`E:\PythonProject\优化` 根目录）。
- 可修改：`run_l1_phase0_diagnostics.py`（仅做向后兼容的扩展）、`工作日志.md` 摘要区、`工作日志条目/`（新增一条）、`README.md`/`CLAUDE.md`（仅登记新入口）、`L1_小目标优化分阶段计划.md`（仅写入 E1/E2 准备状态）。
- 禁止：任何 git 变更；任何 Kaggle 上传/push/提交；修改 `l1_tile_pipeline.py` 的 legacy 评测口径；改动 `dianjienie_ryw18_oofconsensus_v1_dev_only_r1` 内容；读取冻结测试集；往共享 venv 安装新包；启动本机持续监控进程。
- 保留所有无关的既有改动。

# Constraints

- 所有结论标注 `auto_only`；单 Fold 单 seed 不宣称稳定优势。
- Windows 路径用 `pathlib.Path`；新脚本支持 `--help`、固定随机种子、提供 dry-run。
- 遵循 AGENTS.md §6：派生产物只写入 `E:\PythonProject\优化` 下明确命名子目录。
- 完成后按 AGENTS.md §5 写工作日志条目并运行 `python rebuild_worklog_index.py` 重建索引。

# Verification

- E1 预测完整性：三折合计 1127 图、每图有所属 Fold 的 YOLO26m 预测文件、权重哈希与归档一致。
- E1 融合自检：随机抽 5 张图人工核对融合只含同 Fold 两路来源；打印融合前后框数。
- E1 指标脚本以 dry-run/小样本先行验证再全量运行。
- E2 包：`python -m py_compile` 通过；kernel-metadata 校验；生成器 dry-run 输出与预期配置一致。
- 结束前 `git status --short` 检查无意外改动（只读检查，不做任何 git 写操作）。

# Acceptance Criteria

- `l1_oof_dualmodel_wbf_*/report.md` 含四个 IoU 档的整体+逐类指标、与单模型 OOF 的同口径对照、明确闸门判定（pass/keep 或 reject）。
- E2 kernel 包与 Dataset staging 目录齐备且本地校验通过，处于"待用户确认后提交"状态。
- 工作日志新条目 + 索引重建完成；`工作日志.md` 摘要区与本计划书状态同步。

# Required Final Report

Return these headings:

## Context Gathered
## Changes Made
## Verification Results
## Remaining Issues
