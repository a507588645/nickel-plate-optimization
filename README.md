# 镍板缺陷模型优化 — 文档与产物索引

本目录是**研究/实验文档与辅助脚本**的中心，不是 WebAPI 主仓库。  
检测与分类的真实数据、以及现场服务代码在 `E:\PythonProject` 其他路径（见下文「关联路径」）。

**冲突时怎么信：** 磁盘上的图片/标签/`data.yaml` → 可复现训练产物 → `工作日志.md` 近期结论 → 本目录现行规范。  
历史文档中的 18 类示例、固定性能数字、部署选型**不得**直接当已验证事实。

---

## 1. 我该先看哪个？

| 你是谁 / 你要做什么 | 先读 |
| --- | --- |
| 任意 Agent 开工 | `AGENTS.md` → `工作日志.md`（摘要 + 最近条目）→ 本 `README.md` |
| 看当前应该做什么 | `当前工作计划.md`（项目主控：优先级、资源闸道、停止条件） |
| Claude 开工 | 同上，细则见 `CLAUDE.md`（规则继承 `AGENTS.md`） |
| 人工做检测模型优化 | `镍板缺陷模型优化操作手册.md` 第 1–4 章起 |
| 查「上次做到哪」 | **只查** `工作日志.md` |
| 记一次正式训练 | 复制 `模型优化实验记录模板.md`，并在工作日志写摘要 |
| 改 WebAPI / 部署代码 | 仓库 `JinchuanPythonWebapi`，分支约定见 §5；本目录只提供规范与等级分类实验 |

---

## 1.5 三条正式工作线（用户 2026-07-23 定）

| 线 | 目标 | 当前状态 | 关键约束 |
| --- | --- | --- | --- |
| **L1 检测模型优化** | 以 `dianjienie` 历史标签锚点 `dianjienie_ryw18_anchor_v1` 的 OOF 共识候选训练 18 类 YOLO11m | 1325 张、18232 框已完成去重、分组分层重划；OOF 双模型共识形成 1,127 张开发集候选视图与 245 个 accepted 新框，历史标签未覆盖。YOLO11m 主候选（83 epoch）和 YOLO26m 辅助候选（174 epoch）均已归档；预选 YOLO11m 的唯一 198 图历史确认得到 P/R/mAP50/mAP50-95=`0.42582/0.38473/0.35959/0.23147`。G4 高分辨率 YOLO11m、OOF 候选开发集、Fold 1 的 1280/batch 4 试验已完成，最佳 mAP50-95=`0.16264`，低于同 Fold 640 的 `0.16382`，不扩展 Fold 2/3。报告：`l1_auto_gate_oofconsensusv1_20260809/report.md` | 固定历史集只用于已选 YOLO11m 的一次确认，YOLO26m 不读取它；候选五列视图不是人工真值；无人工路线只提供 OOF 证据和自动安全闸门，不宣称逐框正确；不启动本机 medium 训练 |
| **L2 电积镍分类等级** | 缺陷种类+面积规则/LightGBM 判级（3#/9950/9996/bad） | `dianjinie_grade_classification_v1`（1072）+ `dianji_grade_v2/v3` 历史产物；YOLO11m@640 新提取 `dianji_grade_y11m640_v1`：63 特征、test acc=0.7453、macroF1=0.6797、round-trip 通过 | 全部 `unconfirmed_pending_enterprise`；新结果待企业书面标准 + 批次元数据 + Kimi 复核；历史 0.7764/0.6631 仅作参考 |
| **L3 电解镍分类硬编码优化** | `DianjieGradeRule` 阈值/规则/决策层优化（检测权重 `ryw.pt` 不变） | 规则已封装（`dianjie_grade_rule.py`）；电积镍迁移已实锤不可用（acc=0.02） | 商业约束：权重不变，优化只落加载代码/阈值/规则/决策层；改动须在固定人工评测集做前后对比（AGENTS.md §8） |

---

## 2. 文档地图（按可信度与用途）

### A. 现行入口与规则（优先）

| 文件 | 用途 | 备注 |
| --- | --- | --- |
| `README.md` | 总索引（本文件） | 找文档从这里进 |
| `AGENTS.md` | 全体 Agent 强制规则：数据安全、公平实验、工作日志 | 与实现冲突时以数据与产物为准 |
| `CLAUDE.md` | Claude 入口：开始/结束任务、关键事实摘要 | 不替代 `AGENTS.md` |
| `工作日志.md` | 跨会话进度与交接 | **只追加**；最新在上 |
| `工作日志条目模板.md` | 日志单条复制模板 | 空白格式 |
| `当前工作计划.md` | 当前项目主控计划 | 交付优先；高分辨率是非阻塞研究闸道 |
| `研究生毕业与论文路线图.md` | 毕业研究与论文旁路计划 | 与 L1 交付隔离；先完成研究协议、人工金标准和外部泛化证据 |
| `research_thesis/literature/20260929_可行方案检索与论文路线.md` | 论文方向广度检索、13 项来源与12个候选、师兄工作去重 | 结合较新 L2 企业标签及 B3clean 证据；A 判级/B 可信检测/C 多光路线待导师约束，不授权训练 |
| `跨电脑工作区与数据集同步规范.md` | 多电脑协作、逻辑数据集 ID 和本机路径映射 | 新电脑初始化与数据差异处理按此执行 |
| `镍板缺陷模型优化操作手册.md` | 检测优化主执行规范（审计→训练→验证→部署） | 现行操作主文档 |
| `检测切换与等级提升路线_20260904.md` | WebAPI 检测器切换与等级提升跨 Agent 执行路线：A 电解镍 parity 切换 / B 电积镍等级提升 / C 数据线 | 2026-09-04 建立；含当日同口径复评结论与用户裁决（电积镍等级=企业真值） |
| `模型优化实验记录模板.md` | 单次训练/对比实验记录 | 正式实验必填 |

### B. 专题规范（按需读）

| 文件 | 何时读 |
| --- | --- |
| `三线线扫三光采集与数据规范.md` | 新三光硬件、标定、命名、通道语义 |
| `无人工伪标签闭环方案.md` | 无人工标注时的伪标签/自训练流程 |
| `YOLO冻结与迁移学习指南.md` | 冻结 backbone、续训、权重兼容（勿跨架构硬 load） |
| `旧数据与新三光数据混合训练指南.md` | 同镍种旧单光 + 新三光混合，防遗忘 |
| `L1_Kaggle高分辨率验证计划.md` | L1-hr Kaggle-only 执行计划：本机不跑 medium 训练，优先验证 `imgsz=1280` 是否改善小目标 mAP |
| `L1_小目标优化分阶段计划.md` | L1 小目标研究计划：切片路线、E2 Mosaic=0 与 E1b 严格原始锚点扫描均已停止；E1 WBF 保留，E3 P2 与 E4 Copy-Paste Fold 1 正在两个 GPU 槽并行筛选，仍受独立自动闸门约束 |
| `YOLO_mAP5095_后续优化任务书_20260910.md` | 已试方法去重、合成框确认的评测匹配口径差异、AP75/误差预算与定位/真实补采/ROI 二阶段条件式方案；不自动启动训练或重启停止分支 |
| `config/workspace.example.json` / `config/dataset_catalog.json` | 本机路径映射模板与逻辑数据集校验规则 | 真实 `workspace.local.json` 不提交；用 `validate_workspace.py` 验证 |

### C. 已完成实验记录（专项，非通用规范）

| 文件 / 目录 | 内容 |
| --- | --- |
| `伪标签重标注报告_v1.md` | 两数据集伪标签 v1 结果报告（含历史 18 类教师设定，解读时注意与当前 YAML 差异） |
| `伪标签审查流程记录_v1.md` | 审查 UI/进度与流程说明 |
| `visualizations/` | 伪标签抽检图与 HTML（`pseudo_v1_*.html` 等） |
| `分类模型优化工作总结.md` | 本次电解镍/电积镍等级分类优化工作的完整过程、发现与产物索引 |
| `电积镍等级划分标准调研与分类解决方案.md` | 基于 GB/T 6516-2025 的电积镍牌号、外观质量边界与双层判定方案 |
| `企业0309电积镍等级规则总结.md` | ⚠️ 未验证：基于权重特征的个人总结（用户 2026-07-22 声明不得作真值）；电积镍判级方向已改为「缺陷种类+面积规则」 |
| `类别映射_dianjienie17_to_ryw18.md` | dianjienie 17 类 ↔ ryw.pt 18 类按序映射（用户 2026-07-23 确认：oilstain=yellowstain、unknown_14/15/16=youpao/heibian/mijixiaokeng、无 fanxitong；15/16 已经目测实物确认） |
| `experiments/README.md` | 正式实验卡的索引、状态、证据等级和比较边界 |
| `dianjienie_ryw18_anchor_v1_训练交接.md` | `dianjienie` 18 类历史标签锚点的固定三折训练入口、统一参数和后续自动共识路线 |
| `experiments/dianjienie_ryw18_anchor_v1_cv3_summary.md` | RYW18 anchor v1 的 YOLO11m/YOLO26m 三折开发集统一复评汇总与 OOF 候选边界 |
| `experiments/dianjienie_ryw18_oofconsensus_v1_cv3_summary.md` | RYW18 OOF 共识候选 v1 的六个 run 归档、三折统一复评汇总与无人工候选训练决策 |
| `experiments/dianjienie_ryw18_oofconsensus_v1_yolo11m_hr1280_fold01_kaggle.md` | L1-hr 1280 Fold 1 的 Kaggle 归档、同 Fold 640 对比与“不扩展三折”结论 |
| `experiments/dianjienie_ryw18_oofconsensus_v1_yolo11m_tile1024o25_focus6_mix_fold01_kaggle.md` | L1 小目标切片 Fold 1 的中断训练证据、数据边界、独立原图映射评测口径与恢复入口 |
| `experiments/dianjienie_ryw18_oofconsensus_v1_yolo11m_tile1024o25_phase1_wbfedge_fold01_kaggle.md` | L1 小目标 Phase 1 后处理评测的完整归档校验、逐框归因与停止切片路线结论 |
| `experiments/dianjienie_ryw18_oofconsensus_v1_yolo11m_e2_mosaicpair_fold01_kaggle.md` | L1 E2 Mosaic 成对消融实验卡：完整成对验收后为 `not_passed_do_not_expand`，不扩展 Fold 2/3 |
| `experiments/dianjienie_ryw18_oofconsensus_v1_e1b_perclass_confscan_20260823.md` | L1 E1b 逐类别阈值扫描：局部运行对照有召回候选，但相对固定原始 E1 锚点为 `not_passed_strict_fixed_anchor` |
| `l1_auto_gate_oofconsensusv1_20260809/report.md` | OOF 候选权重按未改动历史标签复评的自动非回退闸门、模型选择和边界 |
| `类别体系清单_v1.md` | 全部数据集类别体系总览（ryw.pt 18 类 / dianjienie 17 类 / Dianjinie0211 与 slf.pt 5 类 / 等级分类 / 0309 缺陷目录）+ 缺口与处置候选 |
| `ryw_pt模型身份与电积镍检测结果说明.md` | 明确 `ryw.pt` 是 YOLOv5 检测模型、`ryw.pth` 是 ResNet 分类模型，并汇总两个电积镍分类集的检测与硬编码结果 |
| `原项目YOLOv5训练配置与mAP口径说明.md` | `ryw.pt`/项目 `train.py` 真实训练配置；与 YOLO11 默认差异；**项目 `val.py` 将 mAP50/mAP50-95 误标为 mAP@0.1 与 mAP@0.1:0.55** |
| `JinchuanPythonWebapi-算法解析.md` | WebAPI 六大算法源码解析底稿（**描述 `308` 分支**，HEAD `5cb215a`；2026-08-30 已对照源码与权重逐条核实，8 条红旗全部成立）；后续算法优化融合到 `优化分支` 时的基准参照，引用行号前先确认目标分支 |

### D. 历史 / 仅供背景（默认不执行）

| 文件 | 说明 |
| --- | --- |
| `旧方案与现行优化桥接索引.md` | **旧方案的唯一现行入口**：先按此处的依据优先级、冲突裁决和触发点使用旧方案，不改变 L1/L2/L3 |
| `Instruction.md` | 早期对话约束与选型摘要；部分目标/类别数已过时 |
| `nickel_defect_masterplan.md` | 已归档的原始总方案正文；通过桥接索引使用，不直接作为现行结论 |
| `nickel_defect_masterplan.svg` / `architecture_complete.svg` | 已归档的两张原始架构图；只读背景，不据图中状态判断完成度 |
| `nickel_defect_system_execution_doc.md` | 早期系统执行方案 |
| `nickel_defect_fullstack_guide.pdf` | 早期 AI 生成全栈方案；**示例数字不可当现状** |

---

## 3. 脚本与代码（本目录）

根目录脚本多为**伪标签审查**与**等级分类**辅助，不是 Ultralytics 训练主入口。训练命令以操作手册与外部训练环境为准。

| 文件 | 角色 |
| --- | --- |
| `generate_pseudo_label_review.py` | 生成伪标签审查页 |
| `generate_comprehensive_review.py` | 综合审查页 |
| `generate_status_visualizations.py` | 状态可视化 |
| `visualize_pseudo_labels.py` / `visualize_pseudo_labels_accepted.py` | 伪标签可视化 |
| `apply_review_results.py` | 应用审查结果（写版本化标签，勿覆盖人工原标） |
| `build_grade_dataset.py` | 从检测结果构建等级特征集 |
| `build_enterprise_nickel_datasets.py` | 从两批企业原图构建无版本号硬链接数据集，清单写在当前执行目录 |
| `audit_enterprise_dataset_duplicates.py` | 用文件大小预筛选 + SHA-256 审计企业集与原始集的完全重复图片 |
| `prepare_label_incremental_merge.py` | 整体迁移 Pictures/label、SHA-256 去重、与旧汇总集增量查重并构建新版本；支持 `prepare/finalize` 和 `--dry-run` |
| `build_flat_pseudo_label_dataset.py` | 多来源 SHA-256 去重并硬链接为无划分的扁平 YOLO 汇总输入 |
| `finalize_flat_yolo_dataset.py` | 校验扁平 YOLO 图片/标签、写入 data.yaml、classes.txt 和审计报告 |
| `generate_flat_yolo_review.py` | 为扁平 YOLO 数据生成 Canvas 可视化审查页；支持多视图（原图/增强/强对比）切换，框始终来自 `labels_with_conf` |
| `review_enhance_utils.py` | 审查用显示增强：Lab/HSV/灰阶/CLAHE、top-hat/black-hat、边缘、Gamma（几何不变） |
| `prepare_review_enhance_sample.py` | 低置信度分层抽样 + 多颜色空间视图 + 多视图审查页（**不重跑模型**） |
| `extend_review_views.py` | 为已有抽样集补生成 Lab/HSV/形态学等视图并刷新 HTML |
| `generate_box_roi_review.py` | **按框审查**：低 conf 框 ROI 放大 + 8 面板 montage + 结构分启发式（全图仍看不清时用） |
| `summarize_review_visibility.py` | 汇总审查 JSONL 的可见性提升率（原图不可审→增强救回） |
| `generate_paired_yolo_review.py` | 原图 vs 增强图左右对照审查页（历史/对照实验用） |
| `prepare_enhance_annotation_test.py` | 原图伪标签 vs 增强图重推理对照测试（域迁移实验，非日常审查主路径） |
| `finalize_confidence_pseudo_labels.py` | 验证六列置信度标签并生成 accepted/uncertain/rejected 三状态清单 |
| `organize_grade_datasets.py` | 统一整理电解镍/电积镍等级分类软连接数据集 |
| `build_dianji_enterprise_0309_dataset.py` | 审计并构建独立企业 0309 数据集及包含旧数据的 v2 软连接视图 |
| `merge_dianji_feature_datasets.py` | 合并同模式的旧/新电积镍检测特征和软连接树 |
| `relabel_dianji_enterprise_with_ryw.py` | 用电解镍 `ryw.pt` 重标企业 0309 图片并生成七类规则辅助结果 |
| `analyze_dianji_grade_features_with_ryw.py` | 回连版本化 `ryw.pt` 六列标签，统计电积镍各原等级缺陷类别/框面积，并复现电解镍硬编码七状态 |
| `build_dianji_enterprise_grade_v3.py` | 明确目录等级固定，其余缺陷目录用原电积镍四分类模型判级 |
| `build_dianjie_multiclass_dataset.py` | 基于 308 规则把电解镍 2 类扩展为多类软连接 folder |
| `train_resnet_folder_classifier.py` | 在 folder 分类数据上微调 ResNet18 |
| `train_cnn_folder_classifier.py` | 轻量 CNN folder 分类基线 |
| `infer_grade_classifier.py` | 等级推理 |
| `grade_classifier_api.py` | 检测特征 LightGBM 分类 API |
| `grade_classifier_unified_api_v2.py` | 统一 API：电解镍规则 / 电积镍检测+图像特征+LightGBM |
| `dianjie_grade_rule.py` | 电解镍原项目多类规则封装 |
| `dianji_grade_rule.py` / `mine_dianji_grade_rules.py` | 电积镍 18 类面积规则引擎与 0309 v1 候选规则挖掘 |
| `build_dianjinie_grade_view.py` | 合并 4types+0309，去重后硬链接视图 + 70/15/15 分组分层 |
| `build_detection_union_dataset.py` | L1 去重汇总数据集构建（五源+企业集，原标签，SHA-256 去重，不划分） |
| `join_union_grade_labels.py` | 汇总集等级标签哈希接入（4types+0309 → samples.csv 增 grade 列） |
| `check_union_labels_with_ryw.py` | ryw.pt 对 union 原标签做一致性检查（召回网格/混淆/额外预测/TOP50） |
| `generate_union_pseudo_candidates.py` | 无标签图六列候选标签生成（版本化，禁直接训练） |
| `build_label_check_review.py` | 标签检查 TOP50 静态复核页 |
| `build_label_review_views.py` | 双轨交互审查视图构建（GT 轨 + 候选轨，复用旧审查流程；已被双源页取代） |
| `build_dual_review.py` | **现行**单页双源对比审查页（GT 绿框无 conf + 候选红框真实 conf，逐框判决） |
| `apply_label_review_decisions.py` | 双轨审查应用：GT 留删 + 候选补入 → union `labels_reviewed_v1` |
| `audit_union_label_quality.py` | **L1 标签优化入口**：只读审计原标签的结构异常、类别稀缺、完全重复，并生成不含模型裁决的人工复核优先级；视觉近重复仅在显式 `--near-duplicates` 时运行 |
| `build_manual_reannotation_review.py` | **L1 人工重标页**：以原标签为可编辑草稿，支持新增框、删框、移动/缩放、改类别、备注和完成状态；不展示旧模型候选 |
| `apply_manual_reannotation.py` | 校验人工重标 JSONL 并写入 `优化\manual_reannotation_v1\labels_human_reviewed_v1`，不覆盖 union 原始 `labels/`；支持 scope + 完成状态强制校验 |
| `build_dianjienie_ryw18_anchor.py` | 从原 `dianjienie` 构建只增不改的 RYW18 历史标签锚点：结构校验、精确/近重复分组、冻结测试集与 3 折划分 |
| `train_dianjienie_ryw18_anchor.py` | 训练一个 RYW18 锚点 Fold 的 YOLO11m 或 YOLO26m；训练前校验类别、全部标签和无交集划分，并自动归档运行时清单、配置、环境和 run README |
| `build_kaggle_dianjienie_ryw18_dataset.py` | **灾备回退**：完整图片 Kaggle 上传视图；既有私有图片集不可用时才可加 `--allow-redundant-image-upload --publish` 重传约 21 GiB 图片 |
| `kaggle_dianjienie_ryw18_runner.py` | **灾备回退**：配套全量图片 Dataset 的 Kaggle GPU 单 Fold 运行器 |
| `build_kaggle_dianjienie_ryw18_config_dataset.py` | **当前 Kaggle 上传入口**：复用私有 `lly3323w/dianjienie` 图片集，全量核验后只上传标签、固定 Fold 与脚本配置包 |
| `kaggle_dianjienie_ryw18_existing_data_runner.py` | **当前 Kaggle 训练入口**：同时挂载图片集和配置包，链接 1325 张已核验源图后训练单 Fold |
| `kaggle_dianjienie_ryw18_archive_bootstrap.py` | **Kaggle Notebook 直接入口**：兼容 Kaggle 自动展开配置归档，校验清单后在 working 目录恢复配置，再调用现有图片训练运行器 |
| `kaggle_l1_ryw18_anchorv1_preflight/l1-ryw18-anchor-v1-preflight.py` | 已完成的 Kaggle 实测挂载预检：动态发现自动展开后的图片/配置 Dataset，再以 `--dry-run` 校验锚点与 Fold |
| `kaggle_l1_ryw18_yolo11m_anchorv1_fold01/l1-ryw18-yolo11m-anchorv1-fold01.py` | 初版 L1 YOLO11m Fold 1 私有 GPU Kernel 定义；因 9 小时请求上限在 epoch 250 后取消，仅保留失败证据，不是有效 run |
| `kaggle_l1_ryw18_yolo11m_anchorv1_fold01_retry1/l1-ryw18-yolo11m-anchorv1-fold01-retry1.py` | L1 YOLO11m Fold 1 独立重试 Kernel；已完成 ZIP CRC、权重哈希、固定清单校验与本机固定 val 复评，归档 run 位于 `E:\PythonProject\models\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold01` |
| `kaggle_l1_ryw18_yolo26m_anchorv1_fold01/l1-ryw18-yolo26m-anchorv1-fold01.py` | 已完成的 L1 YOLO26m Fold 1 Kernel：结果 ZIP、权重和固定清单已校验并归档；同口径本机 val 复评见 `l1_reeval_yolo26m_fold01/metrics_per_class_reeval.json` |
| `kaggle_l1_ryw18_yolo11m_anchorv1_fold02/l1-ryw18-yolo11m-anchorv1-fold02.py` | 已完成并归档的 L1 YOLO11m Fold 2 Kernel：ZIP、权重和固定清单已校验；同口径本机 val 复评见 `l1_reeval_yolo11m_fold02/metrics_per_class_reeval.json` |
| `kaggle_l1_ryw18_yolo26m_anchorv1_fold02/l1-ryw18-yolo26m-anchorv1-fold02.py` | 已归档并完成本机固定 val 复评的 L1 YOLO26m Fold 2 Kernel：Kaggle best mAP50/mAP50-95=`0.25637/0.16023`，本机统一复评=`0.20602/0.11342`；固定 `train/val/test=752/375/198` |
| `kaggle_l1_ryw18_yolo11m_anchorv1_fold03/l1-ryw18-yolo11m-anchorv1-fold03.py` | 已归档并完成本机固定 val 复评的 L1 YOLO11m Fold 3 Kernel：Kaggle best mAP50/mAP50-95=`0.32951/0.20305`，本机统一复评=`0.33382/0.20488`；固定 `train/val/test=751/376/198` |
| `kaggle_l1_ryw18_yolo26m_anchorv1_fold03/l1-ryw18-yolo26m-anchorv1-fold03.py` | 已归档并完成本机固定 val 复评的 L1 YOLO26m Fold 3 Kernel：Kaggle best mAP50/mAP50-95=`0.32647/0.20398`，本机统一复评=`0.32710/0.20539`；固定 `train/val/test=751/376/198` |
| `collect_l1_kaggle_run.py` | L1 完成后的 Kaggle 归档入口：仅接受 COMPLETE 状态，下载结果 ZIP，执行 CRC/必需文件/路径安全校验后才安装至 `models/runs`；候选视图必须传 `--development-only`，最终全开发集发布模型另加 `--final-release` 以强制核验固定 epoch/发布证据，普通 Fold 不得传该参数 |
| `reeval_l1_yolo_run.py` | 对已归档 L1 权重在固定开发 Fold 做本机标准复评；候选视图传 `--development-only` 时不读取冻结测试集，并兼容相对 Fold 路径 |
| `generate_l1_oof_consensus.py` | L1 三折开发集 OOF 自动共识：每图只用未见 Fold 的 YOLO11m/YOLO26m 预测，`ryw.pt` 仅作教师证据；输出可追溯六列候选与历史标签加 accepted 框的五列候选视图，不读冻结测试集 |
| `build_l1_oof_consensus_dev_view.py` | 从 OOF 五列候选生成仅含 1,127 张开发图的硬链接训练视图；显式拒绝读取、链接或写出冻结测试集 |
| `l1_box_size_diag.py` | L1 锚点标签框尺寸诊断；默认从三折 `val.txt` 并集取 1,127 张开发图，不读取冻结 test，输出 `val_letterbox` 主口径与训练方形 resize 参考报告 |
| `prepare_l1_oofconsensus_kaggle_kernel.py` | 生成私有 Kaggle 无 GPU 预检或单 Fold GPU Kernel；固定挂载旧图片集和 OOF 候选配置集，训练时强制 `--development-only` |
| `prepare_l1_hr_kaggle_kernel.py` | 生成 L1-hr 的独立 Kaggle 预检或 GPU Kernel：YOLO11m、OOF 候选开发集、Fold 1，首选 `imgsz=1280`；不读取冻结历史测试集 |
| `l1_tile_pipeline.py` | L1 小目标切片研究流水线：创建派生混合整图/切片视图、训练后或中断检查点评测的原图坐标回映射；默认保留 legacy 全局 NMS，并可研究性比较逐类阈值 NMS、WBF、边缘过滤及逐框归因；不修改源图或源标签 |
| `prepare_l1_tile_baseline_weights.py` | 打包固定 Fold 1 640 OOF 基线权重为私有 Kaggle Dataset，供切片评测统一对照 |
| `prepare_l1_tile_interrupted_checkpoints.py` | 打包已回收的 L1 切片中断检查点为私有 Kaggle Dataset，只供独立开发集评测或同配置分段续训 |
| `prepare_l1_tile_kaggle_kernel.py` | 生成 L1 切片 CPU 预检、GPU 训练或中断检查点独立评测 Kernel；可嵌入 Phase 0 阈值 JSON 并转发 WBF/边缘过滤/逐框审计参数，兼容 Kaggle 自动展开的扁平配置包 |
| `run_l1_phase0_diagnostics.py` | CPU-only L1 小目标 Phase 0：三折 OOF 原始预测持久化、逐类阈值扫描、PR 图与 G4b 逐框归因缺口审计；不读冻结集 |
| `run_l1_oof_dualmodel_wbf.py` | CPU-only L1 E1：严格按图配对同 Fold 的 YOLO11m/YOLO26m 原始 OOF 预测，补推理 YOLO26m、扫描固定四个 WBF IoU，并输出指标、哈希与无泄漏审计；不读冻结集 |
| `prepare_l1_nomosaic_kaggle_kernel.py` | E2 的历史单侧 Mosaic=0 staging，仅用于追溯；因缺少匹配对照，不得提交或用于结论 |
| `prepare_l1_e2_mosaic_pair_kaggle_kernels.py` | **现行 E2 入口**：本地生成共享检查点审计镜像、Mosaic=1 对照和 Mosaic=0 候选的两个私有 Kaggle Kernel；固定共享契约，生成器本身不上传、不提交 |
| `verify_l1_e2_mosaic_pair.py` | E2 完成后成对校验两个结果 ZIP：CRC、路径、共享契约、运行时、清单、权重、评测锚与逐类 Recall 后才输出自动闸门 |
| `test_verify_l1_e2_mosaic_pair.py` | E2 成对归档校验器的合成端到端单元测试，不读取真实数据或权重 |
| `collect_l1_e2_minimal_evidence.py` | 从包含无关大型工作区的 Kaggle E2 完整结果 ZIP 按中央目录范围读取必需成员，逐成员 CRC 校验后生成最小可验收证据 ZIP；不读取源数据集或改写 Kaggle 输出 |
| `run_l1_oof_perclass_conf_scan.py` | CPU-only L1 E1b：只重评 Phase 0/E1 持久化 OOF 预测的单类别置信度覆盖，明确区分运行阈值对照与固定原始 mAP 锚点；不读冻结集、不做新推理 |
| `l1_e3_p2_smoke_20260823/run_l1_e3_p2_smoke.py` | E3 原生 Ultralytics P2 检测头零 GPU 冒烟：校验 YAML、构建、合成前反向与参数/FLOPs，不训练、不读数据集 |
| `build_l1_e4_copypaste_dataset.py` | E4 Copy-Paste 派生视图构建器：按 Fold 隔离、原图硬链接、训练内供体/目标、原始像素尺度与零标签相交；支持 `--dry-run` |
| `verify_l1_e4_copypaste_dataset.py` | 独立验证 E4 派生视图的冻结边界、硬链接、训练/验证隔离、五列标签、供体/目标和粘贴记录 |
| `prepare_l1_e3_e4_parallel_kaggle.py` | 生成 E3 P2 与 E4 Copy-Paste 的两条 Kaggle Fold 1 筛选任务包；固定开发集边界、9 小时时间上限、原图级评测器和自动闸门，生成器本身不提交任务 |
| `derive_l1_phase0_holdout_thresholds.py` | 从非目标 Fold 的 Phase 0 原始预测生成留出 Fold 专用阈值 JSON，避免在本 Fold 评测数据上挑阈值 |
| `verify_l1_phase1_tile_evaluation.py` | 只读校验并安全解压 Phase 1 评测 ZIP：验证 SHA-256、CRC、开发集边界、权重、profile 和逐框明细，生成停止/继续闸门报告 |
| `summarize_l1_oof_auto_gate.py` | 汇总 OOF 候选的三折自评与历史锚点自动非回退复评；只读归档 JSON，不访问 release-validation |
| `select_l1_oof_final_epoch.py` | 从已归档的 YOLO11m OOF 三折曲线选择固定最终训练时长；本次结果固定为 83 epoch |
| `train_l1_oof_final.py` | 在 OOF 候选的 1,127 张开发图上训练固定时长的 YOLO11m；发布权重固定为 `last.pt` 副本，不以最终开发集日志选择权重 |
| `build_l1_oof_final_kaggle_config.py` / `prepare_l1_oof_final_kaggle_kernel.py` | 构建仅标签/清单/脚本的私有 Kaggle 配置包，并生成无 GPU 预检或最终 GPU Kernel；不上传图片或 `release-validation` |
| `train_l1_freeze_two_stage.py` / `prepare_l1_freeze_kaggle_kernel.py` | L1 冻结层独立成对消融：同一 83 epoch 两阶段流程下比较 Stage 1 `freeze=0` 对照与 `freeze=10` 候选；仅 OOF 开发视图、仅 Fold 1 筛选 |
| `verify_l1_freeze_pair.py` | 冻结层成对验收：核验两个归档 run 的参数/运行时/开发边界，并比较统一本机复评的 mAP50-95 与六个重点类别 Recall；未通过不得训练全开发集或导出新 ONNX |
| `kaggle_l1_ryw18_oofconsensusv1_preflight/l1-ryw18-oofconsensusv1-preflight.py` | 已提交的 OOF 候选无 GPU 挂载预检：只恢复、校验 1,127 图/标签和三份 train/val，不读取冻结测试集 |
| `monitor_l1_kaggle_chain.py` | L1 可选串行监控入口：仅在用户明确要求自动推进时使用；当前已停用，训练完成后由用户通知再手动归档与提交下一 Kernel |
| `rebuild_worklog_index.py` | 工作日志索引维护：扫描 `工作日志条目/` 全部条目文件，重建可点击「条目索引」表（日期/标题链接/主线），标签按标题继承、新条目人工补标；支持 `--dry-run` 只审计 |
| `split_worklog_entries.py` | 一次性迁移（2026-07-27 已执行）：把主文件/分片正文拆分为 `工作日志条目/` 独立文件并截断主文件；保留备查 |
| `extract_dianji_grade_features.py` | 视图上 v2a join + v2b ryw.pt@1536 特征提取 |
| `extract_dianji_grade_y11m640_features.py` | 冻结 YOLO11m ONNX@640 推理，按历史定义重建 43 BASE 特征，并复用 20 个 `tx__` 特征 |
| `train_dianji_grade_lgbm.py` | 四分类模型对比（majority/规则/树/LightGBM），test 仅最终评估 |
| `build_dianji_grade_v2_dataset.py` | 历史草稿（上一任务中断遗留），**非**当前流水线入口 |
| `extract_dianji_grade_v3_features.py` | v3 空间（3×3 分区/边缘带/聚簇度，源自 detections_1536）+ 纹理（LBP/GLCM，numpy 手写）特征提取 |
| `train_dianji_grade_v3.py` | v3 四特征集对照训练 + 拒判曲线（阈值仅用 valid），test 仅最终一次 |
| `compare_dianji_classifiers.py` | 电积镍分类方案对比 |
| `requirements_classify.txt` | 分类相关依赖 |
| `PY` | 空占位文件，可忽略 |

特征提取与等级软连接产物已按用户要求删除，相关脚本暂保留以便重新规划后复用。

---

## 4. 推荐工作顺序（检测优化）

1. 读 `工作日志.md`，确认进度与阻塞。
2. 读 `当前工作计划.md`，确认当前闸道和资源优先级。
3. 新三光：先 `三线线扫三光采集与数据规范.md`。
4. 无人工标：`无人工伪标签闭环方案.md`（版本化输出，禁止覆盖历史标签）。
5. 续训/冻结：`YOLO冻结与迁移学习指南.md`。
6. 旧+新混合：`旧数据与新三光数据混合训练指南.md`。
7. 数据审计与业务类别确认 → 不合格则先清洗，不调参。
8. 固定独立测试集 + 可复现 3 折开发集 → 统一基线。
9. 按操作手册优先级消融；每次实验填实验记录 + 工作日志摘要。
10. 只比较同数据、同划分、同验证程序的指标。
11. 任务结束更新 `工作日志.md`。

等级分类线需要重新规划；当前仅保留原始分类文件夹，见 §5。

---

## 5. 关联路径与分支（用户约定）

| 用途 | 路径 / 名称 |
| --- | --- |
| 电解镍 **检测** | `E:\PythonProject\datasets\Dianjinie0211`（YAML 约 5 类） |
| `dianjienie` **检测** | `E:\PythonProject\datasets\dianjienie`（YAML 约 17 类；历史文档对中文镍种称呼有冲突，当前以目录名为准） |
| `dianjienie` **18 类历史标签锚点** | `E:\PythonProject\datasets\dianjienie_ryw18_anchor_v1`（1325 图/18232 框；198 图冻结测试集 + 1127 图开发集 3 折；原集不改） |
| 电解镍 **分类** | `E:\PythonProject\datasets\classify_2types`（good/bad） |
| 电积镍 **分类** | `E:\PythonProject\datasets\classify_4types_dianji`（3#/9950/9996/bad） |
| 企业 0309 **原始示例归档** | `E:\PythonProject\datasets\enterprise_label_raw_20260716\电积镍板0309`（由原 Pictures/label 整体迁移） |
| 企业 label **原始迁移归档** | `E:\PythonProject\datasets\enterprise_label_raw_20260716`（原 `Pictures\label` 整体迁移，含 ZIP 与原目录结构） |
| 企业 label **精确去重视图** | `E:\PythonProject\datasets\enterprise_label_dedup_v1`（628 个图片路径去重为 191 张唯一图） |
| 当前增量汇总伪标签集 | `E:\PythonProject\datasets\combined_all_pseudo_v2`（3264 张；label 去重后 191 张均已存在于 v1，无新增推理） |
| 企业电解镍 0306 **待标注检测集** | `E:\PythonProject\datasets\enterprise_dianjie_0306_detection_unlabeled`（硬链接） |
| 企业电积镍 0309 **明确等级分类集** | `E:\PythonProject\datasets\enterprise_dianji_0309_classification`（硬链接，仅 9950/3#） |
| 企业电积镍 0309 **待标注检测集** | `E:\PythonProject\datasets\enterprise_dianji_0309_detection_unlabeled`（硬链接，缺陷提示目录） |
| 电积镍等级分类 **合并硬链接视图 v1** | `E:\PythonProject\datasets\dianjinie_grade_classification_v1`（4types+0309 去重后 1072 张，70/15/15 分组分层；标签口头确认，模型未企业确认） |
| 电积镍等级 v2 **特征/模型产物** | `E:\PythonProject\优化\dianji_grade_v2`（v2a/v2b 特征、detections_1536、report.md；`unconfirmed_pending_enterprise`） |
| 电积镍等级 v3 **空间/纹理+拒判产物** | `E:\PythonProject\优化\dianji_grade_v3`（features_v3_full、model_comparison、reject_curve、report.md、robustness_4types_only；`unconfirmed_pending_enterprise`） |
| 电积镍等级 YOLO11m@640 **重提取/重训产物** | `E:\PythonProject\优化\dianji_grade_y11m640_v1`（43 新 BASE + 20 `tx__`、LightGBM、contract、实验记录；`unconfirmed_pending_enterprise`） |
| L1 **汇总数据集（检测+等级双用途）** | `E:\PythonProject\datasets\union_original_labels_v1`（3265 图/1455 原标签 + 1072 张等级真值哈希接入；SHA-256 去重；**不划分**；datasets 目录最终形态 = 5 原始集 + 本集） |
| 归档（已迁出 datasets） | `E:\PythonProject\archives\`：`combined_all_pseudo_v2`（伪标签主版本）、`enterprise_label_raw_20260716`、`enterprise_dianji_0309_classification`、`dianjinie_grade_classification_v1`（L2 脚本默认路径需同步改） |
| 伪标签审查（已归档） | `E:\PythonProject\优化\_archived\pseudo_label_reviews_20260723`（用户 2026-07-23 决定弃用伪标签审查线） |
| 电积镍等级 `ryw.pt` **最新特征分析** | `E:\PythonProject\优化\grade_feature_analysis\dianji_ryw_20260713_223701\report.md` |
| **工具包** | `E:\PythonProject\Tools` |
| └ 自动标注 | `Tools\ModelAutoLabelTool`（教师权重如 `weights\yolov5\ryw.pt`、`slf.pt`） |
| └ 图像预处理 | `Tools\ImageProcessTool` |
| WebAPI 仓库 | `E:\PythonProject\ImetricCodes\JinchuanPythonWebapi` |
| WebAPI **主分支** | `308`（`origin/308`） |
| WebAPI **优化分支** | `优化分支`（本地开发，默认未 push） |
| **重构 UI（Tauri 桌面端）** | `E:\PythonProject\UI`（原 C# WinForms 应用的 Tauri 2 + Rust / Vue 3 + TS + Vite 重写，pnpm 工作区；该目录自带 `AGENTS.md`/`CLAUDE.md`/`README.md`，UI 侧规则以其为准） |
| 模型训练运行档案 | `E:\PythonProject\models\runs`：每个模型训练 run 的权重、配置、日志、曲线、标准重评和 run README；顶层 `AGENTS.md` / `CLAUDE.md` / `README.md` / `RUN_IDENTITY_CATALOG.md` 是 Agent 索引 |
| 模型结果与工作总结 | `E:\PythonProject\models\results`：经核验后的比较报告、指标汇总、图表和工作总结；不存放权重、训练配置或原始训练日志 |

### 5.1 数据集改动原则（强制）

1. **不要改原有数据集唯一副本**：原图、`labels/` 人工标签、主 `data.yaml` 默认只读。  
2. 新标注 → **新建版本目录**（例：`labels_pseudo_v1/`、`labels_pseudo_v1_v2/`），禁止覆盖 `labels/`。  
3. 新划分 / 新实验视图 → **软连接或路径清单** 引用原图，在新版本根目录组装；不要复制整库除非必要。  
4. 清洗、重划分 → 输出到新版本目录 + 映射/清单（见 `AGENTS.md`）。

### 5.2 伪标签重标注与审查（历史记录）

| 状态 | 内容 |
| --- | --- |
| 已删除 | 两检测集的 `labels_pseudo_v1`、`labels_pseudo_v1_raw` 和 `manifests/` 已按用户要求删除；原 `labels/` 未改 |
| 保留代码/报告 | 综合审查脚本、历史报告和 `visualizations/` 仍在本目录，仅供追溯，不代表数据集内伪标签仍存在 |

流程细节：`伪标签重标注报告_v1.md`、`伪标签审查流程记录_v1.md`。

- **检测集 ≠ 分类集**，路径与标签格式都不同，禁止混用。  
- 改 WebAPI 前确认 git 分支；优化工作在 `优化分支`，勿默认改 `308`。  
- 细节与核实记录见 `工作日志.md` 对应条目。

---

## 6. 目录结构（逻辑视图）

```text
优化/
├── README.md                          ← 你在这里（总索引）
├── AGENTS.md / CLAUDE.md              ← Agent 规则
├── 工作日志.md / 工作日志条目模板.md
├── 镍板缺陷模型优化操作手册.md         ← 检测优化主规范
├── 模型优化实验记录模板.md
├── 三线… / 无人工… / YOLO冻结… / 旧数据混合…   ← 专题规范
├── 伪标签* v1 报告与流程 / visualizations/     ← 历史记录，数据集内伪标签目录已删除
├── *grade* / build_* / visualize_* / apply_*   ← 辅助脚本
└── Instruction.md / nickel_defect_*            ← 历史背景
```

物理文件仍平铺在根目录（未强行搬迁，避免打断脚本与外链）；**分类靠本索引，不靠文件夹嵌套。**

---

## 7. 维护约定

- 新增**规范类**文档：在本 README §2 登记一行，并视需要在 `AGENTS.md` / `CLAUDE.md` 挂入口。  
- 新增**实验结论**：写实验记录或专项报告，并在 `工作日志.md` 留摘要链接。  
- 不在此目录提交大权重、原始大数据集；产物路径写进日志即可。  
- 空文件 `PY` 无文档职责；若确认无用可日后删除。
