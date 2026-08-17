# Claude 项目操作说明

Claude 在本项目中必须首先读取并遵守根目录 `AGENTS.md`。若本文件与 `AGENTS.md` 冲突，以 `AGENTS.md` 为准。  
**完整文档地图、脚本清单与路径表见 `README.md`**（勿在本文件重复维护长列表）。

## 开始任务

1. **必须首先读取**根目录 `AGENTS.md` 与本文件。
2. **必须读取** `工作日志.md` 的「当前状态摘要」和最近 3～5 条条目；续作时从其中的「下一步」接着做，不得重复已完成步骤。
   - **日志形态（2026-07-27 条目文件夹化后）**：主文件只含摘要区 + 全量「条目索引表」（日期 / 标题链接 / 主线标签）；全部条目正文在 `工作日志条目/` 的独立 `.md` 文件中。读「最近 3～5 条」时：先看摘要区 + 索引表（最新在上）→ 按标题链接打开条目文件，避免逐文件遍历。主线标签为人工归号，可能模糊，**下结论前以正文字段为准**。写入新条目 = 在 `工作日志条目/` 新建文件后运行 `python rebuild_worklog_index.py` 重建索引（标签按标题继承）。
3. 浏览 `README.md` 文档地图，只打开与当前任务相关的规范/报告。
4. 使用搜索工具定位实际文件，再读取配置、脚本或实验产物；禁止仅按历史文档直接生成结论。
5. 涉及数据集时，先做只读审计并报告将要修改的文件范围。
6. 涉及训练方案时，区分统一基线、模型独立调优和最终测试，不能混用指标。
7. 涉及 `JinchuanPythonWebapi` 时，先确认 git 分支（优化工作用 `优化分支`，主线为 `308`）。

## 结束任务

1. **无论任务大小，只要修改了文件、运行了脚本或形成了可复用结论，都必须按 `工作日志条目模板.md` 在 `工作日志.md` 顶部条目区追加一条记录（最新在上）。**
2. 更新 `工作日志.md` 顶部的「当前状态摘要」。
3. 若为正式训练/验证实验，另填实验记录，并在工作日志中链接路径。
4. 若新增了规范类文档或改变了目录职责，更新 `README.md` 对应章节。
5. 回复用户时列出改动文件、验证结果、未解决风险，且与日志一致。
6. 纯问答且未改文件、未形成可复用结论时可不写日志；其余实质工作必须写。

细则与检查清单见 `AGENTS.md` 第 5 节。

## 当前关键事实

- **当前项目主控计划（2026-08-16 22:24）**：以 `当前工作计划.md` 为当前执行入口。用户明确不做人工标注或审核，L1 使用无人工 OOF 候选路线：两个架构三折完成后，六个候选权重已按未改动 `anchor_v1` 标签在对应 Fold 自动复评。`l1_auto_gate_oofconsensusv1_20260809/report.md` 的固定工程安全闸门结果为 `auto_continue_oof_candidate`，预先选择 YOLO11m 为自动主候选。两项最终 Kaggle 训练均已完成归档：YOLO11m 为 1,127 图、15,671 框、640、固定 83 epoch，固定权重 SHA-256=`bb501e224b48e610aac5c05293d0b3c63ef9eccad792712466a25803bd4de521`；YOLO26m 为相同开发集、固定 174 epoch，SHA-256=`d0cea7c8ff52534734ddae5ebb54473f3f6d70d4c1a369296e05fe8a709dd227`。仅 YOLO11m 在预选后做了唯一一次历史 198 图/2,806 框 release-validation，P/R/mAP50/mAP50-95=`0.42582/0.38473/0.35959/0.23147`；访问登记 `l1_release_validation_access_registry.json` 已锁定同一冻结清单，YOLO26m 不读取它。YOLO11m ONNX 已导出并在 8 张开发图上通过语义一致性门槛（同类 IoU>=0.90 全局匹配率 `0.96617`、单图检测数最大差 `1`）。L1-hr G4 Kernel `lly3323w/l1-ryw18-yolo11m-oof-v1-hr1280-b4-e300-fold01` 已 `COMPLETE` 并归档：同一开发 Fold 1、1280/batch 4 的最佳训练期 mAP50-95=`0.16264`，低于 640/batch 8 的 `0.16382`；上限 300 epoch、实际 183 epoch，符合 `patience=100` 的早停。两次 train/val 清单哈希一致且未读 release-validation；1280 不扩展 Fold 2/3，不改变 640 自动候选。以上均为 `auto_only`，不得表述为人工真值或稳定架构优势。
- **L1 小目标切片研究（2026-08-17 18:32，等待评测）**：1280 无收益后，Kaggle CPU 预检 `lly3323w/l1-ryw18-oof-v1-tile1024-o25-preflight-fold01` 已验证 OOF 自动候选 Fold 1 的 751/376 清单和无 test 边界。GPU Kernel `lly3323w/l1-ryw18-y11m-oof-v1-tile1024-o25-b8-e300-f01` version 1 在 33.1 秒、`model.train()` 前因 `output_root` 初始化冲突失败；version 2 完成派生切片后在 391.6 秒、`model.train()` 前因缺少 `ultralytics` 失败。固定 `torch==2.6.0`/`torchvision==0.21.0`/`numpy==1.26.4`/`ultralytics==8.4.66` 前置后的 version 3 实际训练至第 109 轮完成验证，随后第 110 轮中被取消；Kaggle 当前状态为 `CANCEL_ACKNOWLEDGED`，无结果 ZIP、无 `tile_evaluation`。回收目录 `kaggle_l1_ryw18_y11m_oof_v1_tile1024_o25_b8_e300_f01_r3/recovered_interrupted_v3/` 已校验 `best.pt` SHA-256=`c4a903e8e8438798d44fc0548d666923a8728509f012d124101a8b980d7d875e`（checkpoint epoch=74，第 75 轮最高训练期 mAP50-95=`0.15178`）及 `last.pt` SHA-256=`882e952b3d96243b03564706856f5ffa9821d79357714c1945b2893b0dde76dd`（checkpoint epoch=108）。配置仍为 YOLO11m、640/batch 8、1024/25% 切片，训练视图为 751 整图 + 4,935 焦点切片 + 1,233 负切片；它不是完成实验，也不能与历史 Ultralytics 指标直接排名。`l1_tile_pipeline.py` 现有独立 `evaluate` 模式，将回收的 `best.pt` 与固定 Fold 1 640 基线按原图坐标回映射/类别全局 NMS输出四组同口径报告；当前 Kaggle API 返回 30 小时周 GPU 额度用尽，等恢复后先跑评测，再决定是否分段续训。历史 `release-validation` 未挂载、不读取；用户不需要人工审核或本机监控器。
- **冻结 holdout 访问边界更正（2026-08-03）**：早期框尺寸诊断曾读取 198 张 holdout 的文件名、标签和部分图片头部，但未用于训练、选模、调阈值或评估；后续不得再表述为“从未访问”。若需要严格零访问测试集，应重新建立未读 group holdout。

- **模型产物目录职责（2026-07-23 用户确认）**：`E:\PythonProject\models\runs` 是各模型的可复现训练结果存放处，每个 run 保存权重、配置、日志、曲线、标准重评指标与对应 `README.md`；顶层 `AGENTS.md`、`CLAUDE.md`、`README.md`、`RUN_IDENTITY_CATALOG.md` 供 Agent 维护和定位。`E:\PythonProject\models\results` 只保存经证据核验后的比较结果、报告、图表与工作总结，不得保存训练权重、原始 epoch 日志或唯一训练证据。具体规则见 `models/README.md`、`models/runs/AGENTS.md`、`models/results/README.md`。

### 检测数据集（YOLO）

- 电解镍检测：`E:\PythonProject\datasets\Dianjinie0211`，当前 YAML 为 5 类。
- `dianjienie` 检测：`E:\PythonProject\datasets\dianjienie`，当前 YAML 为 17 类。历史文档对其中文镍种称呼存在冲突；在业务侧确认前，当前锚点和训练结论仅使用目录名，不推断镍种语义。
- 两个检测数据集必须分别训练和报告，除非先完成经过业务确认的类别映射。
- 企业电解镍 0306 待标注检测硬链接集：`E:\PythonProject\datasets\enterprise_dianjie_0306_detection_unlabeled`，68 张，无可靠标签。
- 企业电积镍 0309 待标注检测硬链接集：`E:\PythonProject\datasets\enterprise_dianji_0309_detection_unlabeled`，37 张，按缺陷提示目录组织，无框标签。

### 分类 / 等级数据集（文件夹分类，非 YOLO 检测标签）

- 用户已要求删除全部特征提取产物和等级划分软连接数据集，后续需要重新规划后再生成。
- 电积镍分类（原 4 类）：`E:\PythonProject\datasets\classify_4types_dianji`（`train`/`valid`，子类如 `3#`、`9950`、`9996`、`bad`）。
- 企业电积镍 0309 明确等级分类硬链接集：`E:\PythonProject\datasets\enterprise_dianji_0309_classification`，`9950=74`、`3#=12`，未划分 train/valid。
- 电解镍分类（2 类）：`E:\PythonProject\datasets\classify_2types`（`train`/`valid`，子类 `good`/`bad`）。
- 原 `E:\Users\23916\Pictures\label` 已按用户要求整体迁移到 `E:\PythonProject\datasets\enterprise_label_raw_20260716`；内部 628 个图片路径经 SHA-256 去重后为 191 张唯一图，规范化硬链接视图为 `enterprise_label_dedup_v1`。
- 当前只保留原始分类数据集，不存在推荐的派生等级数据集或特征集。
- **（2026-07-23 新一代，取代上一条的空窗状态）电积镍等级 v1 视图 + v2/v3 产物**：`datasets\dianjinie_grade_classification_v1`（4types+0309 合并去重 1072 张、近重复 16 组按组划分、70/15/15、硬链接；两来源标签均经用户确认：0309=企业真值且 9950=2号板/3#=3号板、4types 用户已逐张审查）。产物 `优化\dianji_grade_v2`（ryw.pt@1536 全量特征）与 `优化\dianji_grade_v3`（空间/纹理+拒判）。主结果：v2 基线 LGBM test 0.640/0.413；**v3 +纹理 0.776/0.663**，拒判 66% 覆盖 acc=0.897；单源稳健性通过但**等级↔拍摄批次混淆未排除**（需批次元数据/新会话图像外部验证）。电解镍 DianjieGradeRule 迁移 acc=0.02 实锤不可用。端到端图像分类路线已被用户否决（论文 90% 无落地证据）。全部产物 `unconfirmed_pending_enterprise`，企业确认前不进 WebAPI。
- `企业0309电积镍等级规则总结.md` 为用户基于权重特征的未验证个人总结（2026-07-22 声明），仅作假设不得作真值；判级方向已定为「缺陷种类+面积规则」（18 类 ryw.pt）。
- **类别映射已确认（2026-07-23）**：`dianjienie` 17 类按 id 顺序对应 `ryw.pt` 18 类（oilstain=yellowstain；unknown_14/15/16=youpao/heibian/mijixiaokeng；dianjienie 无 fanxitong），见 `类别映射_dianjienie17_to_ryw18.md`；dianjienie 原始人工标签可按序映射并入 L1。仍零监督的类：youpao（0 框）、fanxitong（体系外）。
- **dianjienie RYW18 历史标签锚点（2026-07-24）**：`E:\PythonProject\datasets\dianjienie_ryw18_anchor_v1` 是新建、只增不改的 18 类训练视图，不得再使用原 `dianjienie` train/valid 划分。锚点含 1325 图/18232 框，固定 198 图独立测试集、1127 图开发集 3 折；10 组标签冲突精确重复的 20 图已排除，6 对近重复被强制同组；只删除了新标签副本中的 1 个确定性越界框。训练入口与后续 OOF 共识路线见 `dianjienie_ryw18_anchor_v1_训练交接.md`。`youpao`、`fanxitong` 零监督不可评，`heibian`、`mijixiaokeng` 不足以覆盖每个 Fold。
- **RYW18 Kaggle 路径（2026-07-27）**：本机无法训练时使用私有 Kaggle Dataset + Kaggle GPU Kernel；既有私有 `lly3323w/dianjienie` 已全量核验覆盖 RYW18 的 1325 张源图（路径和文件大小一致），但它在当前 L1 中仅是只读图片仓库，旧 17 类标签、旧 `train/valid` 划分及其余 25 张图片均不使用。私有配置 Dataset `lly3323w/dianjienie-ryw18-anchor-v1-config` 已 `ready`；Kaggle 自动展开的 1342 个 payload 文件已与 manifest 全量核验一致。无 GPU 预检 `lly3323w/l1-ryw18-anchor-v1-preflight` 已在实际挂载布局 `/kaggle/input/datasets/lly3323w/...` 通过，恢复结果为 1325 图/1325 标签与 Fold 1 的 751/376/198。初版正式 Kernel `lly3323w/l1-ryw18-yolo11m-anchorv1-fold01` 因提交时请求 32400 秒在已完成 epoch 250 后变为 `CANCEL_ACKNOWLEDGED`；只留下 0 字节 `best.pt`、无 `last.pt`，不得归档、续训或用于模型结论。失败证据位于 `优化\l1_kaggle_chain_monitor\failed_run_probe`。独立 Kernel `lly3323w/l1-ryw18-yolo11m-anchorv1-fold01-retry1` 已以同一数据、Fold、参数和官方 `yolo11m.pt` 在 43200 秒请求下完成，结果 ZIP 已经 CRC、必需文件、权重 SHA-256 和固定 train/val/test 清单校验后归档至 `models\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold01`。训练期 `best.pt` 是 epoch 177（mAP50/mAP50-95=`0.29306/0.15646`）；同一固定 val 的本机独立复评为 `0.27848/0.14971`，逐类指标在 `优化\l1_reeval_yolo11m_fold01_retry2\metrics_per_class_reeval.json`。二者属于不同运行环境的开发集口径，后续 YOLO11m/YOLO26m 比较必须复用本机复评入口；冻结测试集未访问。YOLO26m Fold 1 Kernel `lly3323w/l1-ryw18-yolo26m-anchorv1-fold01` 已按完全相同配置（仅 `yolo26m.pt`）提交，最近确认状态为 `RUNNING`。当前没有本地监控进程；用户可关闭本机，完成后再通知 Agent 手动收集。Kernel 用动态发现的配置包路径调用 `kaggle_dianjienie_ryw18_existing_data_runner.py`，避免依赖 Kaggle 的自动展开层级。固定依赖为 Torch 2.6.0 CUDA 12.4、Torchvision 0.21.0、NumPy 1.26.4、Ultralytics 8.4.66。不要将约 21.25 GiB 图像上传到 GitHub、Google Drive 或新的 Kaggle Dataset；旧全量视图只作灾备回退。
- **L1 汇总检测集（2026-07-23，最终形态）**：`datasets` 目录 = 5 个原始集（`Dianjinie0211 / classify_2types / classify_4types_dianji / dianjienie / nietiao`）+ `union_original_labels_v1`。union：3265 图（SHA-256 去重跳过 1834）、1455 个原标签（dianjienie 1335 + labels_original 120）、**1072 张等级真值经哈希接入**（9950=546、9996=452、3#=46、bad=28，用户确认 4types 与 Dianjinie0211 同图=同一批照片复用）；**不划分**，类别表 ryw.pt 18 类基准，youpao/fanxitong 标记不可评。已迁出至 `E:\PythonProject\archives\`：`combined_all_pseudo_v2`、`enterprise_label_raw_20260716`、`enterprise_dianji_0309_classification`、`dianjinie_grade_classification_v1`（**L2 脚本/文档中的 grade 视图路径需同步改**）。已删除（内容零损失）：`enterprise_label_dedup_v1`、0309/0306 待标注集。伪标签审查已归档 `优化\_archived\pseudo_label_reviews_20260723`。
- 最新只读分析：`grade_feature_analysis/dianji_ryw_20260713_223701/report.md`。统一 `ryw.pt` 后，企业 3# 的大面积 bubble/B 组特征更突出，但“检出 overfire”不能区分 3#/9950；电解镍七状态硬编码与企业等级语义不一致。**用户 2026-07-16 澄清：该“语义不一致”结论基于仅含部分等级（9950/3#）的企业目录 + 电解镍硬编码特征提取，比较基础存疑，待企业正式判级标准确认后重评。**
- 模型身份必须区分：`ryw.pt` 是 YOLOv5 缺陷检测权重，输出框/类别/置信度；`ryw.pth` 才是 ResNet `bad/good` 二分类权重。固定说明文档为 `ryw_pt模型身份与电积镍检测结果说明.md`。
- **`ryw.pt` 训练配置（权重元数据）**：`yolov5s.pt` 预训练 + 名为 `yolov5m.yaml` 实为 s 规模（depth/width=0.33/0.50，约 7M）+ `hyp.scratch-low` 官方默认 + epochs=300 / batch=32 / imgsz=640 / SGD / patience=100 / seed=0；NWD 未启用。详见 `原项目YOLOv5训练配置与mAP口径说明.md`。
- **项目检测 mAP 口径错误**：`JinchuanYolov5sTrain/val.py` 使用 `linspace(0.1, 0.55, 10)`，报表 mAP50≈**mAP@0.1**，mAP50-95≈**mAP@0.1:0.55**，不是标准 mAP@0.5 / mAP@0.5:0.95。与官方 YOLOv5/YOLO11 比指标前必须先统一验证 IoU。分割 `segment/val.py` 仍是标准 0.5:0.95。
- **Dianjinie0211 新 YOLOv5 产物身份**：目录已更正为 `E:\PythonProject\models\runs\dianjinie_small_yolov5s_640_projcode`，主要权重位于 `exp_dianjinie_yolov5s2`。最终 checkpoint 为 5 类 s 规模（0.33/0.50、7,033,114 参数）。首次运行使用官方 `yolov5m.pt`，但非空本地 `cfg=models/yolov5m.yaml` 优先，只迁移 57/349 项。标准 FP32 重评为 mAP50=0.18086、mAP50-95=0.06233；验证集无 `nodule` 标注，所有模型该类均不可评。`ryw.pt` 同为 s 规模但属于 18 类 `nieban1100_xiangmu` 任务，未经企业类别映射和同一人工测试集不得与该 5 类权重直接排名或替换。以目标目录 `README.md`、`metrics_standard_iou.json` 和 `models/results/电积镍640八模型统一对比实验报告.docx` 为准。
- **Dianjinie0211 640 八模型统一对比（2026-07-23）**：同一 5 类 val、640、标准 IoU 的完整逐缺陷表纳入 RF-DETR Small、YOLO11m、YOLO26m、YOLOv8m、YOLOv8s、YOLO26s、YOLO11s、项目 YOLOv5-s；按 mAP50-95 为 `0.10587/0.10495/0.10347/0.09686/0.09390/0.09363/0.09110/0.06233`。RF-DETR 约 32.1M，应该直接与 YOLO11m=20.03M、YOLO26m=20.35M、YOLOv8m=25.86M 一同解读，不能只放入 Small 组。YOLO11m/YOLO26m/YOLOv8s 的 per-class 指标已用现有 best.pt 在本机 Ultralytics 8.4.90、FP32、640、NMS IoU=0.7、max_det=300 复验并保存；其余使用已归档标准指标。报告为 `models/results/电积镍640八模型统一对比实验报告.docx`，含逐缺陷 mAP50/mAP50-95；`nodule=0` 真值框，统一不可评。单 seed、训练预算不一，仅作现有方案效果对比。
- **runs 命名与目录映射（2026-07-22）**：5 类 640 标准 Ultralytics run 统一采用 `dianjinie_<size>_<model规模>_640_kaggle_seed<seed>_cm<close_mosaic>_standard`；当前目录、历史非标准目录和旧名映射以 `E:/PythonProject/models/runs/RUN_IDENTITY_CATALOG.md` 为唯一索引。不得依据无模型规模或无 seed/cm 的旧名判断实验条件。
- **YOLOv8m / YOLOv8s 640（2026-07-23 已完成）**：YOLOv8m Kaggle `lly3323w/yolov8m-dianjinie-640-project-baseline` version 1 使用 5 类、640、300 epochs、batch=8、SGD、seed=0、patience=100、close_mosaic=0；早停于 158 epochs，最佳 epoch=58，mAP50=`0.24381`、mAP50-95=`0.09686`。YOLOv8s Kaggle `lly3323w/yolov8s-dianjinie-640-project-baseline` version 1 已完成 ZIP 校验和 `runs` 归档，早停于 151 epochs、最佳 epoch=51；同口径 FP32 逐类复验 mAP50=`0.23662`、mAP50-95=`0.09390`。原始产物分别位于 `models/runs/dianjinie_medium_yolov8m_640_kaggle_seed0_cm0_standard` 与 `models/runs/dianjinie_small_yolov8s_640_kaggle_seed0_cm0_standard`。
- **RF-DETR Small 640 对比（2026-07-21 已完成）**：Kaggle `lly3323w/rf-detr-small-dianjinie-640-project-baseline` version 2 完成 100 epochs；`checkpoint_best_total.pth` 标准 COCO 重评为 mAP50=`0.27502`、mAP50-95=`0.10587`。RF-DETR run 位于 `models/runs/dianjinie_small_rfdetr_640_kaggle/rfdetr_small_dianjinie_640_project_baseline`；统一八模型逐缺陷报告为 `models/results/电积镍640八模型统一对比实验报告.docx` 和 `.pdf`。验证集 216 张/4337 框，`nodule=0` 仍为 `null/不可评`。参数量约 32.1M，应与 Medium YOLO 一同解读；单 seed 与非同预算训练条件下不可宣称稳定架构优势。
- 原分类调用链审计：电解镍为 `ryw.pt` 检测框面积硬编码为主、`ryw.pth` 仅辅助大结瘤分支；电积镍原设计为 `slf.pth` 四分类 + `slf.pt` 检测，但当前 `308` 暂用 `ryw.pth` 二分类且 `slf.pth` 未调用。两个 `.pth` 加载后均被新 `nn.Linear` 覆盖分类头，当前 ResNet结果不可靠，详见固定说明文档 §9。
- 派生伪标签、软连接视图、特征和分类结果必须写入 `E:\PythonProject\优化`，不得再混入原始数据集目录；用户明确要求的新硬链接数据集除外。
- 汇总伪标签目录命名：`dianjienie=电解镍`、`dianjinie=电积镍`，后接时间、版本和 `+` 连接的来源名；汇总集使用扁平 `images/labels`，不继承原 train/valid/test。
- 当前两套汇总伪标签统一由 `ryw.pt` 生成，不使用 `slf.pt`。
- 两套汇总伪标签的可视化审查页位于 `E:\PythonProject\优化\pseudo_label_reviews\<对应数据集名>\review.html`，审查结果从页面导出 JSONL。
- 当前唯一保留的汇总版本为 `E:\PythonProject\datasets\combined_all_pseudo_v2`：3264 张、74331 个框；与企业 label 的 191 张唯一图交叉查重后没有新增图片，因此没有新增模型推理。旧 v1 数据集及其审查、等级特征目录已删除。
- v2 等级与特征产物位于 `grade_feature_analysis/combined_all_pseudo_v2_grade`；Windows 中文路径读取已改为 `np.fromfile + cv2.imdecode`，审计为 3264 张、缺失标签 0、不可读图片 0。
- 伪标签主版本必须位于 `labels_with_conf/` 并保留第六列置信度；默认状态阈值为 accepted `>=0.70`、uncertain `0.40~0.70`、rejected `0.15~0.40`。五列训练标签只能由审查结果另行生成。

- **RYW18 Kaggle Fold 1 状态更新（2026-07-27 13:45）**：YOLO26m Kernel `lly3323w/l1-ryw18-yolo26m-anchorv1-fold01` 已 `COMPLETE`，结果 ZIP SHA-256=`2e01f70a615e4a4a472b487e2c7e9ae06db86582b20e3206d14299703c8fab48`，CRC、30 个文件、必需文件、权重和固定 `751/376/198` 清单均已校验，归档至 `models\\runs\\dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_anchorv1_fold01`。Kaggle `best.pt` 为 epoch 244（mAP50/mAP50-95=`0.30990/0.15720`）；同一 376 张开发 val 的本机复评为 P/R/mAP50/mAP50-95=`0.48451/0.29343/0.30540/0.15454`，证据位于 `优化\\l1_reeval_yolo26m_fold01\\metrics_per_class_reeval.json`。冻结 198 张 test 未访问。YOLO11m 和 YOLO26m 均只有一个开发 Fold/seed，Fold 2/3 尚未开始，暂不能选择最终架构；对比必须使用本机复评口径。

- **RYW18 Kaggle Fold 2 状态更新（2026-07-27 13:55）**：YOLO11m Fold 2 Kernel `lly3323w/l1-ryw18-yolo11m-anchorv1-fold02` 已由 Kaggle CLI 成功推送并确认 `RUNNING`。本机 `train_dianjienie_ryw18_anchor.py --model yolo11m.pt --fold 2 --dry-run` 已验证固定清单 `train/val/test=752/375/198`，三集合无交集、总计 1325 图；模型、300 epochs、640、batch 8、SGD、seed 0、patience 100、`close_mosaic=0`、`resume=false` 与 Fold 1 相同。本地未启动训练、监控或下载进程；Kaggle 完成后仍须先归档校验并做对应 Fold 2 本机复评，冻结测试集不访问。

- **RYW18 Kaggle Fold 2 归档与配额暂停（2026-07-28 13:45）**：YOLO11m Kernel `lly3323w/l1-ryw18-yolo11m-anchorv1-fold02` 已 `COMPLETE`。结果 ZIP SHA-256=`a0a8ea9793ebc06b1a2cd95789e397fb393d60117dfba6192cb889d5c11eccfb`，CRC、30 个文件、必需文件、权重和归档内固定 `752/375/198` 清单均已校验，归档至 `models\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold02`。训练期 `results.csv` 的 fitness 最大行是 epoch 146，P/R/mAP50/mAP50-95=`0.36560/0.36931/0.29995/0.18877`；同一 375 张开发 val 的本机统一复评为 `0.28965/0.37252/0.30358/0.19036`，证据在 `优化\l1_reeval_yolo11m_fold02\metrics_per_class_reeval.json`。冻结 198 张 test 未读取。用户已确认 Kaggle GPU 配额耗尽，因此不提交 YOLO26m Fold 2 或 Fold 3；配额恢复后严格按 YOLO26m Fold 2 -> YOLO11m Fold 3 -> YOLO26m Fold 3 的顺序继续。当前仅有 YOLO11m 两个 Fold、YOLO26m 一个 Fold，不能选择最终架构。

- **RYW18 Kaggle YOLO26m Fold 2 已提交（2026-08-01 18:13）**：用户确认 Kaggle GPU 配额已恢复后，私有 Kernel `lly3323w/l1-ryw18-yolo26m-anchorv1-fold02` 已以 version 1 成功推送，并查询到 `RUNNING`。本机 `train_dianjienie_ryw18_anchor.py --model yolo26m.pt --fold 2 --dry-run` 已重新验证固定 `train/val/test=752/375/198`、三集合无交集、1325 图、标签完整和 RYW18 类别顺序。Kernel 复用两份既有私有 Dataset，只有官方模型切换为 `yolo26m.pt`；训练固定为 300 epochs、640、batch 8、SGD、seed 0、patience 100、`close_mosaic=0`、`resume=false`。未启动本机监控、训练或下载进程；完成后必须先归档校验并做 375 张 Fold 2 开发 val 复评，冻结 198 张 test 不访问。

- **RYW18 Fold 2 完成与 Fold 3 并行（2026-08-02 11:15）**：YOLO26m Fold 2 Kernel `lly3323w/l1-ryw18-yolo26m-anchorv1-fold02` 已查询为 `COMPLETE`，本机已启动可续传归档下载；ZIP CRC、权重、固定清单和本机 375 张 val 复评尚未完成，因此不得读取训练期指标或用该权重比较。用户明确提出后续可并行两个模型；基于不可变 RYW18 锚点、不同 Kernel slug 和不同预期 run 目录，YOLO11m Fold 3 `lly3323w/l1-ryw18-yolo11m-anchorv1-fold03` 与 YOLO26m Fold 3 `lly3323w/l1-ryw18-yolo26m-anchorv1-fold03` 已各自以 version 1 推送并确认 `RUNNING`。两者均先通过本机 dry-run：固定 `751/376/198`、无集合交集、官方预训练、300 epochs、640、batch 8、SGD、seed 0、patience 100、`close_mosaic=0`、`resume=false`；冻结 test 持续不访问。F2 下载与 F3 训练无数据或输出目录冲突；F3 完成后也必须逐一归档和同口径本机复评，三折完整前仍不选架构。

- **RYW18 YOLO26m Fold 2 正式归档与复评（2026-08-02 11:35）**：Kernel `lly3323w/l1-ryw18-yolo26m-anchorv1-fold02` 的结果 ZIP 已通过 CRC、30 个文件、必需文件、权重 SHA-256、归档内固定 `752/375/198` 清单校验，归档至 `models\runs\dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_anchorv1_fold02`；ZIP SHA-256=`bd4e6038d39d5b989749fce24aca0f000f7ca5d76de4374847cf0e473548e283`。训练期 best fitness 在 epoch 58，P/R/mAP50/mAP50-95=`0.56126/0.19116/0.25637/0.16023`；同一 375 张开发 val 的本机固定复评为 `0.56254/0.21363/0.20602/0.11342`，证据位于 `优化\l1_reeval_yolo26m_fold02\metrics_per_class_reeval.json`。冻结 198 张 test 未读取。YOLO11m/YOLO26m Fold 3 均仍为并行 `RUNNING`；两个完成并完成本机复评前不比较架构。

- **RYW18 双架构 Fold 3 远端完成并行归档（2026-08-02 21:10）**：Kaggle CLI 已分别确认 YOLO11m Kernel `lly3323w/l1-ryw18-yolo11m-anchorv1-fold03`、YOLO26m Kernel `lly3323w/l1-ryw18-yolo26m-anchorv1-fold03` 均为 `COMPLETE`。两个独立的本机收集器已同时启动，结果 ZIP 分别下载至其 Kernel 专属目录并保留 `.part` 续传文件；这不是本机训练或持续监控器。CRC、必需文件、权重 SHA-256、固定 `751/376/198` 清单及各自 376 张开发 val 复评尚未完成，因此不得读取训练期指标、作架构比较或访问冻结 test。

- **RYW18 Fold 3 归档、复评和三折开发集汇总（2026-08-02 22:56）**：YOLO11m/YOLO26m Fold 3 的结果 ZIP 分别以 SHA-256=`5b2c0002b2e28695a517efc13eb725e5cffeef16f162145a007bc2dfb6cdf491` / `b73d83e62b861fe57fac0743e79def142f30f66a9f03cd5a797f48b74c4fd0cf` 通过 CRC、30 文件、必需文件、权重和固定 `751/376/198` 清单校验，归档至相应 Fold 3 run。两者本机 376 张开发 val 复评 P/R/mAP50/mAP50-95 分别为 YOLO11m `0.56997/0.33965/0.33382/0.20488`，YOLO26m `0.31344/0.31744/0.32710/0.20539`；冻结 test 未读取。六个 run 的本机三折 mAP50-95 均值±样本标准差为 YOLO11m `0.18165±0.02860`，YOLO26m `0.15778±0.04607`，YOLO11m 仅作为 OOF 暂定候选，不能声称多 seed 稳定优势。正式汇总见 `优化/experiments/dianjienie_ryw18_anchor_v1_cv3_summary.md`；下一步是以每图未见 Fold 权重执行 OOF 自动共识，锚点和冻结 test 不改不读。

- **RYW18 三折 OOF 双模型一致性候选（2026-08-02 23:45）**：`优化\generate_l1_oof_consensus.py` 已通过 dry-run 和正式推理，对 1,127 张开发集图片分别使用其未参与训练的 YOLO11m/YOLO26m Fold 权重；六个 checkpoint、`ryw.pt` 和归档 val 清单均与 RYW18 锚点逐项验证。`ryw.pt` 仅记为 `teacher_only` 证据。输出 `优化\l1_oof_consensus_20260802_232333_v1`：9,239 个双 OOF 同类 IoU>=0.55 一致框，5,306 个历史同类支持、316 个历史异类冲突（不自动改类）、3,617 个新候选；其中仅 245 个达到 conf>=0.70 且不与历史框重叠，被加入五列候选视图。开发集历史 15,426 框原始字节均保留为五列输出前缀，新增后为 15,671 框；`labels_with_conf/` 保留 3,617 个新候选的第六列置信度，`decisions.jsonl` 保留全证据。冻结 198 张 test 未读、未推理、未包含。该视图不是人工真值；若作为下一轮数据版本，必须重新跑相同 Fold，旧指标仅作历史参考。

- **RYW18 OOF 候选开发集与 Kaggle 预检（2026-08-03）**：`优化\\dianjienie_ryw18_oofconsensus_v1_dev_only_r1` 已由上述唯一候选来源构建为 1,127 图、1,127 标签、15,671 框的仅开发集硬链接视图；其 245 个新增框不覆盖任何锚点标签，`test.txt` 和冻结 198 图未读、未链接。私有配置 Dataset `lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config` 已 `ready`，只含候选标签、三份 train/val、运行脚本和 manifest，不含图片或测试清单；既有 `lly3323w/dianjienie` 仍只读提供图片。无 GPU Kernel `lly3323w/l1-ryw18-oof-consensus-v1-preflight` 已提交且最近状态为 `RUNNING`，只传递 `--prepare-only`。YOLO11m/YOLO26m Fold 1 的候选 Kernel 包已通过本地语法/元数据检查但尚未提交；预检完成并核验后才可并行提交。不得启动持续监控器，完成后由用户通知再收集。

- **RYW18 OOF 预检 v1 失败与 v2 重提（2026-08-03 01:03）**：v1 的 Kaggle 日志已下载至 `优化\\kaggle_l1_ryw18_oofconsensusv1_preflight\\remote_output_v1\\l1-ryw18-oof-consensus-v1-preflight.log`。两份 Dataset 已成功动态发现，但生成的预检命令遗漏了共享 runner 必填的 `--model`、`--fold`，在参数解析时退出；未请求 GPU、未训练且未写入数据。`prepare_l1_oofconsensus_kaggle_kernel.py` 和预检 Kernel 已补齐 `yolo11m.pt/Fold 1` 参数（仅为 runner 参数校验，仍执行 `--prepare-only`）；本地完整恢复通过 1,127 图、1,127 标签、`oofconsensusv1`、`development_only=true`。实际 Kaggle Kernel id 已对齐自动规范化后的 `lly3323w/l1-ryw18-oof-consensus-v1-preflight`，version 2 已推送且最近状态为 `RUNNING`。预检通过前不得推送 GPU Fold 1。

- **RYW18 OOF 预检通过与 Fold 1 并行提交（2026-08-03 01:14）**：预检 version 2 已 `COMPLETE`；日志确认 Kaggle 挂载了候选配置和只读图片集，并成功恢复 1,127 图、1,127 标签、`data_version=oofconsensusv1`、`development_only=true`、Fold 1。预检只运行 `--prepare-only`，正常结束并非停止错误。私有 GPU Kernel `lly3323w/l1-ryw18-yolo11m-oof-consensus-v1-fold01` 与 `lly3323w/l1-ryw18-yolo26m-oof-consensus-v1-fold01` 已各自成功推送 version 1，提交后均为 `RUNNING`；二者固定 300 epochs、640、batch 8、SGD、seed 0、patience 100、`close_mosaic=0`、`resume=false`，且仅使用开发集 train/val。`collect_l1_kaggle_run.py --development-only` 和 `reeval_l1_yolo_run.py --development-only` 已补齐候选归档/复评支持，后者不读取冻结 test；本地单元/干运行验证通过。无持续监控器，用户通知完成后再收集。

- **RYW18 OOF 候选 Fold 1 归档与配额阻塞（2026-08-03 12:49）**：两个候选 Kernel 均为 `COMPLETE`，并分别通过 29 文件 ZIP、CRC、必需文件、权重、`requested_config.json` 和仅开发集清单校验后归档。相同 376 图 Fold 1 val 的本机复评为：YOLO11m P/R/mAP50/mAP50-95=`0.49024/0.30787/0.31292/0.16513`，YOLO26m=`0.50327/0.32701/0.30722/0.15927`；两份复评均显式 `development_only=true`，冻结测试字段为 `null`。候选标签版本改变了评测标签，不能与 anchor v1 旧指标直接排名，且单 Fold/单 seed 不能选择架构。两个 Fold 2 定义已通过 `752/375/null` dry-run、Kernel 元数据和语法校验，但 Kaggle push 均返回“Maximum weekly GPU quota of 30.00 hours reached”；未产生新 run、未启动监控器，配额恢复后直接重试两个现有 Kernel。
- **RYW18 OOF 候选 CPU 可行性复核（2026-08-03 13:10）**：当前 Fold 2 Kernel 训练路径强制 `enable_gpu=true` 和 `--device 0`，不能直接以 CPU 提交。Fold 1 在 Tesla P100 上实际训练耗时：YOLO11m 最近 epoch 290 时 `36,865.4 s`（约 10.2 小时）、YOLO26m 最近 epoch 209 时 `27,908.6 s`（约 7.8 小时）。因此 CPU 全量 300-epoch 训练既不属于与 GPU 公平的实验配置，也不应预期在可接受的 Kaggle 运行时间内完成；仅可在用户明确授权时创建 CPU 单 epoch 冒烟任务验证启动链路，不能代替候选 Fold 2 正式训练。
- **RYW18 锚点框尺寸诊断更正（2026-08-03 15:17）**：`l1_box_size_diag.py` 已改为默认 development-only，只从 `fold_01/02/03/val.txt` 并集统计 1,127 张开发图和 15,426 框，冻结 test 不读取、不纳入；报告重出至 `优化\l1_box_size_diag\box_size_report.md` / `.json`。抽样 200 张得到中位分辨率 4096x3554，`val_letterbox@640` 主口径缩放系数 0.15625、等效区域 640x555、padding 0x85；`greenstain/wuzi/waterstain` 中位最小边约 5.5/8.6/12.3px，`<32px` 占比约 96.11%/90.93%/84.98%。小目标瓶颈方向成立，但这仍是几何相关性证据，高分辨率是否提高最终 mAP 必须另跑单变量训练验证。
- **L1 后续计划改为 Kaggle-only medium 路线（2026-08-04 更新）**：用户明确本机跑不了 medium 模型，后续 YOLO11m/YOLO26m medium 训练不在本机执行；完整加载 medium 权重的验证也优先使用 Kaggle validation-only Kernel。本机职责限于不加载 medium 权重的脚本生成、语法检查、清单校验、ZIP/哈希/日志解析和报告整理。技术子计划见 `L1_Kaggle高分辨率验证计划.md`；当前默认先完成 OOF 候选 Fold 2/3、三折汇总和固定人工评测，L1-hr `imgsz=1280` 后置。若有现场截止时间，可先训练 anchor v1 640 兜底，但仍需完成 OOF 闭环。

- **RYW18 OOF 评测契约与 Fold 2 启动（2026-08-08 14:37）**：Kaggle 账户设置页在提交前显示 `Kaggle GPU 00:00 / 30 hrs`。`优化\l1_oof_evaluation_contract_v1` 已从唯一 OOF 来源冻结 200 个开发集分组、292 个框级审查单元（100 个新增候选分组、50 个类别冲突分组、50 个历史支持对照分组）；它只读 1,127 张开发图，不读、不链接 198 张 `release-validation`，不修改任何标签，审查结论尚未填写。`lly3323w/l1-ryw18-yolo11m-oof-consensus-v1-fold02` 的本地 Kernel 包与生成器逐字节一致，已成功推送 version 1，随后查询为 `RUNNING`。配置固定为 OOF 开发集 `752/375`、YOLO11m 官方预训练、300 epochs、640、batch 8、SGD、seed 0、patience 100、`resume=false`；未启动本机持续监控。完成后先执行归档校验与同口径 Fold 2 开发集复评，再决定是否串行提交 YOLO26m Fold 2。

- **RYW18 OOF Fold 2 双模型并行（2026-08-08 14:43）**：用户明确确认 Kaggle 可并行两个 GPU 训练。启动 YOLO11m Fold 2 后账户页显示 `00:05 / 30 hrs`；本地 YOLO26m Fold 2 Kernel 脚本 SHA-256=`5aa71cbf4a4a6a834013990af2e2010681d367a2640fc389fcc7440c9a485402`、元数据均与生成器一致。私有 Kernel `lly3323w/l1-ryw18-yolo26m-oof-consensus-v1-fold02` 已成功推送 version 1；随后与 YOLO11m Fold 2 的 CLI 状态均为 `RUNNING`。仅此同 Fold 的两个独立模型/Kernel/run 目录并行，二者仍须独立 ZIP 校验和开发集复评；不并行启动 Fold 3、高分辨率或其他工作线，也不启动本机持续监控。

- **RYW18 YOLO11m OOF Fold 2 归档并流水启动 Fold 3（2026-08-08 21:35）**：YOLO11m Fold 2 Kernel 已 `COMPLETE`，结果 ZIP SHA-256=`c8dc6d28ac551cf3ec4df54210e9b883e6d3b445218c3126f0f9f34f9edf48fb`，通过 29 文件、CRC、必需文件、权重、`requested_config.json` 和 `752/375/null` 开发集清单校验，归档至 `models\runs\dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold02`。训练期最大 mAP50-95 在 epoch 83，P/R/mAP50/mAP50-95=`0.43059/0.22185/0.27236/0.17269`，仅是候选标签训练期指标。当前本机 Python 缺少 `ultralytics`，统一复评改由 Kaggle validation-only 入口完成，未读取 `release-validation`。提交 YOLO11m Fold 3 前 Kaggle 显示 `13:23 / 30 hrs`；本地 F3 脚本/元数据通过生成器和语法校验，脚本 SHA-256=`f3fee5475e22ba4fef606bd510d78fdb9b2a144e06b29def06da54acfe1d40a7`，version 1 已推送并查询为 `RUNNING`。YOLO26m Fold 2 同时仍为 `RUNNING`；最多两个独立槽位，未启动其他工作线或本机监控。

- **RYW18 OOF YOLO26m Fold 2 归档与配额决策（2026-08-09 01:42）**：Kernel `lly3323w/l1-ryw18-yolo26m-oof-consensus-v1-fold02` 已 `COMPLETE` 并归档至 `models\runs\dianjienie_medium_yolo26m_640_kaggle_seed0_cm0_oofconsensusv1_fold02`。ZIP SHA-256=`e5bd8ef88e3ab66bc5d87c70548d1f4b8c8d08d61639603cf8fbc2450a4ae0d6`，CRC、29 个文件条目、必需文件、权重和固定 `752/375/null` 开发集清单均已校验；训练期 mAP50-95 峰值为 epoch 163 的 `0.19229`，仍不可用于架构比较，统一复评必须在 Kaggle validation-only 完成。检查时 GPU 为 `19:58 / 30 hrs`、YOLO11m Fold 3 仍 `RUNNING`；YOLO26m Fold 3 已生成并校验本地包但未提交，等待额度恢复后重新核验。

- **RYW18 OOF YOLO26m Fold 3 已提交（2026-08-09 01:58）**：用户明确确认 Kaggle 会自行完成，要求忽略余量预测直接增加任务。已将本地校验包 `lly3323w/l1-ryw18-yolo26m-oof-consensus-v1-fold03` 推送为 version 1，随后查询为 `RUNNING`；固定 OOF 开发集 Fold 3=`751/376`、官方 `yolo26m.pt`、300 epochs、640、batch 8、SGD、seed 0、patience 100、`resume=false`。YOLO11m Fold 3 同时仍 `RUNNING`，总并行数为 2；未启动本机训练或持续监控。完成后两者必须独立归档、校验和进行 Kaggle validation-only 统一复评。

- **RYW18 OOF 三折归档与统一复评完成（2026-08-09）**：YOLO11m/YOLO26m Fold 3 均为 `COMPLETE` 后已独立下载、CRC/必需文件/权重/配置/固定 `751/376` 清单校验并归档。Fold 3 ZIP SHA-256：YOLO11m=`199c86ef7572e76e7409b566cbb3deec186cbad0212fd4ac0672164bebba5c58`，YOLO26m=`3fbeadb462ecfd0c7955de9ae0603c37bfb953eec9d3dfad6e31fdbe97a0d8d4`。六个 `best.pt` 均使用同一 `E:\PythonProject\models\.venv`（Python 3.10.11、Ultralytics 8.4.66、Torch 2.6.0+cu124）在固定开发 Fold 做 `development_only=true` 复评，不读取 `release-validation`。候选三折 P/R/mAP50/mAP50-95 均值±样本标准差：YOLO11m=`0.46927±0.03975/0.31066±0.08001/0.31005±0.03907/0.18539±0.02939`；YOLO26m=`0.50154±0.04730/0.31034±0.02800/0.31220±0.04135/0.18166±0.03795`，详见 `experiments\dianjienie_ryw18_oofconsensus_v1_cv3_summary.md`。差异小于折间波动且仅单 seed，不能选模型；候选标签自评也不能替代 `l1_oof_evaluation_contract_v1` 的固定人工评测。
- **RYW18 OOF 无人工自动闸门（2026-08-09）**：用户否决人工评测后，6 个候选权重已在原始历史标签和同一开发 Fold 上自动复评，所有复评均显式 `development_only=true`，未读取 `release-validation`。闸门要求每个架构平均 mAP50-95 相比相应 anchor 不下降超过 0.01，且至少 2 个 Fold 非回退；YOLO11m 变化为 `-0.00030`（2/3 Fold 非回退），YOLO26m 为 `+0.01956`（2/3 Fold 非回退），均通过。YOLO11m 在候选标签自评和历史锚点复评的绝对均值都略高，被选为无人工候选训练模型；报告 `优化\l1_auto_gate_oofconsensusv1_20260809\report.md`。这是事后固定的工程安全规则，不是人工真值证明或统计显著性检验。

### 工具包


- 根目录：`E:\PythonProject\Tools`。
- `ModelAutoLabelTool`：批量自动标注（适配 YOLOv5 / Ultralytics / RF-DETR）；伪标签 v1 教师权重 `weights\yolov5\ryw.pt`（另有 `slf.pt`）。
- `ImageProcessTool`：几何不变的图像预处理，便于后续自动标注。

### 数据集改动原则

- **尽量不改原数据集**；新标注写版本化目录；新视图用软连接或路径清单。
- 两检测集中的 `labels_pseudo_v1`、`labels_pseudo_v1_raw` 和 `manifests` 已按用户要求删除；原 `labels/` 保留。审查脚本和历史报告仍在文档目录，仅供追溯。
- 详见 `伪标签审查流程记录_v1.md` 与 `工作日志.md`。

### 项目代码（JinchuanPythonWebapi）

- 仓库：`E:\PythonProject\ImetricCodes\JinchuanPythonWebapi`。
- **主分支 `308`**：现场/项目主线，跟踪 `origin/308`。
- **优化分支 `优化分支`**：用户未来改进用的本地开发分支，**尚未 push**，无 upstream。
- 改 WebAPI 代码前必须确认当前 git 分支；优化工作应在 `优化分支` 上进行，不得默认改 `308`。
- 跨会话进度以 `工作日志.md` 为准。

### 其他

- 当前存在稀有类别、标签异常、缺失标签和划分问题；数据清洗优先于大规模调参。
- 正式比较对象为 YOLO11m、YOLO26m 与待独立重训的真 YOLOv5m；现有项目权重是 YOLOv5-s 历史效果参考。
- 新数据计划采用三线线扫三光方案 A；硬件时序、行距、图像高度和通道映射在准确型号确认前均属于待验证项。
- `Instruction.md`、`nickel_defect_system_execution_doc.md`、`nickel_defect_fullstack_guide.pdf` 仅为历史背景。

## Claude 输出要求

- 面向用户使用中文，结论简洁并注明依据文件或实验产物。
- 修改前说明计划，修改后列出文件、验证结果和未解决风险。
- 不虚构运行结果、模型指标、文件路径或官方参数。
- 默认不修改数据集原件；需要写入时创建版本化输出并保留映射清单。
- 不把 `nickel_defect_fullstack_guide.pdf` 中标注为示例或 AI 生成的内容当作当前配置。
- 对话中的进度说明不能替代 `工作日志.md` 落盘。

## 最小入口（其余见 README）

| 需要 | 文件 |
| --- | --- |
| 总索引 | `README.md` |
| 强制规则 | `AGENTS.md` |
| 进度交接 | `工作日志.md` |
| 检测执行 | `镍板缺陷模型优化操作手册.md` |
| 实验填写 | `模型优化实验记录模板.md` |
