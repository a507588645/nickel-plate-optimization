# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo26m_fold02_kaggle_seed0` |
| 日期 | 2026-08-08 至 2026-08-09 |
| 执行人 | Codex |
| 运行环境 | Kaggle 私有 Script Kernel |
| Kernel | `lly3323w/l1-ryw18-yolo26m-oof-consensus-v1-fold02` |
| 当前状态 | 已完成：结果已归档并通过结构校验，2026-08-09 已完成固定开发集 Fold 2 统一复评 |
| 证据等级 | `provisional`：归档和统一复评完整；固定人工评测未完成，不能用于最终模型选择 |

## 2. 假设与比较边界

- 目标：在与 YOLO11m Fold 2 完全相同的 OOF 候选开发集和固定划分上，补齐 YOLO26m 的 Fold 2 证据。
- 本次经用户明确许可与 YOLO11m Fold 2 并行；并行仅改变日历时间，不改变数据、参数、模型权重来源或总 GPU 配额。
- 两个 Kernel 与预期 run 目录彼此独立，任何一项完成后仍必须单独归档、校验和开发集复评。

## 3. 数据信息

| 项目 | 内容 |
| --- | --- |
| 数据集 | `dianjienie` RYW18 OOF 自动共识候选开发集 |
| 数据版本 | `oofconsensusv1` / `dianjienie_ryw18_oofconsensus_v1_dev_only_r1` |
| Kaggle 图片源 | `lly3323w/dianjienie`，只读图片仓库 |
| Kaggle 配置源 | `lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config` |
| train / val 图片数 | 752 / 375 |
| 冻结 test / release-validation | 不包含、不读取、不评估 |
| 类别数 | 18 类 RYW18 |
| 标签版本 | 15,426 个历史框 + 245 个 OOF accepted 新框 = 15,671 框；非人工真值 |
| 固定评测契约 | `E:/PythonProject/优化/l1_oof_evaluation_contract_v1`，仅开发集、未填写人工决定 |

## 4. 训练配置

| 参数 | 值 |
| --- | --- |
| model / pretrained weights | `yolo26m.pt` / 官方预训练，不使用 `ryw.pt` 初始化 |
| epochs / imgsz / batch | 300 / 640 / 8 |
| optimizer | SGD |
| lr0 / lrf | 0.01 / 0.01 |
| momentum / weight_decay | 0.937 / 0.0005 |
| warmup_epochs / patience | 3.0 / 100 |
| seed / deterministic | 0 / true |
| close_mosaic / resume | 0 / false |

## 5. 提交前验证与归档核验

- 在第二个任务提交前，Kaggle 账户设置页显示 GPU `00:05 / 30 hrs`。
- 本地 Kernel 脚本与 `prepare_l1_oofconsensus_kaggle_kernel.py` 的 Fold 2 渲染结果逐字节一致，脚本 SHA-256=`5aa71cbf4a4a6a834013990af2e2010681d367a2640fc389fcc7440c9a485402`。
- Kernel 元数据与生成器一致；两个私有 Dataset 引用保持不变，GPU 已启用。
- `kaggle kernels push` 成功返回 version 1；2026-08-09 查询状态为 `COMPLETE`。
- `collect_l1_kaggle_run.py --development-only --install` 已归档并安装结果 ZIP：SHA-256=`e5bd8ef88e3ab66bc5d87c70548d1f4b8c8d08d61639603cf8fbc2450a4ae0d6`，ZIP CRC 通过，含 29 个文件条目（另有 3 个目录条目）。
- 10 个必需文件均存在，`best.pt` / `last.pt` SHA-256 分别为 `eab8c360e5e4995c78018196a8751462cd1ed187115821da995ca28b8f245a8f` / `769d6b9dfdbd009cd0cf84fbcfcf1e1efd72b4cdc933ffeb339aec5a463aea38`。
- 归档配置为 `development_only=true`、Fold 2、`752/375/null`；归档 train/val 清单按图片名与本地固定 Fold 2 清单逐项一致。

## 6. 结果与后续核验

- `results.csv` 最后记录为 epoch 263，累计时间 `33,968.8 s`（约 9 小时 26 分）；这只是训练期原始证据。
- 训练期 mAP50-95 峰值位于 epoch 163，P/R/mAP50/mAP50-95=`0.51955/0.27525/0.30411/0.19229`。候选标签版本改变了评测标签，此数字不能与 anchor 或另一个架构直接排名。
- 2026-08-09 使用与 Fold 1 相同的 `E:/PythonProject/models/.venv` 完成 development-only 固定 Fold 2 复评：P/R/mAP50/mAP50-95=`0.45340/0.32600/0.27356/0.16023`。入口为 `reeval_l1_yolo_run.py --development-only`，未读取 `release-validation`。
- 复评证据：`E:/PythonProject/优化/l1_reeval_yolo26m_oofconsensusv1_fold02/metrics_per_class_reeval.json`；环境为 Python 3.10.11、Ultralytics 8.4.66、Torch 2.6.0+cu124、`imgsz=640`、`batch=4`、`conf=0.001`、`iou=0.7`、`max_det=300`。

## 7. 风险与决策

- 2026-08-09 配额检查为 GPU `19:58 / 30 hrs`，且 YOLO11m Fold 3 仍在运行；本任务约 9 小时 26 分的实际训练时间表明不能安全再开 YOLO26m Fold 3。
- 2026-08-09 用户明确确认 Kaggle 会自行完成并要求直接增加任务；YOLO26m Fold 3 已推送 version 1 并为 `RUNNING`，与 YOLO11m Fold 3 构成两个并行任务。
- 单个 Fold / 单个 seed 不能选择最终架构。
- 该 Fold 已纳入候选版三折统一汇总；固定人工评测契约尚未填写，候选标签自评不能证明其标签质量优于历史标签锚点。
- `youpao`、`fanxitong` 无监督实例；`heibian`、`mijixiaokeng` 等稀有类可能不可评。
