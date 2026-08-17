# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | `dianjienie_ryw18_oofconsensus_v1_yolo11m_tile1024o25_focus6_mix_fold01_kaggle_seed0` |
| 所属主线 | L1 |
| 日期 | 2026-08-16 至 2026-08-17 |
| 执行人 | Codex |
| 实验状态 | `interrupted_pending_evaluation`；version 3 已完成 109 轮验证后在第 110 轮中断，检查点与训练曲线已回收，原图级切片评测尚未运行 |
| 证据等级 | `inconclusive`；已有中断训练期指标，但尚无同一原图级评测器的基线/切片对比 |
| 父计划 / 决策闸道 | G4 后的小目标切片研究，独立于已停止的 1280 整图方向 |
| 研究假设 | 对 OOF 自动候选开发集进行混合整图/切片训练，并将切片预测映射回原图后评测，可改善小目标检测。 |
| 主要变量 | 训练样本视图（751 张整图加选择性 1024 切片）及切片推理；不是与历史 640 Ultralytics 指标可直接排名的严格单变量试验。 |
| 运行环境 | Kaggle 私有 Script Kernel，GPU |

## 2. 数据信息

| 项目 | 内容 |
| --- | --- |
| 数据集 | `dianjienie` RYW18 OOF 自动共识候选开发集 |
| 数据版本 | `oofconsensusv1` / `dianjienie_ryw18_oofconsensus_v1_dev_only_r1` |
| 标签属性 | `auto_only` 五列候选标签；不等同人工真值 |
| Kaggle 图片源 | `lly3323w/dianjienie`，只读 |
| Kaggle 配置源 | `lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config`，在 working 目录按 manifest 重建临时目录结构 |
| Fold 1 整图 train / val | 751 / 376 |
| 混合训练输入 | 751 张整图 + 4,935 张焦点切片 + 1,233 张确定性抽样负切片 = 6,919 |
| 验证切片数 | 9,400；仅用于把预测映射回原图坐标后评测 |
| 切片规格 | 1024 px，重叠 25%，框可见面积阈值 0.50 |
| 焦点类别 | `wuzi, waterstain, airhole, bubble, greenstain, watermark` |
| 冻结历史测试集 | 不包含、不挂载、不读取、不评估 |

大框或在任何切片中可见面积不足 0.50 的焦点框保留在整图训练样本中，避免产生截断伪标签。所有切片及其标签只写入 Kaggle working 目录。

## 3. 训练与评测配置

| 参数 | 值 |
| --- | --- |
| model / 预训练 | `yolo11m.pt` / 官方预训练；不使用 `ryw.pt` 初始化 |
| epochs / imgsz / batch | 300 上限 / 640 / 8 |
| optimizer / seed / patience | SGD / 0 / 100 |
| 其他训练边界 | `resume=false`、`close_mosaic=0`、仅 Fold 1 开发集 |
| 固定基线权重 | 同一 Fold 的 640 OOF 训练 `best.pt`，SHA-256=`b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce` |
| 切片 NMS | 回映射至原图后按类别全局 NMS，IoU=0.70 |
| 评测候选阈值 | 生成 AP 时 0.001；报告 P/R 时 0.25 |
| AP 口径 | 原图级 101 点精度包络；不与历史 Ultralytics 训练期 mAP 直接排名 |

完成后将分别计算 `baseline_full`、`baseline_tiled`、`tile_model_full`、`tile_model_tiled`。主比较是同一自定义原图评测器中的 `tile_model_tiled - baseline_full`，并保留逐类结果。

## 4. 已完成预检与当前状态

- 本机通过配置恢复、划分、标签和数量预检；随后 Kaggle CPU 预检 `lly3323w/l1-ryw18-oof-v1-tile1024-o25-preflight-fold01` version 1 已 `COMPLETE`。
- 远端预检的 Fold 1 清单哈希为 train=`dde70976f1864ebf6022f2595f33ec32c9b039d2ddbbd6946ab0ae24c30aa535`、val=`dc2fc3162b0368a5bc193333c2e5457f666ad5a4f3144eb05e970ef590c744a1`，与本地计划一致。
- GPU Kernel `lly3323w/l1-ryw18-y11m-oof-v1-tile1024-o25-b8-e300-f01` version 1 在 33.1 秒后为 `ERROR`。Kaggle 可见日志显示脚本已创建 `output_root`，随后 `prepare_runtime_view()` 又将该目录存在视为错误，抛出 `FileExistsError`；发生在 `model.train()` 之前，未产生模型、训练曲线或评测结果。
- `l1_tile_pipeline.py` 已修复为只接受主入口刚创建的空 `output_root`，仍拒绝非空目录，并显式创建 `dataset_view`。version 2 因此完成了切片视图生成，Kaggle 显示输出 865.54 MB，但在 391.6 秒、调用 `model.train()` 前因最新容器缺少 `ultralytics` 抛出 `ModuleNotFoundError`，仍未产生模型或指标。
- `prepare_l1_tile_kaggle_kernel.py` 已复用已完成 L1 最终任务的固定运行时：在任何切片生成前安装并验证 `torch==2.6.0`、`torchvision==0.21.0`、`numpy==1.26.4`、`ultralytics==8.4.66` 和 GPU 可用性。
- version 3 实际训练完成第 109 轮验证后，在第 110 轮中被取消；当前 Kaggle 状态为 `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`。日志末尾没有训练异常栈；结合约 11 小时 58 分的训练时长，推断为会话上限相关取消，但取消来源未能从日志单独证明。
- 已逐文件回收并校验 `args.yaml`、`results.csv`、`best.pt`、`last.pt`。`best.pt` SHA-256=`c4a903e8e8438798d44fc0548d666923a8728509f012d124101a8b980d7d875e`，checkpoint epoch=74，对应第 75 轮最高训练期 P/R/mAP50/mAP50-95=`0.57318/0.26795/0.29081/0.15178`；`last.pt` SHA-256=`882e952b3d96243b03564706856f5ffa9821d79357714c1945b2893b0dde76dd`，checkpoint epoch=108。训练曲线共 109 行，未产生 `requested_config.json`、`tile_evaluation/` 或最终 ZIP。
- 新增 `evaluate` 模式和检查点打包器，独立运行固定基线与回收 `best.pt` 的四组同口径报告，不重建训练视图、不训练、不读取历史冻结集。当前 Kaggle API 返回周 GPU 30 小时额度耗尽，评测尚不能提交。

## 5. 文件位置与归档入口

| 文件 | 路径 |
| --- | --- |
| 切片流水线 | `E:/PythonProject/优化/l1_tile_pipeline.py` |
| Kaggle Kernel 生成器 | `E:/PythonProject/优化/prepare_l1_tile_kaggle_kernel.py` |
| 基线权重包生成器 | `E:/PythonProject/优化/prepare_l1_tile_baseline_weights.py` |
| GPU Kernel 定义 | version 1：`E:/PythonProject/优化/kaggle_l1_ryw18_y11m_oof_v1_tile1024_o25_b8_e300_f01/`；version 2：`E:/PythonProject/优化/kaggle_l1_ryw18_y11m_oof_v1_tile1024_o25_b8_e300_f01_r2/`；修复后的 version 3：`E:/PythonProject/优化/kaggle_l1_ryw18_y11m_oof_v1_tile1024_o25_b8_e300_f01_r3/` |
| version 1 失败输出 | `E:/PythonProject/优化/kaggle_l1_ryw18_y11m_oof_v1_tile1024_o25_b8_e300_f01/remote_output_error_v1/` |
| version 2 本地部分输出 | `E:/PythonProject/优化/kaggle_l1_ryw18_y11m_oof_v1_tile1024_o25_b8_e300_f01_r2/remote_output_error_v2/` |
| version 3 中断证据 | `E:/PythonProject/优化/kaggle_l1_ryw18_y11m_oof_v1_tile1024_o25_b8_e300_f01_r3/recovered_interrupted_v3/` |
| 检查点 Kaggle 打包器 | `E:/PythonProject/优化/prepare_l1_tile_interrupted_checkpoints.py` |
| 本机预检 | `E:/PythonProject/优化/l1_tile_preflight_flatpayload_smoke_20260816/preflight_report.json` |
| Kaggle CPU 预检输出 | `E:/PythonProject/优化/kaggle_l1_ryw18_oof_v1_tile1024_o25_preflight_fold01/remote_output_v1/` |
| GPU 任务 | `https://www.kaggle.com/code/lly3323w/l1-ryw18-y11m-oof-v1-tile1024-o25-b8-e300-f01` |
| 中断恢复步骤 | GPU 额度恢复后先上传回收检查点私有 Dataset，再运行 `python prepare_l1_tile_kaggle_kernel.py --mode evaluate` 生成独立评测 Kernel；正常完成后再下载评测 ZIP |

## 6. 下一步与风险

1. 等 Kaggle GPU 周额度恢复后，上传回收的 `best.pt`/`last.pt` 私有检查点包并提交独立 `evaluate` Kernel；不重新训练。
2. 下载四份原图级评测报告和逐类结果；只有同口径收益明确，才评估用 `last.pt` 分段续训是否值得。
3. 不再访问历史 `release-validation`，不因本实验改变既有 640 自动候选交付或 ONNX 产物。

风险：切片评测是自定义的原图映射口径，必须始终将基线和候选放在同一评测器内比较；无人工 OOF 标签仍可能包含系统性标签误差。
