# L1 Kaggle 高分辨率验证计划

更新时间：2026-08-03 15:33

## 结论

继续 L1 检测线，但下一步优先验证高分辨率路线，不再建议本机跑 medium 模型。

当前 L1 的主要不确定性不是“换哪个 medium 架构”，而是 `imgsz=640` 是否把小目标压得过小。更正版框尺寸报告显示，在 `val_letterbox@640` 下：

- `greenstain` 中位最小边约 5.5px，`<32px` 占 96.11%。
- `wuzi` 中位最小边约 8.6px，`<32px` 占 90.93%。
- `waterstain` 中位最小边约 12.3px，`<32px` 占 84.98%。

因此 L1-hr 是当前最值得花 Kaggle 配额验证的单变量方向。

## 硬约束

- 本机不跑 YOLO11m/YOLO26m medium 训练。
- 需要加载 medium 权重做完整验证时，优先走 Kaggle validation-only Kernel；本机只做脚本生成、清单校验、ZIP/哈希/日志解析和报告整理。
- 冻结 198 张 test 不参与调参、早停、模型选择、阈值选择或 L1-hr 试验。
- 不把 OOF 候选五列视图当人工真值；跨标签版本指标不直接排名。
- Kaggle GPU 配额有限时，优先把配额用于回答最大不确定性：`640 -> 1280` 是否显著提升开发集 mAP。

## 阶段 1：L1-hr 单 Fold 试验

目标：验证提高输入分辨率是否能显著改善小目标 AP 和整体 mAP50-95。

推荐首跑配置：

- 数据版本：`E:\PythonProject\datasets\dianjienie_ryw18_anchor_v1`。
- Fold：先用 Fold 1。
- 模型：优先 `yolo11m.pt`，因为 anchor v1 三折均值暂优。
- 平台：Kaggle GPU。
- `imgsz=1280`。
- `epochs=300`、`optimizer=SGD`、`patience=100`、`seed=0`、`close_mosaic=0` 保持 640 基线一致。
- `batch`：先尝试 4；若 Kaggle OOM，降到 2，并把该 run 标记为资源约束实验，不与 batch=8 的 640 基线做严格单变量宣称。

对比口径：

- 只比较同一 Fold、同一数据版本、同一验证清单、同一 Ultralytics/Kaggle 环境下的结果。
- 主比较对象是 anchor v1 YOLO11m Fold 1 的 640 Kaggle 结果和 L1-hr 1280 Kaggle 结果。
- 重点看 `mAP50-95`、`mAP50`、Recall，以及 `wuzi/waterstain/greenstain/airhole/bubble/watermark` 的逐类 AP。

决策门槛：

- 若 Fold 1 的 `mAP50-95` 绝对提升 >= 0.02，且小目标类 AP/Recall 同向改善，则继续跑 1280 的 Fold 2/3。
- 若提升在 0.01 以内，或只表现为单类偶然波动，则暂不铺开 1280 三折，转向切片或标签复核。
- 若 1280 OOM 或耗时不可接受，改测 `imgsz=960` 或切片方案，不强行消耗整周配额。

## 阶段 2：OOF 候选线排队

OOF 候选线不取消，但不应抢占 L1-hr 首次验证配额。

- Kaggle 配额恢复后，若只能跑一个 medium 任务，优先提交 L1-hr YOLO11m Fold 1。
- 若配额足够，再继续重试两个已准备好的 OOF 候选 Fold 2 Kernel。
- OOF 候选至少完成三折后，才能判断候选标签路线是否优于 anchor v1。

## 阶段 3：若高分辨率有效

- 完成 YOLO11m 1280 三折开发集验证。
- 只在 YOLO11m 1280 三折结果稳定后，才考虑是否补 YOLO26m 1280。
- 仍不访问冻结 test，直到最终模型、输入尺寸和阈值策略全部固定。

## 阶段 4：若高分辨率无效

优先转向以下路线，而不是继续盲目换 medium 模型：

- 生成切片训练/推理视图，专门解决 `greenstain/wuzi/waterstain` 的极小框。
- 做人工复核优先级清单，先查高频小目标类漏标、错标和框尺寸偏差。
- 做逐类阈值/NMS 标定，但必须在同一批人工审查过的固定开发评测集上对比。

## 当前下一步

等待 Kaggle GPU 配额可用后，先生成并提交 `yolo11m.pt + anchor_v1 + Fold 1 + imgsz=1280` 的 Kaggle Kernel。提交前只在本机做不加载 medium 权重的脚本语法、配置和清单校验。
