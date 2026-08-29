# 模型优化实验记录

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 实验编号 | L2-LGBM-Y11M640-20260828-v1 |
| 所属主线 | L2 电积镍分类等级 |
| 日期 | 2026-08-28 |
| 执行人 | Codex |
| 实验状态 | complete |
| 证据等级 | provisional |
| 父计划 / 决策闸道 | 用户 2026-08-28 决策：用冻结 YOLO11m@640 重提 43 列检测特征并重训 LightGBM；企业标准闸门仍未确认 |
| 研究假设 | 以冻结 YOLO11m@640 替换历史 `ryw.pt@1536` 后，43 列检测特征与既有 20 列 `tx__` 可形成可序列化的 63 列等级模型输入。 |
| 只改变的主要变量 | detector-derived BASE 43 列的来源：历史 `ryw.pt@1536` → 冻结 YOLO11m ONNX `imgsz=640`。20 列 `tx__`、划分、seed 和 LightGBM 参数保持不变。 |
| 停止条件 | 不调参；valid 用于早停和拒判阈值，test 只在最终目标配置上作一次性报告。 |
| 运行环境 | 本地 CPU |
| GPU | 未使用 |
| Python | 3.11.9（LightGBM 环境）；推理另用 WebAPI `.venv` Python 3.11.14 |
| Torch | 仅 detector 推理环境记录：WebAPI `.venv` 2.12.1；未训练 detector |
| Ultralytics | 8.4.66（WebAPI `.venv`） |

## 2. 数据信息

| 项目 | 内容 |
| --- | --- |
| 数据集 | 电积镍等级分类视图 |
| 数据版本 | `archives/dianjinie_grade_classification_v1`，1072 张 |
| data.yaml | 不适用；文件夹等级视图 + `manifests/samples.csv` |
| train 图片数 | 750 |
| val 图片数 | 161 |
| test 图片数 | 161 |
| 类别数与顺序 | 4；`3#`=0、`9950`=1、`9996`=2、`bad`=3 |
| 是否清理重复图 | 是；沿用归档视图，dup-group 跨 split=0 |
| 是否通过标签审计 | 等级视图身份/路径审计通过；企业等级语义仍未确认 |
| split manifest / 文件哈希 | `archives/dianjinie_grade_classification_v1/manifests/samples.csv`；按 SHA-256 join 1072/1072 |
| 冻结测试集访问状态 | 已评估；valid 选择/早停/定阈值完成后，目标配置 test 一次性评估 |
| 数据变更说明 | 原始 archives、datasets、models、WebAPI 均未写入；新特征和模型只写当前输出目录 |

## 3. 训练配置

| 参数 | 值 |
| --- | --- |
| model | LightGBM multiclass |
| pretrained weights | 不适用 |
| epochs | num_boost_round=400 上限 |
| completed epochs | best_iteration=186 |
| imgsz | detector extraction=640 |
| batch | 不适用 |
| optimizer | LightGBM 默认 boosting |
| lr0 | learning_rate=0.05 |
| lrf | 不适用 |
| momentum | 不适用 |
| weight_decay | 不适用 |
| warmup_epochs | 不适用 |
| patience | early_stopping_rounds=40 |
| seed | 42 |
| deterministic | true |
| box | 不适用 |
| cls | 不适用 |
| dfl | 不适用 |
| mosaic | 不适用 |
| close_mosaic | 不适用 |
| freeze | 不适用 |
| 其他变量 | num_leaves=8，max_depth=4，min_data_in_leaf=10，feature_fraction=0.9，bagging_fraction=0.9，bagging_freq=1；63 列=43 BASE+20 tx |
| 预计 GPU 小时 | 0 |
| 实际 GPU 小时 | 0 |
| 资源约束或中断原因 | CPU-only；无训练中断。首次 Ultralytics 推理触发自动依赖更新后，模型环境 ONNX Runtime 出现 Python/二进制混杂；后续推理改用已存在的 WebAPI `.venv`，未安装新包。 |

## 4. 结果

| 指标 | 数值 |
| --- | ---: |
| best epoch | 186 boosting iteration |
| valid accuracy | 0.7453 |
| valid macroF1 | 0.7058 |
| test accuracy | 0.7453 |
| test macroF1 | 0.6797 |
| 训练时长 | LightGBM 训练约数秒；完整提取约 17.4 分钟 |
| 单图推理时长 | detector 全量平均约 0.97 图/秒（含历史 plate-area 计算） |
| 峰值显存 | 不适用 |
| 参数量 | 见 `lgbm_model.txt` |
| 模型大小 | 744947 bytes |
| 产物完整性 | 完整 |

## 5. 逐类别指标

| split | 类别 | Precision | Recall | F1 | 样本数 |
| --- | --- | ---: | ---: | ---: | ---: |
| valid | 3# | 1.0000 | 0.5714 | 0.7273 | 7 |
| valid | 9950 | 0.7805 | 0.7805 | 0.7805 | 82 |
| valid | 9996 | 0.7101 | 0.7206 | 0.7153 | 68 |
| valid | bad | 0.5000 | 0.7500 | 0.6000 | 4 |
| test | 3# | 0.7500 | 0.4286 | 0.5455 | 7 |
| test | 9950 | 0.7750 | 0.7561 | 0.7654 | 82 |
| test | 9996 | 0.7067 | 0.7794 | 0.7413 | 68 |
| test | bad | 1.0000 | 0.5000 | 0.6667 | 4 |

完整混淆矩阵见 `metrics_v3_texture_only.json`。

## 6. 与基线比较

| 项目 | 本实验 | 历史参考 | 差值/说明 |
| --- | ---: | ---: | --- |
| test accuracy | 0.7453 | 0.7764 | -0.0311；历史为 `ryw.pt@1536`，仅作历史参考 |
| test macroF1 | 0.6797 | 0.6631 | +0.0166；不可视为同一 detector 条件下的直接排名 |
| coverage 0.70 threshold | 0.7362 | 0.7397 | 阈值均由各自 valid 置信度分位得到 |
| coverage 0.70 test coverage | 0.7391 | 0.7140 | 仅报告，不调参 |
| coverage 0.70 accepted accuracy | 0.8487 | 0.8696 | 历史值仅供追溯 |

## 7. 结论

本次只改变的主要变量：

```text
使用冻结 YOLO11m ONNX（SHA-256=901ccf6006f44689483aaf6f0d1d1b8c18132a85c7d881be7e62fcdccc255b21，imgsz=640，conf=0.25，iou=0.45，CPU）重新生成 43 列 detector BASE；复用同一 1072 张图的 20 列 tx__；LightGBM 配置、split 和 seed=42 不变。
```

结果是否有效：需要复验。工程链路、schema、指标和序列化均通过；企业判级标准和外部新批次泛化尚未验证。

是否进入下一轮：否。下一轮应先完成企业标准确认和 WebAPI 集成验收，不在本实验上继续调参。

决策闸道结果：仅作参考

是否阻塞主线交付：否；WebAPI 集成另行执行，本实验不改 WebAPI。

证据边界：

```text
固定 1072 图等级视图；检测器输出不是人工逐框真值；3# 和 bad test 各只有 7/4 张；历史 ryw.pt@1536 指标不是本次模型目标；全部状态为 unconfirmed_pending_enterprise。
```

失败原因或风险：

```text
等级标签企业语义未书面确认；等级与拍摄批次/纹理伪相关未排除；首次错误环境触发了模型 venv 的 Ultralytics 自动更新，当前该 venv 的 ONNX Runtime 需要后续环境治理。
```

下一步操作：

```text
由 Kimi 复核产物和报告；随后另行编写 WebAPI 端特征提取、LightGBM 推理、拒判标记和契约接入任务。不要把本实验状态改为企业已确认。
```

## 8. 文件位置

| 文件 | 路径 |
| --- | --- |
| best.pt | 不适用 |
| last.pt | 不适用 |
| args.yaml | 不适用；训练参数见 `deploy_contract.json` 和 `serialization_summary.json` |
| results.csv | 不适用；指标见 `metrics_v3_texture_only.json` |
| metrics.json | `dianji_grade_y11m640_v1/metrics_v3_texture_only.json` |
| 训练日志 | `dianji_grade_y11m640_v1/report.md` |
| LightGBM 模型 | `dianji_grade_y11m640_v1/lgbm_model.txt` |
| 部署契约 | `dianji_grade_y11m640_v1/deploy_contract.json` |
| Kaggle URL | 不适用 |
