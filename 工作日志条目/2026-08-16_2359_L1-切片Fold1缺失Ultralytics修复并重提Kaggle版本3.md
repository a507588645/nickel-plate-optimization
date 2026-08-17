## 2026-08-16 23:59 | L1-切片 Fold1 缺失Ultralytics修复并重提 Kaggle 版本 3

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex |
| 任务目标 | 诊断用户发现的 Kaggle 切片任务再次停止原因，并仅基于 version 2 可见日志修复远端运行时后重提。 |
| 状态 | 进行中：同一 Kaggle Kernel 的 version 3 已成功推送，提交后及 45 秒启动期复查均为 `RUNNING`。 |
| 关联用户请求 | 用户反馈修复后的任务仍然停止。 |

### version 2 失败证据与根因

- `lly3323w/l1-ryw18-y11m-oof-v1-tile1024-o25-b8-e300-f01` version 2 在 GPU P100 上运行 391.6 秒后为 `ERROR`，Kaggle 输出显示 865.54 MB。
- 日志确认配置恢复、图片与固定 Fold 1 基线权重发现、751/376 开发集清单以及 6,919 个训练输入/9,400 个验证切片计划均正常；version 2 已实际开始构建派生切片视图。
- 在 `l1_tile_pipeline.py:844` 导入训练库时抛出 `ModuleNotFoundError: No module named 'ultralytics'`。此时尚未进入 `model.train()`，没有模型权重、训练曲线或评测指标。
- 原因是 Kaggle 最新容器不预装 `ultralytics`；这与 version 1 的目录初始化错误不同，也不是 GPU 配额、数据集或标签错误。

### 修复与验证

- `prepare_l1_tile_kaggle_kernel.py`：新增 `install_dependencies()`，复用已完成 L1 最终 Kaggle 任务的固定 `torch==2.6.0`、`torchvision==0.21.0`、`numpy==1.26.4`、`ultralytics==8.4.66` 组合，并在切片生成前验证 CUDA GPU 可用性。使用 `--no-cache-dir`，避免保留重复安装缓存。
- `kaggle_l1_ryw18_y11m_oof_v1_tile1024_o25_b8_e300_f01_r3/`：version 3 本地 Kernel 包；源生成器和生成脚本均通过 `python -m py_compile`。
- `kaggle kernels push -p E:\PythonProject\优化\kaggle_l1_ryw18_y11m_oof_v1_tile1024_o25_b8_e300_f01_r3` 已返回 version 3 推送成功；两次 `kaggle kernels status` 查询均为 `KernelWorkerStatus.RUNNING`，第二次在提交 45 秒后。

### 边界与结论

- version 3 不改变数据版本、类别、Fold、切片参数、训练超参数、固定基线、标签或评测器；仅补齐远端运行依赖并将失败前置。
- 未修改源图片/源标签，未读取或挂载历史 `release-validation`，未增加人工审核或本机持续监控。

### 未完成项与下一步

1. 等待 version 3 正常完成后下载并核验 ZIP、权重、配置和开发集边界。
2. 仅在归档完整后读取同一原图级评测器的基线/切片对比和逐类结果。

### 风险与待确认

- 依赖安装需要 Kaggle Internet 和临时磁盘，version 3 尚未完成，不能宣称已开始或完成训练。
- 即使完成，单 Fold、单 seed 和 `auto_only` 标签仍只能得到研究性结论。
