## 2026-08-28 14:44 | L1 YOLO11m Freeze10 成对 Fold 1 训练已提交 Kaggle

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex |
| 任务目标 | 在不触碰已发布 ONNX、冻结测试集或历史 `release-validation` 的前提下，执行用户明确授权的 YOLO11m 冻结层独立成对消融。 |
| 状态 | 进行中 |
| 关联用户请求 | 用户询问是否可通过冻结层重新获得 YOLO11m ONNX，并于 2026-08-28 回复“允许”。 |

### 改动文件

- `train_l1_freeze_two_stage.py`：新增固定 20+63 epoch 的两阶段训练入口，Fold 模式只读取 OOF 开发视图。
- `prepare_l1_freeze_kaggle_kernel.py`：新增私有 Kaggle GPU 任务包生成器；后续标题规则已修正为与 Kernel id 一致。
- `verify_l1_freeze_pair.py`：新增成对验收器，比较实际训练合同、运行时和统一本机复评指标。
- `experiments/dianjienie_ryw18_oofconsensus_v1_yolo11m_freeze10pair_fold01_kaggle_20260828.md`：登记正式实验卡、边界、预注册闸门和提交状态。
- `README.md`、`experiments/README.md`：登记训练、任务包和验收入口。
- `l1_freeze_pair_20260828_r1/submission_record.json`：记录已提交包、实际 Kernel id、版本、哈希和后续归档入口。

### 产物路径

- `l1_freeze_pair_20260828_r1/control_kernel/`：对照任务包，Stage 1 `freeze=0`。
- `l1_freeze_pair_20260828_r1/freeze10_kernel/`：候选任务包，Stage 1 `freeze=10`。
- `l1_freeze_pair_20260828_r1/submission_record.json`：提交记录；两个任务提交后均为 `RUNNING`。
- `experiments/dianjienie_ryw18_oofconsensus_v1_yolo11m_freeze10pair_fold01_kaggle_20260828.md`：正式实验卡。

### 关键命令或入口

```text
E:\PythonProject\models\.venv\Scripts\python.exe kaggle_ipv4_cli.py kernels push -p E:\PythonProject\优化\l1_freeze_pair_20260828_r1\control_kernel
E:\PythonProject\models\.venv\Scripts\python.exe kaggle_ipv4_cli.py kernels push -p E:\PythonProject\优化\l1_freeze_pair_20260828_r1\freeze10_kernel
```

### 结论

- 两个任务均已成功推送为 Kaggle version `1`，实际任务为 `lly3323w/l1-ryw18-yolo11m-control-s1e20-s2e63-fold-01` 与 `lly3323w/l1-ryw18-yolo11m-freeze10-s1e20-s2e63-fold-01`，提交后状态均为 `RUNNING`。
- 两个任务均使用同一开发视图、Fold 1 `751/376`、总计 83 epoch、`imgsz=640`、`batch=8`、`SGD`、`seed=0` 和 `plots=False`；唯一训练变量是 Stage 1 的 `freeze=0` 与 `freeze=10`。
- 本地预检已确认 1,127 张开发图、15,671 个合法候选框、无 `test.txt`，生成的两份 Kernel 脚本归一化后完全一致。
- 未完成训练、未下载任何结果、未运行复评、未生成新的 ONNX；现有部署 ONNX 未被读取、修改或替换。

### 未完成项

- [ ] 等待两个 Kaggle 任务完成并安全归档。
- [ ] 在固定开发 Fold 1 上重评两个归档权重并执行 `verify_l1_freeze_pair.py`。
- [ ] 仅在闸门通过且获得下一阶段 GPU 授权后，训练 Freeze10 `finaldev` 候选并导出/验证新的 ONNX。

### 下一步

1. 用户或后续 Agent 查询两个实际 Kaggle Kernel 状态；仅在两者均为 `COMPLETE` 后下载 ZIP。
2. 用 `collect_l1_kaggle_run.py --development-only --install` 安全归档，再用 `reeval_l1_yolo_run.py --development-only --fold 1` 固定口径复评。
3. 执行 `verify_l1_freeze_pair.py`，未通过即停止 ONNX 路线。

### 风险与待确认

- 风险：单 Fold、单 seed 和 `auto_only` 候选标签只能提供暂定的开发集证据，不能宣称稳定提升或人工真值性能。
- 风险：Kaggle 可能因资源、依赖安装或训练运行时失败；必须以完成 ZIP 和本机复评为准。
- 待确认：Freeze10 是否在预注册闸门下胜过同流程对照；在此之前没有新 ONNX 可交付。
