## 2026-09-05 01:55 | B3 已提交 Kaggle：preflight v3 全绿 → train v4 RUNNING

| 字段 | 内容 |
| --- | --- |
| 执行者 | ZCode (GLM) |
| 任务目标 | 用户授权「现在可以提交给kaggle」后，执行 B3 电积镍域内 YOLO11m@640 训练的实际提交（config 数据集上传 + preflight 验证 + 正式训练） |
| 状态 | 进行中（train v4 RUNNING）；训练完成后的回捞与特征提取待后续会话 |
| 关联用户请求 | 「现在可以提交给kaggle」；前序条目 [2026-09-05_0015_B3任务包建立与Kaggle授权](2026-09-05_0015_B3任务包建立与Kaggle授权.md) |

### 提交结果

- **config 数据集已创建**：`lly3323w/dianjinie0211-b3-config-v1`（10 文件/约 300 KB，`kaggle datasets status` = ready）。远端文件清单已核对（manifest/双哈希表/meta 清单/训练器均在）。
- **Kernel**：`lly3323w/b3-dianjinie0211-yolo11m-640-seed0`（https://www.kaggle.com/code/lly3323w/b3-dianjinie0211-yolo11m-640-seed0 ），共 4 个 version：
  - v1 preflight **ERROR**：`locate_datasets` 在 `/kaggle/input` 顶层找不到 config 数据集——**根因：Kaggle 新版把数据集挂载到嵌套路径 `/kaggle/input/datasets/<owner>/<slug>`**（与 L1 finaldev kernel 的 `scoped_root` 扫描同一问题）；
  - v2 preflight ERROR：同因（本次日志的挂载清单打印确认了嵌套布局）；
  - v3 preflight **COMPLETE**：修复 `locate_datasets` 同时扫描顶层与嵌套布局后——config manifest/payload 哈希 ✓ → data.yaml 语义（nc:5 + 5 类顺序）✓ → **990 标签 + 990 图片全量 SHA-256 对账 ✓（约 18 GiB 读取，本地 vs 远端逐字节一致）** → trainer `--dry-run` 配置核对 ✓（774 train/216 val、17,655/4,337 框、类别分布与本地审计一致）；
  - v4 **train（当前 RUNNING，2026-09-05 01:31 推送）**：seed0/300ep/batch8/imgsz640/SGD/patience100，运行名 `dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211v1`；已确认 RUNNING 超过 5 分钟（v1/v2 的错误在启动 1 分钟内即暴露，挂载/哈希/依赖阶段已全部越过）。产出预期：`/kaggle/working/<run>_results.zip`（best/last.pt、args.yaml、requested_config.json、environment.json、results.csv、dataset_config/、README）。

### 提交过程中的坑（后续 Agent 必读）

1. **`kaggle datasets create` 经 CLI 会挂死**（前台/后台、`--dir-mode skip` 与默认模式、系统与 venv 解释器均复现；netstat 显示进程无任何到 Kaggle 的活动连接）。**可用方式：进程内调 API**——`import kaggle_ipv4_cli`（应用 IPv4 补丁）+ `KaggleApi().dataset_create_new(<config目录>)` + `socket.setdefaulttimeout(60)`，91.8 秒完成。读取类 API（list/files/status）不受影响。
2. kernel push 正常姿势：`E:\PythonProject\models\.venv\Scripts\python.exe -u kaggle_ipv4_cli.py kernels push -p <目录>`。
3. **kernel 实际 slug 由 title 生成**（`b3-dianjinie0211-yolo11m-640-seed0`），metadata id 与 title 不一致时会推到意外 slug 且 status 查询 403；v2 起已把 metadata id/title 对齐为实际 slug。
4. 新版 Kaggle 数据集挂载在 `/kaggle/input/datasets/<owner>/<slug>` 嵌套路径；kernel 的 `locate_datasets` 现兼容两种布局，并会在失败时打印完整挂载清单便于诊断。

### 改动文件

- `优化\b3_dianjinie0211_y11m640_task_v1\kaggle_b3_dianjinie0211_y11m640_seed0\b3-dianjinie0211-y11m640-seed0.py`：`locate_datasets` 兼容嵌套挂载 + 失败时打印挂载清单；MODE 于 v4 切回 `train`
- `优化\b3_dianjinie0211_y11m640_task_v1\kaggle_b3_dianjinie0211_y11m640_seed0\kernel-metadata.json`：id/title 对齐实际 slug `b3-dianjinie0211-yolo11m-640-seed0`
- `优化\b3_dianjinie0211_y11m640_task_v1\kaggle_b3_dianjinie0211_y11m640_seed0\preflight_v1_output\`：v1/v2 失败日志与 v3 全绿日志（回捞自 Kaggle）
- `优化\b3_dianjinie0211_y11m640_task_v1\README.md`：§3 补实际执行记录（CLI 挂死规避方法、嵌套挂载、slug 对齐）

### 未完成项

- [ ] train v4 完成后回捞 `_results.zip`（预计数小时；参照 L1 惯例不设本机常驻监控，用户通知或下次会话用 `kaggle kernels output` 回收）
- [ ] 归档校验（environment/requested_config/results.csv）→ 特征提取（slf 5 类 schema，需为 ultralytics .pt 适配加载）→ LGBM 重训 → 与 B0 对比表（闸门 ≥+0.01 macroF1）
- [ ] 7 张无标签图与 55 张「- 副本」最终处置待用户确认

### 下一步

1. 训练完成（或用户通知）后：下载 v4 输出 ZIP 至 `models/runs/dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211v1/`（参照 finaldev 归档惯例），校验证据齐全。
2. 按任务包 README §5 做特征提取与 LGBM 重训，出对比表。

### 风险与待确认

- 训练中段失败风险低但非零（pre 验证全绿 + L1 同栈先例）；若 ERROR，用 `kaggle kernels output` 取日志按行诊断。
- 结果解读必须带泄漏前提（986/1072 等级图在检测器训练/验证集内，见 0015 条目）。
- Kaggle 周额度消耗：preflight 两次失败约数分钟 + preflight v3 约 8 分钟 + train v4 全程（预计数小时，远低于 30 小时周额度）。
