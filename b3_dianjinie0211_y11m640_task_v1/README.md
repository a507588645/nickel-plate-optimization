# B3 电积镍域内检测器训练任务包（v1）

| 项 | 内容 |
| --- | --- |
| 目的 | 路线 B3：在 `datasets/Dianjinie0211`（电积镍 5 类）上训 YOLO11m@640，用于电积镍等级特征提取对比（见 `检测切换与等级提升路线_20260904.md` §3 B3） |
| 授权 | 用户 2026-09-04 授权「明天 Kaggle GPU 额度刷新后提交」（对话确认，走 AskUserQuestion 记录）；2026-09-05 用户明示「现在可以提交给kaggle」后执行 |
| 状态 | **训练已完成（2026-09-05，train v4）**：139 epoch 早停（best epoch 39，patience 100）；best mAP50-95=0.09951 / mAP50=0.24292（best.pt 终检 all: P=0.342 R=0.278 mAP50=0.243 mAP50-95=0.0997；waterstain 0.127 / waterblock 0.158 / colorstain 0.0142 / bubble 0.0987，nodule val 无 GT 不可评）。产物归档 `models\runs\dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211v1\`（best.pt SHA-256=`95b56f70…`、last.pt=`cc21077b…`，`weights_sha256.json` 在档）。**下游特征提取 + LGBM 重训已完成（pass，ΔmacroF1 vs B0=+0.1383，带泄漏前提），见 §5** |
| 训练器 bug 修复 | v4 结束时 kernel 报 `run 缺少必要证据文件：['environment.json']`——训练器移植时丢了 environment.json 落盘行；**训练与全部指标产物完整，证据闸门按设计拦截了 zip 归档**。本地已修 `train_dianjinie0211_b3.py`（补 `(save_dir/"environment.json").write_text(...)`），config 数据集已推 v2（含修复版训练器，status=ready）。run 目录的 environment.json 由 kernel 日志 `[environment]` 行重构（torch=2.6.0+cu124, cuda=12.4, gpu=Tesla P100-PCIE-16GB, ultralytics=8.4.66），文件内 provenance 字段标明非训练器原生生成 |
| 协议 | 与 L1 公平训练基线一致：`yolo11m.pt` 预训练、seed=0、epochs=300、batch=8、imgsz=640、optimizer=SGD（lr0=0.01 / lrf=0.01 / momentum=0.937 / weight_decay=0.0005 / warmup=3ep）、patience=100、deterministic=True、close_mosaic=0、增强同 `train_dianjienie_ryw18_anchor.py` |
| 权重用途 | 只做等级特征提取实验；**不导出、不替换部署 ONNX**（红线 2） |

## 1. 目录内容

| 文件/目录 | 说明 |
| --- | --- |
| `audit_b3_dianjinie0211.py` + `audit/` | 只读数据审计脚本与报告（2026-09-04） |
| `train_dianjinie0211_b3.py` | B3 训练器（5 类固定划分，L1 协议超参，dry-run 支持，证据产物齐全） |
| `build_b3_kaggle_dataset.py` + `kaggle_dataset_upload_flat_v1/` | 18 GiB 扁平上传视图（**回退方案**，硬链接 png，不入库） |
| `build_b3_config_dataset.py` + `kaggle_config_upload_v1/` | **主方案**：config 小数据集（296 KB，含全量 SHA-256 校验表） |
| `probe_dianjinie_remote_reuse.py` + `dianjinie_remote_reuse_probe.json` + `remote_file_listing.json` | 远端图片集复用核验（990 图大小全对 + 1021 标签全在 + data.yaml 语义一致） |
| `splits/` | train/val 清单（本地形态）与 `split_summary.json`（含排除规则记录） |
| `kaggle_b3_dianjinie0211_y11m640_seed0/` | Kaggle kernel（脚本 + metadata） |

## 2. 数据事实与处理规则（2026-09-04 审计）

- 原始划分：train 836 png / val 216 png（另有同名 `.npy` 原始转储，**不使用不上传**）。
- train 缺标签 62 张 = 55 张 Windows「 - 副本」复制件（原图全部在且有标签）+ 7 张真无标签（`110035bottom_0`、`111615bottom_0`、`112613bottom_0`、`20230512164219043_0_2`、`20230512165910725_0_2`、`20230512174425429_0_2`、`20230513162403395_0_2`）。
- **纳入规则（固定、确定性）**：带标签 PNG 全收；train 额外排除「 - 副本」件；无标签图未经人工确认为负样本一律排除（AGENTS §6）。结果：**train 774 图 / 17,655 框，val 216 图 / 4,337 框**。排除清单在 `splits/split_summary.json`。
- `labels/val/classes.txt` 是放错位置的元数据（违反 AGENTS §6 存放规则），不随标签上传；类别顺序以 `data.yaml` 为准：waterstain / waterblock / colorstain / nodule / bubble。
- **nodule 类近乎缺失**：train 仅 5 框、val 0 框——逐类指标不可评，已写入 run README。
- train/val 沿用原始划分（与 slf.pt 同条件），未按板重划分；文件名前缀码跨 split 重叠为 0。
- **泄漏警示（解读 B3 结果必读）**：等级视图 1072 图（`archives/dianjinie_grade_classification_v1`）中 **986 张**出现在 Dianjinie0211（**770 张在 B3 train、216 张在 val**）。B3 检测器对等级视图的 LGBM 评测集（含 test 161 图）存在训练期泄漏，与四臂中域内 slf@640（acc=0.7702）同性质；B0 的部署 ONNX 为电解镍跨域权重，无此泄漏。**B3 若提升，可能部分来自检测器记忆**，结论须带此前提。

## 3. 提交步骤（额度刷新后）

> **实际执行记录（2026-09-05，已按此完成提交）**：`kaggle datasets create` 经 CLI 直接调用会挂死（网络/管道问题），可用方式是进程内调 API：`python kaggle_ipv4_cli.py` 的 IPv4 补丁逻辑（本包上级目录脚本）+ `api.dataset_create_new(<config目录>)` 并设 `socket.setdefaulttimeout(60)`。kernel 推送用 `models\.venv\python kaggle_ipv4_cli.py kernels push` 正常。**Kaggle 现将数据集挂载在 `/kaggle/input/datasets/<owner>/<slug>` 嵌套路径**（kernel 的 `locate_datasets` 已兼容两种布局）。kernel 实际 slug 由 title 生成：`b3-dianjinie0211-yolo11m-640-seed0`（metadata id 已对齐）。

```bash
# 1) 上传 config 数据集（296 KB；首次提交前核对 dataset-metadata.json 的 id）
cd 优化/b3_dianjinie0211_y11m640_task_v1/kaggle_config_upload_v1
kaggle datasets create --dir-mode skip

# 2) 推送 kernel（GPU 开启、Internet 开启；dataset_sources 已在 metadata 中）
cd ../kaggle_b3_dianjinie0211_y11m640_seed0
kaggle kernels push

# 3) 建议先跑一次 preflight：把 kernel 脚本顶部 MODE 改为 "preflight" 再 push，
#    观察 hash 校验与 --dry-run 全绿后改回 "train" 正式跑（多耗约 18GB 读取的数分钟）。
```

- kernel 启动即全量校验：config 内部文件 sha256 → 远端 data.yaml 语义（nc:5 + 5 类顺序）→ **990 标签 sha256 → 990 图片 sha256**（约 18 GiB 读取，`VERIFY_IMAGE_HASHES=True`；失败立即中止，不会误训）。
- 图片 Dataset 复用 `lly3323w/dianjinie`（2026-06-14 上传的单版本镜像，探针核验通过），**无需上传 18 GiB**；`kaggle_dataset_upload_flat_v1/` 仅作回退（若复用被否，`kaggle datasets create --dir-mode skip` 上传该目录并改 kernel dataset_sources）。

## 4. 预期产物与回捞

- 运行名：`dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211v1`。
- kernel 产出 `/kaggle/working/<run>_results.zip`，内含 `weights/best.pt|last.pt`、`args.yaml`、`requested_config.json`、`environment.json`、`results.csv`、PR 曲线/混淆矩阵、`dataset_config/`（清单与 data.yaml）、README。
- 回捞方式同 L1：kernel 完成后下载输出 zip 至本地归档目录（建议 `models/runs/` 同名目录，参照 finaldev 先例），校验 zip 内证据齐全后再进入特征提取。

## 5. 训练后步骤（后续会话）

1. ~~归档校验~~（已完成 2026-09-05）：核对 `environment.json`（torch 2.6.0 / ultralytics 8.4.66）、`requested_config.json` 与本协议一致、results.csv 完整。
2. ~~特征提取~~（已完成 2026-09-05）：新脚本 `优化\extract_dianji_grade_b3_y11m640_features.py`（复用 slf640 的 5 类 schema + y11m640 的 ultralytics 加载，SHA 守卫 = `95b56f70…`）；1072 图 / 16,605 框（nodule 0），纹理 join 1072/0，33 列（13 BASE + 20 纹理）CSV；产物 `优化\dianji_grade_b3_y11m640_v1\`。
3. ~~LGBM 重训~~（已完成 2026-09-05）：`train_dianji_grade_v3.py --features dianji_grade_b3_y11m640_v1\features_b3y11m640_v1.csv --output-dir dianji_grade_b3_y11m640_v1 --seed 42`；slf_texture33 test **acc=0.8385 / macroF1=0.8180**，vs B0（0.7453/0.6797）**ΔmacroF1=+0.1383 ≥ +0.01 闸门通过**；同泄漏前提对比 arm3 slf.pt@640（0.6528）Δ=+0.1652。**泄漏前提复核（行级 stem 匹配）：grade-test 161 中 146 张（90.7%）在 B3 train/val 内**，结论引用必须携带此前提。实验记录 `优化\experiments\dianji_grade_b3_domain_detector_20260905.md`。
4. ~~工作日志 + 实验记录 + 路线文档 §7 回填~~（已完成 2026-09-05，条目 `2026-09-05_1355_B3特征提取与LGBM重训闸门通过`）。

## 6. 风险与注意

- 远端镜像为单版本（2026-06-14），kernel 端全量哈希校验兜底；若未来有人更新该 Kaggle 数据集，校验会失败而不是静默错训。
- 无标签 7 图与「 - 副本」55 件的最终处置（确认负样本/删除副本）留待用户决定；本任务包只做排除，不动原数据集。
- 训练约 300ep × 774 图 batch8，预计单 GPU 数小时，远低于 Kaggle 周额度；preflight + 正式跑两连发也在额度内。
