# B3-clean 电积镍域内检测器重训任务包（v1）

| 项 | 内容 |
| --- | --- |
| 目的 | 路线 B3 的洁净重训：排除等级视图 **test** 泄漏后重训 YOLO11m@640，使 LGBM 等级评测的 test 侧零泄漏（B3 泄漏前提见 `b3_dianjinie0211_y11m640_task_v1/README.md` §2 与 `检测切换与等级提升路线_20260904.md`） |
| 授权 | 用户 2026-09-05 裁决 B3 血缘时选择「A) 授权 B3-clean 重训」（红线 5 的 Kaggle GPU 单独授权已满足；预计 9-11h，周额度 30h 已用约 11.4h） |
| 状态 | **已完成（2026-09-05）**：train v2 COMPLETE（203 epoch 早停，best epoch 103，训练 6.29h）；证据闸门通过、zip 回捞归档 `models\runs\dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211clean1\`（best.pt SHA-256=`849fe3c9…`，environment.json 为训练器原生）；特征提取 1072 图/22,109 框 → LGBM slf_texture33 **test acc=0.7578 / macroF1=0.7225，零泄漏口径 vs B0 ΔmacroF1=+0.0428 ≥ +0.01 闸门通过**；实验记录 `experiments/dianji_grade_b3clean_domain_detector_20260905.md` |
| 协议 | 与 B3/L1 完全一致：`yolo11m.pt` 预训练、seed=0、epochs=300、batch=8、imgsz=640、SGD、patience=100、deterministic=True、close_mosaic=0；训练器复用 B3 修复版（environment.json 落盘已修） |
| 权重用途 | 只做等级特征提取实验；**不导出、不替换部署 ONNX**（红线 2） |

## 1. 排除规则（b3clean_split_v1）

- 泄漏判定同 `analyze_b3_leakage_isolated_eval.py`：grade-test 行命中 B3 990 图池 = sha256 ∪ stem（image_stem/image_name_raw）∪ dup_group 传染 → **146/161 行泄漏**。
- 采集键分族（990 stem：B 族 853 / A 族 105 / OTHER 32）：
  - B 族 `^(\d{17})_\d+_\d+$` → 键 = 17 位时间戳（同原图切片共键）；
  - A 族 `^(\d{6})(bottom|top)_\d+$` → 键 = 6 位前缀（同板 bottom/top 共键）；
  - OTHER → 键 = 全 stem。
  - 键带族前缀（`B:`/`A:`/`O:`）防跨族碰撞。
- 146 个泄漏行 → **145 个采集键**；B3 图中键命中者全部排除：**train −121 图 / −2,506 框（774→653，17,655→15,149），val −38 图 / −764 框（216→178，4,337→3,573）**。
- **只排 grade-test**；grade-train/valid 与 B3 的重叠保留（LGBM 训练侧重叠不影响 test 评测洁净性；全排会毁掉检测器——train 774 中 764 被视图覆盖）。残留 skew 仅作记录。
- 逐排除项审计轨迹：`splits/excluded_images.csv`（每张被排图 → 采集键 → 命中的 grade-test 图）与 `splits/split_summary.json`。
- **nodule 注意**：train nodule 框 5 → 4（1 框随排除离开），val 本就 0 框——该类依旧不可评，与 B3 结论一致。

## 2. 目录内容

| 文件/目录 | 说明 |
| --- | --- |
| `build_b3clean_splits.py` | 划分构建器（确定性；内置 2026-09-05 探针数字断言：146 泄漏/145 键/−121/−38/−2,506，复算不符即失败） |
| `splits/` | `train_local.txt`（653）、`val_local.txt`（178）、`excluded_images.csv`、`split_summary.json` |
| `train_dianjinie0211_b3.py` | B3 修复版训练器副本（与 config 包内上传件字节一致） |
| `build_b3clean_config_dataset.py` + `kaggle_config_upload_v1/` | config 数据集构建器与暂存（831 图/标签全量 SHA-256，manifest 带 `variant=b3clean_grade_test_key_excluded_v1`） |
| `push_b3clean_config_v3.py` | 进程内 API 推新版本（CLI 直调挂死的规避，见 B3 任务包 README §3） |
| `kaggle_b3clean_dianjinie0211_y11m640_seed0/` | kernel 脚本 + metadata（id/title 对齐 slug `b3clean-dianjinie0211-yolo11m-640-seed0`） |

## 3. 提交记录

- config：`push_b3clean_config_v3.py`（IPv4 补丁 + `dataset_create_version` + `setdefaulttimeout(60)`），v3 status=ready，远端 9 文件核对一致（2026-09-05 14:31 UTC）。
- kernel：`PYTHONUTF8=1 models\.venv\Scripts\python.exe -u kaggle_ipv4_cli.py kernels push -p kaggle_b3clean_dianjinie0211_y11m640_seed0`（不带 UTF8 环境变量会在打印响应时 GBK 解码报错）。v1 = preflight（2026-09-05 14:43 COMPLETE，全绿，日志在 `preflight_v1_output/`）；v2 = train（MODE="train"，14:45 推送，RUNNING 已超 5 分钟）。
- kernel 相对 B3 版的改动：`DATA_VERSION=d0211clean1`、`RUN_NAME=dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211clean1`、`verify_config` 额外断言 `variant=b3clean_grade_test_key_excluded_v1`。
- 图片集复用 `lly3323w/dianjinie`（18 GiB 不重传）；kernel 启动时 831 标签 + 831 图片全量 SHA-256 对账。

## 4. 预期产物与回捞

- `/kaggle/working/dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211clean1_results.zip`（best/last.pt、args.yaml、requested_config.json、environment.json——本次应为训练器原生、results.csv、曲线/混淆矩阵、dataset_config/、README）。
- 回捞：`kaggle kernels output lly3323w/b3clean-dianjinie0211-yolo11m-640-seed0 -p <目录>`，归档至 `models\runs\dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211clean1\`。

## 5. 训练后步骤（全部完成，2026-09-05）

1. ~~归档校验~~：`verify_b3clean_archive.py` 通过——证据闸门全套齐全（environment.json 训练器原生：torch 2.6.0+cu124 / P100 / ultralytics 8.4.66），results.csv 203 epoch（best 103 + patience 100 干净早停），run 内 dataset_config 清单与本地 653/178 集合一致；best.pt SHA-256=`849fe3c9467bfabc4113fd442723fbb65482ed2e5a1475ba9342f52011144459`、last.pt=`e3c37d696ad436607877ad7136ae2a96a8a1432576bc9bd16ef91dbf63538aed`（`weights_sha256.json` 在档）。训练 6.292h；终检 val178：P=0.352 R=0.295 mAP50=0.234 mAP50-95=0.0924。
2. ~~特征提取~~：`run_b3clean_pt_extract.py --weights <best.pt> --output-dir dianji_grade_b3clean_y11m640_v1`（新包装器，顺带把 provenance/leakage 文案提升为模块常量并随守卫一起覆盖）；1072 图 / **22,109 框**（nodule 2；比 B3 同视图 16,605 框多 33%），纹理 join 1072/0。
3. ~~LGBM~~：`train_dianji_grade_v3.py --features …\features_b3y11m640_v1.csv --output-dir dianji_grade_b3clean_y11m640_v1 --seed 42`；slf_texture33 **test acc=0.7578 / macroF1=0.7225**。
4. ~~对比~~：**vs B0（0.7453/0.6797）ΔmacroF1=+0.0428 ≥ +0.01 闸门通过（test 侧零泄漏）**；vs B3-leaked（0.8385/0.8180）差 0.0955 = 泄漏记忆与训练集缩小的混合上限——B3 表观增益 +0.1383 中约 2/3 为泄漏成分，真实域内收益保守估计 +0.0428；B3-clean 仍高于 arm3-leaked（0.6528）。实验记录 `experiments/dianji_grade_b3clean_domain_detector_20260905.md`。
5. ~~工作日志 + 实验记录 + 路线文档回填~~（条目 `2026-09-05_2311_B3clean完成零泄漏闸门通过`）。全部结论 `unconfirmed_pending_enterprise`。

## 6. 风险与注意

- 排除后 colorstain 475 框 / nodule 4 框，稀有类更弱；B3-clean 的检测指标预计低于 B3（分母小了且泄漏图被排），属预期，不做跨 run 指标排名（AGENTS §8：不同数据版本不直接排名）。
- val 缩到 178 仍承担早停/选模；与 LGBM 评测的 grade-test 已无采集键交集。
- 若 kernel ERROR：`kaggle kernels output` 取日志诊断；preflight 失败的常见原因（嵌套挂载、slug 对齐）已在 B3 任务包记录。
