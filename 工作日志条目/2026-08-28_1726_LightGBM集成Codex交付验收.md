## 2026-08-28 17:26 | LightGBM WebAPI 集成 Codex 交付并经 Kimi 复核验收

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi Code（复核验收）；Codex（执行，session 01a0476e-8614-7783-ac38-19cc368cc60c，含一轮误报修正） |
| 任务目标 | 复核 Codex 的 LightGBM 等级模型 WebAPI 集成（任务书 `codex_tasks/lgbm_webapi_integration_task.md` + 修正轮 `_fix1.md`），独立复跑关键验证后验收 |
| 状态 | 已完成（验收通过，改动未 commit） |

### 过程备注

- 首轮 Codex 因控制台把中文分支名 `优化分支` 显示为 `????` 误判分支不符而停工（87 秒空跑，exit 0）；Kimi 核实为编码误报后用 `-ResumeSession` 发修正轮，Codex 完成全部实现（16:53–17:18）。
- 监测坑：状态文件 JSON 键是 `state` 不是 `status`，首个 watcher 读错键空转，已纠正。

### 改动文件（WebAPI 仓库 `优化分支`，未 commit）

- 新增 `inference/grade_features.py`（314 行：43 检测特征 + 20 纹理特征移植，纯 numpy/cv2，特征顺序运行时从合同 JSON 读取并校验缺失/多余）
- `inference/detectors.py`：`_dianji_detect` 等级分支换为 LGBM（复用 `model_state.yolo` YOLO11m，conf 0.25/iou 0.45/imgsz 640 取自合同）；新增 `grade/confidence/manual_review/grade_status/grade_source` 字段；LGBM 失败回退 legacy resnet（`resnet_fallback`，manual_review=True）；slf_yolo 缺陷展示路径逐字段不变
- `inference/models.py`：`load_dianji_grade_model()`（启动期校验特征数量/顺序/类别映射/阈值/num_class，合同错误即抛，其余异常留空走回退）；`runtime_state.py` 预加载登记；`models.yaml` 新增 `grade_models.dianji_lgbm`（threshold 0.7362）
- 新增 `weights/dianji_grade_lgbm_v1.txt` + `..._contract.json`、`tools/smoke_dianji_grade.py`、`tools/parity_dianji_grade.py`
- 依赖：lightgbm==4.6.0 入 `.venv`，`pyproject.toml`/`requirements.txt`/`uv.lock` 已更新

### Kimi 独立验证（非复述 Codex）

1. **parity 复跑**：12 图（4 等级×3，sha256 校验）max abs 特征差 **0.0**，等级 id **12/12** 与参考 booster 一致。
2. **标签分组常量**与参考实现逐项一致（A={0,1} B={2,4,5,6,10,12,13} C={7,8,11} Airhole={4} Nodule=3）。
3. **冒烟复跑**：`smoke_dianji_grade.py` PASS（4 图，新字段与 manual_review 阈值一致性校验通过）；`smoke_dianjie.py` PASS（176/1/10 框，与 D1-D3 验收值逐项一致，电解镍无回归）。
4. **ndarray 灰度路径对照**（Kimi 自建 `pi_tasks/_ndarray_gray_consistency.py`）：生产内存直通路径（cvtColor 灰度）与训练语义（imdecode 灰度）灰度像素差 ±1 → 纹理特征最大差 0.028、置信度最大漂移 0.098，6 图等级 0 翻转。

### 结论

集成验收通过，可进入用户试用。等级→type 映射 3#→4/9950→3/9996→2/bad→1 与历史语义一致；全部输出带 `grade_status=unconfirmed_pending_enterprise`。

### 未完成项

- WebAPI 仓库改动**未 commit**（等用户授权）。
- 全量启动回归（`JINCHUAN_PRELOAD_MODELS` 默认路径含分割模型）未做。

### 下一步

1. 用户确认后授权 commit（建议信息含"电积镍等级切换 LightGBM"）。
2. 现场试用时关注 manual_review 比例是否接近预期 ~26%。
3. D4 闸门（`review.html`）仍待用户裁决。

### 风险与待确认

- **ndarray/路径灰度语义漂移**：±1 灰度 → 置信度最大漂移 0.098，接近拒判阈值 0.7362 的样本 manual_review 可能翻转（方向安全：多转人工）；等级本身在抽查中未翻转。彻底消除需按 cvtColor 语义重提 1072 图特征并重训，当前判定不值得。
- LightGBM 原生接口无法直接打开中文路径模型文件；部署权重在仓库 ASCII 路径不受影响（parity 脚本内对参考模型做了临时 ASCII 副本）。
- 验证集指标（test 0.7453/0.6797）是在 imdecode 灰度语义下测得，现场 ndarray 路径会有轻微系统性偏差。
