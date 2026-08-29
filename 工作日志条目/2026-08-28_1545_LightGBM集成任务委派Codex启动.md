## 2026-08-28 15:45 | LightGBM WebAPI 集成任务委派 Codex 启动

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi Code（规划与验收）；Codex（执行，后台任务 bash-c0ie0g2y） |
| 任务目标 | 把已验收的 `dianji_grade_y11m640_v1` LightGBM 等级模型集成进 WebAPI `优化分支`：移植 63 列特征提取（43 检测 + 20 纹理），替换 `_dianji_detect` 的 resnet 等级分支，保持前端契约并新增 confidence/manual_review/grade_status 字段 |
| 状态 | 进行中 |

### 任务书

- `codex_tasks/lgbm_webapi_integration_task.md`（含目标代码位置、契约 JSON 路径、参考实现、等级→type 映射 3#→4/9950→3/9996→2/bad→1、验收标准）

### 关键设计决策（Kimi 已核证）

1. **复用 `model_state.yolo`**（D1-D3 已加载的 YOLO11m 640 ONNX，conf 0.25/iou 0.45）做等级特征检测，不切换 `detectors.dianji`（slf.pt 仍负责显示缺陷，D4 闸门待用户裁决）。
2. 纹理特征为纯 numpy/cv2（无 skimage 依赖）；`compute_plate_area` 依赖的 `preprocess.findcent` 本就位于 WebAPI 仓库，属原生回迁。
3. WebAPI `.venv` 缺 lightgbm，任务书允许安装 `lightgbm==4.6.0`（与训练版本一致）。
4. 运行时从随模型打包的契约 JSON 校验 63 特征顺序，防止手工抄录漂移；LGBM 失败时回退 legacy resnet 分支（`grade_source: resnet_fallback`），接口不 500。
5. 验收以 ≥10 张归档视图图的特征级对齐为准（max abs diff ≤1e-6、预测等级 100% 一致）+ 新旧冒烟测试 PASS。

### 下一步

- Codex 完成后 Kimi 复核实际 diff、parity 与冒烟输出再验收。

### 风险与待确认

- D4 闸门（`d4_dianji_business_compare_20260828_024719/review.html`）仍待用户裁决；本集成不改变电积镍显示缺陷来源，与 D4 裁决解耦。
- 全部等级输出标 `unconfirmed_pending_enterprise`，企业确认前不作为生产判级依据。
