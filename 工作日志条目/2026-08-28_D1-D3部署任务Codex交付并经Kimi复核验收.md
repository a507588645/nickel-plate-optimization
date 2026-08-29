## 2026-08-28 | D1-D3 部署任务 Codex 交付并经 Kimi 复核验收

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex（执行，session 01a04449-c3a3-7d42-a6b2-b7ed091f6f56）+ Kimi Code（复核验收） |
| 任务目标 | WebAPI `优化分支`：D1 修 `.pth` 分类头覆盖 bug；D2 电解镍检测器切换冻结 YOLO11m 640 ONNX；D3 FastAPI 端到端冒烟。 |
| 状态 | 已完成（改动未提交，留在工作区待用户决定 commit） |

### 改动文件（`E:\PythonProject\ImetricCodes\JinchuanPythonWebapi`，`git diff --stat`：5 文件 +290/-105）

- `inference/models.py`：删除两处 `fc = nn.Linear(...)` 覆盖；新增 `_validate_classifier_head()`，加载后断言 `fc.out_features` 与 `models.yaml` 的 `num_classes`（2/4）一致，不一致即 `RuntimeError` 并指明权重文件与两个维度。
- `adapters/ultralytics.py`：占位适配器补全为可用——`YOLO(weights, task="detect")` 支持 `.onnx`，ndarray 输入不再退化为字符串（保留 BGR 内存直通）。
- `models.yaml`：`detectors.dianjie` → `kind: ultralytics` + `weights/dianjienie_yolo11m_ryw18_640.onnx`，`threshold: 0.25` 不变，`options.imgsz: 640`，注释记录来源 run 与 SHA-256；`detectors.dianji` 保持 `slf.pt`/YOLOv5 未动；补充「YOLO11 ONNX 必须走 ultralytics 适配器」说明。
- `pyproject.toml`/`uv.lock`：新增固定依赖 `ultralytics==8.4.66`、`onnx==1.22.0`、`onnxruntime==1.29.0`。
- 新增 `tools/smoke_dianjie.py`（冒烟脚本）与 `weights/dianjienie_yolo11m_ryw18_640.onnx`（未跟踪文件）。

### Kimi 独立复核结果（非复述 Codex 结论）

- 权重 SHA-256 实测 = `901ccf6006f44689483aaf6f0d1d1b8c18132a85c7d881be7e62fcdccc255b21`，与冻结交付一致。
- 逐行复核全部 diff：无越范围改动，无业务规则、阈值、`dianji` 配置变更；`308` 分支未触碰。
- 独立复跑冒烟 `uv run --no-sync python tools/smoke_dianjie.py <dev images> --limit 3`：ONNX Runtime CPU 加载正常，3 图均 HTTP 200/`code=1`，`api_counts=[176,1,10]`，与 Codex 的 5 图结果前 3 项逐项一致；直接推理 176 框 vs API 176 框，差值 0；标签均在 0-17。判定 PASS。
- Codex 报告的 D1 两进程确定性（`predicted_idx=0`、logits 相同）与类别顺序比对（`ordered names identical: True`）未复跑，采纳其输出（冒烟链路已实际经过修复后的 `load_resnet`）。

### 结论

- D1-D3 验收通过：电解镍检测链路已在 `优化分支` 切换至冻结 YOLO11m 640 ONNX（config 级可回退，`ryw.pt` 保留）；`.pth` 分类头修复后电积镍等级输出不再是随机头。
- 冒烟样本为开发集前若干张 PNG（该目录实为 PNG 非 JPG），仅验证链路正确性，不构成精度结论。

### 未完成项

- [ ] WebAPI 改动尚未 git commit（需用户明确授权）。
- [ ] D4 电积镍跨域闸门：按用户选定「业务输出层对比」设计，未立项。
- [ ] LightGBM 接入：按 `v3_texture_only` 纹理路线（检测器无关）设计，未立项。
- [ ] 现场目标设备延迟回归、Tauri 打包验证未做。

### 下一步

1. 用户确认是否 commit WebAPI 当前改动。
2. 立项 D4 + LightGBM 接入（可合并为一个 Codex 任务，工作目录分别为 `优化` 与 WebAPI，需拆分）。
3. D4 通过后回切 `detectors.dianji`。

### 风险与待确认

- 生产阈值仍为 0.25（技术兼容参数），决策层阈值调整仍需固定人工评测集，未动。
- 冒烟用例取自 dianjienie 开发集（电解镍图）；电积镍图上的新权重行为完全未知，正是 D4 要回答的问题。
- `tools/smoke_dianjie.py` 中 `JINCHUAN_PRELOAD_MODELS=false` 环境变量为冒烟专用路径；全量加载（含 slf.pt、分割模型）的启动回归未做。
