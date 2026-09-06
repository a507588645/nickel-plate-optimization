# Tauri 集成运行说明

## 启动方式

桌面打包场景统一通过 `server.py` 启动服务：

```powershell
uv run python server.py
```

新入口特性：

- `FastAPI + uvicorn` 作为外层运行壳。
- 业务接口 `/preprocess`、`/abdetect`、`/dianjiEclass`、`/dianjiclass`、`/cutting`、`/nietiaoSeg`、`/isNieban` 已迁移为原生 FastAPI 路由。
- 新增管理接口：`/health`、`/config`、`/metrics`。
- 默认监听 `127.0.0.1:5005`，更适合 Tauri 本地调用。

## Tauri 侧建议

开发阶段，Tauri 可以直接拉起 `uv run python server.py`，然后轮询：

```text
GET http://127.0.0.1:5005/health
```

返回 `200` 且 `ready=true` 后，再调用业务接口。

如果后面做正式桌面打包，建议先在构建阶段执行一次 `uv sync`，然后让 Tauri 直接拉起：

```powershell
.venv\Scripts\python.exe server.py
```

这样不会在最终用户启动 GUI 时再次触发依赖解析。

## 可调环境变量

```text
JINCHUAN_HOST=127.0.0.1
JINCHUAN_PORT=5005
JINCHUAN_WORKERS=1
JINCHUAN_MODE=desktop
JINCHUAN_PRELOAD_MODELS=true
JINCHUAN_RUNTIME_DIR=C:\your\runtime\dir
JINCHUAN_DEBUG_ARTIFACTS_ENABLED=false
```

## 模型和规则配置

- 服务运行配置：`app_settings.yaml`
- 模型配置：`models.yaml`

已经抽出的内容：

- 服务端口、日志、CORS、健康检查、metrics
- 电解镍判级阈值
- 镍条抓取区域和最小尺寸
- 检测器/分类器/分割模型权重路径

## ONNX / TensorRT 预留方式

后续如果要切换检测模型，不需要改业务代码，只改 `models.yaml`：

```yaml
detectors:
  dianjie:
    kind: onnxruntime
    weights: 'weights/ryw.onnx'
    threshold: 0.25
```

或：

```yaml
detectors:
  dianjie:
    kind: tensorrt
    weights: 'weights/ryw.engine'
    threshold: 0.25
```

当前默认仍然是现有 PyTorch / 加密权重加载链路，不会强制你现在就改 ONNX/TensorRT。

## 多 worker 说明

桌面端如果启用 GPU，多 worker 会重复加载模型，显存和内存占用会直接翻倍甚至更多。

因此现在的策略是：

- CPU 场景允许你自己把 `JINCHUAN_WORKERS` 调大。
- GPU 场景默认强制回落到 `1` 个 worker。
- 如果你明确知道自己要承担重复加载成本，再把 `app_settings.yaml` 里的 `allow_multi_worker_with_gpu` 改成 `true`。
