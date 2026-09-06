# JinchuanPythonWebapi

本项目是金川视觉推理本地服务，当前默认运行形态为：

- `FastAPI + uvicorn` 作为服务外壳
- 原生 FastAPI 业务接口
- 供 Tauri 桌面端通过 `127.0.0.1` 本地调用

## 环境要求

- Windows 为主
- Python `3.11`
- `uv`

## 使用 uv 初始化环境

首次进入项目目录后执行：

```powershell
uv sync
```

这会根据 `pyproject.toml` 创建并同步 `.venv`。

如果你本机配置了自定义 `uv` 镜像，且遇到 `403 Forbidden` 或某些 wheel 无法下载，可以临时改为：

```powershell
uv sync --default-index https://pypi.org/simple
```

如果你还没有对应 Python，可先执行：

```powershell
uv python install 3.11
```

## 启动服务

开发时直接运行：

```powershell
uv run python server.py
```

服务默认监听：

```text
http://127.0.0.1:5005
```

## 常用接口

- `GET /health`
- `GET /config`
- `GET /metrics`
- `POST /preprocess`
- `POST /abdetect`
- `POST /detect/plate`
- `POST /dianjiEclass`
- `POST /dianjiclass`
- `POST /cutting`
- `POST /nietiaoSeg`
- `POST /isNieban`

### 组合式镍板检测

`POST /detect/plate` 按顺序执行 AB 检测、预处理和镍板分类，`plate_type` 支持
`dianji` 和 `dianjie`。接口支持两种请求形式：

二进制图片：

```powershell
curl.exe -X POST "http://127.0.0.1:5005/detect/plate?surface=top&plate_type=dianjie&out_dir=data\detect" `
  -H "Content-Type: application/octet-stream" --data-binary "@image.png"
```

JSON 路径：

```json
{
  "image": "C:\\images\\plate.png",
  "surface": "top",
  "plate_type": "dianjie",
  "out_dir": "data/detect"
}
```

成功响应中的 `data` 包含 `abdetect`、`preprocess`、`classify` 和
`elapsed_ms`；分类阶段使用预处理输出的图片。

## Tauri 集成

开发阶段可以让 Tauri 拉起：

```powershell
uv run python server.py
```

如果你后面做桌面打包，不建议在最终用户机器上每次启动时再调用 `uv run` 做解析；更稳妥的方式是先 `uv sync` 生成 `.venv`，再由 Tauri 直接拉起：

```powershell
.venv\Scripts\python.exe server.py
```

更详细的桌面集成说明见 `TAURI_PACKAGING.md`。

## 模型与规则配置

- 服务配置：`app_settings.yaml`
- 模型配置：`models.yaml`

## 可选后端说明

当前默认还是现有 PyTorch / 加密模型链路。

如果后续要切换额外后端，建议按需安装：

```powershell
uv add ultralytics
uv add onnx onnxruntime
```

`TensorRT` 依赖平台和 NVIDIA 环境更强，建议在确认目标机器环境后再单独接入。
