## 2026-07-24 20:34 | RYW18 私有 Kaggle Dataset 训练包与本地端到端干运行

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex（/root） |
| 任务目标 | 在本机无法训练的条件下，为 `dianjienie_ryw18_anchor_v1` 准备可追溯、不会覆盖历史标签的 Kaggle GPU 训练通路 |
| 状态 | 已完成（上传视图与运行器已生成并校验；未向 Kaggle 上传，未启动实际训练） |

- 核验锚点图片负载为 22,815,397,943 bytes（约 21.25 GiB）。GitHub 不作为图片数据载体；主路径确定为私有 Kaggle Dataset + Kaggle GPU Notebook，Google Drive 仅在上传失败时作中转备选。
- 新增 `build_kaggle_dianjienie_ryw18_dataset.py`：创建 `优化\kaggle_dianjienie_ryw18_anchor_v1\dataset_upload_flat_v3`。其中 1325 张图片是原锚点硬链接，1325 个标签与 13 个划分/类别元数据是独立副本；包含 `flat_manifest.json`、训练器和 Kaggle 运行器。显式 `--publish` 才会读取现有 Kaggle API 凭据并创建私有 Dataset，默认没有外部上传。
- 新增 `kaggle_dianjienie_ryw18_runner.py`：在 `/kaggle/working` 恢复 `images/`、`labels/`、`splits/` 视图，校验 RYW18 和固定 Fold 后调用训练器；训练成功会生成完整 run 的 zip。依赖固定为项目既有 Kaggle 基线的 Torch 2.6.0 CUDA 12.4、Torchvision 0.21.0、NumPy 1.26.4、Ultralytics 8.4.66。
- 修正 `train_dianjienie_ryw18_anchor.py` 的路径返回逻辑：运行时清单保留 `images/` 路径而不解析 Kaggle 符号链接，确保 Ultralytics 能正确派生相邻 `labels/` 路径。
- 本地验证：三个 Python 文件 `py_compile` 通过；上传视图构建成功，图片抽检与源锚点 `samefile=true`、标签 `samefile=false`；Kaggle 运行器 Fold 1 dry-run 重建 1325 图片/1325 标签，并确认 train/val/冻结 test 为 751/376/198。上传视图中训练器和运行器 SHA-256 均与源脚本一致。
- Windows 当前账户没有创建符号链接权限，运行器仅在本地验证时退回硬链接；Kaggle Linux 环境仍使用目录符号链接，不复制图片。未修改原 `dianjienie`、冻结锚点、历史标签或模型权重。

### 改动文件

- `build_kaggle_dianjienie_ryw18_dataset.py`
- `kaggle_dianjienie_ryw18_runner.py`
- `train_dianjienie_ryw18_anchor.py`
- `dianjienie_ryw18_anchor_v1_训练交接.md`
- `README.md`
- `CLAUDE.md`
- `工作日志.md`

### 产物路径

- 上传视图：`E:\PythonProject\优化\kaggle_dianjienie_ryw18_anchor_v1\dataset_upload_flat_v3`。
- 本地验证锚点：`E:\PythonProject\优化\kaggle_dianjienie_ryw18_anchor_v4`（仅测试产物，图片通过链接引用上传视图）。

### 下一步

```powershell
cd E:\PythonProject\优化
python .\build_kaggle_dianjienie_ryw18_dataset.py --publish
```

用户明确同意上传后执行上述命令；随后在 Kaggle 私有 GPU Notebook 运行交接文档中的 YOLO11m 与 YOLO26m Fold 1 命令。每个 run 完成后保留 Kaggle 输出 zip 并归档至 `E:\PythonProject\models\runs`，再补 Fold 2/3 和后续 OOF 共识。

### 风险与待确认

- 约 21.25 GiB 的上传能否完成取决于当前 Kaggle 账户存储、网络稳定性与服务端配额；实际创建私有 Dataset 后再记录结果。
- Kaggle GPU/会话时长也需实际启动 Fold 1 后确认；若显存不足，两个模型必须同步下调 batch 并以新配置记录。
- `youpao`、`fanxitong` 零监督，`heibian`、`mijixiaokeng` 极稀少；Kaggle 训练不会消除这些数据边界。
