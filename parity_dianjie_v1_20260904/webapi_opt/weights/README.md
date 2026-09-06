# 模型权重说明

本目录集中存放所有模型权重。**换检测模型**改 `../models.yaml`；分类/分割权重路径写在 `../models.yaml`。

| 文件 | 模型类型 | 用途 | 被谁加载 | 状态 |
|---|---|---|---|---|
| `ryw.pt` | YOLOv5 检测 | 电解镍缺陷检测 | `models.yaml` → dianjie | ✅ 在用 |
| `slf.pt` | YOLOv5 检测 | 电积镍缺陷检测 | `models.yaml` → dianji | ✅ 在用 |
| `ryw.pth` | ResNet 分类 | 电解镍分级（2 类） | `models.yaml` → `dianjie_resnet` | ✅ 在用 |
| `slf.pth` | ResNet 分类 | 电积镍分级（4 类） | `models.yaml` → `dianji_resnet` | ✅ 在用 |
| `best.pt` | YOLOv5-seg 分割 | 镍条分割 | `models.yaml` → `nietiao` | ✅ 在用 |
| `unused/seg.pt` | 分割（旧/备用） | — | 无 | ❓ 代码未加载，归档 |
| `unused/v5l-seg.pt` | YOLOv5l-seg（旧/备用） | — | 无 | ❓ 代码未加载，归档 |

## 说明

- **检测模型**（电解镍/电积镍）现由 `../models.yaml` 配置，可在 `yolov5`、`yolo26`、`ultralytics` 与 `rfdetr` 之间切换。
- YOLO26 权重放入本目录后，在 `../models.yaml` 中把对应任务改为 `kind: yolo26`，并把 `weights` 指向该 `.pt` 文件。
- `unused/` 里的两个权重当前没有任何代码加载，先归档保留；确认不需要后可自行删除。
- RF-DETR 权重不在本目录，而在 `E:\PythonProject\rf-detr\outputs\...`，需要切换时由 `models.yaml` 用绝对路径引用。
- 若启用加密部署，加密产物 `*.enc` 应与对应权重放在一起（同目录同名加 `.enc`）。
