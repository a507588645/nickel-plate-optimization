## 2026-07-13 | 删除两检测集伪标签与 manifests 派生目录

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex |
| 任务目标 | 补充清理截图中仍存在的 pseudo 标签和 manifests 派生目录 |
| 状态 | 已完成 |

### 已删除

- `E:\PythonProject\datasets\Dianjinie0211\labels_pseudo_v1`
- `E:\PythonProject\datasets\Dianjinie0211\labels_pseudo_v1_raw`
- `E:\PythonProject\datasets\Dianjinie0211\manifests`
- `E:\PythonProject\datasets\dianjienie\labels_pseudo_v1`
- `E:\PythonProject\datasets\dianjienie\labels_pseudo_v1_raw`
- `E:\PythonProject\datasets\dianjienie\manifests`

### 验证

- 六个目标删除后均不存在。
- 两数据集原始 `images/` 和 `labels/` 均确认仍存在。
- `train/`、`valid/`、`data.yaml` 未删除。
