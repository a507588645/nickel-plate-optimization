## 2026-07-14 03:05 | 将 combined_all_pseudo_v1 图片扁平化并恢复 train/val/test 划分

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi Code CLI |
| 任务目标 | 按用户要求：图片直接汇总、不保留来源子目录；同时恢复训练/验证/测试划分 |
| 状态 | 已完成 |
| 关联用户请求 | 汇总里面不应继续分成以前文件夹，应该直接汇总软连接图片；划分训练集验证集测试集还是要的 |

### 改动文件

- `flatten_and_split_14class.py`：新增脚本，负责文件名扁平化、14 类过滤、恢复 70/15/15 划分。
- `combined_all_pseudo_v1/images/`、`combined_all_pseudo_v1/labels/`：重新组织为 images/train、images/val、images/test，根目录下无子文件夹。
- `combined_all_pseudo_v1/data.yaml`：恢复为 14 类 + train/val/test 指向。
- `combined_all_pseudo_v1/splits/distribution.json` 与 `distribution.txt`：更新。
- `工作日志.md`：本条目与状态摘要。

### 产物路径

- `E:\PythonProject\datasets\combined_all_pseudo_v1\`
  - `images/train/`：5177 张图片软连接，文件名扁平化。
  - `images/val/`：1109 张图片软连接。
  - `images/test/`：1110 张图片软连接。
  - `labels/train/` / `labels/val/` / `labels/test/`：对应过滤后的五列 YOLO 标签。
  - `data.yaml`：14 类，train/val/test 指向正确。
  - `labels_with_conf/`：原始 18 类含置信度伪标签主版本（可回溯）。

### 关键命令或入口

```text
cd E:\PythonProject\优化
.venv_classify\Scripts\python.exe flatten_and_split_14class.py
```

### 结论

- 图片不再保留 `Dianjinie0211`、`classify_2types` 等来源子目录，全部扁平放入 `images/train|val|test` 根目录。
- 文件名通过把完整相对路径编码为文件名来保证唯一，例如 `classify_2types_train_bad_114.png`、`Dianjinie0211_images_train_xxx.jpg`。
- 恢复 70/15/15 划分：train=5177、val=1109、test=1110，每个 split 内 images 与 labels 数量一致。
- 仍为 14 类（0–13），过滤后共保留 29564 个框。
- 验证通过：无子目录、无文件名冲突、最大类别 ID 为 13。

### 风险与待确认

- 图片文件名较长且包含原路径信息，若训练脚本依赖简洁文件名，可能需要进一步重命名。
- 其余风险（跨镍种噪声、类别不平衡、无物理镍板 group 划分）同前两条日志。
