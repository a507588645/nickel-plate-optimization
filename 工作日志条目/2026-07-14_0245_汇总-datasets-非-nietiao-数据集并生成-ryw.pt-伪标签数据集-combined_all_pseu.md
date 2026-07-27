## 2026-07-14 02:45 | 汇总 datasets 非 nietiao 数据集并生成 ryw.pt 伪标签数据集 combined_all_pseudo_v1

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi Code CLI |
| 任务目标 | 按用户要求汇总 datasets 下除 nietiao 外全部数据集，用 ryw.pt 生成伪标签，丢弃低置信度框，划分 train/val/test |
| 状态 | 已完成数据集构建；存在类别缺失待决策 |
| 关联用户请求 | 将 datasets 除 nietiao 外所有数据集汇总，软连接，用 ryw.pt 生成伪标签，丢弃低置信度，划分训练/验证/测试 |

### 改动文件

- `build_combined_pseudo_dataset.py`：新增脚本，负责软连接汇总、伪标签生成、置信度过滤、数据集划分与报告。
- `工作日志.md`：本条目与状态摘要。

### 产物路径

- `E:\PythonProject\datasets\combined_all_pseudo_v1\`：最终 YOLO 数据集。
  - `images/train`、`images/val`、`images/test`：按 70/15/15 划分的图片软连接。
  - `labels/train`、`labels/val`、`labels/test`：过滤后的五列 YOLO 标签（conf≥0.70）。
  - `labels_with_conf/`：含第六列置信度的伪标签主版本。
  - `images_all/`：全部原图软连接（中间产物，划分来源）。
  - `data.yaml`：Ultralytics 格式数据配置，18 类。
  - `splits/distribution.json`、`splits/distribution.txt`：划分与类别分布报告。
- `E:\PythonProject\优化\build_combined_pseudo_dataset.log`：脚本运行日志。

### 关键命令或入口

```text
cd E:\PythonProject\优化
.venv_classify\Scripts\python.exe build_combined_pseudo_dataset.py --out-dir E:\PythonProject\datasets\combined_all_pseudo_v1 --img-size 640 --device cpu
```

### 结论

- 共汇总 7 个数据集、7396 张图片：Dianjinie0211(2104)、dianjienie(2700)、classify_2types(1415)、classify_4types_dianji(986)、enterprise_dianji_0309_classification(86)、enterprise_dianji_0309_detection_unlabeled(37)、enterprise_dianjie_0306_detection_unlabeled(68)。
- 全部图片通过 `ryw.pt`（YOLOv5m）在 CPU/img-size=640 下重新生成伪标签；原始检测标签与分类 folder 信息均未保留。
- 原始候选框 102393 个，按 conf≥0.70 过滤后保留 29564 个，空标签图片由脚本写入空 `.txt`。
- 划分结果：train=5177、val=1109、test=1110；每个 split 内 images 与 labels 数量一一对应。
- `data.yaml` 使用 ryw.pt 的 18 类名称：`wuzi, waterstain, yellowstain, nodule, airhole, bubble, bump, fawu, overfire, NiOH, greenstain, suanwuran, quejiao, watermark, youpao, heibian, mijixiaokeng, fanxitong`。
- 类别分布极不平衡：waterstain 14703 个框，而 fawu 17、NiOH 15、greenstain 92。

### 未完成项

- [ ] 确认是否接受 4 个空类别（youpao/heibian/mijixiaokeng/fanxitong 在过滤后无样本）。
- [ ] 确认是否降低过滤阈值（如 0.40）以保留 uncertain 框，弥补稀有类别。
- [ ] 用 Ultralytics 加载 `data.yaml` 做最终一致性验证。

### 下一步

1. 若用户接受当前 0.70 阈值，可删除 data.yaml 中 14-17 类或保留但标注空类别风险，随后启动训练。
2. 若用户要求保留更多稀有类别样本，应重新以 conf≥0.40 过滤生成第二版数据集（如 `combined_all_pseudo_v2_uncertain`），并比较两版类别覆盖。

### 风险与待确认

- **跨镍种噪声**：ryw.pt 是电解镍检测权重，对电积镍数据生成的伪标签可靠性未知，混训会引入域迁移噪声。
- **分类 folder 信息丢失**：classify_2types / classify_4types_dianji / enterprise_dianji_0309_classification 仅作为无标签原图输入，原目录等级/好坏标签未使用。
- **类别缺失**：过滤后类别 14(youpao)、15(heibian)、16(mijixiaokeng)、17(fanxitong) 无训练样本；16 在原始伪标签中有 375 个框但全部 conf<0.70。
- **划分泄漏风险**：未按物理镍板 group 划分，仅按图片随机划分；存在同源/同板图片跨集合的可能。
- **CPU 推理参数**：伪标签使用 img-size=640，非历史审计常用的 1024；若需更高质量可改用 1024 或 GPU 重跑。
