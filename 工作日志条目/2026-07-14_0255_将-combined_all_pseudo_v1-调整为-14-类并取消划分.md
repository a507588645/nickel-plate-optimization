## 2026-07-14 02:55 | 将 combined_all_pseudo_v1 调整为 14 类并取消划分

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi Code CLI |
| 任务目标 | 按用户要求把汇总数据集改为 14 类，并取消 train/val/test 划分 |
| 状态 | 已完成 |
| 关联用户请求 | 改为 14 类；不需要划分，直接放在一起 |

### 改动文件

- `modify_to_14class_no_split.py`：新增脚本，负责删除空类别、合并划分、重写 data.yaml。
- `combined_all_pseudo_v1/data.yaml`：由 18 类改为 14 类，val/test 均指向 images/train。
- `combined_all_pseudo_v1/images/`、`combined_all_pseudo_v1/labels/`：删除 val/test 子目录，全部归入 train。
- `工作日志.md`：本条目与状态摘要。

### 产物路径

- `E:\PythonProject\datasets\combined_all_pseudo_v1\`
  - `images/train/`：7396 张图片软连接。
  - `labels/train/`：7396 个 YOLO 标签（五列，conf≥0.70，类别 0–13）。
  - `data.yaml`：14 类配置，无划分。
  - `labels_with_conf/`：保留原始 18 类含置信度伪标签主版本（可回溯）。
  - `splits/distribution.txt`：更新为无划分摘要。

### 关键命令或入口

```text
cd E:\PythonProject\优化
.venv_classify\Scripts\python.exe modify_to_14class_no_split.py
```

### 结论

- 删除 `youpao / heibian / mijixiaokeng / fanxitong` 4 个空类别，保留 0–13 共 14 类。
- 取消 70/15/15 划分，`images/val`、`images/test`、`labels/val`、`labels/test` 已删除。
- 全部 7396 张图片与对应标签集中在 `images/train` 与 `labels/train`。
- `data.yaml` 中 `val` 与 `test` 也指向 `images/train`，以满足 Ultralytics 解析需要；用户理解这意味着没有独立验证集。
- 验证通过：images/train=7396，labels/train=7396，标签中仅出现 0–13 类，无 14–17 类。

### 风险与待确认

- **无独立验证集**：训练过程无法监控过拟合，也无法评估模型在未见过图片上的泛化能力；最终指标只能依赖训练集 loss，可信度低。
- **类别不平衡**：waterstain(14703) 与 watermark(5569) 占绝大多数，fawu(17)、NiOH(15) 等稀有类别样本极少，模型可能学不到。
- **跨镍种噪声与伪标签可靠性**：同前一条日志；ryw.pt 对电积镍数据的伪标签质量未经验证。
- **用户提到的截图**：`C:\Users\23916\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{03B8E775-AE79-4898-B103-01B92E024DDC}.png` 无法被当前工具读取，若图片中包含额外要求，请用户用文字补充。
