## 2026-07-21 23:02 | 修复 combined_all_pseudo_v2 审查页无类别名（只显示数字 ID）

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi |
| 任务目标 | 用户反馈审查页「不知道是啥标签」无法审查；定位并修复框标注只显示数字类别 ID 的问题 |
| 状态 | 已完成 |

### 根因

- `generate_flat_yolo_review.py` 只从 `<数据集根>/classes.txt` 读取类别名；`combined_all_pseudo_v2` 根目录无该文件（classes.txt 在 `labels_with_conf/` 内），导致生成页 `const CLASSES=[]`，所有框只渲染数字 ID。

### 改动文件

- `generate_flat_yolo_review.py`：类别名加载改为回退链 `classes.txt` → `labels_with_conf/classes.txt` → `labels/classes.txt` → `data.yaml names`，并新增 `--classes-file` 可选显示名文件（最高优先级）；全部缺失时打印警告。
- 新增 `classes_display_18class.txt`：18 类「英文名 中文名」对照（中文名取自 `E:/PythonProject/整合技术文档.md` §15.5 权威映射，非自行翻译）。

### 产物路径

- 重新生成 `pseudo_label_reviews/combined_all_pseudo_v2/review.html`（3264 张 / 74331 框不变）；`const CLASSES` 现为 18 个双语名。
- 浏览器 localStorage 键 `review::combined_all_pseudo_v2` 未变，已有审查进度不受影响。
- 导出 JSONL 的 `class_name` 变为双语名；下游 `apply_flat_review_results.py` 只用 `class_id`，不受影响（已核对）。

### 关键命令

```text
.venv_classify/Scripts/python.exe generate_flat_yolo_review.py E:/PythonProject/datasets/combined_all_pseudo_v2 --output-root E:/PythonProject/优化/pseudo_label_reviews --absolute --classes-file E:/PythonProject/优化/classes_display_18class.txt
```

### 结论与备注

- 18 类映射（`build_combined_pseudo_dataset.py:39` 与 `labels_with_conf/classes.txt` 一致）：0 wuzi 污渍 / 1 waterstain 水渍 / 2 yellowstain 黄渍 / 3 nodule 边部结瘤 / 4 airhole 气孔 / 5 bubble 结粒 / 6 bump 疙瘩 / 7 fawu 发乌 / 8 overfire 烧灼烧板 / 9 NiOH 氢氧化镍 / 10 greenstain 绿色结晶绿渍 / 11 suanwuran 酸污染 / 12 quejiao 缺角 / 13 watermark 严重水渍水印 / 14 youpao 油泡 / 15 heibian 黑边 / 16 mijixiaokeng 密集小坑 / 17 fanxitong 返析铜。
- 当前伪标签实际只用到 ID 0–13 和 16；14（油泡）、15（黑边）、17（返析铜）框数为 0，审查时不会遇到。
- 其他用同一脚本生成的审查页若数据集根无 classes.txt，也存在同样问题；重跑脚本即自动回退修复。

### 下一步

- 用户刷新/重开审查页即可按双语类别名审查；按计划分层抽审 200~300 张建可信评测集。

### 风险与待确认

- `labels_with_conf/classes.txt` 位于标签目录内，违反「classes.txt 不得放在 YOLO labels 目录」的元数据约定；本次未移动它（保持只增不改），后续整理数据集时可上移到根目录。
