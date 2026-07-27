## 2026-07-13 | 删除 visualizations 并构建企业硬链接数据集

| 字段 | 内容 |
| --- | --- |
| 执行者 | Codex |
| 任务目标 | 清理伪标签可视化；规定派生产物不得写入原始数据集；从两批企业原图构建新的硬链接分类/检测数据集 |
| 状态 | 已完成 |

### 删除与规则修改

- 删除 `E:\PythonProject\优化\visualizations`。
- `AGENTS.md` 新增强制规则：伪标签、软连接视图、特征、分类结果和临时清单统一写入当前执行目录；不得混入原始数据集根目录。
- 企业源目录中的自制 `classify`、`detect`、`defect` 结果未经确认不得作为真值。

### 新数据集

| 路径 | 内容 |
| --- | --- |
| `E:\PythonProject\datasets\enterprise_dianjie_0306_detection_unlabeled` | 电解镍 0306，68 张待标注检测图片硬链接 |
| `E:\PythonProject\datasets\enterprise_dianji_0309_classification` | 电积镍明确等级：9950=74、3#=12 |
| `E:\PythonProject\datasets\enterprise_dianji_0309_detection_unlabeled` | 电积镍其余缺陷提示图片 37 张，待检测标注 |

### 构建入口与记录

- 脚本：`build_enterprise_nickel_datasets.py`，支持 `--help` 和 `--dry-run`。
- 清单：`E:\PythonProject\优化\enterprise_dataset_records`。
- 未创建 train/valid 划分，避免再次引入未经确认的人工划分。

### 验证

- 68、86、37 个目标全部通过 `os.path.samefile` 验证为 NTFS 硬链接，软链接数量为 0。
- 电积镍分类集与检测集完全重复图片交叉数量为 0。
- 原企业源目录及原始检测/分类数据集未修改。

### 下一步

1. 为两套待标注检测集制定独立、版本化的自动标注方案，产物写入 `E:\PythonProject\优化`。
2. 明确分组字段后再创建 train/valid/test，不在本次擅自划分。
