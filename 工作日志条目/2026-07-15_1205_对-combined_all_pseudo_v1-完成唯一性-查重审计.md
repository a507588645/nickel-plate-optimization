## 2026-07-15 12:05 | 对 combined_all_pseudo_v1 完成唯一性/查重审计

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi Code CLI |
| 任务目标 | 对汇总数据集 `combined_all_pseudo_v1` 做唯一性检查与查重，识别完全重复、文件名副本、同源物理板及跨 split 泄漏 |
| 状态 | 已完成 |
| 关联用户请求 | 对汇总数据进行唯一性检查，查重 |

### 改动文件

- 新增：`E:/PythonProject/优化/audit_combined_uniqueness.py`

### 产物路径

- 审计目录：`E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/uniqueness_audit/`
- 总览报告：`report.md`
- 完全重复清单：`exact_duplicates.csv`
- 文件名副本清单：`filename_copy_groups.csv`
- 同源物理板组清单：`physical_groups.csv`
- 每张图片元数据：`all_images_md5.csv`
- 统计摘要：`summary.json`

### 关键命令

```text
E:/PythonProject/优化/.venv_classify/Scripts/python.exe E:/PythonProject/优化/audit_combined_uniqueness.py
```

### 审计结论

- 总图片数：7396
- 完全重复组：2352 组，涉及图片：6484 张（87.7%）
- **跨 split 完全重复组：1411 组** — 存在严重数据泄漏
- 文件名副本组：0（副本文件已混入重复组，未单独形成 original+copy 对）
- 近重复对：未启用（默认关闭，可加 `--near-duplicate`）
- 疑似同源物理板组：1773 组，其中跨 split 1024 组

### 典型重复示例

- 同一张图同时出现在 `classify_2types_train_good_7.png`、`dianjienie_images_train_7.png`、`dianjienie_images_val_55555(1).png`、`dianjienie_valid_images_55555(1).png` 等 6 个路径，横跨 train/val。
- 大量 `Dianjinie0211_images_train_... - 副本.png` 与无副本版本、classify 版本互为重复，并跨 train/val/test。

### 风险

- 当前 70/15/15 划分不可信：同一物理样本的多个副本出现在不同 split，会导致验证/测试指标虚高。
- 训练前必须去重，否则任何模型评估都失去意义。

### 下一步

1. 按 `exact_duplicates.csv` 去重：每组只保留一份（建议保留 train 中的版本，或按来源保留最可信的一份）。
2. 同步删除对应标签文件。
3. 对去重后的图片重新做 70/15/15 划分（同一物理板 group 必须整体划入同一 split）。
4. 重新生成 `labels/`、`labels_with_conf/`、伪标签审查页和等级特征。
5. 去重完成后再进行 LightGBM 训练或 YOLO 训练。
