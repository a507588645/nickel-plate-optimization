# L1 人工重标页使用说明

本页只显示原始人工标签，**不显示 `ryw.pt` 候选**。原框只是起点，最终以人工观察为准。

1. 选择一张图；优先级排序来自标签质量审计，不代表旧模型裁决。
2. 点击框选择。拖动框内部移动；拖动边或角调整大小；类别下拉框可改类别；删除可移除误标；“恢复原框”撤销单框改动。
3. 点“新增框”后，在图像空白位置拖出框；用类别下拉框选定正确类别；不需要的新增框删除即可。
4. 人工确认该图已完整重标后，点“标为已完成”。没有缺陷时，删除所有框再标为已完成，导出器会写同名空标签。
5. 绑定保存文件后每次改动自动写入 JSONL；也可最后点“导出 JSONL”。
6. 不能确定时写备注并保持“未完成”，不要把不确定图导入冻结测试集。

导出后运行：

```powershell
python apply_manual_reannotation.py `
  --review-jsonl <导出的 JSONL> `
  --scope-file E:/PythonProject/优化/manual_reannotation_v1/review_scope.txt `
  --require-completed
```

脚本只导出 `completed=true` 的图片。带上 `--scope-file --require-completed` 时，必须本批次
所有图片都已经完成，避免漏审图片被悄悄遗漏。

结果只会写入 `E:\PythonProject\优化\manual_reannotation_v1\labels_human_reviewed_v1`，不会覆盖原始数据集标签。
