## 2026-07-23 21:40 | 双源审查页补聚焦与自动保存（File System Access API）

| 字段 | 内容 |
| --- | --- |
| 执行者 | Kimi |
| 任务目标 | 补齐用户指出的两个缺失：点击框自动聚焦、进度自动保存 |
| 状态 | 已完成 |

- `build_dual_review.py` 重写并重新生成 `label_review_v1\dual\review.html`：
  - **自动聚焦**：点表格行或图上的框 → 自动缩放（小框放到视口中心）+ 黄色高亮 + 行定位；浏览器实测 z/selBox/scroll/行高亮全部生效。
  - **自动保存**：新增「绑定保存文件」——File System Access API 让用户选定本地 `dual_review_progress.jsonl` 后，每次判决防抖 800ms 自动写盘（内容与导出 JSONL 同构，可直接喂 `apply_label_review_decisions.py --dual-jsonl`）；句柄存 IndexedDB，重开页面自动恢复绑定并回读进度；不支持的浏览器退回 localStorage + 手动导出。
- 顶栏常驻保存状态（未绑定/已自动保存 时间）。localStorage 仍作兜底同步。
- 浏览器验证：聚焦高亮截图确认（GT 绿框与候选红虚线框同目标重叠的对比可见）；文件选择器需用户手势，绑定路径未自动测，代码为标准 FS API 用法。
- 下一步：用户开始正式审查；审查完成后用导出/自动保存的 JSONL 跑 apply 生成 `labels_reviewed_v1`。
