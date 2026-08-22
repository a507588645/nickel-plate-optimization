## 2026-08-22 19:40 | 原始转储防上传规则固化到 .gitignore 与 AGENTS.md

- **执行者**：Kimi Code
- **任务目标**：防止大文件/中间产物以后再次误提交（承接同日 19:20 条目）
- **状态**：已完成
- **改动文件**：
  - `.gitignore`：将 19:20 的两条具体规则推广为通用模式——`*.jsonl.gz`、`raw_predictions/`、`*_raw_*.jsonl`、`*_raw_*.jsonl.gz`；并注释说明 `kaggle_*/remote_output_v*/` 下归档文件仍入库（曾尝试整体忽略该目录，发现与 495 个已跟踪归档文件冲突后放弃）
  - `AGENTS.md` §10：新增「原始预测转储与中间产物不入库」正式规则，要求新增可复现大文件产物时同步补 `.gitignore`
- **验证**：`git check-ignore -v` 确认新旧转储文件均被忽略，`metrics.json` 等归档 JSON 不受影响；`git ls-files` 确认无已跟踪文件被新规则误伤
- **结论**：规则层（AGENTS.md）+ 强制层（.gitignore）双保险已落地
- **风险与待确认**：历史提交中的旧大文件 blob 仍在 GitHub（含约 89MB LFS 对象），彻底清理需改写历史强推，未执行
- **下一步**：无
