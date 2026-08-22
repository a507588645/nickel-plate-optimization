## 2026-08-22 17:35 | GitHub 远程仓库改名为 nickel-plate-optimization

- **执行者**：Kimi Code
- **任务目标**：用户反馈 git push 后 GitHub 仓库名显示为 `------`，希望改为「镍板优化」含义的英文名
- **状态**：已完成
- **原因**：GitHub 仓库名不支持中文，建仓库时中文「镍板项目优化」被过滤后只剩分隔符，变成 `------`
- **改动**：
  - 下载 gh CLI v2.98.0 便携版（`/tmp/bin/gh.exe`，winget 不可用）并完成网页设备授权（首次因 `github.com/login/oauth/access_token` 网络 EOF 失败，重试成功）
  - `gh repo rename nickel-plate-optimization --repo a507588645/------`
  - 本地 `git remote set-url origin git@github.com:a507588645/nickel-plate-optimization.git`
- **验证**：`git fetch origin --dry-run` 成功（exit 0）
- **产物/路径**：远程仓库 https://github.com/a507588645/nickel-plate-optimization
- **未完成项**：无
- **风险与待确认**：本地分支与目录名不受影响；本地有未提交的 `README.md`、`工作日志.md` 改动（与本次操作无关，未触碰）
