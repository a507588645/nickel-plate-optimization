"""一次性脚本：把 UI config.json 的 webapi 段配置为 autoLaunch 指向优化分支 worktree。"""
import json
import os

p = os.path.expandvars(r"%APPDATA%\com.jinchuan.nickel-detection\config.json")
with open(p, encoding="utf-8") as f:
    cfg = json.load(f)

cfg["webapi"].update(
    {
        "executable": r"C:\Users\23916\AppData\Local\Temp\jcw_b3clean_wt\.venv\Scripts\python.exe",
        "args": ["server.py"],
        "workingDir": r"C:\Users\23916\AppData\Local\Temp\jcw_b3clean_wt",
        "autoLaunch": True,
    }
)

with open(p, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
print(json.dumps(cfg["webapi"], ensure_ascii=False, indent=2))
