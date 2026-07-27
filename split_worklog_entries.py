#!/usr/bin/env python3
"""一次性迁移：把工作日志正文条目拆分为「条目文件夹」。

从主文件 `工作日志.md`（条目区）与历史分片 `工作日志_pre-2026-07-23.md`
解析全部条目头（``## YYYY-MM-DD[ HH:MM] | 标题``），将每条正文**逐字**写入
``工作日志条目/`` 下的独立文件，随后把主文件截断到条目区之前（保留规则、
摘要区与索引表；索引表由 `rebuild_worklog_index.py` 随后重建为链接版）。

- 顺序：主文件条目（自上而下）+ 分片条目（自上而下），保持原时间倒序；
- 文件名：`YYYY-MM-DD[_HHMM]_标题.md`，标题中的非法字符按规则替换；
- 分片文件不被修改或删除，由人工在全部验证通过后处理；
- 支持 `--dry-run`：只打印迁移计划，不写任何文件。

用法::

    python split_worklog_entries.py            # 执行迁移
    python split_worklog_entries.py --dry-run  # 只审计
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ENTRY_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})(?:\s+(\d{2}):(\d{2}))?\s*\|\s*(.+?)\s*$")
ILLEGAL_MAP = str.maketrans({"/": "-", "\\": "-", ":": "_", "*": "_", "?": "_",
                             '"': "_", "<": "_", ">": "_", "|": "_"})


def read_lines(path: Path) -> tuple[list[str], str]:
    raw = path.read_bytes().decode("utf-8")
    ending = "\r\n" if raw.count("\r\n") * 2 > raw.count("\n") else "\n"
    return raw.replace("\r\n", "\n").split("\n"), ending


def split_entries(lines: list[str], source: str) -> tuple[int, list[dict]]:
    """返回 (首个条目起始行, 条目列表)；条目正文逐字保留到下一个条目头或文件尾。"""
    starts = [i for i, ln in enumerate(lines) if ENTRY_RE.match(ln)]
    if not starts:
        sys.exit(f"[error] {source} 中未扫到任何条目头")
    entries = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        body = lines[start:end]
        while body and not body[-1].strip():  # 去掉条目间空行，文件尾补单换行
            body.pop()
        m = ENTRY_RE.match(lines[start])
        entries.append({
            "date": m.group(1),
            "time": f"{m.group(2)}{m.group(3)}" if m.group(2) else "",
            "title": m.group(4),
            "body": body,
            "source": source,
            "src_line": start + 1,
        })
    return starts[0], entries


def make_filename(entry: dict, used: set[str]) -> str:
    title = entry["title"].translate(ILLEGAL_MAP)
    title = re.sub(r"\s+", "-", title).strip("-.")
    title = re.sub(r"-{2,}", "-", title)[:60].rstrip("-.")
    stem = entry["date"] + (f"_{entry['time']}" if entry["time"] else "") + "_" + title
    name, n = stem, 2
    while f"{name}.md" in used:
        name, n = f"{stem}_{n}", n + 1
    used.add(f"{name}.md")
    return f"{name}.md"


def main() -> int:
    ap = argparse.ArgumentParser(description="工作日志正文一次性拆分为条目文件夹")
    ap.add_argument("--main", default="工作日志.md")
    ap.add_argument("--shard", default="工作日志_pre-2026-07-23.md")
    ap.add_argument("--out-dir", default="工作日志条目")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不写文件")
    args = ap.parse_args()

    main_path, shard_path = Path(args.main), Path(args.shard)
    out_dir = Path(args.out_dir)
    main_lines, main_ending = read_lines(main_path)
    shard_lines, _ = read_lines(shard_path)

    first_main_entry, entries_main = split_entries(main_lines, main_path.name)
    _, entries_shard = split_entries(shard_lines, shard_path.name)
    entries = entries_main + entries_shard
    print(f"解析条目：主 {len(entries_main)} + 片 {len(entries_shard)} = {len(entries)}")

    used: set[str] = set()
    for e in entries:
        e["filename"] = make_filename(e, used)

    if args.dry_run:
        print("[dry-run] 计划生成的文件（前 5 / 后 5）：")
        for e in entries[:5] + entries[-5:]:
            print(f"  {e['filename']}  <- {e['source']}:{e['src_line']}")
        print("[dry-run] 未写文件。")
        return 0

    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"[error] {out_dir} 已存在且非空，为避免覆盖未继续；请先清空或换 --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    for e in entries:
        (out_dir / e["filename"]).write_bytes(("\n".join(e["body"]) + "\n").encode("utf-8"))

    # 回读校验：每个文件的条目头必须与来源一致。
    bad = []
    for e in entries:
        first = (out_dir / e["filename"]).read_bytes().decode("utf-8").split("\n", 1)[0]
        m = ENTRY_RE.match(first)
        if not m or m.group(1) != e["date"] or m.group(4) != e["title"]:
            bad.append(e["filename"])
    if bad:
        print(f"[error] 回读校验失败 {len(bad)} 个文件：{bad[:5]}")
        return 2
    print(f"[ok] 已写入 {len(entries)} 个条目文件到 {out_dir}/，回读校验全部通过。")

    # 截断主文件到条目区之前（保留规则/摘要/索引表；分片文件不动，待人工处理）。
    main_path.write_bytes(main_ending.join(main_lines[:first_main_entry]).encode("utf-8"))
    print(f"[ok] 主文件已截断到第 {first_main_entry - 1} 行（原条目区已迁入文件夹）。")
    print("[next] 运行 python rebuild_worklog_index.py 重建链接版索引；")
    print("       全部验证通过后，由人工删除分片文件。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
