#!/usr/bin/env python3
"""汇总多视图审查导出 JSONL 中的可见性提升率。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总审查 JSONL 的可见性提升统计")
    parser.add_argument("jsonl", type=Path, help="review.html 导出的 JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选：写入 summary JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    total = len(rows)
    filled = 0
    raw_no = 0
    enh_yes_from_no = 0
    strong_yes_from_no = 0
    any_yes_from_no = 0
    still_no = 0
    decisions = Counter()
    false_texture_notes = 0

    for row in rows:
        vis = row.get("visibility") or {}
        raw = vis.get("raw") or ""
        enh = vis.get("enhanced") or ""
        strong = vis.get("strong") or ""
        if raw or enh or strong:
            filled += 1
        if raw == "no":
            raw_no += 1
            e_ok = enh == "yes"
            s_ok = strong == "yes"
            if e_ok:
                enh_yes_from_no += 1
            if s_ok:
                strong_yes_from_no += 1
            if e_ok or s_ok:
                any_yes_from_no += 1
            if not e_ok and not s_ok and (enh == "no" or strong == "no"):
                still_no += 1
        decisions[row.get("image_status") or "pending"] += 1
        note = (row.get("note") or "").lower()
        if "假" in note or "false" in note or "纹理" in note:
            false_texture_notes += 1

    summary = {
        "jsonl": str(args.jsonl.resolve()),
        "total_images": total,
        "visibility_filled": filled,
        "raw_not_visible": raw_no,
        "enhanced_rescue_from_raw_no": enh_yes_from_no,
        "strong_rescue_from_raw_no": strong_yes_from_no,
        "any_enhance_rescue_from_raw_no": any_yes_from_no,
        "still_not_visible_after_enhance": still_no,
        "enhanced_lift_rate": (enh_yes_from_no / raw_no) if raw_no else None,
        "strong_lift_rate": (strong_yes_from_no / raw_no) if raw_no else None,
        "any_lift_rate": (any_yes_from_no / raw_no) if raw_no else None,
        "image_status_counts": dict(decisions),
        "notes_mentioning_false_texture_approx": false_texture_notes,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
