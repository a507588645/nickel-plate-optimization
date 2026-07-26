#!/usr/bin/env python3
"""
给 union_original_labels_v1 汇总集接等级标签。

原理（用户 2026-07-23 确认「发现1=同一批照片复用」）：
classify_4types_dianji 与 enterprise_dianji_0309_classification 的等级标签
可通过 SHA-256 传递给汇总集中的字节级同图。

产物：更新 manifests/samples.csv（增列 grade、grade_source），
写出 manifests/grade_join_report.json，并更新 dataset_summary.json。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

UNION = Path(r"E:/PythonProject/datasets/union_original_labels_v1")
GRADE_SOURCES = [
    ("classify_4types_dianji", Path(r"E:/PythonProject/datasets/classify_4types_dianji")),
    ("enterprise_dianji_0309_classification", Path(r"E:/PythonProject/datasets/enterprise_dianji_0309_classification/images")),
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256_of(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="汇总集等级标签哈希传递", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--union", type=Path, default=UNION)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    samples_path = args.union / "manifests" / "samples.csv"
    if not samples_path.exists():
        print(f"缺少 {samples_path}", file=sys.stderr)
        return 2

    # 1) sha -> (grade, source)
    grade_map: dict[str, tuple[str, str]] = {}
    grade_src_count: dict[str, int] = {}
    for name, root in GRADE_SOURCES:
        if not root.exists():
            print(f"!! 缺目录 {root}", file=sys.stderr)
            continue
        for grade_dir in sorted(root.rglob("*")):
            if not grade_dir.is_dir():
                continue
            grade = grade_dir.name
            if grade not in {"3#", "9950", "9996", "bad", "good"}:
                continue
            for img in sorted(grade_dir.glob("*")):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                grade_map.setdefault(sha256_of(img), (grade, name))
                grade_src_count[name] = grade_src_count.get(name, 0) + 1

    print(f"等级图索引: {len(grade_map)} 张（来源 {grade_src_count}）")

    rows = list(csv.DictReader(samples_path.open(encoding="utf-8-sig")))
    matched = 0
    per_grade: dict[str, int] = {}
    unmatched_grade_images = len(grade_map)
    for r in rows:
        hit = grade_map.get(r["sha256"])
        if hit:
            r["grade"], r["grade_source"] = hit
            matched += 1
            per_grade[hit[0]] = per_grade.get(hit[0], 0) + 1
        else:
            r["grade"], r["grade_source"] = "", ""

    print(f"汇总集 {len(rows)} 张中匹配到等级标签: {matched}（{per_grade}）")
    if args.dry_run:
        print("[dry-run] 未写入")
        return 0

    fields = list(rows[0].keys())
    for f in ("grade", "grade_source"):
        if f not in fields:
            fields.append(f)
    with samples_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "SHA-256 同图传递（用户确认发现1=同一批照片复用）",
        "grade_index_size": len(grade_map),
        "grade_source_counts": grade_src_count,
        "union_images": len(rows),
        "matched": matched,
        "per_grade": per_grade,
        "unmatched_grade_index": unmatched_grade_images - matched,
        "note": "grade 列为空=该图无等级真值；检测标签与等级标签维度独立，可分别划分",
    }
    (args.union / "manifests" / "grade_join_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 更新 summary
    sp = args.union / "dataset_summary.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        s["grade_join"] = report
        sp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已更新 samples.csv / grade_join_report.json / dataset_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
