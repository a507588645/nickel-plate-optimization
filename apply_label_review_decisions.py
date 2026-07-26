#!/usr/bin/env python3
"""
双轨审查结果应用：GT 留删 + 候选补入 → union labels_reviewed_v1（五列）。

输入：
- GT 轨 JSONL（pseudo_label_reviews/gt/review.html 导出）：keep/accepted 保留原框；
  remove/rejected 删除；uncertain 默认保留（--drop-uncertain 剔除）；未出现在 JSONL 的图 = 未审，原标签原样保留。
- 候选轨 JSONL（pseudo_label_reviews/candidates/review.html 导出）：keep/accepted 补入；
  其余（remove/rejected/uncertain/pending/未出现）默认不补（--add-uncertain 可把 uncertain 也补入）。

输出：
- `E:\\PythonProject\\datasets\\union_original_labels_v1\\labels_reviewed_v1\\`（五列，新标签版本，
  不覆盖 labels/ 原标签与任何 labels_with_conf）
- `manifests/labels_reviewed_v1_report.json`
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

UNION = Path(r"E:/PythonProject/datasets/union_original_labels_v1")
KEEP = {"keep", "accepted"}
DROP = {"remove", "removed", "delete", "deleted", "rejected"}

CLASSES_18 = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="双轨审查结果应用（GT 留删 + 候选补入）",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--union", type=Path, default=UNION)
    p.add_argument("--gt-jsonl", type=Path, default=None, help="GT 轨审查导出 JSONL（可选）")
    p.add_argument("--cand-jsonl", type=Path, default=None, help="候选轨审查导出 JSONL（可选）")
    p.add_argument("--dual-jsonl", type=Path, default=None,
                   help="双源审查页（label_review_v1/dual/review.html）导出的 JSONL；与 gt/cand 互斥")
    p.add_argument("--output-suffix", default="labels_reviewed_v1")
    p.add_argument("--drop-uncertain", action="store_true", help="GT 中 uncertain 框也删除")
    p.add_argument("--add-uncertain", action="store_true", help="候选中 uncertain 框也补入")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def read_jsonl(path: Path | None) -> dict[str, list[dict]]:
    """image_name -> kept box list [[class_id, cx, cy, w, h], ...]（按各自规则过滤后）"""
    return {}


def load_decisions(path: Path, mode: str, args) -> tuple[dict[str, list], dict]:
    """mode='gt' 或 'cand'。返回 image_stem -> boxes + stats。"""
    kept: dict[str, list] = {}
    stats = defaultdict(int)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stem = Path(rec["image_name"]).stem
            boxes = []
            for bd in rec.get("box_decisions", []):
                d = bd.get("decision", "pending")
                if mode == "gt":
                    if d in KEEP:
                        boxes.append([bd["class_id"], *bd["xywh"]])
                        stats["gt_kept"] += 1
                    elif d in DROP:
                        stats["gt_dropped"] += 1
                    elif d == "uncertain":
                        if args.drop_uncertain:
                            stats["gt_uncertain_dropped"] += 1
                        else:
                            boxes.append([bd["class_id"], *bd["xywh"]])
                            stats["gt_kept"] += 1
                    else:
                        stats["gt_pending_skipped"] += 1
                else:
                    if d in KEEP:
                        boxes.append([bd["class_id"], *bd["xywh"]])
                        stats["cand_added"] += 1
                    elif d == "uncertain" and args.add_uncertain:
                        boxes.append([bd["class_id"], *bd["xywh"]])
                        stats["cand_added_uncertain"] += 1
                    else:
                        stats["cand_skipped"] += 1
            kept[stem] = boxes
            stats[f"{mode}_images_reviewed"] += 1
    return kept, stats


def iou_xywh(a, b) -> float:
    """归一化 xywh 的 IoU。"""
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_dual(path: Path, args) -> tuple[dict[str, list], dict[str, list], dict[str, list], dict]:
    """读双源 JSONL：返回 gt_drops、cand_adds、cand_replaces（stem->boxes）、stats。"""
    gt_drops: dict[str, list] = defaultdict(list)
    cand_adds: dict[str, list] = defaultdict(list)
    cand_replaces: dict[str, list] = defaultdict(list)
    stats = defaultdict(int)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stem = Path(rec["image_name"]).stem
            stats["dual_images_reviewed"] += 1
            for bd in rec.get("box_decisions", []):
                src = bd.get("source")
                d = bd.get("decision", "pending")
                box = [bd["class_id"], *bd["xywh"]]
                if src == "gt":
                    if d in DROP:
                        gt_drops[stem].append(box)
                        stats["gt_dropped"] += 1
                    elif d == "uncertain" and args.drop_uncertain:
                        gt_drops[stem].append(box)
                        stats["gt_uncertain_dropped"] += 1
                    else:
                        stats["gt_kept"] += 1  # keep / uncertain(保留) / pending 默认保留
                elif src == "candidate":
                    if d in KEEP:
                        cand_adds[stem].append(box)
                        stats["cand_added"] += 1
                    elif d == "replace":
                        cand_replaces[stem].append({"box": box, "gt": bd.get("replace_gt")})
                        stats["cand_replace_marked"] += 1
                    elif d == "uncertain" and args.add_uncertain:
                        cand_adds[stem].append(box)
                        stats["cand_added_uncertain"] += 1
                    else:
                        stats["cand_skipped"] += 1  # remove/pending/未标注 默认不补
    return gt_drops, cand_adds, cand_replaces, stats


def boxes_match(a, b, tol=1e-4) -> bool:
    return a[0] == b[0] and all(abs(x - y) < tol for x, y in zip(a[1:], b[1:]))


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = args.union / args.output_suffix
    print("=== 双轨审查应用 → labels_reviewed_v1 ===")
    print(f"gt_jsonl={args.gt_jsonl}  cand_jsonl={args.cand_jsonl}  dual_jsonl={args.dual_jsonl}")

    gt_dec, cand_dec, stats = {}, {}, defaultdict(int)
    gt_drops, cand_adds, cand_replaces = {}, {}, {}
    replaced_pairs: list[dict] = []
    if args.dual_jsonl:
        gt_drops, cand_adds, cand_replaces, s = load_dual(args.dual_jsonl, args)
        stats.update(s)
    else:
        if args.gt_jsonl:
            gt_dec, s = load_decisions(args.gt_jsonl, "gt", args)
            stats.update(s)
        if args.cand_jsonl:
            cand_dec, s = load_decisions(args.cand_jsonl, "cand", args)
            stats.update(s)

    rows = list(csv.DictReader((args.union / "manifests" / "samples.csv").open(encoding="utf-8-sig")))
    n_written = 0
    per_class = defaultdict(int)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.txt"):
            old.unlink()

    for r in rows:
        name = r["union_name"]
        has_gt = r.get("has_label") == "True"
        boxes: list[list] = []
        if has_gt:
            if args.dual_jsonl:
                # 双源模式：原标签减去判决删除的框
                lf = args.union / "labels" / f"{name}.txt"
                if lf.exists():
                    for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                        p = line.split()
                        if len(p) < 5:
                            continue
                        box = [int(float(p[0]))] + [float(x) for x in p[1:5]]
                        if any(boxes_match(box, d) for d in gt_drops.get(name, [])):
                            stats["gt_dropped_applied"] += 1
                            continue
                        boxes.append(box)
                        stats["gt_kept_total"] += 1
            elif name in gt_dec:
                boxes.extend(gt_dec[name])  # 已审：按决定
            else:
                # 未审：原标签原样保留
                lf = args.union / "labels" / f"{name}.txt"
                if lf.exists():
                    for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                        p = line.split()
                        if len(p) >= 5:
                            boxes.append([int(float(p[0]))] + [float(x) for x in p[1:5]])
                            stats["gt_unreviewed_kept"] += 1
        if args.dual_jsonl:
            # 替换GT：优先用户显式配对的 GT（联动已标删除），否则同类最大 IoU≥0.1 自动配对；配对不上转补入
            for ent in cand_replaces.get(name, []):
                rb, gtc = ent["box"], ent.get("gt")
                target = None
                explicit = False
                if gtc:
                    target = [rb[0]] + [float(v) for v in gtc]
                    explicit = True
                else:
                    best_i, best_v = -1, 0.0
                    for i, gb in enumerate(boxes):
                        if gb[0] != rb[0]:
                            continue
                        v = iou_xywh(gb, rb)
                        if v > best_v:
                            best_i, best_v = i, v
                    if best_i >= 0 and best_v >= 0.1:
                        target = boxes[best_i]
                if target is not None:
                    removed = False
                    for j in range(len(boxes) - 1, -1, -1):
                        if boxes_match(boxes[j], target):
                            boxes.pop(j)
                            removed = True
                    boxes.append(rb)
                    replaced_pairs.append({
                        "image": name, "class": CLASSES_18[rb[0]],
                        "gt_box": target[1:], "cand_box": rb[1:],
                        "explicit": explicit, "gt_removed_here": removed,
                    })
                    stats["cand_replaced_explicit" if explicit else "cand_replaced_auto"] += 1
                else:
                    boxes.append(rb)
                    stats["cand_replace_as_add"] += 1
            boxes.extend(cand_adds.get(name, []))
        elif name in cand_dec:
            boxes.extend(cand_dec[name])

        if not args.dry_run and (boxes or has_gt):
            with (out_dir / f"{name}.txt").open("w", encoding="utf-8") as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
            n_written += 1
        for b in boxes:
            per_class[b[0]] += 1
        stats["images_total"] += 1
        if boxes:
            stats["images_with_boxes"] += 1

    stats["label_files_written"] = n_written
    stats["per_class_boxes"] = {CLASSES_18[k]: v for k, v in sorted(per_class.items())}
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gt_jsonl": str(args.gt_jsonl), "cand_jsonl": str(args.cand_jsonl),
        "output_dir": str(out_dir),
        "policy": {
            "gt_unreviewed": "原样保留", "gt_uncertain": "删除" if args.drop_uncertain else "保留",
            "cand_uncertain": "补入" if args.add_uncertain else "不补",
            "note": "labels/ 原标签与 labels_with_conf 均未改动；本目录为新标签版本 v1",
        },
        "stats": dict(stats),
        "replaced_pairs": replaced_pairs,
    }
    if not args.dry_run:
        (args.union / "manifests" / f"{args.output_suffix}_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report["stats"], ensure_ascii=False, indent=2))
    print(f"产物: {out_dir}" + ("（dry-run 未写入）" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
