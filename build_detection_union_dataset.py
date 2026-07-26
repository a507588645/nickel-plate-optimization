#!/usr/bin/env python3
"""
L1 检测线：去重汇总数据集构建（原标签版，不划分）。

用户决策（2026-07-23）：
- 标签只用原始标签：dianjienie（17 类按序 = ryw.pt 18 类前 17 个）、enterprise_label_dedup_v1/labels_original（18 类 id 0-13）。
- Dianjinie0211 只进图不进标签；nietiao 不纳入。
- classify_2types / classify_4types_dianji / 0309 / 0306 待标注集：进图、暂不生成标签。
- 不涉及等级分类数据集（0309_classification、dianjinie_grade_classification_v1 排除）。
- 不划分 train/valid/test，直接汇总。

产物：`E:\\PythonProject\\datasets\\detection_union_original_labels_v1\\`
- images/ 硬链接（统一命名 <source>__<split>__<stem>）
- labels/ 硬链接（仅 dianjienie 与 dedup_v1 有标签的图）
- classes.txt（18 类基准，含不可评标记；**在数据集根目录，不在 labels/**）
- manifests/samples.csv、dataset_summary.json、README.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(r"E:/PythonProject/datasets")
OUT_DEFAULT = ROOT / "detection_union_original_labels_v1"

# (source 名, 图片目录列表[(split, dir)], 标签目录函数 or None, 是否有标签)
SOURCES = [
    {
        "name": "dianjienie",
        "splits": [("train", ROOT / "dianjienie/train/images"), ("valid", ROOT / "dianjienie/valid/images")],
        "label_dir": lambda split: ROOT / "dianjienie" / split / "labels",
        "has_labels": True,
        "note": "17 类按序映射 18 类（id 不变；id2 oilstain=yellowstain）",
    },
    {
        "name": "enterprise_label_dedup_v1",
        "splits": [("all", ROOT / "enterprise_label_dedup_v1/images")],
        "label_dir": lambda split: ROOT / "enterprise_label_dedup_v1/labels_original",
        "has_labels": True,
        "note": "用户手标，18 类 id 0-13",
    },
    {
        "name": "Dianjinie0211",
        "splits": [("train", ROOT / "Dianjinie0211/train/images"), ("valid", ROOT / "Dianjinie0211/valid/images")],
        "label_dir": None,
        "has_labels": False,
        "note": "只进图（5 类标签不映射，待 ryw.pt 检查重建）",
    },
    {
        "name": "classify_2types",
        "splits": [("train", ROOT / "classify_2types/train"), ("valid", ROOT / "classify_2types/valid")],
        "label_dir": None,
        "has_labels": False,
        "note": "分类目录，进图无标签",
    },
    {
        "name": "classify_4types_dianji",
        "splits": [("train", ROOT / "classify_4types_dianji/train"), ("valid", ROOT / "classify_4types_dianji/valid")],
        "label_dir": None,
        "has_labels": False,
        "note": "分类目录，进图无标签（等级语义不进本检测集）",
    },
    {
        "name": "enterprise_dianji_0309_detection_unlabeled",
        "splits": [("all", ROOT / "enterprise_dianji_0309_detection_unlabeled/images")],
        "label_dir": None,
        "has_labels": False,
        "note": "缺陷提示目录，进图无标签",
    },
    {
        "name": "enterprise_dianjie_0306_detection_unlabeled",
        "splits": [("all", ROOT / "enterprise_dianjie_0306_detection_unlabeled/images")],
        "label_dir": None,
        "has_labels": False,
        "note": "进图无标签",
    },
]

CLASSES_18 = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]
NOT_EVALUABLE = {"youpao", "fanxitong"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="L1 去重汇总数据集构建（原标签版，不划分）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def sha256_of(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.hardlink_to(src)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    print("=== L1 去重汇总数据集构建（原标签版）===")
    print(f"output: {out}")
    print(f"sources: {len(SOURCES)}（nietiao 与等级分类集已排除）")

    seen: dict[str, str] = {}  # sha256 -> 保留名
    rows = []
    dup_skipped = []
    label_class_hist: dict[int, int] = {}
    label_img_hist: dict[int, int] = {}
    n_label_files = 0
    missing_label = []

    for src in SOURCES:
        for split, img_dir in src["splits"]:
            if not img_dir.exists():
                print(f"!! 缺目录: {img_dir}", file=sys.stderr)
                continue
            for img in sorted(img_dir.rglob("*")):
                if img.suffix.lower() not in IMG_EXTS or not img.is_file():
                    continue
                digest = sha256_of(img)
                new_stem = f"{src['name']}__{split}__{img.stem}"
                if digest in seen:
                    dup_skipped.append({
                        "skipped": new_stem, "kept": seen[digest], "sha256": digest[:16],
                    })
                    continue
                seen[digest] = new_stem

                lab_src = None
                if src["has_labels"]:
                    ld = src["label_dir"](split)
                    cand = ld / f"{img.stem}.txt"
                    if cand.exists():
                        lab_src = cand
                    else:
                        missing_label.append(f"{src['name']}/{split}/{img.stem}")

                if not args.dry_run:
                    dst_img = out / "images" / f"{new_stem}{img.suffix.lower()}"
                    link_or_copy(img, dst_img)
                    if lab_src is not None:
                        link_or_copy(lab_src, out / "labels" / f"{new_stem}.txt")

                cls_seen = set()
                n_box = 0
                if lab_src is not None:
                    n_label_files += 0 if args.dry_run else 1
                    for line in lab_src.read_text(encoding="utf-8", errors="ignore").splitlines():
                        p = line.split()
                        if len(p) < 5:
                            continue
                        try:
                            cid = int(float(p[0]))
                        except ValueError:
                            continue
                        label_class_hist[cid] = label_class_hist.get(cid, 0) + 1
                        cls_seen.add(cid)
                        n_box += 1
                    for cid in cls_seen:
                        label_img_hist[cid] = label_img_hist.get(cid, 0) + 1
                rows.append({
                    "union_name": new_stem,
                    "source": src["name"],
                    "source_split": split,
                    "source_relpath": str(img.relative_to(img_dir)),
                    "has_label": bool(lab_src),
                    "n_boxes": n_box,
                    "sha256": digest,
                })

    print(f"图片: 保留 {len(rows)}，完全重复跳过 {len(dup_skipped)}")
    labeled = sum(1 for r in rows if r["has_label"])
    print(f"含标签: {labeled}；无标签: {len(rows) - labeled}；dianjienie/dedup 缺标签: {len(missing_label)}")
    if args.dry_run:
        print("[dry-run] 未写入文件")
        print("类分布(框):", {CLASSES_18[k]: v for k, v in sorted(label_class_hist.items())})
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text(
        "# L1 汇总数据集类别表（ryw.pt 18 类基准）\n"
        + "\n".join(
            f"{i}: {n}" + ("  # 不可评：全库零实例" if n in NOT_EVALUABLE else "")
            for i, n in enumerate(CLASSES_18)
        )
        + "\n",
        encoding="utf-8",
    )

    man_dir = out / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    with (man_dir / "samples.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (man_dir / "dedup_skipped.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["skipped", "kept", "sha256"])
        w.writeheader()
        w.writerows(dup_skipped)
    if missing_label:
        (man_dir / "missing_labels.txt").write_text("\n".join(missing_label), encoding="utf-8")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "name": "detection_union_original_labels_v1",
        "policy": {
            "labels": "仅原始标签（dianjienie 17 类按序映射、labels_original 18 类 id0-13）；Dianjinie0211 标签不并入",
            "excluded": ["nietiao", "enterprise_dianji_0309_classification", "dianjinie_grade_classification_v1"],
            "split": "不划分（用户 2026-07-23：等 ryw.pt 检查标签准确性后再定）",
            "dedup": "SHA-256 完全去重，保留先出现者（有标签来源优先处理）",
        },
        "images_total": len(rows),
        "images_labeled": labeled,
        "images_unlabeled": len(rows) - labeled,
        "label_files": n_label_files,
        "dup_skipped": len(dup_skipped),
        "missing_label_in_labeled_sources": len(missing_label),
        "per_source": {},
        "class_box_hist": {CLASSES_18[k]: v for k, v in sorted(label_class_hist.items())},
        "class_image_hist": {CLASSES_18[k]: v for k, v in sorted(label_img_hist.items())},
        "classes_not_evaluable": sorted(NOT_EVALUABLE),
    }
    for r in rows:
        s = summary["per_source"].setdefault(r["source"], {"images": 0, "labeled": 0})
        s["images"] += 1
        s["labeled"] += int(r["has_label"])
    (out / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (out / "README.md").write_text(
        "# detection_union_original_labels_v1\n\n"
        "L1 检测线去重汇总数据集（原标签版）。用户 2026-07-23 决策构建。\n\n"
        "- 标签：仅 `dianjienie`（17 类按序=18 类前 17 个，见 `类别映射_dianjienie17_to_ryw18.md`）"
        "与 `enterprise_label_dedup_v1/labels_original`（18 类 id 0-13，用户手标）。\n"
        "- 仅进图：Dianjinie0211、classify_2types、classify_4types_dianji、0309/0306 待标注集（标签待 ryw.pt 检查重建）。\n"
        "- 排除：nietiao（分割任务）、等级分类集（0309_classification、grade 视图）。\n"
        "- 去重：SHA-256 完全去重（保留先出现者，有标签来源先处理）；未做近重复分组。\n"
        "- **不划分** train/valid/test；后续用 ryw.pt 检查标签准确性后再划分。\n"
        "- youpao / fanxitong 标记不可评（全库零实例）。\n\n"
        f"图片 {len(rows)}（有标签 {labeled} / 无标签 {len(rows) - labeled}），重复跳过 {len(dup_skipped)}。\n",
        encoding="utf-8",
    )
    print(f"产物: {out}")
    print(json.dumps(summary["per_source"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
