#!/usr/bin/env python3
"""
将 combined_all_pseudo_v1 重新组织为：
1. 14 类（删除 youpao/heibian/mijixiaokeng/fanxitong）
2. 图片文件名扁平化，不再保留来源数据集子目录
3. 恢复 train/val/test 70/15/15 划分
"""
from __future__ import annotations

import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"E:\PythonProject\datasets\combined_all_pseudo_v1")
IMAGES_ALL = ROOT / "images_all"
LABELS_WITH_CONF = ROOT / "labels_with_conf"
NAMES_14 = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark",
]
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def symlink_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def flatten_name(img: Path) -> str:
    """返回 images_all 中图片的扁平文件名。images_all 已扁平化，直接取原文件名。"""
    return img.name


def main() -> None:
    # 清理旧的 images/ 和 labels/
    images_root = ROOT / "images"
    labels_root = ROOT / "labels"
    if images_root.exists():
        shutil.rmtree(images_root)
    if labels_root.exists():
        shutil.rmtree(labels_root)

    # 收集全部图片并随机划分
    all_images = sorted(p for p in IMAGES_ALL.rglob("*") if p.is_file() or p.is_symlink())
    rng = random.Random(SEED)
    rng.shuffle(all_images)

    n = len(all_images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:],
    }

    class_counts = Counter()
    per_dataset_counts = defaultdict(lambda: Counter())
    total_boxes = 0
    kept_boxes = 0

    for split, imgs in splits.items():
        img_dir = ensure_dir(images_root / split)
        lbl_dir = ensure_dir(labels_root / split)
        for img in imgs:
            new_name = flatten_name(img)
            dst_img = img_dir / new_name
            symlink_or_copy(img, dst_img)

            # 来源统计
            rel = img.relative_to(IMAGES_ALL)
            ds_name = rel.parts[0]
            per_dataset_counts[split][ds_name] += 1

            # 对应标签
            rel_txt = rel.with_suffix(".txt")
            src_lbl = LABELS_WITH_CONF / rel_txt
            dst_lbl = (lbl_dir / new_name).with_suffix(".txt")

            if not src_lbl.exists():
                dst_lbl.write_text("", encoding="utf-8")
                continue

            kept_lines: list[str] = []
            for line in src_lbl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                total_boxes += 1
                try:
                    cid = int(parts[0])
                    conf = float(parts[5])
                except ValueError:
                    continue
                if cid >= 14 or conf < 0.70:
                    continue
                kept_lines.append(" ".join(parts[:5]))
                kept_boxes += 1
                class_counts[cid] += 1

            dst_lbl.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")

        print(f"[{split}] images={len(imgs)}")

    # 重写 data.yaml
    yaml_path = ROOT / "data.yaml"
    yaml_path.write_text(
        f"path: {ROOT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(NAMES_14)}\n"
        f"names: {NAMES_14}\n",
        encoding="utf-8",
    )

    # 报告
    report = {
        "split_counts": {k: len(v) for k, v in splits.items()},
        "per_dataset_counts": {k: dict(v) for k, v in per_dataset_counts.items()},
        "class_counts": dict(class_counts),
        "total_boxes_raw": total_boxes,
        "kept_boxes": kept_boxes,
    }
    splits_dir = ensure_dir(ROOT / "splits")
    import json
    (splits_dir / "distribution.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = [
        "14-class flattened split dataset summary",
        "========================================",
        "",
        f"train: {len(splits['train'])} images",
        f"val:   {len(splits['val'])} images",
        f"test:  {len(splits['test'])} images",
        "",
        f"raw boxes: {total_boxes}",
        f"kept boxes (conf>=0.70, class<14): {kept_boxes}",
        "",
        "Class distribution:",
    ]
    for cid in sorted(class_counts):
        summary_lines.append(f"  {cid:2d} {NAMES_14[cid]:15s}: {class_counts[cid]}")
    (splits_dir / "distribution.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[done] raw_boxes={total_boxes}, kept_boxes={kept_boxes}")
    print(f"[yaml] {yaml_path}")


if __name__ == "__main__":
    main()
