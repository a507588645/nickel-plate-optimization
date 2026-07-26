#!/usr/bin/env python3
"""
将 combined_all_pseudo_v1 调整为：
1. 14 类（删除 youpao/heibian/mijixiaokeng/fanxitong）
2. 取消 train/val/test 划分，全部图片放入 images/train/ 与 labels/train/
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(r"E:\PythonProject\datasets\combined_all_pseudo_v1")
NAMES_14 = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark",
]


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


def main() -> None:
    images_all = ROOT / "images_all"
    labels_with_conf = ROOT / "labels_with_conf"
    images_root = ROOT / "images"
    labels_root = ROOT / "labels"

    # 删除旧的划分目录
    for split in ("train", "val", "test"):
        shutil.rmtree(images_root / split, ignore_errors=True)
        shutil.rmtree(labels_root / split, ignore_errors=True)

    train_img_dir = ensure_dir(images_root / "train")
    train_lbl_dir = ensure_dir(labels_root / "train")

    total_images = 0
    kept_boxes = 0
    total_boxes = 0
    empty_labels = 0

    for img in sorted(p for p in images_all.rglob("*") if p.is_file() or p.is_symlink()):
        rel = img.relative_to(images_all)
        rel_txt = rel.with_suffix(".txt")

        # 图片软连接
        dst_img = train_img_dir / rel
        symlink_or_copy(img, dst_img)
        total_images += 1

        # 标签过滤 + 14 类映射
        src_lbl = labels_with_conf / rel_txt
        dst_lbl = train_lbl_dir / rel_txt
        ensure_dir(dst_lbl.parent)

        if not src_lbl.exists():
            dst_lbl.write_text("", encoding="utf-8")
            empty_labels += 1
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
            if cid >= 14:
                continue
            if conf < 0.70:
                continue
            kept_lines.append(" ".join(parts[:5]))
            kept_boxes += 1

        dst_lbl.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
        if not kept_lines:
            empty_labels += 1

    # 重写 data.yaml
    yaml_path = ROOT / "data.yaml"
    yaml_path.write_text(
        f"path: {ROOT}\n"
        "train: images/train\n"
        "val: images/train\n"
        "test: images/train\n"
        f"nc: {len(NAMES_14)}\n"
        f"names: {NAMES_14}\n",
        encoding="utf-8",
    )

    # 重写报告
    report_path = ROOT / "splits" / "distribution.txt"
    report_path.write_text(
        f"14-class no-split dataset summary\n"
        f"=================================\n\n"
        f"total images: {total_images}\n"
        f"all in: images/train, labels/train\n"
        f"total candidate boxes (raw): {total_boxes}\n"
        f"kept boxes (conf>=0.70, class<14): {kept_boxes}\n"
        f"empty labels: {empty_labels}\n",
        encoding="utf-8",
    )

    print(f"[done] images={total_images}, kept_boxes={kept_boxes}, empty_labels={empty_labels}")
    print(f"[yaml] {yaml_path}")


if __name__ == "__main__":
    main()
