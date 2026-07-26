#!/usr/bin/env python3
"""
将 combined_all_pseudo_v1 中的中间产物 images_all/ 和 labels_with_conf/ 也扁平化，
使其与 images/train|val|test、labels/train|val|test 的扁平结构一致。
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(r"E:\PythonProject\datasets\combined_all_pseudo_v1")
IMAGES_ALL = ROOT / "images_all"
LABELS_WITH_CONF = ROOT / "labels_with_conf"


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


def flatten_name(img: Path, base_dir: Path) -> str:
    rel = img.relative_to(base_dir)
    stem = str(rel.with_suffix("")).replace("\\", "_").replace("/", "_")
    return f"{stem}{rel.suffix}"


def main() -> None:
    if not IMAGES_ALL.exists():
        raise FileNotFoundError(f"{IMAGES_ALL} not found")
    if not LABELS_WITH_CONF.exists():
        raise FileNotFoundError(f"{LABELS_WITH_CONF} not found")

    images_flat = ROOT / "images_all_flat"
    labels_flat = ROOT / "labels_with_conf_flat"

    # 清理临时目录
    if images_flat.exists():
        shutil.rmtree(images_flat)
    if labels_flat.exists():
        shutil.rmtree(labels_flat)

    images_flat = ensure_dir(images_flat)
    labels_flat = ensure_dir(labels_flat)

    total_images = 0
    total_labels = 0

    for img in sorted(p for p in IMAGES_ALL.rglob("*") if p.is_file() or p.is_symlink()):
        new_name = flatten_name(img, IMAGES_ALL)

        # 扁平化图片软连接
        dst_img = images_flat / new_name
        symlink_or_copy(img, dst_img)
        total_images += 1

        # 扁平化对应标签
        rel_txt = img.relative_to(IMAGES_ALL).with_suffix(".txt")
        src_lbl = LABELS_WITH_CONF / rel_txt
        dst_lbl = (labels_flat / new_name).with_suffix(".txt")
        if src_lbl.exists():
            ensure_dir(dst_lbl.parent)
            shutil.copy2(src_lbl, dst_lbl)
            total_labels += 1
        else:
            ensure_dir(dst_lbl.parent)
            dst_lbl.write_text("", encoding="utf-8")

    # 删除旧的，重命名新的
    shutil.rmtree(IMAGES_ALL)
    shutil.rmtree(LABELS_WITH_CONF)
    images_flat.rename(IMAGES_ALL)
    labels_flat.rename(LABELS_WITH_CONF)

    # 顺手把 classes.txt 放到旁边（不在 labels 目录内）
    classes_path = ROOT / "classes.txt"
    classes_path.write_text(
        "\n".join([
            "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
            "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
            "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
        ]) + "\n",
        encoding="utf-8",
    )

    print(f"[done] flattened images_all={total_images}, labels_with_conf={total_labels}")


if __name__ == "__main__":
    main()
