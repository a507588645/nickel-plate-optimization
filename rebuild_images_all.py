#!/usr/bin/env python3
"""重建 combined_all_pseudo_v1/images_all：为 labels_with_conf 中的每张图找到源数据集真实图片并硬链接/复制。

命名规则与 build_combined_pseudo_dataset.py 一致：
  {dataset_name}_{relative_path.replace('/', '_').replace('\\', '_')}
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重建 images_all")
    parser.add_argument("--dataset-root", type=Path, default=Path(r"E:/PythonProject/datasets/combined_all_pseudo_v1"))
    parser.add_argument("--source-roots", type=Path, nargs="+", default=[
        Path(r"E:/PythonProject/datasets/Dianjinie0211"),
        Path(r"E:/PythonProject/datasets/dianjienie"),
        Path(r"E:/PythonProject/datasets/classify_2types"),
        Path(r"E:/PythonProject/datasets/classify_4types_dianji"),
        Path(r"E:/PythonProject/datasets/enterprise_dianji_0309_classification"),
        Path(r"E:/PythonProject/datasets/enterprise_dianji_0309_detection_unlabeled"),
        Path(r"E:/PythonProject/datasets/enterprise_dianjie_0306_detection_unlabeled"),
    ])
    parser.add_argument("--copy", action="store_true", help="强制复制而非硬链接")
    return parser.parse_args()


def collect_flat_mapping(source_roots: list[Path]) -> dict[str, Path]:
    """按 build_combined_pseudo_dataset.py 的命名规则，建立扁平名到源图片的映射。"""
    mapping: dict[str, Path] = {}
    for root in source_roots:
        if not root.exists():
            continue
        ds_name = root.name
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                rel = path.relative_to(root)
                flat_name = f"{ds_name}_" + str(rel).replace("\\", "_").replace("/", "_")
                mapping[flat_name] = path
    return mapping


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    labels_dir = dataset_root / "labels_with_conf"
    images_all = dataset_root / "images_all"

    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_with_conf 不存在: {labels_dir}")

    print("扫描源数据集图片并按扁平命名规则建立映射...")
    source_map = collect_flat_mapping(args.source_roots)
    print(f"源数据集扁平名映射数: {len(source_map)}")

    label_paths = sorted(labels_dir.glob("*.txt"))
    label_paths = [p for p in label_paths if p.stem != "classes"]
    print(f"labels_with_conf 标签数: {len(label_paths)}")

    images_all.mkdir(parents=True, exist_ok=True)

    matched = 0
    missing = []
    for label_path in label_paths:
        stem = label_path.stem
        # 汇总图名就是 label stem + .png
        image_name = stem + ".png"
        src = source_map.get(image_name)
        if src is None:
            missing.append(stem)
            continue
        dst = images_all / image_name
        link_or_copy(src, dst, args.copy)
        matched += 1

    print(f"成功链接/复制: {matched}/{len(label_paths)}")
    if missing:
        print(f"未找到源图片: {len(missing)} 张")
        missing_path = dataset_root / "images_all_missing.txt"
        missing_path.write_text("\n".join(missing), encoding="utf-8")
        print(f"清单保存至: {missing_path}")


if __name__ == "__main__":
    main()
