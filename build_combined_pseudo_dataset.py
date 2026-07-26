#!/usr/bin/env python3
"""
汇总 datasets 下除 nietiao 外的所有数据集，使用 ryw.pt 生成伪标签，
过滤低置信度框（仅保留 conf >= 0.70），并划分为 train/val/test。

输出目录：E:\PythonProject\datasets\combined_all_pseudo_v1
结构：
  images_all/          全部原图软连接（临时，划分后移动到 images/train|val|test）
  images/              划分后的图片软连接
  labels_with_conf/    带第六列置信度的伪标签主版本
  labels/              划分后的五列 YOLO 标签（训练用），与 images/ 对应
  splits/              划分清单与统计报告
  data.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


DATASETS_ROOT = Path(r"E:\PythonProject\datasets")
OPT_ROOT = Path(r"E:\PythonProject\优化")
AUTO_LABEL = Path(r"E:\PythonProject\Tools\ModelAutoLabelTool\auto_label.py")
DEFAULT_PYTHON_EXE = Path(r"E:\PythonProject\models\.venv\Scripts\python.exe")

# 排除的数据集
EXCLUDE_DATASETS = {"nietiao"}

IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}

NAMES = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_images(root: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def create_symlink_or_copy(src: Path, dst: Path, prefer_symlink: bool = True) -> None:
    """优先创建文件软连接；失败则复制文件。"""
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if prefer_symlink:
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def collect_images(
    out_root: Path,
    datasets: Sequence[str] | None = None,
    prefer_symlink: bool = True,
) -> dict[str, list[Path]]:
    """收集指定数据集的图片，在 images_all/ 下扁平化创建软连接，返回每个数据集的图片列表。"""
    images_all = ensure_dir(out_root / "images_all")
    per_dataset: dict[str, list[Path]] = defaultdict(list)

    candidates = sorted(d.name for d in DATASETS_ROOT.iterdir() if d.is_dir())
    for ds_name in candidates:
        if ds_name in EXCLUDE_DATASETS:
            continue
        if datasets is not None and ds_name not in datasets:
            continue
        ds_path = DATASETS_ROOT / ds_name
        # 避免把输出目录自身当作源数据集
        try:
            ds_path.relative_to(out_root)
            print(f"[skip] {ds_name}: inside output directory")
            continue
        except ValueError:
            pass
        imgs = iter_images(ds_path, recursive=True)
        if not imgs:
            print(f"[warn] {ds_name}: no images found")
            continue
        for img in imgs:
            rel = img.relative_to(ds_path)
            # 扁平化文件名：数据集名_原相对路径（分隔符替换为_）
            flat_name = f"{ds_name}_" + str(rel).replace("\\", "_").replace("/", "_")
            link_path = images_all / flat_name
            create_symlink_or_copy(img, link_path, prefer_symlink=prefer_symlink)
            per_dataset[ds_name].append(link_path)
        print(f"[collect] {ds_name}: {len(imgs)} images")

    return dict(per_dataset)


def run_auto_label(
    out_root: Path,
    python_exe: Path,
    conf_thres: float = 0.15,
    img_size: int = 640,
    iou_thres: float = 0.45,
    device: str = "cpu",
) -> None:
    """调用 ModelAutoLabelTool 生成带置信度的伪标签。"""
    images_dir = out_root / "images_all"
    labels_dir = ensure_dir(out_root / "labels_with_conf")

    cmd = [
        str(python_exe),
        str(AUTO_LABEL),
        "--task", "dianjie",
        "--images-dir", str(images_dir),
        "--labels-dir", str(labels_dir),
        "--conf-thres", str(conf_thres),
        "--img-size", str(img_size),
        "--iou-thres", str(iou_thres),
        "--device", device,
        "--include-conf",
        "--recursive",
        "--write-classes",
        "--overwrite",
    ]
    print("[auto_label] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def split_dataset(
    out_root: Path,
    conf_keep: float = 0.70,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path], dict[int, int]]:
    """按图片随机划分为 train/val/test，移动软连接并复制过滤后的标签。"""
    images_all = out_root / "images_all"
    labels_with_conf = out_root / "labels_with_conf"
    images_root = ensure_dir(out_root / "images")
    labels_root = ensure_dir(out_root / "labels")

    # 清理旧划分
    for split in ("train", "val", "test"):
        shutil.rmtree(images_root / split, ignore_errors=True)
        shutil.rmtree(labels_root / split, ignore_errors=True)

    all_images = sorted(iter_images(images_all, recursive=True))
    rng = random.Random(seed)
    rng.shuffle(all_images)

    n = len(all_images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_imgs = all_images[:n_train]
    val_imgs = all_images[n_train:n_train + n_val]
    test_imgs = all_images[n_train + n_val:]

    class_counts = Counter()
    total_boxes = 0
    kept_boxes = 0

    def move_to_split(split: str, imgs: list[Path]) -> None:
        nonlocal total_boxes, kept_boxes
        split_img_dir = ensure_dir(images_root / split)
        split_lbl_dir = ensure_dir(labels_root / split)
        for img in imgs:
            rel = img.relative_to(images_all)
            dst_img = split_img_dir / rel
            create_symlink_or_copy(img, dst_img, prefer_symlink=True)

            rel_txt = rel.with_suffix(".txt")
            src_lbl = labels_with_conf / rel_txt
            dst_lbl = split_lbl_dir / rel_txt
            ensure_dir(dst_lbl.parent)

            if not src_lbl.exists():
                # 无伪标签文件时写入空标签，明确这是无检出样本
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
                    conf = float(parts[5])
                except ValueError:
                    continue
                if conf >= conf_keep:
                    kept_lines.append(" ".join(parts[:5]))
                    class_counts[int(parts[0])] += 1
                    kept_boxes += 1

            dst_lbl.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")

    move_to_split("train", train_imgs)
    move_to_split("val", val_imgs)
    move_to_split("test", test_imgs)

    print(f"[split] train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    print(f"[filter] total_boxes={total_boxes}, kept_boxes={kept_boxes}")
    return train_imgs, val_imgs, test_imgs, dict(class_counts)


def write_data_yaml(out_root: Path) -> None:
    yaml_path = out_root / "data.yaml"
    content = (
        f"path: {out_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(NAMES)}\n"
        f"names: {NAMES}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[yaml] wrote {yaml_path}")


def report_distribution(
    out_root: Path,
    split_imgs: dict[str, list[Path]],
    class_counts: dict[str, int],
) -> None:
    """生成划分与类别分布报告。"""
    report: dict = {
        "dataset": str(out_root),
        "split_counts": {k: len(v) for k, v in split_imgs.items()},
        "per_dataset_counts": {},
        "class_counts_overall": class_counts,
        "class_counts_per_split": {},
    }

    for split, imgs in split_imgs.items():
        ds_counter: dict[str, int] = defaultdict(int)
        cls_counter = Counter()
        for img in imgs:
            rel = img.relative_to(out_root / "images_all")
            ds_name = rel.parts[0]
            ds_counter[ds_name] += 1
            lbl_path = out_root / "labels" / split / rel.with_suffix(".txt")
            if lbl_path.exists():
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        cls_counter[int(line.split()[0])] += 1
        report["per_dataset_counts"][split] = dict(ds_counter)
        report["class_counts_per_split"][split] = dict(cls_counter)

    report_path = ensure_dir(out_root / "splits") / "distribution.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[report] wrote {report_path}")

    # 同时输出文本摘要
    summary_path = report_path.with_suffix(".txt")
    lines = ["Combined pseudo-label dataset summary", "=" * 40, ""]
    for split, count in report["split_counts"].items():
        lines.append(f"{split}: {count} images")
    lines.append("")
    lines.append("Class distribution (overall):")
    for cid in sorted(report["class_counts_overall"]):
        lines.append(f"  {cid:2d} {NAMES[cid]:15s}: {report['class_counts_overall'][cid]}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined pseudo-label YOLO dataset")
    parser.add_argument("--out-dir", default=str(DATASETS_ROOT / "combined_all_pseudo_v1"),
                        help="Output dataset root")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific dataset names to include; default all except nietiao")
    parser.add_argument("--conf-keep", type=float, default=0.70,
                        help="Confidence threshold to keep pseudo boxes")
    parser.add_argument("--conf-min", type=float, default=0.15,
                        help="Minimum confidence for pseudo-label generation")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Inference image size for ryw.pt")
    parser.add_argument("--iou-thres", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0", help="Device for inference, e.g. cpu, 0, cuda:0")
    parser.add_argument("--python-exe", default=str(DEFAULT_PYTHON_EXE),
                        help="Python executable to run ModelAutoLabelTool")
    parser.add_argument("--skip-auto-label", action="store_true",
                        help="Skip auto-label generation (reuse existing labels_with_conf)")
    parser.add_argument("--skip-symlink", action="store_true",
                        help="Skip image collection (reuse existing images_all)")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    python_exe = Path(args.python_exe)
    ensure_dir(out_root)

    print(f"[out] {out_root}")
    print(f"[python] {python_exe}")
    print(f"[config] keep_conf>={args.conf_keep}, gen_conf_min={args.conf_min}, "
          f"img_size={args.img_size}, split={args.train_ratio}/{args.val_ratio}/"
          f"{1 - args.train_ratio - args.val_ratio:.2f}, seed={args.seed}")

    # 1. 收集图片软连接
    if not args.skip_symlink:
        per_dataset = collect_images(out_root, datasets=args.datasets)
        total = sum(len(v) for v in per_dataset.values())
        print(f"[collect] total {total} images")
    else:
        print("[collect] skipped")

    # 2. 生成伪标签
    if not args.skip_auto_label:
        run_auto_label(
            out_root,
            python_exe=python_exe,
            conf_thres=args.conf_min,
            img_size=args.img_size,
            iou_thres=args.iou_thres,
            device=args.device,
        )
    else:
        print("[auto_label] skipped")

    # 3. 划分并过滤低置信度
    train_imgs, val_imgs, test_imgs, class_counts = split_dataset(
        out_root,
        conf_keep=args.conf_keep,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 5. 写 data.yaml
    write_data_yaml(out_root)

    # 6. 报告
    split_imgs = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs,
    }
    report_distribution(out_root, split_imgs, class_counts)

    print("[done]")


if __name__ == "__main__":
    main()
