#!/usr/bin/env python3
"""为电解镍伪标签人工审查准备「原图 vs 增强图」对照测试集。

流程：
1. 从现有电解镍伪标签数据集中随机抽取 N 张图。
2. 复制原图及其现有 labels_with_conf 到 <out>/raw。
3. 用 ImageProcessTool 对原图做增强，输出到 <out>/enhanced。
4. 用 ModelAutoLabelTool（ryw.pt）在增强图上重新生成伪标签（带置信度）。
5. 分别为 raw 和 enhanced 生成 review.html。
6. 输出 summary.csv，列出每张图在两个视图下的框数/平均置信度。

注意：ImageProcessTool 的 OpenCV 在当前环境下对含中文路径读取不稳定，
因此增强过程先在一个无中文临时目录完成，再复制回项目目录。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_TOOL_DIR = Path(r"E:/PythonProject/Tools/ImageProcessTool")
AUTO_LABEL_TOOL_DIR = Path(r"E:/PythonProject/Tools/ModelAutoLabelTool")
REVIEW_GENERATOR = Path(r"E:/PythonProject/优化/generate_flat_yolo_review.py")


@dataclass
class BoxInfo:
    class_id: int
    confidence: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare raw vs enhanced annotation test set")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            r"E:/PythonProject/优化/pseudo_label_datasets/"
            r"dianjienie_20260713_131840_v1_Dianjinie0211+classify_2types+enterprise_dianjie_0306"
        ),
        help="源电解镍伪标签数据集根目录",
    )
    parser.add_argument("--n-images", type=int, default=20, help="抽取图片数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化"),
        help="测试集输出根目录",
    )
    parser.add_argument("--conf-thres", type=float, default=0.15, help="自动标注置信度阈值")
    parser.add_argument(
        "--device", default="", help="推理设备，空字符串表示自动选择 cuda/cpu"
    )
    parser.add_argument(
        "--no-run-model",
        action="store_true",
        help="只准备原图和增强图，不运行 ryw.pt 生成增强图伪标签",
    )
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def copy_images_and_labels(
    image_paths: Sequence[Path],
    src_images_dir: Path,
    src_labels_dir: Path,
    dst_root: Path,
) -> None:
    dst_images = dst_root / "images"
    dst_labels = dst_root / "labels_with_conf"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for src_img in image_paths:
        dst_img = dst_images / src_img.name
        shutil.copy2(src_img, dst_img)
        src_label = src_labels_dir / f"{src_img.stem}.txt"
        if src_label.is_file():
            shutil.copy2(src_label, dst_labels / f"{src_img.stem}.txt")


def read_boxes(label_path: Path) -> list[BoxInfo]:
    if not label_path.is_file():
        return []
    boxes: list[BoxInfo] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) not in (5, 6):
            continue
        conf = float(parts[5]) if len(parts) == 6 else None
        boxes.append(BoxInfo(class_id=int(parts[0]), confidence=conf))
    return boxes


def run_image_process(input_dir: Path, output_dir: Path) -> None:
    cmd = [
        "uv", "run", "python", "image.py",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--recursive",
    ]
    subprocess.run(cmd, cwd=IMAGE_TOOL_DIR, check=True)


def run_auto_label(images_dir: Path, labels_dir: Path, conf_thres: float, device: str) -> None:
    cmd = [
        "uv", "run", "python", "auto_label.py",
        "--task", "dianjie",
        "--images-dir", str(images_dir),
        "--labels-dir", str(labels_dir),
        "--include-conf",
        "--write-classes",
        "--conf-thres", str(conf_thres),
    ]
    if device:
        cmd.extend(["--device", device])
    subprocess.run(cmd, cwd=AUTO_LABEL_TOOL_DIR, check=True)


def generate_review(dataset_root: Path, review_root: Path) -> Path:
    cmd = [
        sys.executable,
        str(REVIEW_GENERATOR),
        str(dataset_root),
        "--output-root", str(review_root),
    ]
    subprocess.run(cmd, check=True)
    return review_root / dataset_root.name / "review.html"


def write_summary(out_root: Path, image_names: Sequence[str]) -> None:
    raw_root = out_root / "raw"
    enhanced_root = out_root / "enhanced"
    raw_labels = raw_root / "labels_with_conf"
    enhanced_labels = enhanced_root / "labels_with_conf"

    rows = []
    for name in image_names:
        stem = Path(name).stem
        raw_boxes = read_boxes(raw_labels / f"{stem}.txt")
        enhanced_boxes = read_boxes(enhanced_labels / f"{stem}.txt")
        rows.append(
            {
                "image_name": name,
                "raw_boxes": len(raw_boxes),
                "raw_avg_conf": (
                    round(sum(b.confidence for b in raw_boxes if b.confidence is not None) / len(raw_boxes), 4)
                    if raw_boxes else ""
                ),
                "enhanced_boxes": len(enhanced_boxes),
                "enhanced_avg_conf": (
                    round(sum(b.confidence for b in enhanced_boxes if b.confidence is not None) / len(enhanced_boxes), 4)
                    if enhanced_boxes else ""
                ),
            }
        )

    csv_path = out_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "raw_boxes",
                "raw_avg_conf",
                "enhanced_boxes",
                "enhanced_avg_conf",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_root / "summary.json"
    json_path.write_text(
        json.dumps(
            {
                "n_images": len(image_names),
                "raw_total_boxes": sum(r["raw_boxes"] for r in rows),
                "enhanced_total_boxes": sum(r["enhanced_boxes"] for r in rows),
                "per_image": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels_with_conf"
    classes_file = dataset_root / "classes.txt"

    if not images_dir.is_dir():
        raise NotADirectoryError(f"images dir not found: {images_dir}")

    all_images = list_images(images_dir)
    if len(all_images) < args.n_images:
        raise ValueError(f"数据集只有 {len(all_images)} 张图，少于请求数量 {args.n_images}")

    random.seed(args.seed)
    sampled = sorted(random.sample(all_images, args.n_images))
    image_names = [p.name for p in sampled]

    timestamp = subprocess.check_output(["date", "+%Y%m%d_%H%M%S"], text=True).strip()
    out_root = args.output_root / f"test_enhance_annotation_dianjie_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    raw_root = out_root / "raw"
    enhanced_root = out_root / "enhanced"

    print(f"[1/5] 复制 {len(sampled)} 张原图及现有伪标签到 {raw_root}")
    copy_images_and_labels(sampled, images_dir, labels_dir, raw_root)
    if classes_file.is_file():
        shutil.copy2(classes_file, raw_root / "classes.txt")

    print(f"[2/5] 用 ImageProcessTool 生成增强图")
    with tempfile.TemporaryDirectory(prefix="proc_", dir=r"E:/PythonProject") as tmpdir:
        tmp_path = Path(tmpdir)
        tmp_raw = tmp_path / "raw"
        tmp_raw.mkdir()
        for img in sampled:
            shutil.copy2(img, tmp_raw / img.name)

        tmp_enhanced = tmp_path / "enhanced"
        run_image_process(tmp_raw, tmp_enhanced)

        print(f"[3/5] 复制增强图到 {enhanced_root}")
        enhanced_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_enhanced, enhanced_root / "images", dirs_exist_ok=True)

        if not args.no_run_model:
            print(f"[4/5] 用 ryw.pt 在增强图上生成伪标签（conf_thres={args.conf_thres}）")
            tmp_labels = tmp_path / "labels_with_conf"
            run_auto_label(tmp_enhanced, tmp_labels, args.conf_thres, args.device)
            shutil.copytree(tmp_labels, enhanced_root / "labels_with_conf", dirs_exist_ok=True)
            (enhanced_root / "classes.txt").write_text(
                (tmp_labels / "classes.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        else:
            enhanced_root.mkdir(parents=True, exist_ok=True)

    print(f"[5/5] 生成 review.html")
    raw_review = generate_review(raw_root, out_root / "reviews")
    if not args.no_run_model:
        enhanced_review = generate_review(enhanced_root, out_root / "reviews")
    else:
        enhanced_review = None

    if not args.no_run_model:
        write_summary(out_root, image_names)

    print("\n完成。产物：")
    print(f"  测试集根目录：{out_root}")
    print(f"  原图审查页：   {raw_review}")
    if enhanced_review:
        print(f"  增强图审查页： {enhanced_review}")
    if not args.no_run_model:
        print(f"  统计摘要：     {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
