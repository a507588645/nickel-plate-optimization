"""
抽查可视化 pseudo-label v1 的质量。
从 accepted / uncertain / rejected 中各抽若干张图，绘制预测框、类别和置信度。
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# 给每个类别固定一个颜色 (BGR)
CLASS_COLORS = {
    "wuzi": (50, 120, 240),
    "waterstain": (240, 180, 50),
    "oilstain": (180, 50, 180),
    "nodule": (50, 180, 50),
    "airhole": (240, 50, 50),
    "bubble": (50, 50, 240),
    "bump": (180, 180, 50),
    "fawu": (50, 180, 180),
    "overfire": (180, 50, 50),
    "NiOH": (240, 50, 240),
    "greenstain": (50, 240, 50),
    "suanwuran": (120, 50, 240),
    "quejiao": (240, 120, 50),
    "watermark": (100, 100, 100),
}


def load_manifest(manifest_path: Path):
    """读取 jsonl，按 image_stem 聚合所有框。"""
    groups = defaultdict(list)
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            groups[rec["image_stem"]].append(rec)
    return groups


def pick_samples(groups: dict, n: int = 6, seed: int = 42):
    """
    抽样策略：优先选择框数适中（1~5 个框）且包含不同类别的图。
    返回 [(image_stem, boxes), ...]
    """
    random.seed(seed)
    candidates = []
    for stem, boxes in groups.items():
        n_boxes = len(boxes)
        if 1 <= n_boxes <= 5:
            classes = {b["class_name"] for b in boxes}
            candidates.append((stem, boxes, len(classes)))

    # 按包含类别数降序，再按框数升序，优先多样且干净
    candidates.sort(key=lambda x: (-x[2], len(x[1]), x[0]))

    selected = []
    seen_classes = set()
    # 第一轮：尽量覆盖不同类别组合
    for stem, boxes, _ in candidates:
        classes = {b["class_name"] for b in boxes}
        if classes - seen_classes:
            selected.append((stem, boxes))
            seen_classes |= classes
            if len(selected) >= n:
                break

    # 如果不够，随机补
    if len(selected) < n:
        remaining = [c for c in candidates if c[0] not in {s[0] for s in selected}]
        random.shuffle(remaining)
        for stem, boxes, _ in remaining:
            selected.append((stem, boxes))
            if len(selected) >= n:
                break

    return selected[:n]


def draw_boxes(image: np.ndarray, boxes: list):
    """在图上绘制 bbox、类别和置信度。"""
    h, w = image.shape[:2]
    for box in boxes:
        cls = box["class_name"]
        conf = box["confidence"]
        color = CLASS_COLORS.get(cls, (200, 200, 200))

        x_c, y_c, bw, bh = box["x_center"], box["y_center"], box["width"], box["height"]
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        # 防止越界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, max(1, int(min(h, w) / 300)))

        label = f"{cls}:{conf:.2f}"
        font_scale = max(0.4, min(h, w) / 1200)
        thickness = max(1, int(min(h, w) / 500))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return image


def find_image(image_path_str: str):
    """manifest 中的路径可能指向不存在的位置，尝试多个候选。"""
    candidates = [Path(image_path_str)]
    p = Path(image_path_str)
    # 同名 png
    candidates.append(p.with_suffix(".png"))
    # 切换到 images/train 或 images/val
    parts = p.parts
    if "images" in parts:
        idx = parts.index("images")
        base = Path(*parts[:idx])
        stem = p.stem
        for split in ["train", "val", "valid"]:
            candidates.append(base / "images" / split / f"{stem}.png")
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def visualize_dataset(
    dataset: str,
    manifest_root: Path,
    out_dir: Path,
    samples_per_status: int = 6,
    seed: int = 42,
):
    """对一个数据集的三个状态分别抽样并可视化。"""
    status_order = ["accepted", "uncertain", "rejected"]
    for status in status_order:
        manifest = manifest_root / dataset / "manifests" / f"{status}.jsonl"
        if not manifest.exists():
            print(f"[跳过] 未找到 {manifest}")
            continue

        groups = load_manifest(manifest)
        samples = pick_samples(groups, n=samples_per_status, seed=seed)
        print(f"[{dataset}/{status}] 抽到 {len(samples)} 张图用于可视化")

        for idx, (stem, boxes) in enumerate(samples, 1):
            image_path = find_image(boxes[0]["image_path"])
            if image_path is None:
                print(f"  [警告] 找不到图片: {stem}")
                continue

            img = cv2.imread(image_path)
            if img is None:
                print(f"  [警告] 读取失败: {image_path}")
                continue

            img = draw_boxes(img, boxes)

            out_name = f"{dataset}_{status}_{idx:02d}_{stem}.jpg"
            out_path = out_dir / out_name
            # OpenCV 在部分 Windows 环境写入可能失败，使用 PIL 保存
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(out_path, quality=95)
            print(f"  已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="可视化 pseudo-label v1 质量抽查")
    parser.add_argument("--datasets", nargs="+", default=["Dianjinie0211", "dianjienie"], help="数据集名称")
    parser.add_argument("--manifest-root", type=Path, default=Path("E:/PythonProject/datasets"), help="manifest 根目录")
    parser.add_argument("--out-dir", type=Path, default=Path("E:/PythonProject/优化/pseudo_v1_visualization"), help="输出目录")
    parser.add_argument("--samples", type=int, default=6, help="每个状态抽多少张")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--clean", action="store_true", help="清空输出目录")
    args = parser.parse_args()

    out_dir = args.out_dir
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        visualize_dataset(ds, args.manifest_root, out_dir, args.samples, args.seed)

    print(f"\n全部完成，输出目录: {out_dir}")


if __name__ == "__main__":
    main()
