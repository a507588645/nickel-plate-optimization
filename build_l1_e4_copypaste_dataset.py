#!/usr/bin/env python3
"""Build development-only, fold-isolated Copy-Paste views for L1 E4.

The source view is read only. Each output fold keeps original images as hard
links, then adds a bounded number of synthetic training images. A donor and a
target always belong to that fold's train list, preventing validation leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = ROOT / "dianjienie_ryw18_oofconsensus_v1_dev_only_r1"
DEFAULT_OUTPUT_ROOT = ROOT / "l1_e4_oofconsensus_v1_cp_v1_20260823"
EXPECTED_CLASSES = (
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
)


@dataclass(frozen=True)
class Box:
    class_id: int
    x: float
    y: float
    width: float
    height: float
    line_number: int


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class Donor:
    image_name: str
    box: Box
    image_width: int
    image_height: int
    box_rect: Rect
    crop_rect: Rect


@dataclass(frozen=True)
class Placement:
    target_name: str
    target_width: int
    target_height: int
    left: int
    top: int


def parse_folds(value: str) -> tuple[int, ...]:
    try:
        folds = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--folds 必须是 1,2,3 形式的逗号分隔整数") from error
    if not folds or any(fold not in {1, 2, 3} for fold in folds):
        raise argparse.ArgumentTypeError("--folds 仅允许 1、2、3，且不能为空")
    return folds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 L1 E4 开发集内、按 Fold 隔离的稀有小目标 Copy-Paste 训练视图。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", type=parse_folds, default=(1, 2, 3))
    parser.add_argument("--data-version", default="oofconsensus_v1_cp_v1")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-class-boxes", type=int, default=100)
    parser.add_argument("--max-donor-area-ratio", type=float, default=0.0025)
    parser.add_argument("--patch-padding-px", type=int, default=16)
    parser.add_argument("--max-synthetic-per-fold", type=int, default=48)
    parser.add_argument("--max-replicas-per-donor", type=int, default=2)
    parser.add_argument("--max-per-class", type=int, default=8)
    parser.add_argument("--placement-attempts", type=int, default=300)
    parser.add_argument("--target-attempts", type=int, default=80)
    parser.add_argument("--blend-edge-px", type=int, default=6)
    parser.add_argument("--png-compress-level", type=int, choices=range(10), default=3)
    parser.add_argument("--dry-run", action="store_true", help="只审计输入并输出构建计划，不写入任何文件。")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_class_boxes < 1:
        raise ValueError("--max-class-boxes 必须为正数")
    if not 0 < args.max_donor_area_ratio <= 1:
        raise ValueError("--max-donor-area-ratio 必须在 (0, 1] 内")
    if args.patch_padding_px < 1:
        raise ValueError("--patch-padding-px 必须至少为 1")
    if args.max_synthetic_per_fold < 1 or args.max_replicas_per_donor < 1 or args.max_per_class < 1:
        raise ValueError("合成数量和复用上限必须为正数")
    if args.placement_attempts < 1 or args.target_attempts < 1:
        raise ValueError("目标和位置尝试次数必须为正数")
    if args.blend_edge_px < 0 or args.blend_edge_px >= args.patch_padding_px:
        raise ValueError("--blend-edge-px 必须不小于 0 且小于 --patch-padding-px")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def relative_image_name(entry: str) -> str:
    name = PurePosixPath(entry.strip().replace("\\", "/")).name
    if not name or Path(name).suffix.lower() != ".png":
        raise ValueError(f"划分清单包含非 PNG 图片：{entry}")
    return name


def load_classes(source_root: Path) -> list[str]:
    classes = [line.strip() for line in (source_root / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if tuple(classes) != EXPECTED_CLASSES:
        raise ValueError("源视图 classes.txt 不是固定 RYW18 顺序")
    return classes


def parse_label(path: Path, class_count: int) -> list[Box]:
    boxes: list[Box] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"标签字段数不是 5：{path}:{line_number}")
        try:
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"标签含非数值字段：{path}:{line_number}") from error
        if not 0 <= class_id < class_count:
            raise ValueError(f"标签类别越界：{path}:{line_number}")
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"标签坐标非有限：{path}:{line_number}")
        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"标签宽高非法：{path}:{line_number}")
        epsilon = 1e-9
        if x - width / 2 < -epsilon or x + width / 2 > 1 + epsilon or y - height / 2 < -epsilon or y + height / 2 > 1 + epsilon:
            raise ValueError(f"标签越过图像边界：{path}:{line_number}")
        boxes.append(Box(class_id, x, y, width, height, line_number))
    return boxes


def box_to_rect(box: Box, width: int, height: int) -> Rect:
    left = max(0, math.floor((box.x - box.width / 2) * width))
    top = max(0, math.floor((box.y - box.height / 2) * height))
    right = min(width, math.ceil((box.x + box.width / 2) * width))
    bottom = min(height, math.ceil((box.y + box.height / 2) * height))
    if right <= left or bottom <= top:
        raise ValueError(f"像素化后框为空：line={box.line_number}")
    return Rect(left, top, right, bottom)


def rects_intersect(left: Rect, right: Rect) -> bool:
    return left.left < right.right and left.right > right.left and left.top < right.bottom and left.bottom > right.top


def expand_rect(rect: Rect, padding: int, width: int, height: int) -> Rect | None:
    left, top, right, bottom = rect.left - padding, rect.top - padding, rect.right + padding, rect.bottom + padding
    if left < 0 or top < 0 or right > width or bottom > height:
        return None
    return Rect(left, top, right, bottom)


def read_split(source_root: Path, fold: int, split_name: str) -> list[str]:
    path = source_root / "splits" / "folds" / f"fold_{fold:02d}" / f"{split_name}.txt"
    names = [relative_image_name(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{path} 为空或含重复图片")
    return names


def validate_source_boundary(source_root: Path) -> dict[str, Any]:
    path = source_root / "view_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少开发集边界清单：{path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "development_only":
        raise ValueError("源视图不是 development_only，拒绝继续")
    if manifest.get("frozen_test_access") != "not read; not linked; not included":
        raise ValueError("源视图未声明冻结集隔离，拒绝继续")
    if manifest.get("counts", {}).get("images") != 1127 or manifest.get("counts", {}).get("labels") != 1127:
        raise ValueError("源视图图片或标签数量不是预期的 1127")
    return manifest


def inspect_source(source_root: Path, class_count: int) -> tuple[dict[str, tuple[int, int]], dict[str, list[Box]], dict[str, Any]]:
    images = sorted((source_root / "images").glob("*.png"), key=lambda path: path.name)
    labels = sorted((source_root / "labels").glob("*.txt"), key=lambda path: path.name)
    image_names = {path.name for path in images}
    label_names = {path.with_suffix(".png").name for path in labels}
    missing_labels, orphan_labels = image_names - label_names, label_names - image_names
    if missing_labels or orphan_labels:
        raise ValueError(f"图片/标签不配对：missing={len(missing_labels)} orphan={len(orphan_labels)}")
    if len(images) != 1127 or len(labels) != 1127:
        raise ValueError(f"源图或标签数量不正确：images={len(images)} labels={len(labels)}")
    dimensions: dict[str, tuple[int, int]] = {}
    boxes_by_image: dict[str, list[Box]] = {}
    class_counts: Counter[int] = Counter()
    image_sizes: Counter[str] = Counter()
    empty_labels = 0
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                if image.mode != "RGB":
                    raise ValueError(f"图像不是 RGB：{image_path} mode={image.mode}")
                dimensions[image_path.name] = image.size
                image_sizes[f"{image.size[0]}x{image.size[1]}"] += 1
        except Exception as error:
            raise RuntimeError(f"无法读取图像：{image_path}") from error
        boxes = parse_label(source_root / "labels" / image_path.with_suffix(".txt").name, class_count)
        boxes_by_image[image_path.name] = boxes
        if not boxes:
            empty_labels += 1
        class_counts.update(box.class_id for box in boxes)
    digest = hashlib.sha256()
    for label_path in labels:
        digest.update(label_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(label_path).encode("ascii"))
        digest.update(b"\n")
    return dimensions, boxes_by_image, {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": 0,
        "orphan_labels": 0,
        "empty_labels": empty_labels,
        "boxes": sum(class_counts.values()),
        "class_box_counts": {str(class_id): class_counts[class_id] for class_id in range(class_count)},
        "image_sizes": dict(sorted(image_sizes.items())),
        "source_label_tree_sha256": digest.hexdigest(),
    }


def validate_folds(source_root: Path, image_names: set[str], folds: Iterable[int]) -> dict[int, dict[str, list[str]]]:
    result: dict[int, dict[str, list[str]]] = {}
    for fold in folds:
        train, val = read_split(source_root, fold, "train"), read_split(source_root, fold, "val")
        if set(train) & set(val):
            raise ValueError(f"Fold {fold} train/val 有重叠")
        if set(train) | set(val) != image_names:
            raise ValueError(f"Fold {fold} train/val 并集不等于开发图片集")
        result[fold] = {"train": train, "val": val}
    return result


def make_donor_candidates(
    image_names: Iterable[str], dimensions: dict[str, tuple[int, int]], boxes_by_image: dict[str, list[Box]],
    rare_class_ids: set[int], max_area_ratio: float, padding: int,
) -> tuple[dict[int, list[Donor]], dict[str, int]]:
    candidates: dict[int, list[Donor]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    for image_name in sorted(image_names):
        image_width, image_height = dimensions[image_name]
        boxes = boxes_by_image[image_name]
        pixel_boxes = [box_to_rect(box, image_width, image_height) for box in boxes]
        for index, box in enumerate(boxes):
            if box.class_id not in rare_class_ids:
                continue
            if box.width * box.height > max_area_ratio:
                rejected["too_large"] += 1
                continue
            box_rect = pixel_boxes[index]
            crop_rect = expand_rect(box_rect, padding, image_width, image_height)
            if crop_rect is None:
                rejected["too_close_to_image_edge"] += 1
                continue
            if any(rects_intersect(crop_rect, other) for other_index, other in enumerate(pixel_boxes) if other_index != index):
                rejected["near_other_annotated_box"] += 1
                continue
            candidates[box.class_id].append(Donor(image_name, box, image_width, image_height, box_rect, crop_rect))
    for items in candidates.values():
        items.sort(key=lambda donor: (donor.image_name, donor.box.line_number))
    return dict(candidates), dict(sorted(rejected.items()))


def choose_donors(
    candidates_by_class: dict[int, list[Donor]], max_total: int, max_per_class: int,
    max_replicas_per_donor: int, rng: random.Random,
) -> list[Donor]:
    pools: dict[int, list[Donor]] = {}
    for class_id, donors in candidates_by_class.items():
        shuffled = list(donors)
        rng.shuffle(shuffled)
        pools[class_id] = shuffled
    class_ids = sorted(pools, key=lambda class_id: (len(pools[class_id]), class_id))
    selected: list[Donor] = []
    per_class: Counter[int] = Counter()
    donor_uses: Counter[tuple[str, int]] = Counter()
    cursor: dict[int, int] = defaultdict(int)
    progressed = True
    while len(selected) < max_total and progressed:
        progressed = False
        for class_id in class_ids:
            if len(selected) >= max_total:
                break
            if per_class[class_id] >= max_per_class:
                continue
            donors = pools[class_id]
            for _ in range(len(donors)):
                donor = donors[cursor[class_id] % len(donors)]
                cursor[class_id] += 1
                key = (donor.image_name, donor.box.line_number)
                if donor_uses[key] >= max_replicas_per_donor:
                    continue
                selected.append(donor)
                donor_uses[key] += 1
                per_class[class_id] += 1
                progressed = True
                break
    return selected


def find_placement(
    donor: Donor, target_names: list[str], dimensions: dict[str, tuple[int, int]], boxes_by_image: dict[str, list[Box]],
    used_targets: set[str], rng: random.Random, target_attempts: int, placement_attempts: int,
) -> Placement | None:
    eligible = [name for name in target_names if name != donor.image_name and name not in used_targets]
    rng.shuffle(eligible)
    for target_name in eligible[:target_attempts]:
        target_width, target_height = dimensions[target_name]
        if donor.crop_rect.width > target_width or donor.crop_rect.height > target_height:
            continue
        target_boxes = [box_to_rect(box, target_width, target_height) for box in boxes_by_image[target_name]]
        for _ in range(placement_attempts):
            left = rng.randint(0, target_width - donor.crop_rect.width)
            top = rng.randint(0, target_height - donor.crop_rect.height)
            patch_rect = Rect(left, top, left + donor.crop_rect.width, top + donor.crop_rect.height)
            if not any(rects_intersect(patch_rect, target_box) for target_box in target_boxes):
                return Placement(target_name, target_width, target_height, left, top)
    return None


def yolo_line(box: Box) -> str:
    return f"{box.class_id} {box.x:.8f} {box.y:.8f} {box.width:.8f} {box.height:.8f}"


def pasted_box(donor: Donor, placement: Placement) -> Box:
    width_px, height_px = donor.box_rect.width, donor.box_rect.height
    left_px = placement.left + donor.box_rect.left - donor.crop_rect.left
    top_px = placement.top + donor.box_rect.top - donor.crop_rect.top
    result = Box(
        donor.box.class_id,
        (left_px + width_px / 2) / placement.target_width,
        (top_px + height_px / 2) / placement.target_height,
        width_px / placement.target_width,
        height_px / placement.target_height,
        0,
    )
    if not (0 < result.x < 1 and 0 < result.y < 1 and 0 < result.width <= 1 and 0 < result.height <= 1):
        raise ValueError("合成框归一化后越界")
    return result


def hard_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"目标已存在，拒绝覆盖：{destination}")
    os.link(source, destination)
    if not os.path.samefile(source, destination):
        raise RuntimeError(f"硬链接验证失败：{destination}")


def blend_mask(width: int, height: int, edge: int) -> Image.Image:
    if edge == 0:
        return Image.new("L", (width, height), 255)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle((edge, edge, width - edge - 1, height - edge - 1), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=max(1, edge // 2)))


def render_synthetic(source_root: Path, output_image: Path, donor: Donor, placement: Placement, edge: int, compress_level: int) -> None:
    with Image.open(source_root / "images" / donor.image_name) as donor_image, Image.open(source_root / "images" / placement.target_name) as target_image:
        patch = donor_image.convert("RGB").crop((donor.crop_rect.left, donor.crop_rect.top, donor.crop_rect.right, donor.crop_rect.bottom))
        output = target_image.convert("RGB").copy()
        if patch.size != (donor.crop_rect.width, donor.crop_rect.height):
            raise RuntimeError("供体裁剪尺寸与计划不一致")
        output.paste(patch, (placement.left, placement.top), blend_mask(patch.width, patch.height, edge))
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output.save(output_image, format="PNG", compress_level=compress_level)


def yaml_content(published_fold_root: Path) -> str:
    lines = [f'path: "{published_fold_root.as_posix()}"', 'train: "splits/train.txt"', 'val: "splits/val.txt"', "", "nc: 18", "names:"]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(EXPECTED_CLASSES))
    return "\n".join(lines) + "\n"


def copy_fold_base(source_root: Path, fold_root: Path, image_names: Iterable[str], source_labels: dict[str, list[Box]]) -> None:
    for image_name in sorted(image_names):
        hard_link(source_root / "images" / image_name, fold_root / "images" / image_name)
        write_text(fold_root / "labels" / Path(image_name).with_suffix(".txt").name, "\n".join(yolo_line(box) for box in source_labels[image_name]) + "\n")


def build_fold(
    source_root: Path, fold_root: Path, published_fold_root: Path, fold: int, split: dict[str, list[str]],
    dimensions: dict[str, tuple[int, int]], boxes_by_image: dict[str, list[Box]], rare_class_ids: set[int],
    classes: list[str], args: argparse.Namespace,
) -> dict[str, Any]:
    train_names, val_names = split["train"], split["val"]
    candidates_by_class, donor_rejections = make_donor_candidates(
        train_names, dimensions, boxes_by_image, rare_class_ids, args.max_donor_area_ratio, args.patch_padding_px
    )
    rng = random.Random(args.seed + fold * 1009)
    donors = choose_donors(candidates_by_class, args.max_synthetic_per_fold, args.max_per_class, args.max_replicas_per_donor, rng)
    selected: list[tuple[Donor, Placement]] = []
    used_targets: set[str] = set()
    no_placement = 0
    for donor in donors:
        placement = find_placement(donor, train_names, dimensions, boxes_by_image, used_targets, rng, args.target_attempts, args.placement_attempts)
        if placement is None:
            no_placement += 1
            continue
        used_targets.add(placement.target_name)
        selected.append((donor, placement))
    candidate_counts = {classes[class_id]: len(candidates_by_class.get(class_id, [])) for class_id in sorted(rare_class_ids)}
    selected_counts = Counter(donor.box.class_id for donor, _ in selected)
    summary = {
        "fold": fold,
        "original_train_images": len(train_names),
        "original_val_images": len(val_names),
        "candidate_donors_by_class": candidate_counts,
        "selected_synthetic_by_class": {classes[class_id]: selected_counts[class_id] for class_id in sorted(rare_class_ids)},
        "selected_synthetic_total": len(selected),
        "requested_synthetic_cap": args.max_synthetic_per_fold,
        "donor_rejections": donor_rejections,
        "donors_without_safe_target_placement": no_placement,
    }
    if args.dry_run:
        return summary
    copy_fold_base(source_root, fold_root, [*train_names, *val_names], boxes_by_image)
    synthetic_names: list[str] = []
    records: list[dict[str, Any]] = []
    for index, (donor, placement) in enumerate(selected, start=1):
        synthetic_name = f"cp_f{fold:02d}_{index:04d}_{classes[donor.box.class_id]}.png"
        render_synthetic(source_root, fold_root / "images" / synthetic_name, donor, placement, args.blend_edge_px, args.png_compress_level)
        pasted = pasted_box(donor, placement)
        target_boxes = boxes_by_image[placement.target_name]
        target_rects = [box_to_rect(box, placement.target_width, placement.target_height) for box in target_boxes]
        patch_rect = Rect(placement.left, placement.top, placement.left + donor.crop_rect.width, placement.top + donor.crop_rect.height)
        pasted_rect = box_to_rect(pasted, placement.target_width, placement.target_height)
        if any(rects_intersect(patch_rect, target_rect) or rects_intersect(pasted_rect, target_rect) for target_rect in target_rects):
            raise RuntimeError("安全放置验证失败：粘贴 patch 或新增框与目标原标签相交")
        write_text(
            fold_root / "labels" / Path(synthetic_name).with_suffix(".txt").name,
            "\n".join(yolo_line(box) for box in [*target_boxes, pasted]) + "\n",
        )
        synthetic_names.append(synthetic_name)
        records.append(
            {
                "synthetic_image": f"images/{synthetic_name}",
                "synthetic_label": f"labels/{Path(synthetic_name).with_suffix('.txt').name}",
                "class_id": donor.box.class_id,
                "class_name": classes[donor.box.class_id],
                "source_image": f"images/{donor.image_name}",
                "source_label_line": donor.box.line_number,
                "source_box_yolo": asdict(donor.box),
                "source_box_pixels": asdict(donor.box_rect),
                "source_crop_pixels": asdict(donor.crop_rect),
                "target_image": f"images/{placement.target_name}",
                "target_placement_pixels": {"left": placement.left, "top": placement.top, "right": patch_rect.right, "bottom": patch_rect.bottom},
                "pasted_box_yolo": asdict(pasted),
                "target_original_box_count": len(target_boxes),
                "safe_placement": "patch and pasted box have zero intersection with every target original annotation",
                "geometry": "original_pixel_scale_no_resize",
                "blend": {"edge_px": args.blend_edge_px, "method": "feathered crop boundary; object remains inside full-alpha region"},
            }
        )
    train_entries = [*(f"images/{name}" for name in train_names), *(f"images/{name}" for name in synthetic_names)]
    val_entries = [f"images/{name}" for name in val_names]
    write_text(fold_root / "splits" / "train.txt", "\n".join(train_entries) + "\n")
    write_text(fold_root / "splits" / "val.txt", "\n".join(val_entries) + "\n")
    write_text(fold_root / "data.yaml", yaml_content(published_fold_root))
    write_text(fold_root / "classes.txt", "\n".join(classes) + "\n")
    manifest_path = fold_root / "manifests" / "synthetic_samples.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary.update({
        "train_images_after_augmentation": len(train_entries),
        "val_images_after_augmentation": len(val_entries),
        "synthetic_manifest_sha256": sha256_file(manifest_path),
    })
    write_json(fold_root / "manifests" / "fold_summary.json", summary)
    return summary


def write_source_assets(output_root: Path, source_root: Path, dimensions: dict[str, tuple[int, int]], boxes_by_image: dict[str, list[Box]]) -> None:
    path = output_root / "manifests" / "source_assets.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_image", "source_label", "width", "height", "box_count"))
        writer.writeheader()
        for name in sorted(dimensions):
            writer.writerow({
                "source_image": str(source_root / "images" / name),
                "source_label": str(source_root / "labels" / Path(name).with_suffix(".txt")),
                "width": dimensions[name][0], "height": dimensions[name][1], "box_count": len(boxes_by_image[name]),
            })


def markdown_report(source_root: Path, source_audit: dict[str, Any], rare_class_ids: set[int], classes: list[str], fold_summaries: dict[int, dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "# L1 E4 Copy-Paste 开发集派生视图", "", "## 边界", "",
        f"- 源视图：`{source_root}`，仅含 1,127 张 `development_only` 图片。",
        "- 冻结 `release-validation` 未挂载、未读取、未链接，也不出现在 train/val 清单。",
        "- 所有合成图仅用于对应 Fold 的训练清单；供体和目标均属于该 Fold 的 train。",
        "- 此视图为 `auto_only`：放大的是现有自动候选标签，不是人工真值。", "", "## 固定策略", "",
        f"- 稀有类条件：0 < 总框数 <= {args.max_class_boxes}；候选：{', '.join(classes[class_id] for class_id in sorted(rare_class_ids)) or '无'}。",
        f"- 微小框条件：归一化面积 <= {args.max_donor_area_ratio:.6f}；供体框四周固定 {args.patch_padding_px}px 上下文，禁止几何缩放。",
        f"- 每 Fold 最多 {args.max_synthetic_per_fold} 张合成图，同一供体框最多复用 {args.max_replicas_per_donor} 次，同类别最多 {args.max_per_class} 张。",
        "- 目标位置要求整个粘贴 patch 与目标原有所有标注框零相交；每张目标原图每 Fold 最多生成一个合成副本。", "", "## 源审计", "",
        f"- 图片/标签/框：{source_audit['images']}/{source_audit['labels']}/{source_audit['boxes']}；缺失标签={source_audit['missing_labels']}，孤立标签={source_audit['orphan_labels']}，空标签={source_audit['empty_labels']}。",
        f"- 标签树 SHA-256：`{source_audit['source_label_tree_sha256']}`。", "", "## Fold 摘要", "",
        "| Fold | 原 train | 原 val | 合成 train 图 | train 总数 | 合格供体 | 无安全位置 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold, summary in sorted(fold_summaries.items()):
        lines.append(
            f"| {fold} | {summary['original_train_images']} | {summary['original_val_images']} | {summary['selected_synthetic_total']} | "
            f"{summary.get('train_images_after_augmentation', summary['original_train_images'] + summary['selected_synthetic_total'])} | "
            f"{sum(summary['candidate_donors_by_class'].values())} | {summary['donors_without_safe_target_placement']} |"
        )
    lines.extend([
        "", "## 限制", "",
        "- `fawu`、`NiOH`、`youpao`、`fanxitong` 若没有符合微小框条件的真实供体，不会被人工放大或伪造；详细供体计数见各 Fold 的 `manifests/fold_summary.json`。",
        "- 目标区域只能对现有标签做到零相交，无法证明其没有未标注缺陷；这是 auto_only 标签噪声的固有限制。",
        "- 本次只准备训练输入，未训练、未评测、未选择模型或阈值。任何 E4 GPU 训练仍需独立授权及固定闸门。", "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    source_root, output_root = args.source_root.resolve(), args.output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"源视图不存在：{source_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("输出目录不能位于源视图内部或与源视图相同")
    # A dry run only reads the source view, so an existing published output must
    # not block its reproducibility audit. Real builds still refuse overwrite.
    if output_root.exists() and not args.dry_run:
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_root}")
    source_manifest = validate_source_boundary(source_root)
    classes = load_classes(source_root)
    dimensions, boxes_by_image, source_audit = inspect_source(source_root, len(classes))
    splits = validate_folds(source_root, set(dimensions), args.folds)
    global_counts = Counter(box.class_id for boxes in boxes_by_image.values() for box in boxes)
    rare_class_ids = {class_id for class_id in range(len(classes)) if 0 < global_counts[class_id] <= args.max_class_boxes}
    dry_args = argparse.Namespace(**{**vars(args), "dry_run": True})
    preflight = {
        fold: build_fold(source_root, output_root / f"fold_{fold:02d}", output_root / f"fold_{fold:02d}", fold, splits[fold], dimensions, boxes_by_image, rare_class_ids, classes, dry_args)
        for fold in args.folds
    }
    plan = {
        "stage": "E4 Copy-Paste offline preparation", "data_version": args.data_version,
        "source_root": str(source_root), "output_root": str(output_root), "development_only": True,
        "frozen_test_access": "not read; not linked; not included", "auto_only": True,
        "source_view_name": source_manifest.get("view_name"), "source_view_manifest_sha256": sha256_file(source_root / "view_manifest.json"),
        "source_audit": source_audit, "rare_class_ids": sorted(rare_class_ids),
        "rare_classes": [classes[class_id] for class_id in sorted(rare_class_ids)],
        "arguments": {key: (list(value) if isinstance(value, tuple) else value) for key, value in vars(args).items() if key not in {"source_root", "output_root"}},
        "fold_preflight": {str(fold): summary for fold, summary in sorted(preflight.items())},
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f"{output_root.name}.building_{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"发现未完成构建目录，请先人工检查：{staging_root}")
    staging_root.mkdir()
    try:
        fold_summaries = {
            fold: build_fold(source_root, staging_root / f"fold_{fold:02d}", output_root / f"fold_{fold:02d}", fold, splits[fold], dimensions, boxes_by_image, rare_class_ids, classes, args)
            for fold in args.folds
        }
        write_source_assets(staging_root, source_root, dimensions, boxes_by_image)
        write_text(staging_root / "classes.txt", "\n".join(classes) + "\n")
        final_manifest = {
            **plan, "generated_at": datetime.now().isoformat(timespec="seconds"), "output_root": str(output_root),
            "folds": {str(fold): summary for fold, summary in sorted(fold_summaries.items())},
            "image_storage": "original images are NTFS hard links; synthetic images are newly rendered PNG files",
            "synthetic_label_format": "five-column YOLO training labels; source candidate labels remain unmodified",
        }
        write_json(staging_root / "build_manifest.json", final_manifest)
        write_text(staging_root / "README.md", markdown_report(source_root, source_audit, rare_class_ids, classes, fold_summaries, args))
        staging_root.replace(output_root)
    except Exception:
        # Preserve failed staging for inspection. Source data is never deleted or altered.
        raise
    print(json.dumps({"status": "ready", "output_root": str(output_root), "fold_synthetic_counts": {str(fold): summary["selected_synthetic_total"] for fold, summary in fold_summaries.items()}, "build_manifest_sha256": sha256_file(output_root / "build_manifest.json")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
