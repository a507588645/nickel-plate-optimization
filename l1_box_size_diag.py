"""L1 anchor box-size diagnostics.

Default scope is the development set only: the union of fold_01/02/03 val.txt.
The frozen test split is not opened unless --include-test is explicitly passed.

Outputs:
  l1_box_size_diag/box_size_report.json
  l1_box_size_diag/box_size_report.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ANCHOR_ROOT = Path(r"E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1")
DEFAULT_OUT_DIR = THIS_DIR / "l1_box_size_diag"
DEFAULT_IMGSZ = 640
SMALL_PX = 32
TINY_PX = 16
SMALL_AREA_PX = SMALL_PX * SMALL_PX
FALLBACK_MEDIAN_WH = (4096, 3580)

# Classes with many boxes and/or poor AP in the current L1 discussions.
HIGH_INTEREST = {
    "airhole",
    "bubble",
    "bump",
    "greenstain",
    "overfire",
    "watermark",
    "waterstain",
    "wuzi",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="L1 RYW18 anchor box-size diagnostics. Default is dev-only and does not open frozen test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument(
        "--sample-img",
        type=int,
        default=200,
        help="Number of selected images whose headers are sampled for resolution; 0 means do not read image headers.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly include the frozen test split. Omit for decision-safe dev-only diagnostics.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print diagnostics without writing report files.")
    return parser.parse_args(argv)


def read_classes(anchor_root: Path) -> list[str]:
    classes_file = anchor_root / "classes.txt"
    if not classes_file.is_file():
        raise FileNotFoundError(f"Missing classes file: {classes_file}")
    names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Empty classes file: {classes_file}")
    return names


def split_line_to_name(line: str) -> str:
    return PurePosixPath(line.strip().replace("\\", "/")).name


def read_split_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing split list: {path}")
    names = [split_line_to_name(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Empty split list: {path}")
    return names


def ordered_unique(items: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    unique: list[str] = []
    duplicates = 0
    for item in items:
        if item in seen:
            duplicates += 1
            continue
        seen.add(item)
        unique.append(item)
    return unique, duplicates


def resolve_image_paths(anchor_root: Path, names: list[str]) -> list[Path]:
    images_dir = anchor_root / "images"
    paths: list[Path] = []
    missing: list[str] = []
    for name in names:
        path = images_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        paths.append(path)
    if missing:
        raise FileNotFoundError(f"Split image missing from images/: {missing[:3]}")
    return paths


def load_selected_images(anchor_root: Path, include_test: bool) -> tuple[list[Path], dict[str, Any]]:
    folds_root = anchor_root / "splits" / "folds"
    fold_rows: list[dict[str, Any]] = []
    dev_names: list[str] = []
    for fold in (1, 2, 3):
        val_list = folds_root / f"fold_{fold:02d}" / "val.txt"
        names = read_split_names(val_list)
        fold_rows.append({"fold": fold, "val_list": str(val_list), "val_images": len(names)})
        dev_names.extend(names)

    dev_unique, dev_duplicates = ordered_unique(dev_names)
    selected_names = list(dev_unique)
    scope = "development"
    frozen_test_access = "not read; not included"
    test_count = None
    test_duplicates = None
    test_list = None
    if include_test:
        test_path = anchor_root / "splits" / "test.txt"
        test_names = read_split_names(test_path)
        test_unique, test_duplicates = ordered_unique(test_names)
        selected_names = dev_unique + [name for name in test_unique if name not in set(dev_unique)]
        scope = "development_plus_frozen_test"
        frozen_test_access = "read and included because --include-test was passed"
        test_count = len(test_unique)
        test_list = str(test_path)

    selected_unique, selected_duplicates = ordered_unique(selected_names)
    paths = resolve_image_paths(anchor_root, selected_unique)
    split_info = {
        "scope": scope,
        "frozen_test_access": frozen_test_access,
        "fold_val_source": "union of fold_01/fold_02/fold_03 val.txt",
        "fold_val_lists": fold_rows,
        "development_images": len(dev_unique),
        "development_duplicate_names_dropped": dev_duplicates,
        "frozen_test_list": test_list,
        "frozen_test_images": test_count,
        "frozen_test_duplicate_names_dropped": test_duplicates,
        "selected_images": len(paths),
        "selected_duplicate_names_dropped": selected_duplicates,
    }
    return paths, split_info


def parse_label(path: Path, nc: int) -> tuple[list[tuple[int, float, float]], Counter[str]]:
    rows: list[tuple[int, float, float]] = []
    problems: Counter[str] = Counter()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            problems["bad_field_count"] += 1
            continue
        try:
            class_id = int(parts[0])
            x, y, width, height = (float(value) for value in parts[1:])
        except ValueError:
            problems["bad_numeric_value"] += 1
            continue
        if not 0 <= class_id < nc:
            problems["bad_class_id"] += 1
            continue
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            problems["non_finite_value"] += 1
            continue
        if width <= 0 or height <= 0:
            problems["non_positive_size"] += 1
            continue
        if width > 1 or height > 1:
            problems["size_gt_1"] += 1
            continue
        rows.append((class_id, width, height))
    return rows, problems


def human_pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if value < 0.01:
        return f"{value * 100:.3f}%"
    return f"{value * 100:.2f}%"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def measure_image_resolutions(image_paths: list[Path], sample_n: int) -> tuple[dict[str, tuple[int, int]], list[str]]:
    if sample_n < 0:
        raise ValueError("--sample-img must be >= 0")
    if sample_n == 0:
        return {}, ["sample-img=0: image headers were not read; fallback median resolution is used."]
    if not image_paths:
        return {}, ["No selected images."]
    try:
        from PIL import Image
    except Exception as error:  # pragma: no cover
        return {}, [f"PIL unavailable: {error}; fallback median resolution is used."]

    if sample_n >= len(image_paths):
        chosen = image_paths
    else:
        step = max(1, len(image_paths) // sample_n)
        chosen = image_paths[::step][:sample_n]

    resolutions: dict[str, tuple[int, int]] = {}
    warnings: list[str] = []
    for path in chosen:
        try:
            with Image.open(path) as image:
                resolutions[path.name] = (image.width, image.height)
        except Exception as error:
            warnings.append(f"Failed to read {path.name}: {error}")
    return resolutions, warnings


def summarize_contexts(median_w: int, median_h: int, imgsz: int) -> dict[str, dict[str, Any]]:
    letterbox_scale = imgsz / max(median_w, median_h)
    stretch_w_scale = imgsz / median_w
    stretch_h_scale = imgsz / median_h
    resized_w = round(median_w * letterbox_scale)
    resized_h = round(median_h * letterbox_scale)
    return {
        "val_letterbox": {
            "description": "Ultralytics validation/inference letterbox estimate: preserve aspect ratio and fit long side to imgsz.",
            "width_scale": letterbox_scale,
            "height_scale": letterbox_scale,
            "resized_w": resized_w,
            "resized_h": resized_h,
            "pad_w": imgsz - resized_w,
            "pad_h": imgsz - resized_h,
        },
        "train_square_stretch_reference": {
            "description": "Square-resize reference for non-rect train input before augmentation effects; actual mosaic/affine can change apparent sizes.",
            "width_scale": stretch_w_scale,
            "height_scale": stretch_h_scale,
            "resized_w": imgsz,
            "resized_h": imgsz,
            "pad_w": 0,
            "pad_h": 0,
        },
    }


def context_metrics(
    ws: list[float],
    hs: list[float],
    median_w: int,
    median_h: int,
    width_scale: float,
    height_scale: float,
) -> dict[str, Any]:
    width_px = [w * median_w * width_scale for w in ws]
    height_px = [h * median_h * height_scale for h in hs]
    min_side = [min(w, h) for w, h in zip(width_px, height_px)]
    area_px = [w * h for w, h in zip(width_px, height_px)]
    n = len(ws)
    return {
        "w_px_median": statistics.median(width_px),
        "h_px_median": statistics.median(height_px),
        "min_side_px_p10": percentile(min_side, 0.10),
        "min_side_px_median": statistics.median(min_side),
        "min_side_px_p90": percentile(min_side, 0.90),
        "pct_min_side_lt_32": sum(value < SMALL_PX for value in min_side) / n,
        "pct_min_side_lt_16": sum(value < TINY_PX for value in min_side) / n,
        "pct_width_lt_32": sum(value < SMALL_PX for value in width_px) / n,
        "pct_height_lt_32": sum(value < SMALL_PX for value in height_px) / n,
        "pct_area_lt_1024": sum(value < SMALL_AREA_PX for value in area_px) / n,
        "count_min_side_lt_32": sum(value < SMALL_PX for value in min_side),
        "count_min_side_lt_16": sum(value < TINY_PX for value in min_side),
    }


def build_markdown(report: dict[str, Any]) -> str:
    split_info = report["split"]
    image_resolution = report["image_resolution"]
    contexts = report["contexts"]
    per_class = report["per_class"]
    primary = "val_letterbox"
    context = contexts[primary]

    md = [
        "# L1 锚点框尺寸诊断更正版",
        "",
        "## 边界",
        "",
        f"- 锚点：`{report['anchor_root']}`",
        f"- 范围：`{split_info['scope']}`；选中图片 {split_info['selected_images']} 张；标签文件 {report['selected_label_files']} 个；总框 {report['total_boxes']} 个。",
        f"- 冻结 test：{split_info['frozen_test_access']}。",
        f"- 开发集来源：{split_info['fold_val_source']}；开发集唯一图片 {split_info['development_images']} 张。",
        f"- 原图分辨率：中位 {image_resolution['median_w']}x{image_resolution['median_h']}；抽样 {image_resolution['sampled']} 张；唯一尺寸 {image_resolution['unique_shapes'][:5]}。",
        f"- 主口径：`{primary}`，缩放系数 {context['width_scale']:.6f}，等效无 padding 区域 {context['resized_w']}x{context['resized_h']}，padding {context['pad_w']}x{context['pad_h']}。",
        "",
    ]
    if report["label_problems"]:
        md += ["## 标签问题", ""]
        for key, value in sorted(report["label_problems"].items()):
            md.append(f"- {key}: {value}")
        md.append("")

    def table_for(context_name: str, title: str) -> list[str]:
        rows = [
            f"## {title}",
            "",
            "| 类别 | 框数 | 中位最小边(px) | <32px占比 | <16px占比 | 中位宽(px) | 中位高(px) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for item in per_class:
            if item["status"] != "evaluated":
                rows.append(f"| {item['class_name']} | {item['count']} | n/a | n/a | n/a | n/a | n/a |")
                continue
            metrics = item["contexts"][context_name]
            rows.append(
                f"| {item['class_name']} | {item['count']} | {metrics['min_side_px_median']:.1f} | "
                f"{human_pct(metrics['pct_min_side_lt_32'])} | {human_pct(metrics['pct_min_side_lt_16'])} | "
                f"{metrics['w_px_median']:.1f} | {metrics['h_px_median']:.1f} |"
            )
        rows.append("")
        return rows

    md += table_for("val_letterbox", "每类摘要：val_letterbox 口径")
    md += table_for("train_square_stretch_reference", "每类摘要：train_square_stretch_reference 口径")

    md += ["## 高兴趣类结论（主口径 val_letterbox）", ""]
    for line in report["high_interest_verdict"]:
        md.append(line)
    md += [
        "",
        "## 说明",
        "",
        "- 本报告默认不读取、不统计冻结 test；只有显式 `--include-test` 才会改变该边界。",
        "- `val_letterbox` 口径按 Ultralytics 验证/推理的长边适配估计，适合判断 mAP 验证时目标等效尺寸。",
        "- `train_square_stretch_reference` 是非 rect 训练输入的方形 resize 参考；实际 mosaic/affine 会继续改变训练期可见尺寸。",
        "- 小目标尺寸与低 mAP 目前是相关性证据，高分辨率是否提升仍需单变量训练验证。",
    ]
    if report["resolution_warnings"]:
        md += ["", "## 警告", ""]
        for warning in report["resolution_warnings"]:
            md.append(f"- {warning}")
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    anchor_root = args.anchor_root.resolve()
    output_dir = args.output_dir.resolve()
    images_dir = anchor_root / "images"
    labels_dir = anchor_root / "labels"
    if not anchor_root.is_dir():
        print(f"[ERR] Anchor root does not exist: {anchor_root}", file=sys.stderr)
        return 2
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"[ERR] Anchor images/labels directory missing under: {anchor_root}", file=sys.stderr)
        return 2
    if args.imgsz <= 0:
        print("[ERR] --imgsz must be positive.", file=sys.stderr)
        return 2
    if args.sample_img < 0:
        print("[ERR] --sample-img must be >= 0.", file=sys.stderr)
        return 2

    classes = read_classes(anchor_root)
    selected_images, split_info = load_selected_images(anchor_root, args.include_test)
    selected_stems = {path.stem for path in selected_images}
    print(f"[info] classes={len(classes)}; scope={split_info['scope']}; selected_images={len(selected_images)}")
    print(f"[info] frozen_test_access={split_info['frozen_test_access']}")

    per_class_box: dict[int, list[tuple[float, float]]] = defaultdict(list)
    total_boxes = 0
    empty_files = 0
    missing_label_files: list[str] = []
    label_problems: Counter[str] = Counter()
    class_id_seen: set[int] = set()

    for image_path in selected_images:
        if image_path.stem not in selected_stems:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_label_files.append(label_path.name)
            continue
        rows, problems = parse_label(label_path, len(classes))
        label_problems.update(problems)
        if not rows:
            empty_files += 1
        for class_id, width, height in rows:
            per_class_box[class_id].append((width, height))
            total_boxes += 1
            class_id_seen.add(class_id)

    res_by_name, res_warnings = measure_image_resolutions(selected_images, args.sample_img)
    if res_by_name:
        widths = [width for width, _ in res_by_name.values()]
        heights = [height for _, height in res_by_name.values()]
        median_w = int(statistics.median(widths))
        median_h = int(statistics.median(heights))
        resolution_source = "sampled_selected_images"
    else:
        median_w, median_h = FALLBACK_MEDIAN_WH
        resolution_source = "fallback_constant"

    shape_counter = Counter(f"{width}x{height}" for width, height in res_by_name.values())
    contexts = summarize_contexts(median_w, median_h, args.imgsz)
    report: dict[str, Any] = {
        "schema_version": 2,
        "generated_by": "l1_box_size_diag.py",
        "anchor_root": str(anchor_root),
        "classes": classes,
        "nc": len(classes),
        "split": split_info,
        "selected_label_files": len(selected_images) - len(missing_label_files),
        "missing_label_files": missing_label_files[:20],
        "missing_label_file_count": len(missing_label_files),
        "empty_label_files": empty_files,
        "total_boxes": total_boxes,
        "class_ids_seen": sorted(class_id_seen),
        "label_problems": dict(label_problems),
        "image_resolution": {
            "source": resolution_source,
            "sample_requested": args.sample_img,
            "sampled": len(res_by_name),
            "median_w": median_w,
            "median_h": median_h,
            "unique_shapes": [f"{shape} ({count})" for shape, count in shape_counter.most_common()],
            "samples_first5": {name: list(value) for name, value in list(res_by_name.items())[:5]},
        },
        "imgsz": args.imgsz,
        "small_px_threshold": SMALL_PX,
        "tiny_px_threshold": TINY_PX,
        "small_area_px_threshold": SMALL_AREA_PX,
        "contexts": contexts,
        "per_class": [],
        "resolution_warnings": res_warnings,
    }

    for class_id, class_name in enumerate(classes):
        boxes = per_class_box.get(class_id, [])
        entry: dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "count": len(boxes),
            "interest": class_name in HIGH_INTEREST,
        }
        if not boxes:
            entry.update({"status": "no_box", "note": "no boxes in selected scope"})
            report["per_class"].append(entry)
            continue
        ws = [width for width, _ in boxes]
        hs = [height for _, height in boxes]
        entry.update(
            {
                "status": "evaluated",
                "w_norm_median": statistics.median(ws),
                "h_norm_median": statistics.median(hs),
                "area_norm_median": statistics.median([width * height for width, height in boxes]),
                "min_side_norm_median": statistics.median(min(width, height) for width, height in boxes),
                "contexts": {
                    context_name: context_metrics(
                        ws,
                        hs,
                        median_w,
                        median_h,
                        context["width_scale"],
                        context["height_scale"],
                    )
                    for context_name, context in contexts.items()
                },
            }
        )
        report["per_class"].append(entry)

    verdict_lines: list[str] = []
    for entry in report["per_class"]:
        if not entry.get("interest") or entry["status"] != "evaluated":
            continue
        metrics = entry["contexts"]["val_letterbox"]
        verdict_lines.append(
            f"- {entry['class_name']}: 中位最小边约 {metrics['min_side_px_median']:.1f}px，"
            f"<32px 占 {human_pct(metrics['pct_min_side_lt_32'])}，"
            f"<16px 占 {human_pct(metrics['pct_min_side_lt_16'])}，样本 {entry['count']}"
        )
    report["high_interest_verdict"] = verdict_lines

    print("\n=== val_letterbox per-class summary ===")
    print(f"{'class':>14} {'n':>6} {'min_med':>9} {'<32px':>8} {'<16px':>8} {'w_med':>9} {'h_med':>9}")
    for entry in report["per_class"]:
        if entry["status"] != "evaluated":
            print(f"{entry['class_name']:>14} {entry['count']:>6} {'n/a':>9} {'n/a':>8} {'n/a':>8} {'n/a':>9} {'n/a':>9}")
            continue
        metrics = entry["contexts"]["val_letterbox"]
        print(
            f"{entry['class_name']:>14} {entry['count']:>6} {metrics['min_side_px_median']:>9.1f} "
            f"{human_pct(metrics['pct_min_side_lt_32']):>8} {human_pct(metrics['pct_min_side_lt_16']):>8} "
            f"{metrics['w_px_median']:>9.1f} {metrics['h_px_median']:>9.1f}"
        )

    if args.dry_run:
        print("\n[dry-run] Report files were not written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "box_size_report.json"
    out_md = output_dir / "box_size_report.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")
    print(f"\n[ok] wrote {out_json}")
    print(f"[ok] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
