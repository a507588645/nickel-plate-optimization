"""D4 phase-1 business-output comparison for the dianji detectors.

The script deliberately loads both detectors directly through the repository
adapters instead of changing models.yaml or the production inference path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CONFIDENCE_THRESHOLD = 0.25
CANDIDATE_IMAGE_SIZE = 640
SLF_MODEL_NAME = "slf.pt"
CANDIDATE_MODEL_NAME = "dianjienie_yolo11m_ryw18_640.onnx"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

SLF_CLASS_NAMES = {
    0: "waterstain",
    1: "waterblock",
    2: "colorstain",
    3: "nodule",
    4: "bubble",
}

RYW18_CLASS_NAMES = {
    0: "wuzi",
    1: "waterstain",
    2: "yellowstain",
    3: "nodule",
    4: "airhole",
    5: "bubble",
    6: "bump",
    7: "fawu",
    8: "overfire",
    9: "NiOH",
    10: "greenstain",
    11: "suanwuran",
    12: "quejiao",
    13: "watermark",
    14: "youpao",
    15: "heibian",
    16: "mijixiaokeng",
    17: "fanxitong",
}

SHARED_CLASS_IDS = {
    "waterstain": (0, 1),
    "nodule": (3, 3),
    "bubble": (4, 5),
}

# This is the production dianji mapping, recorded here only for audit context.
SLF_PRODUCTION_LABELS = {0: 1, 1: 0, 2: 2, 3: 3, 4: 5}

LAYOUT_CANDIDATES = (
    ("images/val", "labels/val"),
    ("valid/images", "valid/labels"),
    ("val/images", "val/labels"),
    ("images/valid", "labels/valid"),
)

PER_IMAGE_FIELDS = [
    "image",
    "image_path",
    "image_width",
    "image_height",
    "image_area_px",
    "slf_defect_count",
    "candidate_defect_count",
    "count_delta_candidate_minus_slf",
    "slf_box_area_px",
    "candidate_box_area_px",
    "slf_area_ratio",
    "candidate_area_ratio",
    "area_ratio_delta_candidate_minus_slf",
    "divergence_score",
]
for _class_id, _class_name in SLF_CLASS_NAMES.items():
    PER_IMAGE_FIELDS.append(f"slf_class_{_class_id}_{_class_name}_count")
for _class_name, (_slf_id, _candidate_id) in SHARED_CLASS_IDS.items():
    PER_IMAGE_FIELDS.append(f"candidate_class_{_candidate_id}_{_class_name}_count")
PER_IMAGE_FIELDS.append("candidate_only_count")

PER_BOX_FIELDS = [
    "model",
    "image",
    "image_path",
    "image_width",
    "image_height",
    "class_id",
    "class_name",
    "confidence",
    "x_center",
    "y_center",
    "width",
    "height",
    "x1",
    "y1",
    "x2",
    "y2",
    "box_area_px",
    "box_area_ratio",
    "production_slf_label",
    "is_shared_name",
    "is_candidate_only",
]


def _format_float(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def _format_pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def _format_int(value: int | float) -> str:
    return f"{int(value):,}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_files(directory: Path) -> tuple[list[Path], Counter[str]]:
    files = sorted((path for path in directory.iterdir() if path.is_file()), key=lambda path: path.name)
    extension_counts = Counter(path.suffix.lower() or "<no extension>" for path in files)
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    return images, extension_counts


def _label_names(directory: Path | None) -> set[str]:
    if directory is None or not directory.is_dir():
        return set()
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt" and path.name.lower() != "classes.txt"
    }


def collect_validation_images(dataset_root: Path) -> tuple[list[Path], dict[str, Any]]:
    """Discover valid layouts and deduplicate exact mirror files by filename/content."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"验证集目录不存在: {dataset_root}")

    detected_layouts: list[dict[str, Any]] = []
    selected_by_name: dict[str, Path] = {}
    selected_hashes: dict[str, str] = {}
    hash_cache: dict[Path, str] = {}
    duplicate_conflicts: list[str] = []

    def content_hash(path: Path) -> str:
        if path not in hash_cache:
            hash_cache[path] = _sha256(path)
        return hash_cache[path]

    for image_rel, label_rel in LAYOUT_CANDIDATES:
        image_dir = dataset_root / Path(image_rel)
        label_dir = dataset_root / Path(label_rel)
        if not image_dir.is_dir():
            continue

        images, extension_counts = _image_files(image_dir)
        labels = _label_names(label_dir if label_dir.is_dir() else None)
        detected_layouts.append(
            {
                "image_rel": image_rel,
                "label_rel": label_rel if label_dir.is_dir() else None,
                "image_dir": image_dir,
                "label_dir": label_dir if label_dir.is_dir() else None,
                "all_file_count": sum(extension_counts.values()),
                "image_count": len(images),
                "extension_counts": dict(sorted(extension_counts.items())),
                "label_count": len(labels),
                "classes_txt": (label_dir / "classes.txt").is_file() if label_dir.is_dir() else False,
            }
        )

        for image_path in images:
            name = image_path.name
            image_hash = content_hash(image_path)
            previous = selected_by_name.get(name)
            if previous is None:
                selected_by_name[name] = image_path
                selected_hashes[name] = image_hash
            elif selected_hashes[name] != image_hash:
                duplicate_conflicts.append(f"{name}: {previous} != {image_path}")

    if duplicate_conflicts:
        conflicts = "\n".join(duplicate_conflicts[:10])
        raise RuntimeError(f"验证布局中同名图像内容不一致，无法安全去重:\n{conflicts}")
    if not selected_by_name:
        expected = ", ".join(image_rel for image_rel, _ in LAYOUT_CANDIDATES)
        raise FileNotFoundError(f"未找到验证图像目录，已检查: {expected}")

    image_paths = sorted(selected_by_name.values(), key=lambda path: path.name)
    label_dirs = [layout["label_dir"] for layout in detected_layouts if layout["label_dir"] is not None]
    all_label_names: set[str] = set()
    for label_dir in label_dirs:
        all_label_names.update(_label_names(label_dir))
    missing_labels = [path.name for path in image_paths if path.stem not in all_label_names]

    layout_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(detected_layouts):
        left_paths = {path.name: path for path in _image_files(left["image_dir"])[0]}
        for right in detected_layouts[left_index + 1 :]:
            right_paths = {path.name: path for path in _image_files(right["image_dir"])[0]}
            common_names = sorted(set(left_paths) & set(right_paths))
            same_names = bool(common_names) and set(left_paths) == set(right_paths)
            same_content = same_names and all(
                content_hash(left_paths[name]) == content_hash(right_paths[name]) for name in common_names
            )
            layout_pairs.append(
                {
                    "left": left["image_rel"],
                    "right": right["image_rel"],
                    "left_count": len(left_paths),
                    "right_count": len(right_paths),
                    "common_count": len(common_names),
                    "same_names": same_names,
                    "same_content": same_content,
                }
            )

    dataset_info = {
        "dataset_root": dataset_root,
        "layouts": detected_layouts,
        "layout_pairs": layout_pairs,
        "selected_image_count": len(image_paths),
        "selected_image_paths": image_paths,
        "missing_labels": missing_labels,
        "label_count_union": len(all_label_names),
        "ignored_non_image_extensions": dict(
            sorted(
                Counter(
                    extension
                    for layout in detected_layouts
                    for extension, count in layout["extension_counts"].items()
                    if extension not in IMAGE_SUFFIXES
                    for _ in range(count)
                ).items()
            )
        ),
    }
    return image_paths, dataset_info


def _decode_bgr(image_path: Path) -> np.ndarray:
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取验证图像: {image_path}")
    return np.ascontiguousarray(image)


def _result_rows(result: Any, model_name: str, image_name: str) -> np.ndarray:
    values = result.xyxy[0]
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    rows = np.asarray(values, dtype=np.float32)
    if rows.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    rows = rows.reshape(-1, 6)
    if not np.isfinite(rows).all():
        raise ValueError(f"{model_name} 在 {image_name} 返回了非有限检测框")
    return rows


def _load_detectors(repo_root: Path) -> tuple[Any, Any]:
    from adapters.ultralytics import UltralyticsDetector
    from adapters.yolov5 import YoloV5Detector

    slf_weights = repo_root / "weights" / "slf.pt"
    candidate_weights = repo_root / "weights" / "dianjienie_yolo11m_ryw18_640.onnx"
    if not slf_weights.is_file():
        raise FileNotFoundError(f"找不到当前 dianji 权重: {slf_weights}")
    if not candidate_weights.is_file():
        raise FileNotFoundError(f"找不到候选权重: {candidate_weights}")

    # YOLOv5 loader currently concatenates a possible .enc suffix as a string.
    slf = YoloV5Detector(
        weights=str(slf_weights.resolve()),
        device="cpu",
        threshold=CONFIDENCE_THRESHOLD,
    )
    candidate = UltralyticsDetector(
        weights=str(candidate_weights.resolve()),
        device="cpu",
        threshold=CONFIDENCE_THRESHOLD,
        imgsz=CANDIDATE_IMAGE_SIZE,
    )
    return slf, candidate


def _validate_class_ids(rows: np.ndarray, model_name: str, image_name: str) -> None:
    class_ids = rows[:, 5].astype(int) if len(rows) else np.empty(0, dtype=int)
    max_class_id = 4 if model_name == SLF_MODEL_NAME else 17
    if np.any(class_ids < 0) or np.any(class_ids > max_class_id):
        raise ValueError(f"{model_name} 在 {image_name} 返回了超出范围的 class_id: {class_ids.tolist()}")


def _summarize_rows(rows: np.ndarray, width: int, height: int, model_name: str, image_name: str) -> dict[str, Any]:
    _validate_class_ids(rows, model_name, image_name)
    class_names = SLF_CLASS_NAMES if model_name == SLF_MODEL_NAME else RYW18_CLASS_NAMES
    class_counts = Counter(int(row[5]) for row in rows)
    box_area_px = 0.0
    for row in rows:
        x1, y1, x2, y2 = (float(value) for value in row[:4])
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        box_area_px += box_width * box_height
    image_area_px = float(width * height)
    return {
        "count": int(len(rows)),
        "box_area_px": box_area_px,
        "area_ratio": box_area_px / image_area_px,
        "class_counts": class_counts,
        "class_names": class_names,
    }


def _per_image_record(
    image_path: Path,
    width: int,
    height: int,
    slf_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    count_delta = candidate_summary["count"] - slf_summary["count"]
    area_delta = candidate_summary["area_ratio"] - slf_summary["area_ratio"]
    record: dict[str, Any] = {
        "image": image_path.name,
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "image_area_px": width * height,
        "slf_defect_count": slf_summary["count"],
        "candidate_defect_count": candidate_summary["count"],
        "count_delta_candidate_minus_slf": count_delta,
        "slf_box_area_px": slf_summary["box_area_px"],
        "candidate_box_area_px": candidate_summary["box_area_px"],
        "slf_area_ratio": slf_summary["area_ratio"],
        "candidate_area_ratio": candidate_summary["area_ratio"],
        "area_ratio_delta_candidate_minus_slf": area_delta,
        "divergence_score": abs(count_delta) + abs(area_delta),
    }
    for class_id, class_name in SLF_CLASS_NAMES.items():
        record[f"slf_class_{class_id}_{class_name}_count"] = slf_summary["class_counts"].get(class_id, 0)
    for class_name, (_slf_id, candidate_id) in SHARED_CLASS_IDS.items():
        record[f"candidate_class_{candidate_id}_{class_name}_count"] = candidate_summary["class_counts"].get(
            candidate_id, 0
        )
    candidate_shared_ids = {candidate_id for _slf_id, candidate_id in SHARED_CLASS_IDS.values()}
    record["candidate_only_count"] = sum(
        count for class_id, count in candidate_summary["class_counts"].items() if class_id not in candidate_shared_ids
    )
    return record


def _per_box_records(
    model_name: str,
    image_path: Path,
    width: int,
    height: int,
    rows: np.ndarray,
) -> list[dict[str, Any]]:
    class_names = SLF_CLASS_NAMES if model_name == SLF_MODEL_NAME else RYW18_CLASS_NAMES
    shared_candidate_ids = {candidate_id for _slf_id, candidate_id in SHARED_CLASS_IDS.values()}
    output: list[dict[str, Any]] = []
    image_area = float(width * height)
    for row in rows:
        x1, y1, x2, y2, confidence, class_value = (float(value) for value in row)
        class_id = int(class_value)
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        is_shared = class_id in (
            {slf_id for slf_id, _candidate_id in SHARED_CLASS_IDS.values()}
            if model_name == SLF_MODEL_NAME
            else shared_candidate_ids
        )
        output.append(
            {
                "model": model_name,
                "image": image_path.name,
                "image_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": confidence,
                "x_center": (x1 + x2) / 2.0,
                "y_center": (y1 + y2) / 2.0,
                "width": x2 - x1,
                "height": y2 - y1,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "box_area_px": box_area,
                "box_area_ratio": box_area / image_area,
                "production_slf_label": SLF_PRODUCTION_LABELS[class_id] if model_name == SLF_MODEL_NAME else "",
                "is_shared_name": int(is_shared),
                "is_candidate_only": int(model_name != SLF_MODEL_NAME and not is_shared),
            }
        )
    return output


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _distribution_stats(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0, "min": 0.0}
    return {
        "median": float(np.median(data)),
        "mean": float(np.mean(data)),
        "p90": float(np.percentile(data, 90)),
        "max": float(np.max(data)),
        "min": float(np.min(data)),
    }


def _stat_table_row(label: str, stats: dict[str, float], digits: int = 4) -> str:
    return (
        f"| {label} | {stats['median']:.{digits}f} | {stats['mean']:.{digits}f} | "
        f"{stats['p90']:.{digits}f} | {stats['max']:.{digits}f} |"
    )


def _layout_report(dataset_info: dict[str, Any]) -> list[str]:
    lines = [
        "### 验证集布局",
        f"- 数据集根目录：`{dataset_info['dataset_root']}`。",
        "- 图像枚举规则：只使用常见图像后缀，按文件名排序；`.npy` 不作为检测输入。",
    ]
    for layout in dataset_info["layouts"]:
        extension_text = ", ".join(
            f"`{extension}`={count}" for extension, count in layout["extension_counts"].items()
        )
        label_text = "无" if layout["label_dir"] is None else f"{layout['label_count']} 个 txt 标签"
        if layout["classes_txt"]:
            label_text += "，含 `classes.txt`"
        lines.append(
            f"- `{layout['image_rel']}`：{layout['image_count']} 个图像文件，目录总文件 {layout['all_file_count']}；"
            f"扩展名统计：{extension_text}；标签目录 `{layout['label_rel']}`：{label_text}。"
        )
    if dataset_info["layout_pairs"]:
        for pair in dataset_info["layout_pairs"]:
            lines.append(
                f"- 布局核对 `{pair['left']}` vs `{pair['right']}`：共同文件 {pair['common_count']}，"
                f"文件名集合相同={pair['same_names']}，内容相同={pair['same_content']}。"
            )
    lines.append(
        f"- 去重后实际评估 {dataset_info['selected_image_count']} 张独立 PNG/图像；"
        f"同名标签缺失 {len(dataset_info['missing_labels'])} 张。标签只做布局记录，不参与本次 detector-output 比较。"
    )
    if dataset_info["missing_labels"]:
        examples = ", ".join(dataset_info["missing_labels"][:5])
        lines.append(f"- 缺失标签示例：{examples}。")
    ignored = ", ".join(
        f"`{extension}`={count}" for extension, count in dataset_info["ignored_non_image_extensions"].items()
    )
    if ignored:
        lines.append(f"- 发现但忽略的非图像文件：{ignored}。")
    return lines


def _render_panel(
    image: np.ndarray,
    rows: np.ndarray,
    title: str,
    class_names: dict[int, str],
    max_width: int = 1400,
) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = min(1.0, max_width / float(source_width))
    if scale != 1.0:
        panel = cv2.resize(image, (int(round(source_width * scale)), int(round(source_height * scale))), interpolation=cv2.INTER_AREA)
    else:
        panel = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    header_height = 60
    canvas = np.full((panel.shape[0] + header_height, panel.shape[1], 3), 245, dtype=np.uint8)
    canvas[header_height:] = panel
    cv2.putText(canvas, title, (16, 38), font, 0.78, (30, 30, 30), 2, cv2.LINE_AA)

    colors = [
        (51, 153, 255),
        (60, 200, 60),
        (255, 120, 40),
        (190, 80, 220),
        (0, 190, 210),
        (220, 180, 30),
        (40, 120, 220),
        (150, 90, 40),
    ]
    for row in rows:
        x1, y1, x2, y2, confidence, class_value = (float(value) for value in row)
        class_id = int(class_value)
        left = max(0, min(canvas.shape[1] - 1, int(round(x1 * scale))))
        top = max(header_height, min(canvas.shape[0] - 1, int(round(y1 * scale)) + header_height))
        right = max(0, min(canvas.shape[1] - 1, int(round(x2 * scale))))
        bottom = max(header_height, min(canvas.shape[0] - 1, int(round(y2 * scale)) + header_height))
        color = colors[class_id % len(colors)]
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 3, cv2.LINE_AA)
        label = f"id={class_id} {class_names[class_id]} conf={confidence:.2f}"
        text_size, baseline = cv2.getTextSize(label, font, 0.52, 1)
        label_top = max(header_height, top - text_size[1] - baseline - 4)
        label_bottom = label_top + text_size[1] + baseline + 4
        label_right = min(canvas.shape[1] - 1, left + text_size[0] + 8)
        cv2.rectangle(canvas, (left, label_top), (label_right, label_bottom), color, -1)
        cv2.putText(canvas, label, (left + 4, label_bottom - baseline - 2), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _render_comparison(
    image_path: Path,
    slf_rows: np.ndarray,
    candidate_rows: np.ndarray,
    output_path: Path,
    rank: int,
) -> None:
    image = _decode_bgr(image_path)
    slf_area = _summarize_rows(slf_rows, image.shape[1], image.shape[0], SLF_MODEL_NAME, image_path.name)["area_ratio"]
    candidate_area = _summarize_rows(
        candidate_rows, image.shape[1], image.shape[0], CANDIDATE_MODEL_NAME, image_path.name
    )["area_ratio"]
    left = _render_panel(
        image,
        slf_rows,
        f"Current slf.pt | count={len(slf_rows)} | area={slf_area * 100:.2f}%",
        SLF_CLASS_NAMES,
    )
    right = _render_panel(
        image,
        candidate_rows,
        f"Candidate YOLO11m ONNX | count={len(candidate_rows)} | area={candidate_area * 100:.2f}%",
        RYW18_CLASS_NAMES,
    )
    target_height = max(left.shape[0], right.shape[0])
    if left.shape[0] != target_height:
        left = cv2.copyMakeBorder(left, 0, target_height - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    if right.shape[0] != target_height:
        right = cv2.copyMakeBorder(right, 0, target_height - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    comparison = cv2.hconcat([left, right])
    cv2.putText(comparison, f"review rank {rank:02d}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), comparison):
        raise IOError(f"无法写入审查图像: {output_path}")


def _safe_filename(image_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(image_name).stem)


def _write_review_html(
    output_path: Path,
    top_rows: list[dict[str, Any]],
    render_names: list[str],
    total_images: int,
) -> None:
    cards: list[str] = []
    for rank, (row, render_name) in enumerate(zip(top_rows, render_names), start=1):
        cards.append(
            "\n".join(
                [
                    '<article class="card">',
                    f"<h2>#{rank:02d} {html.escape(row['image'])}</h2>",
                    "<div class=\"metrics\">"
                    f"SLF {row['slf_defect_count']} boxes / candidate {row['candidate_defect_count']} boxes; "
                    f"count delta {row['count_delta_candidate_minus_slf']:+d}; "
                    f"area delta {row['area_ratio_delta_candidate_minus_slf'] * 100:+.2f}%; "
                    f"score {row['divergence_score']:.4f}"
                    "</div>",
                    f'<img src="review_images/{html.escape(render_name)}" alt="{html.escape(row["image"])} comparison">',
                    "</article>",
                ]
            )
        )
    content = "\n".join(cards) if cards else '<p class="empty">没有非零 divergence image。</p>'
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>D4 dianji business comparison review</title>
<style>
:root {{ color-scheme: light; --ink: #1d2927; --muted: #5b6965; --paper: #f2eee6; --card: #fffdf8; --line: #d8d0c2; --accent: #086b66; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: var(--ink); background: linear-gradient(135deg, #e7eee9, var(--paper)); font-family: "Microsoft YaHei", "Segoe UI", sans-serif; }}
header {{ padding: 24px clamp(16px, 4vw, 56px) 18px; border-bottom: 1px solid var(--line); background: #fffdf8dd; position: sticky; top: 0; z-index: 2; backdrop-filter: blur(8px); }}
h1 {{ margin: 0 0 8px; font-family: Georgia, "Times New Roman", serif; font-size: clamp(24px, 3vw, 38px); }}
.summary {{ margin: 0; color: var(--muted); line-height: 1.6; }}
main {{ width: min(1800px, 100%); margin: 0 auto; padding: 24px clamp(12px, 3vw, 42px) 60px; display: grid; gap: 22px; }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; box-shadow: 0 10px 30px #30413b12; padding: 14px; overflow: hidden; }}
.card h2 {{ margin: 0 0 5px; font-size: 18px; }}
.metrics {{ margin: 0 0 12px; color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }}
.card img {{ display: block; width: 100%; height: auto; border: 1px solid #cfc7b9; background: #eee; }}
.empty {{ padding: 40px; background: var(--card); border: 1px solid var(--line); border-radius: 12px; }}
@media (min-width: 1500px) {{ main {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .card:first-child {{ grid-column: 1 / -1; }} }}
</style>
</head>
<body>
<header>
<h1>D4 dianji business-output review</h1>
<p class="summary">Top {len(top_rows)} non-zero divergence comparisons from {total_images} evaluated images. Left: current <code>slf.pt</code>; right: candidate YOLO11m 640 ONNX. Both use original BGR images, CPU inference, confidence threshold 0.25. Open this file with its sibling <code>review_images</code> directory.</p>
</header>
<main>
{content}
</main>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def _write_report(
    report_path: Path,
    run_started: datetime,
    total_wall_seconds: float,
    inference_seconds: float,
    repo_root: Path,
    dataset_info: dict[str, Any],
    image_rows: list[dict[str, Any]],
    box_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    review_count: int,
    stride: int,
) -> None:
    slf_weight = repo_root / "weights" / "slf.pt"
    candidate_weight = repo_root / "weights" / "dianjienie_yolo11m_ryw18_640.onnx"
    total = len(image_rows)
    slf_counts = [row["slf_defect_count"] for row in image_rows]
    candidate_counts = [row["candidate_defect_count"] for row in image_rows]
    count_deltas = [row["count_delta_candidate_minus_slf"] for row in image_rows]
    area_deltas = [row["area_ratio_delta_candidate_minus_slf"] for row in image_rows]
    abs_count_deltas = [abs(value) for value in count_deltas]
    abs_area_deltas = [abs(value) for value in area_deltas]
    slf_detected = sum(count > 0 for count in slf_counts)
    candidate_detected = sum(count > 0 for count in candidate_counts)

    slf_total_by_class = Counter()
    candidate_total_by_class = Counter()
    for row in box_rows:
        if row["model"] == SLF_MODEL_NAME:
            slf_total_by_class[int(row["class_id"])] += 1
        else:
            candidate_total_by_class[int(row["class_id"])] += 1

    lines = [
        "# D4 电积镍 detector business-output 对比报告",
        "",
        f"- 运行开始：`{run_started.isoformat(timespec='seconds')}`。",
        f"- 总墙钟耗时（布局核验、模型加载、推理、渲染、CSV/HTML 写盘）：`{total_wall_seconds:.2f}` 秒。",
        f"- detector 推理循环耗时：`{inference_seconds:.2f}` 秒。",
        f"- 评估图像：`{total}` 张；图像选择 stride=`{stride}`，`stride=1` 表示全量去重集。",
        f"- 产物目录：`{report_path.parent}`。",
        "- 本阶段仅分析，不切换 `models.yaml`，不调用生产 classifier；生产 grade `type` 按要求排除，只比较 defects 列表。",
        "",
        "## 方法",
        "- 当前模型：`weights/slf.pt`，YOLOv5 adapter，5 类，类序 `0=waterstain, 1=waterblock, 2=colorstain, 3=nodule, 4=bubble`，阈值 `0.25`。",
        "- 候选模型：`weights/dianjienie_yolo11m_ryw18_640.onnx`，Ultralytics adapter，RYW18 18 类，`imgsz=640`，阈值 `0.25`。",
        "- 两个 adapter 均接收同一张原始 BGR ndarray；没有把业务层的 `(1024, 888)` 副本传入 detector。框坐标按 adapter 契约解释为原图像素坐标。",
        "- 每张图记录 detector business defect-list 的框数和所有框面积之和。面积比为 `sum(box_area_px) / image_area_px`，重叠框不做 union。",
        "- 同名对照只使用已确认映射：`waterstain`（SLF 0 / RYW18 1）、`nodule`（3 / 3）、`bubble`（4 / 5）。`waterblock`、`colorstain` 不参与伪映射，候选其他 15 类单列为 candidate-only。",
        f"- 权重 SHA-256：`slf.pt={_sha256(slf_weight)}`；`dianjienie_yolo11m_ryw18_640.onnx={_sha256(candidate_weight)}`。",
        "",
    ]
    lines.extend(_layout_report(dataset_info))
    lines.extend(
        [
            "",
            "## 总体结果",
            "",
            "| 模型 | 平均 defect count | 检出图像数 | 图像级 detection rate |",
            "|---|---:|---:|---:|",
            f"| `slf.pt` | {np.mean(slf_counts):.4f} | {slf_detected}/{total} | {_format_pct(slf_detected / total)} |",
            f"| `yolo11m_640.onnx` | {np.mean(candidate_counts):.4f} | {candidate_detected}/{total} | {_format_pct(candidate_detected / total)} |",
            f"| candidate - slf | {np.mean(candidate_counts) - np.mean(slf_counts):+.4f} | - | {(candidate_detected - slf_detected) / total:+.4%} |",
            "",
            "### 差异分布",
            "",
            "以下 signed delta 定义为 `candidate - slf`；面积 delta 的单位为 ratio。",
            "",
            "| 指标 | median | mean | p90 | max |",
            "|---|---:|---:|---:|---:|",
            _stat_table_row("count delta", _distribution_stats(count_deltas), digits=4),
            _stat_table_row("area-ratio delta", _distribution_stats(area_deltas), digits=6),
            _stat_table_row("absolute count delta", _distribution_stats(abs_count_deltas), digits=4),
            _stat_table_row("absolute area-ratio delta", _distribution_stats(abs_area_deltas), digits=6),
            "",
            "### 已确认同名类别总数",
            "",
            "| 类别 | SLF class id | RYW18 class id | SLF 总框数 | candidate 总框数 | candidate - SLF |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for class_name, (slf_id, candidate_id) in SHARED_CLASS_IDS.items():
        slf_count = slf_total_by_class[slf_id]
        candidate_count = candidate_total_by_class[candidate_id]
        lines.append(
            f"| `{class_name}` | {slf_id} | {candidate_id} | {slf_count} | {candidate_count} | {candidate_count - slf_count:+d} |"
        )
    lines.extend(
        [
            "",
            "### SLF 类别总数",
            "",
            "| SLF class id | 类别 | 总框数 | 生产输出 label |",
            "|---:|---|---:|---:|",
        ]
    )
    for class_id, class_name in SLF_CLASS_NAMES.items():
        lines.append(f"| {class_id} | `{class_name}` | {slf_total_by_class[class_id]} | {SLF_PRODUCTION_LABELS[class_id]} |")
    lines.extend(
        [
            "",
            "### Candidate-only 类别总数",
            "",
            "candidate-only 定义为 RYW18 class id 不在 `{1, 3, 5}` 的检测框。",
            "",
            "| RYW18 class id | 类别 | 总框数 |",
            "|---:|---|---:|",
        ]
    )
    for class_id, class_name in RYW18_CLASS_NAMES.items():
        if class_id not in {candidate_id for _slf_id, candidate_id in SHARED_CLASS_IDS.values()}:
            lines.append(f"| {class_id} | `{class_name}` | {candidate_total_by_class[class_id]} |")

    lines.extend(
        [
            "",
            f"## Top {len(top_rows)} divergence images",
            "",
            "排序分数严格为 `abs(count_delta) + abs(area_ratio_delta)`，其中 delta 均为 `candidate - slf`。审查图像见 [`review.html`](review.html)。",
            "",
            "| rank | image | SLF count | candidate count | count delta | SLF area | candidate area | area delta | score | rendered |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(top_rows, start=1):
        render_name = f"{rank:02d}_{_safe_filename(row['image'])}"
        lines.append(
            f"| {rank} | `{row['image']}` | {row['slf_defect_count']} | {row['candidate_defect_count']} | "
            f"{row['count_delta_candidate_minus_slf']:+d} | {_format_pct(row['slf_area_ratio'])} | "
            f"{_format_pct(row['candidate_area_ratio'])} | {_format_pct(row['area_ratio_delta_candidate_minus_slf'])} | "
            f"{row['divergence_score']:.6f} | `review_images/{render_name}.png` |"
        )
    lines.extend(
        [
            "",
            "## 产物",
            "",
            "- [`per_image.csv`](per_image.csv)：每张图的框数、面积比、delta 和指定类别计数。",
            "- [`per_box.csv`](per_box.csv)：两模型所有原始框的 model/image/class id/confidence/xywh 及审计字段。",
            "- [`review.html`](review.html)：无外部脚本或 CSS 依赖的 side-by-side 审查页，图像使用同目录相对路径。",
            f"- 已渲染审查比较：`{review_count}` 张；HTML 中每张比较对应一个 `<img>` 引用。",
            "",
            "## 限制与闸门",
            "",
            "- 本次没有 ground-truth comparison；标签仅用于核对目录布局，没有计算 precision、recall 或 mAP。",
            "- `waterblock` 与 `colorstain` 没有确认的 RYW18 映射；它们保持为 SLF-only，不做猜测或合并。",
            "- 全部检测使用单一 confidence threshold `0.25`，没有类别专用阈值或后处理调参扫描。",
            "- 两个 detector 均为 CPU inference；耗时和数值表现不代表 GPU 部署性能。",
            "- pass/fail gate 不由本报告自动判定，而是由用户对 `review.html` 中 largest disagreements 的人工 adjudication 决定。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:\PythonProject\datasets\Dianjinie0211"),
        help="只读验证集根目录",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(r"E:\PythonProject\优化"),
        help="D4 产物根目录；默认使用项目外的分析目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="可选的固定输出目录；未指定时自动创建 d4_dianji_business_compare_<timestamp>",
    )
    parser.add_argument(
        "--image-stride",
        type=int,
        default=1,
        help="按排序列表每 N 张取 1 张；默认 1 为全量。用于长时间运行时的确定性子样本。",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    run_started = datetime.now()
    total_wall_started = time.perf_counter()
    if args.image_stride < 1:
        raise ValueError("--image-stride 必须 >= 1")

    all_image_paths, dataset_info = collect_validation_images(dataset_root)
    image_paths = all_image_paths[:: args.image_stride]
    if not image_paths:
        raise ValueError("按 image-stride 选择后没有验证图像")
    dataset_info = dict(dataset_info)
    dataset_info["selected_image_count"] = len(image_paths)
    dataset_info["selected_image_paths"] = image_paths

    if args.output_dir is None:
        output_dir = args.artifact_root.expanduser().resolve() / (
            f"d4_dianji_business_compare_{run_started.strftime('%Y%m%d_%H%M%S')}"
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()
    if dataset_root == output_dir or dataset_root in output_dir.parents:
        raise ValueError(f"产物目录不能位于只读数据集目录内: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    review_dir = output_dir / "review_images"
    review_dir.mkdir()

    print(f"dataset_root={dataset_root}", flush=True)
    print(f"unique_validation_images={len(all_image_paths)}", flush=True)
    print(f"evaluated_images={len(image_paths)} stride={args.image_stride}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print("loading detectors on CPU...", flush=True)
    slf_detector, candidate_detector = _load_detectors(repo_root)

    image_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    detections: dict[str, tuple[Path, np.ndarray, np.ndarray]] = {}
    inference_started = time.perf_counter()
    try:
        import torch

        inference_context = torch.inference_mode()
    except AttributeError:
        inference_context = None

    if inference_context is None:
        context_manager = None
    else:
        context_manager = inference_context

    if context_manager is None:
        image_iterator = image_paths
        for index, image_path in enumerate(image_iterator, start=1):
            image = _decode_bgr(image_path)
            slf_rows = _result_rows(slf_detector(image), SLF_MODEL_NAME, image_path.name)
            candidate_rows = _result_rows(candidate_detector(image), CANDIDATE_MODEL_NAME, image_path.name)
            height, width = image.shape[:2]
            slf_summary = _summarize_rows(slf_rows, width, height, SLF_MODEL_NAME, image_path.name)
            candidate_summary = _summarize_rows(candidate_rows, width, height, CANDIDATE_MODEL_NAME, image_path.name)
            image_rows.append(_per_image_record(image_path, width, height, slf_summary, candidate_summary))
            box_rows.extend(_per_box_records(SLF_MODEL_NAME, image_path, width, height, slf_rows))
            box_rows.extend(_per_box_records(CANDIDATE_MODEL_NAME, image_path, width, height, candidate_rows))
            detections[image_path.name] = (image_path, slf_rows, candidate_rows)
            if index == 1 or index % 10 == 0 or index == len(image_paths):
                print(
                    f"[{index}/{len(image_paths)}] {image_path.name} slf={len(slf_rows)} candidate={len(candidate_rows)}",
                    flush=True,
                )
    else:
        with context_manager:
            for index, image_path in enumerate(image_paths, start=1):
                image = _decode_bgr(image_path)
                slf_rows = _result_rows(slf_detector(image), SLF_MODEL_NAME, image_path.name)
                candidate_rows = _result_rows(candidate_detector(image), CANDIDATE_MODEL_NAME, image_path.name)
                height, width = image.shape[:2]
                slf_summary = _summarize_rows(slf_rows, width, height, SLF_MODEL_NAME, image_path.name)
                candidate_summary = _summarize_rows(candidate_rows, width, height, CANDIDATE_MODEL_NAME, image_path.name)
                image_rows.append(_per_image_record(image_path, width, height, slf_summary, candidate_summary))
                box_rows.extend(_per_box_records(SLF_MODEL_NAME, image_path, width, height, slf_rows))
                box_rows.extend(_per_box_records(CANDIDATE_MODEL_NAME, image_path, width, height, candidate_rows))
                detections[image_path.name] = (image_path, slf_rows, candidate_rows)
                if index == 1 or index % 10 == 0 or index == len(image_paths):
                    print(
                        f"[{index}/{len(image_paths)}] {image_path.name} slf={len(slf_rows)} candidate={len(candidate_rows)}",
                        flush=True,
                    )

    inference_seconds = time.perf_counter() - inference_started
    ranked_rows = sorted(
        (row for row in image_rows if row["divergence_score"] > 0),
        key=lambda row: (-row["divergence_score"], row["image"]),
    )
    top_rows = ranked_rows[:20]
    render_names: list[str] = []
    for rank, row in enumerate(top_rows, start=1):
        image_path, slf_rows, candidate_rows = detections[row["image"]]
        render_name = f"{rank:02d}_{_safe_filename(row['image'])}.png"
        _render_comparison(image_path, slf_rows, candidate_rows, review_dir / render_name, rank)
        render_names.append(render_name)

    _write_csv(output_dir / "per_image.csv", PER_IMAGE_FIELDS, image_rows)
    _write_csv(output_dir / "per_box.csv", PER_BOX_FIELDS, box_rows)
    _write_review_html(output_dir / "review.html", top_rows, render_names, len(image_rows))
    total_wall_seconds = time.perf_counter() - total_wall_started
    _write_report(
        output_dir / "report.md",
        run_started,
        total_wall_seconds,
        inference_seconds,
        repo_root,
        dataset_info,
        image_rows,
        box_rows,
        top_rows,
        len(render_names),
        args.image_stride,
    )

    print(f"comparison_wall_seconds={total_wall_seconds:.2f}", flush=True)
    print(f"inference_wall_seconds={inference_seconds:.2f}", flush=True)
    print(f"mean_slf_defect_count={np.mean([row['slf_defect_count'] for row in image_rows]):.6f}", flush=True)
    print(
        f"mean_candidate_defect_count={np.mean([row['candidate_defect_count'] for row in image_rows]):.6f}",
        flush=True,
    )
    print(f"slf_detection_rate={sum(row['slf_defect_count'] > 0 for row in image_rows) / len(image_rows):.6%}", flush=True)
    print(
        f"candidate_detection_rate={sum(row['candidate_defect_count'] > 0 for row in image_rows) / len(image_rows):.6%}",
        flush=True,
    )
    print(f"review_count={len(render_names)}", flush=True)
    print(f"report={output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    if str(Path(__file__).resolve().parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
