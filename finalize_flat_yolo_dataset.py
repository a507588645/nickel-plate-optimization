#!/usr/bin/env python3
"""验证并补齐扁平、无划分的 YOLO 伪标签数据集。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


RYW_CLASSES = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="验证扁平 YOLO 数据集并写入配置")
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    image_paths = sorted(p for p in (root / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    label_paths = sorted((root / "labels").glob("*.txt"))
    image_stems = {p.stem for p in image_paths}
    label_stems = {p.stem for p in label_paths}
    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)
    class_counts: Counter[int] = Counter()
    invalid: list[dict[str, object]] = []
    empty_labels = 0
    total_boxes = 0
    for label in label_paths:
        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
        for line_no, line in enumerate(lines, 1):
            fields = line.split()
            if len(fields) != 5:
                invalid.append({"label": str(label), "line": line_no, "reason": f"fields={len(fields)}"})
                continue
            try:
                class_id = int(fields[0])
                coords = [float(value) for value in fields[1:]]
            except ValueError:
                invalid.append({"label": str(label), "line": line_no, "reason": "non_numeric"})
                continue
            if not 0 <= class_id < len(RYW_CLASSES):
                invalid.append({"label": str(label), "line": line_no, "reason": f"class={class_id}"})
            if any(value < 0 or value > 1 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
                invalid.append({"label": str(label), "line": line_no, "reason": "invalid_coordinates"})
            class_counts[class_id] += 1
            total_boxes += 1

    (root / "classes.txt").write_text("\n".join(RYW_CLASSES) + "\n", encoding="utf-8")
    data_yaml = (
        f"path: {root.as_posix()}\n"
        "train: images\n"
        "val: images  # 未划分；仅用于工具识别，禁止据此报告验证指标\n"
        f"nc: {len(RYW_CLASSES)}\n"
        "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(RYW_CLASSES))
    )
    (root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    audit = {
        "layout": "flat_yolo_no_split",
        "images": len(image_paths),
        "labels": len(label_paths),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "empty_labels": empty_labels,
        "total_boxes": total_boxes,
        "invalid_entries": invalid,
        "class_distribution": {RYW_CLASSES[k]: v for k, v in sorted(class_counts.items())},
        "teacher_model": "ryw.pt",
    }
    (root / "label_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path = root / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    metadata.update({
        "label_status": "complete",
        "teacher_model": "ryw.pt",
        "label_format": "YOLO 5-column xywh normalized",
        "images": len(image_paths),
        "labels": len(label_paths),
        "empty_labels": empty_labels,
        "total_boxes": total_boxes,
        "split_status": "none",
    })
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
