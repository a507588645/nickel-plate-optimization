#!/usr/bin/env python3
"""Re-evaluate one archived L1 run on its fixed local development fold."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在本地固定开发 Fold 上重评已归档 L1 YOLO 权重，不访问冻结测试集。"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="只检查固定 Fold 输入，不创建硬链接或运行验证。")
    return parser.parse_args()


def read_validation_images(val_list: Path, anchor_root: Path) -> list[Path]:
    image_root = (anchor_root / "images").resolve()
    images = [Path(line.strip()).resolve() for line in val_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not images:
        raise ValueError(f"验证清单为空：{val_list}")
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(f"验证图片不存在：{image}")
        try:
            image.relative_to(image_root)
        except ValueError as error:
            raise ValueError(f"验证图片不属于锚点 images：{image}") from error
    if len({image.name for image in images}) != len(images):
        raise ValueError("Fold 验证图片文件名冲突，不能创建扁平硬链接视图")
    return images


def hardlink(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError as error:
        raise RuntimeError(f"无法创建硬链接：{source} -> {target}") from error


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    anchor_root = args.anchor_root.resolve()
    output_root = args.output_root.resolve()
    weights = run_dir / "weights" / "best.pt"
    source_data = run_dir / "dataset_config" / "data.yaml"
    fold_root = anchor_root / "splits" / "folds" / f"fold_{args.fold:02d}"
    val_list = fold_root / "val.txt"
    test_list = anchor_root / "splits" / "test.txt"

    if not weights.is_file() or not source_data.is_file() or not val_list.is_file() or not test_list.is_file():
        raise FileNotFoundError("缺少归档权重、配置或固定 Fold 清单")
    images = read_validation_images(val_list, anchor_root)
    labels_root = (anchor_root / "labels").resolve()
    labels = [labels_root / f"{image.stem}.txt" for image in images]
    missing_labels = [label for label in labels if not label.is_file()]
    if missing_labels:
        raise FileNotFoundError(f"验证标签不存在：{missing_labels[:3]}")

    audit = {
        "scope": "development_fold_validation_only",
        "run_dir": str(run_dir),
        "weights": str(weights),
        "anchor_root": str(anchor_root),
        "fold": args.fold,
        "val_list": str(val_list),
        "val_images": len(images),
        "frozen_test_list_not_evaluated": str(test_list),
        "frozen_test_images_not_evaluated": len(
            [line for line in test_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        ),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seed": args.seed,
    }
    if args.dry_run:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有复评目录：{output_root}")

    source = yaml.safe_load(source_data.read_text(encoding="utf-8"))
    view_root = output_root / "dataset_view"
    view_images = view_root / "images"
    view_labels = view_root / "labels"
    output_root.mkdir(parents=True)
    view_images.mkdir(parents=True)
    view_labels.mkdir(parents=True)
    view_manifest = []
    for image, label in zip(images, labels, strict=True):
        linked_image = view_images / image.name
        linked_label = view_labels / label.name
        hardlink(image, linked_image)
        hardlink(label, linked_label)
        view_manifest.append(
            {
                "source_image": str(image),
                "source_label": str(label),
                "linked_image": str(linked_image),
                "linked_label": str(linked_label),
            }
        )
    view_val_list = view_root / "val.txt"
    view_val_list.write_text("\n".join(str(view_images / image.name) for image in images) + "\n", encoding="utf-8")
    data = {
        "path": str(view_root),
        "train": str(view_val_list),
        "val": str(view_val_list),
        "nc": source["nc"],
        "names": source["names"],
    }
    local_data = output_root / "data_local_reeval.yaml"
    local_data.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output_root / "dataset_view_manifest.json").write_text(
        json.dumps({"audit": audit, "files": view_manifest}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = YOLO(weights)
    results = model.val(
        data=str(local_data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        conf=0.001,
        iou=0.7,
        max_det=300,
        half=False,
        rect=False,
        plots=True,
        project=str(output_root),
        name="ultralytics_val",
        exist_ok=False,
    )

    names = model.names
    metric_positions = {int(class_id): position for position, class_id in enumerate(results.box.ap_class_index)}
    per_class = []
    for index, class_name in names.items():
        class_id = int(index)
        # Per-class target counts belong to DetMetrics, while AP values belong to its box metric.
        instances = int(results.nt_per_class[class_id])
        images_with_class = int(results.nt_per_image[class_id])
        metric_position = metric_positions.get(class_id)
        values: dict[str, object] = {
            "class_id": int(index),
            "class_name": str(class_name),
            "images": images_with_class,
            "instances": instances,
        }
        if metric_position is None:
            if instances:
                raise RuntimeError(f"类别 {class_id} 有 {instances} 个标签但缺少 AP 指标")
            values.update({"status": "not_evaluable_no_ground_truth", "precision": None, "recall": None, "map50": None, "map50_95": None})
        else:
            precision, recall, map50, map50_95 = results.box.class_result(metric_position)
            values.update(
                {
                    "status": "evaluated",
                    "precision": float(precision),
                    "recall": float(recall),
                    "map50": float(map50),
                    "map50_95": float(map50_95),
                }
            )
        per_class.append(values)

    metrics = {
        **audit,
        "split": "val",
        "local_view_root": str(view_root),
        "local_val_list": str(view_val_list),
        "ultralytics": __import__("ultralytics").__version__,
        "torch": torch.__version__,
        "metrics": {
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
        },
        "per_class": per_class,
    }
    (output_root / "metrics_per_class_reeval.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(local_data, output_root / "data.yaml")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
