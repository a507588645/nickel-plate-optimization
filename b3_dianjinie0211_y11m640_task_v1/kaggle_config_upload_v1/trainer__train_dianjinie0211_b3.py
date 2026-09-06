#!/usr/bin/env python3
"""Train B3: YOLO11m@640 on the dianjinie 5-class Dianjinie0211 split.

Route-B3 trainer (检测切换与等级提升路线_20260904.md §3 B3).  Protocol is the
same fair-training baseline as L1 (seed0 / 300ep / batch8 / SGD / patience100).
The script never mutates the source dataset: it only reads the images/labels
mounted at ``--dataset-root`` and writes run evidence under ``--project``.

Split lists are plain text files with one dataset-root-relative image path per
line (the path must contain an ``images`` component so Ultralytics can derive
the paired ``labels`` path, e.g. ``images/train/x.png`` or ``images/x.png``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path(r"E:/PythonProject/datasets/Dianjinie0211")
DEFAULT_PROJECT = Path(r"E:/PythonProject/models/runs")
DEFAULT_RUNTIME_CONFIG_ROOT = ROOT / "training_runtime_configs"
EXPECTED_CLASSES = ["waterstain", "waterblock", "colorstain", "nodule", "bubble"]
MODEL_SPECS = {"yolo11m.pt": {"size": "medium", "run_model": "yolo11m"}}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 Dianjinie0211 电积镍 5 类固定划分上训练 B3 YOLO11m@640",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), default="yolo11m.pt")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="含 images/ 与 labels/ 的数据集根目录（只读）")
    parser.add_argument("--train-list", type=Path, required=True,
                        help="训练清单：每行一条相对 --dataset-root 的图片路径，需含 images 分量")
    parser.add_argument("--val-list", type=Path, required=True,
                        help="验证清单：格式同 --train-list")
    parser.add_argument("--data-version", default="d0211v1",
                        help="写入 run 名与配置的数据版本标识")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--platform", choices=("lab", "kaggle"), default="lab")
    parser.add_argument("--runtime-config-root", type=Path, default=DEFAULT_RUNTIME_CONFIG_ROOT,
                        help="运行时 data.yaml/list 文件目录，不写入 project")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="校验划分与标签并打印配置，不导入或训练 Ultralytics")
    return parser.parse_args(argv)


def read_list(list_path: Path) -> list[str]:
    if not list_path.is_file():
        raise FileNotFoundError(f"缺少划分清单：{list_path}")
    lines = [line.strip().replace("\\", "/") for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"划分清单为空：{list_path}")
    if len(set(lines)) != len(lines):
        raise ValueError(f"划分清单存在重复条目：{list_path}")
    for line in lines:
        if line.startswith("/") or ".." in Path(line).parts:
            raise ValueError(f"清单路径必须是相对路径且不得上跳：{line}")
        if "images" not in Path(line).parts:
            raise ValueError(f"清单路径必须包含 images 分量以便推导 labels 路径：{line}")
    return lines


def image_to_label(dataset_root: Path, relative_image: str) -> Path:
    """Mirror Ultralytics' img2label_paths: swap the last images/ for labels/."""
    image_path = dataset_root / relative_image
    label_relative = re.sub(r"(?i)(.*)images/", r"\1labels/", relative_image, count=1)
    label_path = (dataset_root / label_relative).with_suffix(".txt")
    if not label_path.is_file():
        raise FileNotFoundError(f"图片缺少对应标签：{image_path} -> {label_path}")
    return label_path


def validate_labels(dataset_root: Path, relative_images: list[str]) -> tuple[int, dict[str, int]]:
    """Reject malformed labels before a costly GPU job starts."""
    class_counter: dict[str, int] = {}
    box_count = 0
    seen_labels: set[Path] = set()
    for relative_image in relative_images:
        image_path = dataset_root / relative_image
        if not image_path.is_file():
            raise FileNotFoundError(f"清单图片不存在：{image_path}")
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"清单包含非图片文件：{image_path}")
        label_path = image_to_label(dataset_root, relative_image)
        if label_path in seen_labels:
            continue
        seen_labels.add(label_path)
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"标签字段数错误：{label_path}:{line_number}")
            try:
                class_id = int(fields[0])
                x, y, width, height = (float(value) for value in fields[1:])
            except ValueError as error:
                raise ValueError(f"标签含非数值字段：{label_path}:{line_number}") from error
            if not 0 <= class_id < len(EXPECTED_CLASSES):
                raise ValueError(f"标签类别越界：{label_path}:{line_number}")
            if not all(math.isfinite(value) for value in (x, y, width, height)):
                raise ValueError(f"标签坐标非有限数：{label_path}:{line_number}")
            if not 0 < width <= 1 or not 0 < height <= 1:
                raise ValueError(f"标签宽高非法：{label_path}:{line_number}")
            epsilon = 1e-9
            if (
                x - width / 2 < -epsilon
                or x + width / 2 > 1 + epsilon
                or y - height / 2 < -epsilon
                or y + height / 2 > 1 + epsilon
            ):
                raise ValueError(f"标签框越过图像边界：{label_path}:{line_number}")
            class_counter[EXPECTED_CLASSES[class_id]] = class_counter.get(EXPECTED_CLASSES[class_id], 0) + 1
            box_count += 1
    return box_count, class_counter


def validate_dataset_yaml(dataset_root: Path) -> None:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.is_file():
        return
    content = data_yaml.read_text(encoding="utf-8")
    if f"nc: {len(EXPECTED_CLASSES)}" not in content:
        raise ValueError(f"dataset data.yaml 类别数不是 {len(EXPECTED_CLASSES)}：{data_yaml}")
    for name in EXPECTED_CLASSES:
        if f": {name}" not in content:
            raise ValueError(f"dataset data.yaml 缺少类别 {name}：{data_yaml}")


def make_run_name(model: str, args: argparse.Namespace) -> str:
    spec = MODEL_SPECS[model]
    return (
        f"dianjinie0211_{spec['size']}_{spec['run_model']}_{args.imgsz}_{args.platform}"
        f"_seed{args.seed}_cm0_{args.data_version}"
    )


def training_kwargs(data_yaml: Path, run_name: str, args: argparse.Namespace) -> dict[str, Any]:
    """与 train_dianjienie_ryw18_anchor.py 完全一致的公平训练协议。"""
    return {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(args.project),
        "name": run_name,
        "exist_ok": False,
        "pretrained": True,
        "optimizer": "SGD",
        "seed": args.seed,
        "deterministic": True,
        "patience": args.patience,
        "cache": False,
        "resume": False,
        "close_mosaic": 0,
        "amp": True,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "iou": 0.7,
        "max_det": 300,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "copy_paste_mode": "flip",
        "auto_augment": "randaugment",
        "erasing": 0.4,
        "save_period": -1,
        "plots": True,
    }


def write_path_list(path: Path, paths: list[Path]) -> None:
    path.write_text("\n".join(item.as_posix() for item in paths) + "\n", encoding="utf-8")


def create_runtime_data_yaml(
    dataset_root: Path,
    split_paths: dict[str, list[Path]],
    run_name: str,
    runtime_config_root: Path,
) -> Path:
    """Create host-local list files without modifying the source dataset."""
    runtime_dir = runtime_config_root / run_name
    if runtime_dir.exists():
        raise FileExistsError(f"运行时数据配置已存在，为防止覆盖已拒绝：{runtime_dir}")
    runtime_dir.mkdir(parents=True)
    train_list, val_list = runtime_dir / "train.txt", runtime_dir / "val.txt"
    write_path_list(train_list, split_paths["train"])
    write_path_list(val_list, split_paths["val"])
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(EXPECTED_CLASSES))
    runtime_yaml = runtime_dir / "data.yaml"
    yaml_lines = [
        f'path: "{dataset_root.as_posix()}"',
        f'train: "{train_list.as_posix()}"',
        f'val: "{val_list.as_posix()}"',
        "",
        f"nc: {len(EXPECTED_CLASSES)}",
        "names:",
        names,
        "",
    ]
    runtime_yaml.write_text("\n".join(yaml_lines), encoding="utf-8")
    return runtime_yaml


def prepare_runtime_data_yaml(
    dataset_root: Path,
    split_paths: dict[str, list[Path]],
    run_name: str,
    runtime_config_root: Path,
    config: dict[str, Any],
) -> Path:
    """Create once or safely reuse an identical retry configuration."""
    runtime_dir = runtime_config_root / run_name
    config_path = runtime_dir / "requested_config.json"
    runtime_yaml = runtime_dir / "data.yaml"
    if runtime_dir.exists():
        if not config_path.is_file() or not runtime_yaml.is_file():
            raise FileExistsError(f"运行时配置目录不完整，拒绝覆盖：{runtime_dir}")
        try:
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"运行时配置 JSON 无法解析，拒绝覆盖：{runtime_dir}") from error
        if existing_config != config:
            raise ValueError(f"运行时配置与本次请求不同，拒绝混用：{runtime_dir}")
        return runtime_yaml

    runtime_yaml = create_runtime_data_yaml(dataset_root, split_paths, run_name, runtime_config_root)
    (runtime_yaml.parent / "requested_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return runtime_yaml


def environment_record() -> dict[str, str]:
    import torch
    import ultralytics

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
    }


def last_metrics(results_path: Path) -> dict[str, str]:
    if not results_path.is_file():
        return {"precision": "未找到 results.csv", "recall": "", "map50": "", "map50_95": ""}
    rows = list(csv.DictReader(results_path.open(encoding="utf-8-sig", newline="")))
    if not rows:
        return {"precision": "results.csv 为空", "recall": "", "map50": "", "map50_95": ""}
    row = rows[-1]

    def metric(*candidates: str) -> str:
        for candidate in candidates:
            if candidate in row:
                return row[candidate]
        return "未记录"

    return {
        "precision": metric("metrics/precision(B)", "metrics/precision"),
        "recall": metric("metrics/recall(B)", "metrics/recall"),
        "map50": metric("metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": metric("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }


def write_run_readme(save_dir: Path, config: dict[str, Any], environment: dict[str, str], split_summary: dict[str, Any]) -> None:
    training = config["training"]
    metrics = last_metrics(save_dir / "results.csv")
    model_spec = MODEL_SPECS[config["model"]]
    content = f"""# Dianjinie0211 电积镍 5 类 B3 {model_spec['run_model'].upper()} 实验说明（{save_dir.name}）

## 数据集

| 项 | 值 |
| --- | --- |
| 数据集根目录 | `{config['dataset_root']}` |
| 数据版本 | `{config['data_version']}` |
| 类别 | {', '.join(EXPECTED_CLASSES)} |
| train / val | {config['split_counts']['train']} / {config['split_counts']['val']} |
| 划分来源 | 固定清单（`--train-list` / `--val-list`），非随机生成 |
| 运行时 YAML | `dataset_config/data.yaml` |

## 基础配置

| 参数 | 值 |
| --- | --- |
| model | `{config['model']}` |
| epochs | {training['epochs']} |
| imgsz | {training['imgsz']} |
| batch | {training['batch']} |
| device | `{training['device']}` |
| workers | {training['workers']} |
| optimizer | `{training['optimizer']}` |
| seed | {training['seed']} |
| deterministic | {training['deterministic']} |
| pretrained | {training['pretrained']} |
| resume | {training['resume']} |
| close_mosaic | {training['close_mosaic']} |
| patience | {training['patience']} |
| amp | {training['amp']} |
| iou / max_det | {training['iou']} / {training['max_det']} |

## 优化器与学习率

| lr0 | lrf | momentum | weight_decay | warmup_epochs | warmup_momentum | warmup_bias_lr |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {training['lr0']} | {training['lrf']} | {training['momentum']} | {training['weight_decay']} | {training['warmup_epochs']} | {training['warmup_momentum']} | {training['warmup_bias_lr']} |

## 损失与增强

| box | cls | dfl | hsv_h/s/v | translate | scale | fliplr | mosaic | mixup | cutmix | auto_augment | erasing |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| {training['box']} | {training['cls']} | {training['dfl']} | {training['hsv_h']}/{training['hsv_s']}/{training['hsv_v']} | {training['translate']} | {training['scale']} | {training['fliplr']} | {training['mosaic']} | {training['mixup']} | {training['cutmix']} | `{training['auto_augment']}` | {training['erasing']} |

## 训练结果

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| {metrics['precision']} | {metrics['recall']} | {metrics['map50']} | {metrics['map50_95']} |

## 划分与排除

{split_summary['readme_lines']}

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `weights/best.pt` / `weights/last.pt` | 训练权重 |
| `args.yaml` / `requested_config.json` | Ultralytics 实际参数与本入口请求参数 |
| `dataset_config/` | train/val 清单、data.yaml 与划分摘要 |
| `results.csv`、曲线、混淆矩阵 | 训练和验证证据 |
| `environment.json` | Python、Torch、CUDA、Ultralytics、GPU 环境 |

## 备注

- 该 run 是路线 B3（域内检测器）训练：权重只用于电积镍等级特征提取实验，不导出、不替换部署 ONNX。
- nodule 类在 train 仅 {split_summary['train_class_distribution'].get('nodule', 0)} 框、val 0 框，逐类指标不可评。
- train/val 继承 Dianjinie0211 原始划分（与 slf.pt 同条件），未做按板重划分。
- 环境：Python `{environment['python']}`；Torch `{environment['torch']}`；CUDA `{environment['cuda']}`；Ultralytics `{environment['ultralytics']}`；GPU `{environment['gpu']}`。
"""
    (save_dir / "README.md").write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.data_version):
        raise ValueError("--data-version 只能包含 ASCII 字母、数字、下划线和连字符")
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在：{dataset_root}")
    args.project = args.project.resolve()
    args.runtime_config_root = args.runtime_config_root.resolve()

    validate_dataset_yaml(dataset_root)
    train_relative = read_list(args.train_list)
    val_relative = read_list(args.val_list)
    if set(train_relative) & set(val_relative):
        raise ValueError("train/val 清单存在交集，已拒绝训练")
    train_boxes, train_classes = validate_labels(dataset_root, train_relative)
    val_boxes, val_classes = validate_labels(dataset_root, val_relative)

    run_name = make_run_name(args.model, args)
    expected_run_dir = args.project / run_name
    if expected_run_dir.exists() and not args.dry_run:
        raise FileExistsError(f"训练目录已存在，为防止覆盖已拒绝：{expected_run_dir}")

    # Keep the images component in runtime list entries so Ultralytics can
    # derive the paired labels/ path through the Kaggle directory symlinks.
    split_paths = {
        "train": [(dataset_root / relative).absolute() for relative in train_relative],
        "val": [(dataset_root / relative).absolute() for relative in val_relative],
    }
    runtime_data_yaml = args.runtime_config_root / run_name / "data.yaml"
    config = {
        "model": args.model,
        "run_name": run_name,
        "dataset_root": str(dataset_root),
        "data_version": args.data_version,
        "train_list": str(args.train_list),
        "val_list": str(args.val_list),
        "split_counts": {"train": len(train_relative), "val": len(val_relative)},
        "runtime_data_yaml": str(runtime_data_yaml),
        "expected_run_dir": str(expected_run_dir),
        "training": training_kwargs(runtime_data_yaml, run_name, args),
    }
    split_summary = {
        "train_count": len(train_relative),
        "val_count": len(val_relative),
        "train_box_count": train_boxes,
        "val_box_count": val_boxes,
        "train_class_distribution": train_classes,
        "val_class_distribution": val_classes,
    }
    print(json.dumps({**config, "split_summary": split_summary}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    # A fresh 5-class model starts from its architecture's public pretrained
    # weights, never from any incompatible teacher checkpoint.
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError("缺少 ultralytics；请在训练环境安装后重试") from error

    model = YOLO(args.model)
    runtime_data_yaml = prepare_runtime_data_yaml(
        dataset_root, split_paths, run_name, args.runtime_config_root, config
    )
    model.train(**config["training"])
    save_dir = Path(model.trainer.save_dir)
    shutil.copy2(runtime_data_yaml.parent / "requested_config.json", save_dir / "requested_config.json")
    shutil.copytree(runtime_data_yaml.parent, save_dir / "dataset_config", dirs_exist_ok=False)
    environment = environment_record()
    (save_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    split_summary["readme_lines"] = "\n".join(
        [
            f"- train {len(train_relative)} 图 / {train_boxes} 框；val {len(val_relative)} 图 / {val_boxes} 框。",
            f"- train 类别分布：{json.dumps(train_classes, ensure_ascii=False)}。",
            f"- val 类别分布：{json.dumps(val_classes, ensure_ascii=False)}。",
            f"- 清单：`{args.train_list}` 与 `{args.val_list}`（副本文件与无标签图已在清单生成时排除）。",
        ]
    )
    write_run_readme(save_dir, config, environment, split_summary)
    print(f"[done] {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
