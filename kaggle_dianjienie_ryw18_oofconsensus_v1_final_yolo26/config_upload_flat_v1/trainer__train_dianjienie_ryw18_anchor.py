#!/usr/bin/env python3
"""Train one fair YOLO11m or YOLO26m fold on the dianjienie RYW18 anchor.

The script deliberately refuses the original dianjienie split.  It uses only
the fixed group-aware fold files in dianjienie_ryw18_anchor_v1 and never uses
the frozen test set during training.
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
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_ANCHOR = Path(r"E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1")
DEFAULT_PROJECT = Path(r"E:/PythonProject/models/runs")
DEFAULT_RUNTIME_CONFIG_ROOT = ROOT / "training_runtime_configs"
EXPECTED_CLASSES = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]
MODEL_SPECS = {
    "yolo11m.pt": {"size": "medium", "run_model": "yolo11m"},
    "yolo26m.pt": {"size": "medium", "run_model": "yolo26m"},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 dianjienie 18 类历史锚点上训练一个固定 Fold 的 YOLO11m 或 YOLO26m",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument(
        "--data-version",
        default="anchorv1",
        help="写入 run 名与配置的数据版本标识；默认保持历史 anchorv1 命名。",
    )
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="只读取和训练固定开发集 train/val，不读取或写出冻结 test 清单。",
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--platform", choices=("lab", "kaggle"), default="lab")
    parser.add_argument("--runtime-config-root", type=Path, default=DEFAULT_RUNTIME_CONFIG_ROOT,
                        help="运行时 data.yaml/list 文件目录，不写入 models/runs")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="校验 Fold 和打印配置，但不导入或训练 Ultralytics")
    return parser.parse_args(argv)


def resolve_split_paths(list_path: Path, anchor_root: Path) -> list[Path]:
    """Rebase stored anchor paths so the dataset is portable to another host."""
    if not list_path.is_file():
        raise FileNotFoundError(f"缺少划分清单：{list_path}")
    source_lines = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # The immutable anchor was built on Windows. Normalize separators before
    # extracting the basename so the same split files work on Linux/Kaggle.
    paths = [
        anchor_root / "images" / PurePosixPath(source_line.replace("\\", "/")).name
        for source_line in source_lines
    ]
    if not paths:
        raise ValueError(f"划分清单为空：{list_path}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"锚点图片不存在或清单文件名不匹配，例如：{missing[0]}")
    missing_labels = [anchor_root / "labels" / f"{path.stem}.txt" for path in paths]
    missing_labels = [path for path in missing_labels if not path.is_file()]
    if missing_labels:
        raise FileNotFoundError(f"锚点图片缺少对应标签，例如：{missing_labels[0]}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"划分清单存在重复图片：{list_path}")
    # Keep the ``images/`` component in the runtime list.  Resolving a Kaggle
    # symbolic link here would turn the path into a flat uploaded filename,
    # preventing Ultralytics from deriving the paired ``labels/`` location.
    return [path.absolute() for path in paths]


def validate_yolo_labels(paths: list[Path], anchor_root: Path) -> int:
    """Reject malformed labels before a costly GPU job starts."""
    seen_labels: set[Path] = set()
    box_count = 0
    for image_path in paths:
        label_path = anchor_root / "labels" / f"{image_path.stem}.txt"
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
            if width <= 0 or height <= 0 or width > 1 or height > 1:
                raise ValueError(f"标签宽高非法：{label_path}:{line_number}")
            epsilon = 1e-9
            if x - width / 2 < -epsilon or x + width / 2 > 1 + epsilon or y - height / 2 < -epsilon or y + height / 2 > 1 + epsilon:
                raise ValueError(f"标签框越过图像边界：{label_path}:{line_number}")
            box_count += 1
    return box_count


def validate_fold(
    anchor_root: Path, fold: int, development_only: bool = False
) -> tuple[Path, dict[str, list[Path]]]:
    fold_root = anchor_root / "splits" / "folds" / f"fold_{fold:02d}"
    data_yaml = fold_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"不存在 Fold 配置：{data_yaml}")
    content = data_yaml.read_text(encoding="utf-8")
    required_lines = ["nc: 18", *[f"  {index}: {name}" for index, name in enumerate(EXPECTED_CLASSES)]]
    missing_lines = [line for line in required_lines if line not in content]
    if missing_lines:
        raise ValueError(f"Fold 类别定义不是预期 RYW18：{missing_lines[0]}")

    train = resolve_split_paths(fold_root / "train.txt", anchor_root)
    val = resolve_split_paths(fold_root / "val.txt", anchor_root)
    train_set, val_set = set(train), set(val)
    if train_set & val_set:
        raise ValueError("Fold 划分存在 train/val/test 交集，已拒绝训练")
    split_paths = {"train": train, "val": val}
    if not development_only:
        test = resolve_split_paths(anchor_root / "splits" / "test.txt", anchor_root)
        test_set = set(test)
        if train_set & test_set or val_set & test_set:
            raise ValueError("Fold 划分存在 train/val/test 交集，已拒绝训练")
        split_paths["test"] = test
    validate_yolo_labels([path for paths in split_paths.values() for path in paths], anchor_root)
    return data_yaml, split_paths


def make_run_name(model: str, fold: int, args: argparse.Namespace) -> str:
    spec = MODEL_SPECS[model]
    return (
        f"dianjienie_{spec['size']}_{spec['run_model']}_{args.imgsz}_{args.platform}"
        f"_seed{args.seed}_cm0_{args.data_version}_fold{fold:02d}"
    )


def training_kwargs(data_yaml: Path, run_name: str, args: argparse.Namespace) -> dict[str, Any]:
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
    anchor_root: Path,
    fold_data_yaml: Path,
    split_paths: dict[str, list[Path]],
    run_name: str,
    runtime_config_root: Path,
) -> Path:
    """Create host-local list files without modifying the immutable anchor."""
    runtime_dir = runtime_config_root / run_name
    if runtime_dir.exists():
        raise FileExistsError(f"运行时数据配置已存在，为防止覆盖已拒绝：{runtime_dir}")
    runtime_dir.mkdir(parents=True)
    train_list, val_list = runtime_dir / "train.txt", runtime_dir / "val.txt"
    write_path_list(train_list, split_paths["train"])
    write_path_list(val_list, split_paths["val"])
    test_list = runtime_dir / "test.txt" if "test" in split_paths else None
    if test_list is not None:
        write_path_list(test_list, split_paths["test"])
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(EXPECTED_CLASSES))
    runtime_yaml = runtime_dir / "data.yaml"
    yaml_lines = [
        f'path: "{anchor_root.as_posix()}"',
        f'train: "{train_list.as_posix()}"',
        f'val: "{val_list.as_posix()}"',
    ]
    if test_list is not None:
        yaml_lines.append(f'test: "{test_list.as_posix()}"')
    yaml_lines.extend(["", "nc: 18", "names:", names, ""])
    runtime_yaml.write_text("\n".join(yaml_lines), encoding="utf-8")
    shutil.copy2(fold_data_yaml, runtime_dir / "source_fold_data.yaml")
    return runtime_yaml


def prepare_runtime_data_yaml(
    anchor_root: Path,
    fold_data_yaml: Path,
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
            raise ValueError(f"运行时配置 JSON 无法解析，拒绝覆盖：{config_path}") from error
        if existing_config != config:
            raise ValueError(f"运行时配置与本次请求不同，拒绝混用：{runtime_dir}")
        return runtime_yaml

    runtime_yaml = create_runtime_data_yaml(
        anchor_root, fold_data_yaml, split_paths, run_name, runtime_config_root
    )
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


def write_run_readme(save_dir: Path, config: dict[str, Any], environment: dict[str, str]) -> None:
    training = config["training"]
    metrics = last_metrics(save_dir / "results.csv")
    model_spec = MODEL_SPECS[config["model"]]
    test_display = "未读取（development_only）" if config["development_only"] else config["split_counts"]["test_frozen"]
    list_description = "train/val" if config["development_only"] else "train/val/test"
    test_note = (
        "此 run 明确不读取、不写出冻结 `test.txt`。"
        if config["development_only"]
        else "此 run 只使用开发集 Fold；`test.txt` 未参与训练、早停或模型选择。"
    )
    content = f"""# dianjienie Medium {model_spec['run_model'].upper()} 实验说明（{save_dir.name})

## 数据集

| 项 | 值 |
| --- | --- |
| 数据锚点 | `{config['anchor_root']}` |
| 数据版本 | `{config['data_version']}` |
| 类别数 | 18 |
| Fold | {config['fold']} |
| train / val / 冻结 test | {config['split_counts']['train']} / {config['split_counts']['val']} / {test_display} |
| 源 Fold YAML | `{config['source_fold_data_yaml']}` |
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

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `weights/best.pt` / `weights/last.pt` | 训练权重 |
| `args.yaml` / `requested_config.json` | Ultralytics 实际参数与本入口请求参数 |
| `dataset_config/` | 本机重建的 {list_description} 清单与 data.yaml |
| `results.csv`、曲线、混淆矩阵 | 训练和开发验证证据 |
| `environment.json` | Python、Torch、CUDA、Ultralytics、GPU 环境 |

## 备注

- {test_note}
- RYW18 保留 `youpao`、`fanxitong`，但两类当前零监督，不可评；`heibian`、`mijixiaokeng` 极稀少。
- 该 run 属于 `{config['data_version']}` 的同条件 YOLO11m/YOLO26m 组。不可与旧 5 类数据集指标直接排名。
- 环境：Python `{environment['python']}`；Torch `{environment['torch']}`；CUDA `{environment['cuda']}`；Ultralytics `{environment['ultralytics']}`；GPU `{environment['gpu']}`。
"""
    (save_dir / "README.md").write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.data_version):
        raise ValueError("--data-version 只能包含 ASCII 字母、数字、下划线和连字符")
    anchor_root = args.anchor_root.resolve()
    args.project = args.project.resolve()
    args.runtime_config_root = args.runtime_config_root.resolve()
    fold_data_yaml, split_paths = validate_fold(anchor_root, args.fold, args.development_only)
    run_name = make_run_name(args.model, args.fold, args)
    expected_run_dir = args.project / run_name
    if expected_run_dir.exists() and not args.dry_run:
        raise FileExistsError(f"训练目录已存在，为防止覆盖已拒绝：{expected_run_dir}")
    runtime_data_yaml = args.runtime_config_root / run_name / "data.yaml"
    split_counts = {
        "train": len(split_paths["train"]),
        "val": len(split_paths["val"]),
        "test_frozen": None if args.development_only else len(split_paths["test"]),
    }
    config = {
        "model": args.model,
        "run_name": run_name,
        "anchor_root": str(anchor_root),
        "data_version": args.data_version,
        "development_only": args.development_only,
        "fold": args.fold,
        "split_counts": split_counts,
        "frozen_test_set": None if args.development_only else str(anchor_root / "splits" / "test.txt"),
        "source_fold_data_yaml": str(fold_data_yaml),
        "runtime_data_yaml": str(runtime_data_yaml),
        "expected_run_dir": str(expected_run_dir),
        "training": training_kwargs(runtime_data_yaml, run_name, args),
    }
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    # A fresh 18-class model starts from its architecture's public pretrained
    # weights, never from the incompatible YOLOv5 ryw.pt teacher checkpoint.
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError("缺少 ultralytics；请在训练环境安装后重试") from error

    model = YOLO(args.model)
    runtime_data_yaml = prepare_runtime_data_yaml(
        anchor_root, fold_data_yaml, split_paths, run_name, args.runtime_config_root, config
    )
    model.train(**config["training"])
    save_dir = Path(model.trainer.save_dir)
    shutil.copy2(runtime_data_yaml.parent / "requested_config.json", save_dir / "requested_config.json")
    shutil.copytree(runtime_data_yaml.parent, save_dir / "dataset_config", dirs_exist_ok=False)
    environment = environment_record()
    (save_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_readme(save_dir, config, environment)
    print(f"[done] {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
