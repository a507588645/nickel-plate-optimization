#!/usr/bin/env python3
"""Train the no-manual L1 OOF candidate on all development images once.

The final run intentionally does not use development images to choose a best
checkpoint.  Its epoch count is fixed from the earlier three-fold OOF run and
the release candidate is a copy of ``last.pt`` at that fixed epoch.
"""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_ANCHOR = ROOT / "dianjienie_ryw18_oofconsensus_v1_dev_only_r1"
DEFAULT_PROJECT = Path(r"E:/PythonProject/models/runs")
DEFAULT_RUNTIME_CONFIG_ROOT = ROOT / "training_runtime_configs"
DEFAULT_SELECTION_SUMMARY = ROOT / "l1_final_epoch_selection_oofconsensusv1_yolo11m_20260809" / "summary.json"
EXPECTED_CLASSES = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]
EXPECTED_MODEL = "yolo11m.pt"
EXPECTED_DATA_VERSION = "oofconsensusv1"
EXPECTED_DEVELOPMENT_IMAGES = 1127
EXPECTED_CANDIDATE_BOXES = 15671


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在 OOF 共识候选的全部开发集上训练一次固定轮数 YOLO11m；"
            "不读取 release-validation，也不用全开发集自身验证选择 checkpoint。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=(EXPECTED_MODEL,), default=EXPECTED_MODEL)
    parser.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--data-version", default=EXPECTED_DATA_VERSION)
    parser.add_argument("--selection-summary", type=Path, default=DEFAULT_SELECTION_SUMMARY)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--runtime-config-root", type=Path, default=DEFAULT_RUNTIME_CONFIG_ROOT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="仅允许等于已冻结的 OOF 轮数选择。")
    parser.add_argument("--dry-run", action="store_true", help="只校验清单并打印配置，不导入或训练 Ultralytics。")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selection(path: Path, requested_epochs: int | None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结训练轮数选择摘要：{path}")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"训练轮数选择摘要无法解析：{path}") from error
    selected = summary.get("selected")
    policy = summary.get("final_training_policy")
    if (
        summary.get("model") != EXPECTED_MODEL
        or summary.get("data_version") != EXPECTED_DATA_VERSION
        or summary.get("development_only") is not True
        or not isinstance(selected, dict)
        or not isinstance(policy, dict)
        or not isinstance(selected.get("epoch"), int)
        or policy.get("epochs") != selected["epoch"]
        or policy.get("validation") is not False
        or policy.get("patience") != 0
    ):
        raise ValueError("训练轮数选择摘要不是预期的 OOF YOLO11m 固定训练策略")
    epochs = selected["epoch"]
    if epochs <= 0:
        raise ValueError("冻结训练轮数必须为正数")
    if requested_epochs is not None and requested_epochs != epochs:
        raise ValueError(f"--epochs={requested_epochs} 与已冻结轮数 {epochs} 不一致")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "selected_epoch": epochs,
        "selected_mean_map50_95": selected.get("mean_map50_95"),
        "policy": policy,
    }


def read_classes(anchor_root: Path) -> None:
    classes_path = anchor_root / "classes.txt"
    if not classes_path.is_file():
        raise FileNotFoundError(f"缺少 classes.txt：{classes_path}")
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if classes != EXPECTED_CLASSES:
        raise ValueError("候选开发集的类别顺序不是预期 RYW18")


def read_split_paths(list_path: Path, anchor_root: Path) -> list[Path]:
    if not list_path.is_file():
        raise FileNotFoundError(f"缺少固定 Fold 清单：{list_path}")
    paths = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        # Splits may retain the source host's absolute paths.  Their basename
        # identifies the portable hard-link view without resolving a symlink.
        image = anchor_root / "images" / PurePosixPath(value.replace("\\", "/")).name
        if not image.is_file():
            raise FileNotFoundError(f"Fold 清单图片不存在：{image}")
        paths.append(image.absolute())
    if not paths:
        raise ValueError(f"Fold 清单为空：{list_path}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"Fold 清单包含重复图片：{list_path}")
    return paths


def validate_labels(paths: list[Path], anchor_root: Path) -> int:
    labels_root = anchor_root / "labels"
    count = 0
    for image in paths:
        label = labels_root / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"候选开发集缺少标签：{label}")
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"标签字段数错误：{label}:{line_number}")
            try:
                class_id = int(fields[0])
                x, y, width, height = (float(value) for value in fields[1:])
            except ValueError as error:
                raise ValueError(f"标签含非数值字段：{label}:{line_number}") from error
            if not 0 <= class_id < len(EXPECTED_CLASSES):
                raise ValueError(f"标签类别越界：{label}:{line_number}")
            if not all(math.isfinite(value) for value in (x, y, width, height)):
                raise ValueError(f"标签坐标非有限数：{label}:{line_number}")
            if width <= 0 or height <= 0 or width > 1 or height > 1:
                raise ValueError(f"标签宽高非法：{label}:{line_number}")
            epsilon = 1e-9
            if x - width / 2 < -epsilon or x + width / 2 > 1 + epsilon or y - height / 2 < -epsilon or y + height / 2 > 1 + epsilon:
                raise ValueError(f"标签框越界：{label}:{line_number}")
            count += 1
    return count


def collect_all_development(anchor_root: Path) -> tuple[list[Path], dict[str, int]]:
    if (anchor_root / "splits" / "test.txt").exists():
        raise ValueError("最终候选训练只接受 development-only 视图，不能传入含 release-validation 的锚点")
    fold_train: set[Path] = set()
    fold_val: set[Path] = set()
    for fold in (1, 2, 3):
        fold_root = anchor_root / "splits" / "folds" / f"fold_{fold:02d}"
        train = set(read_split_paths(fold_root / "train.txt", anchor_root))
        val = set(read_split_paths(fold_root / "val.txt", anchor_root))
        if train & val:
            raise ValueError(f"Fold {fold} 的 train/val 存在交集")
        fold_train |= train
        fold_val |= val
    if fold_train != fold_val:
        raise ValueError("三个 Fold 的 train 并集与 val 并集不一致，拒绝生成全开发集训练")
    paths = sorted(fold_val, key=lambda path: path.name)
    image_names = {path.name for path in paths}
    all_images = {path.name for path in (anchor_root / "images").glob("*.png")}
    if image_names != all_images:
        raise ValueError("Fold 并集与候选开发集 images/ 不一致")
    if len(paths) != EXPECTED_DEVELOPMENT_IMAGES:
        raise ValueError(f"开发集图片数为 {len(paths)}，预期 {EXPECTED_DEVELOPMENT_IMAGES}")
    boxes = validate_labels(paths, anchor_root)
    if boxes != EXPECTED_CANDIDATE_BOXES:
        raise ValueError(f"候选标签框数为 {boxes}，预期 {EXPECTED_CANDIDATE_BOXES}")
    return paths, {"development_images": len(paths), "candidate_boxes": boxes}


def make_run_name(args: argparse.Namespace, selected_epoch: int) -> str:
    return (
        f"dianjienie_medium_yolo11m_{args.imgsz}_kaggle_seed{args.seed}_cm0_"
        f"{args.data_version}_finaldev_ep{selected_epoch:03d}"
    )


def write_path_list(path: Path, paths: list[Path]) -> None:
    path.write_text("\n".join(item.as_posix() for item in paths) + "\n", encoding="utf-8")


def training_kwargs(data_yaml: Path, run_name: str, args: argparse.Namespace, epochs: int) -> dict[str, Any]:
    return {
        "data": str(data_yaml),
        "epochs": epochs,
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
        "patience": 0,
        "val": False,
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


def prepare_runtime(
    anchor_root: Path,
    paths: list[Path],
    run_name: str,
    runtime_root: Path,
    config: dict[str, Any],
    selection_summary: Path,
) -> Path:
    runtime_dir = runtime_root / run_name
    if runtime_dir.exists():
        raise FileExistsError(f"运行时配置目录已存在，拒绝覆盖：{runtime_dir}")
    runtime_dir.mkdir(parents=True)
    train_list = runtime_dir / "train.txt"
    val_list = runtime_dir / "val.txt"
    # The identical val list is required by Ultralytics' data parser, but
    # training has val=False and release selection uses last.pt, not best.pt.
    write_path_list(train_list, paths)
    write_path_list(val_list, paths)
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(EXPECTED_CLASSES))
    data_yaml = runtime_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f'path: "{anchor_root.as_posix()}"',
                f'train: "{train_list.as_posix()}"',
                f'val: "{val_list.as_posix()}"',
                "",
                "nc: 18",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    (runtime_dir / "requested_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(selection_summary, runtime_dir / "epoch_selection_summary.json")
    return data_yaml


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


def write_readme(save_dir: Path, config: dict[str, Any], environment: dict[str, str], release_weight: dict[str, Any]) -> None:
    training = config["training"]
    content = f"""# L1 OOF 无人工最终候选训练

## 数据与边界

| 项 | 值 |
| --- | --- |
| 数据版本 | `{config['data_version']}` |
| 训练图片 / 候选框 | {config['split_counts']['train']} / {config['candidate_boxes']} |
| 训练范围 | 仅 1,127 张开发图；不包含、不读取 `release-validation` |
| 模型 | `{config['model']}` |
| 固定训练轮数 | {training['epochs']} |
| 选择依据 | `dataset_config/epoch_selection_summary.json` |

## checkpoint 规则

- `weights/final_fixed_epoch.pt` 是固定第 {training['epochs']} 个 epoch 的 `last.pt` 副本，SHA-256 为 `{release_weight['sha256']}`。
- `val=false`、`patience=0`，不以全开发集自身验证选择 epoch 或选择发布权重。
- Ultralytics 在最终 epoch 可能因框架收尾写出 `best.pt`；它不是本候选的 release 权重。

## 固定训练配置

| imgsz | batch | optimizer | seed | resume | close_mosaic |
| ---: | ---: | --- | ---: | --- | ---: |
| {training['imgsz']} | {training['batch']} | `{training['optimizer']}` | {training['seed']} | {training['resume']} | {training['close_mosaic']} |

## 限制

- 本 run 是 `auto_only` 候选：OOF 双模型共识和历史标签自动非回退闸门支持继续训练，但不等于逐框人工真值。
- `results.csv` 中若出现全开发集验证指标，仅为框架运行日志，不用于选择模型或作为发布性能。
- 后续仅对历史 `release-validation` 自动验证 `final_fixed_epoch.pt`，然后执行 ONNX 一致性检查。

环境：Python `{environment['python']}`；Torch `{environment['torch']}`；CUDA `{environment['cuda']}`；Ultralytics `{environment['ultralytics']}`；GPU `{environment['gpu']}`。
"""
    (save_dir / "README.md").write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.data_version != EXPECTED_DATA_VERSION:
        raise ValueError(f"--data-version 必须为 {EXPECTED_DATA_VERSION}")
    if args.imgsz != 640:
        raise ValueError("本次最终候选训练固定 imgsz=640；高分辨率另走研究闸门")
    if args.batch <= 0 or args.workers < 0:
        raise ValueError("--batch 必须为正，--workers 不能为负")
    anchor_root = args.anchor_root.resolve()
    args.project = args.project.resolve()
    args.runtime_config_root = args.runtime_config_root.resolve()
    selection_summary = args.selection_summary.resolve()
    selection = load_selection(selection_summary, args.epochs)
    read_classes(anchor_root)
    paths, counts = collect_all_development(anchor_root)
    run_name = make_run_name(args, selection["selected_epoch"])
    expected_run_dir = args.project / run_name
    if expected_run_dir.exists() and not args.dry_run:
        raise FileExistsError(f"训练输出已存在，拒绝覆盖：{expected_run_dir}")
    runtime_data_yaml = args.runtime_config_root / run_name / "data.yaml"
    config = {
        "model": args.model,
        "run_name": run_name,
        "anchor_root": str(anchor_root),
        "data_version": args.data_version,
        "development_only": True,
        "final_development_training": True,
        "fold": None,
        "split_counts": {"train": counts["development_images"], "val_same_as_train_for_parser": counts["development_images"], "test_frozen": None},
        "candidate_boxes": counts["candidate_boxes"],
        "release_validation_access": "not read by this training run",
        "runtime_data_yaml": str(runtime_data_yaml),
        "expected_run_dir": str(expected_run_dir),
        "epoch_selection": selection,
        "training": training_kwargs(runtime_data_yaml, run_name, args, selection["selected_epoch"]),
    }
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError("缺少 ultralytics；请在 Kaggle GPU 训练环境运行") from error
    runtime_data_yaml = prepare_runtime(
        anchor_root, paths, run_name, args.runtime_config_root, config, selection_summary
    )
    model = YOLO(args.model)
    model.train(**config["training"])
    save_dir = Path(model.trainer.save_dir)
    last_weight = save_dir / "weights" / "last.pt"
    if not last_weight.is_file() or last_weight.stat().st_size == 0:
        raise FileNotFoundError("固定训练完成后缺少非空 weights/last.pt")
    final_weight = save_dir / "weights" / "final_fixed_epoch.pt"
    shutil.copy2(last_weight, final_weight)
    release_weight = {
        "source": "weights/last.pt",
        "release_weight": "weights/final_fixed_epoch.pt",
        "fixed_epoch": selection["selected_epoch"],
        "sha256": sha256_file(final_weight),
    }
    (save_dir / "release_weight.json").write_text(
        json.dumps(release_weight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(runtime_data_yaml.parent / "requested_config.json", save_dir / "requested_config.json")
    shutil.copytree(runtime_data_yaml.parent, save_dir / "dataset_config", dirs_exist_ok=False)
    environment = environment_record()
    (save_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(save_dir, config, environment, release_weight)
    print(f"[done] {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
