#!/usr/bin/env python3
"""Run a reproducible two-stage YOLO11m freeze ablation on the L1 dev view."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


CLASS_NAMES = (
    "wuzi",
    "waterstain",
    "yellowstain",
    "nodule",
    "airhole",
    "bubble",
    "bump",
    "fawu",
    "overfire",
    "NiOH",
    "greenstain",
    "suanwuran",
    "quejiao",
    "watermark",
    "youpao",
    "heibian",
    "mijixiaokeng",
    "fanxitong",
)
EXPECTED_DATA_VERSION = "oofconsensusv1"
EXPECTED_DEVELOPMENT_IMAGES = 1127
EXPECTED_CANDIDATE_BOXES = 15671
DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = DEFAULT_ROOT / "freeze_runs"
DEFAULT_RUNTIME_ROOT = DEFAULT_ROOT / "freeze_runtime_configs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a development-only two-stage YOLO11m freeze experiment. "
            "Stage 1 uses --freeze; stage 2 always unfreezes the full model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("fold", "final"), required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3))
    parser.add_argument("--data-version", default=EXPECTED_DATA_VERSION)
    parser.add_argument("--model", choices=("yolo11m.pt",), default="yolo11m.pt")
    parser.add_argument("--variant", choices=("control", "freeze10"), required=True)
    parser.add_argument("--freeze", type=int, required=True)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=63)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--runtime-config-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--source-file", type=Path, help="Copy this trainer source into the finished run evidence.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the fixed view and print the request only.")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.data_version != EXPECTED_DATA_VERSION:
        raise ValueError(f"--data-version must be {EXPECTED_DATA_VERSION}")
    if args.mode == "fold" and args.fold is None:
        raise ValueError("--mode fold requires --fold")
    if args.mode == "final" and args.fold is not None:
        raise ValueError("--mode final must not receive --fold")
    expected_freeze = 0 if args.variant == "control" else 10
    if args.freeze != expected_freeze:
        raise ValueError(f"Variant {args.variant!r} requires --freeze {expected_freeze}")
    if args.stage1_epochs <= 0 or args.stage2_epochs <= 0:
        raise ValueError("Both training stages must have a positive epoch count")
    if args.stage1_epochs + args.stage2_epochs != 83:
        raise ValueError("The paired experiment fixes the total budget at 83 epochs")
    if args.imgsz != 640 or args.batch != 8:
        raise ValueError("The paired experiment fixes --imgsz 640 and --batch 8")
    if args.seed != 0:
        raise ValueError("The paired experiment fixes --seed 0")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")


def read_classes(anchor_root: Path) -> None:
    classes_path = anchor_root / "classes.txt"
    if not classes_path.is_file():
        raise FileNotFoundError(f"Missing classes file: {classes_path}")
    actual = tuple(line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if actual != CLASS_NAMES:
        raise ValueError("The restored development view does not match the frozen RYW18 class order")


def read_split(list_path: Path, anchor_root: Path) -> list[Path]:
    if not list_path.is_file():
        raise FileNotFoundError(f"Missing fixed split list: {list_path}")
    images: list[Path] = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        # Lists can carry paths from the machine where the view was first built.
        image = anchor_root / "images" / PurePosixPath(value.replace("\\", "/")).name
        if not image.is_file():
            raise FileNotFoundError(f"Split image is missing from the restored view: {image}")
        images.append(image.resolve())
    if not images or len(images) != len(set(images)):
        raise ValueError(f"Split list is empty or contains duplicate images: {list_path}")
    return images


def audit_labels(images: list[Path], anchor_root: Path) -> int:
    count = 0
    for image in images:
        label = anchor_root / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing YOLO label: {label}")
        for line_number, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            fields = raw.split()
            if len(fields) != 5:
                raise ValueError(f"Label must contain five fields: {label}:{line_number}")
            try:
                class_id = int(fields[0])
                x, y, width, height = (float(value) for value in fields[1:])
            except ValueError as error:
                raise ValueError(f"Non-numeric YOLO label: {label}:{line_number}") from error
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(f"YOLO class index is out of range: {label}:{line_number}")
            if not all(math.isfinite(value) for value in (x, y, width, height)):
                raise ValueError(f"YOLO box is non-finite: {label}:{line_number}")
            if width <= 0 or height <= 0 or width > 1 or height > 1:
                raise ValueError(f"YOLO box dimensions are invalid: {label}:{line_number}")
            epsilon = 1e-9
            if x - width / 2 < -epsilon or x + width / 2 > 1 + epsilon:
                raise ValueError(f"YOLO box exceeds horizontal bounds: {label}:{line_number}")
            if y - height / 2 < -epsilon or y + height / 2 > 1 + epsilon:
                raise ValueError(f"YOLO box exceeds vertical bounds: {label}:{line_number}")
            count += 1
    return count


def collect_fold_images(anchor_root: Path, fold: int) -> tuple[list[Path], list[Path]]:
    fold_root = anchor_root / "splits" / "folds" / f"fold_{fold:02d}"
    train = read_split(fold_root / "train.txt", anchor_root)
    val = read_split(fold_root / "val.txt", anchor_root)
    if set(train) & set(val):
        raise ValueError(f"Fold {fold} train and val memberships overlap")
    expected = {1: (751, 376), 2: (752, 375), 3: (751, 376)}[fold]
    if (len(train), len(val)) != expected:
        raise ValueError(f"Fold {fold} count mismatch: actual={(len(train), len(val))}, expected={expected}")
    return train, val


def collect_final_images(anchor_root: Path) -> list[Path]:
    train_union: set[Path] = set()
    val_union: set[Path] = set()
    for fold in (1, 2, 3):
        train, val = collect_fold_images(anchor_root, fold)
        train_union.update(train)
        val_union.update(val)
    if train_union != val_union:
        raise ValueError("The union of fixed OOF train and val lists is inconsistent")
    images = sorted(val_union, key=lambda item: item.name)
    actual_images = {item.resolve() for item in (anchor_root / "images").glob("*.png")}
    if set(images) != actual_images or len(images) != EXPECTED_DEVELOPMENT_IMAGES:
        raise ValueError("The fixed OOF lists do not cover exactly the development images")
    return images


def run_name(args: argparse.Namespace) -> str:
    scope = f"fold{args.fold:02d}" if args.mode == "fold" else "finaldev"
    return (
        "dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_"
        f"{args.data_version}_{args.variant}_s1e{args.stage1_epochs:03d}_s2e{args.stage2_epochs:03d}_{scope}"
    )


def write_path_list(path: Path, images: list[Path]) -> None:
    path.write_text("\n".join(image.as_posix() for image in images) + "\n", encoding="utf-8")


def make_runtime_data(
    anchor_root: Path,
    train_images: list[Path],
    val_images: list[Path],
    name: str,
    runtime_root: Path,
) -> Path:
    runtime = runtime_root / name
    if runtime.exists():
        raise FileExistsError(f"Refusing to overwrite runtime configuration: {runtime}")
    runtime.mkdir(parents=True)
    train_list = runtime / "train.txt"
    val_list = runtime / "val.txt"
    write_path_list(train_list, train_images)
    write_path_list(val_list, val_images)
    names = "\n".join(f"  {index}: {class_name}" for index, class_name in enumerate(CLASS_NAMES))
    data_yaml = runtime / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            (
                f'path: "{anchor_root.as_posix()}"',
                f'train: "{train_list.as_posix()}"',
                f'val: "{val_list.as_posix()}"',
                "",
                "nc: 18",
                "names:",
                names,
                "",
            )
        ),
        encoding="utf-8",
    )
    return data_yaml


def training_kwargs(
    data_yaml: Path,
    name: str,
    args: argparse.Namespace,
    epochs: int,
    freeze: int,
    validation: bool,
) -> dict[str, Any]:
    return {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(args.project),
        "name": name,
        "exist_ok": False,
        "pretrained": True,
        "optimizer": "SGD",
        "seed": args.seed,
        "deterministic": True,
        "patience": 0,
        "val": validation,
        "cache": False,
        "resume": False,
        "close_mosaic": 0,
        "amp": True,
        "freeze": freeze,
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
        # This only suppresses image rendering, avoiding an observed Kaggle callback crash.
        "plots": False,
    }


def environment_payload() -> dict[str, Any]:
    import numpy
    import torch
    import torchvision
    import ultralytics

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "numpy": numpy.__version__,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
    }


def nonempty_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def write_readme(save_dir: Path, request: dict[str, Any]) -> None:
    stage1 = request["training"]["stage1"]
    stage2 = request["training"]["stage2"]
    scope = f"Fold {request['fold']}" if request["fold"] is not None else "all 1,127 development images"
    save_dir.joinpath("README.md").write_text(
        "\n".join(
            (
                "# L1 YOLO11m two-stage freeze experiment",
                "",
                f"- Variant: `{request['variant']}`; Stage-1 freeze: `{stage1['freeze']}`.",
                f"- Scope: `{scope}`; development-only: `{request['development_only']}`.",
                "- Frozen historical release-validation is not mounted, read, or evaluated.",
                f"- Stage 1: {stage1['epochs']} epochs from public `yolo11m.pt`.",
                f"- Stage 2: {stage2['epochs']} epochs from Stage-1 `last.pt`, with `freeze=0` and `resume=false`.",
                "- `weights/best.pt` is the Fold candidate when validation is enabled.",
                "- For finaldev, `weights/final_fixed_epoch.pt` is copied from Stage-2 `last.pt`.",
                "",
            )
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    args.anchor_root = args.anchor_root.resolve()
    args.project = args.project.resolve()
    args.runtime_config_root = args.runtime_config_root.resolve()
    if (args.anchor_root / "splits" / "test.txt").exists():
        raise ValueError("This experiment accepts only the development-only OOF view and must not receive test.txt")
    read_classes(args.anchor_root)

    if args.mode == "fold":
        train_images, val_images = collect_fold_images(args.anchor_root, args.fold)
        audit_images = sorted({*train_images, *val_images}, key=lambda item: item.name)
    else:
        train_images = collect_final_images(args.anchor_root)
        val_images = train_images
        audit_images = train_images
    box_count = audit_labels(audit_images, args.anchor_root)
    if len(audit_images) != EXPECTED_DEVELOPMENT_IMAGES or box_count != EXPECTED_CANDIDATE_BOXES:
        raise ValueError(
            "The freeze experiment requires the exact 1,127-image / 15,671-box OOF development view; "
            f"got images={len(audit_images)}, boxes={box_count}"
        )

    name = run_name(args)
    stage1_name = f"{name}_stage1"
    runtime_yaml = args.runtime_config_root / name / "data.yaml"
    stage1_kwargs = training_kwargs(
        runtime_yaml, stage1_name, args, args.stage1_epochs, args.freeze, args.mode == "fold"
    )
    stage2_kwargs = training_kwargs(
        runtime_yaml, name, args, args.stage2_epochs, 0, args.mode == "fold"
    )
    source = (args.source_file or Path(__file__)).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Trainer source is missing: {source}")
    request: dict[str, Any] = {
        "experiment": "L1 YOLO11m paired two-stage freeze ablation",
        "run_name": name,
        "variant": args.variant,
        "model": args.model,
        "data_version": args.data_version,
        "development_only": True,
        "frozen_test_set": None,
        "release_validation_access": "not read by this experiment",
        "mode": args.mode,
        "fold": args.fold,
        "split_counts": {
            "train": len(train_images),
            "val": len(val_images),
            "test_frozen": None,
            "audited_development_images": len(audit_images),
            "audited_candidate_boxes": box_count,
        },
        "anchor_root": str(args.anchor_root),
        "runtime_data_yaml": str(runtime_yaml),
        "expected_run_dir": str(args.project / name),
        "trainer_source": {"path": str(source), "sha256": sha256_file(source)},
        "training": {"stage1": stage1_kwargs, "stage2": stage2_kwargs},
        "selection": {
            "fold_candidate_weight": "weights/best.pt (stage 2 only)",
            "final_weight": "weights/final_fixed_epoch.pt copied from stage-2 last.pt",
            "total_epochs": args.stage1_epochs + args.stage2_epochs,
            "notes": "The paired control uses the same two-stage optimizer reset and differs only by stage-1 freeze.",
        },
    }
    print(json.dumps(request, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0

    if (args.project / name).exists() or (args.project / stage1_name).exists():
        raise FileExistsError("A stage output already exists; refusing to overwrite a prior experiment")
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError("Ultralytics is unavailable; use the pinned Kaggle GPU runtime") from error

    data_yaml = make_runtime_data(args.anchor_root, train_images, val_images, name, args.runtime_config_root)
    if data_yaml != runtime_yaml:
        raise RuntimeError("Runtime data path unexpectedly changed")
    write_json(data_yaml.parent / "requested_config.json", request)

    stage1_model = YOLO(args.model)
    stage1_model.train(**stage1_kwargs)
    stage1_dir = Path(stage1_model.trainer.save_dir)
    stage1_last = stage1_dir / "weights" / "last.pt"
    stage1_best = stage1_dir / "weights" / "best.pt"
    nonempty_file(stage1_last, "stage-1 last.pt")
    nonempty_file(stage1_best, "stage-1 best.pt")

    stage2_model = YOLO(str(stage1_last))
    stage2_model.train(**stage2_kwargs)
    save_dir = Path(stage2_model.trainer.save_dir)
    best_weight = save_dir / "weights" / "best.pt"
    last_weight = save_dir / "weights" / "last.pt"
    for artifact, label in (
        (best_weight, "stage-2 best.pt"),
        (last_weight, "stage-2 last.pt"),
        (save_dir / "args.yaml", "stage-2 args.yaml"),
        (save_dir / "results.csv", "stage-2 results.csv"),
    ):
        nonempty_file(artifact, label)

    request["artifacts"] = {
        "stage1_last_pt_sha256": sha256_file(stage1_last),
        "stage1_best_pt_sha256": sha256_file(stage1_best),
        "stage2_last_pt_sha256": sha256_file(last_weight),
        "stage2_best_pt_sha256": sha256_file(best_weight),
    }
    write_json(save_dir / "requested_config.json", request)
    shutil.copytree(data_yaml.parent, save_dir / "dataset_config", dirs_exist_ok=False)
    shutil.copytree(stage1_dir, save_dir / "freeze_stage1", dirs_exist_ok=False)
    source_dir = save_dir / "source"
    source_dir.mkdir()
    shutil.copy2(source, source_dir / source.name)
    write_json(save_dir / "environment.json", environment_payload())
    write_readme(save_dir, request)

    if args.mode == "final":
        final_weight = save_dir / "weights" / "final_fixed_epoch.pt"
        shutil.copy2(last_weight, final_weight)
        write_json(
            save_dir / "release_weight.json",
            {
                "source": "weights/last.pt",
                "sha256": sha256_file(final_weight),
                "fixed_total_epochs": args.stage1_epochs + args.stage2_epochs,
                "release_validation_access": "not read",
            },
        )
        write_json(
            save_dir / "dataset_config" / "epoch_selection_summary.json",
            {
                "purpose": "pre-registered fixed total epoch budget for the freeze10 finaldev candidate",
                "total_epochs": args.stage1_epochs + args.stage2_epochs,
                "stage1_epochs": args.stage1_epochs,
                "stage2_epochs": args.stage2_epochs,
                "selection_source": "paired Fold 1 freeze ablation gate",
            },
        )
    print(json.dumps({"run_dir": str(save_dir), "run_name": name}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
