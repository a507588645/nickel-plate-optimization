#!/usr/bin/env python3
"""Prepare, but never submit, the E2 Mosaic=0 L1 Kaggle training package.

E2 is a full-image fine-tune from the archived Fold 1 YOLO11m checkpoint. It
does not reuse the tile kernel generator because tile generation, mixed train
views, and cross-tile evaluation would introduce unrelated variables.  The
generated Kaggle script restores the existing development-only config Dataset,
checks the checkpoint SHA-256, starts a new `resume=False` run with
`mosaic=0.0` as the only model-training augmentation change, and evaluates it
with one full-image crop per source image using the established original-image
scorer. `save_period=1` is an operational checkpoint-recovery policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import textwrap
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any


ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = ROOT / "l1_tile_pipeline.py"
IMAGE_DATASET_REF = "lly3323w/dianjienie"
CONFIG_DATASET_REF = "lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config"
DEFAULT_OWNER = "lly3323w"
DEFAULT_SLUG = "l1-ryw18-y11m-oof-v1-nomosaic-ft30-f01"
DEFAULT_TITLE = "L1 RYW18 YOLO11m OOF v1 Mosaic 0 Fine Tune 30 Epoch Fold01"
DEFAULT_DATASET_SLUG = "dianjienie-ryw18-oof-y11m-fold01-640-nomosaic-base"
DEFAULT_DATASET_TITLE = "RYW18 OOF YOLO11m Fold01 640 Mosaic0 Fine-Tune Base"
BASELINE_FULL_MAP50_95_REFERENCE = 0.168001
DEFAULT_WEIGHT = (
    Path(r"E:/PythonProject/models/runs")
    / "dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold01"
    / "weights"
    / "best.pt"
)
EXPECTED_WEIGHT_SHA256 = "b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce"
CHECKPOINT_NAME = "baseline_fold01_640_best.pt"
EXPECTED_CLASSES = (
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


KERNEL_TEMPLATE = Template(
    r'''#!/usr/bin/env python3
"""Run E2: a development-only Fold 1 full-image Mosaic=0 fine-tune."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


INPUT_ROOT = Path("/kaggle/input")
RUN_NAME = "$run_name"
CHECKPOINT_NAME = "$checkpoint_name"
EXPECTED_CHECKPOINT_SHA256 = "$expected_checkpoint_sha256"
PIPELINE_SOURCE = $pipeline_source
EXPECTED_CLASSES = $expected_classes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mounted_roots() -> list[Path]:
    roots = [path for path in INPUT_ROOT.iterdir() if path.is_dir()]
    scoped = INPUT_ROOT / "datasets"
    if scoped.is_dir():
        roots.extend(path for path in scoped.rglob("*") if path.is_dir())
    return roots


def inventory() -> dict[str, list[str]]:
    return {str(root.relative_to(INPUT_ROOT)): sorted(path.name for path in root.iterdir())[:12] for root in mounted_roots()}


def safe_relative(value: str) -> Path:
    candidate = PurePosixPath(value)
    if "\\" in value or candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"Unsafe config relative path: {value!r}")
    return Path(*candidate.parts)


def resolve_config_payload() -> tuple[Path, Path]:
    candidates = [root for root in mounted_roots() if (root / "config_archive_manifest.json").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate development-only config Dataset: {inventory()}")
    package_root = candidates[0]
    archive_manifest = json.loads((package_root / "config_archive_manifest.json").read_text(encoding="utf-8"))
    archive_name = archive_manifest.get("config_archive")
    payload_files = archive_manifest.get("payload_files")
    source_manifest_sha256 = archive_manifest.get("source_manifest_sha256")
    if not isinstance(archive_name, str) or not isinstance(payload_files, list) or not isinstance(source_manifest_sha256, str):
        raise ValueError("Invalid development config archive manifest")
    payload_root = package_root / Path(archive_name).stem
    actual = {path.name for path in payload_root.iterdir() if path.is_file()} if payload_root.is_dir() else set()
    if actual != set(payload_files):
        raise ValueError(f"Expanded config payload mismatch: expected={len(payload_files)}, actual={len(actual)}")
    manifest_path = payload_root / "config_manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != source_manifest_sha256:
        raise ValueError("Expanded config manifest hash mismatch")
    return payload_root, manifest_path


def restore_config(payload_root: Path, manifest_path: Path) -> tuple[Path, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("data_version") != "oofconsensusv1" or manifest.get("development_only") is not True:
        raise ValueError("E2 requires the OOF-consensus development-only config")
    if manifest.get("classes") != EXPECTED_CLASSES:
        raise ValueError("Config class names do not match the frozen RYW18 contract")
    if manifest.get("counts", {}).get("images") != 1127 or manifest.get("counts", {}).get("labels") != 1127:
        raise ValueError("Config does not declare the expected 1127-image development view")
    labels, metadata = manifest.get("labels"), manifest.get("metadata")
    if not isinstance(labels, list) or not isinstance(metadata, list):
        raise ValueError("Config manifest does not describe flattened labels and metadata")
    runtime = Path("/kaggle/working/l1_nomosaic_config_runtime")
    if runtime.exists():
        raise FileExistsError(f"Refusing to overwrite runtime config: {runtime}")
    runtime.mkdir(parents=True)
    shutil.copy2(manifest_path, runtime / "config_manifest.json")
    for record in labels + metadata:
        upload_name = record.get("upload_name")
        relative_path = record.get("relative_path")
        if not isinstance(upload_name, str) or Path(upload_name).name != upload_name or not isinstance(relative_path, str):
            raise ValueError("Invalid flattened config record")
        source = payload_root / upload_name
        target = runtime / safe_relative(relative_path)
        if not source.is_file():
            raise FileNotFoundError(f"Missing config input: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    required = (
        runtime / "labels",
        runtime / "splits" / "folds" / "fold_01" / "train.txt",
        runtime / "splits" / "folds" / "fold_01" / "val.txt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Restored E2 config is incomplete: {missing}")
    return runtime, manifest


def resolve_image_dataset(manifest: dict) -> Path:
    images = manifest.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], dict):
        raise ValueError("Config manifest has no source-image probe")
    source_relative = images[0].get("source_relative")
    if not isinstance(source_relative, str):
        raise ValueError("Config source-image probe is invalid")
    parts = PurePosixPath(source_relative).parts
    candidates = [root for root in mounted_roots() if root.joinpath(*parts).is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate image Dataset: candidates={candidates}, inputs={inventory()}")
    return candidates[0]


def resolve_checkpoint() -> Path:
    candidates = [root / CHECKPOINT_NAME for root in mounted_roots() if (root / CHECKPOINT_NAME).is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate E2 base checkpoint: {inventory()}")
    checkpoint = candidates[0]
    actual = sha256_file(checkpoint)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Base checkpoint SHA-256 mismatch: expected={EXPECTED_CHECKPOINT_SHA256}, actual={actual}")
    return checkpoint


def install_dependencies() -> None:
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "--force-reinstall",
            "torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cu124",
        ],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "torchaudio"], check=False)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "numpy==1.26.4", "ultralytics==8.4.66"],
        check=True,
    )
    import numpy
    import torch
    import torchvision
    import ultralytics

    if (
        torch.__version__ != "2.6.0+cu124"
        or torchvision.__version__.split("+", 1)[0] != "0.21.0"
        or numpy.__version__ != "1.26.4"
        or ultralytics.__version__ != "8.4.66"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Pinned runtime verification failed: "
            f"torch={torch.__version__}, torchvision={torchvision.__version__}, numpy={numpy.__version__}, "
            f"ultralytics={ultralytics.__version__}, cuda={torch.cuda.is_available()}"
        )


def make_runtime_data(config_root: Path, image_root: Path, manifest: dict) -> tuple[Path, list[Path], list[Path]]:
    runtime_root = Path("/kaggle/working/dianjienie_ryw18_oofconsensusv1_nomosaic")
    if runtime_root.exists():
        raise FileExistsError(f"Refusing to overwrite E2 runtime view: {runtime_root}")
    runtime_root.mkdir(parents=True)
    source_by_anchor = {item["anchor_name"]: item["source_relative"] for item in manifest["images"]}
    fold_root = config_root / "splits" / "folds" / "fold_01"

    def resolve_entries(name: str) -> list[tuple[Path, Path]]:
        resolved = []
        for value in [line.strip() for line in (fold_root / name).read_text(encoding="utf-8").splitlines() if line.strip()]:
            anchor_name = Path(value).name
            source_relative = source_by_anchor.get(anchor_name)
            if source_relative is None:
                raise ValueError(f"Fold list image missing from config manifest: {anchor_name}")
            image = image_root.joinpath(*PurePosixPath(source_relative).parts)
            label = config_root / "labels" / f"{Path(anchor_name).stem}.txt"
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f"E2 runtime source missing: image={image}, label={label}")
            linked_image = runtime_root / "images" / anchor_name
            linked_label = runtime_root / "labels" / f"{Path(anchor_name).stem}.txt"
            linked_image.parent.mkdir(parents=True, exist_ok=True)
            linked_label.parent.mkdir(parents=True, exist_ok=True)
            # The image Dataset remains read-only; only a working symlink and
            # a copied five-column auto_only label are created for training.
            linked_image.symlink_to(image)
            shutil.copy2(label, linked_label)
            resolved.append((linked_image, linked_label))
        return resolved

    train_pairs = resolve_entries("train.txt")
    val_pairs = resolve_entries("val.txt")
    train_images = [pair[0] for pair in train_pairs]
    val_images = [pair[0] for pair in val_pairs]
    if set(train_images) & set(val_images) or len(train_images) != 751 or len(val_images) != 376:
        raise ValueError("E2 Fold 1 train/val membership is invalid")
    train_list = runtime_root / "train.txt"
    val_list = runtime_root / "val.txt"
    train_list.write_text("\n".join(path.as_posix() for path in train_images) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(path.as_posix() for path in val_images) + "\n", encoding="utf-8")
    yaml = runtime_root / "data.yaml"
    yaml.write_text(
        "\n".join(
            [
                f'path: "{runtime_root.as_posix()}"',
                f'train: "{train_list.as_posix()}"',
                f'val: "{val_list.as_posix()}"',
                "nc: 18",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(EXPECTED_CLASSES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml, train_images, val_images


def evaluate_full_image_pair(pipeline: Path, config_root: Path, image_root: Path, baseline: Path, candidate: Path) -> None:
    # A 4096px, zero-overlap grid produces one full-image crop for every
    # 4096x3554/3600 source image. This retains the existing evaluator's four
    # report files while introducing no tile inference or postprocessing.
    workspace = Path("/kaggle/working") / f"{RUN_NAME}_evaluation_workspace"
    command = [
        sys.executable, str(pipeline),
        "--mode", "evaluate",
        "--config-root", str(config_root),
        "--image-dataset-root", str(image_root),
        "--output-root", str(workspace),
        "--run-name", RUN_NAME,
        "--fold", "1",
        "--baseline-weights", str(baseline),
        "--candidate-weights", str(candidate),
        "--tile-size", "4096",
        "--overlap", "0.0",
        "--negative-ratio", "0.0",
        "--min-visible-fraction", "1.0",
        "--focus-classes", "wuzi,waterstain,airhole,bubble,greenstain,watermark",
        "--epochs", "30",
        "--imgsz", "640",
        "--batch", "8",
        "--predict-batch", "8",
        "--device", "0",
        "--workers", "8",
        "--seed", "0",
        "--patience", "100",
        "--tiled-eval-profiles", "legacy",
        "--primary-tiled-profile", "legacy",
    ]
    subprocess.run(command, check=True)
    archive = workspace / f"{RUN_NAME}_evaluation_results.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Expected evaluation archive is missing: {archive}")
    published = Path("/kaggle/working") / archive.name
    if published.exists():
        raise FileExistsError(f"Refusing to overwrite published evaluation archive: {published}")
    shutil.copy2(archive, published)


def main() -> None:
    payload_root, manifest_path = resolve_config_payload()
    config_root, manifest = restore_config(payload_root, manifest_path)
    image_root = resolve_image_dataset(manifest)
    checkpoint = resolve_checkpoint()
    install_dependencies()
    pipeline = Path("/kaggle/working/l1_tile_pipeline.py")
    pipeline.write_text(PIPELINE_SOURCE, encoding="utf-8")
    data_yaml, train_images, val_images = make_runtime_data(config_root, image_root, manifest)
    from ultralytics import YOLO

    run_root = Path("/kaggle/working/runs")
    run_dir = run_root / RUN_NAME
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite E2 run directory: {run_dir}")
    requested = {
        "experiment": "E2 Mosaic=0 fine-tune",
        "data_version": "oofconsensusv1",
        "development_only": True,
        "frozen_test_set": None,
        "fold": 1,
        "split_counts": {"train": len(train_images), "val": len(val_images), "test_frozen": None},
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": sha256_file(checkpoint),
        "training": {
            "epochs": 30, "imgsz": 640, "batch": 8, "optimizer": "SGD", "seed": 0,
            "deterministic": True, "resume": False, "mosaic": 0.0, "close_mosaic": 0,
            "save_period": 1, "patience": 100,
        },
        "evaluation": {
            "entry": "l1_tile_pipeline.py --mode evaluate",
            "baseline_full_mAP50_95_reference": 0.168001,
            "tile_size": 4096, "overlap": 0.0,
            "semantic": "one full-image crop per source image; no actual tiling",
        },
    }
    write_json(Path("/kaggle/working") / "e2_requested_config.json", requested)
    model = YOLO(str(checkpoint))
    model.train(
        data=str(data_yaml),
        epochs=30,
        imgsz=640,
        batch=8,
        device="0",
        workers=8,
        project=str(run_root),
        name=RUN_NAME,
        exist_ok=False,
        pretrained=True,
        optimizer="SGD",
        seed=0,
        deterministic=True,
        patience=100,
        cache=False,
        resume=False,
        close_mosaic=0,
        amp=True,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        iou=0.7,
        max_det=300,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",
        auto_augment="randaugment",
        erasing=0.4,
        save_period=1,
        plots=True,
    )
    best, last = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
    required = (best, last, run_dir / "args.yaml", run_dir / "results.csv")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"E2 training output is incomplete: {missing}")
    shutil.copy2(Path("/kaggle/working") / "e2_requested_config.json", run_dir / "requested_config.json")
    shutil.copy2(data_yaml, run_dir / "data.yaml")
    evaluate_full_image_pair(pipeline, config_root, image_root, checkpoint, best)
    archive = Path(shutil.make_archive(str(Path("/kaggle/working") / f"{RUN_NAME}_results"), "zip", run_root, RUN_NAME))
    print(json.dumps({"run_dir": str(run_dir), "archive": str(archive)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_slug(value: str, field: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,62}", value):
        raise ValueError(f"{field} must be 3-63 lowercase letters, digits, or hyphens")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare only the E2 full-image Mosaic=0 Kaggle kernel and checkpoint Dataset staging directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--expected-weight-sha256", default=EXPECTED_WEIGHT_SHA256)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--dataset-title", default=DEFAULT_DATASET_TITLE)
    parser.add_argument("--config-dataset-ref", default=CONFIG_DATASET_REF)
    parser.add_argument("--image-dataset-ref", default=IMAGE_DATASET_REF)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset-output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan without creating staging or kernel files.")
    return parser.parse_args(argv)


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, str, str]:
    if not PIPELINE_PATH.is_file():
        raise FileNotFoundError(f"Missing established evaluator source: {PIPELINE_PATH}")
    source_weight = args.source_weight.resolve()
    if not source_weight.is_file():
        raise FileNotFoundError(f"Missing archived Fold 1 base checkpoint: {source_weight}")
    actual_hash = sha256_file(source_weight)
    expected_hash = str(args.expected_weight_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("--expected-weight-sha256 must be a lowercase 64-character SHA-256")
    if actual_hash != expected_hash:
        raise ValueError(f"Base checkpoint SHA-256 mismatch: expected={expected_hash}, actual={actual_hash}")
    slug = validate_slug(str(args.slug), "--slug")
    dataset_slug = validate_slug(str(args.dataset_slug), "--dataset-slug")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", str(args.owner)):
        raise ValueError("--owner must be a Kaggle-compatible lowercase slug")
    output_root = (args.output_root or ROOT / f"kaggle_{slug.replace('-', '_')}").resolve()
    dataset_output_root = (
        args.dataset_output_root or ROOT / f"kaggle_{dataset_slug.replace('-', '_')}"
    ).resolve()
    if output_root == dataset_output_root:
        raise ValueError("Kernel and Dataset staging directories must be distinct")
    return source_weight, output_root, dataset_output_root, actual_hash, slug


def build_dataset_manifest(source_weight: Path, actual_hash: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "E2 Mosaic=0 full-image fine-tune base checkpoint; staging only, not uploaded",
        "data_version": "oofconsensusv1",
        "development_only": True,
        "frozen_test_set": None,
        "fold": 1,
        "weights": {
            CHECKPOINT_NAME: {
                "source": str(source_weight),
                "sha256": actual_hash,
                "role": "archived OOF Fold 1 YOLO11m best.pt initialization weight",
            }
        },
        "training_contract": {
            "resume": False,
            "epochs": 30,
            "imgsz": 640,
            "batch": 8,
            "optimizer": "SGD",
            "seed": 0,
            "mosaic": 0.0,
            "only_intended_model_training_variable": "mosaic",
            "operational_artifact_change": "save_period=1 to preserve recoverable checkpoints",
            "save_period": 1,
        },
    }


def build_kernel_metadata(args: argparse.Namespace, slug: str, checkpoint_dataset_ref: str) -> dict[str, Any]:
    return {
        "id": f"{args.owner}/{slug}",
        "title": args.title,
        "code_file": f"{slug}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [args.image_dataset_ref, args.config_dataset_ref, checkpoint_dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def build_kernel_script(slug: str) -> str:
    run_name = "dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_nomosaic_ft30_fold01"
    return textwrap.dedent(
        KERNEL_TEMPLATE.substitute(
            run_name=run_name,
            checkpoint_name=CHECKPOINT_NAME,
            expected_checkpoint_sha256=EXPECTED_WEIGHT_SHA256,
            pipeline_source=json.dumps(PIPELINE_PATH.read_text(encoding="utf-8")),
            expected_classes=repr(list(EXPECTED_CLASSES)),
        )
    )


def create_dataset_staging(output_root: Path, source_weight: Path, args: argparse.Namespace, actual_hash: str) -> None:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint Dataset staging directory: {output_root}")
    output_root.mkdir(parents=True)
    shutil.copy2(source_weight, output_root / CHECKPOINT_NAME)
    metadata = {
        "title": args.dataset_title,
        "id": f"{args.owner}/{args.dataset_slug}",
        "licenses": [{"name": "other"}],
    }
    write_json(output_root / "dataset-metadata.json", metadata)
    write_json(output_root / "checkpoint_manifest.json", build_dataset_manifest(source_weight, actual_hash))
    (output_root / "README.md").write_text(
        "# E2 Mosaic=0 base checkpoint staging\n\n"
        "Private Kaggle Dataset staging only. It contains the archived Fold 1 YOLO11m `best.pt` "
        "for a new `resume=false`, 30-epoch, full-image Mosaic=0 fine-tune. `save_period=1` is only "
        "a checkpoint-recovery policy. Do not upload or modify "
        "without user confirmation. The frozen release-validation set is not included.\n",
        encoding="utf-8",
    )


def create_kernel_staging(output_root: Path, metadata: dict[str, Any], script: str, source_weight: Path, actual_hash: str) -> None:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite E2 kernel staging directory: {output_root}")
    output_root.mkdir(parents=True)
    write_json(output_root / "kernel-metadata.json", metadata)
    code_file = output_root / str(metadata["code_file"])
    code_file.write_text(script, encoding="utf-8", newline="\n")
    requested = {
        "experiment": "E2 Mosaic=0 full-image fine-tune",
        "status": "prepared_not_submitted_pending_user_gpu_confirmation",
        "reason_not_reusing_prepare_l1_tile_kaggle_kernel": (
            "E2 is an un-tiled full-image fine-tune. The tile generator changes training-view composition and "
            "cross-tile evaluation behavior, so reuse would violate the single-variable Mosaic comparison."
        ),
        "data_version": "oofconsensusv1",
        "development_only": True,
        "frozen_test_set": None,
        "fold": 1,
        "split_counts": {"train": 751, "val": 376, "test_frozen": None},
        "base_checkpoint": str(source_weight),
        "base_checkpoint_sha256": actual_hash,
        "runtime_pins": {
            "torch": "2.6.0",
            "torchvision": "0.21.0",
            "numpy": "1.26.4",
            "ultralytics": "8.4.66",
        },
        "training": {
            "model": "archived Fold 1 best.pt",
            "resume": False,
            "epochs": 30,
            "imgsz": 640,
            "batch": 8,
            "optimizer": "SGD",
            "seed": 0,
            "deterministic": True,
            "mosaic": 0.0,
            "close_mosaic": 0,
            "save_period": 1,
            "patience": 100,
        },
        "evaluation": {
            "entry": "l1_tile_pipeline.py --mode evaluate",
            "baseline_full_mAP50_95_reference": BASELINE_FULL_MAP50_95_REFERENCE,
            "reports": ["baseline_full", "baseline_tiled", "tile_model_full", "tile_model_tiled"],
            "tile_size": 4096,
            "overlap": 0.0,
            "semantic": "one full-image crop per 4096x3554/3600 source image; no tile generation or cross-tile WBF",
        },
    }
    write_json(output_root / "requested_config.json", requested)
    (output_root / "README.md").write_text(
        "# E2 Mosaic=0 Kaggle kernel package\n\n"
        "Prepared locally only. Do not run `kaggle kernels push` until the user confirms GPU consumption. "
        "The kernel verifies the archived Fold 1 checkpoint hash before it starts a new `resume=false` run.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_weight, output_root, dataset_output_root, actual_hash, slug = validate_inputs(args)
    checkpoint_dataset_ref = f"{args.owner}/{args.dataset_slug}"
    metadata = build_kernel_metadata(args, slug, checkpoint_dataset_ref)
    plan = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "kernel_output_root": str(output_root),
        "checkpoint_dataset_output_root": str(dataset_output_root),
        "kernel_metadata": metadata,
        "checkpoint_manifest": build_dataset_manifest(source_weight, actual_hash),
        "submission": "not performed; this generator contains no Kaggle API or push command",
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    create_dataset_staging(dataset_output_root, source_weight, args, actual_hash)
    try:
        create_kernel_staging(output_root, metadata, build_kernel_script(slug), source_weight, actual_hash)
    except Exception:
        # Preserve the completed checkpoint staging for diagnosis rather than silently removing it.
        raise
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print(f"[prepared] kernel={output_root}")
    print(f"[prepared] checkpoint_dataset={dataset_output_root}")
    print("[pending] User GPU confirmation is required before any Kaggle Dataset upload or kernel submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
