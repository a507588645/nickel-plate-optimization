#!/usr/bin/env python3
"""Prepare, but never submit, the paired L1 E2 Mosaic ablation kernels.

Both variants begin from the exact same archived Fold 1 YOLO11m checkpoint and
restart a fresh 30-epoch SGD fine-tune.  The control keeps ``mosaic=1.0`` and
the candidate uses ``mosaic=0.0``.  This makes the post-restart training
effect common to both jobs, so their difference can be attributed to Mosaic.

The script only creates local Kaggle Dataset/Kernel staging folders.  It has no
Kaggle API calls and never uploads, pushes, trains, or reads release-validation.
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
DEFAULT_PAIR_ID = "l1_e2_mosaicpair_f01_ft30_v1"
DEFAULT_DATASET_SLUG = "dianjienie-ryw18-oof-yolo11m-fold01-640-weight"
DEFAULT_DATASET_TITLE = "RYW18 OOF YOLO11m Fold01 640 Weight"
# The first local draft remains intact for provenance.  This is the canonical
# staging root after adding archive runtime evidence needed by pair validation.
# r2 used a Dataset slug that Kaggle rejected for exceeding its 50-character cap.
# r3 was not uploaded because its new Dataset upload endpoint was unreachable.
# r4 surfaced Kaggle's title-to-id validation. Reuse the already verified
# private Dataset and keep titles that normalize to their fixed Kernel IDs.
DEFAULT_PAIR_OUTPUT_ROOT = ROOT / "l1_e2_mosaic_pair_20260822_r5"
BASELINE_FULL_MAP50_95 = 0.16800147577890728
BASELINE_TOLERANCE = 1e-6
PAIR_DELTA_THRESHOLD = 0.01
FOCUS_CLASSES = ("wuzi", "waterstain", "airhole", "bubble", "greenstain", "watermark")
CHECKPOINT_NAME = "baseline_fold01_640_best.pt"
EXPECTED_WEIGHT_SHA256 = "b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce"
DEFAULT_WEIGHT = (
    Path(r"E:/PythonProject/models/runs")
    / "dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_fold01"
    / "weights"
    / "best.pt"
)
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

VARIANT_DEFAULTS = {
    "control": {
        "slug": "l1-ryw18-y11m-oof-v1-mosaic1-ft30-f01",
        "title": "L1 RYW18 Y11m OOF V1 Mosaic1 FT30 F01",
        "run_name": "dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_mosaic1_ft30_fold01",
        "mosaic": 1.0,
    },
    "candidate": {
        "slug": "l1-ryw18-y11m-oof-v1-mosaic0-ft30-f01",
        "title": "L1 RYW18 Y11m OOF V1 Mosaic0 FT30 F01",
        "run_name": "dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_mosaic0_ft30_fold01",
        "mosaic": 0.0,
    },
}


KERNEL_TEMPLATE = Template(
    r'''#!/usr/bin/env python3
"""Run one member of the paired L1 E2 Mosaic ablation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


INPUT_ROOT = Path("/kaggle/input")
RUN_NAME = "$run_name"
ROLE = "$role"
MOSAIC = $mosaic
CHECKPOINT_NAME = "$checkpoint_name"
EXPECTED_CHECKPOINT_SHA256 = "$expected_checkpoint_sha256"
BASELINE_FULL_MAP50_95 = $baseline_full_map50_95
BASELINE_TOLERANCE = $baseline_tolerance
FOCUS_CLASSES = $focus_classes
PAIR_CONTRACT = json.loads($pair_contract_json)
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
    return {
        str(root.relative_to(INPUT_ROOT)): sorted(path.name for path in root.iterdir())[:12]
        for root in mounted_roots()
    }


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
    runtime = Path("/kaggle/working/l1_e2_mosaic_pair_config_runtime")
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
    runtime_root = Path("/kaggle/working/dianjienie_ryw18_oofconsensusv1_e2_mosaicpair")
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


def evaluate_full_image_pair(
    pipeline: Path,
    config_root: Path,
    image_root: Path,
    baseline: Path,
    candidate: Path,
    run_dir: Path,
) -> dict:
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
        "--focus-classes", ",".join(FOCUS_CLASSES),
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
    source = workspace / "evaluation"
    if not source.is_dir():
        raise FileNotFoundError(f"Expected full-image evaluation evidence is missing: {source}")
    destination = run_dir / "evaluation"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation evidence: {destination}")
    shutil.copytree(source, destination)
    baseline_report = json.loads((destination / "tile_evaluation" / "baseline_full.json").read_text(encoding="utf-8"))
    candidate_report = json.loads((destination / "tile_evaluation" / "tile_model_full.json").read_text(encoding="utf-8"))
    candidate_hash = sha256_file(candidate)
    if candidate_report.get("model_sha256") != candidate_hash:
        raise ValueError("Evaluation candidate report does not match the saved best.pt")
    baseline_metric = float(baseline_report["mAP50_95"])
    if abs(baseline_metric - BASELINE_FULL_MAP50_95) > BASELINE_TOLERANCE:
        raise ValueError(
            f"Fixed full-image baseline did not reproduce: expected={BASELINE_FULL_MAP50_95}, actual={baseline_metric}"
        )
    per_class = candidate_report.get("per_class")
    if not isinstance(per_class, dict):
        raise ValueError("Candidate full-image report has no per-class metrics")
    focus_recall = {}
    for class_name in FOCUS_CLASSES:
        item = per_class.get(class_name)
        if not isinstance(item, dict) or not isinstance(item.get("recall_conf025_iou50"), (int, float)):
            raise ValueError(f"Candidate report is missing focus recall: {class_name}")
        focus_recall[class_name] = float(item["recall_conf025_iou50"])
    summary = {
        "pair_id": PAIR_CONTRACT["pair_id"],
        "role": ROLE,
        "mosaic": MOSAIC,
        "evaluation_mode": "full_image",
        "baseline_full_mAP50_95": baseline_metric,
        "candidate_best_mAP50_95": float(candidate_report["mAP50_95"]),
        "candidate_minus_fixed_baseline_mAP50_95": float(candidate_report["mAP50_95"]) - baseline_metric,
        "candidate_focus_recall_conf025_iou50": focus_recall,
        "baseline_model_sha256": baseline_report.get("model_sha256"),
        "candidate_best_model_sha256": candidate_hash,
        "tile_size": 4096,
        "overlap": 0.0,
    }
    write_json(destination / "evaluation_summary.json", summary)
    return summary


def environment_payload() -> dict:
    import numpy
    import torch
    import torchvision
    import ultralytics

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": numpy.__version__,
        "ultralytics": ultralytics.__version__,
    }


def write_run_evidence(
    run_dir: Path,
    config_root: Path,
    data_yaml: Path,
    train_images: list[Path],
    val_images: list[Path],
    checkpoint: Path,
    evaluation_summary: dict,
) -> None:
    dataset_config = run_dir / "dataset_config"
    dataset_config.mkdir(parents=True, exist_ok=False)
    shutil.copy2(data_yaml, dataset_config / "data.yaml")
    fold_root = config_root / "splits" / "folds" / "fold_01"
    shutil.copy2(fold_root / "train.txt", dataset_config / "train.txt")
    shutil.copy2(fold_root / "val.txt", dataset_config / "val.txt")
    requested = {
        "experiment": "L1 E2 paired Mosaic ablation",
        "run_name": RUN_NAME,
        "pair_id": PAIR_CONTRACT["pair_id"],
        "role": ROLE,
        "mosaic": MOSAIC,
        "data_version": "oofconsensusv1",
        "development_only": True,
        "frozen_test_set": None,
        "fold": 1,
        "split_counts": {"train": len(train_images), "val": len(val_images), "test_frozen": None},
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": sha256_file(checkpoint),
        "training": {
            "epochs": 30, "imgsz": 640, "batch": 8, "optimizer": "SGD", "seed": 0,
            "deterministic": True, "resume": False, "mosaic": MOSAIC, "close_mosaic": 0,
            "save_period": 1, "patience": 100,
        },
        "evaluation": evaluation_summary,
    }
    write_json(run_dir / "requested_config.json", requested)
    write_json(run_dir / "e2_pair_contract.json", PAIR_CONTRACT)
    write_json(run_dir / "environment.json", environment_payload())
    (run_dir / "README.md").write_text(
        "# L1 E2 paired Mosaic ablation run\n\n"
        f"Role: `{ROLE}`; Mosaic: `{MOSAIC}`; Pair: `{PAIR_CONTRACT['pair_id']}`. "
        "This run is development-only Fold 1 and starts a new `resume=false` fine-tune from the shared archived "
        "checkpoint.  Its only intended training variable relative to its paired sibling is `mosaic`. "
        "The frozen release-validation set is not included.\n",
        encoding="utf-8",
    )


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
        mosaic=MOSAIC,
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
    evaluation_summary = evaluate_full_image_pair(pipeline, config_root, image_root, checkpoint, best, run_dir)
    write_run_evidence(run_dir, config_root, data_yaml, train_images, val_images, checkpoint, evaluation_summary)
    archive = Path(shutil.make_archive(str(Path("/kaggle/working") / f"{RUN_NAME}_results"), "zip", run_root, RUN_NAME))
    print(json.dumps({"role": ROLE, "run_dir": str(run_dir), "archive": str(archive)}, ensure_ascii=False))


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


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_slug(value: str, field: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,49}", value):
        raise ValueError(f"{field} must be 3-50 lowercase letters, digits, or hyphens")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare only the paired E2 Mosaic=1 control and Mosaic=0 candidate Kaggle packages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--expected-weight-sha256", default=EXPECTED_WEIGHT_SHA256)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--pair-id", default=DEFAULT_PAIR_ID)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--dataset-title", default=DEFAULT_DATASET_TITLE)
    parser.add_argument("--control-slug", default=VARIANT_DEFAULTS["control"]["slug"])
    parser.add_argument("--candidate-slug", default=VARIANT_DEFAULTS["candidate"]["slug"])
    parser.add_argument("--control-title", default=VARIANT_DEFAULTS["control"]["title"])
    parser.add_argument("--candidate-title", default=VARIANT_DEFAULTS["candidate"]["title"])
    parser.add_argument("--config-dataset-ref", default=CONFIG_DATASET_REF)
    parser.add_argument("--image-dataset-ref", default=IMAGE_DATASET_REF)
    parser.add_argument("--pair-output-root", type=Path, default=DEFAULT_PAIR_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the contract without creating files.")
    return parser.parse_args(argv)


def validate_inputs(args: argparse.Namespace) -> tuple[Path, str, Path, dict[str, dict[str, Any]]]:
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
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", str(args.owner)):
        raise ValueError("--owner must be a Kaggle-compatible lowercase slug")
    pair_id = str(args.pair_id)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,96}", pair_id):
        raise ValueError("--pair-id must contain only lowercase letters, digits, underscores, or hyphens")
    dataset_slug = validate_slug(str(args.dataset_slug), "--dataset-slug")
    variants = {
        "control": {
            **VARIANT_DEFAULTS["control"],
            "slug": validate_slug(str(args.control_slug), "--control-slug"),
            "title": str(args.control_title),
        },
        "candidate": {
            **VARIANT_DEFAULTS["candidate"],
            "slug": validate_slug(str(args.candidate_slug), "--candidate-slug"),
            "title": str(args.candidate_title),
        },
    }
    if variants["control"]["slug"] == variants["candidate"]["slug"]:
        raise ValueError("Control and candidate Kernel slugs must differ")
    pair_output_root = args.pair_output_root.resolve()
    return source_weight, actual_hash, pair_output_root, variants


def build_common_contract(source_weight: Path, actual_hash: str, pipeline_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "L1 E2 paired Mosaic ablation",
        "data_version": "oofconsensusv1",
        "development_only": True,
        "frozen_test_set": None,
        "fold": 1,
        "split_counts": {"train": 751, "val": 376, "test_frozen": None},
        "base_checkpoint": {
            "source": str(source_weight),
            "file_name": CHECKPOINT_NAME,
            "sha256": actual_hash,
        },
        "runtime_pins": {
            "torch": "2.6.0+cu124",
            "torchvision": "0.21.0",
            "numpy": "1.26.4",
            "ultralytics": "8.4.66",
        },
        "training": {
            "epochs": 30,
            "imgsz": 640,
            "batch": 8,
            "optimizer": "SGD",
            "seed": 0,
            "deterministic": True,
            "resume": False,
            "close_mosaic": 0,
            "patience": 100,
            "save_period": 1,
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "augmentations_except_mosaic": {
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "translate": 0.1,
                "scale": 0.5,
                "fliplr": 0.5,
                "mixup": 0.0,
                "cutmix": 0.0,
                "copy_paste": 0.0,
                "auto_augment": "randaugment",
                "erasing": 0.4,
            },
        },
        "evaluation": {
            "entry": "l1_tile_pipeline.py --mode evaluate",
            "pipeline_sha256": pipeline_sha256,
            "model_weight": "best.pt",
            "tile_size": 4096,
            "overlap": 0.0,
            "semantic": "one full-image crop per source image; no actual tiling or cross-tile WBF",
            "fixed_baseline_full_mAP50_95": BASELINE_FULL_MAP50_95,
            "baseline_tolerance": BASELINE_TOLERANCE,
            "focus_classes": list(FOCUS_CLASSES),
        },
        "gate": {
            "causal_mosaic_delta_mAP50_95_min": PAIR_DELTA_THRESHOLD,
            "utility_delta_vs_fixed_baseline_mAP50_95_min": PAIR_DELTA_THRESHOLD,
            "focus_recall_requirement": "candidate Mosaic=0 recall@0.25 must not decline versus Mosaic=1 control for every focus class",
        },
    }


def build_variant_contract(
    pair_id: str,
    common_contract: dict[str, Any],
    role: str,
    variants: dict[str, dict[str, Any]],
    owner: str,
) -> dict[str, Any]:
    sibling_role = "candidate" if role == "control" else "control"
    return {
        "pair_id": pair_id,
        "common_contract_sha256": hashlib.sha256(canonical_json(common_contract).encode("utf-8")).hexdigest(),
        "common_contract": common_contract,
        "variant": {
            "role": role,
            "mosaic": variants[role]["mosaic"],
            "kernel": f"{owner}/{variants[role]['slug']}",
            "run_name": variants[role]["run_name"],
        },
        "expected_sibling": {
            "role": sibling_role,
            "mosaic": variants[sibling_role]["mosaic"],
            "kernel": f"{owner}/{variants[sibling_role]['slug']}",
            "run_name": variants[sibling_role]["run_name"],
        },
    }


def build_kernel_metadata(
    owner: str,
    variant: dict[str, Any],
    image_dataset_ref: str,
    config_dataset_ref: str,
    checkpoint_dataset_ref: str,
) -> dict[str, Any]:
    slug = str(variant["slug"])
    return {
        "id": f"{owner}/{slug}",
        "title": str(variant["title"]),
        "code_file": f"{slug}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [image_dataset_ref, config_dataset_ref, checkpoint_dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def build_kernel_script(variant: dict[str, Any], contract: dict[str, Any]) -> str:
    return textwrap.dedent(
        KERNEL_TEMPLATE.substitute(
            run_name=variant["run_name"],
            role=contract["variant"]["role"],
            mosaic=repr(float(variant["mosaic"])),
            checkpoint_name=CHECKPOINT_NAME,
            expected_checkpoint_sha256=contract["common_contract"]["base_checkpoint"]["sha256"],
            baseline_full_map50_95=repr(BASELINE_FULL_MAP50_95),
            baseline_tolerance=repr(BASELINE_TOLERANCE),
            focus_classes=repr(list(FOCUS_CLASSES)),
            pair_contract_json=json.dumps(json.dumps(contract, ensure_ascii=False)),
            pipeline_source=json.dumps(PIPELINE_PATH.read_text(encoding="utf-8")),
            expected_classes=repr(list(EXPECTED_CLASSES)),
        )
    )


def create_checkpoint_dataset_staging(
    output_root: Path,
    source_weight: Path,
    actual_hash: str,
    args: argparse.Namespace,
    common_contract: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_weight, output_root / CHECKPOINT_NAME)
    write_json(
        output_root / "dataset-metadata.json",
        {
            "title": args.dataset_title,
            "id": f"{args.owner}/{args.dataset_slug}",
            "licenses": [{"name": "other"}],
        },
    )
    write_json(
        output_root / "checkpoint_manifest.json",
        {
            "schema_version": 1,
            "purpose": "Local audit mirror of the immutable checkpoint reused by the paired E2 Mosaic ablation.",
            "checkpoint": {"file_name": CHECKPOINT_NAME, "sha256": actual_hash, "source": str(source_weight)},
            "common_contract_sha256": hashlib.sha256(canonical_json(common_contract).encode("utf-8")).hexdigest(),
            "development_only": True,
            "frozen_test_set": None,
        },
    )
    (output_root / "README.md").write_text(
        "# L1 E2 paired Mosaic checkpoint staging\n\n"
        "This local audit mirror verifies the immutable Fold 1 YOLO11m base checkpoint shared by the "
        "Mosaic=1 control and Mosaic=0 candidate. The Kernels reuse the already-uploaded private Dataset "
        "identified by this staging metadata; this folder must not be uploaded as a new Dataset version.\n",
        encoding="utf-8",
    )


def create_kernel_staging(
    output_root: Path,
    metadata: dict[str, Any],
    contract: dict[str, Any],
    script: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root / "kernel-metadata.json", metadata)
    (output_root / str(metadata["code_file"])).write_text(script, encoding="utf-8", newline="\n")
    requested = {
        "experiment": "L1 E2 paired Mosaic ablation",
        "status": "prepared_not_submitted_pending_user_gpu_confirmation",
        "pair_id": contract["pair_id"],
        "role": contract["variant"]["role"],
        "mosaic": contract["variant"]["mosaic"],
        "expected_sibling": contract["expected_sibling"],
        "common_contract_sha256": contract["common_contract_sha256"],
        "development_only": True,
        "frozen_test_set": None,
        "submission": "not performed; this generator contains no Kaggle API or push command",
    }
    write_json(output_root / "requested_config.json", requested)
    write_json(output_root / "e2_pair_contract.json", contract)
    (output_root / "README.md").write_text(
        "# L1 E2 paired Mosaic Kernel staging\n\n"
        f"Role: `{contract['variant']['role']}`; Mosaic: `{contract['variant']['mosaic']}`. "
        "This local-only package must be submitted together with its sibling after explicit GPU authorization. "
        "The frozen release-validation set is not included.\n",
        encoding="utf-8",
    )


def validate_staging(pair_root: Path, variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    validation: dict[str, Any] = {"pair_root": str(pair_root), "variants": {}}
    contract_hashes: set[str] = set()
    for role, variant in variants.items():
        kernel_root = pair_root / f"{role}_kernel"
        metadata = json.loads((kernel_root / "kernel-metadata.json").read_text(encoding="utf-8"))
        requested = json.loads((kernel_root / "requested_config.json").read_text(encoding="utf-8"))
        contract = json.loads((kernel_root / "e2_pair_contract.json").read_text(encoding="utf-8"))
        source = (kernel_root / metadata["code_file"]).read_text(encoding="utf-8")
        compile(source, str(kernel_root / metadata["code_file"]), "exec")
        if metadata["id"].rsplit("/", 1)[-1] != variant["slug"]:
            raise ValueError(f"Staged metadata slug mismatch for {role}")
        if requested.get("role") != role or float(requested.get("mosaic")) != float(variant["mosaic"]):
            raise ValueError(f"Staged requested config mismatch for {role}")
        if contract.get("variant", {}).get("role") != role or float(contract["variant"].get("mosaic")) != float(variant["mosaic"]):
            raise ValueError(f"Staged pair contract mismatch for {role}")
        if any("release" in str(dataset_ref).lower() for dataset_ref in metadata["dataset_sources"]):
            raise ValueError(f"Staged Kernel declares a forbidden frozen-data Dataset source: {role}")
        contract_hashes.add(str(contract.get("common_contract_sha256")))
        validation["variants"][role] = {
            "kernel": metadata["id"],
            "run_name": contract["variant"]["run_name"],
            "mosaic": contract["variant"]["mosaic"],
            "script_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }
    if len(contract_hashes) != 1:
        raise ValueError("Control and candidate do not share one common contract hash")
    validation["common_contract_sha256"] = contract_hashes.pop()
    return validation


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_weight, actual_hash, pair_root, variants = validate_inputs(args)
    pipeline_sha256 = sha256_file(PIPELINE_PATH)
    common_contract = build_common_contract(source_weight, actual_hash, pipeline_sha256)
    contracts = {
        role: build_variant_contract(args.pair_id, common_contract, role, variants, args.owner)
        for role in variants
    }
    checkpoint_dataset_ref = f"{args.owner}/{args.dataset_slug}"
    plan = {
        "pair_id": args.pair_id,
        "pair_root": str(pair_root),
        "checkpoint_dataset_root": str(pair_root / "checkpoint_dataset"),
        "checkpoint_dataset": checkpoint_dataset_ref,
        "checkpoint_dataset_mode": "reuse_existing_private_dataset",
        "common_contract": common_contract,
        "variants": {
            role: {
                "kernel_root": str(pair_root / f"{role}_kernel"),
                "kernel": f"{args.owner}/{variant['slug']}",
                "run_name": variant["run_name"],
                "mosaic": variant["mosaic"],
            }
            for role, variant in variants.items()
        },
        "submission": "not performed; local staging only",
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if pair_root.exists():
        raise FileExistsError(f"Refusing to overwrite paired E2 staging root: {pair_root}")
    pair_root.mkdir(parents=True)
    create_checkpoint_dataset_staging(pair_root / "checkpoint_dataset", source_weight, actual_hash, args, common_contract)
    for role, variant in variants.items():
        metadata = build_kernel_metadata(
            args.owner,
            variant,
            args.image_dataset_ref,
            args.config_dataset_ref,
            checkpoint_dataset_ref,
        )
        create_kernel_staging(
            pair_root / f"{role}_kernel",
            metadata,
            contracts[role],
            build_kernel_script(variant, contracts[role]),
        )
    write_json(pair_root / "pair_submission_plan.json", plan)
    write_json(pair_root / "pair_contract_common.json", common_contract)
    (pair_root / "README.md").write_text(
        "# L1 E2 paired Mosaic ablation staging\n\n"
        "Contains an audit mirror of one verified existing private checkpoint Dataset and two private Kernel "
        "stagings: Mosaic=1 control and Mosaic=0 candidate. No new checkpoint Dataset upload is required. "
        "Submit both only after explicit GPU authorization, then verify their completed archives together with "
        "`verify_l1_e2_mosaic_pair.py`.\n",
        encoding="utf-8",
    )
    validation = validate_staging(pair_root, variants)
    write_json(pair_root / "local_staging_validation.json", validation)
    print(json.dumps({**plan, "validation": validation}, ensure_ascii=False, indent=2))
    print("[prepared] No Kaggle Dataset upload or Kernel submission was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
