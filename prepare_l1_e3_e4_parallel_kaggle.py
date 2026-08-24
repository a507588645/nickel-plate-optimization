#!/usr/bin/env python3
"""Prepare two bounded, development-only Kaggle GPU jobs for L1 E3 and E4.

This script only creates local staging folders.  It never calls the Kaggle API,
starts training, or changes a source dataset.  Submission is deliberately kept
as a separate, auditable action after the generated packages pass local checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import textwrap
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = ROOT / "l1_tile_pipeline.py"
P2_YAML_PATH = ROOT / "l1_e3_p2_smoke_20260823" / "yolo11m_p2.yaml"
P2_SMOKE_PATH = ROOT / "l1_e3_p2_smoke_20260823" / "smoke_report.json"
DEV_CONFIG_PAYLOAD = ROOT / "kaggle_dianjienie_ryw18_oofconsensus_v1_dev_only" / "config_upload_archive_v1"
CP_SOURCE_ROOT = ROOT / "l1_e4_oofconsensus_v1_cp_v1_20260823"
BASELINE_WEIGHT = ROOT / "kaggle_dianjienie_ryw18_oof_yolo11m_fold01_640_weight" / "baseline_fold01_640_best.pt"

OWNER = "lly3323w"
IMAGE_DATASET_REF = "lly3323w/dianjienie"
DEV_CONFIG_DATASET_REF = "lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config"
BASELINE_DATASET_REF = "lly3323w/dianjienie-ryw18-oof-yolo11m-fold01-640-weight"
E4_DATASET_SLUG = "dianjienie-ryw18-oof-cp-v1-fold01-r2"
E4_DATASET_REF = f"{OWNER}/{E4_DATASET_SLUG}"
BASELINE_WEIGHT_NAME = "baseline_fold01_640_best.pt"
BASELINE_WEIGHT_SHA256 = "b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce"
BASELINE_FULL_MAP50_95 = 0.16800147577890728
BASELINE_TOLERANCE = 1e-6
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
FOCUS_CLASSES = ("wuzi", "waterstain", "airhole", "bubble", "greenstain", "watermark")
RUNTIME_PINS = {
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0",
    "numpy": "1.26.4",
    "ultralytics": "8.4.66",
}


KERNEL_TEMPLATE = r'''#!/usr/bin/env python3
"""Run one bounded L1 E3/E4 Fold 1 development-only GPU experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageFilter


CONFIG = json.loads(__CONFIG_JSON_LITERAL__)
PIPELINE_SOURCE = __PIPELINE_SOURCE_LITERAL__
INPUT_ROOT = Path("/kaggle/input")


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
        raise ValueError(f"Unsafe relative path: {value!r}")
    return Path(*candidate.parts)


def blend_mask(width: int, height: int, edge: int) -> Image.Image:
    if edge == 0:
        return Image.new("L", (width, height), 255)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle((edge, edge, width - edge - 1, height - edge - 1), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=max(1, edge // 2)))


def render_e4_synthetic(
    output_image: Path,
    image_root: Path,
    mapping: dict[str, str],
    record: dict,
) -> None:
    donor_name = Path(str(record["source_image"])).name
    target_name = Path(str(record["target_image"])).name
    donor_relative = mapping.get(donor_name)
    target_relative = mapping.get(target_name)
    if donor_relative is None or target_relative is None:
        raise ValueError(f"E4 manifest image is outside the development mapping: donor={donor_name}, target={target_name}")
    crop = record.get("source_crop_pixels")
    placement = record.get("target_placement_pixels")
    blend = record.get("blend")
    if not isinstance(crop, dict) or not isinstance(placement, dict) or not isinstance(blend, dict):
        raise ValueError("E4 synthetic record has incomplete crop, placement, or blend data")
    crop_box = tuple(int(crop[key]) for key in ("left", "top", "right", "bottom"))
    left, top = int(placement["left"]), int(placement["top"])
    edge = int(blend["edge_px"])
    donor_path = image_root.joinpath(*PurePosixPath(donor_relative).parts)
    target_path = image_root.joinpath(*PurePosixPath(target_relative).parts)
    with Image.open(donor_path) as donor_image, Image.open(target_path) as target_image:
        patch = donor_image.convert("RGB").crop(crop_box)
        expected_size = (int(placement["right"]) - left, int(placement["bottom"]) - top)
        if patch.size != expected_size:
            raise ValueError(f"E4 patch dimensions do not match frozen placement: {patch.size} != {expected_size}")
        output = target_image.convert("RGB").copy()
        output.paste(patch, (left, top), blend_mask(patch.width, patch.height, edge))
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output.save(output_image, format="PNG", compress_level=3)


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

    expected = CONFIG["runtime_pins"]
    if (
        torch.__version__ != expected["torch"]
        or torchvision.__version__.split("+", 1)[0] != expected["torchvision"]
        or numpy.__version__ != expected["numpy"]
        or ultralytics.__version__ != expected["ultralytics"]
        or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Pinned runtime verification failed: "
            f"torch={torch.__version__}, torchvision={torchvision.__version__}, numpy={numpy.__version__}, "
            f"ultralytics={ultralytics.__version__}, cuda={torch.cuda.is_available()}"
        )


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


def resolve_dev_config_payload() -> tuple[Path, Path]:
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


def restore_dev_config(payload_root: Path, manifest_path: Path) -> tuple[Path, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("data_version") != "oofconsensusv1" or manifest.get("development_only") is not True:
        raise ValueError("Only the OOF-consensus development-only config may be used")
    if manifest.get("classes") != CONFIG["expected_classes"]:
        raise ValueError("Development config class names do not match the frozen RYW18 contract")
    if manifest.get("counts", {}).get("images") != 1127 or manifest.get("counts", {}).get("labels") != 1127:
        raise ValueError("Development config must declare exactly 1127 images and labels")
    labels, metadata = manifest.get("labels"), manifest.get("metadata")
    if not isinstance(labels, list) or not isinstance(metadata, list):
        raise ValueError("Config manifest does not describe flattened labels and metadata")
    runtime = Path("/kaggle/working/l1_e3e4_dev_config_runtime")
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
        raise RuntimeError(f"Cannot uniquely locate source-image Dataset: candidates={candidates}, inputs={inventory()}")
    return candidates[0]


def resolve_baseline_weight() -> Path:
    candidates = [root / CONFIG["baseline_weight_name"] for root in mounted_roots() if (root / CONFIG["baseline_weight_name"]).is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate fixed Fold 1 baseline checkpoint: {inventory()}")
    weight = candidates[0]
    actual = sha256_file(weight)
    if actual != CONFIG["baseline_weight_sha256"]:
        raise ValueError(f"Baseline checkpoint SHA-256 mismatch: expected={CONFIG['baseline_weight_sha256']}, actual={actual}")
    return weight


def source_mapping(manifest: dict) -> dict[str, str]:
    mapping = {item["anchor_name"]: item["source_relative"] for item in manifest["images"]}
    if len(mapping) != 1127:
        raise ValueError("The development config image mapping is incomplete")
    return mapping


def resolve_e4_payload() -> tuple[Path, dict]:
    candidates = []
    seen_payload_manifests = set()
    for root in mounted_roots():
        for manifest_path in root.rglob("payload_manifest.json"):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("experiment") == "L1 E4 Copy-Paste Fold 1 Kaggle payload":
                # mounted_roots() deliberately includes both owner and dataset paths.
                # Deduplicate a manifest found through those overlapping roots.
                manifest_key = str(manifest_path.resolve())
                if manifest_key not in seen_payload_manifests:
                    seen_payload_manifests.add(manifest_key)
                    candidates.append((manifest_path.parent, payload))
    if not candidates:
        archives = []
        seen_archives = set()
        for root in mounted_roots():
            for archive_manifest_path in root.rglob("e4_cp_f01_archive_manifest.json"):
                archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
                archive_name = archive_manifest.get("archive")
                if not isinstance(archive_name, str):
                    continue
                archive_path = archive_manifest_path.parent / archive_name
                if archive_path.is_file():
                    archive_key = str(archive_path.resolve())
                    if archive_key not in seen_archives:
                        seen_archives.add(archive_key)
                        archives.append((archive_path, archive_manifest))
        if len(archives) == 1:
            archive_path, archive_manifest = archives[0]
            if sha256_file(archive_path) != archive_manifest.get("sha256"):
                raise ValueError("E4 Copy-Paste archive checksum mismatch")
            extracted = Path("/kaggle/working/l1_e4_payload_archive_runtime")
            if extracted.exists():
                raise FileExistsError(f"Refusing to overwrite E4 payload extraction: {extracted}")
            shutil.unpack_archive(str(archive_path), str(extracted), "zip")
            manifest_path = extracted / "payload_manifest.json"
            if manifest_path.is_file():
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if payload.get("experiment") == "L1 E4 Copy-Paste Fold 1 Kaggle payload":
                    candidates.append((extracted, payload))
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate E4 Copy-Paste payload: {inventory()}")
    root, payload = candidates[0]
    if payload.get("development_only") is not True or payload.get("frozen_test_set") is not None:
        raise ValueError("E4 payload does not declare development-only frozen-test isolation")
    if payload.get("split_counts") != {"train": 787, "val": 376, "synthetic_train": 36}:
        raise ValueError("E4 payload split counts are not the approved Fold 1 view")
    if payload.get("classes") != CONFIG["expected_classes"]:
        raise ValueError("E4 payload classes do not match the frozen RYW18 contract")
    for record in payload.get("files", []):
        relative = record.get("relative_path")
        expected_hash = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("Invalid E4 payload file record")
        path = root / safe_relative(relative)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"E4 payload integrity check failed: {relative}")
    return root, payload


def write_data_yaml(runtime_root: Path, train_images: list[Path], val_images: list[Path]) -> Path:
    train_list = runtime_root / "train.txt"
    val_list = runtime_root / "val.txt"
    train_list.write_text("\n".join(path.as_posix() for path in train_images) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(path.as_posix() for path in val_images) + "\n", encoding="utf-8")
    data_yaml = runtime_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f'path: "{runtime_root.as_posix()}"',
                f'train: "{train_list.as_posix()}"',
                f'val: "{val_list.as_posix()}"',
                "nc: 18",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CONFIG["expected_classes"])],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def materialize_original_pair(
    runtime_root: Path,
    image_root: Path,
    config_root: Path,
    mapping: dict[str, str],
    image_name: str,
    label_source: Path,
) -> tuple[Path, Path]:
    source_relative = mapping.get(image_name)
    if source_relative is None:
        raise ValueError(f"No source-image mapping for {image_name}")
    image = image_root.joinpath(*PurePosixPath(source_relative).parts)
    if not image.is_file() or not label_source.is_file():
        raise FileNotFoundError(f"Runtime source missing: image={image}, label={label_source}")
    linked_image = runtime_root / "images" / image_name
    linked_label = runtime_root / "labels" / label_source.name
    linked_image.parent.mkdir(parents=True, exist_ok=True)
    linked_label.parent.mkdir(parents=True, exist_ok=True)
    linked_image.symlink_to(image)
    shutil.copy2(label_source, linked_label)
    return linked_image, linked_label


def build_e3_runtime(config_root: Path, manifest: dict, image_root: Path) -> tuple[Path, list[Path], list[Path], dict]:
    runtime_root = Path("/kaggle/working/l1_e3_p2_runtime")
    if runtime_root.exists():
        raise FileExistsError(f"Refusing to overwrite E3 runtime view: {runtime_root}")
    runtime_root.mkdir(parents=True)
    mapping = source_mapping(manifest)
    fold_root = config_root / "splits" / "folds" / "fold_01"

    def materialize(name: str) -> list[Path]:
        images: list[Path] = []
        for value in [line.strip() for line in (fold_root / name).read_text(encoding="utf-8").splitlines() if line.strip()]:
            image_name = Path(value).name
            label = config_root / "labels" / f"{Path(image_name).stem}.txt"
            linked_image, _ = materialize_original_pair(runtime_root, image_root, config_root, mapping, image_name, label)
            images.append(linked_image)
        return images

    train_images, val_images = materialize("train.txt"), materialize("val.txt")
    if len(train_images) != 751 or len(val_images) != 376 or set(train_images) & set(val_images):
        raise ValueError("E3 Fold 1 runtime membership is invalid")
    data_yaml = write_data_yaml(runtime_root, train_images, val_images)
    return data_yaml, train_images, val_images, {"synthetic_train": 0, "e4_payload_manifest_sha256": None}


def build_e4_runtime(
    config_root: Path,
    manifest: dict,
    image_root: Path,
    payload_root: Path,
    payload: dict,
) -> tuple[Path, list[Path], list[Path], dict]:
    runtime_root = Path("/kaggle/working/l1_e4_copypaste_runtime")
    if runtime_root.exists():
        raise FileExistsError(f"Refusing to overwrite E4 runtime view: {runtime_root}")
    runtime_root.mkdir(parents=True)
    mapping = source_mapping(manifest)
    fold_root = payload_root / "fold_01"
    labels_root = fold_root / "labels"
    train_entries = [line.strip() for line in (fold_root / "splits" / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    val_entries = [line.strip() for line in (fold_root / "splits" / "val.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(train_entries) != 787 or len(val_entries) != 376 or set(train_entries) & set(val_entries):
        raise ValueError("E4 Fold 1 payload membership is invalid")
    train_images: list[Path] = []
    val_images: list[Path] = []
    synthetic_names: list[str] = []
    synthetic_manifest = fold_root / "manifests" / "synthetic_samples.jsonl"
    if not synthetic_manifest.is_file():
        raise FileNotFoundError(f"E4 frozen synthetic manifest is missing: {synthetic_manifest}")
    synthetic_records = [json.loads(line) for line in synthetic_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    records_by_name = {Path(str(record["synthetic_image"])).name: record for record in synthetic_records}
    original_train_names = {Path(entry).name for entry in train_entries if not Path(entry).name.startswith("cp_")}
    if len(records_by_name) != 36:
        raise ValueError("E4 frozen synthetic manifest must describe exactly 36 unique images")

    def materialize(entry: str, target: list[Path], allow_synthetic: bool) -> None:
        relative = safe_relative(entry)
        if relative.parts[0] != "images" or len(relative.parts) != 2:
            raise ValueError(f"Unexpected E4 image entry: {entry}")
        image_name = relative.name
        label = labels_root / f"{Path(image_name).stem}.txt"
        if image_name.startswith("cp_"):
            if not allow_synthetic:
                raise ValueError("Synthetic image entered E4 validation split")
            record = records_by_name.get(image_name)
            if record is None or not label.is_file():
                raise FileNotFoundError(f"E4 synthetic record or label is missing: image={image_name}, label={label}")
            donor_name = Path(str(record["source_image"])).name
            target_name = Path(str(record["target_image"])).name
            if donor_name not in original_train_names or target_name not in original_train_names:
                raise ValueError("E4 frozen donor or target is outside the Fold 1 original training membership")
            target_image = runtime_root / "images" / image_name
            target_label = runtime_root / "labels" / label.name
            render_e4_synthetic(target_image, image_root, mapping, record)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label, target_label)
            target.append(target_image)
            synthetic_names.append(image_name)
            return
        linked_image, _ = materialize_original_pair(runtime_root, image_root, config_root, mapping, image_name, label)
        target.append(linked_image)

    for entry in train_entries:
        materialize(entry, train_images, True)
    for entry in val_entries:
        materialize(entry, val_images, False)
    if len(synthetic_names) != 36 or len(set(synthetic_names)) != 36:
        raise ValueError("E4 runtime must contain exactly 36 unique synthetic training images")
    data_yaml = write_data_yaml(runtime_root, train_images, val_images)
    return data_yaml, train_images, val_images, {
        "synthetic_train": len(synthetic_names),
        "e4_payload_manifest_sha256": sha256_file(payload_root / "payload_manifest.json"),
        "e4_source_build_manifest_sha256": payload.get("source_build_manifest_sha256"),
    }


def build_model() -> tuple[object, dict]:
    from ultralytics import YOLO

    baseline_pretrained = YOLO("yolo11m.pt")
    initial_weight = Path(getattr(baseline_pretrained, "ckpt_path", "yolo11m.pt"))
    if not initial_weight.is_file():
        initial_weight = Path("yolo11m.pt")
    if CONFIG["variant"] == "E3_P2":
        model_yaml = Path("/kaggle/working/yolo11m_p2.yaml")
        model_yaml.write_text(CONFIG["p2_model_yaml"], encoding="utf-8")
        model = YOLO(str(model_yaml))
        model.load(str(initial_weight))
        return model, {
            "architecture": "YOLO11m native P2/P3/P4/P5 Detect",
            "model_yaml_sha256": CONFIG["p2_model_yaml_sha256"],
            "pretrained_source": str(initial_weight),
            "pretrained_source_sha256": sha256_file(initial_weight),
            "transfer": "Ultralytics partial load from official YOLO11m checkpoint; P2-specific parameters remain architecture-initialized.",
        }
    return baseline_pretrained, {
        "architecture": "YOLO11m",
        "model_yaml_sha256": None,
        "pretrained_source": str(initial_weight),
        "pretrained_source_sha256": sha256_file(initial_weight),
        "transfer": "Official YOLO11m pretrained checkpoint.",
    }


def run_e3_gpu_preflight(data_yaml: Path) -> dict | None:
    """Exercise the exact P2 GPU training path once before the full run."""
    if CONFIG["variant"] != "E3_P2":
        return None
    import gc
    import torch

    training = CONFIG["training"]
    preflight_root = Path("/kaggle/working/e3_p2_gpu_preflight")
    preflight_name = f"{CONFIG['run_name']}_one_epoch_preflight"
    preflight_dir = preflight_root / preflight_name
    if preflight_dir.exists():
        raise FileExistsError(f"Refusing to overwrite E3 GPU preflight: {preflight_dir}")
    model, initialization = build_model()
    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=training["imgsz"],
        batch=training["batch"],
        device="0",
        workers=training["workers"],
        project=str(preflight_root),
        name=preflight_name,
        exist_ok=False,
        pretrained=True,
        optimizer="SGD",
        seed=0,
        deterministic=True,
        patience=0,
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
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",
        auto_augment="randaugment",
        erasing=0.4,
        save=False,
        plots=False,
    )
    results = preflight_dir / "results.csv"
    if not results.is_file():
        raise FileNotFoundError("E3 GPU preflight did not write results.csv")
    result = {
        "status": "passed",
        "epochs": 1,
        "workers": training["workers"],
        "amp": True,
        "results_csv": str(results),
        "initialization": initialization,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def evaluate_full_image(
    pipeline: Path,
    config_root: Path,
    image_root: Path,
    baseline: Path,
    candidate: Path,
    run_dir: Path,
) -> dict:
    workspace = Path("/kaggle/working") / f"{CONFIG['run_name']}_evaluation_workspace"
    command = [
        sys.executable, str(pipeline),
        "--mode", "evaluate",
        "--config-root", str(config_root),
        "--image-dataset-root", str(image_root),
        "--output-root", str(workspace),
        "--run-name", CONFIG["run_name"],
        "--fold", "1",
        "--baseline-weights", str(baseline),
        "--candidate-weights", str(candidate),
        "--tile-size", "4096",
        "--overlap", "0.0",
        "--negative-ratio", "0.0",
        "--min-visible-fraction", "1.0",
        "--focus-classes", ",".join(CONFIG["focus_classes"]),
        "--epochs", str(CONFIG["training"]["epochs"]),
        "--imgsz", str(CONFIG["training"]["imgsz"]),
        "--batch", str(CONFIG["training"]["batch"]),
        "--predict-batch", str(CONFIG["evaluation_predict_batch"]),
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
    shutil.copytree(source, destination)
    baseline_report = json.loads((destination / "tile_evaluation" / "baseline_full.json").read_text(encoding="utf-8"))
    candidate_report = json.loads((destination / "tile_evaluation" / "tile_model_full.json").read_text(encoding="utf-8"))
    candidate_hash = sha256_file(candidate)
    if candidate_report.get("model_sha256") != candidate_hash:
        raise ValueError("Evaluation candidate report does not match saved best.pt")
    baseline_metric = float(baseline_report["mAP50_95"])
    if abs(baseline_metric - CONFIG["baseline_full_map50_95"]) > CONFIG["baseline_tolerance"]:
        raise ValueError(
            f"Fixed full-image baseline did not reproduce: expected={CONFIG['baseline_full_map50_95']}, actual={baseline_metric}"
        )
    per_class = candidate_report.get("per_class")
    if not isinstance(per_class, dict):
        raise ValueError("Candidate full-image report has no per-class metrics")
    focus = {}
    for class_name in CONFIG["focus_classes"]:
        item = per_class.get(class_name)
        if not isinstance(item, dict):
            raise ValueError(f"Candidate report is missing focus class: {class_name}")
        focus[class_name] = {
            "AP50_95": item.get("AP50_95"),
            "recall_conf025_iou50": item.get("recall_conf025_iou50"),
        }
    return {
        "entry": "l1_tile_pipeline.py --mode evaluate",
        "pipeline_sha256": CONFIG["pipeline_sha256"],
        "mode": "one original full-image crop per source validation image",
        "baseline_full_mAP50_95": baseline_metric,
        "candidate_best_mAP50_95": float(candidate_report["mAP50_95"]),
        "candidate_minus_fixed_baseline_mAP50_95": float(candidate_report["mAP50_95"]) - baseline_metric,
        "candidate_focus": focus,
        "baseline_model_sha256": baseline_report.get("model_sha256"),
        "candidate_best_model_sha256": candidate_hash,
        "tile_size": 4096,
        "overlap": 0.0,
    }


def copy_evidence(run_dir: Path, data_yaml: Path, config_root: Path) -> None:
    evidence = run_dir / "dataset_config"
    evidence.mkdir(parents=True, exist_ok=False)
    shutil.copy2(data_yaml, evidence / "data.yaml")
    shutil.copy2(config_root / "splits" / "folds" / "fold_01" / "train.txt", evidence / "original_fold01_train.txt")
    shutil.copy2(config_root / "splits" / "folds" / "fold_01" / "val.txt", evidence / "original_fold01_val.txt")


def main() -> None:
    payload_root, manifest_path = resolve_dev_config_payload()
    config_root, manifest = restore_dev_config(payload_root, manifest_path)
    image_root = resolve_image_dataset(manifest)
    baseline = resolve_baseline_weight()
    payload_details = {"synthetic_train": 0, "e4_payload_manifest_sha256": None}
    if CONFIG["variant"] == "E4_COPYPASTE":
        e4_payload_root, e4_payload = resolve_e4_payload()
    else:
        e4_payload_root, e4_payload = None, None
    install_dependencies()
    pipeline = Path("/kaggle/working/l1_tile_pipeline.py")
    pipeline.write_text(PIPELINE_SOURCE, encoding="utf-8")
    if CONFIG["variant"] == "E3_P2":
        data_yaml, train_images, val_images, payload_details = build_e3_runtime(config_root, manifest, image_root)
    else:
        data_yaml, train_images, val_images, payload_details = build_e4_runtime(
            config_root, manifest, image_root, e4_payload_root, e4_payload
        )
    gpu_preflight = run_e3_gpu_preflight(data_yaml)
    model, initialization = build_model()
    run_root = Path("/kaggle/working/runs")
    run_dir = run_root / CONFIG["run_name"]
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {run_dir}")
    training = CONFIG["training"]
    model.train(
        data=str(data_yaml),
        epochs=training["epochs"],
        imgsz=training["imgsz"],
        batch=training["batch"],
        device="0",
        workers=training["workers"],
        project=str(run_root),
        name=CONFIG["run_name"],
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
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",
        auto_augment="randaugment",
        erasing=0.4,
        save_period=training["save_period"],
        plots=training["plots"],
    )
    best, last = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
    required = (best, last, run_dir / "args.yaml", run_dir / "results.csv")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Training output is incomplete: {missing}")
    evaluation = evaluate_full_image(pipeline, config_root, image_root, baseline, best, run_dir)
    copy_evidence(run_dir, data_yaml, config_root)
    requested = {
        "experiment": CONFIG["experiment"],
        "variant": CONFIG["variant"],
        "run_name": CONFIG["run_name"],
        "development_only": True,
        "auto_only": True,
        "frozen_test_set": None,
        "fold": 1,
        "split_counts": {"train": len(train_images), "val": len(val_images), "test_frozen": None},
        "runtime_pins": CONFIG["runtime_pins"],
        "baseline_weight": {"sha256": sha256_file(baseline), "fixed_full_mAP50_95": CONFIG["baseline_full_map50_95"]},
        "training": training,
        "initialization": initialization,
        "gpu_preflight": gpu_preflight,
        "dataset_payload": payload_details,
        "evaluation": evaluation,
        "gate": CONFIG["gate"],
    }
    write_json(run_dir / "requested_config.json", requested)
    write_json(run_dir / "environment.json", environment_payload())
    write_json(run_dir / "initialization.json", initialization)
    (run_dir / "README.md").write_text(
        f"# {CONFIG['experiment']}\n\n"
        "This is a single-seed Fold 1 development-only, auto_only screening run. "
        "The 198-image release-validation set is not mounted, read, linked, or evaluated. "
        "No Ultralytics time-hour limit is passed; per-epoch checkpoints are kept, while training-batch plots stay disabled for runtime stability.\n",
        encoding="utf-8",
    )
    archive = Path(shutil.make_archive(str(Path("/kaggle/working") / f"{CONFIG['run_name']}_results"), "zip", run_root, CONFIG["run_name"]))
    print(json.dumps({"run_dir": str(run_dir), "archive": str(archive), "best_sha256": sha256_file(best)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create local Kaggle staging packages for the approved L1 E3/E4 parallel Fold 1 jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "l1_e3_e4_parallel_kaggle_20260823")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the planned paths without creating staging files.")
    return parser.parse_args(argv)


def expected_kernel_definitions() -> list[dict[str, Any]]:
    return [
        {
            "variant": "E3_P2",
            "experiment": "L1 E3 YOLO11m native P2 head Fold 1",
            # Kaggle canonicalizes "Fold01" to "fold01"; keep the local id aligned
            # so a future revision updates this submitted kernel instead of creating a fork.
            "slug": "l1-ryw18-y11m-p2-e3-fold01",
            "title": "L1 RYW18 Y11m P2 E3 Fold01",
            "run_name": "dianjienie_medium_yolo11m_p2_640_kaggle_seed0_oofconsensusv1_fold01",
            "dataset_sources": [IMAGE_DATASET_REF, DEV_CONFIG_DATASET_REF, BASELINE_DATASET_REF],
            "training": {"epochs": 300, "time_hours": None, "imgsz": 640, "batch": 8, "workers": 0, "optimizer": "SGD", "seed": 0, "deterministic": True, "resume": False, "patience": 100, "save_period": 1, "plots": False},
            "evaluation_predict_batch": 8,
            "gate": {
                "screening_only": True,
                "continue_only_if": "original-image Fold 1 mAP50-95 improves by at least 0.02 and focus-class AP/Recall show the same-direction improvement; otherwise do not expand to Fold 2/3.",
                "comparison": "fixed full-image evaluator and baseline SHA-256 only",
            },
        },
        {
            "variant": "E4_COPYPASTE",
            "experiment": "L1 E4 Copy-Paste Fold 1",
            "slug": "l1-ryw18-y11m-cp-e4-fold01",
            "title": "L1 RYW18 Y11m CP E4 Fold01",
            "run_name": "dianjienie_medium_yolo11m_cp_640_kaggle_seed0_oofconsensusv1_fold01",
            "dataset_sources": [IMAGE_DATASET_REF, DEV_CONFIG_DATASET_REF, BASELINE_DATASET_REF, E4_DATASET_REF],
            "training": {"epochs": 300, "time_hours": None, "imgsz": 640, "batch": 8, "workers": 8, "optimizer": "SGD", "seed": 0, "deterministic": True, "resume": False, "patience": 100, "save_period": 1, "plots": False},
            "evaluation_predict_batch": 8,
            "gate": {
                "screening_only": True,
                "continue_only_if": "original-image Fold 1 mAP50-95 improves by at least 0.02 and rare-class AP/Recall improve without material regression; otherwise do not expand to Fold 2/3.",
                "comparison": "fixed full-image evaluator and baseline SHA-256 only",
            },
        },
    ]


def validate_inputs() -> dict[str, Any]:
    required = (PIPELINE_PATH, P2_YAML_PATH, P2_SMOKE_PATH, DEV_CONFIG_PAYLOAD / "config_archive_manifest.json", CP_SOURCE_ROOT / "build_manifest.json", CP_SOURCE_ROOT / "manifests" / "verification_report.json", BASELINE_WEIGHT)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required local inputs are missing: {missing}")
    smoke = read_json(P2_SMOKE_PATH)
    if smoke.get("status") != "passed" or smoke.get("gate") != "pass_local_smoke":
        raise ValueError("E3 P2 CPU smoke is not in the required passed state")
    if smoke.get("model_yaml_sha256") != sha256_file(P2_YAML_PATH):
        raise ValueError("E3 P2 model YAML no longer matches its passed smoke report")
    verification = read_json(CP_SOURCE_ROOT / "manifests" / "verification_report.json")
    if verification.get("status") != "passed" or verification.get("development_only") is not True:
        raise ValueError("E4 Copy-Paste view has not passed its development-only verification")
    fold_one = next((item for item in verification.get("folds", []) if item.get("fold") == 1), None)
    if fold_one != {"fold": 1, "original_hardlinks_verified": 1127, "synthetic_samples_verified": 36, "train_images": 787, "val_images": 376, "train_val_overlap": 0, "manifest_sha256": fold_one.get("manifest_sha256") if isinstance(fold_one, dict) else None}:
        raise ValueError("E4 Fold 1 verification values are not the approved 787/376/36 view")
    actual_weight_sha256 = sha256_file(BASELINE_WEIGHT)
    if actual_weight_sha256 != BASELINE_WEIGHT_SHA256:
        raise ValueError(f"Fixed baseline checksum mismatch: expected={BASELINE_WEIGHT_SHA256}, actual={actual_weight_sha256}")
    archive_manifest = read_json(DEV_CONFIG_PAYLOAD / "config_archive_manifest.json")
    if archive_manifest.get("source_manifest_sha256") != "349c3629f0ade7b8397b42a1cc3926a904daa2e5d117219381f6afbc03cd0e14":
        raise ValueError("Unexpected development config manifest hash")
    classes = [line.strip() for line in (CP_SOURCE_ROOT / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if tuple(classes) != EXPECTED_CLASSES:
        raise ValueError("E4 class order differs from the fixed RYW18 contract")
    return {
        "p2_yaml_sha256": sha256_file(P2_YAML_PATH),
        "p2_smoke_sha256": sha256_file(P2_SMOKE_PATH),
        "pipeline_sha256": sha256_file(PIPELINE_PATH),
        "dev_config_manifest_sha256": archive_manifest["source_manifest_sha256"],
        "baseline_weight_sha256": actual_weight_sha256,
        "e4_build_manifest_sha256": sha256_file(CP_SOURCE_ROOT / "build_manifest.json"),
        "e4_verification_report_sha256": sha256_file(CP_SOURCE_ROOT / "manifests" / "verification_report.json"),
    }


def copy_file_record(source: Path, target: Path, relative_path: str, records: list[dict[str, Any]]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    records.append({"relative_path": relative_path.replace("\\", "/"), "bytes": target.stat().st_size, "sha256": sha256_file(target)})


def create_e4_dataset_payload(payload_root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    if payload_root.exists():
        raise FileExistsError(f"Refusing to overwrite E4 Dataset payload: {payload_root}")
    records: list[dict[str, Any]] = []
    source_fold = CP_SOURCE_ROOT / "fold_01"
    label_sources = sorted((source_fold / "labels").glob("*.txt"))
    if len(label_sources) != 1163:
        raise ValueError(f"E4 Fold 1 should have 1163 labels, found {len(label_sources)}")
    for source in label_sources:
        copy_file_record(source, payload_root / "fold_01" / "labels" / source.name, f"fold_01/labels/{source.name}", records)
    synthetic_manifest = source_fold / "manifests" / "synthetic_samples.jsonl"
    synthetic_record_count = sum(1 for line in synthetic_manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    if synthetic_record_count != 36:
        raise ValueError(f"E4 Fold 1 should have 36 frozen synthetic records, found {synthetic_record_count}")
    fixed_files = (
        (source_fold / "splits" / "train.txt", "fold_01/splits/train.txt"),
        (source_fold / "splits" / "val.txt", "fold_01/splits/val.txt"),
        (source_fold / "classes.txt", "fold_01/classes.txt"),
        (source_fold / "data.yaml", "fold_01/data.yaml"),
        (source_fold / "manifests" / "fold_summary.json", "fold_01/manifests/fold_summary.json"),
        (source_fold / "manifests" / "synthetic_samples.jsonl", "fold_01/manifests/synthetic_samples.jsonl"),
        (CP_SOURCE_ROOT / "build_manifest.json", "source_evidence/build_manifest.json"),
        (CP_SOURCE_ROOT / "manifests" / "verification_report.json", "source_evidence/verification_report.json"),
        (CP_SOURCE_ROOT / "classes.txt", "source_evidence/classes.txt"),
    )
    for source, relative in fixed_files:
        copy_file_record(source, payload_root / relative, relative, records)
    payload = {
        "schema_version": 1,
        "experiment": "L1 E4 Copy-Paste Fold 1 Kaggle payload",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "auto_only": True,
        "frozen_test_set": None,
        "source_view": "dianjienie_ryw18_oofconsensus_v1_dev_only_r1",
        "classes": list(EXPECTED_CLASSES),
        "split_counts": {"train": 787, "val": 376, "synthetic_train": 36},
        "synthetic_image_delivery": "runtime reconstruction from frozen manifest coordinates and the existing development image Dataset; no source image bytes are uploaded",
        "source_build_manifest_sha256": evidence["e4_build_manifest_sha256"],
        "source_verification_report_sha256": evidence["e4_verification_report_sha256"],
        "files": records,
    }
    write_json(payload_root / "payload_manifest.json", payload)
    write_json(
        payload_root / "dataset-metadata.json",
        {"title": "dianjienie RYW18 OOF CP v1 Fold01", "id": E4_DATASET_REF, "licenses": [{"name": "other"}]},
    )
    (payload_root / "README.md").write_text(
        "# L1 E4 Copy-Paste Fold 1 Kaggle payload\n\n"
        "Contains only Fold 1 labels, splits, 36 frozen Copy-Paste geometry records, and verification evidence. "
        "The runtime deterministically recreates the 36 images from the existing `dianjienie` development image Dataset. "
        "The historical 198-image release-validation set is not included.\n",
        encoding="utf-8",
    )
    return payload


def create_e4_dataset_upload(upload_root: Path, payload_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Package the nested E4 payload in Kaggle's proven archive upload layout."""
    if upload_root.exists():
        raise FileExistsError(f"Refusing to overwrite E4 Dataset upload package: {upload_root}")
    upload_root.mkdir(parents=True)
    archive_name = "e4_cp_f01_payload_v1.zip"
    archive_path = upload_root / archive_name
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(payload_root.rglob("*")):
            if path.is_file() and path.name != "dataset-metadata.json":
                archive.write(path, path.relative_to(payload_root).as_posix())
    archive_manifest = {
        "schema_version": 1,
        "experiment": "L1 E4 Copy-Paste Fold 1 Kaggle archive wrapper",
        "archive": archive_name,
        "sha256": sha256_file(archive_path),
        "payload_manifest_sha256": sha256_file(payload_root / "payload_manifest.json"),
        "payload_file_count": len(payload["files"]),
        "development_only": True,
        "frozen_test_set": None,
    }
    write_json(upload_root / "e4_cp_f01_archive_manifest.json", archive_manifest)
    write_json(
        upload_root / "dataset-metadata.json",
        {"title": "dianjienie RYW18 OOF CP v1 Fold01", "id": E4_DATASET_REF, "licenses": [{"name": "other"}]},
    )
    (upload_root / "README.md").write_text(
        "# L1 E4 Copy-Paste Fold 1 Kaggle Dataset upload\n\n"
        "The ZIP contains the verified nested payload. The runtime first uses Kaggle's expanded archive when available, "
        "then has a checksum-verified extraction fallback. The frozen historical validation set is not included.\n",
        encoding="utf-8",
    )
    return archive_manifest


def build_kernel_script(definition: dict[str, Any], evidence: dict[str, Any]) -> str:
    config = {
        **definition,
        "runtime_pins": RUNTIME_PINS,
        "expected_classes": list(EXPECTED_CLASSES),
        "focus_classes": list(FOCUS_CLASSES),
        "baseline_weight_name": BASELINE_WEIGHT_NAME,
        "baseline_weight_sha256": BASELINE_WEIGHT_SHA256,
        "baseline_full_map50_95": BASELINE_FULL_MAP50_95,
        "baseline_tolerance": BASELINE_TOLERANCE,
        "pipeline_sha256": evidence["pipeline_sha256"],
        "p2_model_yaml": P2_YAML_PATH.read_text(encoding="utf-8") if definition["variant"] == "E3_P2" else None,
        "p2_model_yaml_sha256": evidence["p2_yaml_sha256"] if definition["variant"] == "E3_P2" else None,
    }
    return KERNEL_TEMPLATE.replace("__CONFIG_JSON_LITERAL__", repr(json.dumps(config, ensure_ascii=False))).replace(
        "__PIPELINE_SOURCE_LITERAL__", repr(PIPELINE_PATH.read_text(encoding="utf-8"))
    )


def create_kernel_staging(kernel_root: Path, definition: dict[str, Any], script: str, evidence: dict[str, Any]) -> dict[str, Any]:
    if kernel_root.exists():
        raise FileExistsError(f"Refusing to overwrite kernel staging: {kernel_root}")
    kernel_root.mkdir(parents=True)
    code_file = f"{definition['slug']}.py"
    metadata = {
        "id": f"{OWNER}/{definition['slug']}",
        "title": definition["title"],
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": definition["dataset_sources"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    write_json(kernel_root / "kernel-metadata.json", metadata)
    (kernel_root / code_file).write_text(script, encoding="utf-8", newline="\n")
    requested = {
        "experiment": definition["experiment"],
        "variant": definition["variant"],
        "status": "prepared_pending_submission",
        "run_name": definition["run_name"],
        "development_only": True,
        "auto_only": True,
        "frozen_test_set": None,
        "fold": 1,
        "gpu_slots_authorized": 2,
        "runtime_pins": RUNTIME_PINS,
        "training": definition["training"],
        "fixed_baseline": {"sha256": BASELINE_WEIGHT_SHA256, "full_image_mAP50_95": BASELINE_FULL_MAP50_95},
        "pipeline_sha256": evidence["pipeline_sha256"],
        "gate": definition["gate"],
        "submission": "not yet performed by this preparation script",
    }
    write_json(kernel_root / "requested_config.json", requested)
    (kernel_root / "README.md").write_text(
        f"# {definition['experiment']} Kaggle staging\n\n"
        "This package is development-only and auto_only. It must run alone in one Kaggle GPU slot. "
        "Training uses the 300-epoch and early-stopping protocol without an Ultralytics time-hour cap or batch-plot callback, then runs the fixed original-image evaluator.\n",
        encoding="utf-8",
    )
    return {"kernel": metadata["id"], "root": str(kernel_root), "code_sha256": sha256_file(kernel_root / code_file)}


def validate_staging(
    output_root: Path,
    definitions: list[dict[str, Any]],
    e4_payload: dict[str, Any],
    e4_archive: dict[str, Any],
) -> dict[str, Any]:
    if e4_payload["split_counts"] != {"train": 787, "val": 376, "synthetic_train": 36}:
        raise ValueError("Generated E4 payload has incorrect split counts")
    archive_path = output_root / "e4_dataset_upload" / str(e4_archive["archive"])
    if not archive_path.is_file() or sha256_file(archive_path) != e4_archive["sha256"]:
        raise ValueError("Generated E4 upload archive failed integrity validation")
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    required_archive_members = {"payload_manifest.json", "fold_01/splits/train.txt", "fold_01/splits/val.txt"}
    if not required_archive_members.issubset(names):
        raise ValueError("Generated E4 upload archive is missing required payload members")
    kernels: list[dict[str, Any]] = []
    for definition in definitions:
        kernel_root = output_root / "kernels" / definition["variant"].lower()
        metadata = read_json(kernel_root / "kernel-metadata.json")
        requested = read_json(kernel_root / "requested_config.json")
        source = (kernel_root / metadata["code_file"]).read_text(encoding="utf-8")
        compile(source, str(kernel_root / metadata["code_file"]), "exec")
        if metadata["id"] != f"{OWNER}/{definition['slug']}":
            raise ValueError(f"Kernel metadata id mismatch: {definition['variant']}")
        if requested["variant"] != definition["variant"] or requested["training"].get("time_hours") is not None:
            raise ValueError(f"Kernel requested configuration mismatch: {definition['variant']}")
        if requested["training"].get("plots") is not False:
            raise ValueError(f"Kernel must disable unstable batch plotting: {definition['variant']}")
        if "time=training[\"time_hours\"]" in source:
            raise ValueError(f"Kernel must not pass an Ultralytics time-hour cap: {definition['variant']}")
        if "plots=training[\"plots\"]" not in source:
            raise ValueError(f"Kernel must use its explicit batch-plot setting: {definition['variant']}")
        # The code contains only prohibitions and no-access evidence wording for
        # the frozen historical validation set, never a mounted source.
        if '"frozen_test_set": None' not in source:
            raise ValueError("Kernel does not encode frozen-test isolation")
        if "release-validation" in " ".join(metadata["dataset_sources"]).lower():
            raise ValueError("Forbidden frozen dataset attached to kernel")
        kernels.append({"variant": definition["variant"], "kernel": metadata["id"], "code_sha256": sha256_file(kernel_root / metadata["code_file"])})
    return {
        "status": "passed",
        "kernels": kernels,
        "e4_payload_files": len(e4_payload["files"]),
        "e4_archive_sha256": e4_archive["sha256"],
        "frozen_test_set": None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    evidence = validate_inputs()
    definitions = expected_kernel_definitions()
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Prepare exactly two user-authorized parallel Kaggle GPU jobs: E3 P2 Fold 1 and E4 Copy-Paste Fold 1.",
        "max_parallel_gpu_jobs": 2,
        "jobs": [{"variant": item["variant"], "kernel": f"{OWNER}/{item['slug']}", "training_time_hours": item["training"]["time_hours"]} for item in definitions],
        "development_only": True,
        "frozen_test_set": None,
        "evidence": evidence,
    }
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "plan": plan}, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing staging root: {output_root}")
    output_root.mkdir(parents=True)
    write_json(output_root / "submission_plan.json", plan)
    e4_payload = create_e4_dataset_payload(output_root / "e4_dataset_payload", evidence)
    e4_archive = create_e4_dataset_upload(output_root / "e4_dataset_upload", output_root / "e4_dataset_payload", e4_payload)
    jobs = [create_kernel_staging(output_root / "kernels" / item["variant"].lower(), item, build_kernel_script(item, evidence), evidence) for item in definitions]
    validation = validate_staging(output_root, definitions, e4_payload, e4_archive)
    write_json(output_root / "validation_report.json", validation)
    write_json(output_root / "prepared_jobs.json", {"jobs": jobs, "e4_dataset": E4_DATASET_REF, "validation": validation})
    print(json.dumps({"output_root": str(output_root), "jobs": jobs, "e4_dataset": E4_DATASET_REF, "validation": validation}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
