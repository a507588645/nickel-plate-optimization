#!/usr/bin/env python3
"""Validate one preselected L1 final candidate on the frozen release split once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO


EXPECTED_DATA_VERSION = "oofconsensusv1"
DEFAULT_SELECTED_MODEL = "yolo11m.pt"
DEFAULT_EXPECTED_TEST_IMAGES = 198
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the one-time frozen release validation for the preselected L1 candidate."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--access-registry",
        type=Path,
        default=Path(__file__).resolve().parent / "l1_release_validation_access_registry.json",
        help="Append-only registry that prevents automatic reuse of the same frozen split.",
    )
    parser.add_argument("--selected-model", default=DEFAULT_SELECTED_MODEL)
    parser.add_argument("--expected-test-images", type=int, default=DEFAULT_EXPECTED_TEST_IMAGES)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot parse JSON: {path}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return loaded


def normalize_names(value: Any, nc: int) -> list[str]:
    if isinstance(value, list):
        names = [str(item) for item in value]
    elif isinstance(value, dict):
        normalized: dict[int, str] = {}
        for key, item in value.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid class-name key: {key!r}") from error
            normalized[index] = str(item)
        names = [normalized[index] for index in range(nc) if index in normalized]
        if len(names) != nc:
            raise ValueError("Class names do not cover every class id")
    else:
        raise ValueError("data.yaml names must be a list or mapping")
    if len(names) != nc:
        raise ValueError(f"Class-name count {len(names)} does not equal nc={nc}")
    return names


def read_classes(anchor_root: Path) -> list[str]:
    classes_path = anchor_root / "classes.txt"
    if not classes_path.is_file():
        raise FileNotFoundError(f"Missing classes file: {classes_path}")
    names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError("classes.txt is empty")
    return names


def required_run_paths(run_dir: Path) -> dict[str, Path]:
    relative_paths = {
        "final_weight": "weights/final_fixed_epoch.pt",
        "last_weight": "weights/last.pt",
        "best_weight": "weights/best.pt",
        "release_weight": "release_weight.json",
        "requested_config": "requested_config.json",
        "data_yaml": "dataset_config/data.yaml",
        "train_list": "dataset_config/train.txt",
        "val_list": "dataset_config/val.txt",
        "epoch_selection": "dataset_config/epoch_selection_summary.json",
        "environment": "environment.json",
        "results": "results.csv",
        "readme": "README.md",
    }
    paths = {key: run_dir / relative for key, relative in relative_paths.items()}
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Final run is missing required evidence: {missing}")
    return paths


def preflight_run(run_dir: Path, anchor_root: Path, selected_model: str) -> dict[str, Any]:
    paths = required_run_paths(run_dir)
    config = json_load(paths["requested_config"])
    release_weight = json_load(paths["release_weight"])
    if config.get("run_name") != run_dir.name:
        raise ValueError("requested_config run_name does not match --run-dir")
    if config.get("model") != selected_model:
        raise ValueError(f"This release gate is fixed to {selected_model}; run uses {config.get('model')!r}")
    if config.get("data_version") != EXPECTED_DATA_VERSION:
        raise ValueError("Unexpected L1 data version")
    if config.get("development_only") is not True or config.get("final_development_training") is not True:
        raise ValueError("Run is not an all-development final-training artifact")
    if config.get("fold") is not None:
        raise ValueError("A fold run cannot enter final release validation")
    training = config.get("training")
    selection = config.get("epoch_selection")
    if not isinstance(training, dict) or not isinstance(selection, dict):
        raise ValueError("Final run lacks training or epoch-selection metadata")
    fixed_epoch = release_weight.get("fixed_epoch")
    if fixed_epoch != training.get("epochs") or fixed_epoch != selection.get("selected_epoch"):
        raise ValueError("Release weight epoch does not match the frozen selection")
    if release_weight.get("source") != "weights/last.pt" or release_weight.get("release_weight") != "weights/final_fixed_epoch.pt":
        raise ValueError("Release-weight provenance is not the fixed final checkpoint")
    final_weight_sha256 = sha256_file(paths["final_weight"])
    if release_weight.get("sha256") != final_weight_sha256:
        raise ValueError("final_fixed_epoch.pt SHA-256 does not match release_weight.json")

    try:
        data = yaml.safe_load(paths["data_yaml"].read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("Cannot parse archived data.yaml") from error
    if not isinstance(data, dict):
        raise ValueError("Archived data.yaml root must be a mapping")
    try:
        nc = int(data["nc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Archived data.yaml lacks a valid nc") from error
    archived_names = normalize_names(data.get("names"), nc)
    anchor_names = read_classes(anchor_root)
    if archived_names != anchor_names:
        raise ValueError("Archived class order does not match the RYW18 anchor classes")
    return {
        "paths": paths,
        "requested_config": config,
        "release_weight": release_weight,
        "final_weight_sha256": final_weight_sha256,
        "class_names": anchor_names,
        "nc": nc,
    }


def read_release_images(test_list: Path, anchor_root: Path, expected_count: int) -> list[Path]:
    image_root = (anchor_root / "images").resolve()
    values = [line.strip() for line in test_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != expected_count:
        raise ValueError(f"Frozen test image count is {len(values)}, expected {expected_count}")
    images: list[Path] = []
    for value in values:
        candidate = Path(value)
        image = (candidate if candidate.is_absolute() else anchor_root / candidate).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Frozen test image missing: {image}")
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported frozen-test image suffix: {image}")
        try:
            image.relative_to(image_root)
        except ValueError as error:
            raise ValueError(f"Frozen test image escapes anchor images: {image}") from error
        images.append(image)
    if len({image.name for image in images}) != len(images):
        raise ValueError("Frozen test images have duplicate filenames; flat validation view is unsafe")
    return images


def audit_labels(images: list[Path], anchor_root: Path, nc: int) -> tuple[list[Path], dict[str, Any]]:
    labels_root = (anchor_root / "labels").resolve()
    labels: list[Path] = []
    empty_files = 0
    boxes = 0
    per_class = [0] * nc
    digest = hashlib.sha256()
    for image in images:
        label = labels_root / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Frozen test label missing: {label}")
        text = label.read_text(encoding="utf-8")
        digest.update(image.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            empty_files += 1
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO field count in {label}:{line_number}")
            try:
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
            except ValueError as error:
                raise ValueError(f"Non-numeric YOLO label in {label}:{line_number}") from error
            if not 0 <= class_id < nc:
                raise ValueError(f"Out-of-range class id in {label}:{line_number}")
            x_center, y_center, width, height = values
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite YOLO coordinate in {label}:{line_number}")
            if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                raise ValueError(f"Out-of-range YOLO coordinate in {label}:{line_number}")
            boxes += 1
            per_class[class_id] += 1
        labels.append(label)
    return labels, {
        "label_files": len(labels),
        "empty_label_files": empty_files,
        "boxes": boxes,
        "per_class_instances": per_class,
        "labels_content_sha256": digest.hexdigest(),
    }


def hardlink(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError as error:
        raise RuntimeError(f"Cannot create hard link: {source} -> {target}") from error


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "records": []}
    registry = json_load(path)
    if registry.get("schema_version") != 1 or not isinstance(registry.get("records"), list):
        raise ValueError(f"Invalid release-validation access registry: {path}")
    return registry


def write_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def register_access(registry_path: Path, anchor_root: Path, test_list: Path, test_sha256: str, run_dir: Path, output_root: Path) -> str:
    registry = load_registry(registry_path)
    anchor = str(anchor_root.resolve())
    for record in registry["records"]:
        if record.get("anchor_root") == anchor and record.get("test_list_sha256") == test_sha256:
            raise RuntimeError(
                "This frozen split already has a recorded release-validation access; automatic reruns are prohibited"
            )
    record_id = uuid.uuid4().hex
    registry["records"].append(
        {
            "record_id": record_id,
            "state": "started",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "anchor_root": anchor,
            "test_list": str(test_list.resolve()),
            "test_list_sha256": test_sha256,
            "run_dir": str(run_dir.resolve()),
            "output_root": str(output_root.resolve()),
        }
    )
    write_registry(registry_path, registry)
    return record_id


def finish_access(registry_path: Path, record_id: str, state: str, detail: str) -> None:
    registry = load_registry(registry_path)
    for record in registry["records"]:
        if record.get("record_id") == record_id:
            record["state"] = state
            record["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            record["detail"] = detail
            write_registry(registry_path, registry)
            return
    raise RuntimeError("Release-validation registry record disappeared")


def build_per_class_metrics(results: Any, names: list[str]) -> list[dict[str, Any]]:
    positions = {int(class_id): index for index, class_id in enumerate(results.box.ap_class_index)}
    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(names):
        instances = int(results.nt_per_class[class_id])
        images = int(results.nt_per_image[class_id])
        position = positions.get(class_id)
        row: dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "images": images,
            "instances": instances,
        }
        if position is None:
            if instances:
                raise RuntimeError(f"Class {class_id} has targets but no AP metric")
            row.update({"status": "not_evaluable_no_ground_truth", "precision": None, "recall": None, "map50": None, "map50_95": None})
        else:
            precision, recall, map50, map50_95 = results.box.class_result(position)
            row.update(
                {
                    "status": "evaluated",
                    "precision": float(precision),
                    "recall": float(recall),
                    "map50": float(map50),
                    "map50_95": float(map50_95),
                }
            )
        rows.append(row)
    return rows


def write_report(output_root: Path, metrics: dict[str, Any]) -> None:
    overall = metrics["metrics"]
    content = "\n".join(
        [
            "# L1 Final Release Validation",
            "",
            "- Scope: one frozen historical release-validation pass for the preselected candidate.",
            f"- Run: `{metrics['run_dir']}`",
            f"- Release weight SHA-256: `{metrics['final_weight_sha256']}`",
            f"- Frozen images: {metrics['release_images']}",
            f"- Frozen labels: {metrics['label_audit']['label_files']} files / {metrics['label_audit']['boxes']} boxes",
            f"- Precision: {overall['precision']:.6f}",
            f"- Recall: {overall['recall']:.6f}",
            f"- mAP50: {overall['map50']:.6f}",
            f"- mAP50-95: {overall['map50_95']:.6f}",
            "",
            "This result confirms only the already selected candidate. It must not be used to select between architectures or tune thresholds.",
            "",
        ]
    )
    (output_root / "report.md").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.expected_test_images <= 0 or args.batch <= 0 or args.imgsz <= 0:
        raise ValueError("expected-test-images, batch, and imgsz must be positive")
    run_dir = args.run_dir.resolve()
    anchor_root = args.anchor_root.resolve()
    output_root = args.output_root.resolve()
    registry_path = args.access_registry.resolve()
    test_list = anchor_root / "splits" / "test.txt"
    if not run_dir.is_dir() or not anchor_root.is_dir() or not test_list.is_file():
        raise FileNotFoundError("run-dir, anchor-root, or frozen test list is missing")
    evidence = preflight_run(run_dir, anchor_root, args.selected_model)
    preflight = {
        "scope": "preselected_l1_final_release_validation_once",
        "run_dir": str(run_dir),
        "anchor_root": str(anchor_root),
        "selected_model": args.selected_model,
        "release_weight": str(evidence["paths"]["final_weight"]),
        "final_weight_sha256": evidence["final_weight_sha256"],
        "expected_test_images": args.expected_test_images,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "frozen_test_images_read": False,
    }
    if args.dry_run:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite release-validation output: {output_root}")

    images = read_release_images(test_list, anchor_root, args.expected_test_images)
    labels, label_audit = audit_labels(images, anchor_root, evidence["nc"])
    test_list_sha256 = sha256_file(test_list)
    record_id = register_access(registry_path, anchor_root, test_list, test_list_sha256, run_dir, output_root)
    try:
        output_root.mkdir(parents=True)
        view_root = output_root / "dataset_view"
        view_images = view_root / "images"
        view_labels = view_root / "labels"
        view_images.mkdir(parents=True)
        view_labels.mkdir(parents=True)
        manifest: list[dict[str, Any]] = []
        for image, label in zip(images, labels, strict=True):
            linked_image = view_images / image.name
            linked_label = view_labels / label.name
            hardlink(image, linked_image)
            hardlink(label, linked_label)
            manifest.append(
                {
                    "source_image": str(image),
                    "source_image_size": image.stat().st_size,
                    "source_image_mtime_ns": image.stat().st_mtime_ns,
                    "source_label": str(label),
                    "linked_image": str(linked_image),
                    "linked_label": str(linked_label),
                }
            )
        val_list = view_root / "val.txt"
        val_list.write_text("\n".join(str(view_images / image.name) for image in images) + "\n", encoding="utf-8")
        local_data = {
            "path": str(view_root),
            "train": str(val_list),
            "val": str(val_list),
            "nc": evidence["nc"],
            "names": {index: name for index, name in enumerate(evidence["class_names"])},
        }
        local_data_path = output_root / "data_local_release.yaml"
        local_data_path.write_text(yaml.safe_dump(local_data, allow_unicode=False, sort_keys=False), encoding="utf-8")
        (output_root / "dataset_view_manifest.json").write_text(
            json.dumps(
                {
                    "scope": preflight["scope"],
                    "test_list": str(test_list),
                    "test_list_sha256": test_list_sha256,
                    "files": manifest,
                    "label_audit": label_audit,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = YOLO(evidence["paths"]["final_weight"])
        model_names = normalize_names(model.names, evidence["nc"])
        if model_names != evidence["class_names"]:
            raise ValueError("Release model class order does not match the RYW18 anchor")
        results = model.val(
            data=str(local_data_path),
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
            name="ultralytics_release_val",
            exist_ok=False,
        )
        metrics = {
            **preflight,
            "frozen_test_images_read": True,
            "access_registry": str(registry_path),
            "access_record_id": record_id,
            "test_list": str(test_list),
            "test_list_sha256": test_list_sha256,
            "release_images": len(images),
            "label_audit": label_audit,
            "local_view_root": str(view_root),
            "local_val_list": str(val_list),
            "ultralytics": __import__("ultralytics").__version__,
            "torch": torch.__version__,
            "metrics": {
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
                "map50": float(results.box.map50),
                "map50_95": float(results.box.map),
            },
            "per_class": build_per_class_metrics(results, evidence["class_names"]),
        }
        metrics_path = output_root / "metrics_per_class_release.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(local_data_path, output_root / "data.yaml")
        write_report(output_root, metrics)
        finish_access(registry_path, record_id, "complete", str(metrics_path))
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    except BaseException as error:
        finish_access(registry_path, record_id, "failed_after_access", f"{type(error).__name__}: {error}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
