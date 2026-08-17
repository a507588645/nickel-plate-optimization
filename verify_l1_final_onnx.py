#!/usr/bin/env python3
"""Export a selected L1 final weight and compare PyTorch and ONNX on dev-only images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


EXPECTED_DATA_VERSION = "oofconsensusv1"
DEFAULT_SELECTED_MODEL = "yolo11m.pt"
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one selected L1 final candidate to ONNX and compare it with PyTorch on dev-only images."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selected-model", default=DEFAULT_SELECTED_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1, help="Static ONNX export requires batch=1.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--pt-device", default="cpu")
    parser.add_argument("--min-match-iou", type=float, default=0.90)
    parser.add_argument("--min-global-match-coverage", type=float, default=0.95)
    parser.add_argument("--max-image-count-delta", type=int, default=1)
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
        indexed: dict[int, str] = {}
        for key, item in value.items():
            try:
                indexed[int(key)] = str(item)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid class-name key: {key!r}") from error
        if set(indexed) != set(range(nc)):
            raise ValueError("Class-name mapping does not cover every class id")
        names = [indexed[index] for index in range(nc)]
    else:
        raise ValueError("data.yaml names must be a list or mapping")
    if len(names) != nc:
        raise ValueError(f"Class-name count {len(names)} does not equal nc={nc}")
    return names


def preflight_run(run_dir: Path, selected_model: str) -> dict[str, Any]:
    required = {
        "weight": run_dir / "weights" / "final_fixed_epoch.pt",
        "release_weight": run_dir / "release_weight.json",
        "requested_config": run_dir / "requested_config.json",
        "data_yaml": run_dir / "dataset_config" / "data.yaml",
        "epoch_selection": run_dir / "dataset_config" / "epoch_selection_summary.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Final run is missing required export evidence: {missing}")
    config = json_load(required["requested_config"])
    release_weight = json_load(required["release_weight"])
    if config.get("run_name") != run_dir.name:
        raise ValueError("requested_config run_name does not match --run-dir")
    if config.get("model") != selected_model:
        raise ValueError(f"This ONNX gate is fixed to {selected_model}; run uses {config.get('model')!r}")
    if config.get("data_version") != EXPECTED_DATA_VERSION:
        raise ValueError("Unexpected L1 data version")
    if config.get("development_only") is not True or config.get("final_development_training") is not True:
        raise ValueError("Run is not an all-development final-training artifact")
    if config.get("fold") is not None:
        raise ValueError("A fold run cannot enter final ONNX verification")
    training = config.get("training")
    selection = config.get("epoch_selection")
    if not isinstance(training, dict) or not isinstance(selection, dict):
        raise ValueError("Final run lacks training or epoch-selection metadata")
    if release_weight.get("fixed_epoch") != training.get("epochs") or release_weight.get("fixed_epoch") != selection.get("selected_epoch"):
        raise ValueError("Release weight epoch does not match the frozen selection")
    if release_weight.get("source") != "weights/last.pt" or release_weight.get("release_weight") != "weights/final_fixed_epoch.pt":
        raise ValueError("Release-weight provenance is not the fixed final checkpoint")
    weight_sha256 = sha256_file(required["weight"])
    if release_weight.get("sha256") != weight_sha256:
        raise ValueError("final_fixed_epoch.pt SHA-256 does not match release_weight.json")
    try:
        data = yaml.safe_load(required["data_yaml"].read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("Cannot parse archived data.yaml") from error
    if not isinstance(data, dict):
        raise ValueError("Archived data.yaml root must be a mapping")
    try:
        nc = int(data["nc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Archived data.yaml lacks a valid nc") from error
    return {
        "paths": required,
        "config": config,
        "weight_sha256": weight_sha256,
        "nc": nc,
        "names": normalize_names(data.get("names"), nc),
    }


def read_dev_images(development_root: Path, sample_count: int) -> list[Path]:
    if (development_root / "splits" / "test.txt").exists():
        raise ValueError("Development-only view unexpectedly contains test.txt")
    image_root = (development_root / "images").resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Development images directory missing: {image_root}")
    images = sorted(
        (path.resolve() for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    if len(images) < sample_count:
        raise ValueError(f"Only {len(images)} dev images exist, fewer than sample-count={sample_count}")
    if len({image.name for image in images}) != len(images):
        raise ValueError("Development images have duplicate filenames")
    return images[:sample_count]


def hardlink(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError as error:
        raise RuntimeError(f"Cannot create hard link: {source} -> {target}") from error


def detections(result: Any) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    rows = [
        {"class_id": int(class_id), "confidence": float(confidence), "xyxy": [float(value) for value in coords]}
        for coords, confidence, class_id in zip(xyxy, confidences, classes, strict=True)
    ]
    return sorted(rows, key=lambda row: (row["class_id"], -row["confidence"], row["xyxy"]))


def box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def compare_detections(
    pt_rows: list[dict[str, Any]],
    onnx_rows: list[dict[str, Any]],
    min_match_iou: float,
) -> dict[str, Any]:
    candidates: list[tuple[float, float, int, int]] = []
    for pt_index, pt_row in enumerate(pt_rows):
        for onnx_index, onnx_row in enumerate(onnx_rows):
            if pt_row["class_id"] != onnx_row["class_id"]:
                continue
            iou = box_iou(pt_row["xyxy"], onnx_row["xyxy"])
            if iou >= min_match_iou:
                candidates.append((iou, -abs(pt_row["confidence"] - onnx_row["confidence"]), pt_index, onnx_index))
    candidates.sort(reverse=True)
    used_pt: set[int] = set()
    used_onnx: set[int] = set()
    matched: list[dict[str, Any]] = []
    for iou, _negative_conf_delta, pt_index, onnx_index in candidates:
        if pt_index in used_pt or onnx_index in used_onnx:
            continue
        used_pt.add(pt_index)
        used_onnx.add(onnx_index)
        pt_row = pt_rows[pt_index]
        onnx_row = onnx_rows[onnx_index]
        box_delta = max(abs(first - second) for first, second in zip(pt_row["xyxy"], onnx_row["xyxy"], strict=True))
        conf_delta = abs(pt_row["confidence"] - onnx_row["confidence"])
        matched.append(
            {
                "pt_index": pt_index,
                "onnx_index": onnx_index,
                "class_id": pt_row["class_id"],
                "iou": iou,
                "box_abs_delta": box_delta,
                "confidence_abs_delta": conf_delta,
            }
        )
    unmatched_pt = sorted(set(range(len(pt_rows))) - used_pt)
    unmatched_onnx = sorted(set(range(len(onnx_rows))) - used_onnx)

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return float(ordered[round((len(ordered) - 1) * fraction)])

    box_deltas = [row["box_abs_delta"] for row in matched]
    confidence_deltas = [row["confidence_abs_delta"] for row in matched]
    max_box_delta = max((row["box_abs_delta"] for row in matched), default=0.0)
    max_conf_delta = max((row["confidence_abs_delta"] for row in matched), default=0.0)
    min_iou = min((row["iou"] for row in matched), default=None)
    return {
        "pt_detections": len(pt_rows),
        "onnx_detections": len(onnx_rows),
        "matched_detections": len(matched),
        "match_coverage": len(matched) / max(len(pt_rows), len(onnx_rows), 1),
        "image_count_delta": abs(len(pt_rows) - len(onnx_rows)),
        "unmatched_pt_count": len(unmatched_pt),
        "unmatched_onnx_count": len(unmatched_onnx),
        "unmatched_pt_indices_preview": unmatched_pt[:20],
        "unmatched_onnx_indices_preview": unmatched_onnx[:20],
        "min_matched_iou": min_iou,
        "box_abs_delta_p50": percentile(box_deltas, 0.50),
        "box_abs_delta_p95": percentile(box_deltas, 0.95),
        "max_box_abs_delta": max_box_delta,
        "confidence_abs_delta_p50": percentile(confidence_deltas, 0.50),
        "confidence_abs_delta_p95": percentile(confidence_deltas, 0.95),
        "max_confidence_abs_delta": max_conf_delta,
    }


def write_report(output_root: Path, report: dict[str, Any]) -> None:
    aggregate = report["aggregate"]
    content = "\n".join(
        [
            "# L1 ONNX Consistency Verification",
            "",
            "- Scope: deterministic development-only image sample; frozen release images were not used.",
            f"- Run: `{report['run_dir']}`",
            f"- Release weight SHA-256: `{report['weight_sha256']}`",
            f"- ONNX SHA-256: `{report['onnx_sha256']}`",
            f"- Samples: {aggregate['images']}",
            f"- Passed: {aggregate['passed']}",
            f"- Global same-class match coverage: {aggregate['global_match_coverage']:.6f}",
            f"- Maximum per-image count delta: {aggregate['max_image_count_delta']}",
            f"- Maximum box absolute delta: {aggregate['max_box_abs_delta']:.8f}",
            f"- Maximum confidence absolute delta: {aggregate['max_confidence_abs_delta']:.8f}",
            "",
            "The semantic gate uses same-class matches at the configured IoU threshold, global match coverage, and per-image count delta.",
            "",
        ]
    )
    (output_root / "report.md").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.imgsz <= 0 or args.sample_count <= 0 or args.max_det <= 0 or args.batch <= 0:
        raise ValueError("imgsz, sample-count, batch, and max-det must be positive")
    if args.batch != 1:
        raise ValueError("This static ONNX consistency gate requires --batch 1")
    if not (
        0.0 <= args.conf <= 1.0
        and 0.0 < args.iou <= 1.0
        and 0.0 < args.min_match_iou <= 1.0
        and 0.0 <= args.min_global_match_coverage <= 1.0
    ):
        raise ValueError("Invalid confidence or IoU threshold")
    if args.max_image_count_delta < 0:
        raise ValueError("max-image-count-delta must be non-negative")
    run_dir = args.run_dir.resolve()
    development_root = args.development_root.resolve()
    output_root = args.output_root.resolve()
    if not run_dir.is_dir() or not development_root.is_dir():
        raise FileNotFoundError("run-dir or development-root is missing")
    evidence = preflight_run(run_dir, args.selected_model)
    images = read_dev_images(development_root, args.sample_count)
    sample_manifest = [
        {"path": str(image), "sha256": sha256_file(image), "bytes": image.stat().st_size}
        for image in images
    ]
    preflight = {
        "scope": "development_only_onnx_consistency",
        "run_dir": str(run_dir),
        "development_root": str(development_root),
        "selected_model": args.selected_model,
        "weight": str(evidence["paths"]["weight"]),
        "weight_sha256": evidence["weight_sha256"],
        "imgsz": args.imgsz,
        "sample_count": args.sample_count,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "semantic_gate": {
            "min_match_iou": args.min_match_iou,
            "min_global_match_coverage": args.min_global_match_coverage,
            "max_image_count_delta": args.max_image_count_delta,
        },
        "pt_device": args.pt_device,
        "onnx_device": "cpu",
        "frozen_release_images_used": False,
        "samples": sample_manifest,
    }
    if args.dry_run:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite ONNX verification output: {output_root}")
    try:
        import onnx
        import onnxruntime
    except ModuleNotFoundError as error:
        raise RuntimeError("ONNX and onnxruntime must be installed in the active environment") from error

    output_root.mkdir(parents=True)
    model_input = output_root / "model_input"
    model_input.mkdir()
    linked_weight = model_input / "final_fixed_epoch.pt"
    hardlink(evidence["paths"]["weight"], linked_weight)
    pt_model = YOLO(linked_weight)
    if normalize_names(pt_model.names, evidence["nc"]) != evidence["names"]:
        raise ValueError("Release model class order does not match archived RYW18 data.yaml")
    exported = pt_model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=False,
        simplify=False,
        half=False,
        device=args.pt_device,
    )
    onnx_path = Path(str(exported)).resolve()
    if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
        raise FileNotFoundError(f"ONNX export did not produce a file: {onnx_path}")
    onnx.checker.check_model(str(onnx_path))
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if not session.get_inputs() or not session.get_outputs():
        raise ValueError("ONNX Runtime session has no inputs or outputs")

    common_predict = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "batch": args.batch,
        "half": False,
        "verbose": False,
        "save": False,
        "stream": False,
    }
    onnx_model = YOLO(onnx_path, task="detect")
    pt_results = []
    onnx_results = []
    # Passing a path one at a time prevents Ultralytics' list loader from batching all samples into a static ONNX input.
    for image in images:
        pt_prediction = pt_model.predict(source=str(image), **common_predict, device=args.pt_device)
        onnx_prediction = onnx_model.predict(source=str(image), **common_predict, device="cpu")
        if len(pt_prediction) != 1 or len(onnx_prediction) != 1:
            raise RuntimeError(f"Single-image inference returned an unexpected result count for {image}")
        pt_results.append(pt_prediction[0])
        onnx_results.append(onnx_prediction[0])

    comparisons: list[dict[str, Any]] = []
    for image, pt_result, onnx_result in zip(images, pt_results, onnx_results, strict=True):
        comparison = compare_detections(
            detections(pt_result), detections(onnx_result), args.min_match_iou
        )
        comparisons.append({"image": str(image), **comparison})
    pt_total = sum(item["pt_detections"] for item in comparisons)
    onnx_total = sum(item["onnx_detections"] for item in comparisons)
    matched_total = sum(item["matched_detections"] for item in comparisons)
    global_match_coverage = matched_total / max(pt_total, onnx_total, 1)
    max_image_count_delta = max((item["image_count_delta"] for item in comparisons), default=0)
    aggregate = {
        "images": len(comparisons),
        "passed": global_match_coverage >= args.min_global_match_coverage
        and max_image_count_delta <= args.max_image_count_delta,
        "pt_detections": pt_total,
        "onnx_detections": onnx_total,
        "matched_detections": matched_total,
        "global_match_coverage": global_match_coverage,
        "max_image_count_delta": max_image_count_delta,
        "unmatched_pt": sum(item["unmatched_pt_count"] for item in comparisons),
        "unmatched_onnx": sum(item["unmatched_onnx_count"] for item in comparisons),
        "max_box_abs_delta": max((item["max_box_abs_delta"] for item in comparisons), default=0.0),
        "max_confidence_abs_delta": max((item["max_confidence_abs_delta"] for item in comparisons), default=0.0),
    }
    report = {
        **preflight,
        "onnx_path": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "onnxruntime_providers": session.get_providers(),
        "aggregate": aggregate,
        "per_image": comparisons,
    }
    report_path = output_root / "onnx_consistency.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_root, report)
    print(json.dumps({"output": str(report_path), "aggregate": aggregate, "semantic_gate": preflight["semantic_gate"]}, ensure_ascii=False, indent=2))
    return 0 if aggregate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
