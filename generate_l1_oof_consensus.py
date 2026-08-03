#!/usr/bin/env python
"""Generate traceable L1 OOF consensus candidates without touching frozen test data.

The script predicts each development image only with the two models whose
corresponding fold did not train on that image.  ``ryw.pt`` is retained as a
teacher-only source: it can add provenance, but cannot by itself create an
automatic label.  Historical anchor labels are copied into a new candidate
view and are never deleted or reclassified here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from workspace_config import resolve_path


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_ANCHOR_ROOT = resolve_path(
    "l1_anchor_ryw18_v1", Path(r"E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1")
)
DEFAULT_RUNS_ROOT = resolve_path("model_runs", Path(r"E:/PythonProject/models/runs"))
DEFAULT_TOOL_ROOT = resolve_path(
    "model_auto_label_tool", Path(r"E:/PythonProject/Tools/ModelAutoLabelTool")
)
DEFAULT_RYW_WEIGHTS = DEFAULT_TOOL_ROOT / "weights" / "yolov5" / "ryw.pt"
MODEL_ARCHITECTURES = ("yolo11m", "yolo26m")


@dataclass(frozen=True)
class Assignment:
    """One development image and the only fold model allowed to predict it."""

    key: str
    image_path: Path
    source_label_path: Path
    fold: int
    width: int
    height: int


@dataclass(frozen=True)
class Detection:
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass
class Candidate:
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    yolo11m: Detection
    yolo26m: Detection
    teacher_matches: list[Detection]


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Generate L1 development-set OOF consensus pseudo-label candidates. "
            "The frozen test list and its images are intentionally never read."
        )
    )
    parser.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--tool-root", type=Path, default=DEFAULT_TOOL_ROOT)
    parser.add_argument("--ryw-weights", type=Path, default=DEFAULT_RYW_WEIGHTS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_ROOT / f"l1_oof_consensus_{timestamp}_v1",
        help="New output directory. It must not already exist.",
    )
    parser.add_argument("--device", default="0", help="Ultralytics/YOLOv5 device, e.g. 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf-min", type=float, default=0.15)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--consensus-iou", type=float, default=0.55)
    parser.add_argument("--label-overlap-iou", type=float, default=0.55)
    parser.add_argument("--accepted-confidence", type=float, default=0.70)
    parser.add_argument("--uncertain-confidence", type=float, default=0.40)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--expected-dev-images", type=int, default=1127)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run static/checkpoint preflight only; do not write any output or infer images.",
    )
    return parser.parse_args()


def print_info(message: str) -> None:
    print(f"[l1-oof] {message}", flush=True)


def fail(message: str) -> None:
    raise RuntimeError(message)


def batched(items: list[Assignment], size: int) -> Iterator[list[Assignment]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str, anchor_root: Path) -> Path:
    raw = Path(value.strip().strip('"'))
    candidates = (raw, anchor_root / raw, anchor_root / "images" / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    fail(f"Image path in fold list does not exist: {value}")


def ordered_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, dict):
        try:
            return [str(raw_names[index]) for index in range(len(raw_names))]
        except KeyError:
            return [str(value) for _, value in sorted(raw_names.items(), key=lambda pair: int(pair[0]))]
    if isinstance(raw_names, (list, tuple)):
        return [str(item) for item in raw_names]
    fail(f"Unsupported model names type: {type(raw_names)!r}")


def read_anchor_classes(anchor_root: Path) -> list[str]:
    data_yaml = anchor_root / "data.yaml"
    if not data_yaml.is_file():
        fail(f"Missing anchor data.yaml: {data_yaml}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read data.yaml.") from exc
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = ordered_names(payload.get("names"))
    if len(names) != 18:
        fail(f"Anchor must contain 18 classes, found {len(names)}: {names}")
    return names


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def collect_assignments(anchor_root: Path, expected_total: int) -> tuple[list[Assignment], dict[int, list[Path]]]:
    images_root = (anchor_root / "images").resolve()
    labels_root = (anchor_root / "labels").resolve()
    if not images_root.is_dir() or not labels_root.is_dir():
        fail(f"Anchor needs images/ and labels/: {anchor_root}")

    assignments: list[Assignment] = []
    fold_lists: dict[int, list[Path]] = {}
    seen: dict[Path, int] = {}
    for fold in range(1, 4):
        fold_list = anchor_root / "splits" / "folds" / f"fold_{fold:02d}" / "val.txt"
        if not fold_list.is_file():
            fail(f"Missing OOF validation list for fold {fold}: {fold_list}")
        values = [line.strip() for line in fold_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not values:
            fail(f"Empty validation list for fold {fold}: {fold_list}")
        paths = [resolve_path(value, anchor_root) for value in values]
        if len(paths) != len(set(paths)):
            fail(f"Duplicate image path inside fold {fold} validation list.")
        fold_lists[fold] = paths
        for path in paths:
            if path in seen:
                fail(f"Development image occurs in two OOF folds: {path} ({seen[path]}, {fold})")
            seen[path] = fold
            try:
                relative = path.relative_to(images_root)
            except ValueError as exc:
                raise RuntimeError(f"Fold image is outside anchor images/: {path}") from exc
            source_label = labels_root / relative.with_suffix(".txt")
            if not source_label.is_file():
                fail(f"Missing historical anchor label for OOF image: {source_label}")
            width, height = image_size(path)
            if width <= 0 or height <= 0:
                fail(f"Invalid image size: {path}")
            assignments.append(
                Assignment(
                    key=relative.as_posix(),
                    image_path=path,
                    source_label_path=source_label,
                    fold=fold,
                    width=width,
                    height=height,
                )
            )
    if len(assignments) != expected_total:
        fail(
            f"Expected {expected_total} development images from the three validation folds, "
            f"found {len(assignments)}. Do not proceed with an unrecognized split."
        )
    return sorted(assignments, key=lambda item: item.key), fold_lists


def model_weight_path(runs_root: Path, architecture: str, fold: int) -> Path:
    run_name = f"dianjienie_medium_{architecture}_640_kaggle_seed0_cm0_anchorv1_fold{fold:02d}"
    return runs_root / run_name / "weights" / "best.pt"


def image_key_from_list_entry(value: str, anchor_images_root: Path) -> str:
    """Normalize local and Kaggle image paths to the anchor-relative key."""

    normalized = value.strip().strip('"').replace("\\", "/")
    marker = "/images/"
    if marker in normalized:
        return normalized.split(marker, maxsplit=1)[1]
    local = Path(normalized)
    if local.is_file():
        try:
            return local.resolve().relative_to(anchor_images_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"Run validation image is outside anchor images/: {value}") from exc
    if normalized.startswith("images/"):
        return normalized.removeprefix("images/")
    fail(f"Cannot normalize archived validation-list image key: {value}")


def validate_run_lists(
    runs_root: Path, fold_lists: dict[int, list[Path]], anchor_root: Path
) -> dict[str, dict[str, Any]]:
    """Verify only train-independent OOF val lists; frozen test lists are not read."""

    metadata: dict[str, dict[str, Any]] = {}
    for architecture in MODEL_ARCHITECTURES:
        architecture_info: dict[str, Any] = {}
        for fold in range(1, 4):
            weight = model_weight_path(runs_root, architecture, fold)
            if not weight.is_file():
                fail(f"Missing completed {architecture} Fold {fold} checkpoint: {weight}")
            run_val = weight.parents[1] / "dataset_config" / "val.txt"
            if not run_val.is_file():
                fail(f"Missing archived validation list next to checkpoint: {run_val}")
            entries = [line.strip() for line in run_val.read_text(encoding="utf-8").splitlines() if line.strip()]
            anchor_images_root = (anchor_root / "images").resolve()
            recorded = {image_key_from_list_entry(value, anchor_images_root) for value in entries}
            expected = {path.relative_to(anchor_images_root).as_posix() for path in fold_lists[fold]}
            if len(recorded) != len(entries):
                fail(f"{architecture} Fold {fold} archived val list contains duplicate image keys.")
            if recorded != expected:
                fail(
                    f"{architecture} Fold {fold} archived val list differs from the anchor OOF list. "
                    "Refusing to use a non-identical validation fold."
                )
            architecture_info[f"fold_{fold:02d}"] = {
                "weight": str(weight),
                "sha256": sha256_file(weight),
                "validated_val_images": len(recorded),
            }
        metadata[architecture] = architecture_info
    return metadata


def clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def validate_ultralytics_class_names(runs_root: Path, expected_names: list[str]) -> None:
    from ultralytics import YOLO

    for architecture in MODEL_ARCHITECTURES:
        for fold in range(1, 4):
            weight = model_weight_path(runs_root, architecture, fold)
            print_info(f"checking {architecture} Fold {fold} class order")
            model = YOLO(str(weight))
            names = ordered_names(model.names)
            del model
            clear_cuda()
            if names != expected_names:
                fail(
                    f"Class order mismatch in {architecture} Fold {fold}. "
                    f"Expected {expected_names}; got {names}"
                )


def load_teacher(tool_root: Path, weights: Path, device: str, imgsz: int, conf_min: float, nms_iou: float) -> Any:
    """Load the legacy teacher through the project tool without auto-installing packages."""

    os.environ["YOLOv5_AUTOINSTALL"] = "False"
    tool_root = tool_root.resolve()
    if not tool_root.is_dir():
        fail(f"ModelAutoLabelTool root is missing: {tool_root}")
    if str(tool_root) not in sys.path:
        sys.path.insert(0, str(tool_root))
    from auto_label import yolo_checkpoint_load_compat
    from adapters.yolov5 import YoloV5Detector

    with yolo_checkpoint_load_compat():
        return YoloV5Detector(
            weights=weights,
            device=device,
            threshold=conf_min,
            imgsz=imgsz,
            iou=nms_iou,
        )


def validate_teacher_class_names(args: argparse.Namespace, expected_names: list[str]) -> dict[str, Any]:
    if not args.ryw_weights.is_file():
        fail(f"Missing ryw.pt teacher weight: {args.ryw_weights}")
    print_info("checking ryw.pt class order through ModelAutoLabelTool")
    teacher = load_teacher(
        args.tool_root,
        args.ryw_weights,
        "cpu",
        args.imgsz,
        args.conf_min,
        args.nms_iou,
    )
    names = ordered_names(teacher.model.names)
    del teacher
    clear_cuda()
    if names != expected_names:
        fail(f"ryw.pt class order differs from anchor: expected {expected_names}; got {names}")
    return {"weight": str(args.ryw_weights), "sha256": sha256_file(args.ryw_weights), "class_order_verified": True}


def xywh_to_xyxy(row: list[float]) -> tuple[float, float, float, float]:
    _, center_x, center_y, width, height = row[:5]
    return (
        max(0.0, center_x - width / 2),
        max(0.0, center_y - height / 2),
        min(1.0, center_x + width / 2),
        min(1.0, center_y + height / 2),
    )


def xyxy_to_xywh(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def read_historical_labels(path: Path, class_count: int) -> list[Detection]:
    labels: list[Detection] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        values = raw_line.split()
        if len(values) != 5:
            fail(f"Historical label is not five-column YOLO at {path}:{line_number}")
        try:
            parsed = [float(value) for value in values]
        except ValueError as exc:
            raise RuntimeError(f"Non-numeric historical label at {path}:{line_number}") from exc
        class_id = int(parsed[0])
        if parsed[0] != class_id or not 0 <= class_id < class_count:
            fail(f"Invalid historical class id at {path}:{line_number}")
        if not all(0.0 <= value <= 1.0 for value in parsed[1:]):
            fail(f"Out-of-range historical normalized coordinate at {path}:{line_number}")
        labels.append(Detection(class_id, xywh_to_xyxy(parsed), 1.0))
    return labels


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def normalise_detection(row: Any, width: int, height: int, class_count: int) -> Detection:
    values = [float(value) for value in row[:6]]
    x1, y1, x2, y2, confidence, raw_class = values
    class_id = int(raw_class)
    if raw_class != class_id or not 0 <= class_id < class_count:
        fail(f"Model produced invalid class id: {raw_class}")
    xyxy = (
        min(1.0, max(0.0, x1 / width)),
        min(1.0, max(0.0, y1 / height)),
        min(1.0, max(0.0, x2 / width)),
        min(1.0, max(0.0, y2 / height)),
    )
    if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
        fail(f"Model produced a degenerate normalized box: {xyxy}")
    return Detection(class_id, xyxy, confidence)


def serialise_detection(detection: Detection) -> dict[str, Any]:
    return {
        "class_id": detection.class_id,
        "xyxy_norm": [round(value, 8) for value in detection.xyxy],
        "confidence": round(detection.confidence, 8),
    }


def predict_ultralytics(
    architecture: str,
    assignments: list[Assignment],
    args: argparse.Namespace,
    predictions: dict[str, dict[str, list[Detection]]],
    jsonl_path: Path,
    class_count: int,
) -> None:
    from ultralytics import YOLO

    for fold in range(1, 4):
        allowed = [assignment for assignment in assignments if assignment.fold == fold]
        weight = model_weight_path(args.runs_root, architecture, fold)
        source_name = f"{architecture}_oof_fold_{fold:02d}"
        print_info(f"predicting {len(allowed)} OOF images with {source_name}")
        model = YOLO(str(weight))
        with jsonl_path.open("a", encoding="utf-8") as output:
            completed = 0
            for chunk in batched(allowed, args.batch):
                results = model.predict(
                    source=[str(item.image_path) for item in chunk],
                    imgsz=args.imgsz,
                    conf=args.conf_min,
                    iou=args.nms_iou,
                    max_det=args.max_det,
                    device=args.device,
                    batch=len(chunk),
                    verbose=False,
                    stream=False,
                )
                if len(results) != len(chunk):
                    fail(f"{source_name} returned {len(results)} results for a batch of {len(chunk)} images")
                for assignment, result in zip(chunk, results, strict=True):
                    rows = [] if result.boxes is None else result.boxes.data.detach().cpu().tolist()
                    detections = [normalise_detection(row, assignment.width, assignment.height, class_count) for row in rows]
                    predictions[assignment.key][architecture] = detections
                    output.write(
                        json.dumps(
                            {
                                "image": assignment.key,
                                "fold": assignment.fold,
                                "source": source_name,
                                "oof": True,
                                "detections": [serialise_detection(item) for item in detections],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                completed += len(chunk)
                if completed % max(args.batch * 10, 80) == 0 or completed == len(allowed):
                    print_info(f"{source_name}: {completed}/{len(allowed)} images")
        del model
        clear_cuda()


def predict_teacher(
    assignments: list[Assignment],
    args: argparse.Namespace,
    predictions: dict[str, dict[str, list[Detection]]],
    jsonl_path: Path,
    class_count: int,
) -> None:
    print_info(f"predicting {len(assignments)} development images with ryw.pt teacher (not OOF)")
    teacher = load_teacher(
        args.tool_root,
        args.ryw_weights,
        args.device,
        args.imgsz,
        args.conf_min,
        args.nms_iou,
    )
    import torch

    with jsonl_path.open("a", encoding="utf-8") as output, torch.no_grad():
        completed = 0
        for chunk in batched(assignments, args.batch):
            results = teacher.model([str(item.image_path) for item in chunk], size=args.imgsz)
            rows_by_image = results.xyxy
            if len(rows_by_image) != len(chunk):
                fail(f"ryw.pt returned {len(rows_by_image)} results for a batch of {len(chunk)} images")
            for assignment, rows in zip(chunk, rows_by_image, strict=True):
                detections = [
                    normalise_detection(row.detach().cpu().tolist(), assignment.width, assignment.height, class_count)
                    for row in rows
                ]
                predictions[assignment.key]["ryw_teacher"] = detections
                output.write(
                    json.dumps(
                        {
                            "image": assignment.key,
                            "fold": assignment.fold,
                            "source": "ryw_teacher",
                            "oof": False,
                            "role": "teacher_only",
                            "detections": [serialise_detection(item) for item in detections],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            completed += len(chunk)
            if completed % max(args.batch * 10, 80) == 0 or completed == len(assignments):
                print_info(f"ryw_teacher: {completed}/{len(assignments)} images")
    del teacher
    clear_cuda()


def build_consensus(
    yolo11m: list[Detection], yolo26m: list[Detection], teacher: list[Detection], threshold: float
) -> list[Candidate]:
    """Require a direct same-class IoU match between the two OOF architectures."""

    raw_pairs: list[Candidate] = []
    for first in yolo11m:
        for second in yolo26m:
            if first.class_id != second.class_id or iou(first.xyxy, second.xyxy) < threshold:
                continue
            canonical = first if first.confidence >= second.confidence else second
            raw_pairs.append(
                Candidate(
                    class_id=first.class_id,
                    xyxy=canonical.xyxy,
                    confidence=min(first.confidence, second.confidence),
                    yolo11m=first,
                    yolo26m=second,
                    teacher_matches=[],
                )
            )

    kept: list[Candidate] = []
    for candidate in sorted(raw_pairs, key=lambda item: item.confidence, reverse=True):
        duplicate = next(
            (
                existing
                for existing in kept
                if existing.class_id == candidate.class_id and iou(existing.xyxy, candidate.xyxy) >= threshold
            ),
            None,
        )
        if duplicate is None:
            kept.append(candidate)

    for candidate in kept:
        candidate.teacher_matches = [
            detection
            for detection in teacher
            if detection.class_id == candidate.class_id and iou(detection.xyxy, candidate.xyxy) >= threshold
        ]
    return kept


def historical_relation(candidate: Candidate, historical: list[Detection], threshold: float) -> tuple[str, float]:
    # Either OOF box overlapping history is enough to suppress an automatic addition.
    oof_boxes = (candidate.xyxy, candidate.yolo11m.xyxy, candidate.yolo26m.xyxy)
    same_class = [
        iou(box, item.xyxy)
        for box in oof_boxes
        for item in historical
        if item.class_id == candidate.class_id
    ]
    other_class = [
        iou(box, item.xyxy)
        for box in oof_boxes
        for item in historical
        if item.class_id != candidate.class_id
    ]
    max_same = max(same_class, default=0.0)
    max_other = max(other_class, default=0.0)
    if max_same >= threshold:
        return "existing_supported", max_same
    if max_other >= threshold:
        return "existing_class_conflict", max_other
    return "new_candidate", max(max_same, max_other)


def confidence_status(confidence: float, args: argparse.Namespace) -> str:
    if confidence >= args.accepted_confidence:
        return "accepted"
    if confidence >= args.uncertain_confidence:
        return "uncertain"
    return "rejected"


def candidate_record(
    assignment: Assignment,
    candidate: Candidate,
    class_names: list[str],
    relation: str,
    overlap: float,
    status: str,
    action: str,
) -> dict[str, Any]:
    return {
        "image": assignment.key,
        "fold": assignment.fold,
        "class_id": candidate.class_id,
        "class_name": class_names[candidate.class_id],
        "xyxy_norm": [round(value, 8) for value in candidate.xyxy],
        "confidence": round(candidate.confidence, 8),
        "confidence_definition": "min(yolo11m_oof_confidence, yolo26m_oof_confidence)",
        "status": status,
        "historical_relation": relation,
        "max_historical_iou": round(overlap, 8),
        "action": action,
        "evidence": {
            "yolo11m_oof": serialise_detection(candidate.yolo11m),
            "yolo26m_oof": serialise_detection(candidate.yolo26m),
            "ryw_teacher_matches": [serialise_detection(item) for item in candidate.teacher_matches],
        },
    }


def write_confidence_label(path: Path, candidate: Candidate) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    center_x, center_y, width, height = xyxy_to_xywh(candidate.xyxy)
    with path.open("a", encoding="utf-8") as output:
        output.write(
            f"{candidate.class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f} {candidate.confidence:.8f}\n"
        )


def append_training_candidate(path: Path, candidate: Candidate) -> None:
    center_x, center_y, width, height = xyxy_to_xywh(candidate.xyxy)
    with path.open("a", encoding="utf-8") as output:
        if path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n"):
            output.write("\n")
        output.write(f"{candidate.class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}\n")


def write_outputs(
    output_root: Path,
    assignments: list[Assignment],
    predictions: dict[str, dict[str, list[Detection]]],
    class_names: list[str],
    args: argparse.Namespace,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    labels_with_conf = output_root / "labels_with_conf"
    training_labels = output_root / "labels_consensus_training_candidate_5col"
    decisions_path = output_root / "decisions.jsonl"
    labels_with_conf.mkdir(parents=True)
    training_labels.mkdir(parents=True)
    (output_root / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    counters: Counter[str] = Counter()
    by_class: defaultdict[str, Counter[str]] = defaultdict(Counter)
    with decisions_path.open("w", encoding="utf-8") as decision_output:
        for assignment in assignments:
            output_label = training_labels / Path(assignment.key).with_suffix(".txt")
            output_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(assignment.source_label_path, output_label)
            historical = read_historical_labels(assignment.source_label_path, len(class_names))
            source_predictions = predictions[assignment.key]
            candidates = build_consensus(
                source_predictions["yolo11m"],
                source_predictions["yolo26m"],
                source_predictions["ryw_teacher"],
                args.consensus_iou,
            )
            counters["oof_consensus_total"] += len(candidates)
            accepted_added: list[Candidate] = []
            for candidate in candidates:
                relation, overlap = historical_relation(candidate, historical, args.label_overlap_iou)
                status = confidence_status(candidate.confidence, args)
                if relation == "existing_supported":
                    action = "preserve_historical_label"
                elif relation == "existing_class_conflict":
                    action = "record_conflict_no_auto_reclass"
                elif status == "accepted":
                    duplicate = any(
                        item.class_id == candidate.class_id
                        and iou(item.xyxy, candidate.xyxy) >= args.label_overlap_iou
                        for item in accepted_added
                    )
                    action = "duplicate_new_candidate_suppressed" if duplicate else "add_to_training_candidate_view"
                    if not duplicate:
                        accepted_added.append(candidate)
                        append_training_candidate(output_label, candidate)
                else:
                    action = "candidate_only_not_added_to_training_view"

                record = candidate_record(
                    assignment,
                    candidate,
                    class_names,
                    relation,
                    overlap,
                    status,
                    action,
                )
                decision_output.write(json.dumps(record, ensure_ascii=False) + "\n")
                counters[f"relation:{relation}"] += 1
                counters[f"status:{status}"] += 1
                counters[f"action:{action}"] += 1
                by_class[class_names[candidate.class_id]][f"relation:{relation}"] += 1
                by_class[class_names[candidate.class_id]][f"status:{status}"] += 1
                if relation == "new_candidate":
                    confidence_path = labels_with_conf / Path(assignment.key).with_suffix(".txt")
                    write_confidence_label(confidence_path, candidate)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "L1 OOF two-architecture consensus candidate generation",
        "frozen_test_access": "not read; not inferred; not included",
        "images": len(assignments),
        "class_names": class_names,
        "config": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "model_confidence_min": args.conf_min,
            "model_nms_iou": args.nms_iou,
            "oof_consensus_iou": args.consensus_iou,
            "historical_label_overlap_iou": args.label_overlap_iou,
            "accepted_confidence": args.accepted_confidence,
            "uncertain_confidence": args.uncertain_confidence,
            "consensus_confidence": "min of direct YOLO11m/YOLO26m OOF pair",
        },
        "safety": {
            "historical_labels": "copied unchanged into five-column candidate view; never deleted or reclassified",
            "automatic_addition_rule": "direct same-class YOLO11m/YOLO26m OOF IoU match only",
            "ryw_role": "teacher-only evidence; never sufficient for an automatic addition",
            "labels_with_conf": "new candidates only, six columns including confidence; decisions.jsonl retains status and provenance",
            "training_candidate_view": "historical labels plus accepted new candidates; not reviewed ground truth",
        },
        "counts": dict(sorted(counters.items())),
        "by_class": {name: dict(sorted(count.items())) for name, count in sorted(by_class.items())},
        "preflight": preflight,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "output_root": str(output_root),
        "files": {
            "classes": "classes.txt",
            "predictions": ["predictions/yolo11m_oof.jsonl", "predictions/yolo26m_oof.jsonl", "predictions/ryw_teacher.jsonl"],
            "decisions": "decisions.jsonl",
            "labels_with_conf": "labels_with_conf/",
            "training_candidate_labels": "labels_consensus_training_candidate_5col/",
            "summary": "summary.json",
        },
        "hashes": {path.name: sha256_file(path) for path in (output_root / "predictions").glob("*.jsonl")},
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def write_report(output_root: Path, summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    config = summary["config"]
    lines = [
        "# L1 OOF 双模型一致性候选报告",
        "",
        "## 边界",
        "",
        "- 输入仅为三个开发集验证折；冻结测试集未读取、未推理、未参与任何阈值或模型选择。",
        "- 每张图片仅由其未参与训练的 YOLO11m 与 YOLO26m 对应 Fold 权重推理。",
        "- `ryw.pt` 仅为教师证据，不能单独或与单一 OOF 模型一起生成自动新增标签。",
        "- 历史锚点标签不删除、不改类；同类重叠只记录为支持，异类重叠只记录为冲突。",
        "",
        "## 规则",
        "",
        f"- 模型最低候选置信度：`{config['model_confidence_min']:.2f}`；NMS IoU：`{config['model_nms_iou']:.2f}`。",
        f"- 新增候选必须是 YOLO11m 与 YOLO26m 直接同类匹配，IoU >= `{config['oof_consensus_iou']:.2f}`。",
        f"- 候选置信度为两模型置信度最小值；`accepted >= {config['accepted_confidence']:.2f}`，`uncertain >= {config['uncertain_confidence']:.2f}`，其余为 `rejected`。",
        f"- 与历史框 IoU >= `{config['historical_label_overlap_iou']:.2f}` 的候选不会自动新增。",
        "",
        "## 结果",
        "",
        f"- 开发集图片：`{summary['images']}`。",
        f"- 两 OOF 架构一致框：`{counts.get('oof_consensus_total', 0)}`。",
        f"- 可自动加入候选训练视图的新框：`{counts.get('action:add_to_training_candidate_view', 0)}`。",
        f"- 历史同类支持：`{counts.get('relation:existing_supported', 0)}`；历史异类冲突：`{counts.get('relation:existing_class_conflict', 0)}`。",
        f"- 新候选 accepted/uncertain/rejected：`{counts.get('status:accepted', 0)}` / `{counts.get('status:uncertain', 0)}` / `{counts.get('status:rejected', 0)}`（状态总数含与历史重叠的候选）。",
        "",
        "## 产物说明",
        "",
        "- `predictions/*.jsonl`：三来源逐图原始预测与 OOF 标识。",
        "- `decisions.jsonl`：每个一致框的历史关系、动作、两 OOF 模型证据与教师证据。",
        "- `labels_with_conf/`：仅新增候选的六列标签，保留置信度；不用于直接训练。",
        "- `labels_consensus_training_candidate_5col/`：完整历史五列标签加 accepted 新框，仅是后续审计/训练候选视图，不是人工审查真值。",
    ]
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.conf_min <= 1.0:
        fail("--conf-min must be in [0, 1].")
    if not 0.0 < args.consensus_iou <= 1.0 or not 0.0 < args.label_overlap_iou <= 1.0:
        fail("IoU thresholds must be in (0, 1].")
    if not 0.0 <= args.uncertain_confidence <= args.accepted_confidence <= 1.0:
        fail("Require 0 <= uncertain confidence <= accepted confidence <= 1.")
    if args.batch <= 0 or args.imgsz <= 0 or args.max_det <= 0:
        fail("--batch, --imgsz, and --max-det must be positive.")

    args.anchor_root = args.anchor_root.resolve()
    args.runs_root = args.runs_root.resolve()
    args.tool_root = args.tool_root.resolve()
    args.ryw_weights = args.ryw_weights.resolve()
    args.output_root = args.output_root.resolve()

    print_info("reading 18-class anchor and three development validation folds")
    class_names = read_anchor_classes(args.anchor_root)
    assignments, fold_lists = collect_assignments(args.anchor_root, args.expected_dev_images)
    print_info(
        "development images by OOF fold: "
        + ", ".join(f"Fold {fold}={len(fold_lists[fold])}" for fold in range(1, 4))
    )
    print_info("validating completed checkpoint paths and archived validation lists")
    run_metadata = validate_run_lists(args.runs_root, fold_lists, args.anchor_root)
    validate_ultralytics_class_names(args.runs_root, class_names)
    teacher_metadata = validate_teacher_class_names(args, class_names)
    preflight = {
        "anchor_root": str(args.anchor_root),
        "development_images": len(assignments),
        "fold_sizes": {f"fold_{fold:02d}": len(fold_lists[fold]) for fold in range(1, 4)},
        "ultralytics_runs": run_metadata,
        "ryw_teacher": teacher_metadata,
        "frozen_test_access": "intentionally not read",
    }
    if args.dry_run:
        print(json.dumps({"status": "preflight_ok", **preflight}, ensure_ascii=False, indent=2))
        return 0
    if args.output_root.exists():
        fail(f"Refusing to overwrite existing output directory: {args.output_root}")

    predictions_dir = args.output_root / "predictions"
    predictions_dir.mkdir(parents=True)
    (args.output_root / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    predictions: dict[str, dict[str, list[Detection]]] = {assignment.key: {} for assignment in assignments}
    predict_ultralytics(
        "yolo11m",
        assignments,
        args,
        predictions,
        predictions_dir / "yolo11m_oof.jsonl",
        len(class_names),
    )
    predict_ultralytics(
        "yolo26m",
        assignments,
        args,
        predictions,
        predictions_dir / "yolo26m_oof.jsonl",
        len(class_names),
    )
    predict_teacher(
        assignments,
        args,
        predictions,
        predictions_dir / "ryw_teacher.jsonl",
        len(class_names),
    )
    for assignment in assignments:
        missing = {"yolo11m", "yolo26m", "ryw_teacher"} - set(predictions[assignment.key])
        if missing:
            fail(f"Missing inference source(s) for {assignment.key}: {sorted(missing)}")
    summary = write_outputs(args.output_root, assignments, predictions, class_names, args, preflight)
    write_report(args.output_root, summary)
    print_info(f"complete: {args.output_root}")
    print_info(f"accepted additions: {summary['counts'].get('action:add_to_training_candidate_view', 0)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[l1-oof] ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
