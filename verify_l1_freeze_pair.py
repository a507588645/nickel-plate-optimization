#!/usr/bin/env python3
"""Verify the paired Fold-1 YOLO11m freeze ablation before final training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "l1_freeze_pair_verification_20260828"
EXPECTED_SPLIT_COUNTS = {
    "train": 751,
    "val": 376,
    "test_frozen": None,
    "audited_development_images": 1127,
    "audited_candidate_boxes": 15671,
}
EXPECTED_RUNTIME = {
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0",
    "numpy": "1.26.4",
    "ultralytics": "8.4.66",
}
FOCUS_CLASSES = ("wuzi", "waterstain", "airhole", "bubble", "greenstain", "watermark")
COMMON_TRAINING_KEYS = (
    "epochs",
    "imgsz",
    "batch",
    "device",
    "workers",
    "exist_ok",
    "pretrained",
    "optimizer",
    "seed",
    "deterministic",
    "patience",
    "val",
    "cache",
    "resume",
    "close_mosaic",
    "amp",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "box",
    "cls",
    "dfl",
    "iou",
    "max_det",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "copy_paste_mode",
    "auto_augment",
    "erasing",
    "save_period",
    "plots",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "校验 L1 YOLO11m Freeze10 与两阶段不冻结对照的实际产物、统一复评和自动闸门。"
        )
    )
    parser.add_argument("--control-run-dir", type=Path, required=True)
    parser.add_argument("--freeze10-run-dir", type=Path, required=True)
    parser.add_argument("--control-metrics", type=Path, required=True)
    parser.add_argument("--freeze10-metrics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="只校验输入文件和输出路径，不读取指标或写结论。")
    return parser.parse_args(argv)


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Cannot parse {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def parse_scalar(value: str) -> Any:
    value = value.strip().strip("'\"")
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_flat_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty args.yaml: {path}")
    values: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line[0].isspace() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = parse_scalar(value)
    return values


def require_equal(actual: Any, expected: Any, field: str) -> None:
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{field} differs: expected={expected!r}, actual={actual!r}")
    elif actual != expected:
        raise ValueError(f"{field} differs: expected={expected!r}, actual={actual!r}")


def require_training_contract(
    config: dict[str, Any], args_yaml: dict[str, Any], stage: str, expected_freeze: int, expected_epochs: int, role: str
) -> dict[str, Any]:
    training = config.get("training")
    if not isinstance(training, dict) or not isinstance(training.get(stage), dict):
        raise ValueError(f"{role} requested_config lacks training.{stage}")
    requested = training[stage]
    for key in COMMON_TRAINING_KEYS:
        if key not in requested:
            raise ValueError(f"{role} requested_config.training.{stage} lacks {key!r}")
    require_equal(requested.get("freeze"), expected_freeze, f"{role}.{stage}.requested.freeze")
    require_equal(requested.get("epochs"), expected_epochs, f"{role}.{stage}.requested.epochs")
    require_equal(args_yaml.get("freeze"), expected_freeze, f"{role}.{stage}.args.freeze")
    require_equal(args_yaml.get("epochs"), expected_epochs, f"{role}.{stage}.args.epochs")
    for key in ("imgsz", "batch", "optimizer", "seed", "deterministic", "patience", "resume", "close_mosaic", "plots"):
        require_equal(args_yaml.get(key), requested[key], f"{role}.{stage}.args.{key}")
    return {key: requested[key] for key in COMMON_TRAINING_KEYS}


def read_metrics(path: Path, role: str) -> tuple[float, dict[str, float]]:
    payload = read_json(path, f"{role} re-evaluation metrics")
    for key, expected in {
        "scope": "development_fold_validation_only",
        "fold": 1,
        "development_only": True,
        "val_images": 376,
        "imgsz": 640,
        "seed": 0,
    }.items():
        require_equal(payload.get(key), expected, f"{role}.metrics.{key}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("map50_95"), (int, float)):
        raise ValueError(f"{role} metrics lack numeric map50_95")
    per_class_raw = payload.get("per_class")
    if not isinstance(per_class_raw, list):
        raise ValueError(f"{role} metrics lack per_class list")
    recalls: dict[str, float] = {}
    for record in per_class_raw:
        if not isinstance(record, dict):
            continue
        name = record.get("class_name")
        if name not in FOCUS_CLASSES:
            continue
        if record.get("status") != "evaluated" or not isinstance(record.get("recall"), (int, float)):
            raise ValueError(f"{role} focus class is not evaluable: {name}")
        recalls[name] = float(record["recall"])
    missing = [name for name in FOCUS_CLASSES if name not in recalls]
    if missing:
        raise ValueError(f"{role} re-evaluation lacks focus recalls: {missing}")
    return float(metrics["map50_95"]), recalls


def inspect_run(run_dir: Path, role: str, expected_stage1_freeze: int, metrics_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    config = read_json(run_dir / "requested_config.json", f"{role} requested_config")
    environment = read_json(run_dir / "environment.json", f"{role} environment")
    stage2_args = read_flat_yaml(run_dir / "args.yaml")
    stage1_args = read_flat_yaml(run_dir / "freeze_stage1" / "args.yaml")
    for required in (run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt", run_dir / "results.csv"):
        if not required.is_file() or required.stat().st_size <= 0:
            raise FileNotFoundError(f"{role} run is missing required evidence: {required}")
    expected_variant = "control" if role == "control" else "freeze10"
    for key, expected in {
        "experiment": "L1 YOLO11m paired two-stage freeze ablation",
        "variant": expected_variant,
        "model": "yolo11m.pt",
        "data_version": "oofconsensusv1",
        "development_only": True,
        "frozen_test_set": None,
        "release_validation_access": "not read by this experiment",
        "mode": "fold",
        "fold": 1,
    }.items():
        require_equal(config.get(key), expected, f"{role}.config.{key}")
    require_equal(config.get("split_counts"), EXPECTED_SPLIT_COUNTS, f"{role}.config.split_counts")
    selection = config.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"{role} requested_config lacks selection")
    require_equal(selection.get("total_epochs"), 83, f"{role}.selection.total_epochs")
    stage1_contract = require_training_contract(
        config, stage1_args, "stage1", expected_stage1_freeze, 20, role
    )
    stage2_contract = require_training_contract(config, stage2_args, "stage2", 0, 63, role)
    for key, expected in EXPECTED_RUNTIME.items():
        actual = str(environment.get(key, ""))
        if key == "torchvision":
            actual = actual.split("+", 1)[0]
        require_equal(actual, expected, f"{role}.environment.{key}")
    map50_95, recalls = read_metrics(metrics_path.resolve(), role)
    return {
        "role": role,
        "run_dir": str(run_dir),
        "run_name": config.get("run_name"),
        "variant": expected_variant,
        "stage1_freeze": expected_stage1_freeze,
        "stage1_training": stage1_contract,
        "stage2_training": stage2_contract,
        "runtime": {key: environment.get(key) for key in EXPECTED_RUNTIME},
        "reeval_metrics": str(metrics_path.resolve()),
        "map50_95": map50_95,
        "focus_recall": recalls,
    }


def write_outputs(output_root: Path, decision: dict[str, Any]) -> None:
    output_root.mkdir(parents=True)
    (output_root / "l1_freeze_pair_gate_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checks = decision["checks"]
    lines = [
        "# L1 YOLO11m Freeze10 成对验收",
        "",
        "- 范围：仅 OOF 开发集 Fold 1；未读取冻结测试集或 `release-validation`。",
        f"- 对照 mAP50-95：`{decision['control']['map50_95']:.6f}`。",
        f"- Freeze10 mAP50-95：`{decision['freeze10']['map50_95']:.6f}`。",
        f"- Freeze10 - 对照：`{decision['freeze10_minus_control_map50_95']:+.6f}`。",
        f"- 闸门状态：`{decision['status']}`。",
        "",
        "| 检查 | 结果 |",
        "| --- | --- |",
        f"| mAP50-95 增益至少 +0.010000 | {'通过' if checks['map_gain_pass'] else '未通过'} |",
        f"| 六个重点类别 Recall 均不下降 | {'通过' if checks['focus_recall_pass'] else '未通过'} |",
        "",
        "| 类别 | Freeze10 - 对照 Recall |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {name} | {delta:+.6f} |" for name, delta in decision["focus_recall_deltas"].items()
    )
    lines.extend(
        [
            "",
            "结论："
            + (
                "Freeze10 仅获得进入全开发集固定 epoch 训练的资格；全开发集训练和 ONNX 导出仍需按项目 GPU 授权与归档流程执行。"
                if decision["status"] == "pass_finaldev_eligible"
                else "不训练 Freeze10 全开发集版本，不导出新的部署 ONNX；保留现有已发布 ONNX。"
            ),
            "",
        ]
    )
    (output_root / "l1_freeze_pair_gate_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = {
        "control run": args.control_run_dir,
        "freeze10 run": args.freeze10_run_dir,
        "control metrics": args.control_metrics,
        "freeze10 metrics": args.freeze10_metrics,
    }
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite verification output: {args.output_root}")
    for label, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.dry_run:
        print(json.dumps({"inputs": {key: str(value.resolve()) for key, value in inputs.items()}}, ensure_ascii=False, indent=2))
        return 0

    control = inspect_run(args.control_run_dir, "control", 0, args.control_metrics)
    freeze10 = inspect_run(args.freeze10_run_dir, "freeze10", 10, args.freeze10_metrics)
    control_stage1_common = {key: value for key, value in control["stage1_training"].items() if key != "freeze"}
    freeze10_stage1_common = {key: value for key, value in freeze10["stage1_training"].items() if key != "freeze"}
    if control_stage1_common != freeze10_stage1_common:
        raise ValueError("Control and Freeze10 stage-1 configurations differ outside the expected freeze setting")
    if control["stage2_training"] != freeze10["stage2_training"]:
        raise ValueError("Control and Freeze10 stage-2 configurations differ")
    if control["runtime"] != freeze10["runtime"]:
        raise ValueError("Control and Freeze10 runtime versions differ")

    map_delta = freeze10["map50_95"] - control["map50_95"]
    recall_deltas = {
        name: freeze10["focus_recall"][name] - control["focus_recall"][name] for name in FOCUS_CLASSES
    }
    checks = {
        "map_gain_pass": map_delta >= 0.01,
        "focus_recall_pass": all(delta >= 0.0 for delta in recall_deltas.values()),
    }
    passed = all(checks.values())
    decision = {
        "experiment": "L1 YOLO11m paired two-stage Freeze10 ablation",
        "scope": "development-only Fold 1 auto_only evidence; frozen test and release-validation were not read",
        "control": control,
        "freeze10": freeze10,
        "freeze10_minus_control_map50_95": map_delta,
        "focus_recall_deltas": recall_deltas,
        "gate": {
            "minimum_map50_95_gain": 0.01,
            "focus_classes_recall_must_not_decrease": list(FOCUS_CLASSES),
        },
        "checks": checks,
        "status": "pass_finaldev_eligible" if passed else "not_passed_do_not_train_finaldev",
    }
    write_outputs(args.output_root, decision)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "status": decision["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
