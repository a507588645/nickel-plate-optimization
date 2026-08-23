#!/usr/bin/env python3
"""Safely verify the completed paired L1 E2 Mosaic experiment archives.

The E2 candidate is only interpretable when its Mosaic=1 control completed
under the same frozen development-only contract.  This tool validates both
archives together, extracts them into a new derived directory, and writes the
single automatic gate decision.  It never reads a dataset or uploads files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "l1_e2_mosaic_pair_verification_20260822"
EXPECTED_BASE_CHECKPOINT_SHA256 = "b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce"
EXPECTED_BASELINE_MAP50_95 = 0.16800147577890728
EXPECTED_PIPELINE_SHA256 = "7128b72c1e4fbd30d2cc25ce1daf051692f9da68a2933115540b2a5df863a1b8"
BASELINE_TOLERANCE = 1e-6
CHUNK_SIZE = 1024 * 1024
ROLES = ("control", "candidate")
EXPECTED_MOSAIC = {"control": 1.0, "candidate": 0.0}
EXPECTED_RUN_NAMES = {
    "control": "dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_mosaic1_ft30_fold01",
    "candidate": "dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_mosaic0_ft30_fold01",
}
FOCUS_CLASSES = ("wuzi", "waterstain", "airhole", "bubble", "greenstain", "watermark")
PAIR_VARIANT_ARGS_KEYS = {"mosaic", "name", "save_dir"}
REQUIRED_FILES = (
    "weights/best.pt",
    "weights/last.pt",
    "args.yaml",
    "requested_config.json",
    "e2_pair_contract.json",
    "environment.json",
    "results.csv",
    "README.md",
    "dataset_config/data.yaml",
    "dataset_config/train.txt",
    "dataset_config/val.txt",
    "evaluation/evaluation_summary.json",
    "evaluation/tile_evaluation/baseline_full.json",
    "evaluation/tile_evaluation/tile_model_full.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验 L1 E2 Mosaic=1 对照与 Mosaic=0 候选的两个完成 ZIP，并输出成对闸门结论。"
    )
    parser.add_argument("--control-archive", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-base-checkpoint-sha256", default=EXPECTED_BASE_CHECKPOINT_SHA256)
    parser.add_argument("--dry-run", action="store_true", help="仅检查输入路径和输出目录，不读取或解压 ZIP。")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(source: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
        digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_zip_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if not path.parts or path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise ValueError(f"ZIP contains an unsafe path: {name}")
    return path


def read_json(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    try:
        payload = json.loads(archive.read(member).decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot parse required JSON evidence: {member}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {member}")
    return payload


def as_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    return float(value)


def parse_args_yaml(raw: str) -> dict[str, Any]:
    """Read only the flat scalar keys that form the E2 training contract."""
    values: dict[str, Any] = {}
    for line in raw.splitlines():
        if not line or line[0].isspace() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("'\"")
        if value.lower() in {"true", "false"}:
            values[key] = value.lower() == "true"
        else:
            try:
                values[key] = int(value)
            except ValueError:
                try:
                    values[key] = float(value)
                except ValueError:
                    values[key] = value
    return values


def validate_archive(archive_path: Path, role: str, expected_base_hash: str) -> dict[str, Any]:
    if not archive_path.is_file() or archive_path.stat().st_size <= 0:
        raise FileNotFoundError(f"{role} archive does not exist or is empty: {archive_path}")
    expected_run_name = EXPECTED_RUN_NAMES[role]
    with zipfile.ZipFile(archive_path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise zipfile.BadZipFile(f"{role} archive CRC failure: {corrupt}")
        files: set[str] = set()
        for item in archive.infolist():
            path = normalized_zip_path(item.filename)
            mode = item.external_attr >> 16
            if mode and (mode & 0o170000) == 0o120000:
                raise ValueError(f"{role} archive contains a symlink: {item.filename}")
            if not item.is_dir():
                normalized = path.as_posix()
                if normalized in files:
                    raise ValueError(f"{role} archive contains a duplicate file: {normalized}")
                files.add(normalized)
        prefix = f"{expected_run_name}/"
        if not files or any(not file_name.startswith(prefix) for file_name in files):
            roots = sorted({PurePosixPath(file_name).parts[0] for file_name in files})
            raise ValueError(f"{role} archive root must be {expected_run_name}, actual={roots}")
        missing = [relative for relative in REQUIRED_FILES if f"{prefix}{relative}" not in files]
        if missing:
            raise FileNotFoundError(f"{role} archive lacks required evidence: {missing}")

        requested = read_json(archive, f"{prefix}requested_config.json")
        contract = read_json(archive, f"{prefix}e2_pair_contract.json")
        environment = read_json(archive, f"{prefix}environment.json")
        evaluation_summary = read_json(archive, f"{prefix}evaluation/evaluation_summary.json")
        baseline = read_json(archive, f"{prefix}evaluation/tile_evaluation/baseline_full.json")
        candidate = read_json(archive, f"{prefix}evaluation/tile_evaluation/tile_model_full.json")
        args_yaml = parse_args_yaml(archive.read(f"{prefix}args.yaml").decode("utf-8"))
        train_lines = [line for line in archive.read(f"{prefix}dataset_config/train.txt").decode("utf-8").splitlines() if line.strip()]
        val_lines = [line for line in archive.read(f"{prefix}dataset_config/val.txt").decode("utf-8").splitlines() if line.strip()]
        with archive.open(f"{prefix}weights/best.pt") as source:
            best_hash = sha256_stream(source)

    variant = contract.get("variant")
    common = contract.get("common_contract")
    if not isinstance(variant, dict) or not isinstance(common, dict):
        raise ValueError(f"{role} pair contract is incomplete")
    if variant.get("role") != role or requested.get("role") != role:
        raise ValueError(f"{role} role metadata disagrees")
    if requested.get("run_name") != expected_run_name or variant.get("run_name") != expected_run_name:
        raise ValueError(f"{role} run name metadata disagrees")
    mosaic = as_float(variant.get("mosaic"), f"{role}.contract.mosaic")
    if mosaic != EXPECTED_MOSAIC[role] or as_float(requested.get("mosaic"), f"{role}.requested.mosaic") != mosaic:
        raise ValueError(f"{role} Mosaic value is not the expected paired value")
    sibling_role = "candidate" if role == "control" else "control"
    sibling = contract.get("expected_sibling")
    if not isinstance(sibling, dict) or sibling.get("role") != sibling_role:
        raise ValueError(f"{role} expected sibling metadata is invalid")
    if as_float(sibling.get("mosaic"), f"{role}.sibling.mosaic") != EXPECTED_MOSAIC[sibling_role]:
        raise ValueError(f"{role} expected sibling Mosaic is invalid")
    if sibling.get("run_name") != EXPECTED_RUN_NAMES[sibling_role]:
        raise ValueError(f"{role} expected sibling run name is invalid")
    if requested.get("development_only") is not True or requested.get("frozen_test_set") is not None:
        raise ValueError(f"{role} violates the development-only boundary")
    if common.get("development_only") is not True or common.get("frozen_test_set") is not None:
        raise ValueError(f"{role} common contract violates the development-only boundary")
    if common.get("fold") != 1 or common.get("split_counts") != {"train": 751, "val": 376, "test_frozen": None}:
        raise ValueError(f"{role} fold/split contract is incorrect")
    base = common.get("base_checkpoint")
    if not isinstance(base, dict) or base.get("sha256") != expected_base_hash:
        raise ValueError(f"{role} base checkpoint hash does not match the frozen E2 base")
    evaluation_contract = common.get("evaluation")
    if not isinstance(evaluation_contract, dict):
        raise ValueError(f"{role} evaluation contract is missing")
    expected_evaluation = {
        "entry": "l1_tile_pipeline.py --mode evaluate",
        "pipeline_sha256": EXPECTED_PIPELINE_SHA256,
        "model_weight": "best.pt",
        "tile_size": 4096,
        "overlap": 0.0,
        "fixed_baseline_full_mAP50_95": EXPECTED_BASELINE_MAP50_95,
    }
    for key, expected in expected_evaluation.items():
        actual = evaluation_contract.get(key)
        if isinstance(expected, float):
            if abs(as_float(actual, f"{role}.evaluation.{key}") - expected) > 1e-12:
                raise ValueError(f"{role} evaluation contract field {key!r} differs")
        elif actual != expected:
            raise ValueError(f"{role} evaluation contract field {key!r} differs")
    training = common.get("training")
    if not isinstance(training, dict):
        raise ValueError(f"{role} common training contract is missing")
    expected_training = {"epochs": 30, "imgsz": 640, "batch": 8, "optimizer": "SGD", "seed": 0, "deterministic": True, "resume": False, "close_mosaic": 0, "patience": 100, "save_period": 1}
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"{role} common training field {key!r} differs: {training.get(key)!r}")
    expected_args = {
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
    }
    for key, expected in expected_args.items():
        if requested.get("training", {}).get(key) != expected:
            # The concise requested_config records only the core run settings;
            # the immutable pair contract and args.yaml carry the rest.
            if key in {"lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "hsv_h", "hsv_s", "hsv_v", "translate", "scale", "fliplr", "mixup", "cutmix", "copy_paste", "auto_augment", "erasing"}:
                continue
            raise ValueError(f"{role} requested training field {key!r} differs")
        if args_yaml.get(key) != expected:
            raise ValueError(f"{role} args.yaml field {key!r} differs: {args_yaml.get(key)!r}")
    if as_float(args_yaml.get("mosaic"), f"{role}.args.yaml.mosaic") != mosaic:
        raise ValueError(f"{role} args.yaml Mosaic differs from the pair contract")
    if args_yaml.get("name") != expected_run_name:
        raise ValueError(f"{role} args.yaml run name differs from the expected archive root")
    if requested.get("base_checkpoint_sha256") != expected_base_hash:
        raise ValueError(f"{role} requested base checkpoint hash differs")
    if len(train_lines) != 751 or len(val_lines) != 376 or set(train_lines) & set(val_lines):
        raise ValueError(f"{role} archived train/val manifests violate the 751/376 disjoint split contract")
    runtime_pins = common.get("runtime_pins")
    if not isinstance(runtime_pins, dict):
        raise ValueError(f"{role} runtime contract is missing")
    for name in ("torch", "torchvision", "numpy", "ultralytics"):
        actual = str(environment.get(name, ""))
        expected = str(runtime_pins.get(name, ""))
        if name == "torchvision":
            actual = actual.split("+", 1)[0]
        if actual != expected:
            raise ValueError(f"{role} runtime {name} differs: expected={expected}, actual={actual}")
    if evaluation_summary.get("role") != role or as_float(evaluation_summary.get("mosaic"), f"{role}.evaluation.mosaic") != mosaic:
        raise ValueError(f"{role} evaluation summary identity differs")
    if evaluation_summary.get("candidate_best_model_sha256") != best_hash or candidate.get("model_sha256") != best_hash:
        raise ValueError(f"{role} evaluation does not correspond to archived best.pt")
    baseline_map = as_float(baseline.get("mAP50_95"), f"{role}.baseline.mAP50_95")
    if abs(baseline_map - EXPECTED_BASELINE_MAP50_95) > BASELINE_TOLERANCE:
        raise ValueError(f"{role} full-image baseline did not reproduce: {baseline_map}")
    if baseline.get("model_sha256") != expected_base_hash:
        raise ValueError(f"{role} baseline full report used an unexpected weight")
    if evaluation_summary.get("baseline_model_sha256") != expected_base_hash:
        raise ValueError(f"{role} evaluation summary used an unexpected baseline weight")
    candidate_map = as_float(candidate.get("mAP50_95"), f"{role}.candidate.mAP50_95")
    if abs(as_float(evaluation_summary.get("candidate_best_mAP50_95"), f"{role}.summary.candidate_mAP") - candidate_map) > 1e-12:
        raise ValueError(f"{role} evaluation summary mAP disagrees with the candidate report")
    recalls: dict[str, float] = {}
    per_class = candidate.get("per_class")
    if not isinstance(per_class, dict):
        raise ValueError(f"{role} candidate report lacks per-class metrics")
    for class_name in FOCUS_CLASSES:
        record = per_class.get(class_name)
        if not isinstance(record, dict):
            raise ValueError(f"{role} candidate report lacks focus class: {class_name}")
        recalls[class_name] = as_float(record.get("recall_conf025_iou50"), f"{role}.{class_name}.recall")
    common_hash = hashlib.sha256(canonical_json(common).encode("utf-8")).hexdigest()
    if contract.get("common_contract_sha256") != common_hash:
        raise ValueError(f"{role} common contract hash is invalid")
    return {
        "role": role,
        "archive": str(archive_path.resolve()),
        "archive_sha256": sha256_file(archive_path),
        "run_name": expected_run_name,
        "pair_id": contract.get("pair_id"),
        "common_contract_sha256": common_hash,
        "mosaic": mosaic,
        "best_pt_sha256": best_hash,
        "runtime": {name: environment.get(name) for name in ("torch", "torchvision", "numpy", "ultralytics")},
        "split_counts_checked": {"train": len(train_lines), "val": len(val_lines)},
        "nonvariant_args": {key: value for key, value in args_yaml.items() if key not in PAIR_VARIANT_ARGS_KEYS},
        "baseline_full_mAP50_95": baseline_map,
        "candidate_best_mAP50_95": candidate_map,
        "focus_recall_conf025_iou50": recalls,
    }


def extract_archive(archive_path: Path, run_name: str, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite extracted archive: {destination}")
    parent = destination.parent.resolve()
    with tempfile.TemporaryDirectory(prefix="l1_e2_extract_", dir=parent) as temporary:
        temporary_root = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            for item in archive.infolist():
                safe = normalized_zip_path(item.filename)
                target = (temporary_root / Path(*safe.parts)).resolve()
                if not target.is_relative_to(temporary_root.resolve()):
                    raise ValueError(f"ZIP extraction escapes output root: {item.filename}")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=CHUNK_SIZE)
        source_run = temporary_root / run_name
        if not source_run.is_dir():
            raise FileNotFoundError(f"Expected run root missing after extraction: {run_name}")
        staging = parent / f".{destination.name}.extracting-{uuid.uuid4().hex}"
        try:
            shutil.copytree(source_run, staging)
            staging.replace(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def gate_decision(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if control["pair_id"] != candidate["pair_id"]:
        raise ValueError("Control and candidate pair IDs differ")
    if control["common_contract_sha256"] != candidate["common_contract_sha256"]:
        raise ValueError("Control and candidate common contracts differ")
    if control["nonvariant_args"] != candidate["nonvariant_args"]:
        control_keys = set(control["nonvariant_args"])
        candidate_keys = set(candidate["nonvariant_args"])
        differing = sorted(
            (control_keys | candidate_keys)
            - {key for key in control_keys & candidate_keys if control["nonvariant_args"][key] == candidate["nonvariant_args"][key]}
        )
        raise ValueError(f"Control and candidate args.yaml differ outside Mosaic/run output fields: {differing}")
    nonvariant_args_sha256 = hashlib.sha256(
        canonical_json(control["nonvariant_args"]).encode("utf-8")
    ).hexdigest()
    pair_delta = candidate["candidate_best_mAP50_95"] - control["candidate_best_mAP50_95"]
    utility_delta = candidate["candidate_best_mAP50_95"] - candidate["baseline_full_mAP50_95"]
    recall_deltas = {
        class_name: candidate["focus_recall_conf025_iou50"][class_name] - control["focus_recall_conf025_iou50"][class_name]
        for class_name in FOCUS_CLASSES
    }
    causal_pass = pair_delta >= 0.01
    recall_pass = all(delta >= 0.0 for delta in recall_deltas.values())
    utility_pass = utility_delta >= 0.01
    passed = causal_pass and recall_pass and utility_pass
    return {
        "experiment": "L1 E2 paired Mosaic ablation",
        "pair_id": control["pair_id"],
        "scope": "development-only Fold 1 auto_only evidence; no frozen holdout was used",
        "control": control,
        "candidate": candidate,
        "candidate_minus_control_mAP50_95": pair_delta,
        "candidate_minus_fixed_baseline_mAP50_95": utility_delta,
        "focus_recall_deltas_candidate_minus_control": recall_deltas,
        "shared_nonvariant_args_sha256": nonvariant_args_sha256,
        "gate": {
            "causal_mosaic_delta_mAP50_95_min": 0.01,
            "focus_recall_non_degradation": True,
            "utility_delta_vs_fixed_baseline_mAP50_95_min": 0.01,
        },
        "checks": {
            "causal_mosaic_delta_pass": causal_pass,
            "focus_recall_non_degradation_pass": recall_pass,
            "utility_delta_pass": utility_pass,
        },
        "status": "pass_expand_eligible" if passed else "not_passed_do_not_expand",
        "next_step": (
            "Do not expand to Fold 2/3 without a separate user authorization."
            if passed
            else "Do not expand E2 to Fold 2/3; retain the result as a single-Fold auto_only ablation."
        ),
    }


def write_report(output_root: Path, decision: dict[str, Any]) -> None:
    checks = decision["checks"]
    recalls = decision["focus_recall_deltas_candidate_minus_control"]
    lines = [
        "# L1 E2 成对 Mosaic 校验报告",
        "",
        f"- 配对 ID：`{decision['pair_id']}`",
        f"- 状态：`{decision['status']}`",
        "- 范围：仅开发集 Fold 1、`auto_only`；未使用冻结验证集。",
        f"- 对照 Mosaic=1 mAP50-95：`{decision['control']['candidate_best_mAP50_95']:.6f}`",
        f"- 候选 Mosaic=0 mAP50-95：`{decision['candidate']['candidate_best_mAP50_95']:.6f}`",
        f"- 候选 - 对照：`{decision['candidate_minus_control_mAP50_95']:+.6f}`（门槛 `+0.010000`，{'通过' if checks['causal_mosaic_delta_pass'] else '未通过'}）",
        f"- 候选 - 固定基线：`{decision['candidate_minus_fixed_baseline_mAP50_95']:+.6f}`（门槛 `+0.010000`，{'通过' if checks['utility_delta_pass'] else '未通过'}）",
        f"- 焦点类 Recall 不劣化：{'通过' if checks['focus_recall_non_degradation_pass'] else '未通过'}。",
        "",
        "| 类别 | Mosaic=0 − Mosaic=1 Recall@0.25/IoU0.5 |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {delta:+.6f} |" for name, delta in recalls.items())
    lines.extend(["", f"结论：{decision['next_step']}", ""])
    (output_root / "e2_pair_verification_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archives = {"control": args.control_archive.resolve(), "candidate": args.candidate_archive.resolve()}
    output_root = args.output_root.resolve()
    if args.dry_run:
        for role, archive in archives.items():
            if not archive.is_file() or archive.stat().st_size <= 0:
                raise FileNotFoundError(f"{role} archive does not exist or is empty: {archive}")
        if output_root.exists():
            raise FileExistsError(f"Refusing to overwrite verification output root: {output_root}")
        print(json.dumps({"archives": {role: str(path) for role, path in archives.items()}, "output_root": str(output_root)}, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite verification output root: {output_root}")
    verified = {role: validate_archive(path, role, args.expected_base_checkpoint_sha256) for role, path in archives.items()}
    decision = gate_decision(verified["control"], verified["candidate"])
    output_root.mkdir(parents=True)
    try:
        for role in ROLES:
            extract_archive(archives[role], EXPECTED_RUN_NAMES[role], output_root / role)
        write_json(output_root / "e2_pair_collection_manifest.json", {"archives": verified, "decision": decision})
        write_json(output_root / "e2_pair_gate_decision.json", decision)
        write_report(output_root, decision)
    except Exception:
        # The directory only contains this verification attempt and was just created above.
        shutil.rmtree(output_root)
        raise
    print(json.dumps({"output_root": str(output_root), "status": decision["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
