"""Verify and extract a completed L1 Phase 1 tiled-evaluation archive.

This verifier is intentionally separate from ``collect_l1_kaggle_run.py``:
Phase 1 is an evaluation-only archive and must not be treated as a training
run.  It reads only the downloaded ZIP and writes its report under the chosen
derived output directory; it never reads images, labels, or release-validation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_RUN_NAME = "dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_tile1024o25_focus6_mix_fold01"
DEFAULT_BASELINE_SHA256 = "b00aad9cc8041424f0b1e7263a6c9a2be5675827fd7017fcdec395b40ccac0ce"
DEFAULT_CANDIDATE_SHA256 = "c4a903e8e8438798d44fc0548d666923a8728509f012d124101a8b980d7d875e"
DEFAULT_BASELINE_FULL_MAP50_95 = 0.16800147577890728
FOCUS_CLASSES = ("wuzi", "waterstain", "airhole", "bubble", "greenstain", "watermark")
PROFILES = ("legacy", "perclass_nms", "wbf", "wbf_edge")
CHUNK_SIZE = 1024 * 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and safely extract one completed L1 Phase 1 evaluation ZIP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--archive", type=Path, required=True, help="Downloaded Phase 1 evaluation ZIP.")
    parser.add_argument("--output-root", type=Path, required=True, help="New derived extraction/report directory.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--expected-baseline-sha256", default=DEFAULT_BASELINE_SHA256)
    parser.add_argument("--expected-candidate-sha256", default=DEFAULT_CANDIDATE_SHA256)
    parser.add_argument("--baseline-full-map50-95", type=float, default=DEFAULT_BASELINE_FULL_MAP50_95)
    parser.add_argument("--baseline-tolerance", type=float, default=1e-6)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_zip_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if not path.parts or path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise ValueError(f"Unsafe ZIP path: {name}")
    return path


def read_archive_json(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(archive.read(name).decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in archive: {name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {name}")
    return payload


def required_files(root: str) -> tuple[str, ...]:
    base = (
        "README.md",
        "requested_config.json",
        "plan_summary.json",
        "environment.json",
        "tile_evaluation/baseline_full.json",
        "tile_evaluation/baseline_tiled.json",
        "tile_evaluation/tile_model_full.json",
        "tile_evaluation/tile_model_tiled.json",
        "tile_evaluation/comparison_summary.json",
        "tile_evaluation/tiled_attribution_baseline.json",
        "tile_evaluation/tiled_attribution_tile_model.json",
        "tile_evaluation/baseline_raw_tile_predictions.jsonl.gz",
        "tile_evaluation/tile_model_raw_tile_predictions.jsonl.gz",
    )
    profile_files = tuple(
        f"tile_evaluation/{model}_tiled_{profile}.json"
        for model in ("baseline", "tile_model")
        for profile in PROFILES
    )
    return tuple(f"{root}/{relative}" for relative in (*base, *profile_files))


def validate_config(
    requested: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if requested.get("mode") != "evaluate" or requested.get("run_name") != args.run_name:
        raise ValueError("Evaluation requested_config mode or run_name mismatch")
    if requested.get("data_version") != "oofconsensusv1":
        raise ValueError("Unexpected data_version")
    if requested.get("development_only") is not True or requested.get("frozen_test_set") is not None:
        raise ValueError("Phase 1 archive does not satisfy the development-only frozen-test boundary")
    if requested.get("fold") != 1:
        raise ValueError("Phase 1 must evaluate fixed Fold 1")
    if requested.get("baseline_weights_sha256") != args.expected_baseline_sha256:
        raise ValueError("Fixed baseline weight hash mismatch")
    if requested.get("candidate_weights_sha256") != args.expected_candidate_sha256:
        raise ValueError("Recovered tile checkpoint hash mismatch")
    inference = requested.get("inference")
    if not isinstance(inference, dict):
        raise ValueError("Missing inference configuration")
    postprocessing = inference.get("tiled_postprocessing")
    if not isinstance(postprocessing, dict):
        raise ValueError("Missing tiled post-processing configuration")
    if tuple(postprocessing.get("requested_profiles", ())) != PROFILES:
        raise ValueError("Unexpected tiled post-processing profile set")
    if postprocessing.get("primary_profile") != "wbf_edge":
        raise ValueError("Phase 1 primary tiled profile must be wbf_edge")
    confidence = postprocessing.get("per_class_confidence")
    if not isinstance(confidence, dict):
        raise ValueError("Missing per-class confidence profile")
    metadata = confidence.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("held_out_evaluation_fold") != 1:
        raise ValueError("Per-class confidence profile was not held out from Fold 1")
    if metadata.get("calibration_folds") != [2, 3]:
        raise ValueError("Per-class confidence profile must be calibrated from Fold 2/3")
    return postprocessing


def load_profile_reports(archive: zipfile.ZipFile, root: str) -> dict[str, dict[str, dict[str, Any]]]:
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    for model in ("baseline", "tile_model"):
        reports[model] = {}
        for profile in PROFILES:
            report = read_archive_json(archive, f"{root}/tile_evaluation/{model}_tiled_{profile}.json")
            if report.get("tiled_profile") != profile or report.get("inference_mode") != "tiled":
                raise ValueError(f"Tiled profile report mismatch: {model}/{profile}")
            reports[model][profile] = report
    return reports


def detail_output_count(detail: dict[str, Any]) -> int:
    if "output_raw_ids" in detail:
        values = detail["output_raw_ids"]
    else:
        values = detail.get("fused_outputs")
    if not isinstance(values, list):
        raise ValueError("Post-processing detail has no valid output list")
    return len(values)


def scan_detail_file(path: Path, expected_model: str) -> dict[str, Any]:
    images = 0
    raw_count = 0
    low_confidence_count = 0
    internal_edge_count = 0
    profile_output_counts = {profile: 0 for profile in PROFILES}
    seen_image_ids: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            image_id = payload.get("image_id")
            raw_predictions = payload.get("raw_predictions")
            postprocess = payload.get("postprocess")
            if not isinstance(image_id, str) or image_id in seen_image_ids:
                raise ValueError(f"Invalid or duplicate image id in {path.name}:{line_number}")
            if not isinstance(raw_predictions, list) or not isinstance(postprocess, dict):
                raise ValueError(f"Invalid raw/detail payload in {path.name}:{line_number}")
            seen_image_ids.add(image_id)
            images += 1
            raw_count += len(raw_predictions)
            for expected_id, raw in enumerate(raw_predictions):
                if not isinstance(raw, dict) or raw.get("raw_id") != expected_id:
                    raise ValueError(f"Non-contiguous raw id in {path.name}:{line_number}")
                confidence = raw.get("confidence")
                edges = raw.get("touches_internal_edges")
                if not isinstance(confidence, (int, float)) or not isinstance(edges, list):
                    raise ValueError(f"Invalid raw prediction fields in {path.name}:{line_number}")
                low_confidence_count += float(confidence) < 0.25
                internal_edge_count += bool(edges)
            for profile in PROFILES:
                detail = postprocess.get(profile)
                if not isinstance(detail, dict):
                    raise ValueError(f"Missing {profile} detail in {path.name}:{line_number}")
                profile_output_counts[profile] += detail_output_count(detail)
    if images != 376:
        raise ValueError(f"Expected 376 Fold 1 images in {expected_model} detail file, got {images}")
    return {
        "model": expected_model,
        "images": images,
        "raw_premerge_count": raw_count,
        "raw_confidence_below_025_count": low_confidence_count,
        "raw_touches_internal_edge_count": internal_edge_count,
        "profile_output_counts": profile_output_counts,
    }


def verify_detail_aggregate(detail: dict[str, Any], attribution: dict[str, Any]) -> None:
    for key in ("images", "raw_premerge_count", "raw_confidence_below_025_count", "raw_touches_internal_edge_count"):
        if attribution.get(key) != detail[key]:
            raise ValueError(f"Detail aggregate mismatch for {detail['model']}: {key}")
    profile_counts = attribution.get("profile_counts")
    if not isinstance(profile_counts, dict):
        raise ValueError(f"Missing profile_counts for {detail['model']}")
    for profile, output_count in detail["profile_output_counts"].items():
        current = profile_counts.get(profile)
        if not isinstance(current, dict) or current.get("output_count") != output_count:
            raise ValueError(f"Detail output count mismatch for {detail['model']}/{profile}")


def metric_value(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric metric {key}")
    return float(value)


def phase1_gate(
    baseline_full: dict[str, Any], candidate_wbf_edge: dict[str, Any]
) -> dict[str, Any]:
    baseline_map = metric_value(baseline_full, "mAP50_95")
    candidate_map = metric_value(candidate_wbf_edge, "mAP50_95")
    focus_delta: dict[str, dict[str, float | None]] = {}
    focus_non_degraded = True
    for class_name in FOCUS_CLASSES:
        baseline = baseline_full.get("per_class", {}).get(class_name)
        candidate = candidate_wbf_edge.get("per_class", {}).get(class_name)
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            raise ValueError(f"Missing focus-class report: {class_name}")
        ap_before = baseline.get("ap50_95")
        ap_after = candidate.get("ap50_95")
        recall_before = baseline.get("recall_conf025_iou50")
        recall_after = candidate.get("recall_conf025_iou50")
        ap_delta = None if ap_before is None or ap_after is None else float(ap_after) - float(ap_before)
        recall_delta = None if recall_before is None or recall_after is None else float(recall_after) - float(recall_before)
        focus_delta[class_name] = {"ap50_95_delta": ap_delta, "recall_conf025_iou50_delta": recall_delta}
        if ap_delta is not None and ap_delta < 0:
            focus_non_degraded = False
        if recall_delta is not None and recall_delta < 0:
            focus_non_degraded = False
    delta = candidate_map - baseline_map
    if delta >= 0 and focus_non_degraded:
        status = "pass_phase2_eligible"
    elif delta < -0.01:
        status = "fail_stop_tile_direction"
    else:
        status = "not_passed_do_not_start_phase2"
    return {
        "baseline_full_mAP50_95": baseline_map,
        "tile_model_wbf_edge_mAP50_95": candidate_map,
        "delta_mAP50_95": delta,
        "focus_class_non_degraded_ap_and_recall": focus_non_degraded,
        "focus_class_deltas": focus_delta,
        "status": status,
        "rule": "pass only if delta >= 0 and all focus-class AP50-95/Recall@0.25 do not decline; stop if delta < -0.01",
    }


def safe_extract(archive_path: Path, output_root: Path, root: str) -> Path:
    # Keep the staging path short because the archive root and detail filenames
    # are already long enough to exceed Windows' legacy path limit otherwise.
    with tempfile.TemporaryDirectory(prefix="l1p1_", dir=tempfile.gettempdir()) as temp_dir:
        temp_root = Path(temp_dir)
        with zipfile.ZipFile(archive_path) as archive:
            for item in archive.infolist():
                path = normalized_zip_path(item.filename)
                target = (temp_root / Path(*path.parts)).resolve()
                if not target.is_relative_to(temp_root.resolve()):
                    raise ValueError(f"ZIP extraction escaped staging area: {item.filename}")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=CHUNK_SIZE)
        extracted = temp_root / root
        if not extracted.is_dir():
            raise FileNotFoundError("Expected evaluation root missing after extraction")
        if output_root.exists():
            raise FileExistsError(f"Refusing to overwrite output root: {output_root}")
        shutil.copytree(extracted, output_root)
    return output_root


def render_report(
    archive_path: Path,
    archive_sha256: str,
    config: dict[str, Any],
    profiles: dict[str, dict[str, dict[str, Any]]],
    details: dict[str, dict[str, Any]],
    gate: dict[str, Any],
) -> str:
    lines = [
        "# L1 Phase 1 tiled post-processing evaluation verification",
        "",
        f"- Archive: `{archive_path}`",
        f"- Archive SHA-256: `{archive_sha256}`",
        f"- Scope: development-only Fold {config['fold']}; frozen test: `{config['frozen_test_set']}`",
        f"- Fixed baseline SHA-256: `{config['baseline_weights_sha256']}`",
        f"- Recovered tile checkpoint SHA-256: `{config['candidate_weights_sha256']}`",
        "",
        "## Profile metrics",
        "",
        "| Profile | Baseline tiled mAP50-95 | Tile model tiled mAP50-95 | Tile model minus baseline full |",
        "| --- | ---: | ---: | ---: |",
    ]
    for profile in PROFILES:
        baseline = metric_value(profiles["baseline"][profile], "mAP50_95")
        candidate = metric_value(profiles["tile_model"][profile], "mAP50_95")
        delta = candidate - gate["baseline_full_mAP50_95"]
        lines.append(f"| {profile} | {baseline:.6f} | {candidate:.6f} | {delta:+.6f} |")
    lines.extend(
        [
            "",
            "## Attribution",
            "",
            "| Model | Raw crop predictions | Below 0.25 | Touch internal crop edge | wbf_edge outputs |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in ("baseline", "tile_model"):
        detail = details[model]
        lines.append(
            f"| {model} | {detail['raw_premerge_count']} | {detail['raw_confidence_below_025_count']} | "
            f"{detail['raw_touches_internal_edge_count']} | {detail['profile_output_counts']['wbf_edge']} |"
        )
    lines.extend(
        [
            "",
            "## Phase 1 gate",
            "",
            f"- Status: `{gate['status']}`",
            f"- `tile_model_wbf_edge - baseline_full`: `{gate['delta_mAP50_95']:+.6f}` mAP50-95",
            f"- Focus-class AP/Recall non-degradation: `{gate['focus_class_non_degraded_ap_and_recall']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archive_path = args.archive.resolve()
    output_root = args.output_root.resolve()
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise FileNotFoundError(f"Non-empty archive required: {archive_path}")
    if args.baseline_tolerance < 0:
        raise ValueError("--baseline-tolerance must be non-negative")
    root = f"{args.run_name}_evaluation"
    archive_sha256 = sha256_file(archive_path)
    if args.expected_archive_sha256 and archive_sha256 != args.expected_archive_sha256:
        raise ValueError("Archive SHA-256 does not match the completed Kaggle job record")
    print(json.dumps({"archive": str(archive_path), "sha256": archive_sha256, "output_root": str(output_root)}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    with zipfile.ZipFile(archive_path) as archive:
        corrupt_entry = archive.testzip()
        if corrupt_entry is not None:
            raise zipfile.BadZipFile(f"ZIP CRC failure: {corrupt_entry}")
        names: list[str] = []
        for item in archive.infolist():
            path = normalized_zip_path(item.filename)
            unix_mode = item.external_attr >> 16
            if unix_mode and (unix_mode & 0o170000) == 0o120000:
                raise ValueError(f"ZIP symlink entry is forbidden: {item.filename}")
            if not item.is_dir():
                names.append(path.as_posix())
        if not names or any(not name.startswith(f"{root}/") for name in names):
            raise ValueError("Unexpected ZIP root")
        missing = [name for name in required_files(root) if name not in names]
        if missing:
            raise FileNotFoundError(f"Phase 1 archive is missing required files: {missing}")
        requested = read_archive_json(archive, f"{root}/requested_config.json")
        postprocessing = validate_config(requested, args)
        baseline_full = read_archive_json(archive, f"{root}/tile_evaluation/baseline_full.json")
        if abs(metric_value(baseline_full, "mAP50_95") - args.baseline_full_map50_95) > args.baseline_tolerance:
            raise ValueError("Fixed baseline full-image metric does not reproduce the archived anchor")
        profiles = load_profile_reports(archive, root)
        primary_baseline = read_archive_json(archive, f"{root}/tile_evaluation/baseline_tiled.json")
        primary_candidate = read_archive_json(archive, f"{root}/tile_evaluation/tile_model_tiled.json")
        if primary_baseline.get("tiled_profile") != "wbf_edge" or primary_candidate.get("tiled_profile") != "wbf_edge":
            raise ValueError("Primary tiled reports are not wbf_edge")
        comparison = read_archive_json(archive, f"{root}/tile_evaluation/comparison_summary.json")
        if comparison.get("primary_tiled_profile") != "wbf_edge":
            raise ValueError("Comparison summary primary profile mismatch")
        candidate_wbf_edge = profiles["tile_model"]["wbf_edge"]
        gate = phase1_gate(baseline_full, candidate_wbf_edge)
        attributions = {
            model: read_archive_json(archive, f"{root}/tile_evaluation/tiled_attribution_{model}.json")
            for model in ("baseline", "tile_model")
        }

    extracted_root = safe_extract(archive_path, output_root, root)
    details = {
        "baseline": scan_detail_file(extracted_root / "tile_evaluation" / "baseline_raw_tile_predictions.jsonl.gz", "baseline"),
        "tile_model": scan_detail_file(extracted_root / "tile_evaluation" / "tile_model_raw_tile_predictions.jsonl.gz", "tile_model"),
    }
    for model in ("baseline", "tile_model"):
        verify_detail_aggregate(details[model], attributions[model])
    manifest = {
        "phase": "L1_small_object_phase1",
        "archive": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_file_count": len(names),
        "extracted_root": str(extracted_root),
        "requested_config": requested,
        "postprocessing": postprocessing,
        "detail_audit": details,
        "phase1_gate": gate,
        "development_only": True,
        "frozen_test_set": None,
    }
    write_json(extracted_root / "phase1_collection_manifest.json", manifest)
    (extracted_root / "phase1_verification_report.md").write_text(
        render_report(archive_path, archive_sha256, requested, profiles, details, gate), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "extracted_root": str(extracted_root),
                "gate": gate,
                "verification_report": str(extracted_root / "phase1_verification_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
