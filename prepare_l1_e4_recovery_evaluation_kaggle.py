#!/usr/bin/env python3
"""Prepare an evaluation-only Kaggle GPU kernel for the recovered L1 E4 checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

from prepare_l1_tile_kaggle_kernel import PIPELINE_PATH, SCRIPT_TEMPLATE, build_metadata


ROOT = Path(__file__).resolve().parent
OWNER = "lly3323w"
CANDIDATE_DATASET_REF = "lly3323w/dianjienie-ryw18-oof-e4-cp-f01-v4-best"
CANDIDATE_WEIGHT_NAME = "e4_copypaste_fold01_v4_best.pt"
RUN_NAME = "dianjienie_medium_yolo11m_cp_640_kaggle_seed0_oofconsensusv1_fold01_evaluation"
DEFAULT_SLUG = "l1-ryw18-y11m-e4-cp-eval-f01-v4"
FOCUS_CLASSES = "wuzi,waterstain,airhole,bubble,greenstain,watermark"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the development-only original-image E4 recovery evaluation kernel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--candidate-dataset-ref", default=CANDIDATE_DATASET_REF)
    parser.add_argument("--candidate-weight-name", default=CANDIDATE_WEIGHT_NAME)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--title", default="L1 RYW18 Y11m E4 CP Eval Fold01 V4")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not PIPELINE_PATH.is_file():
        raise FileNotFoundError(f"Missing fixed original-image evaluator: {PIPELINE_PATH}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,62}", args.slug):
        raise ValueError("--slug must be 3-63 lowercase letters, digits, or hyphens")
    if "/" not in args.candidate_dataset_ref or not args.candidate_weight_name:
        raise ValueError("Candidate Dataset reference and weight name are required")
    output_root = (args.output_root or ROOT / f"kaggle_{args.slug.replace('-', '_')}").resolve()
    code_file = f"{args.slug}.py"
    metadata = build_metadata(
        args.owner,
        args.slug,
        args.title,
        code_file,
        True,
        args.candidate_dataset_ref,
    )
    evaluation_request = {
        "schema_version": 1,
        "experiment": "L1 E4 Copy-Paste Fold 1 recovered-checkpoint evaluation",
        "mode": "evaluate_only",
        "development_only": True,
        "auto_only": True,
        "frozen_test_set": None,
        "fold": 1,
        "run_name": RUN_NAME,
        "candidate_dataset": args.candidate_dataset_ref,
        "candidate_weight": args.candidate_weight_name,
        "baseline_dataset": "lly3323w/dianjienie-ryw18-oof-yolo11m-fold01-640-weight",
        "baseline_weight": "baseline_fold01_640_best.pt",
        "evaluator": {
            "pipeline": "l1_tile_pipeline.py",
            "tile_size": 4096,
            "overlap": 0.0,
            "negative_ratio": 0.0,
            "min_visible_fraction": 1.0,
            "imgsz": 640,
            "predict_batch": 8,
            "profiles": ["legacy"],
            "focus_classes": FOCUS_CLASSES.split(","),
        },
        "prohibition": "No training, no checkpoint resume, no source-label modification, and no frozen historical release-validation mount or read.",
    }
    script = textwrap.dedent(
        SCRIPT_TEMPLATE.substitute(
            mode="evaluate",
            run_name=RUN_NAME,
            pipeline_source=json.dumps(PIPELINE_PATH.read_text(encoding="utf-8")),
            baseline_weight_name="baseline_fold01_640_best.pt",
            candidate_weight_name=args.candidate_weight_name,
            tile_size=4096,
            overlap=0.0,
            negative_ratio=0.0,
            min_visible_fraction=1.0,
            focus_classes=FOCUS_CLASSES,
            epochs=300,
            imgsz=640,
            batch=8,
            predict_batch=8,
            tiled_eval_profiles="legacy",
            primary_tiled_profile="legacy",
            wbf_iou=0.25,
            edge_margin=16,
            save_tile_details="False",
            per_class_conf_config_text="None",
        )
    )
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "metadata": metadata, "evaluation_request": evaluation_request}, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing Kernel directory: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "evaluation_request.json").write_text(
        json.dumps(evaluation_request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / code_file).write_text(script, encoding="utf-8", newline="\n")
    print(f"[created] {output_root}")
    print(f"[push] kaggle kernels push -p {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
