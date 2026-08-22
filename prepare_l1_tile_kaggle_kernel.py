#!/usr/bin/env python3
"""Create Kaggle preflight and GPU kernels for the L1 tile experiment."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from string import Template


ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = ROOT / "l1_tile_pipeline.py"
IMAGE_DATASET_REF = "lly3323w/dianjienie"
CONFIG_DATASET_REF = "lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config"
BASELINE_DATASET_REF = "lly3323w/dianjienie-ryw18-oof-yolo11m-fold01-640-weight"
BASELINE_WEIGHT_NAME = "baseline_fold01_640_best.pt"
DEFAULT_CANDIDATE_DATASET_REF = "lly3323w/dianjienie-ryw18-oof-y11m-tile-f01-v3-ckpt"
DEFAULT_CANDIDATE_WEIGHT_NAME = "tile_fold01_v3_best.pt"


SCRIPT_TEMPLATE = Template(
    r'''#!/usr/bin/env python3
"""Run one L1 development-only mixed full-image/tile Kaggle experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


INPUT_ROOT = Path("/kaggle/input")
MODE = "$mode"
RUN_NAME = "$run_name"
PIPELINE_SOURCE = $pipeline_source
CANDIDATE_WEIGHT_NAME = "$candidate_weight_name"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mounted_dataset_roots() -> list[Path]:
    roots = [path for path in INPUT_ROOT.iterdir() if path.is_dir()]
    scoped_root = INPUT_ROOT / "datasets"
    if scoped_root.is_dir():
        roots.extend(path for path in scoped_root.rglob("*") if path.is_dir())
    return roots


def input_inventory() -> dict[str, list[str]]:
    return {
        str(path.relative_to(INPUT_ROOT)): sorted(item.name for item in path.iterdir())[:12]
        for path in mounted_dataset_roots()
    }


def resolve_config_payload() -> tuple[Path, Path]:
    package_roots = [
        path for path in mounted_dataset_roots() if (path / "config_archive_manifest.json").is_file()
    ]
    if len(package_roots) != 1:
        raise RuntimeError(f"Cannot uniquely locate OOF-consensus config Dataset: {input_inventory()}")
    package_root = package_roots[0]
    manifest = json.loads((package_root / "config_archive_manifest.json").read_text(encoding="utf-8"))
    archive_name = manifest.get("config_archive")
    expected_files = manifest.get("payload_files")
    expected_config_hash = manifest.get("source_manifest_sha256")
    if not isinstance(archive_name, str) or not isinstance(expected_files, list) or not isinstance(expected_config_hash, str):
        raise ValueError("Invalid OOF-consensus config archive manifest")
    payload_root = package_root / Path(archive_name).stem
    actual_files = {path.name for path in payload_root.iterdir() if path.is_file()} if payload_root.is_dir() else set()
    if actual_files != set(expected_files):
        raise ValueError(
            f"Expanded OOF-consensus payload mismatch: expected={len(expected_files)}, actual={len(actual_files)}"
        )
    config_manifest = payload_root / "config_manifest.json"
    if sha256_file(config_manifest) != expected_config_hash:
        raise ValueError("Expanded OOF-consensus config_manifest hash mismatch")
    print(f"[job] config_payload={payload_root}", flush=True)
    return payload_root, config_manifest


def safe_relative_path(value: str) -> Path:
    candidate = PurePosixPath(value)
    if (
        "\\" in value
        or candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"Unsafe config relative path: {value!r}")
    return Path(*candidate.parts)


def restore_runtime_config(
    payload_root: Path,
    config_manifest: Path,
    runtime_root: Path = Path("/kaggle/working/l1_tile_config_runtime"),
) -> Path:
    """Restore the archive's flattened payload into the expected directory tree."""
    manifest = json.loads(config_manifest.read_text(encoding="utf-8"))
    labels = manifest.get("labels")
    metadata = manifest.get("metadata")
    if not isinstance(labels, list) or not isinstance(metadata, list):
        raise ValueError("Config manifest does not describe flattened labels and metadata")
    runtime_root = runtime_root.resolve()
    if runtime_root.exists():
        raise FileExistsError(f"Refusing to overwrite tile config runtime: {runtime_root}")
    runtime_root.mkdir(parents=True)
    shutil.copy2(config_manifest, runtime_root / "config_manifest.json")
    for record in labels + metadata:
        if not isinstance(record, dict):
            raise ValueError("Invalid flattened config record")
        upload_name = record.get("upload_name")
        relative_path = record.get("relative_path")
        if not isinstance(upload_name, str) or Path(upload_name).name != upload_name:
            raise ValueError("Invalid flattened config upload name")
        if not isinstance(relative_path, str):
            raise ValueError("Invalid flattened config relative path")
        source = payload_root / upload_name
        target = runtime_root / safe_relative_path(relative_path)
        if not source.is_file():
            raise FileNotFoundError(f"Missing flattened config input: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    required = (
        runtime_root / "labels",
        runtime_root / "splits" / "folds" / "fold_01" / "train.txt",
        runtime_root / "splits" / "folds" / "fold_01" / "val.txt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Restored tile config is incomplete: {missing}")
    print(f"[job] restored_config={runtime_root}", flush=True)
    return runtime_root


def resolve_image_dataset_root(config_manifest: Path) -> Path:
    manifest = json.loads(config_manifest.read_text(encoding="utf-8"))
    if manifest.get("data_version") != "oofconsensusv1" or manifest.get("development_only") is not True:
        raise ValueError("Config manifest is not the required development-only OOF-consensus version")
    images = manifest.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], dict):
        raise ValueError("Config manifest has no source image probe")
    source_relative = images[0].get("source_relative")
    if not isinstance(source_relative, str):
        raise ValueError("Config manifest source image probe is invalid")
    probe_parts = PurePosixPath(source_relative).parts
    candidates = [
        root for root in mounted_dataset_roots()
        if root.is_dir() and root.joinpath(*probe_parts).is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cannot uniquely locate source image Dataset for {source_relative}: "
            f"matches={[str(path.relative_to(INPUT_ROOT)) for path in candidates]}, inputs={input_inventory()}"
        )
    print(f"[job] image_dataset={candidates[0]}", flush=True)
    return candidates[0]


def resolve_baseline_weights() -> Path:
    candidates = [path / "$baseline_weight_name" for path in mounted_dataset_roots() if (path / "$baseline_weight_name").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate fixed Fold 1 baseline weight: {input_inventory()}")
    print(f"[job] baseline_weights={candidates[0]}", flush=True)
    return candidates[0]


def resolve_candidate_weights() -> Path:
    candidates = [path / CANDIDATE_WEIGHT_NAME for path in mounted_dataset_roots() if (path / CANDIDATE_WEIGHT_NAME).is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate recovered tile checkpoint: {input_inventory()}")
    print(f"[job] candidate_weights={candidates[0]}", flush=True)
    return candidates[0]


def install_dependencies() -> None:
    """Use the pinned CUDA stack from the completed L1 final Kaggle jobs."""
    print("[job] installing pinned torch and ultralytics runtime", flush=True)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "--force-reinstall",
            "torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cu124",
        ],
        check=True,
    )
    # Kaggle images can contain an incompatible torchaudio build; this workflow does not use it.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "torchaudio"], check=False)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "numpy==1.26.4", "ultralytics==8.4.66"],
        check=True,
    )
    import torch
    import ultralytics

    if not torch.cuda.is_available():
        raise RuntimeError("Pinned Ultralytics runtime cannot access the requested Kaggle GPU")
    print(
        f"[job] runtime torch={torch.__version__} cuda={torch.version.cuda} "
        f"ultralytics={ultralytics.__version__} gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )


def main() -> None:
    payload_root, config_manifest = resolve_config_payload()
    config_root = restore_runtime_config(payload_root, config_manifest)
    image_dataset_root = resolve_image_dataset_root(config_manifest)
    baseline_weights = resolve_baseline_weights() if MODE in {"train", "evaluate"} else None
    candidate_weights = resolve_candidate_weights() if MODE == "evaluate" else None
    if MODE in {"train", "evaluate"}:
        install_dependencies()
    pipeline = Path("/kaggle/working/l1_tile_pipeline.py")
    pipeline.write_text(PIPELINE_SOURCE, encoding="utf-8")
    output_root = Path("/kaggle/working") / f"{RUN_NAME}_workspace"
    command = [
        sys.executable,
        str(pipeline),
        "--mode", MODE,
        "--config-root", str(config_root),
        "--image-dataset-root", str(image_dataset_root),
        "--output-root", str(output_root),
        "--run-name", RUN_NAME,
        "--fold", "1",
        "--model", "yolo11m.pt",
        "--tile-size", "$tile_size",
        "--overlap", "$overlap",
        "--negative-ratio", "$negative_ratio",
        "--min-visible-fraction", "$min_visible_fraction",
        "--focus-classes", "$focus_classes",
        "--epochs", "$epochs",
        "--imgsz", "$imgsz",
        "--batch", "$batch",
        "--predict-batch", "$predict_batch",
        "--device", "0",
        "--workers", "8",
        "--seed", "0",
        "--patience", "100",
    ]
    if baseline_weights is not None:
        command.extend(["--baseline-weights", str(baseline_weights)])
    if candidate_weights is not None:
        command.extend(["--candidate-weights", str(candidate_weights)])
    print("[job]", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if MODE in {"train", "evaluate"}:
        archive_stem = RUN_NAME if MODE == "train" else f"{RUN_NAME}_evaluation"
        archive = output_root / f"{archive_stem}_results.zip"
        if not archive.is_file():
            raise FileNotFoundError(f"Tile pipeline did not create expected archive: {archive}")
        published_archive = Path("/kaggle/working") / archive.name
        if published_archive.exists():
            raise FileExistsError(f"Refusing to overwrite published Kaggle archive: {published_archive}")
        shutil.copy2(archive, published_archive)
        print(f"[job] published_archive={published_archive}", flush=True)


if __name__ == "__main__":
    main()
'''
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Kaggle-only L1 OOF-consensus tile experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("preflight", "train", "evaluate"), required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument("--min-visible-fraction", type=float, default=0.50)
    parser.add_argument("--focus-classes", default="wuzi,waterstain,airhole,bubble,greenstain,watermark")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--predict-batch", type=int, default=8)
    parser.add_argument("--owner", default="lly3323w")
    parser.add_argument("--candidate-dataset-ref", default=DEFAULT_CANDIDATE_DATASET_REF)
    parser.add_argument("--candidate-weight-name", default=DEFAULT_CANDIDATE_WEIGHT_NAME)
    parser.add_argument("--slug")
    parser.add_argument("--title")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def defaults(args: argparse.Namespace) -> tuple[str, str, str, Path]:
    if args.tile_size < 128 or not 0 <= args.overlap < 1 or args.negative_ratio < 0:
        raise ValueError("Invalid tile configuration")
    if not 0 < args.min_visible_fraction <= 1 or args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("Invalid training configuration")
    overlap_code = int(args.overlap * 100)
    if args.mode == "train":
        default_slug = f"l1-ryw18-y11m-oof-v1-tile{args.tile_size}-o{overlap_code:02d}-b{args.batch}-e{args.epochs}-f01"
    elif args.mode == "evaluate":
        default_slug = f"l1-ryw18-y11m-oof-v1-tile{args.tile_size}-o{overlap_code:02d}-eval-f01"
    else:
        default_slug = f"l1-ryw18-oof-v1-tile{args.tile_size}-o{overlap_code:02d}-preflight-fold01"
    slug = args.slug or default_slug
    if args.mode == "train":
        default_title = f"L1 RYW18 Y11m OOF v1 Tile{args.tile_size} O{overlap_code:02d} B{args.batch} E{args.epochs} F01"
    elif args.mode == "evaluate":
        default_title = f"L1 RYW18 Y11m OOF v1 Tile{args.tile_size} O{overlap_code:02d} Eval F01"
    else:
        default_title = f"L1 RYW18 OOF v1 Tile{args.tile_size} O{overlap_code:02d} Preflight Fold01"
    title = args.title or default_title
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,62}", slug):
        raise ValueError("Kernel slug must be 3-63 lowercase letters, digits, or hyphens")
    run_name = f"dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_tile{args.tile_size}o{overlap_code:02d}_focus6_mix_fold01"
    output_root = args.output_root or ROOT / f"kaggle_{slug.replace('-', '_')}"
    return slug, title, run_name, output_root.resolve()


def build_metadata(
    owner: str,
    slug: str,
    title: str,
    code_file: str,
    use_gpu: bool,
    candidate_dataset_ref: str | None = None,
) -> dict[str, object]:
    datasets = [IMAGE_DATASET_REF, CONFIG_DATASET_REF]
    if use_gpu:
        datasets.append(BASELINE_DATASET_REF)
    if candidate_dataset_ref:
        datasets.append(candidate_dataset_ref)
    return {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": use_gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not PIPELINE_PATH.is_file():
        raise FileNotFoundError(f"Missing embedded tile pipeline: {PIPELINE_PATH}")
    slug, title, run_name, output_root = defaults(args)
    code_file = f"{slug}.py"
    if args.mode == "evaluate" and (not args.candidate_dataset_ref or not args.candidate_weight_name):
        raise ValueError("evaluate mode requires a candidate checkpoint Dataset reference and filename")
    metadata = build_metadata(
        args.owner,
        slug,
        title,
        code_file,
        args.mode in {"train", "evaluate"},
        args.candidate_dataset_ref if args.mode == "evaluate" else None,
    )
    script = textwrap.dedent(
        SCRIPT_TEMPLATE.substitute(
            mode=args.mode,
            run_name=run_name,
            pipeline_source=json.dumps(PIPELINE_PATH.read_text(encoding="utf-8")),
            baseline_weight_name=BASELINE_WEIGHT_NAME,
            candidate_weight_name=args.candidate_weight_name,
            tile_size=args.tile_size,
            overlap=args.overlap,
            negative_ratio=args.negative_ratio,
            min_visible_fraction=args.min_visible_fraction,
            focus_classes=args.focus_classes,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            predict_batch=args.predict_batch,
        )
    )
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing Kernel directory: {output_root}")
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "metadata": metadata, "run_name": run_name}, ensure_ascii=False, indent=2))
        return 0
    output_root.mkdir(parents=True)
    (output_root / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / code_file).write_text(script, encoding="utf-8", newline="\n")
    print(f"[created] {output_root}")
    print(f"[push] kaggle kernels push -p {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
