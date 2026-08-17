#!/usr/bin/env python3
"""Create a Kaggle-only high-resolution L1 Fold 1 experiment.

The experiment reuses the OOF-consensus development-only configuration and the
existing private image Dataset. It intentionally does not mount or read the
frozen historical release-validation images.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from string import Template


ROOT = Path(__file__).resolve().parent
DATA_VERSION = "oofconsensusv1"
IMAGE_DATASET_REF = "lly3323w/dianjienie"
CONFIG_DATASET_REF = "lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config"
MODEL = "yolo11m.pt"
FOLD = 1


SCRIPT_TEMPLATE = Template(
    r'''#!/usr/bin/env python3
"""Run one L1 OOF-consensus high-resolution development-only Kaggle job."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath


INPUT_ROOT = Path("/kaggle/input")
DATA_VERSION = "$data_version"
MODE = "$mode"
IMGSZ = $imgsz
BATCH = $batch
EPOCHS = $epochs


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
    runner = payload_root / "runner__kaggle_dianjienie_ryw18_existing_data_runner.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Missing OOF-consensus runtime runner: {runner}")
    print(f"[job] package={package_root}, payload={payload_root}", flush=True)
    return payload_root, runner


def resolve_image_dataset_root(payload_root: Path) -> Path:
    config_manifest = json.loads((payload_root / "config_manifest.json").read_text(encoding="utf-8"))
    if config_manifest.get("data_version") != DATA_VERSION or not config_manifest.get("development_only"):
        raise ValueError("Config manifest is not the required development-only OOF-consensus version")
    images = config_manifest.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], dict):
        raise ValueError("OOF-consensus config manifest has no source image probe")
    source_relative = images[0].get("source_relative")
    if not isinstance(source_relative, str):
        raise ValueError("OOF-consensus config manifest source image probe is invalid")
    probe_parts = PurePosixPath(source_relative).parts
    candidates = [
        root for root in mounted_dataset_roots()
        if root.is_dir() and root != payload_root.parent and root.joinpath(*probe_parts).is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cannot uniquely locate source image Dataset for {source_relative}: "
            f"matches={[str(path.relative_to(INPUT_ROOT)) for path in candidates]}, inputs={input_inventory()}"
        )
    print(f"[job] image_dataset={candidates[0]}", flush=True)
    return candidates[0]


def main() -> None:
    payload_root, runner = resolve_config_payload()
    image_dataset_root = resolve_image_dataset_root(payload_root)
    command = [
        sys.executable,
        str(runner),
        "--config-root",
        str(payload_root),
        "--data-version",
        DATA_VERSION,
        "--development-only",
        "--image-dataset-root",
        str(image_dataset_root),
        "--model",
        "yolo11m.pt",
        "--fold",
        "1",
    ]
    if MODE == "preflight":
        command.append("--prepare-only")
    else:
        command.extend([
            "--epochs", str(EPOCHS),
            "--imgsz", str(IMGSZ),
            "--batch", str(BATCH),
            "--workers", "8",
            "--device", "0",
            "--seed", "0",
            "--patience", "100",
        ])
    print("[job]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
'''
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Kaggle-only YOLO11m high-resolution L1 Fold 1 experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--imgsz", type=int, choices=(960, 1280), default=1280)
    parser.add_argument("--batch", type=int, choices=(2, 4), default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--owner", default="lly3323w")
    parser.add_argument("--slug", help="Kernel slug without the owner prefix.")
    parser.add_argument("--title", help="Kaggle Kernel title.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def defaults(args: argparse.Namespace) -> tuple[str, str, Path]:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    default_slug = f"l1-ryw18-yolo11m-oof-v1-hr{args.imgsz}-b{args.batch}-e{args.epochs}-fold01"
    slug = args.slug or default_slug
    title = args.title or f"L1 RYW18 YOLO11m OOF v1 HR{args.imgsz} B{args.batch} E{args.epochs} Fold01"
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,62}", slug):
        raise ValueError("Kernel slug must be 3-63 lowercase letters, digits, or hyphens")
    output_root = args.output_root or ROOT / f"kaggle_{slug.replace('-', '_')}"
    return slug, title, output_root.resolve()


def build_metadata(owner: str, slug: str, title: str, code_file: str, use_gpu: bool) -> dict[str, object]:
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
        "dataset_sources": [IMAGE_DATASET_REF, CONFIG_DATASET_REF],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def write_kernel(output_root: Path, metadata: dict[str, object], script: str, dry_run: bool) -> None:
    code_file = str(metadata["code_file"])
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing Kernel directory: {output_root}")
    if dry_run:
        print(json.dumps({"output_root": str(output_root), "metadata": metadata}, ensure_ascii=False, indent=2))
        return
    output_root.mkdir(parents=True)
    (output_root / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / code_file).write_text(script, encoding="utf-8", newline="\n")
    print(f"[created] {output_root}")
    print(f"[push] kaggle kernels push -p {output_root}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    slug, title, output_root = defaults(args)
    code_file = f"{slug}.py"
    metadata = build_metadata(args.owner, slug, title, code_file, args.mode == "train")
    script = textwrap.dedent(
        SCRIPT_TEMPLATE.substitute(
            data_version=DATA_VERSION,
            mode=args.mode,
            imgsz=args.imgsz,
            batch=args.batch,
            epochs=args.epochs,
        )
    )
    write_kernel(output_root, metadata, script, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
