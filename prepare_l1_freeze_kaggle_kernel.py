#!/usr/bin/env python3
"""Create one reproducible private Kaggle Kernel for the L1 freeze ablation."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from string import Template


ROOT = Path(__file__).resolve().parent
TRAINER_SOURCE = ROOT / "train_l1_freeze_two_stage.py"
IMAGE_DATASET_REF = "lly3323w/dianjienie"
CONFIG_DATASET_REF = "lly3323w/dianjienie-ryw18-oof-consensus-v1-dev-config"
DATA_VERSION = "oofconsensusv1"
VARIANTS = {"control": 0, "freeze10": 10}


SCRIPT_TEMPLATE = Template(
    r'''#!/usr/bin/env python3
"""Run one development-only L1 two-stage YOLO11m freeze task."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


INPUT_ROOT = Path("/kaggle/input")
RUN_NAME = "$run_name"
VARIANT = "$variant"
FREEZE = $freeze
MODE = "$mode"
FOLD = $fold
TRAINER_SOURCE = $trainer_source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def resolve_config_payload() -> tuple[Path, Path]:
    candidates = [root for root in mounted_roots() if (root / "config_archive_manifest.json").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate OOF config Dataset: {inventory()}")
    package_root = candidates[0]
    archive_manifest = json.loads((package_root / "config_archive_manifest.json").read_text(encoding="utf-8"))
    archive_name = archive_manifest.get("config_archive")
    payload_files = archive_manifest.get("payload_files")
    manifest_hash = archive_manifest.get("source_manifest_sha256")
    if not isinstance(archive_name, str) or not isinstance(payload_files, list) or not isinstance(manifest_hash, str):
        raise ValueError("Invalid OOF config archive manifest")
    payload_root = package_root / Path(archive_name).stem
    actual = {path.name for path in payload_root.iterdir() if path.is_file()} if payload_root.is_dir() else set()
    if actual != set(payload_files):
        raise ValueError(f"Expanded OOF config payload mismatch: expected={len(payload_files)}, actual={len(actual)}")
    config_manifest = payload_root / "config_manifest.json"
    if not config_manifest.is_file() or sha256_file(config_manifest) != manifest_hash:
        raise ValueError("OOF config manifest hash mismatch")
    runner = payload_root / "runner__kaggle_dianjienie_ryw18_existing_data_runner.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Missing OOF runtime runner: {runner}")
    return payload_root, runner


def resolve_image_root(payload_root: Path) -> Path:
    manifest = json.loads((payload_root / "config_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("data_version") != "oofconsensusv1" or manifest.get("development_only") is not True:
        raise ValueError("This Kernel requires the development-only OOF-consensus configuration")
    images = manifest.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], dict):
        raise ValueError("OOF config has no valid source image probe")
    source_relative = images[0].get("source_relative")
    if not isinstance(source_relative, str):
        raise ValueError("OOF source image probe is invalid")
    parts = PurePosixPath(source_relative).parts
    candidates = [
        root for root in mounted_roots()
        if root.is_dir() and root != payload_root.parent and root.joinpath(*parts).is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate image Dataset: matches={candidates}, inputs={inventory()}")
    return candidates[0]


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

    if (
        torch.__version__ != "2.6.0+cu124"
        or torchvision.__version__.split("+", 1)[0] != "0.21.0"
        or numpy.__version__ != "1.26.4"
        or ultralytics.__version__ != "8.4.66"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Pinned runtime verification failed: "
            f"torch={torch.__version__}, torchvision={torchvision.__version__}, "
            f"numpy={numpy.__version__}, ultralytics={ultralytics.__version__}, cuda={torch.cuda.is_available()}"
        )


def main() -> None:
    payload_root, runner = resolve_config_payload()
    image_root = resolve_image_root(payload_root)
    anchor_root = Path("/kaggle/working") / f"{RUN_NAME}_anchor"
    prepare = [
        sys.executable,
        str(runner),
        "--config-root", str(payload_root),
        "--data-version", "oofconsensusv1",
        "--development-only",
        "--image-dataset-root", str(image_root),
        "--model", "yolo11m.pt",
        "--fold", str(FOLD),
        "--work-root", str(anchor_root),
        "--prepare-only",
    ]
    print("[prepare]", " ".join(prepare), flush=True)
    subprocess.run(prepare, check=True)
    install_dependencies()
    trainer = Path("/kaggle/working/train_l1_freeze_two_stage.py")
    trainer.write_text(TRAINER_SOURCE, encoding="utf-8")
    command = [
        sys.executable,
        str(trainer),
        "--mode", MODE,
        "--anchor-root", str(anchor_root),
        "--data-version", "oofconsensusv1",
        "--variant", VARIANT,
        "--freeze", str(FREEZE),
        "--project", "/kaggle/working/runs",
        "--runtime-config-root", "/kaggle/working/freeze_runtime_configs",
        "--source-file", str(trainer),
    ]
    if MODE == "fold":
        command.extend(["--fold", str(FOLD)])
    print("[train]", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    run_dir = Path("/kaggle/working/runs") / RUN_NAME
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Expected run directory is missing: {run_dir}")
    archive = Path(shutil.make_archive(str(Path("/kaggle/working") / f"{RUN_NAME}_results"), "zip", run_dir.parent, run_dir.name))
    print(json.dumps({"run_dir": str(run_dir), "archive": str(archive), "archive_sha256": sha256_file(archive)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a private Kaggle GPU Kernel for one L1 freeze ablation variant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("fold", "final"), required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3))
    parser.add_argument("--owner", default="lly3323w")
    parser.add_argument("--slug")
    parser.add_argument("--title")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def defaults(args: argparse.Namespace) -> tuple[str, str, Path]:
    if args.mode == "fold" and args.fold is None:
        raise ValueError("--mode fold requires --fold")
    if args.mode == "final" and args.fold is not None:
        raise ValueError("--mode final must not receive --fold")
    freeze = VARIANTS[args.variant]
    if args.mode == "fold":
        slug = args.slug or f"l1-ryw18-y11m-{args.variant}-s1e20-s2e63-f{args.fold:02d}"
        # Keep Kaggle's title-derived slug identical to the requested kernel id.
        title = args.title or f"L1 RYW18 Y11m {args.variant} S1e20 S2e63 F{args.fold:02d}"
        scope = f"fold{args.fold:02d}"
    else:
        if args.variant != "freeze10":
            raise ValueError("Only the validated freeze10 variant may be packaged as finaldev")
        slug = args.slug or "l1-ryw18-y11m-freeze10-s1e20-s2e63-finaldev"
        title = args.title or "L1 RYW18 Y11m Freeze10 S1e20 S2e63 Finaldev"
        scope = "finaldev"
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,62}", slug):
        raise ValueError("Kernel slug must be 3-63 lowercase letters, digits, or hyphens")
    run_name = (
        "dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_"
        f"{DATA_VERSION}_{args.variant}_s1e020_s2e063_{scope}"
    )
    output_root = (args.output_root or ROOT / f"kaggle_{slug.replace('-', '_')}").resolve()
    return slug, title, output_root, run_name, freeze


def metadata(owner: str, slug: str, title: str, code_file: str) -> dict[str, object]:
    return {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [IMAGE_DATASET_REF, CONFIG_DATASET_REF],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def render(mode: str, variant: str, freeze: int, fold: int | None, run_name: str, trainer_source: str) -> str:
    return textwrap.dedent(
        SCRIPT_TEMPLATE.substitute(
            mode=mode,
            variant=variant,
            freeze=freeze,
            fold=fold or 1,
            run_name=run_name,
            trainer_source=json.dumps(trainer_source),
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not TRAINER_SOURCE.is_file():
        raise FileNotFoundError(f"Missing local trainer source: {TRAINER_SOURCE}")
    slug, title, output_root, run_name, freeze = defaults(args)
    code_file = f"{slug}.py"
    rendered = render(args.mode, args.variant, freeze, args.fold, run_name, TRAINER_SOURCE.read_text(encoding="utf-8"))
    package_metadata = metadata(args.owner, slug, title, code_file)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing Kernel package: {output_root}")
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "metadata": package_metadata, "run_name": run_name}, ensure_ascii=False, indent=2))
        return 0
    output_root.mkdir(parents=True)
    (output_root / "kernel-metadata.json").write_text(
        json.dumps(package_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / code_file).write_text(rendered, encoding="utf-8", newline="\n")
    (output_root / "requested_config.json").write_text(
        json.dumps(
            {
                "experiment": "L1 YOLO11m paired two-stage freeze ablation",
                "variant": args.variant,
                "stage1_freeze": freeze,
                "stage1_epochs": 20,
                "stage2_freeze": 0,
                "stage2_epochs": 63,
                "mode": args.mode,
                "fold": args.fold,
                "data_version": DATA_VERSION,
                "development_only": True,
                "frozen_test_set": None,
                "run_name": run_name,
                "trainer_sha256": hashlib_sha256_text(TRAINER_SOURCE.read_text(encoding="utf-8")),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[created] {output_root}")
    print(f"[push] python kaggle_ipv4_cli.py kernels push -p {output_root}")
    return 0


def hashlib_sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
