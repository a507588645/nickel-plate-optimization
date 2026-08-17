#!/usr/bin/env python3
"""Create private Kaggle preflight or GPU Kernel for the L1 OOF final run."""

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
MODEL_OPTIONS = ("yolo11m.pt", "yolo26m.pt")
DEFAULT_MODEL = "yolo11m.pt"
MODEL_SPECS = {
    "yolo11m.pt": {
        "config_dataset_ref": "lly3323w/dianjienie-ryw18-oofconsensus-v1-final-config",
        "fixed_epoch": 83,
        "slug": "l1-ryw18-yolo11m-oof-consensus-v1-finaldev",
        "title": "L1 RYW18 YOLO11m OOF Consensus v1 Finaldev",
    },
    "yolo26m.pt": {
        "config_dataset_ref": "lly3323w/dianjienie-ryw18-oof-y26-final",
        "fixed_epoch": None,
        "slug": "l1-ryw18-yolo26m-oof-consensus-v1-finaldev",
        "title": "L1 RYW18 YOLO26m OOF Consensus v1 Finaldev",
    },
}


SCRIPT_TEMPLATE = Template(
    r'''#!/usr/bin/env python3
"""Run L1 no-manual final OOF preflight or fixed-epoch GPU training."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


INPUT_ROOT = Path("/kaggle/input")
DATA_VERSION = "$data_version"
MODE = "$mode"
RUN_NAME = "$run_name"
EXPECTED_MODEL = "$expected_model"
EXPECTED_FIXED_EPOCH = $expected_fixed_epoch


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


def resolve_payload() -> tuple[Path, dict, Path, Path]:
    package_roots = [path for path in mounted_dataset_roots() if (path / "config_archive_manifest.json").is_file()]
    if len(package_roots) != 1:
        raise RuntimeError(f"Cannot uniquely locate final config Dataset: {input_inventory()}")
    package_root = package_roots[0]
    archive_manifest = json.loads((package_root / "config_archive_manifest.json").read_text(encoding="utf-8"))
    archive_name = archive_manifest.get("config_archive")
    expected_files = archive_manifest.get("payload_files")
    expected_manifest_hash = archive_manifest.get("source_manifest_sha256")
    if not isinstance(archive_name, str) or not isinstance(expected_files, list) or not isinstance(expected_manifest_hash, str):
        raise ValueError("Invalid final config archive manifest")
    payload_root = package_root / Path(archive_name).stem
    actual_files = {path.name for path in payload_root.iterdir() if path.is_file()} if payload_root.is_dir() else set()
    if actual_files != set(expected_files):
        raise ValueError(f"Expanded final config payload mismatch: expected={len(expected_files)}, actual={len(actual_files)}")
    manifest_path = payload_root / "config_manifest.json"
    if sha256_file(manifest_path) != expected_manifest_hash:
        raise ValueError("Expanded final config_manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_runtime = manifest.get("final_runtime")
    if manifest.get("data_version") != DATA_VERSION or not manifest.get("development_only") or not isinstance(final_runtime, dict):
        raise ValueError("Final config Dataset has wrong data version or does not declare final runtime")
    runner_name = manifest.get("scripts", {}).get("runner")
    trainer_name = final_runtime.get("trainer")
    epoch_name = final_runtime.get("epoch_selection")
    if not all(isinstance(value, str) for value in (runner_name, trainer_name, epoch_name)):
        raise ValueError("Final config manifest has invalid runtime paths")
    runner, trainer, epoch_summary = payload_root / runner_name, payload_root / trainer_name, payload_root / epoch_name
    if not runner.is_file() or not trainer.is_file() or not epoch_summary.is_file():
        raise FileNotFoundError("Final config payload is missing runner, trainer, or epoch summary")
    if sha256_file(trainer) != final_runtime.get("trainer_sha256"):
        raise ValueError("Final trainer hash mismatch")
    if sha256_file(epoch_summary) != final_runtime.get("epoch_selection_sha256"):
        raise ValueError("Final epoch summary hash mismatch")
    if final_runtime.get("expected_model") != EXPECTED_MODEL or final_runtime.get("expected_fixed_epoch") != EXPECTED_FIXED_EPOCH:
        raise ValueError(f"Final runtime is not the frozen {EXPECTED_MODEL} epoch-{EXPECTED_FIXED_EPOCH} configuration")
    print(f"[job] package={package_root}, payload={payload_root}", flush=True)
    return payload_root, manifest, runner, trainer


def resolve_image_root(payload_root: Path, manifest: dict) -> Path:
    images = manifest.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], dict):
        raise ValueError("Final config manifest has no image probe")
    source_relative = images[0].get("source_relative")
    if not isinstance(source_relative, str):
        raise ValueError("Final config image probe is invalid")
    parts = PurePosixPath(source_relative).parts
    candidates = [
        root for root in mounted_dataset_roots()
        if root.is_dir() and root != payload_root.parent and root.joinpath(*parts).is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate image Dataset: matches={candidates}, inputs={input_inventory()}")
    print(f"[job] image_dataset={candidates[0]}", flush=True)
    return candidates[0]


def install_dependencies() -> None:
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall",
            "torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cu124",
        ],
        check=True,
    )
    # Kaggle images do not always include torchaudio; it is irrelevant here.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "torchaudio"], check=False)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "numpy==1.26.4", "ultralytics==8.4.66"],
        check=True,
    )


def main() -> None:
    payload_root, manifest, runner, trainer = resolve_payload()
    image_root = resolve_image_root(payload_root, manifest)
    work_root = Path("/kaggle/working/dianjienie_ryw18_oofconsensusv1")
    runtime_root = Path("/kaggle/working/final_runtime_configs")
    prepare = [
        sys.executable, str(runner),
        "--config-root", str(payload_root),
        "--data-version", DATA_VERSION,
        "--development-only",
        "--image-dataset-root", str(image_root),
        "--work-root", str(work_root),
        "--model", EXPECTED_MODEL, "--fold", "1", "--prepare-only",
    ]
    print("[prepare]", " ".join(prepare), flush=True)
    subprocess.run(prepare, check=True)
    epoch_summary = payload_root / manifest["final_runtime"]["epoch_selection"]
    command = [
        sys.executable, str(trainer),
        "--anchor-root", str(work_root),
        "--selection-summary", str(epoch_summary),
        "--project", "/kaggle/working/runs",
        "--runtime-config-root", str(runtime_root),
        "--model", EXPECTED_MODEL,
        "--device", "0", "--batch", "8", "--workers", "8",
    ]
    if MODE == "preflight":
        command.append("--dry-run")
    else:
        install_dependencies()
    print("[final]", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if MODE == "train":
        run_dir = Path("/kaggle/working/runs") / RUN_NAME
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Final run directory missing: {run_dir}")
        required = [
            "weights/best.pt", "weights/last.pt", "weights/final_fixed_epoch.pt", "args.yaml",
            "requested_config.json", "environment.json", "results.csv", "README.md",
            "dataset_config/data.yaml", "dataset_config/train.txt", "dataset_config/val.txt",
            "dataset_config/epoch_selection_summary.json", "release_weight.json",
        ]
        missing = [relative for relative in required if not (run_dir / relative).is_file()]
        if missing:
            raise FileNotFoundError(f"Final run missing required evidence: {missing}")
        archive = Path(shutil.make_archive(str(Path("/kaggle/working") / f"{RUN_NAME}_results"), "zip", run_dir.parent, run_dir.name))
        print(f"[archive] {archive}", flush=True)


if __name__ == "__main__":
    main()
'''
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 L1 OOF 无人工最终训练的 Kaggle preflight 或 GPU Kernel。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--model", choices=MODEL_OPTIONS, default=DEFAULT_MODEL)
    parser.add_argument("--fixed-epoch", type=int, help="与配置包 manifest 一致的冻结训练轮数。")
    parser.add_argument("--config-dataset-ref", help="私有 Kaggle 最终配置 Dataset 引用。")
    parser.add_argument("--owner", default="lly3323w")
    parser.add_argument("--slug")
    parser.add_argument("--title")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def defaults(args: argparse.Namespace) -> tuple[str, str, Path, int, str, str]:
    spec = MODEL_SPECS[args.model]
    if args.mode == "preflight":
        if args.model == DEFAULT_MODEL:
            slug = args.slug or "l1-ryw18-oof-consensus-v1-final-preflight"
            title = args.title or "L1 RYW18 OOF Consensus v1 Final Preflight"
        else:
            model_tag = args.model.removesuffix(".pt")
            slug = args.slug or f"l1-ryw18-{model_tag}-oof-consensus-v1-final-preflight"
            title = args.title or f"L1 RYW18 {model_tag.replace('yolo', 'YOLO', 1)} OOF Consensus v1 Final Preflight"
    else:
        slug = args.slug or str(spec["slug"])
        title = args.title or str(spec["title"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,62}", slug):
        raise ValueError("Kernel slug must be 3-63 lowercase letters, digits, or hyphens")
    fixed_epoch = args.fixed_epoch if args.fixed_epoch is not None else spec["fixed_epoch"]
    if not isinstance(fixed_epoch, int) or fixed_epoch <= 0:
        raise ValueError(f"{args.model} 必须通过 --fixed-epoch 明确指定正数固定训练轮数")
    config_dataset_ref = str(args.config_dataset_ref or spec["config_dataset_ref"])
    model_tag = args.model.removesuffix(".pt")
    run_name = f"dianjienie_medium_{model_tag}_640_kaggle_seed0_cm0_{DATA_VERSION}_finaldev_ep{fixed_epoch:03d}"
    output_root = args.output_root or ROOT / f"kaggle_{slug.replace('-', '_')}"
    return slug, title, output_root.resolve(), fixed_epoch, config_dataset_ref, run_name


def metadata(owner: str, slug: str, title: str, code_file: str, use_gpu: bool, config_dataset_ref: str) -> dict[str, object]:
    return {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": use_gpu,
        "enable_tpu": False,
        "enable_internet": use_gpu,
        "dataset_sources": [IMAGE_DATASET_REF, config_dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    slug, title, output_root, fixed_epoch, config_dataset_ref, run_name = defaults(args)
    code_file = f"{slug}.py"
    record = metadata(args.owner, slug, title, code_file, args.mode == "train", config_dataset_ref)
    if output_root.exists():
        raise FileExistsError(f"Kernel 输出目录已存在，拒绝覆盖：{output_root}")
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "metadata": record}, ensure_ascii=False, indent=2))
        return 0
    output_root.mkdir(parents=True)
    (output_root / "kernel-metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / code_file).write_text(
        textwrap.dedent(
            SCRIPT_TEMPLATE.substitute(
                data_version=DATA_VERSION,
                mode=args.mode,
                run_name=run_name,
                expected_model=args.model,
                expected_fixed_epoch=str(fixed_epoch),
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"[created] {output_root}")
    print(f"[push] kaggle kernels push -p {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
