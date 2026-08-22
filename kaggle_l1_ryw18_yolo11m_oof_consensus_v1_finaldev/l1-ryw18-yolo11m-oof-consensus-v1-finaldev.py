#!/usr/bin/env python3
"""Run L1 no-manual final OOF preflight or fixed-epoch GPU training."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


INPUT_ROOT = Path("/kaggle/input")
DATA_VERSION = "oofconsensusv1"
MODE = "train"
RUN_NAME = "dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_finaldev_ep083"


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
    if final_runtime.get("expected_model") != "yolo11m.pt" or final_runtime.get("expected_fixed_epoch") != 83:
        raise ValueError("Final runtime is not the frozen YOLO11m epoch-83 configuration")
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
        "--model", "yolo11m.pt", "--fold", "1", "--prepare-only",
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
