#!/usr/bin/env python3
"""Run one RYW18 anchor fold in a Kaggle GPU notebook.

This program is shipped inside the flat Kaggle Dataset created by
``build_kaggle_dianjienie_ryw18_dataset.py``.  It maps both ``images/`` and
``labels/`` to the Dataset root in ``/kaggle/working`` and invokes the project
trainer with portable paths; all image bytes remain in the read-only input
mount.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path("/kaggle/input")
DEFAULT_WORK_ROOT = Path("/kaggle/working/dianjienie_ryw18_anchor_v1")
DEFAULT_PROJECT_ROOT = Path("/kaggle/working/runs")
DEFAULT_RUNTIME_ROOT = Path("/kaggle/working/training_runtime_configs")
DEFAULT_TORCH_VERSION = "2.6.0"
DEFAULT_TORCHVISION_VERSION = "0.21.0"
DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
DEFAULT_ULTRALYTICS_VERSION = "8.4.66"
EXPECTED_CLASSES = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 Kaggle GPU 上运行 dianjienie RYW18 锚点的单个 Fold。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=("yolo11m.pt", "yolo26m.pt"), required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--dataset-root", type=Path, help="已挂载 Kaggle Dataset 根目录；省略时自动定位。")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--runtime-config-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--torch-version", default=DEFAULT_TORCH_VERSION)
    parser.add_argument("--torchvision-version", default=DEFAULT_TORCHVISION_VERSION)
    parser.add_argument("--torch-index-url", default=DEFAULT_TORCH_INDEX_URL)
    parser.add_argument("--ultralytics-version", default=DEFAULT_ULTRALYTICS_VERSION)
    parser.add_argument("--skip-dependency-install", action="store_true", help="使用 Notebook 已安装的依赖。")
    parser.add_argument("--prepare-only", action="store_true", help="只恢复符号链接锚点并校验。")
    parser.add_argument("--dry-run", action="store_true", help="传给训练器，仅校验 Fold，不导入或训练模型。")
    return parser.parse_args(argv)


def load_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "flat_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"不是预期的 Kaggle RYW18 数据集：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Kaggle 上传 manifest 无法解析：{manifest_path}") from error
    if manifest.get("schema_version") != 2 or manifest.get("classes") != EXPECTED_CLASSES:
        raise ValueError("Kaggle 上传 manifest 的版本或类别顺序不符合 RYW18")
    for group in ("images", "labels", "metadata"):
        if not manifest.get("files", {}).get(group):
            raise ValueError(f"Kaggle 上传 manifest 缺少 {group} 文件清单")
    if manifest.get("counts", {}).get("images") != len(manifest["files"]["images"]):
        raise ValueError("图片数量与 Kaggle 上传 manifest 不一致")
    if manifest.get("counts", {}).get("labels") != len(manifest["files"]["labels"]):
        raise ValueError("标签数量与 Kaggle 上传 manifest 不一致")
    return manifest


def locate_dataset_root(requested_root: Path | None) -> Path:
    if requested_root is not None:
        return requested_root.resolve()
    if not DEFAULT_INPUT_ROOT.is_dir():
        raise FileNotFoundError("未找到 /kaggle/input；请在 Kaggle GPU Notebook 中运行")
    candidates = [path for path in DEFAULT_INPUT_ROOT.iterdir() if (path / "flat_manifest.json").is_file()]
    if len(candidates) != 1:
        detail = ", ".join(path.name for path in candidates) or "无候选"
        raise RuntimeError(
            "无法唯一定位 RYW18 数据集；请附加一个含 flat_manifest.json 的 Dataset，"
            f"或传入 --dataset-root。当前候选：{detail}"
        )
    return candidates[0].resolve()


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    stable = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def make_symbolic_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        try:
            if os.path.samefile(source, destination):
                return
        except OSError:
            pass
        raise FileExistsError(f"工作锚点中存在非预期文件，拒绝覆盖：{destination}")
    try:
        os.symlink(source, destination)
    except OSError as error:
        # Kaggle/Linux supports symbolic links.  The fallback only exists so
        # the exact preparation path can be dry-run on Windows hosts where
        # the current user lacks SeCreateSymbolicLinkPrivilege.
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            os.link(source, destination)
            return
        raise


def make_directory_link(source: Path, destination: Path) -> bool:
    """Return True when a directory-level link was created or already exists."""
    if destination.exists() or destination.is_symlink():
        try:
            if os.path.samefile(source, destination):
                return True
        except OSError:
            pass
        raise FileExistsError(f"工作锚点中存在非预期目录，拒绝覆盖：{destination}")
    try:
        os.symlink(source, destination, target_is_directory=True)
        return True
    except OSError as error:
        # The local Windows dry-run host may lack symlink privileges.  Return
        # False there so individual hard-link fallback can validate the same
        # layout; Kaggle/Linux uses the directory symlink above.
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            return False
        raise


def restore_anchor(dataset_root: Path, work_root: Path, manifest: dict[str, Any]) -> None:
    marker_path = work_root / ".flat_manifest_sha256"
    fingerprint = manifest_fingerprint(manifest)
    if work_root.exists() and marker_path.is_file():
        existing = marker_path.read_text(encoding="utf-8").strip()
        if existing != fingerprint:
            raise ValueError(f"工作锚点对应其他版本 Dataset，拒绝混用：{work_root}")
    elif work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError(f"工作锚点目录已存在且无版本标记，拒绝覆盖：{work_root}")
    else:
        work_root.mkdir(parents=True, exist_ok=True)

    for group in ("images", "labels", "metadata"):
        for record in manifest["files"][group]:
            source = dataset_root / record["upload_name"]
            if not source.is_file():
                raise FileNotFoundError(f"Kaggle Dataset 缺少上传文件：{source}")

    images_linked = make_directory_link(dataset_root, work_root / "images")
    labels_linked = make_directory_link(dataset_root, work_root / "labels")
    if not (images_linked and labels_linked):
        # Windows-only validation fallback.  It maintains the same visible
        # ``images/<name>.png`` and ``labels/<name>.txt`` tree without copying.
        for group in ("images", "labels"):
            for record in manifest["files"][group]:
                make_symbolic_link(dataset_root / record["upload_name"], work_root / record["relative_path"])
    for record in manifest["files"]["metadata"]:
        make_symbolic_link(dataset_root / record["upload_name"], work_root / record["relative_path"])
    marker_path.write_text(fingerprint + "\n", encoding="utf-8")

    expected_paths = [
        work_root / "images",
        work_root / "labels",
        work_root / "classes.txt",
        work_root / "splits/test.txt",
    ]
    expected_paths.extend(work_root / "splits/folds" / f"fold_{fold:02d}" / "data.yaml" for fold in range(1, 4))
    missing = [path for path in expected_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"符号链接锚点不完整，例如：{missing[0]}")


def install_and_check_dependencies(args: argparse.Namespace) -> None:
    if not args.skip_dependency_install:
        # This is the same CUDA stack used by the project's completed Kaggle
        # YOLO11/YOLO26 baseline runs.  Reinstalling it avoids silently using
        # a changing Notebook image with a different YOLO26 implementation.
        torch_command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--force-reinstall",
            f"torch=={args.torch_version}",
            f"torchvision=={args.torchvision_version}",
            "--index-url",
            args.torch_index_url,
        ]
        print("[dependency]", " ".join(torch_command), flush=True)
        subprocess.run(torch_command, check=True)
        torchaudio_command = [sys.executable, "-m", "pip", "uninstall", "--yes", "torchaudio"]
        print("[dependency]", " ".join(torchaudio_command), flush=True)
        subprocess.run(torchaudio_command, check=True)
        ultralytics_command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "numpy==1.26.4",
            f"ultralytics=={args.ultralytics_version}",
        ]
        print("[dependency]", " ".join(ultralytics_command), flush=True)
        subprocess.run(ultralytics_command, check=True)
    try:
        import torch
        import ultralytics
    except ModuleNotFoundError as error:
        raise RuntimeError("缺少 torch 或 ultralytics；请开启 Kaggle Internet 后重试") from error
    print(f"[environment] torch={torch.__version__}, cuda={torch.version.cuda}", flush=True)
    print(f"[environment] ultralytics={ultralytics.__version__}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 GPU；请在 Kaggle Notebook 设置中选择 GPU accelerator")
    print(f"[environment] gpu={torch.cuda.get_device_name(0)}", flush=True)


def trainer_command(args: argparse.Namespace, dataset_root: Path, manifest: dict[str, Any]) -> list[str]:
    trainer_name = manifest.get("scripts", {}).get("trainer")
    if not trainer_name:
        raise ValueError("Kaggle 上传 manifest 未记录训练器名称")
    trainer = dataset_root / trainer_name
    if not trainer.is_file():
        raise FileNotFoundError(f"Kaggle Dataset 缺少训练器：{trainer}")
    command = [
        sys.executable,
        str(trainer),
        "--model",
        args.model,
        "--fold",
        str(args.fold),
        "--anchor-root",
        str(args.work_root),
        "--project",
        str(args.project),
        "--platform",
        "kaggle",
        "--runtime-config-root",
        str(args.runtime_config_root),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--patience",
        str(args.patience),
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def run_name(args: argparse.Namespace) -> str:
    model_name = args.model.removesuffix(".pt")
    return (
        f"dianjienie_medium_{model_name}_{args.imgsz}_kaggle"
        f"_seed{args.seed}_cm0_anchorv1_fold{args.fold:02d}"
    )


def archive_run(args: argparse.Namespace) -> Path:
    run_dir = args.project / run_name(args)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"训练完成但未找到预期 run 目录：{run_dir}")
    archive_base = Path("/kaggle/working") / f"{run_dir.name}_results"
    archive_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=run_dir.parent,
            base_dir=run_dir.name,
        )
    )
    print(f"[archive] {archive_path}", flush=True)
    return archive_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = locate_dataset_root(args.dataset_root)
    args.work_root = args.work_root.resolve()
    args.project = args.project.resolve()
    args.runtime_config_root = args.runtime_config_root.resolve()
    manifest = load_manifest(dataset_root)
    restore_anchor(dataset_root, args.work_root, manifest)
    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "work_root": str(args.work_root),
                "images": manifest["counts"]["images"],
                "labels": manifest["counts"]["labels"],
                "model": args.model,
                "fold": args.fold,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.prepare_only:
        return 0
    if not args.dry_run:
        install_and_check_dependencies(args)
    command = trainer_command(args, dataset_root, manifest)
    print("[train]", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not args.dry_run:
        archive_run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
