#!/usr/bin/env python3
"""Train one RYW18 fold using the existing private Kaggle image Dataset.

Attach two private Kaggle Datasets to one GPU Notebook:
1. ``lly3323w/dianjienie`` for the original source images.
2. ``dianjienie-ryw18-anchor-v1-config`` for labels, fixed folds and code.

Only symbolic links are created in ``/kaggle/working``; image bytes are never
copied or unpacked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
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
        description="复用既有 Kaggle 图片 Dataset，在 RYW18 固定锚点上运行单个 Fold。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=("yolo11m.pt", "yolo26m.pt"), required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--data-version", default="anchorv1", help="必须与配置包 manifest 的数据版本一致。")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="只恢复并训练开发集 train/val；配置包必须同样声明 development_only。",
    )
    parser.add_argument("--config-root", type=Path, help="RYW18 配置 Dataset 的挂载根目录；省略时自动定位。")
    parser.add_argument("--image-dataset-root", type=Path, help="既有图片 Dataset 的挂载根目录；省略时自动定位。")
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
    parser.add_argument("--prepare-only", action="store_true", help="只恢复锚点链接并校验。")
    parser.add_argument("--dry-run", action="store_true", help="传给训练器，仅校验 Fold，不导入或训练模型。")
    return parser.parse_args(argv)


def load_manifest(config_root: Path) -> dict[str, Any]:
    path = config_root / "config_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"不是预期的 RYW18 配置 Dataset：{path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"配置 manifest 无法解析：{path}") from error
    if manifest.get("schema_version") != 1 or manifest.get("classes") != EXPECTED_CLASSES:
        raise ValueError("配置 manifest 的版本或类别顺序不符合 RYW18")
    counts = manifest.get("counts", {})
    if counts.get("images") != len(manifest.get("images", [])):
        raise ValueError("配置 manifest 图片数量不一致")
    if counts.get("labels") != len(manifest.get("labels", [])):
        raise ValueError("配置 manifest 标签数量不一致")
    if counts.get("metadata") != len(manifest.get("metadata", [])):
        raise ValueError("配置 manifest 元数据数量不一致")
    return manifest


def locate_config_root(requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    if not DEFAULT_INPUT_ROOT.is_dir():
        raise FileNotFoundError("未找到 /kaggle/input；请在 Kaggle GPU Notebook 中运行")
    candidates = [path for path in DEFAULT_INPUT_ROOT.iterdir() if (path / "config_manifest.json").is_file()]
    if len(candidates) != 1:
        detail = ", ".join(path.name for path in candidates) or "无候选"
        raise RuntimeError(f"无法唯一定位 RYW18 配置 Dataset；当前候选：{detail}")
    return candidates[0].resolve()


def source_path(root: Path, relative: str) -> Path:
    return root.joinpath(*PurePosixPath(relative).parts)


def locate_image_dataset_root(requested: Path | None, manifest: dict[str, Any], config_root: Path) -> Path:
    if requested is not None:
        return requested.resolve()
    if not DEFAULT_INPUT_ROOT.is_dir():
        raise FileNotFoundError("未找到 /kaggle/input；请传入 --image-dataset-root")
    preferred_name = str(manifest["image_dataset_ref"]).split("/")[-1]
    candidates = [DEFAULT_INPUT_ROOT / preferred_name]
    candidates.extend(path for path in DEFAULT_INPUT_ROOT.iterdir() if path != config_root and path.name != preferred_name)
    probe = manifest["images"][0]["source_relative"]
    matches = [candidate for candidate in candidates if candidate.is_dir() and source_path(candidate, probe).is_file()]
    if len(matches) != 1:
        detail = ", ".join(path.name for path in matches) or "无候选"
        raise RuntimeError(f"无法唯一定位既有图片 Dataset；当前候选：{detail}")
    return matches[0].resolve()


def validate_payload(config_root: Path, image_root: Path, manifest: dict[str, Any]) -> None:
    for record in manifest["images"]:
        image = source_path(image_root, record["source_relative"])
        if not image.is_file():
            raise FileNotFoundError(f"既有图片 Dataset 缺少锚点源图：{image}")
        if image.stat().st_size != record["source_bytes"]:
            raise ValueError(f"既有图片 Dataset 图片大小与锚点来源不一致：{image}")
    for record in manifest["labels"] + manifest["metadata"]:
        if not (config_root / record["upload_name"]).is_file():
            raise FileNotFoundError(f"配置 Dataset 缺少文件：{record['upload_name']}")
    for script_name in ("trainer", "runner"):
        name = manifest.get("scripts", {}).get(script_name)
        if not name or not (config_root / name).is_file():
            raise FileNotFoundError(f"配置 Dataset 缺少 {script_name} 脚本")


def manifest_fingerprint(manifest: dict[str, Any], image_root: Path) -> str:
    stable = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{stable}\n{image_root}".encode("utf-8")).hexdigest()


def make_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        try:
            if os.path.samefile(source, destination):
                return
        except OSError:
            pass
        raise FileExistsError(f"工作锚点中存在非预期文件，拒绝覆盖：{destination}")
    # Avoid the Windows unprivileged-symlink path, which can leave a dangling
    # destination entry before the hard-link fallback runs.
    if os.name == "nt":
        os.link(source, destination)
        return
    os.symlink(source, destination)


def restore_anchor(config_root: Path, image_root: Path, work_root: Path, manifest: dict[str, Any]) -> None:
    marker = work_root / ".config_manifest_sha256"
    fingerprint = manifest_fingerprint(manifest, image_root)
    if work_root.exists() and marker.is_file():
        if marker.read_text(encoding="utf-8").strip() != fingerprint:
            raise ValueError(f"工作锚点来自其他配置或图片 Dataset，拒绝混用：{work_root}")
    elif work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError(f"工作锚点目录已存在且无版本标记，拒绝覆盖：{work_root}")
    else:
        work_root.mkdir(parents=True, exist_ok=True)

    for record in manifest["images"]:
        make_link(source_path(image_root, record["source_relative"]), work_root / "images" / record["anchor_name"])
    for record in manifest["labels"] + manifest["metadata"]:
        make_link(config_root / record["upload_name"], work_root / record["relative_path"])
    marker.write_text(fingerprint + "\n", encoding="utf-8")

    expected = [work_root / "images", work_root / "labels", work_root / "classes.txt"]
    if not manifest.get("development_only", False):
        expected.append(work_root / "splits/test.txt")
    expected.extend(work_root / "splits/folds" / f"fold_{fold:02d}" / "data.yaml" for fold in range(1, 4))
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"恢复的 RYW18 锚点不完整，例如：{missing[0]}")


def install_and_check_dependencies(args: argparse.Namespace) -> None:
    if not args.skip_dependency_install:
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
        subprocess.run(torch_command, check=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "torchaudio"], check=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "numpy==1.26.4", f"ultralytics=={args.ultralytics_version}"],
            check=True,
        )
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


def run_name(args: argparse.Namespace) -> str:
    return (
        f"dianjienie_medium_{args.model.removesuffix('.pt')}_{args.imgsz}_kaggle"
        f"_seed{args.seed}_cm0_{args.data_version}_fold{args.fold:02d}"
    )


def trainer_command(args: argparse.Namespace, config_root: Path, manifest: dict[str, Any]) -> list[str]:
    trainer = config_root / manifest["scripts"]["trainer"]
    command = [
        sys.executable,
        str(trainer),
        "--model", args.model,
        "--fold", str(args.fold),
        "--data-version", args.data_version,
        "--anchor-root", str(args.work_root),
        "--project", str(args.project),
        "--platform", "kaggle",
        "--runtime-config-root", str(args.runtime_config_root),
        "--epochs", str(args.epochs),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--workers", str(args.workers),
        "--device", str(args.device),
        "--seed", str(args.seed),
        "--patience", str(args.patience),
    ]
    if args.development_only:
        command.append("--development-only")
    if args.dry_run:
        command.append("--dry-run")
    return command


def archive_run(args: argparse.Namespace) -> Path:
    run_dir = args.project / run_name(args)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"训练完成但未找到预期 run 目录：{run_dir}")
    archive_path = Path(shutil.make_archive(str(Path("/kaggle/working") / f"{run_dir.name}_results"), "zip", run_dir.parent, run_dir.name))
    print(f"[archive] {archive_path}", flush=True)
    return archive_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.data_version):
        raise ValueError("--data-version 只能包含 ASCII 字母、数字、下划线和连字符")
    config_root = locate_config_root(args.config_root)
    manifest = load_manifest(config_root)
    manifest_version = str(manifest.get("data_version", "anchorv1"))
    manifest_development_only = bool(manifest.get("development_only", False))
    if args.data_version != manifest_version:
        raise ValueError(f"数据版本与配置包不一致：请求 {args.data_version}，配置包为 {manifest_version}")
    if args.development_only != manifest_development_only:
        raise ValueError("--development-only 必须与配置包 manifest 的 development_only 一致")
    image_root = locate_image_dataset_root(args.image_dataset_root, manifest, config_root)
    if args.work_root == DEFAULT_WORK_ROOT and args.data_version != "anchorv1":
        args.work_root = DEFAULT_WORK_ROOT.with_name(f"dianjienie_ryw18_{args.data_version}")
    args.work_root = args.work_root.resolve()
    args.project = args.project.resolve()
    args.runtime_config_root = args.runtime_config_root.resolve()
    validate_payload(config_root, image_root, manifest)
    restore_anchor(config_root, image_root, args.work_root, manifest)
    print(
        json.dumps(
            {
                "config_root": str(config_root),
                "image_dataset_root": str(image_root),
                "work_root": str(args.work_root),
                "images": manifest["counts"]["images"],
                "labels": manifest["counts"]["labels"],
                "data_version": args.data_version,
                "development_only": args.development_only,
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
    command = trainer_command(args, config_root, manifest)
    print("[train]", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not args.dry_run:
        archive_run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
