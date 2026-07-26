#!/usr/bin/env python3
"""Prepare the small RYW18 Kaggle configuration package and run one fold.

The companion Kaggle Dataset contains this bootstrap script plus one archive.
Kaggle can automatically expand that archive when mounting a Dataset, so this
script accepts either form. It only writes to the Kaggle working directory;
source images remain read-only in the separately attached private image Dataset.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
import argparse
from pathlib import Path, PurePosixPath


DEFAULT_INPUT_ROOT = Path("/kaggle/input")
DEFAULT_RUNTIME_ROOT = Path("/kaggle/working/ryw18_config_runtime_v1")
ARCHIVE_MANIFEST_NAME = "config_archive_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_package_root() -> Path:
    candidates = [
        path
        for path in DEFAULT_INPUT_ROOT.iterdir()
        if (path / ARCHIVE_MANIFEST_NAME).is_file()
    ]
    if len(candidates) != 1:
        detail = ", ".join(path.name for path in candidates) or "无候选"
        raise RuntimeError(f"无法唯一定位 RYW18 archive 配置 Dataset：{detail}")
    return candidates[0].resolve()


def read_archive_manifest(package_root: Path) -> dict[str, object]:
    manifest_path = package_root / ARCHIVE_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"配置归档 manifest 无法解析：{manifest_path}") from error
    if manifest.get("schema_version") != 1:
        raise ValueError("配置归档 schema 不兼容")
    archive_name = manifest.get("config_archive")
    archive_hash = manifest.get("config_archive_sha256")
    source_manifest_hash = manifest.get("source_manifest_sha256")
    bootstrap_name = manifest.get("bootstrap")
    payload_files = manifest.get("payload_files")
    if (
        not isinstance(archive_name, str)
        or Path(archive_name).name != archive_name
        or not isinstance(archive_hash, str)
        or not isinstance(source_manifest_hash, str)
    ):
        raise ValueError("配置归档缺少文件名或哈希")
    if not isinstance(bootstrap_name, str) or not isinstance(payload_files, list):
        raise ValueError("配置归档缺少启动器或 payload 清单")
    if bootstrap_name != Path(__file__).name:
        raise ValueError("启动器名称与配置归档 manifest 不一致")
    if (
        not all(isinstance(name, str) and Path(name).name == name for name in payload_files)
        or len(set(payload_files)) != len(payload_files)
    ):
        raise ValueError("配置归档 payload 文件名不安全")
    return manifest


def validate_config_manifest_hash(config_root: Path, manifest: dict[str, object]) -> None:
    config_manifest = config_root / "config_manifest.json"
    if not config_manifest.is_file():
        raise FileNotFoundError(f"配置包缺少 config_manifest.json：{config_manifest}")
    if sha256_file(config_manifest) != manifest["source_manifest_sha256"]:
        raise ValueError(f"配置包 config_manifest 哈希不一致：{config_manifest}")


def validate_expanded_payload(expanded_root: Path, expected_files: set[str]) -> None:
    if not expanded_root.is_dir():
        raise FileNotFoundError(f"配置 archive 与自动展开目录均不存在：{expanded_root}")
    entries = list(expanded_root.iterdir())
    actual_files = {path.name for path in entries if path.is_file()}
    if any(path.is_dir() for path in entries) or actual_files != expected_files:
        raise ValueError("Kaggle 自动展开的配置内容与 manifest 不一致")


def unpack_config(package_root: Path, runtime_root: Path, manifest: dict[str, object]) -> Path:
    archive_path = package_root / str(manifest["config_archive"])
    expected_files = set(manifest["payload_files"])
    marker = runtime_root / ".config_archive_sha256"
    if runtime_root.exists():
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == manifest["config_archive_sha256"]:
            actual_files = {path.name for path in runtime_root.iterdir() if path.is_file() and path.name != marker.name}
            if actual_files == expected_files:
                validate_config_manifest_hash(runtime_root, manifest)
                return runtime_root
        if any(runtime_root.iterdir()):
            raise FileExistsError(f"运行时配置目录不是当前 archive，拒绝覆盖：{runtime_root}")
    runtime_root.mkdir(parents=True, exist_ok=True)
    if archive_path.is_file():
        if sha256_file(archive_path) != manifest["config_archive_sha256"]:
            raise ValueError(f"配置 archive 哈希不一致：{archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            archive_files = {entry.filename for entry in archive.infolist() if not entry.is_dir()}
            if archive_files != expected_files:
                raise ValueError("配置 archive 内容与 manifest 不一致")
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                path = PurePosixPath(entry.filename)
                if path.is_absolute() or len(path.parts) != 1 or path.name != entry.filename:
                    raise ValueError(f"配置 archive 含不安全路径：{entry.filename}")
                archive.extract(entry, runtime_root)
    else:
        expanded_root = package_root / Path(str(manifest["config_archive"])).stem
        validate_expanded_payload(expanded_root, expected_files)
        for name in expected_files:
            shutil.copy2(expanded_root / name, runtime_root / name)
    validate_config_manifest_hash(runtime_root, manifest)
    marker.write_text(str(manifest["config_archive_sha256"]) + "\n", encoding="utf-8")
    return runtime_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="解压 RYW18 小型配置 archive 并复用既有 Kaggle 图片集训练单个 Fold。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=("yolo11m.pt", "yolo26m.pt"), required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--runtime-config-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--image-dataset-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--torch-version")
    parser.add_argument("--torchvision-version")
    parser.add_argument("--torch-index-url")
    parser.add_argument("--ultralytics-version")
    parser.add_argument("--skip-dependency-install", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    package_root = find_package_root()
    manifest = read_archive_manifest(package_root)
    runtime_root = unpack_config(package_root, args.runtime_config_root.resolve(), manifest)
    runner = runtime_root / "runner__kaggle_dianjienie_ryw18_existing_data_runner.py"
    if not runner.is_file():
        raise FileNotFoundError(f"配置 archive 缺少训练运行器：{runner}")
    print(f"[archive] package={package_root}, runtime={runtime_root}", flush=True)
    command = [sys.executable, str(runner), "--config-root", str(runtime_root)]
    for name in (
        "image_dataset_root", "work_root", "project", "epochs", "imgsz", "batch", "workers",
        "device", "seed", "patience", "torch_version", "torchvision_version",
        "torch_index_url", "ultralytics_version",
    ):
        value = getattr(args, name)
        if value is not None:
            command.extend([f"--{name.replace('_', '-')}", str(value)])
    for name in ("skip_dependency_install", "prepare_only", "dry_run"):
        if getattr(args, name):
            command.append(f"--{name.replace('_', '-')}")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
