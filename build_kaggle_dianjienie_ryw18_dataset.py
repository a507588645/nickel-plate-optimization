#!/usr/bin/env python3
"""Build a fallback full-image Kaggle upload view for the RYW18 anchor.

Kaggle's CLI archives subdirectories when uploading a dataset.  The resulting
archive would have to be unpacked in ``/kaggle/working`` and duplicate the
large image payload.  This builder keeps each PNG and its same-stem label at
the Dataset root.  The Kaggle-side runner maps both ``images/`` and
``labels/`` to that root, so Ultralytics works whether it resolves links or
not.  PNGs are NTFS hard links; labels and metadata are independent copies.

The project already has a private Kaggle image Dataset that matches the source
anchor.  Prefer ``build_kaggle_dianjienie_ryw18_config_dataset.py`` so images
are not uploaded again.  This fallback requires an explicit second opt-in.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_ANCHOR_ROOT = Path(r"E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1")
DEFAULT_OUTPUT_ROOT = ROOT / "kaggle_dianjienie_ryw18_anchor_v1" / "dataset_upload_flat_v3"
DEFAULT_DATASET_SLUG = "dianjienie-ryw18-anchor-v1"
DEFAULT_DATASET_TITLE = "dianjienie RYW18 Anchor v1"
TRAINER_SOURCE = ROOT / "train_dianjienie_ryw18_anchor.py"
RUNNER_SOURCE = ROOT / "kaggle_dianjienie_ryw18_runner.py"
EXPECTED_CLASSES = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="创建 RYW18 全量图片 Kaggle 上传视图；已有图片集可用时仅作回退方案。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--dataset-title", default=DEFAULT_DATASET_TITLE)
    parser.add_argument(
        "--owner-slug",
        help="Kaggle 用户名；省略时，本地视图使用占位符，--publish 时从已登录 Kaggle API 读取。",
    )
    parser.add_argument("--publish", action="store_true", help="明确确认后创建私有 Kaggle Dataset。")
    parser.add_argument(
        "--allow-redundant-image-upload",
        action="store_true",
        help="确认既有私有图片 Dataset 不可用后，允许上传完整约 21 GiB 图片负载。",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅审计源锚点和输出计划，不创建文件。")
    return parser.parse_args(argv)


def expected_metadata_paths() -> list[Path]:
    paths = [Path("classes.txt"), Path("data.yaml"), Path("README.md"), Path("splits/test.txt")]
    for fold in range(1, 4):
        fold_root = Path("splits/folds") / f"fold_{fold:02d}"
        paths.extend(fold_root / name for name in ("train.txt", "val.txt", "data.yaml"))
    return paths


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少必要文件：{path}")


def load_classes(anchor_root: Path) -> list[str]:
    classes_path = anchor_root / "classes.txt"
    require_file(classes_path)
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if classes != EXPECTED_CLASSES:
        raise ValueError("classes.txt 不等于已冻结的 RYW18 类别顺序，拒绝创建上传视图")
    return classes


def collect_source_files(anchor_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    images_root = anchor_root / "images"
    labels_root = anchor_root / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError("锚点缺少 images/ 或 labels/ 目录")

    images = sorted(path for path in images_root.iterdir() if path.is_file() and path.suffix.lower() == ".png")
    if not images:
        raise ValueError("images/ 中没有 PNG 图片")
    labels = []
    for image in images:
        label = labels_root / f"{image.stem}.txt"
        require_file(label)
        labels.append(label)
    extra_labels = sorted(path for path in labels_root.iterdir() if path.is_file() and path.suffix.lower() == ".txt")
    if len(extra_labels) != len(labels):
        raise ValueError("labels/ 含有孤立或重复标签，拒绝上传")

    metadata_paths = expected_metadata_paths()
    for relative_path in metadata_paths:
        require_file(anchor_root / relative_path)
    return images, labels, metadata_paths


def source_file_record(source: Path, upload_name: str, relative_path: str) -> dict[str, str]:
    return {
        "source_name": source.name,
        "upload_name": upload_name,
        "relative_path": relative_path,
    }


def create_hard_link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        try:
            if os.path.samefile(source, destination):
                return
        except OSError:
            pass
        raise FileExistsError(f"目标文件已存在且不是预期硬链接：{destination}")
    os.link(source, destination)


def copy_small_file(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.read_bytes() == source.read_bytes():
            return
        raise FileExistsError(f"目标文件已存在且内容不同：{destination}")
    shutil.copy2(source, destination)


def read_existing_manifest(output_root: Path) -> dict[str, Any] | None:
    manifest_path = output_root / "flat_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"已有上传视图的 manifest 损坏：{manifest_path}") from error


def validate_existing_view(output_root: Path, anchor_root: Path) -> dict[str, Any] | None:
    if not output_root.exists():
        return None
    manifest = read_existing_manifest(output_root)
    if manifest is None:
        raise FileExistsError(f"输出目录已存在但不是完整上传视图，拒绝覆盖：{output_root}")
    if manifest.get("anchor_root") != str(anchor_root):
        raise ValueError(f"上传视图来源不是本次锚点，拒绝混用：{output_root}")
    for group in ("images", "labels", "metadata"):
        for record in manifest.get("files", {}).get(group, []):
            if not (output_root / record["upload_name"]).is_file():
                raise FileNotFoundError(f"已有上传视图缺少文件：{record['upload_name']}")
    for script_name in ("trainer", "runner"):
        file_name = manifest.get("scripts", {}).get(script_name)
        if not file_name or not (output_root / file_name).is_file():
            raise FileNotFoundError(f"已有上传视图缺少 {script_name} 脚本")
    return manifest


def kaggle_owner_slug() -> str:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError as error:
        raise RuntimeError("未安装 Kaggle API，无法执行 --publish") from error
    api = KaggleApi()
    api.authenticate()
    owner = api.config_values.get("username")
    if not owner:
        raise RuntimeError("Kaggle API 凭据中未找到 username")
    return str(owner)


def make_upload_manifest(
    anchor_root: Path,
    images: list[Path],
    labels: list[Path],
    metadata_paths: list[Path],
) -> dict[str, Any]:
    image_records = [
        source_file_record(image, image.name, f"images/{image.name}")
        for image in images
    ]
    label_records = [
        source_file_record(label, label.name, f"labels/{label.name}")
        for label in labels
    ]
    metadata_records = []
    for relative_path in metadata_paths:
        source = anchor_root / relative_path
        upload_name = "meta__" + "__".join(relative_path.parts)
        metadata_records.append(source_file_record(source, upload_name, relative_path.as_posix()))
    return {
        "schema_version": 2,
        "anchor_name": anchor_root.name,
        "anchor_root": str(anchor_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "classes": load_classes(anchor_root),
        "counts": {"images": len(images), "labels": len(labels), "metadata": len(metadata_records)},
        "files": {"images": image_records, "labels": label_records, "metadata": metadata_records},
        "scripts": {
            "trainer": "trainer__train_dianjienie_ryw18_anchor.py",
            "runner": "runner__kaggle_dianjienie_ryw18_runner.py",
        },
    }


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_dataset_metadata(
    output_root: Path,
    owner_slug: str | None,
    dataset_slug: str,
    dataset_title: str,
) -> None:
    metadata = {
        "title": dataset_title,
        "id": f"{owner_slug or '__KAGGLE_USERNAME__'}/{dataset_slug}",
        "licenses": [{"name": "other"}],
    }
    write_text(output_root / "dataset-metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def copy_runtime_scripts(output_root: Path, manifest: dict[str, Any]) -> None:
    trainer_name = manifest["scripts"]["trainer"]
    runner_name = manifest["scripts"]["runner"]
    shutil.copy2(TRAINER_SOURCE, output_root / trainer_name)
    shutil.copy2(RUNNER_SOURCE, output_root / runner_name)


def create_view(
    output_root: Path,
    anchor_root: Path,
    images: list[Path],
    labels: list[Path],
    metadata_paths: list[Path],
    owner_slug: str | None,
    dataset_slug: str,
    dataset_title: str,
) -> dict[str, Any]:
    if output_root.exists():
        existing = validate_existing_view(output_root, anchor_root)
        assert existing is not None
        copy_runtime_scripts(output_root, existing)
        write_dataset_metadata(output_root, owner_slug, dataset_slug, dataset_title)
        return existing

    if not TRAINER_SOURCE.is_file() or not RUNNER_SOURCE.is_file():
        raise FileNotFoundError("缺少 Kaggle 包所需的训练器或运行器脚本")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f"{output_root.name}.building_{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"检测到未完成的上传视图，请人工检查后改名：{staging_root}")
    staging_root.mkdir()
    try:
        manifest = make_upload_manifest(anchor_root, images, labels, metadata_paths)
        for record, source in zip(manifest["files"]["images"], images, strict=True):
            create_hard_link(source, staging_root / record["upload_name"])
        for record, source in zip(manifest["files"]["labels"], labels, strict=True):
            copy_small_file(source, staging_root / record["upload_name"])
        for record in manifest["files"]["metadata"]:
            copy_small_file(anchor_root / record["relative_path"], staging_root / record["upload_name"])

        copy_runtime_scripts(staging_root, manifest)
        write_text(staging_root / "flat_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        write_dataset_metadata(staging_root, owner_slug, dataset_slug, dataset_title)
        write_text(
            staging_root / "README_UPLOAD.md",
            "# Kaggle 上传视图\n\n"
            "此目录由 `build_kaggle_dianjienie_ryw18_dataset.py` 生成。图片为硬链接，标签和元数据为独立副本；"
            "不得在此目录直接编辑。上传时使用 `kaggle datasets create --dir-mode skip`，"
            "以保留单文件挂载；Kaggle 端使用 `runner__kaggle_dianjienie_ryw18_runner.py` 重建符号链接锚点。\n",
        )
        staging_root.replace(output_root)
        return manifest
    except Exception:
        # Preserve a failed staging view for inspection rather than silently deleting data.
        raise


def publish_view(output_root: Path) -> None:
    command = [
        "kaggle",
        "datasets",
        "create",
        "--path",
        str(output_root),
        "--dir-mode",
        "skip",
        "--keep-tabular",
    ]
    print("[publish]", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print("[publish] completed.", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    anchor_root = args.anchor_root.resolve()
    output_root = args.output_root.resolve()
    images, labels, metadata_paths = collect_source_files(anchor_root)
    load_classes(anchor_root)
    print(
        json.dumps(
            {
                "anchor_root": str(anchor_root),
                "output_root": str(output_root),
                "image_count": len(images),
                "label_count": len(labels),
                "metadata_count": len(metadata_paths),
                "publish": args.publish,
                "image_hard_link_mode": True,
                "label_metadata_copy_mode": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    if args.publish and not args.allow_redundant_image_upload:
        raise RuntimeError(
            "已存在可复用的私有 Kaggle 图片 Dataset；拒绝重复上传图片。"
            "请改用 build_kaggle_dianjienie_ryw18_config_dataset.py，"
            "或在确认既有图片集不可用后显式传入 --allow-redundant-image-upload。"
        )

    owner_slug = args.owner_slug
    if args.publish and not owner_slug:
        owner_slug = kaggle_owner_slug()
    manifest = create_view(
        output_root,
        anchor_root,
        images,
        labels,
        metadata_paths,
        owner_slug,
        args.dataset_slug,
        args.dataset_title,
    )
    print(f"[view] ready: {output_root} ({manifest['counts']['images']} images, no image copy)")
    if args.publish:
        publish_view(output_root)
    else:
        print("[view] no external upload was performed. Re-run with --publish only after reviewing the private Dataset metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
