#!/usr/bin/env python3
"""Build the small Kaggle companion Dataset for the RYW18 anchor.

The existing private Kaggle Dataset ``lly3323w/dianjienie`` contains all
1325 source images used by the frozen RYW18 anchor.  This builder therefore
uploads only the corrected anchor labels, fixed split files, source-image
mapping and training code.  It never uploads the 21 GiB image payload again.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_ANCHOR_ROOT = Path(r"E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1")
DEFAULT_SOURCE_ROOT = Path(r"E:/PythonProject/datasets/dianjienie")
DEFAULT_FLAT_OUTPUT_ROOT = ROOT / "kaggle_dianjienie_ryw18_anchor_v1" / "config_upload_flat_v1"
DEFAULT_OUTPUT_ROOT = ROOT / "kaggle_dianjienie_ryw18_anchor_v1" / "config_upload_archive_v1"
DEFAULT_IMAGE_DATASET_REF = "lly3323w/dianjienie"
DEFAULT_CONFIG_DATASET_SLUG = "dianjienie-ryw18-anchor-v1-config"
DEFAULT_CONFIG_DATASET_TITLE = "dianjienie RYW18 Anchor v1 Config"
CONFIG_ARCHIVE_NAME = "ryw18_config_payload_v1.zip"
TRAINER_SOURCE = ROOT / "train_dianjienie_ryw18_anchor.py"
RUNNER_SOURCE = ROOT / "kaggle_dianjienie_ryw18_existing_data_runner.py"
ARCHIVE_BOOTSTRAP_SOURCE = ROOT / "kaggle_dianjienie_ryw18_archive_bootstrap.py"
EXPECTED_CLASSES = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="创建复用既有 Kaggle 图片 Dataset 的 RYW18 小型配置包；默认不上传。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--anchor-root", type=Path, default=DEFAULT_ANCHOR_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--data-version", default="anchorv1", help="写入配置 manifest 的数据版本标识。")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="配置包只包含开发集 train/val；不打包或链接冻结 test 清单。",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--flat-output-root",
        type=Path,
        default=DEFAULT_FLAT_OUTPUT_ROOT,
        help="保存已验证标签和训练文件的内部平铺构建目录。",
    )
    parser.add_argument(
        "--package-format",
        choices=("archive", "flat"),
        default="archive",
        help="archive 将全部配置压为一个 zip，避免 Kaggle 逐文件上传过慢。",
    )
    parser.add_argument("--image-dataset-ref", default=DEFAULT_IMAGE_DATASET_REF)
    parser.add_argument("--dataset-slug", default=DEFAULT_CONFIG_DATASET_SLUG)
    parser.add_argument("--dataset-title", default=DEFAULT_CONFIG_DATASET_TITLE)
    parser.add_argument("--owner-slug", help="Kaggle 用户名；--publish 时可从已登录 API 自动读取。")
    parser.add_argument(
        "--verify-source-dataset",
        action="store_true",
        help="只读核验既有 Kaggle 图片集是否包含每张锚点源图及相同文件大小。",
    )
    parser.add_argument("--publish", action="store_true", help="明确确认后创建私有 Kaggle 配置 Dataset。")
    parser.add_argument("--dry-run", action="store_true", help="只审计构建计划，不写文件、不联网。")
    return parser.parse_args(argv)


def expected_metadata_paths(development_only: bool = False) -> list[Path]:
    paths = [Path("classes.txt"), Path("data.yaml"), Path("README.md")]
    if not development_only:
        paths.append(Path("splits/test.txt"))
    for fold in range(1, 4):
        fold_root = Path("splits/folds") / f"fold_{fold:02d}"
        paths.extend(fold_root / name for name in ("train.txt", "val.txt", "data.yaml"))
    return paths


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少必要文件：{path}")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def load_classes(anchor_root: Path) -> list[str]:
    classes_path = anchor_root / "classes.txt"
    require_file(classes_path)
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if classes != EXPECTED_CLASSES:
        raise ValueError("classes.txt 不等于已冻结的 RYW18 类别顺序，拒绝构建配置包")
    return classes


def source_relative_path(source_image: str, source_root: Path) -> str:
    source = PureWindowsPath(source_image)
    root = PureWindowsPath(str(source_root))
    try:
        return source.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"锚点源图片不在预期 source_root 内：{source_image}") from error


def collect_anchor_records(
    anchor_root: Path, source_root: Path, development_only: bool = False
) -> tuple[list[dict[str, Any]], list[Path]]:
    manifest_path = anchor_root / "manifests" / "source_manifest.csv"
    require_file(manifest_path)
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError("source_manifest.csv 为空")

    records: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    seen_sources: set[str] = set()
    for row in rows:
        anchor_image = Path(row["anchor_image"])
        anchor_label = Path(row["anchor_label"])
        source_image = Path(row["source_image"])
        require_file(anchor_image)
        require_file(anchor_label)
        require_file(source_image)
        if anchor_image.suffix.lower() != ".png" or anchor_label.suffix.lower() != ".txt":
            raise ValueError(f"锚点文件扩展名异常：{anchor_image}")
        if anchor_label.stem != anchor_image.stem:
            raise ValueError(f"锚点图片与标签名不一致：{anchor_image.name}")
        relative = source_relative_path(row["source_image"], source_root)
        if anchor_image.name in seen_images:
            raise ValueError(f"锚点图片名重复：{anchor_image.name}")
        if relative in seen_sources:
            raise ValueError(f"同一源图片重复进入锚点：{relative}")
        seen_images.add(anchor_image.name)
        seen_sources.add(relative)
        records.append(
            {
                "anchor_image": anchor_image.name,
                "anchor_label": anchor_label.name,
                "source_relative": relative,
                "source_bytes": source_image.stat().st_size,
                "label_source": anchor_label,
            }
        )

    anchor_images = sorted(path.name for path in (anchor_root / "images").glob("*.png"))
    anchor_labels = sorted(path.name for path in (anchor_root / "labels").glob("*.txt"))
    if sorted(seen_images) != anchor_images:
        raise ValueError("source_manifest 与锚点 images/ 不一致")
    expected_labels = sorted(Path(image).with_suffix(".txt").name for image in anchor_images)
    if anchor_labels != expected_labels:
        raise ValueError("锚点 labels/ 与 images/ 不能一一对应")
    return sorted(records, key=lambda item: item["anchor_image"]), expected_metadata_paths(development_only)


def make_manifest(
    anchor_root: Path,
    source_root: Path,
    image_dataset_ref: str,
    image_records: list[dict[str, Any]],
    metadata_paths: list[Path],
    data_version: str,
    development_only: bool,
) -> dict[str, Any]:
    labels = [
        {
            "upload_name": f"label__{record['anchor_label']}",
            "relative_path": f"labels/{record['anchor_label']}",
            "source_name": record["anchor_label"],
        }
        for record in image_records
    ]
    metadata = [
        {
            "upload_name": "meta__" + "__".join(relative.parts),
            "relative_path": relative.as_posix(),
        }
        for relative in metadata_paths
    ]
    return {
        "schema_version": 1,
        "anchor_name": anchor_root.name,
        "anchor_root": str(anchor_root),
        "source_root": str(source_root),
        "data_version": data_version,
        "development_only": development_only,
        "image_dataset_ref": image_dataset_ref,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "classes": load_classes(anchor_root),
        "counts": {"images": len(image_records), "labels": len(labels), "metadata": len(metadata)},
        "images": [
            {
                "anchor_name": record["anchor_image"],
                "source_relative": record["source_relative"],
                "source_bytes": record["source_bytes"],
            }
            for record in image_records
        ],
        "labels": labels,
        "metadata": metadata,
        "scripts": {
            "trainer": "trainer__train_dianjienie_ryw18_anchor.py",
            "runner": "runner__kaggle_dianjienie_ryw18_existing_data_runner.py",
        },
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


def write_dataset_metadata(output_root: Path, owner_slug: str | None, slug: str, title: str) -> None:
    metadata = {
        "title": title,
        "id": f"{owner_slug or '__KAGGLE_USERNAME__'}/{slug}",
        "licenses": [{"name": "other"}],
    }
    write_text(output_root / "dataset-metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_existing_view(
    output_root: Path,
    anchor_root: Path,
    image_dataset_ref: str,
    data_version: str,
    development_only: bool,
) -> dict[str, Any]:
    manifest_path = output_root / "config_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"输出目录已存在但不是完整配置包，拒绝覆盖：{output_root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"配置包 manifest 损坏：{manifest_path}") from error
    if manifest.get("schema_version") != 1:
        raise ValueError(f"配置包 schema 不兼容：{output_root}")
    if manifest.get("anchor_root") != str(anchor_root) or manifest.get("image_dataset_ref") != image_dataset_ref:
        raise ValueError(f"配置包来源与本次请求不同，拒绝混用：{output_root}")
    if manifest.get("data_version", "anchorv1") != data_version or bool(manifest.get("development_only", False)) != development_only:
        raise ValueError(f"配置包数据版本或 development_only 状态不匹配，拒绝混用：{output_root}")
    for record in manifest.get("labels", []) + manifest.get("metadata", []):
        if not (output_root / record["upload_name"]).is_file():
            raise FileNotFoundError(f"配置包缺少文件：{record['upload_name']}")
    for script_name in ("trainer", "runner"):
        name = manifest.get("scripts", {}).get(script_name)
        if not name or not (output_root / name).is_file():
            raise FileNotFoundError(f"配置包缺少 {script_name} 脚本")
    return manifest


def copy_runtime_scripts(output_root: Path, manifest: dict[str, Any]) -> None:
    shutil.copy2(TRAINER_SOURCE, output_root / manifest["scripts"]["trainer"])
    shutil.copy2(RUNNER_SOURCE, output_root / manifest["scripts"]["runner"])


def create_view(
    output_root: Path,
    anchor_root: Path,
    source_root: Path,
    image_dataset_ref: str,
    owner_slug: str | None,
    slug: str,
    title: str,
    data_version: str,
    development_only: bool,
) -> dict[str, Any]:
    if output_root.exists():
        manifest = validate_existing_view(
            output_root, anchor_root, image_dataset_ref, data_version, development_only
        )
        copy_runtime_scripts(output_root, manifest)
        write_dataset_metadata(output_root, owner_slug, slug, title)
        return manifest

    if not TRAINER_SOURCE.is_file() or not RUNNER_SOURCE.is_file():
        raise FileNotFoundError("缺少配置包需要的训练器或既有图片集运行器")
    image_records, metadata_paths = collect_anchor_records(anchor_root, source_root, development_only)
    manifest = make_manifest(
        anchor_root,
        source_root,
        image_dataset_ref,
        image_records,
        metadata_paths,
        data_version,
        development_only,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f"{output_root.name}.building_{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"检测到未完成构建目录，请人工检查：{staging_root}")
    staging_root.mkdir()
    try:
        labels_by_name = {record["anchor_label"]: Path(record["label_source"]) for record in image_records}
        for record in manifest["labels"]:
            copy_small_file(labels_by_name[record["source_name"]], staging_root / record["upload_name"])
        for record in manifest["metadata"]:
            copy_small_file(anchor_root / record["relative_path"], staging_root / record["upload_name"])
        copy_runtime_scripts(staging_root, manifest)
        write_text(staging_root / "config_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        write_dataset_metadata(staging_root, owner_slug, slug, title)
        write_text(
            staging_root / "README_CONFIG_UPLOAD.md",
            "# RYW18 Kaggle 配置包\n\n"
            "此包不包含图片。训练时同时附加私有图片集 `lly3323w/dianjienie`，"
            "由运行器根据 `config_manifest.json` 将固定锚点图片、标签和 Fold 恢复到工作目录。"
            "标签是锚点独立副本，不得在此目录直接编辑。"
            + ("此包是 development_only，不包含冻结 test。\n" if development_only else "\n"),
        )
        staging_root.replace(output_root)
        return manifest
    except Exception:
        # Keep a failed staging directory for diagnosis rather than deleting it.
        raise


def archive_payload_files(flat_root: Path) -> list[Path]:
    excluded = {"dataset-metadata.json"}
    files = [path for path in flat_root.iterdir() if path.is_file() and path.name not in excluded]
    if not files:
        raise ValueError(f"平铺配置包没有可归档文件：{flat_root}")
    return sorted(files, key=lambda path: path.name)


def write_config_archive(archive_path: Path, flat_root: Path) -> tuple[list[str], str]:
    payload_files = archive_payload_files(flat_root)
    temporary = archive_path.with_suffix(archive_path.suffix + ".building")
    if temporary.exists():
        raise FileExistsError(f"发现未完成的配置归档，请人工检查：{temporary}")
    with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in payload_files:
            archive.write(path, arcname=path.name)
    temporary.replace(archive_path)
    return [path.name for path in payload_files], sha256_file(archive_path)


def validate_existing_archive_view(output_root: Path, flat_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "config_archive_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"归档输出目录已存在但缺少 manifest，拒绝覆盖：{output_root}")
    try:
        archive_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"归档 manifest 损坏：{manifest_path}") from error
    if archive_manifest.get("schema_version") != 1:
        raise ValueError(f"归档 manifest schema 不兼容：{manifest_path}")
    archive_path = output_root / str(archive_manifest.get("config_archive", ""))
    bootstrap_path = output_root / str(archive_manifest.get("bootstrap", ""))
    if not archive_path.is_file() or not bootstrap_path.is_file():
        raise FileNotFoundError(f"归档发布包不完整：{output_root}")
    if archive_manifest.get("config_archive_sha256") != sha256_file(archive_path):
        raise ValueError(f"归档哈希不一致：{archive_path}")
    source_manifest = flat_root / "config_manifest.json"
    if archive_manifest.get("source_manifest_sha256") != sha256_file(source_manifest):
        raise ValueError("平铺配置包已变化；请使用新的 archive 输出目录重新构建")
    return archive_manifest


def create_archive_upload_view(
    output_root: Path,
    flat_root: Path,
    owner_slug: str | None,
    slug: str,
    title: str,
) -> dict[str, Any]:
    if output_root.exists():
        archive_manifest = validate_existing_archive_view(output_root, flat_root)
        write_dataset_metadata(output_root, owner_slug, slug, title)
        return archive_manifest
    require_file(ARCHIVE_BOOTSTRAP_SOURCE)
    source_manifest = flat_root / "config_manifest.json"
    require_file(source_manifest)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f"{output_root.name}.building_{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"检测到未完成构建目录，请人工检查：{staging_root}")
    staging_root.mkdir()
    try:
        archive_path = staging_root / CONFIG_ARCHIVE_NAME
        payload_files, archive_hash = write_config_archive(archive_path, flat_root)
        bootstrap_name = ARCHIVE_BOOTSTRAP_SOURCE.name
        copy_small_file(ARCHIVE_BOOTSTRAP_SOURCE, staging_root / bootstrap_name)
        write_dataset_metadata(staging_root, owner_slug, slug, title)
        archive_manifest = {
            "schema_version": 1,
            "config_archive": CONFIG_ARCHIVE_NAME,
            "config_archive_sha256": archive_hash,
            "source_manifest_sha256": sha256_file(source_manifest),
            "payload_files": payload_files,
            "bootstrap": bootstrap_name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_text(
            staging_root / "config_archive_manifest.json",
            json.dumps(archive_manifest, ensure_ascii=False, indent=2) + "\n",
        )
        write_text(
            staging_root / "README_CONFIG_UPLOAD.md",
            "# RYW18 Kaggle archive 配置包\n\n"
            "此包不包含图片。zip 保存标签、固定 Fold 和训练器；"
            "顶层启动脚本会在 Kaggle working 目录安全解压该归档，再复用私有图片集。\n",
        )
        staging_root.replace(output_root)
        return archive_manifest
    except Exception:
        # Keep a failed staging directory for diagnosis rather than deleting it.
        raise


def kaggle_owner_slug() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    owner = api.config_values.get("username")
    if not owner:
        raise RuntimeError("Kaggle API 凭据中未找到 username")
    return str(owner)


def verify_source_dataset(manifest: dict[str, Any]) -> dict[str, int]:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    page_token: str | None = None
    remote_sizes: dict[str, int] = {}
    while True:
        response = api.dataset_list_files(manifest["image_dataset_ref"], page_token=page_token, page_size=200)
        entries = list(getattr(response, "dataset_files", None) or getattr(response, "files", None) or [])
        if not entries:
            raise RuntimeError("Kaggle API 未返回图片集文件清单")
        for entry in entries:
            remote_sizes[str(entry.name).replace("\\", "/")] = int(entry.total_bytes)
        page_token = getattr(response, "next_page_token", None)
        if not page_token:
            break
    missing = [record for record in manifest["images"] if record["source_relative"] not in remote_sizes]
    size_mismatch = [
        record
        for record in manifest["images"]
        if record["source_relative"] in remote_sizes
        and remote_sizes[record["source_relative"]] != record["source_bytes"]
    ]
    if missing or size_mismatch:
        detail = missing[0]["source_relative"] if missing else size_mismatch[0]["source_relative"]
        raise ValueError(f"既有 Kaggle 图片集与锚点不一致，例如：{detail}")
    return {"remote_files": len(remote_sizes), "matched_anchor_images": len(manifest["images"])}


def publish_view(output_root: Path) -> None:
    # Use the same authenticated API client as source-dataset verification.
    # The PowerShell-only kaggle.bat shim can otherwise select a different
    # Python runtime and hang before reaching the already-working API path.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"[publish] KaggleApi.dataset_create_new({output_root})", flush=True)
    api.dataset_create_new(str(output_root), public=False, quiet=False, convert_to_csv=True, dir_mode="skip")
    print("[publish] completed.", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.data_version):
        raise ValueError("--data-version 只能包含 ASCII 字母、数字、下划线和连字符")
    anchor_root = args.anchor_root.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    flat_output_root = args.flat_output_root.resolve()
    image_records, metadata_paths = collect_anchor_records(anchor_root, source_root, args.development_only)
    load_classes(anchor_root)
    plan = {
        "anchor_root": str(anchor_root),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "flat_output_root": str(flat_output_root),
        "image_dataset_ref": args.image_dataset_ref,
        "data_version": args.data_version,
        "development_only": args.development_only,
        "image_count": len(image_records),
        "label_count": len(image_records),
        "metadata_count": len(metadata_paths),
        "publish": args.publish,
        "verify_source_dataset": args.verify_source_dataset or args.publish,
        "package_format": args.package_format,
        "uploads_images": False,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    owner_slug = args.owner_slug
    if args.publish and not owner_slug:
        owner_slug = kaggle_owner_slug()
    manifest = create_view(
        flat_output_root,
        anchor_root,
        source_root,
        args.image_dataset_ref,
        owner_slug,
        args.dataset_slug,
        args.dataset_title,
        args.data_version,
        args.development_only,
    )
    if args.package_format == "archive":
        upload_root = output_root
        create_archive_upload_view(
            upload_root,
            flat_output_root,
            owner_slug,
            args.dataset_slug,
            args.dataset_title,
        )
    else:
        upload_root = flat_output_root
    if args.verify_source_dataset or args.publish:
        verified = verify_source_dataset(manifest)
        print(f"[verify] remote_files={verified['remote_files']}, matched_anchor_images={verified['matched_anchor_images']}")
    print(f"[view] ready: {upload_root} (no images included)")
    if args.publish:
        publish_view(upload_root)
    else:
        print("[view] no external upload was performed. Use --publish only after confirming the private companion Dataset creation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
