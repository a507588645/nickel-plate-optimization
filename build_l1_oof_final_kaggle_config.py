#!/usr/bin/env python3
"""Build a private Kaggle config Dataset for the L1 no-manual final run.

The package reuses the already verified OOF development-only labels and image
mapping.  It adds only the final-training entry point and fixed epoch summary;
no image bytes and no release-validation files are uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_FLAT_ROOT = ROOT / "kaggle_dianjienie_ryw18_oofconsensus_v1_dev_only" / "config_upload_flat_v1"
FINAL_TRAINER = ROOT / "train_l1_oof_final.py"
ARCHIVE_NAME = "ryw18_oof_final_config_payload_v1.zip"
EXPECTED_DATA_VERSION = "oofconsensusv1"
MODEL_OPTIONS = ("yolo11m.pt", "yolo26m.pt")
DEFAULT_MODEL = "yolo11m.pt"
MODEL_SPECS: dict[str, dict[str, Path | str]] = {
    "yolo11m.pt": {
        "epoch_selection": ROOT / "l1_final_epoch_selection_oofconsensusv1_yolo11m_20260809" / "summary.json",
        "output_root": ROOT / "kaggle_dianjienie_ryw18_oofconsensus_v1_final" / "config_upload_flat_v1",
        "archive_root": ROOT / "kaggle_dianjienie_ryw18_oofconsensus_v1_final" / "config_upload_archive_v1",
        "dataset_slug": "dianjienie-ryw18-oofconsensus-v1-final-config",
        "dataset_title": "dianjienie RYW18 OOF Consensus v1 Final Config",
    },
    "yolo26m.pt": {
        "epoch_selection": ROOT / "l1_final_epoch_selection_oofconsensusv1_yolo26m_20260815" / "summary.json",
        "output_root": ROOT / "kaggle_dianjienie_ryw18_oofconsensus_v1_final_yolo26" / "config_upload_flat_v1",
        "archive_root": ROOT / "kaggle_dianjienie_ryw18_oofconsensus_v1_final_yolo26" / "config_upload_archive_v1",
        "dataset_slug": "dianjienie-ryw18-oof-y26-final",
        "dataset_title": "Dianjienie RYW18 OOF Y26 Final",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 L1 OOF 无人工最终训练的私有 Kaggle 配置包，不上传图片或 release-validation。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=MODEL_OPTIONS, default=DEFAULT_MODEL)
    parser.add_argument("--source-flat-root", type=Path, default=SOURCE_FLAT_ROOT)
    parser.add_argument("--final-trainer", type=Path, default=FINAL_TRAINER)
    parser.add_argument("--epoch-selection", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--archive-output-root", type=Path)
    parser.add_argument("--dataset-slug")
    parser.add_argument("--dataset-title")
    parser.add_argument("--owner-slug", default="lly3323w")
    parser.add_argument("--publish", action="store_true", help="通过已配置的 Kaggle 凭据创建私有配置 Dataset。")
    parser.add_argument("--dry-run", action="store_true", help="只校验来源和打印计划，不写文件或上传。")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def load_source_manifest(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / "config_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少已验证 OOF 配置 manifest：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("data_version") != EXPECTED_DATA_VERSION
        or manifest.get("development_only") is not True
        or manifest.get("counts", {}).get("images") != 1127
        or manifest.get("counts", {}).get("labels") != 1127
    ):
        raise ValueError("来源配置包不是预期的 1,127 图 OOF development-only 版本")
    for record in manifest.get("labels", []) + manifest.get("metadata", []):
        if not (source_root / record["upload_name"]).is_file():
            raise FileNotFoundError(f"来源配置包缺少文件：{record['upload_name']}")
    for script_name in ("trainer", "runner"):
        filename = manifest.get("scripts", {}).get(script_name)
        if not isinstance(filename, str) or not (source_root / filename).is_file():
            raise FileNotFoundError(f"来源配置包缺少 {script_name}：{filename}")
    return manifest


def load_epoch_selection(path: Path, model: str) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"缺少模型固定 epoch 选择摘要：{path}")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"固定 epoch 选择摘要无法解析：{path}") from error
    selected = summary.get("selected")
    policy = summary.get("final_training_policy")
    if (
        summary.get("model") != model
        or summary.get("data_version") != EXPECTED_DATA_VERSION
        or summary.get("development_only") is not True
        or not isinstance(selected, dict)
        or not isinstance(policy, dict)
        or not isinstance(selected.get("epoch"), int)
        or selected["epoch"] <= 0
        or policy.get("epochs") != selected["epoch"]
        or policy.get("validation") is not False
        or policy.get("patience") != 0
    ):
        raise ValueError(f"固定 epoch 选择摘要不是预期的 {model} OOF 最终训练策略")
    return int(selected["epoch"])


def dataset_metadata(owner_slug: str | None, slug: str, title: str) -> dict[str, Any]:
    return {
        "title": title,
        "id": f"{owner_slug or '__KAGGLE_USERNAME__'}/{slug}",
        "licenses": [{"name": "other"}],
    }


def build_flat_view(
    source_root: Path,
    final_trainer: Path,
    epoch_selection: Path,
    model: str,
    fixed_epoch: int,
    output_root: Path,
    owner_slug: str | None,
    slug: str,
    title: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"配置输出已存在，拒绝覆盖：{output_root}")
    if not final_trainer.is_file() or not epoch_selection.is_file():
        raise FileNotFoundError("缺少最终训练器或冻结 epoch 选择摘要")
    manifest = load_source_manifest(source_root)
    staging = output_root.parent / f"{output_root.name}.building_{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"发现未完成构建目录，请人工检查：{staging}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        for source in source_root.iterdir():
            if source.is_file() and source.name != "dataset-metadata.json":
                shutil.copy2(source, staging / source.name)
        final_trainer_name = "final_trainer__train_l1_oof_final.py"
        epoch_selection_name = "epoch_selection__summary.json"
        shutil.copy2(final_trainer, staging / final_trainer_name)
        shutil.copy2(epoch_selection, staging / epoch_selection_name)
        manifest["final_runtime"] = {
            "trainer": final_trainer_name,
            "trainer_sha256": sha256_file(staging / final_trainer_name),
            "epoch_selection": epoch_selection_name,
            "epoch_selection_sha256": sha256_file(staging / epoch_selection_name),
            "expected_model": model,
            "expected_fixed_epoch": fixed_epoch,
            "release_validation": "not included and not read by final training",
        }
        manifest["generated_at"] = datetime.now().isoformat(timespec="seconds")
        write_text(staging / "config_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        write_text(staging / "dataset-metadata.json", json.dumps(dataset_metadata(owner_slug, slug, title), ensure_ascii=False, indent=2) + "\n")
        write_text(
            staging / "README_CONFIG_UPLOAD.md",
            "# RYW18 OOF 无人工最终训练配置包\n\n"
            "此私有配置包只包含 1,127 张开发图对应的 OOF 候选标签、固定 Fold、运行器、"
            f"最终训练脚本和 `{model}` 的固定 {fixed_epoch} epoch 选择摘要。"
            "它不包含图片、模型权重或 198 张 release-validation。\n",
        )
        staging.replace(output_root)
    except Exception:
        # Preserve incomplete staged files for diagnosis instead of deleting them.
        raise
    return manifest


def create_archive_view(flat_root: Path, archive_root: Path, owner_slug: str | None, slug: str, title: str) -> dict[str, Any]:
    if archive_root.exists():
        raise FileExistsError(f"归档配置输出已存在，拒绝覆盖：{archive_root}")
    manifest_path = flat_root / "config_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少平铺配置 manifest：{manifest_path}")
    staging = archive_root.parent / f"{archive_root.name}.building_{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"发现未完成归档目录，请人工检查：{staging}")
    archive_root.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        archive_path = staging / ARCHIVE_NAME
        payload = sorted(
            path for path in flat_root.iterdir() if path.is_file() and path.name != "dataset-metadata.json"
        )
        if not payload:
            raise ValueError("平铺配置包为空")
        with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for source in payload:
                archive.write(source, arcname=source.name)
        archive_manifest = {
            "schema_version": 1,
            "config_archive": ARCHIVE_NAME,
            "config_archive_sha256": sha256_file(archive_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "payload_files": [source.name for source in payload],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_text(staging / "config_archive_manifest.json", json.dumps(archive_manifest, ensure_ascii=False, indent=2) + "\n")
        write_text(staging / "dataset-metadata.json", json.dumps(dataset_metadata(owner_slug, slug, title), ensure_ascii=False, indent=2) + "\n")
        write_text(
            staging / "README_CONFIG_UPLOAD.md",
            "# RYW18 OOF 无人工最终训练 archive 配置包\n\n"
            "zip 保存仅开发集的标签、清单和最终训练脚本；Kaggle 会在输入目录展开该 zip。\n",
        )
        staging.replace(archive_root)
    except Exception:
        raise
    return archive_manifest


def validate_existing_views(flat_root: Path, archive_root: Path, model: str, fixed_epoch: int) -> None:
    """Permit a separately requested publish without rebuilding immutable files."""
    source_manifest = flat_root / "config_manifest.json"
    archive_manifest_path = archive_root / "config_archive_manifest.json"
    if not source_manifest.is_file() or not archive_manifest_path.is_file():
        raise FileNotFoundError("已有配置包不完整，不能仅发布")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    final_runtime = manifest.get("final_runtime")
    if (
        manifest.get("data_version") != EXPECTED_DATA_VERSION
        or manifest.get("development_only") is not True
        or manifest.get("counts", {}).get("images") != 1127
        or not isinstance(final_runtime, dict)
        or final_runtime.get("expected_model") != model
        or final_runtime.get("expected_fixed_epoch") != fixed_epoch
    ):
        raise ValueError(f"已有配置包不是预期的 {model} OOF 最终训练版本")
    for key in ("trainer", "epoch_selection"):
        filename = final_runtime.get(key)
        if not isinstance(filename, str) or not (flat_root / filename).is_file():
            raise FileNotFoundError(f"已有配置包缺少 final_runtime.{key}")
    if sha256_file(flat_root / final_runtime["trainer"]) != final_runtime.get("trainer_sha256"):
        raise ValueError("已有最终训练器哈希不一致")
    if sha256_file(flat_root / final_runtime["epoch_selection"]) != final_runtime.get("epoch_selection_sha256"):
        raise ValueError("已有 epoch 选择摘要哈希不一致")
    archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
    archive = archive_root / str(archive_manifest.get("config_archive", ""))
    if not archive.is_file() or archive_manifest.get("config_archive_sha256") != sha256_file(archive):
        raise ValueError("已有配置归档缺失或哈希不一致")
    if archive_manifest.get("source_manifest_sha256") != sha256_file(source_manifest):
        raise ValueError("已有配置归档与平铺 manifest 不一致")
    with zipfile.ZipFile(archive) as handle:
        corrupt = handle.testzip()
        payload_files = set(archive_manifest.get("payload_files", []))
        actual_files = {item.filename for item in handle.infolist() if not item.is_dir()}
    if corrupt is not None or actual_files != payload_files:
        raise ValueError("已有配置归档 CRC 或文件清单不一致")


def publish(archive_root: Path) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"[publish] KaggleApi.dataset_create_new({archive_root})", flush=True)
    api.dataset_create_new(str(archive_root), public=False, quiet=False, convert_to_csv=True, dir_mode="skip")
    print("[publish] completed", flush=True)


def main() -> int:
    args = parse_args()
    spec = MODEL_SPECS[args.model]
    source_root = args.source_flat_root.resolve()
    final_trainer = args.final_trainer.resolve()
    epoch_selection = Path(args.epoch_selection or spec["epoch_selection"]).resolve()
    output_root = Path(args.output_root or spec["output_root"]).resolve()
    archive_root = Path(args.archive_output_root or spec["archive_root"]).resolve()
    dataset_slug = str(args.dataset_slug or spec["dataset_slug"])
    dataset_title = str(args.dataset_title or spec["dataset_title"])
    source_manifest = load_source_manifest(source_root)
    fixed_epoch = load_epoch_selection(epoch_selection, args.model)
    plan = {
        "source_flat_root": str(source_root),
        "output_root": str(output_root),
        "archive_root": str(archive_root),
        "dataset_slug": dataset_slug,
        "model": args.model,
        "fixed_epoch": fixed_epoch,
        "development_only": source_manifest["development_only"],
        "images": source_manifest["counts"]["images"],
        "labels": source_manifest["counts"]["labels"],
        "uploads_images": False,
        "includes_release_validation": False,
        "publish": args.publish,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if output_root.exists() or archive_root.exists():
        if not args.publish or not (output_root.exists() and archive_root.exists()):
            raise FileExistsError("配置输出已存在或不完整，拒绝覆盖")
        validate_existing_views(output_root, archive_root, args.model, fixed_epoch)
        print(f"[view] verified existing: {archive_root}")
    else:
        build_flat_view(
            source_root,
            final_trainer,
            epoch_selection,
            args.model,
            fixed_epoch,
            output_root,
            args.owner_slug,
            dataset_slug,
            dataset_title,
        )
        create_archive_view(output_root, archive_root, args.owner_slug, dataset_slug, dataset_title)
        print(f"[view] ready: {archive_root}")
    if args.publish:
        publish(archive_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
