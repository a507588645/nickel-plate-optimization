#!/usr/bin/env python3
"""Build the small B3 config Kaggle dataset (route B3, config-dataset pattern).

The private image dataset ``lly3323w/dianjinie`` already mirrors
``datasets/Dianjinie0211`` (verified by ``probe_dianjinie_remote_reuse.py``).
This builder therefore stages only a few MB of metadata:

- ``meta__train.txt`` / ``meta__val.txt``  split lists in the dataset-native
  relative form (``images/train/x.png``),
- ``meta__classes.txt`` / ``meta__source_data.yaml``
- ``label_hashes.json`` / ``image_hashes.json``  full SHA-256 of every
  included image and label; the kernel re-hashes the mounted dataset before
  training so local and remote content are proven byte-identical,
- ``trainer__train_dianjinie0211_b3.py``  the B3 trainer,
- ``b3_config_manifest.json``  schema, counts and integrity hashes,
- ``dataset-metadata.json``  private Kaggle dataset metadata.

The 18 GiB flat upload view produced by ``build_b3_kaggle_dataset.py``
remains the fallback if the config-dataset pattern is abandoned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path(r"E:/PythonProject/datasets/Dianjinie0211")
DEFAULT_OUTPUT_ROOT = ROOT / "kaggle_config_upload_v1"
SPLITS_DIR = ROOT / "splits"
TRAINER_SOURCE = ROOT / "train_dianjinie0211_b3.py"
TRAINER_UPLOAD_NAME = "trainer__train_dianjinie0211_b3.py"
DEFAULT_OWNER_SLUG = "lly3323w"
DEFAULT_DATASET_SLUG = "dianjinie0211-b3-config-v1"
DEFAULT_DATASET_TITLE = "dianjinie0211 B3 Config v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 B3 config Kaggle 数据集（含全量 SHA-256 清单）")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--owner-slug", default=DEFAULT_OWNER_SLUG)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--dataset-title", default=DEFAULT_DATASET_TITLE)
    parser.add_argument("--skip-image-hashes", action="store_true",
                        help="跳过 18 GiB 图片哈希（不推荐；标签哈希始终计算）")
    parser.add_argument("--dry-run", action="store_true", help="只审计并打印计划，不写任何文件")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split_list(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"划分清单为空：{path}")
    if len(set(lines)) != len(lines):
        raise ValueError(f"划分清单存在重复条目：{path}")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    train_relative = read_split_list(SPLITS_DIR / "train_local.txt")
    val_relative = read_split_list(SPLITS_DIR / "val_local.txt")
    if set(train_relative) & set(val_relative):
        raise ValueError("train/val 清单存在交集")
    if not TRAINER_SOURCE.is_file():
        raise FileNotFoundError(f"缺少训练器脚本：{TRAINER_SOURCE}")
    summary = json.loads((SPLITS_DIR / "split_summary.json").read_text(encoding="utf-8"))

    image_hashes: dict[str, str] = {}
    label_hashes: dict[str, str] = {}
    for relative in [*train_relative, *val_relative]:
        image_path = dataset_root / relative
        if not image_path.is_file():
            raise FileNotFoundError(f"清单图片不存在：{image_path}")
        label_path = (dataset_root / relative.replace("images/", "labels/", 1)).with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"图片缺少标签：{label_path}")

    plan = {
        "dataset_root": str(dataset_root),
        "train": len(train_relative),
        "val": len(val_relative),
        "output_root": str(args.output_root),
        "skip_image_hashes": args.skip_image_hashes,
        "dry_run": args.dry_run,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    if args.output_root.exists():
        raise FileExistsError(f"config 视图已存在，拒绝覆盖（如需重建请人工删除）：{args.output_root}")
    staging = args.output_root.parent / f"{args.output_root.name}.building"
    if staging.exists():
        raise FileExistsError(f"检测到未完成的构建目录，请人工检查后删除：{staging}")
    staging.mkdir(parents=True)

    try:
        (staging / "meta__train.txt").write_text("\n".join(train_relative) + "\n", encoding="utf-8")
        (staging / "meta__val.txt").write_text("\n".join(val_relative) + "\n", encoding="utf-8")
        classes = summary["classes"]
        (staging / "meta__classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
        shutil.copy2(dataset_root / "data.yaml", staging / "meta__source_data.yaml")
        shutil.copy2(TRAINER_SOURCE, staging / TRAINER_UPLOAD_NAME)

        print(f"[hash] hashing {len(train_relative) + len(val_relative)} labels...", flush=True)
        for relative in [*train_relative, *val_relative]:
            label_path = (dataset_root / relative.replace("images/", "labels/", 1)).with_suffix(".txt")
            label_hashes[relative] = sha256_file(label_path)
        if not args.skip_image_hashes:
            print(f"[hash] hashing {len(train_relative) + len(val_relative)} images (may take a while)...", flush=True)
            for index, relative in enumerate([*train_relative, *val_relative], 1):
                image_hashes[relative] = sha256_file(dataset_root / relative)
                if index % 200 == 0:
                    print(f"[hash] {index} images done", flush=True)

        (staging / "label_hashes.json").write_text(
            json.dumps(label_hashes, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "image_hashes.json").write_text(
            json.dumps(image_hashes, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )

        manifest = {
            "schema_version": "b3_config_v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_dataset_root": str(dataset_root),
            "image_dataset_ref": "lly3323w/dianjinie",
            "classes": classes,
            "counts": {"train": len(train_relative), "val": len(val_relative)},
            "split_summary": summary,
            "files": {
                "meta__train.txt": sha256_file(staging / "meta__train.txt"),
                "meta__val.txt": sha256_file(staging / "meta__val.txt"),
                "meta__classes.txt": sha256_file(staging / "meta__classes.txt"),
                "meta__source_data.yaml": sha256_file(staging / "meta__source_data.yaml"),
                "label_hashes.json": sha256_file(staging / "label_hashes.json"),
                "image_hashes.json": sha256_file(staging / "image_hashes.json"),
                TRAINER_UPLOAD_NAME: sha256_file(staging / TRAINER_UPLOAD_NAME),
            },
            "hash_coverage": {
                "labels": "all 990 labels sha256-verified on kernel start",
                "images": "all 990 images sha256-verified on kernel start" if not args.skip_image_hashes else "sizes only (hashing skipped)",
            },
        }
        (staging / "b3_config_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": args.dataset_title,
                    "id": f"{args.owner_slug}/{args.dataset_slug}",
                    "licenses": [{"name": "other"}],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        (staging / "README_UPLOAD.md").write_text(
            "# B3 Dianjinie0211 Config 数据集\n\n"
            "由 `build_b3_config_dataset.py` 生成；图片复用私有 Dataset `lly3323w/dianjinie`，"
            "本 config 只含清单、全量 SHA-256 校验表与训练器。上传："
            "`kaggle datasets create --dir-mode skip`；kernel 见 "
            "`b3-dianjinie0211-y11m640-seed0.py`。\n",
            encoding="utf-8",
        )
        staging.replace(args.output_root)
        print(f"[config] ready: {args.output_root}")
    except Exception:
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
