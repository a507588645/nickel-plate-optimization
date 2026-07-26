#!/usr/bin/env python3
"""汇总多来源图片为无划分的扁平 YOLO 伪标签数据集输入。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将多个图片来源去重并硬链接为扁平 images/ 数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        nargs=2,
        action="append",
        metavar=("SOURCE_ID", "PATH"),
        required=True,
        help="可重复指定来源标识与图片根目录",
    )
    parser.add_argument("--task", choices=["dianjie", "dianji"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.rglob("*")):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {args.output_root}")

    candidates: list[tuple[str, Path]] = []
    for source_id, source_value in args.source:
        source_root = Path(source_value).resolve()
        if not source_root.exists():
            raise FileNotFoundError(source_root)
        for path in sorted(source_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                candidates.append((safe_id(source_id), path))

    by_identity: dict[tuple[int, int], str] = {}
    by_hash: dict[str, tuple[str, Path]] = {}
    records: list[dict[str, str]] = []
    for source_id, source in candidates:
        stat = source.stat()
        identity = (stat.st_dev, stat.st_ino)
        digest = by_identity.get(identity)
        if digest is None:
            digest = sha256(source)
            by_identity[identity] = digest
        if digest in by_hash:
            kept_source_id, kept_source = by_hash[digest]
            records.append({
                "source_id": source_id,
                "source_path": str(source),
                "sha256": digest,
                "status": "duplicate_skipped",
                "output_image": "",
                "duplicate_of": str(kept_source),
                "kept_source_id": kept_source_id,
            })
            continue
        by_hash[digest] = (source_id, source)
        output_name = f"{source_id}__{digest[:12]}__{source.name}"
        output = args.output_root / "images" / output_name
        if not args.dry_run:
            output.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, output)
            if not os.path.samefile(source, output):
                raise OSError(f"硬链接验证失败: {output}")
        records.append({
            "source_id": source_id,
            "source_path": str(source),
            "sha256": digest,
            "status": "linked" if not args.dry_run else "dry_run",
            "output_image": str(output),
            "duplicate_of": "",
            "kept_source_id": source_id,
        })

    if not args.dry_run:
        (args.output_root / "labels").mkdir(parents=True, exist_ok=True)
        with (args.output_root / "source_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        metadata = {
            "task": args.task,
            "layout": "flat_yolo_no_split",
            "sources": [{"source_id": source_id, "path": str(Path(path).resolve())} for source_id, path in args.source],
            "candidate_paths": len(candidates),
            "unique_images": len(by_hash),
            "duplicate_paths_skipped": len(candidates) - len(by_hash),
            "label_status": "pending",
        }
        (args.output_root / "dataset_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"候选图片路径: {len(candidates)}")
    print(f"唯一图片: {len(by_hash)}")
    print(f"跳过重复路径/内容: {len(candidates) - len(by_hash)}")
    print(f"输出: {args.output_root}")


if __name__ == "__main__":
    main()
