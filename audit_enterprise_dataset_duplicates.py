#!/usr/bin/env python3
"""审计企业硬链接数据集与原始数据集的完全重复图片。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用文件大小预筛选和 SHA-256 检查企业数据与原始数据的完全重复",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/enterprise_duplicate_audit"),
    )
    return parser.parse_args()


def images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_by_size(paths: Iterable[Path], allowed_sizes: set[int] | None = None) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    identity_cache: dict[tuple[int, int], str] = {}
    for path in paths:
        stat = path.stat()
        if allowed_sizes is not None and stat.st_size not in allowed_sizes:
            continue
        identity = (stat.st_dev, stat.st_ino)
        digest = identity_cache.get(identity)
        if digest is None:
            digest = sha256(path)
            identity_cache[identity] = digest
        result[digest].append(path)
    return result


def compare(
    enterprise_name: str,
    enterprise_root: Path,
    original_name: str,
    original_root: Path,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    enterprise_paths = images(enterprise_root)
    original_paths = images(original_root)
    enterprise_sizes = {p.stat().st_size for p in enterprise_paths}
    enterprise_hashes = hash_by_size(enterprise_paths)
    original_hashes = hash_by_size(original_paths, enterprise_sizes)
    shared = sorted(set(enterprise_hashes) & set(original_hashes))
    rows: list[dict[str, str]] = []
    enterprise_duplicate_paths: set[str] = set()
    for digest in shared:
        for enterprise_path in enterprise_hashes[digest]:
            enterprise_duplicate_paths.add(str(enterprise_path))
            for original_path in original_hashes[digest]:
                rows.append({
                    "enterprise_dataset": enterprise_name,
                    "enterprise_path": str(enterprise_path),
                    "original_dataset": original_name,
                    "original_path": str(original_path),
                    "sha256": digest,
                })
    summary = {
        "enterprise_dataset": enterprise_name,
        "enterprise_root": str(enterprise_root),
        "enterprise_images": len(enterprise_paths),
        "original_dataset": original_name,
        "original_root": str(original_root),
        "original_images": len(original_paths),
        "original_size_candidates": sum(
            1 for path in original_paths if path.stat().st_size in enterprise_sizes
        ),
        "shared_hashes": len(shared),
        "duplicate_enterprise_images": len(enterprise_duplicate_paths),
        "duplicate_path_pairs": len(rows),
    }
    return summary, rows


def internal_duplicates(name: str, root: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    paths = images(root)
    hashes = hash_by_size(paths)
    groups = {digest: members for digest, members in hashes.items() if len(members) > 1}
    rows = [
        {"dataset": name, "sha256": digest, "path": str(path)}
        for digest, members in groups.items()
        for path in members
    ]
    return {
        "dataset": name,
        "images": len(paths),
        "duplicate_groups": len(groups),
        "duplicate_images": len(rows),
    }, rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    enterprise = {
        "enterprise_dianjie_0306_detection": Path(r"E:/PythonProject/datasets/enterprise_dianjie_0306_detection_unlabeled"),
        "enterprise_dianji_0309_classification": Path(r"E:/PythonProject/datasets/enterprise_dianji_0309_classification"),
        "enterprise_dianji_0309_detection": Path(r"E:/PythonProject/datasets/enterprise_dianji_0309_detection_unlabeled"),
    }
    comparisons = [
        ("enterprise_dianjie_0306_detection", Path(r"E:/PythonProject/datasets/Dianjinie0211"), "Dianjinie0211"),
        ("enterprise_dianji_0309_classification", Path(r"E:/PythonProject/datasets/classify_4types_dianji"), "classify_4types_dianji"),
        ("enterprise_dianji_0309_classification", Path(r"E:/PythonProject/datasets/dianjienie"), "dianjienie"),
        ("enterprise_dianji_0309_detection", Path(r"E:/PythonProject/datasets/classify_4types_dianji"), "classify_4types_dianji"),
        ("enterprise_dianji_0309_detection", Path(r"E:/PythonProject/datasets/dianjienie"), "dianjienie"),
    ]

    summaries: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, str]] = []
    internal_summaries: list[dict[str, object]] = []
    internal_rows: list[dict[str, str]] = []
    for enterprise_name, original_root, original_name in comparisons:
        summary, rows = compare(enterprise_name, enterprise[enterprise_name], original_name, original_root)
        summaries.append(summary)
        duplicate_rows.extend(rows)
        print(
            f"{enterprise_name} vs {original_name}: "
            f"重复企业图片={summary['duplicate_enterprise_images']}，路径对={summary['duplicate_path_pairs']}"
        )
    for name, root in enterprise.items():
        summary, rows = internal_duplicates(name, root)
        internal_summaries.append(summary)
        internal_rows.extend(rows)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "exact_duplicate_pairs.csv", duplicate_rows)
    write_csv(args.output_root / "enterprise_internal_duplicates.csv", internal_rows)
    report = {
        "method": "file-size prefilter + SHA-256 exact-content comparison",
        "scope": "exact duplicates only; visually similar but recompressed/cropped images are not included",
        "comparisons": summaries,
        "enterprise_internal": internal_summaries,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"报告: {args.output_root}")


if __name__ == "__main__":
    main()
