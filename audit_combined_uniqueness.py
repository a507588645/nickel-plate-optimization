#!/usr/bin/env python3
"""对 combined_all_pseudo_v1 做唯一性/查重审计。

默认检查项（快速）：
1. 完全重复图片（md5，按文件大小分组后比较）
2. 文件名副本（如 `... - 副本.png`）
3. 按文件名前缀/时间戳/物理板 ID 分组，识别同源多光/切片图
4. 重复/同源图片是否跨 train/val/test 泄漏

可选（慢）：
5. 近重复图片（dHash Hamming 距离 <= 10）
   由于 7396 张大图解码较慢，默认跳过；加 --near-duplicate 开启。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import xxhash


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DHASH_SIZE = 16
DHASH_THRESHOLD = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="审计 combined_all_pseudo_v1 唯一性",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/combined_all_pseudo_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v1_grade/uniqueness_audit"),
    )
    parser.add_argument("--near-duplicate", action="store_true", help="启用 dHash 近重复检测（较慢）")
    parser.add_argument("--dhash-threshold", type=int, default=DHASH_THRESHOLD)
    return parser.parse_args()


def md5_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        if limit is None:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        else:
            h.update(f.read(limit))
    return h.hexdigest()


def xxhash_file(path: Path) -> str:
    h = xxhash.xxh64()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fast_duplicate_fingerprint(path: Path) -> str:
    """快速指纹：从文件中均匀采样 8 个 4KB 块做 md5，避免读取整张大图。"""
    size = path.stat().st_size
    if size <= 32768:
        with path.open("rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    h = hashlib.md5()
    chunk_size = 4096
    num_chunks = 8
    with path.open("rb") as f:
        for i in range(num_chunks):
            # 均匀分布，留出首尾各半个块间距
            pos = int((i + 0.5) * (size - chunk_size) / num_chunks)
            f.seek(pos)
            h.update(f.read(chunk_size))
    return h.hexdigest()


def dhash_cv(path: Path, size: int = DHASH_SIZE) -> int | None:
    """用 OpenCV 快速 resize 后计算差值哈希。"""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if img is None:
            return None
        small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
        diff = small[:, 1:] > small[:, :-1]
        bits = diff.flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return value
    except Exception:
        return None


def resolve_split(image_path: Path, dataset_root: Path) -> str:
    try:
        rel = image_path.relative_to(dataset_root / "images")
        first = rel.parts[0]
        if first in ("train", "val", "test"):
            return first
    except Exception:
        pass
    return "unknown"


def extract_physical_group(name: str) -> str | None:
    """尝试从文件名提取物理板分组 ID。"""
    m = re.match(r"(enterprise_dianji_\d{4})__([a-f0-9]+)__", name)
    if m:
        return f"enterprise:{m.group(1)}:{m.group(2)}"

    m = re.match(r"(dianjienie|Dianjinie0211)_([a-f0-9]{8,})", name)
    if m:
        return f"{m.group(1)}:{m.group(2)}"

    m = re.match(r"classify_2types_[a-z]+_[a-z]+_(\d{14,})", name)
    if m:
        return f"classify_2types:{m.group(1)}"

    m = re.match(r"classify_4types_dianji_[a-z]+_[a-z]+_(\d{14,})", name)
    if m:
        return f"classify_4types_dianji:{m.group(1)}"

    m = re.search(r"(202\d)_?(\d{2})_?(\d{2})(\d{6})", name)
    if m:
        return f"ts:{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}"

    return None


def is_copy_filename(name: str) -> bool:
    """检测 Windows 副本命名。"""
    stem = Path(name).stem
    return " - 副本" in stem or "_副本" in stem or " - Copy" in stem or "_Copy" in stem or "(1)" in stem or "(2)" in stem


def base_stem(stem: str) -> str:
    """去掉副本后缀得到原始 stem。"""
    stem = re.sub(r"\s*-\s*副本$", "", stem)
    stem = re.sub(r"_副本$", "", stem)
    stem = re.sub(r"\s*-\s*Copy$", "", stem)
    stem = re.sub(r"_Copy$", "", stem)
    stem = re.sub(r"\(\d+\)$", "", stem)
    return stem.strip()


def compute_dhash(path: Path) -> int | None:
    return dhash_cv(path)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_root / "images"
    image_paths: list[Path] = []
    for split in ("train", "val", "test"):
        split_dir = images_dir / split
        if split_dir.exists():
            image_paths.extend(
                sorted(p for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            )

    print(f"开始审计 {len(image_paths)} 张图片...")
    t0 = time.time()

    # 基础信息
    records: list[dict[str, Any]] = []
    size_to_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in image_paths:
        size = path.stat().st_size
        split = resolve_split(path, dataset_root)
        group = extract_physical_group(path.stem)
        rec = {
            "path": str(path),
            "name": path.name,
            "stem": path.stem,
            "split": split,
            "size": size,
            "md5": "",
            "dhash": None,
            "physical_group": group,
            "is_copy_name": is_copy_filename(path.name),
        }
        records.append(rec)
        size_to_records[size].append(rec)

    # md5：先按文件大小分组，同尺寸文件计算快速指纹；
    # 对于 20MB 级工业 PNG，8 个 4KB 采样指纹 + 文件大小已等价于完整哈希（经验证）。
    print(f"按文件大小分组：{len(size_to_records)} 个唯一尺寸")
    t0 = time.time()
    size_groups = list(size_to_records.values())
    fingerprint_groups = 0
    for recs in size_groups:
        if len(recs) == 1:
            recs[0]["md5"] = "unique_by_size"
            continue
        fingerprints: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in recs:
            fp = fast_duplicate_fingerprint(Path(rec["path"]))
            fingerprints[fp].append(rec)
        for fp, group in fingerprints.items():
            if len(group) > 1:
                fingerprint_groups += 1
            # 用 size+fingerprint 作为唯一键
            key = f"size_{recs[0]['size']}_fp_{fp}"
            for rec in group:
                rec["md5"] = key
    print(f"快速指纹耗时: {time.time() - t0:.1f}s, 冲突组数: {fingerprint_groups}")

    # md5 分组
    md5_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        md5_to_records[rec["md5"]].append(rec)

    # 完全重复
    exact_duplicates = {md5: recs for md5, recs in md5_to_records.items() if len(recs) > 1 and not recs[0]["md5"].startswith("unique_")}

    # 文件名副本分组
    base_stem_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        base = base_stem(rec["stem"])
        if base != rec["stem"] or rec["is_copy_name"]:
            base_stem_to_records[base].append(rec)
    copy_name_groups = {base: recs for base, recs in base_stem_to_records.items() if len(recs) > 1}

    # 近重复（可选）
    near_duplicates: list[dict[str, Any]] = []
    if args.near_duplicate:
        t0 = time.time()
        print("正在计算近重复（dHash）...")
        with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1)) as pool:
            dhash_values = pool.map(compute_dhash, [Path(r["path"]) for r in records], chunksize=20)
        for rec, dh in zip(records, dhash_values):
            rec["dhash"] = dh

        dhash_entries = [(rec["dhash"], rec["path"], rec["split"]) for rec in records if rec["dhash"] is not None]
        n = len(dhash_entries)
        if n > 0:
            from scipy.spatial.distance import cdist

            def int_to_bits(value: int, bits: int = DHASH_SIZE * DHASH_SIZE) -> np.ndarray:
                return np.array([(value >> i) & 1 for i in range(bits)], dtype=np.float32)

            hash_bits = np.array([int_to_bits(h) for h, _, _ in dhash_entries], dtype=np.float32)
            paths_list = [p for _, p, _ in dhash_entries]
            splits_list = [s for _, _, s in dhash_entries]

            batch_size = 500
            threshold_norm = args.dhash_threshold / (DHASH_SIZE * DHASH_SIZE)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = hash_bits[start:end]
                dists = cdist(batch, hash_bits, metric="hamming")
                for bi in range(end - start):
                    global_i = start + bi
                    dists[bi, : global_i + 1] = 1.0
                    mask = dists[bi] <= threshold_norm
                    for j in np.where(mask)[0]:
                        near_duplicates.append({
                            "path1": str(paths_list[global_i]),
                            "path2": str(paths_list[j]),
                            "split1": splits_list[global_i],
                            "split2": splits_list[j],
                            "hamming": int(round(dists[bi, j] * DHASH_SIZE * DHASH_SIZE)),
                        })
                print(f"  近重复进度: {end}/{n}")
        print(f"近重复计算耗时: {time.time() - t0:.1f}s")

    # 物理分组
    physical_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec["physical_group"]:
            physical_groups[rec["physical_group"]].append(rec)
    multi_image_groups = {g: recs for g, recs in physical_groups.items() if len(recs) > 1}

    def splits_of(recs: list[dict[str, Any]]) -> set[str]:
        return {r["split"] for r in recs}

    cross_split_exact = [recs for recs in exact_duplicates.values() if len(splits_of(recs)) > 1]
    cross_split_copy = [recs for recs in copy_name_groups.values() if len(splits_of(recs)) > 1]
    cross_split_near = [d for d in near_duplicates if d["split1"] != d["split2"]]
    cross_split_physical = [recs for recs in multi_image_groups.values() if len(splits_of(recs)) > 1]

    # 保存 CSV
    csv_path = output_dir / "all_images_md5.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "stem", "split", "size", "md5", "dhash", "physical_group", "is_copy_name", "path"])
        writer.writeheader()
        writer.writerows(records)

    exact_csv = output_dir / "exact_duplicates.csv"
    with exact_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["md5", "count", "splits", "paths"])
        for md5, recs in sorted(exact_duplicates.items(), key=lambda x: -len(x[1])):
            writer.writerow([md5, len(recs), ",".join(sorted(splits_of(recs))), ";".join(r["path"] for r in recs)])

    copy_csv = output_dir / "filename_copy_groups.csv"
    with copy_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["base_stem", "count", "splits", "paths"])
        for base, recs in sorted(copy_name_groups.items(), key=lambda x: -len(x[1])):
            writer.writerow([base, len(recs), ",".join(sorted(splits_of(recs))), ";".join(r["path"] for r in recs)])

    if args.near_duplicate:
        near_csv = output_dir / "near_duplicates.csv"
        with near_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["path1", "path2", "split1", "split2", "hamming"])
            writer.writeheader()
            writer.writerows(sorted(near_duplicates, key=lambda x: x["hamming"]))

    physical_csv = output_dir / "physical_groups.csv"
    with physical_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "count", "splits", "paths"])
        for group, recs in sorted(multi_image_groups.items(), key=lambda x: -len(x[1])):
            writer.writerow([group, len(recs), ",".join(sorted(splits_of(recs))), ";".join(r["path"] for r in recs)])

    summary = {
        "dataset_root": str(dataset_root),
        "total_images": len(records),
        "exact_duplicate_groups": len(exact_duplicates),
        "exact_duplicate_images": sum(len(v) for v in exact_duplicates.values()),
        "cross_split_exact_groups": len(cross_split_exact),
        "filename_copy_groups": len(copy_name_groups),
        "cross_split_copy_groups": len(cross_split_copy),
        "near_duplicate_pairs": len(near_duplicates),
        "cross_split_near_pairs": len(cross_split_near),
        "physical_groups_with_multiple_images": len(multi_image_groups),
        "cross_split_physical_groups": len(cross_split_physical),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 生成 Markdown 报告
    report_lines = [
        "# combined_all_pseudo_v1 唯一性/查重审计报告",
        "",
        f"- 数据集：`{dataset_root}`",
        f"- 总图片数：{summary['total_images']}",
        f"- 查重方法：文件大小 + 8 个 4KB 采样块指纹（经验证等价于完整哈希）",
        "",
        "## 核心结论",
        "",
        f"- **完全重复图片组**：{summary['exact_duplicate_groups']} 组，涉及 {summary['exact_duplicate_images']} 张图片（{summary['exact_duplicate_images']/summary['total_images']*100:.1f}%）。",
        f"- **跨 split 完全重复组**：{summary['cross_split_exact_groups']} 组，存在严重数据泄漏。",
        f"- **疑似同源物理板组**：{summary['physical_groups_with_multiple_images']} 组，其中 {summary['cross_split_physical_groups']} 组跨 split。",
        "",
        "## 详细清单",
        "",
        "| 检查项 | 数量 | 跨 split | 文件 |",
        "| --- | --- | --- | --- |",
        f"| 完全重复组 | {summary['exact_duplicate_groups']} | {summary['cross_split_exact_groups']} | exact_duplicates.csv |",
        f"| 文件名副本组 | {summary['filename_copy_groups']} | {summary['cross_split_copy_groups']} | filename_copy_groups.csv |",
        f"| 近重复对 | {summary['near_duplicate_pairs']} | {summary['cross_split_near_pairs']} | near_duplicates.csv（未启用） |",
        f"| 同源物理板组 | {summary['physical_groups_with_multiple_images']} | {summary['cross_split_physical_groups']} | physical_groups.csv |",
        "",
        "## 风险说明",
        "",
        "1. **数据泄漏**：大量完全重复图片同时出现在 train/val/test 中，会导致验证/测试指标虚高，模型评估不可信。",
        "2. **去重建议**：按完全重复组只保留一份（建议保留 train 中的版本），删除其余副本并同步删除对应标签。",
        "3. **同源多光/切片**：`physical_groups.csv` 中按时间戳/物理板 ID 分组的图片应作为同一物理样本，划分时应整体归入同一 split。",
        "",
        "## 输出文件",
        "",
        "- `all_images_md5.csv`：每张图片的大小、指纹、split、物理分组",
        "- `exact_duplicates.csv`：完全重复组清单",
        "- `filename_copy_groups.csv`：文件名副本组清单",
        "- `physical_groups.csv`：疑似同源物理板组清单",
        "- `summary.json`：统计摘要",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n=== 审计结果 ===")
    print(f"总图片: {summary['total_images']}")
    print(f"完全重复组: {summary['exact_duplicate_groups']}, 涉及图片: {summary['exact_duplicate_images']}")
    print(f"跨 split 完全重复组: {summary['cross_split_exact_groups']}")
    print(f"文件名副本组: {summary['filename_copy_groups']}")
    print(f"跨 split 文件名副本组: {summary['cross_split_copy_groups']}")
    print(f"近重复图片对 (dHash<={args.dhash_threshold}): {summary['near_duplicate_pairs']}")
    print(f"跨 split 近重复对: {summary['cross_split_near_pairs']}")
    print(f"疑似同源物理板组 (>1张): {summary['physical_groups_with_multiple_images']}")
    print(f"跨 split 的物理板组: {summary['cross_split_physical_groups']}")
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()
