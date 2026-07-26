#!/usr/bin/env python3
"""
合并电积镍已核实等级分类集，去重后按 dup_group 分组分层 70/15/15 重划分，
并以硬链接组装为 datasets/dianjinie_grade_classification_v1 视图。

约束：
- 只读源数据集；禁止移动/覆盖原图
- 完整重复只留一份（优先 4types）；近重复不删，同 dup_group
- 不沿用 4types 原 train/valid（原 split 仅记入 manifest）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CLASSIFY = Path(r"E:/PythonProject/datasets/classify_4types_dianji")
DEFAULT_ENTERPRISE = Path(r"E:/PythonProject/datasets/enterprise_dianji_0309_classification")
DEFAULT_OUT = Path(r"E:/PythonProject/datasets/dianjinie_grade_classification_v1")
DEFAULT_SEED = 42
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
GRADES = ("3#", "9950", "9996", "bad")
SPLITS = ("train", "valid", "test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="构建电积镍等级分类硬链接视图 dianjinie_grade_classification_v1（去重+70/15/15）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--classify-root", type=Path, default=DEFAULT_CLASSIFY)
    p.add_argument("--enterprise-root", type=Path, default=DEFAULT_ENTERPRISE)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--valid-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument(
        "--phash-threshold",
        type=int,
        default=5,
        help="近重复 dHash hamming 阈值（含）；16x16 dHash 默认 5，过松会把异图并成巨组",
    )
    p.add_argument("--phash-size", type=int, default=16, help="dHash 边长，位数=size*size")
    p.add_argument("--dry-run", action="store_true", help="只扫描/划分，不创建硬链接")
    p.add_argument(
        "--force",
        action="store_true",
        help="若输出目录已存在则允许在其内覆盖清单（不删除源图）",
    )
    p.add_argument("--copy-fallback", action="store_true", default=True, help="硬链接失败时复制")
    return p.parse_args(argv)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def difference_hash_bits(path: Path, hash_size: int = 16) -> int | None:
    """
    差分哈希 dHash：对镍板类大图比 aHash 更稳，减少「全图发灰」导致的假近重复。
    输出位数 = hash_size * hash_size。
    """
    from PIL import Image

    try:
        with Image.open(path) as im:
            # width=hash_size+1, height=hash_size
            im = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            arr = list(im.getdata())
    except OSError:
        return None
    w = hash_size + 1
    value = 0
    bit = 0
    for y in range(hash_size):
        row = y * w
        for x in range(hash_size):
            left = arr[row + x]
            right = arr[row + x + 1]
            if left > right:
                value |= 1 << bit
            bit += 1
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def scan_classify(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        raise FileNotFoundError(root)
    for split_name in ("train", "valid", "val", "test"):
        split_dir = root / split_name
        if not split_dir.is_dir():
            continue
        source_split = "valid" if split_name == "val" else split_name
        for grade_dir in sorted(split_dir.iterdir()):
            if not grade_dir.is_dir() or grade_dir.name not in GRADES:
                continue
            grade = grade_dir.name
            for path in sorted(grade_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    rows.append(
                        {
                            "image_name_raw": path.name,
                            "image_stem": path.stem,
                            "image_path": str(path.resolve()),
                            "source_dataset": "classify_4types_dianji",
                            "source_split": source_split,
                            "grade": grade,
                        }
                    )
    return rows


def scan_enterprise(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    images_root = root / "images"
    if not images_root.exists():
        raise FileNotFoundError(images_root)
    for path in sorted(images_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        grade = path.parent.name
        if grade not in GRADES:
            continue
        rows.append(
            {
                "image_name_raw": path.name,
                "image_stem": path.stem,
                "image_path": str(path.resolve()),
                "source_dataset": "enterprise_dianji_0309_classification",
                "source_split": "enterprise_all",
                "grade": grade,
            }
        )
    return rows


def exact_dedup(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_hash: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        digest = sha256_file(Path(row["image_path"]))
        row["sha256"] = digest
        by_hash[digest].append(i)

    groups = {h: idxs for h, idxs in by_hash.items() if len(idxs) > 1}
    keep = [True] * len(rows)
    removed = 0
    for idxs in groups.values():
        ranked = sorted(
            idxs,
            key=lambda i: (
                0 if rows[i]["source_dataset"] == "classify_4types_dianji" else 1,
                0 if rows[i]["source_split"] == "train" else 1,
                rows[i]["image_path"],
            ),
        )
        for j in ranked[1:]:
            keep[j] = False
            removed += 1
    kept = [r for r, k in zip(rows, keep) if k]
    meta = {
        "raw_total": len(rows),
        "exact_duplicate_groups": len(groups),
        "exact_duplicate_images_involved": sum(len(v) for v in groups.values()),
        "exact_removed": removed,
        "kept_after_exact_dedup": len(kept),
    }
    return kept, meta


def near_dup_groups(
    rows: list[dict[str, Any]],
    phash_threshold: int,
    phash_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    近重复：dHash + 同等级才允许合并（跨等级相近外观不并组，避免污染分层标签）。
    若 dHash 仍过松导致单组过大，可降低 threshold。
    """
    phashes: list[int | None] = []
    for i, row in enumerate(rows):
        ph = difference_hash_bits(Path(row["image_path"]), hash_size=phash_size)
        phashes.append(ph)
        # 64 hex 足够装 256 bit
        row["phash"] = f"{ph:064x}" if ph is not None else ""
        if (i + 1) % 200 == 0:
            print(f"  dhash {i + 1}/{len(rows)}")

    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    near_pairs = 0
    # 仅同 grade 内比较，显著降复杂度并避免跨等级误并
    by_grade_idx: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if phashes[i] is not None:
            by_grade_idx[row["grade"]].append(i)

    for grade, idxs in by_grade_idx.items():
        for a in range(len(idxs)):
            i = idxs[a]
            for b in range(a + 1, len(idxs)):
                j = idxs[b]
                if hamming(phashes[i], phashes[j]) <= phash_threshold:  # type: ignore[arg-type]
                    union(i, j)
                    near_pairs += 1

    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        comps[find(i)].append(i)

    multi = {k: v for k, v in comps.items() if len(v) > 1}
    gid = 0
    for root_id, members in sorted(comps.items(), key=lambda kv: min(kv[1])):
        if len(members) > 1:
            group_id = f"near_g{gid:04d}"
            gid += 1
        else:
            group_id = f"single_{members[0]:04d}"
        for m in members:
            rows[m]["dup_group_id"] = group_id

    grade_conflicts = 0
    by_gid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_gid[r["dup_group_id"]].append(r)
    for g, members in by_gid.items():
        grades = [m["grade"] for m in members]
        maj = Counter(grades).most_common(1)[0][0]
        if len(set(grades)) > 1:
            grade_conflicts += 1
        for m in members:
            m["group_grade"] = maj

    multi_sizes = sorted((len(v) for v in multi.values()), reverse=True)
    meta = {
        "hash_method": "dHash",
        "phash_size": phash_size,
        "phash_threshold": phash_threshold,
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_groups": len(multi),
        "near_duplicate_images_in_multi_groups": sum(len(v) for v in multi.values()),
        "largest_near_group_sizes_top5": multi_sizes[:5],
        "grade_conflicts_in_near_groups": grade_conflicts,
        "unique_dup_groups": len(by_gid),
        "same_grade_only_union": True,
    }
    return rows, meta


def stratified_group_split(
    rows: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    以 dup_group 为单位划分，按 group_grade 分层。
    分配时按组内图片数做贪心配额，使各 split 的图片数逼近 70/15/15，
    避免「大组进 valid」导致比例倒置。
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.default_rng(seed)

    by_gid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_gid[r["dup_group_id"]].append(r)

    groups: list[dict[str, Any]] = []
    for gid, members in by_gid.items():
        groups.append(
            {
                "dup_group_id": gid,
                "group_grade": members[0]["group_grade"],
                "n_images": len(members),
            }
        )

    assigned: dict[str, str] = {}
    split_stats: dict[str, Counter[str]] = {s: Counter() for s in SPLITS}
    split_image_totals: dict[str, int] = {s: 0 for s in SPLITS}

    by_grade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in groups:
        by_grade[g["group_grade"]].append(g)

    for grade, glist in by_grade.items():
        order = list(glist)
        rng.shuffle(order)
        # 大组优先放置，便于配额控制
        order.sort(key=lambda g: g["n_images"], reverse=True)
        total_imgs = sum(g["n_images"] for g in order)
        target = {
            "train": total_imgs * train_ratio,
            "valid": total_imgs * valid_ratio,
            "test": total_imgs * test_ratio,
        }
        filled = {"train": 0.0, "valid": 0.0, "test": 0.0}
        # 少数类：尽量每个 split 至少 1 组（若组数够）
        n_groups = len(order)
        must_cover = n_groups >= 3

        for gi, g in enumerate(order):
            n_img = g["n_images"]
            # 计算放入各 split 后相对目标的缺口
            candidates = []
            for sp in SPLITS:
                # 若该 split 还没有任何该等级组，且仍有未分配组，优先覆盖
                prefer = 0.0
                if must_cover and split_stats[sp][grade] == 0 and gi < n_groups:
                    prefer = -1e9  # 强烈优先空 split
                # 缺口：target - filled；选缺口最大的
                gap = target[sp] - filled[sp]
                # 轻微噪声防全塞 train
                noise = float(rng.random() * 1e-3)
                candidates.append((prefer - gap + noise, sp))
            candidates.sort()
            # 若 test/valid 目标已满且 train 未满，偏向 train
            chosen = candidates[0][1]
            # 防止 test/valid 过度：若 filled 已超 target*1.35 且 train 仍缺，改 train
            if filled[chosen] >= target[chosen] * 1.35 and filled["train"] < target["train"]:
                chosen = "train"

            assigned[g["dup_group_id"]] = chosen
            filled[chosen] += n_img
            split_stats[chosen][grade] += n_img
            split_image_totals[chosen] += n_img

    for r in rows:
        r["split"] = assigned[r["dup_group_id"]]

    cross = 0
    for gid, members in by_gid.items():
        if len({m["split"] for m in members}) > 1:
            cross += 1

    total = len(rows)
    meta = {
        "seed": seed,
        "ratios": {"train": train_ratio, "valid": valid_ratio, "test": test_ratio},
        "n_groups": len(groups),
        "images_by_split_grade": {s: dict(c) for s, c in split_stats.items()},
        "images_by_split": {s: int(split_image_totals[s]) for s in SPLITS},
        "image_ratio_by_split": {
            s: (split_image_totals[s] / total if total else 0.0) for s in SPLITS
        },
        "cross_split_groups": cross,
        "note": "dup_group 整组同一 split；按组大小贪心逼近 70/15/15；不沿用 4types 原划分",
    }
    return rows, meta


def unique_link_name(row: dict[str, Any], used: set[str]) -> str:
    """避免不同来源同名冲突。"""
    stem = row["image_stem"]
    suffix = Path(row["image_name_raw"]).suffix.lower() or ".png"
    if row["source_dataset"] == "enterprise_dianji_0309_classification":
        base = f"ent0309_{row['grade']}_{stem}{suffix}"
    else:
        base = f"c4_{row['source_split']}_{row['grade']}_{stem}{suffix}"
    # 清理 Windows 非法字符
    base = base.replace("#", "sharp").replace("/", "_").replace("\\", "_")
    # 还原 3# 显示需求：目录名仍用 3#，文件名用 sharp 更安全；manifest 保留 grade=3#
    name = base
    if name in used:
        h = row.get("sha256", "")[:8]
        name = f"{Path(base).stem}_{h}{suffix}"
    used.add(name)
    return name


def hardlink_or_copy(src: Path, dst: Path, allow_copy: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return "existing_hardlink"
        except OSError:
            pass
        raise FileExistsError(f"目标已存在: {dst}")
    try:
        if src.drive.lower() != dst.drive.lower():
            raise OSError("cross-volume")
        os.link(str(src), str(dst))
        if not os.path.samefile(src, dst):
            raise OSError("hardlink verify failed")
        return "hardlink"
    except OSError as exc:
        if not allow_copy:
            raise
        shutil.copy2(src, dst)
        return f"copy_fallback:{exc.__class__.__name__}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys = fields or list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    link_mode = summary.get("link_mode_counts", {})
    lines = [
        "# dianjinie_grade_classification_v1",
        "",
        "> **状态：未经企业书面标准最终确认。** 标签来自用户口头确认的目录真值（4types + 0309），",
        "> 模型/规则产物不得直接当作企业判级标准。",
        "",
        "## 来源",
        "",
        f"- `classify_4types_dianji`：{summary['source_counts_raw'].get('classify_4types_dianji', 0)} 张",
        f"- `enterprise_dianji_0309_classification`：{summary['source_counts_raw'].get('enterprise_dianji_0309_classification', 0)} 张",
        "",
        "## 去重",
        "",
        f"- 完整重复组：{summary['dedup']['exact_duplicate_groups']}（移除 {summary['dedup']['exact_removed']}）",
        f"- 近重复多成员组：{summary['near']['near_duplicate_groups']}（phash thr≤{summary['near']['phash_threshold']}）",
        f"- 去重后图片：{summary['kept_images']}",
        "",
        "## 划分",
        "",
        f"- 按 `dup_group_id` 整组划分 + 按等级分层，seed={summary['split']['seed']}",
        f"- 目标比例 train/valid/test = {summary['split']['ratios']['train']}/{summary['split']['ratios']['valid']}/{summary['split']['ratios']['test']}",
        f"- **不沿用** 4types 原 train/valid；原 split 记在 `manifests/samples.csv` 的 `source_split`",
        f"- test 仅用于最终确认，不参与调参",
        "",
        "## 链接方式",
        "",
        f"- 统计：{link_mode}",
        "- 同盘优先 `os.link` 硬链接；失败则复制并在本 README 注明",
        "",
        "## 目录结构",
        "",
        "```",
        "images/{train,valid,test}/{3#,9950,9996,bad}/",
        "manifests/samples.csv",
        "splits/{train,valid,test}.txt",
        "dataset_summary.json",
        "README.md",
        "```",
        "",
        f"- 构建时间：{summary.get('created_at', '')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def clear_view_images(out: Path) -> None:
    """仅清理视图内 images/ 硬链接或复制文件，不触碰源数据集。"""
    images = out / "images"
    if not images.exists():
        return
    for p in images.rglob("*"):
        if p.is_file():
            p.unlink()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output_root
    print("=== 构建 dianjinie_grade_classification_v1 ===")
    print(f"输出: {out}")
    if abs(args.train_ratio + args.valid_ratio + args.test_ratio - 1.0) > 1e-6:
        print("错误：train+valid+test 比例之和必须为 1", file=sys.stderr)
        return 2

    if out.exists() and (out / "images").exists() and any((out / "images").rglob("*")):
        if not args.force and not args.dry_run:
            print("输出目录已有 images，拒绝覆盖。使用 --force 重建。", file=sys.stderr)
            return 3
        if args.force and not args.dry_run:
            print("[force] 清理旧 images 硬链接…")
            clear_view_images(out)

    classify_rows = scan_classify(args.classify_root)
    enterprise_rows = scan_enterprise(args.enterprise_root)
    raw = classify_rows + enterprise_rows
    print(f"[扫描] 4types={len(classify_rows)} 0309={len(enterprise_rows)} raw={len(raw)}")

    kept, exact_meta = exact_dedup(raw)
    print(
        f"[完整重复] 组={exact_meta['exact_duplicate_groups']} "
        f"移除={exact_meta['exact_removed']} 保留={exact_meta['kept_after_exact_dedup']}"
    )
    kept, near_meta = near_dup_groups(kept, args.phash_threshold, args.phash_size)
    print(
        f"[近重复] 组={near_meta['near_duplicate_groups']} "
        f"对={near_meta['near_duplicate_pairs']} 冲突等级组={near_meta['grade_conflicts_in_near_groups']}"
    )
    kept, split_meta = stratified_group_split(
        kept, args.seed, args.train_ratio, args.valid_ratio, args.test_ratio
    )
    print(f"[划分] {split_meta['images_by_split']} cross_split_groups={split_meta['cross_split_groups']}")

    # 分配链接文件名
    used_names: set[str] = set()
    for r in kept:
        r["image_name"] = unique_link_name(r, used_names)
        r["link_relpath"] = f"images/{r['split']}/{r['grade']}/{r['image_name']}"

    link_counts: Counter[str] = Counter()
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            for grade in GRADES:
                (out / "images" / split / grade).mkdir(parents=True, exist_ok=True)
        for r in kept:
            src = Path(r["image_path"])
            dst = out / r["link_relpath"]
            mode = hardlink_or_copy(src, dst, allow_copy=args.copy_fallback)
            r["link_mode"] = mode
            link_counts[mode] += 1
            if sum(link_counts.values()) % 200 == 0:
                print(f"  linked {sum(link_counts.values())}/{len(kept)}")
    else:
        for r in kept:
            r["link_mode"] = "dry_run"
            link_counts["dry_run"] += 1

    # 写 manifest / splits
    manifest_fields = [
        "image_name",
        "image_stem",
        "image_name_raw",
        "source_dataset",
        "source_split",
        "grade",
        "group_grade",
        "split",
        "dup_group_id",
        "sha256",
        "phash",
        "link_relpath",
        "link_mode",
        "image_path",
    ]
    if not args.dry_run:
        write_csv(out / "manifests" / "samples.csv", kept, manifest_fields)
        for split in SPLITS:
            lines = [r["link_relpath"] for r in kept if r["split"] == split]
            (out / "splits").mkdir(parents=True, exist_ok=True)
            (out / "splits" / f"{split}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

    grade_counts = dict(Counter(r["grade"] for r in kept))
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "unconfirmed_pending_enterprise",
        "output_root": str(out.resolve()),
        "source_counts_raw": {
            "classify_4types_dianji": len(classify_rows),
            "enterprise_dianji_0309_classification": len(enterprise_rows),
        },
        "dedup": exact_meta,
        "near": near_meta,
        "split": split_meta,
        "kept_images": len(kept),
        "grade_counts": grade_counts,
        "link_mode_counts": dict(link_counts),
        "small_sample_warning": {
            "3#": grade_counts.get("3#", 0),
            "bad": grade_counts.get("bad", 0),
            "message": "3# 与 bad 为小样本，指标方差大",
        },
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        (out / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_readme(out / "README.md", summary)
        # 也把一份摘要拷到优化目录旁路索引用（不强制）
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("完成" if not args.dry_run else "dry-run 完成（未写硬链接）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
