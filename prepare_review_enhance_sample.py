#!/usr/bin/env python3
"""为伪标签审查准备「显示增强」分层抽样集（不重跑模型）。

流程：
1. 从 labels_with_conf 中筛选含低置信度框的图片。
2. 按 亮度档 × 低conf档 × 框面积档 分层抽样。
3. 生成 images_raw / images_enhanced / images_strong（几何不变）。
4. 复制同一套 labels_with_conf（框不因增强改变）。
5. 生成多视图 review.html + sample_manifest.csv + 空评分表。

输出目录默认：E:\\PythonProject\\优化\\review_enhance_samples\\<name>
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from review_enhance_utils import (
    DEFAULT_VIEW_KEYS,
    IMAGE_SUFFIXES,
    VIEW_CATALOG,
    area_bucket,
    brightness_bucket,
    conf_bucket,
    estimate_mean_brightness,
    low_conf_boxes,
    process_one_image,
    read_boxes,
    stratified_sample_indices,
)


DEFAULT_DATASETS = {
    "dianjie": Path(
        r"E:/PythonProject/优化/pseudo_label_datasets/"
        r"dianjienie_20260713_131840_v1_Dianjinie0211+classify_2types+enterprise_dianjie_0306"
    ),
    "dianji": Path(
        r"E:/PythonProject/优化/pseudo_label_datasets/"
        r"dianjinie_20260713_131840_v1_dianjienie+classify_4types_dianji+enterprise_dianji_0309"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分层抽样低置信度难审图，生成原图/增强/强对比多视图审查集（不重推理）"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="源伪标签数据集根目录；也可用 --preset",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(DEFAULT_DATASETS.keys()),
        default="dianjie",
        help="快捷选择电解镍/电积镍当前汇总伪标签集",
    )
    parser.add_argument("--n-images", type=int, default=80, help="抽样图片数，建议 50-100")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--low-conf-threshold",
        type=float,
        default=0.40,
        help="低置信度阈值，默认 <0.40",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/review_enhance_samples"),
    )
    parser.add_argument(
        "--review-output-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/pseudo_label_reviews"),
    )
    parser.add_argument(
        "--name",
        default="",
        help="输出子目录名；默认自动带时间戳",
    )
    parser.add_argument(
        "--skip-review-html",
        action="store_true",
        help="只生成数据，不调用 generate_flat_yolo_review.py",
    )
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def build_label_candidates(
    images: list[Path],
    labels_dir: Path,
    low_thresh: float,
) -> list[dict]:
    """First pass: labels only (fast). Stratum without brightness."""
    rows: list[dict] = []
    total = len(images)
    for idx, img in enumerate(images, 1):
        boxes = read_boxes(labels_dir / f"{img.stem}.txt")
        lows = low_conf_boxes(boxes, low_thresh)
        if not lows:
            continue
        min_conf = min(b.confidence for b in lows if b.confidence is not None)
        mean_area = sum(b.area for b in lows) / len(lows)
        c_bucket = conf_bucket(min_conf)
        a_bucket = area_bucket(mean_area)
        rows.append(
            {
                "path": img,
                "name": img.name,
                "stem": img.stem,
                "mean_brightness": None,
                "brightness_bucket": "unknown",
                "conf_bucket": c_bucket,
                "area_bucket": a_bucket,
                "stratum": f"unknown|{c_bucket}|{a_bucket}",
                "label_stratum": f"{c_bucket}|{a_bucket}",
                "low_box_count": len(lows),
                "total_box_count": len(boxes),
                "min_low_conf": round(min_conf, 4),
                "mean_low_area": round(mean_area, 6),
            }
        )
        if idx % 500 == 0 or idx == total:
            print(f"  label-scan {idx}/{total}, low-conf images={len(rows)}", flush=True)
    return rows


def attach_brightness(rows: list[dict]) -> None:
    total = len(rows)
    for idx, row in enumerate(rows, 1):
        luma = estimate_mean_brightness(row["path"])
        if luma is None:
            print(f"[warn] unreadable image, keep unknown brightness: {row['path']}", flush=True)
            continue
        row["mean_brightness"] = round(luma, 2)
        row["brightness_bucket"] = brightness_bucket(luma)
        row["stratum"] = f"{row['brightness_bucket']}|{row['conf_bucket']}|{row['area_bucket']}"
        if idx % 20 == 0 or idx == total:
            print(f"  brightness {idx}/{total}", flush=True)


def write_score_template(path: Path) -> None:
    headers = [
        "image_name",
        "stratum",
        "raw_visible",
        "enhanced_visible",
        "strong_visible",
        "final_decision",
        "false_texture",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        # empty template; user fills after review, or export JSONL from HTML


def main() -> None:
    args = parse_args()
    dataset_root = (args.dataset_root or DEFAULT_DATASETS[args.preset]).resolve()
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels_with_conf"
    classes_file = dataset_root / "classes.txt"

    if not images_dir.is_dir():
        raise NotADirectoryError(f"images not found: {images_dir}")
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"labels_with_conf not found: {labels_dir}")

    print(f"[1/5] 扫描低置信度图片: {dataset_root}", flush=True)
    all_images = list_images(images_dir)
    print(f"  total images: {len(all_images)}", flush=True)
    candidates = build_label_candidates(all_images, labels_dir, args.low_conf_threshold)
    if not candidates:
        raise RuntimeError("未找到含低置信度框的图片")
    if args.n_images > len(candidates):
        print(
            f"[warn] 请求 {args.n_images} 张，仅有 {len(candidates)} 张含低 conf 框，将全部使用",
            flush=True,
        )
        args.n_images = len(candidates)

    # Two-pass: oversample by label strata, then attach brightness and re-stratify.
    oversample_n = min(len(candidates), max(args.n_images * 3, args.n_images))
    label_strata = [c["label_stratum"] for c in candidates]
    over_idx = stratified_sample_indices(label_strata, oversample_n, args.seed)
    oversample = [candidates[i] for i in over_idx]
    print(
        f"  oversample {len(oversample)} for brightness (target final {args.n_images})",
        flush=True,
    )
    attach_brightness(oversample)

    final_strata = [c["stratum"] for c in oversample]
    selected_idx = stratified_sample_indices(final_strata, args.n_images, args.seed + 1)
    selected = [oversample[i] for i in selected_idx]
    selected.sort(key=lambda x: x["name"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"{args.preset if args.dataset_root is None else dataset_root.name}_n{args.n_images}_{stamp}"
    out_root = args.output_root / name
    if out_root.exists():
        raise FileExistsError(f"输出目录已存在，避免覆盖: {out_root}")

    # also keep images/ as raw copy for tools expecting images/
    images_alias = out_root / "images"
    labels_out = out_root / "labels_with_conf"
    labels_out.mkdir(parents=True)
    images_alias.mkdir(parents=True)

    view_keys = list(DEFAULT_VIEW_KEYS)
    print(f"[2/5] 生成 {len(selected)} 张多视图到 {out_root}: {view_keys}", flush=True)
    for i, row in enumerate(selected, 1):
        src = row["path"]
        process_one_image(
            src,
            out_root,
            view_keys=view_keys,
            stem=row["stem"],
            suffix=src.suffix,
            copy_raw_bytes=True,
        )
        raw_dst = out_root / "images_raw" / f"{row['stem']}{src.suffix}"
        if raw_dst.is_file():
            shutil.copy2(raw_dst, images_alias / raw_dst.name)
        src_label = labels_dir / f"{row['stem']}.txt"
        if src_label.is_file():
            shutil.copy2(src_label, labels_out / f"{row['stem']}.txt")
        else:
            (labels_out / f"{row['stem']}.txt").write_text("", encoding="utf-8")
        if i % 5 == 0 or i == len(selected):
            print(f"  processed {i}/{len(selected)}", flush=True)

    if classes_file.is_file():
        shutil.copy2(classes_file, out_root / "classes.txt")

    # also copy labels/ five-col stripped? not needed for review; keep with conf only
    # write meta
    meta = {
        "source_dataset": str(dataset_root),
        "created_at": stamp,
        "n_images": len(selected),
        "seed": args.seed,
        "low_conf_threshold": args.low_conf_threshold,
        "note": "显示增强仅用于审查；框来自原 labels_with_conf，未重新推理",
        "views": view_keys,
        "view_catalog": {
            k: {"label": VIEW_CATALOG[k][1], "help": VIEW_CATALOG[k][2]}
            for k in view_keys
            if k in VIEW_CATALOG
        },
        "presets": {
            "enhanced": "CLAHE clip=2.0 tile=8 bilateral=9",
            "strong": "CLAHE clip=4.0 tile=8 bilateral=5",
            "lab_l": "Lab L channel CLAHE + TURBO colormap",
            "hsv_v": "HSV V CLAHE",
            "hsv_s": "HSV S CLAHE + MAGMA",
            "tophat/blackhat": "morphology residual (kernel=21)",
        },
        "candidate_pool": len(candidates),
        "stratum_counts_pool": {},
        "stratum_counts_sample": {},
        "images": [],
    }
    for c in candidates:
        key = c.get("label_stratum", c["stratum"])
        meta["stratum_counts_pool"][key] = meta["stratum_counts_pool"].get(key, 0) + 1
    meta["oversample_n"] = len(oversample)
    for c in oversample:
        meta.setdefault("stratum_counts_oversample", {})
        meta["stratum_counts_oversample"][c["stratum"]] = (
            meta["stratum_counts_oversample"].get(c["stratum"], 0) + 1
        )
    for c in selected:
        meta["stratum_counts_sample"][c["stratum"]] = meta["stratum_counts_sample"].get(c["stratum"], 0) + 1
        meta["images"].append(
            {
                "name": c["name"],
                "mean_brightness": c["mean_brightness"],
                "brightness_bucket": c["brightness_bucket"],
                "conf_bucket": c["conf_bucket"],
                "area_bucket": c["area_bucket"],
                "stratum": c["stratum"],
                "low_box_count": c["low_box_count"],
                "total_box_count": c["total_box_count"],
                "min_low_conf": c["min_low_conf"],
                "mean_low_area": c["mean_low_area"],
            }
        )
    (out_root / "sample_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest_path = out_root / "sample_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "image_name",
            "stratum",
            "mean_brightness",
            "brightness_bucket",
            "conf_bucket",
            "area_bucket",
            "low_box_count",
            "total_box_count",
            "min_low_conf",
            "mean_low_area",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in selected:
            writer.writerow(
                {
                    "image_name": c["name"],
                    "stratum": c["stratum"],
                    "mean_brightness": c["mean_brightness"],
                    "brightness_bucket": c["brightness_bucket"],
                    "conf_bucket": c["conf_bucket"],
                    "area_bucket": c["area_bucket"],
                    "low_box_count": c["low_box_count"],
                    "total_box_count": c["total_box_count"],
                    "min_low_conf": c["min_low_conf"],
                    "mean_low_area": c["mean_low_area"],
                }
            )

    write_score_template(out_root / "visibility_score_template.csv")

    # README for human
    readme = f"""# 多视图审查抽样集

- 源数据集: `{dataset_root}`
- 抽样数: {len(selected)} / 候选池 {len(candidates)}
- 随机种子: {args.seed}
- **框来源**: 原 `labels_with_conf`（未重新推理）
- **视图**（框始终同一套 labels_with_conf）:
  - 原图 / Lab增强 / 强对比
  - Lab-L / HSV-V / HSV-S / 灰阶
  - 亮残差 top-hat / 暗残差 black-hat / 残差叠加 / 边缘 / Gamma

## 审查方式

1. 打开生成的 `review.html`
2. 用下拉框或 Q/E、数字键切换视图；框列表点「聚焦」看位置是否对准
3. 填写：原图能否判定 / 最佳增强能否判定 / 框位置是否合理 / 最有用视图
4. 对框 keep/remove/uncertain；导出 JSONL

## 决策规则建议

- 任一结构视图（残差/边缘）在框内能看到缺陷结构 → keep
- 多视图下框内无结构、或明显偏到空白/纹理噪声 → remove
- 仍无法判断 → uncertain
"""
    (out_root / "README.md").write_text(readme, encoding="utf-8")

    review_html = None
    if not args.skip_review_html:
        print("[3/5] 生成多视图 review.html")
        gen = Path(__file__).resolve().parent / "generate_flat_yolo_review.py"
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(gen),
                str(out_root),
                "--output-root",
                str(args.review_output_root),
                "--multi-view",
            ],
            check=True,
        )
        review_html = args.review_output_root / out_root.name / "review.html"
    else:
        print("[3/5] 跳过 review.html")

    print("[4/5] 完成")
    summary = {
        "output_root": str(out_root),
        "review_html": str(review_html) if review_html else None,
        "n_images": len(selected),
        "candidate_pool": len(candidates),
        "stratum_counts_sample": meta["stratum_counts_sample"],
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (out_root / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
