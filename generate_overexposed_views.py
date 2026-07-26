#!/usr/bin/env python3
"""为过曝发白的图片生成审查增强视图（几何不变、不改标签，仅处理亮度超阈值的图片）。

背景：审查页以 file:// 打开时浏览器禁止 canvas 像素级读取（SecurityError），
无法前端实时增强；因此把增强视图预先写成 JPG，由 generate_flat_yolo_review.py
的多视图模式自动发现（images_<suffix>/ 目录）。未生成视图的图片仍显示原图。

用法：
    python generate_overexposed_views.py scan                      # 只扫描亮度分布
    python generate_overexposed_views.py gen --threshold 165       # 为过曝图生成视图
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from review_enhance_utils import (
    IMAGE_SUFFIXES,
    VIEW_CATALOG,
    build_views,
    estimate_mean_brightness,
    imread_unicode,
)

DEFAULT_DATASET = Path(r"E:/PythonProject/datasets/combined_all_pseudo_v2")
DEFAULT_REPORT_DIR = Path(r"E:/PythonProject/优化/pseudo_label_reviews")
DEFAULT_VIEWS = ["enhanced", "strong", "gray"]
JPEG_QUALITY = 88


def iter_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def cmd_scan(args: argparse.Namespace) -> dict:
    images_dir = args.dataset_root.resolve() / args.images_dir
    if not images_dir.is_dir():
        raise NotADirectoryError(f"图片目录不存在: {images_dir}")
    files = iter_images(images_dir)
    print(f"[scan] {images_dir} 共 {len(files)} 张，开始降采样亮度扫描……")
    rows = []
    t0 = time.time()
    for idx, path in enumerate(files, 1):
        luma = estimate_mean_brightness(path)
        rows.append({"name": path.name, "mean_luma": None if luma is None else round(luma, 2)})
        if idx % 200 == 0:
            print(f"  {idx}/{len(files)} ({time.time() - t0:.0f}s)")
    values = np.array([r["mean_luma"] for r in rows if r["mean_luma"] is not None], dtype=float)
    percentiles = {str(p): round(float(np.percentile(values, p)), 1) for p in (50, 75, 90, 95, 99)}
    thresholds = {str(t): int((values >= t).sum()) for t in (150, 160, 165, 170, 180, 190, 200)}
    report = {
        "dataset": str(args.dataset_root),
        "images_dir": str(images_dir),
        "total": len(files),
        "unreadable": len(files) - len(values),
        "luma_percentiles": percentiles,
        "count_ge_threshold": thresholds,
        "rows": rows,
    }
    out = args.report_dir.resolve() / args.dataset_root.name / "overexposed_scan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[scan] 亮度分位数: {percentiles}")
    print(f"[scan] 各阈值过曝图片数: {thresholds}")
    print(f"[scan] 报告: {out}")
    return report


def write_jpg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"JPG 编码失败: {path}")
    buf.tofile(str(path))


def load_scan_names(args: argparse.Namespace) -> dict[str, float] | None:
    scan_path = args.report_dir.resolve() / args.dataset_root.name / "overexposed_scan.json"
    if not scan_path.is_file():
        return None
    data = json.loads(scan_path.read_text(encoding="utf-8"))
    if data.get("dataset") != str(args.dataset_root):
        return None
    return {r["name"]: r["mean_luma"] for r in data["rows"] if r["mean_luma"] is not None}


def cmd_gen(args: argparse.Namespace) -> None:
    dataset = args.dataset_root.resolve()
    images_dir = dataset / args.images_dir
    if not images_dir.is_dir():
        raise NotADirectoryError(f"图片目录不存在: {images_dir}")
    for key in args.views:
        if key not in VIEW_CATALOG or key == "raw":
            raise KeyError(f"未知或不允许的视图 key: {key}（可选: {[k for k in VIEW_CATALOG if k != 'raw']}）")

    cached = load_scan_names(args)
    files = iter_images(images_dir)
    if args.all:
        selected = [(p, None) for p in files]
        print(f"[gen] --all：处理全部 {len(selected)} 张")
    elif cached:
        selected = [(images_dir / n, v) for n, v in cached.items() if v >= args.threshold and (images_dir / n).is_file()]
        print(f"[gen] 复用扫描结果：均值亮度 ≥{args.threshold} 共 {len(selected)} 张")
    else:
        print(f"[gen] 无扫描缓存，逐张读取亮度（建议先运行 scan）……")
        selected = []
        for path in files:
            luma = estimate_mean_brightness(path)
            if luma is not None and luma >= args.threshold:
                selected.append((path, luma))
        print(f"[gen] 均值亮度 ≥{args.threshold} 共 {len(selected)} 张")

    if args.limit:
        selected = selected[: args.limit]
        print(f"[gen] --limit {args.limit}，本次只处理 {len(selected)} 张")

    t0 = time.time()
    done, skipped, failed = 0, 0, []
    bytes_written = 0
    for idx, (path, luma) in enumerate(selected, 1):
        targets = [
            dataset / f"images_{VIEW_CATALOG[key][0]}" / f"{path.stem}.jpg"
            for key in args.views
        ]
        if args.skip_existing and all(t.exists() for t in targets):
            skipped += 1
            continue
        try:
            image = imread_unicode(path)
            if image is None:
                raise RuntimeError("读取失败")
            views = build_views(image, args.views)
            for key, target in zip(args.views, targets):
                write_jpg(target, views[key])
                bytes_written += target.stat().st_size
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed.append({"name": path.name, "error": str(exc)})
        if idx % 20 == 0:
            print(f"  {idx}/{len(selected)} 成功 {done} 跳过 {skipped} 失败 {len(failed)} ({time.time() - t0:.0f}s)")

    report = {
        "dataset": str(dataset),
        "threshold": args.threshold,
        "views": args.views,
        "jpeg_quality": JPEG_QUALITY,
        "selected": len(selected),
        "generated": done,
        "skipped_existing": skipped,
        "failed": failed,
        "bytes_written": bytes_written,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = args.report_dir.resolve() / dataset.name / "overexposed_views_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gen] 完成：生成 {done} 张 × {len(args.views)} 视图，跳过 {skipped}，失败 {len(failed)}，写入 {bytes_written / 1e9:.2f} GB")
    print(f"[gen] 报告: {out}")
    if failed:
        sys.exit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["scan", "gen"], help="scan=只扫描亮度分布；gen=生成增强视图")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--images-dir", default="images_all", help="数据集内源图目录名")
    parser.add_argument("--threshold", type=float, default=105.0, help="过曝判定：均值灰度 ≥ 该值（本数据集 p90≈102、p95≈109）")
    parser.add_argument("--all", action="store_true", help="不按亮度筛选，处理全部图片")
    parser.add_argument("--views", nargs="*", default=DEFAULT_VIEWS, help=f"要生成的视图，默认 {DEFAULT_VIEWS}")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少张（调试用）")
    parser.add_argument("--skip-existing", action="store_true", help="全部视图已存在则跳过")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "scan":
        cmd_scan(args)
    else:
        cmd_gen(args)


if __name__ == "__main__":
    main()
