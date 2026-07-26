#!/usr/bin/env python3
"""为已有多视图审查抽样集补生成颜色空间/形态学视图（不改标签、不重采样）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from review_enhance_utils import (
    DEFAULT_VIEW_KEYS,
    VIEW_CATALOG,
    imread_unicode,
    imwrite_unicode,
    process_one_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为审查抽样集补生成更多显示视图")
    parser.add_argument(
        "sample_root",
        type=Path,
        help="抽样集根目录，如 review_enhance_samples/dianjie_n80_...",
    )
    parser.add_argument(
        "--views",
        nargs="*",
        default=None,
        help="要生成的视图 key 列表；默认除 raw 外全部 DEFAULT_VIEW_KEYS",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="目标文件已存在则跳过该视图",
    )
    parser.add_argument(
        "--regen-html",
        action="store_true",
        help="生成后调用 generate_flat_yolo_review.py 刷新审查页",
    )
    parser.add_argument(
        "--review-output-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/pseudo_label_reviews"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.sample_root.resolve()
    raw_dir = root / "images_raw"
    if not raw_dir.is_dir():
        # fallback images/
        raw_dir = root / "images"
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"找不到 images_raw 或 images: {root}")

    if args.views:
        view_keys = args.views
    else:
        view_keys = [k for k in DEFAULT_VIEW_KEYS if k != "raw"]

    images = sorted(p for p in raw_dir.iterdir() if p.is_file())
    print(f"sample={root.name} images={len(images)} views={view_keys}", flush=True)

    for i, src in enumerate(images, 1):
        # optional skip: if all target views exist
        if args.skip_existing:
            missing = []
            for key in view_keys:
                suffix = VIEW_CATALOG.get(key, (key, key, ""))[0]
                dst = root / f"images_{suffix}" / f"{src.stem}{src.suffix}"
                if not dst.is_file():
                    missing.append(key)
            if not missing:
                if i % 10 == 0 or i == len(images):
                    print(f"  skip-existing {i}/{len(images)}", flush=True)
                continue
            keys = missing
        else:
            keys = view_keys

        image = imread_unicode(src)
        if image is None:
            print(f"[warn] unreadable {src}", flush=True)
            continue
        # write only requested non-raw views
        from review_enhance_utils import build_views

        built = build_views(image, keys)
        for key, img in built.items():
            if key == "raw":
                continue
            suffix = VIEW_CATALOG.get(key, (key, key, ""))[0]
            dst = root / f"images_{suffix}" / f"{src.stem}{src.suffix}"
            if args.skip_existing and dst.is_file():
                continue
            if not imwrite_unicode(dst, img):
                print(f"[warn] write fail {dst}", flush=True)
        if i % 5 == 0 or i == len(images):
            print(f"  processed {i}/{len(images)}", flush=True)

    # update sample_meta presets if present
    meta_path = root / "sample_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["views_extended"] = view_keys
        meta["view_catalog"] = {
            k: {"label": VIEW_CATALOG[k][1], "help": VIEW_CATALOG[k][2]}
            for k in view_keys
            if k in VIEW_CATALOG
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.regen_html:
        gen = Path(__file__).resolve().parent / "generate_flat_yolo_review.py"
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(gen),
                str(root),
                "--output-root",
                str(args.review_output_root),
                "--multi-view",
            ],
            check=True,
        )

    print(json.dumps({"sample_root": str(root), "views": view_keys, "n_images": len(images)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
