# -*- coding: utf-8 -*-
"""C0.1：overfire 标签审计审查页生成器（路线 C0 第一步，零 GPU，只读数据集）。

抽样规则（路线文档 §4.0）：
- 全体 = anchor_v1 开发集（fold_01/02/03 的 train+val 并集，1,127 图）中类别 8（overfire）的全部 GT 框；
- 重板（单板 ≥5 个 overfire 框，共 11 张）的全部框 + 其余框按固定 seed 随机抽 40 个 ≈ 100 个裁决单元。

每单元输出：目标框外扩 2 倍裁剪图 + 整板缩略图（目标框红框、该板其余 overfire 框橙框）+ 元数据。
产物：c0_overfire_audit_<date>/review.html、manifest.json、decisions.jsonl（预填单元、decision 留空）。

用法:
  python generate_c0_overfire_audit.py            # 正式生成
  python generate_c0_overfire_audit.py --dry-run  # 只统计抽样，不渲染
"""
from __future__ import annotations

import argparse
import html
import json
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path("E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1")
OUT = Path("E:/PythonProject/优化/c0_overfire_audit_20260904")
SEED = 20260904
RANDOM_N = 40
HEAVY_MIN = 5
OVERFIRE = 8


def imread_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return img


def imwrite_cn(path: Path, img: np.ndarray, quality: int = 90) -> None:
    """cv2.imwrite 在本环境对含中文的路径返回 False；按项目惯例改 imencode+write_bytes。"""
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"imencode 失败: {path}")
    path.write_bytes(buf.tobytes())


def collect() -> tuple[dict[str, list], dict[str, tuple[int, int]]]:
    dev = set()
    for fold in ("fold_01", "fold_02", "fold_03"):
        for split in ("train.txt", "val.txt"):
            for line in (ROOT / "splits" / "folds" / fold / split).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    dev.add(Path(line.strip()).name)
    plates: dict[str, list] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for name in sorted(dev):
        lbl = ROOT / "labels" / (Path(name).stem + ".txt")
        if not lbl.exists():
            continue
        with Image.open(ROOT / "images" / name) as im:
            w, h = im.size
        sizes[name] = (w, h)
        boxes = []
        for line in lbl.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 5 and int(p[0]) == OVERFIRE:
                _, xc, yc, bw, bh = map(float, p)
                boxes.append((xc, yc, bw, bh))
        if boxes:
            plates[name] = boxes
    return plates, sizes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="只统计抽样，不渲染")
    args = ap.parse_args()

    plates, sizes = collect()
    heavy = {n: b for n, b in plates.items() if len(b) >= HEAVY_MIN}
    rest_units = [(n, i, b) for n, b in plates.items() if len(b) < HEAVY_MIN for i, b in enumerate(b)]
    rng = random.Random(SEED)
    sampled_rest = sorted(rng.sample(rest_units, min(RANDOM_N, len(rest_units))))
    total_boxes = sum(len(b) for b in plates.values())
    print(f"开发集含 overfire 框的板：{len(plates)}；总框数 {total_boxes}；重板 {len(heavy)} 张共 "
          f"{sum(len(b) for b in heavy.values())} 框；随机抽样 {len(sampled_rest)} 框", flush=True)

    units = []
    for name in sorted(heavy, key=lambda n: (-len(plates[n]), n)):
        for i, b in enumerate(plates[name]):
            units.append({"image": name, "box_index": i, "box": b, "plate_overfire_count": len(plates[name]),
                          "sampling": "heavy_plate"})
    for name, i, b in sampled_rest:
        units.append({"image": name, "box_index": i, "box": b, "plate_overfire_count": len(plates[name]),
                      "sampling": "random"})
    for idx, u in enumerate(units, 1):
        u["unit"] = f"U{idx:03d}"

    if args.dry_run:
        print(json.dumps({"n_units": len(units), "n_heavy_plates": len(heavy)}, ensure_ascii=False))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    crops = OUT / "crops"
    plates_dir = OUT / "plates"
    crops.mkdir(exist_ok=True)
    plates_dir.mkdir(exist_ok=True)

    plate_cache: dict[str, np.ndarray] = {}
    cards = []
    for u in units:
        name = u["image"]
        w, h = sizes[name]
        xc, yc, bw, bh = u["box"]
        x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
        x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h

        if name not in plate_cache:
            plate_cache[name] = imread_bgr(ROOT / "images" / name)
        img = plate_cache[name]

        # 整板缩略图（每板一次）：目标框红粗、该板其余 overfire 框橙细
        thumb_path = plates_dir / f"{Path(name).stem}.jpg"
        if not thumb_path.exists():
            thumb = img.copy()
            for j, (oxc, oyc, obw, obh) in enumerate(plates[name]):
                ox1, oy1 = (oxc - obw / 2) * w, (oyc - obh / 2) * h
                ox2, oy2 = (oxc + obw / 2) * w, (oyc + obh / 2) * h
                color = (0, 0, 255) if j == u["box_index"] else (0, 140, 255)
                thick = 6 if j == u["box_index"] else 2
                cv2.rectangle(thumb, (int(ox1), int(oy1)), (int(ox2), int(oy2)), color, thick)
            th = 800 / thumb.shape[1]
            thumb = cv2.resize(thumb, (800, int(thumb.shape[0] * th)))
            imwrite_cn(thumb_path, thumb, quality=85)

        # 裁剪图：外扩 2 倍（宽高各 ×2，中心不变），裁到图界
        cx1 = max(0, int(x1 - bw * w / 2)); cy1 = max(0, int(y1 - bh * h / 2))
        cx2 = min(w, int(x2 + bw * w / 2)); cy2 = min(h, int(y2 + bh * h / 2))
        crop = img[cy1:cy2, cx1:cx2].copy()
        cv2.rectangle(crop, (int(x1 - cx1), int(y1 - cy1)), (int(x2 - cx1), int(y2 - cy1)), (0, 0, 255), 3)
        crop_name = f"{u['unit']}_{Path(name).stem}_b{u['box_index']}.jpg"
        imwrite_cn(crops / crop_name, crop, quality=90)

        cards.append(
            f"<article class='card'><h2>{u['unit']} · {html.escape(name)} · 框#{u['box_index']}"
            f"{' · <span class=heavy>重板</span>' if u['sampling'] == 'heavy_plate' else ''}</h2>"
            f"<div class='meta'>该板 overfire GT 框数 {u['plate_overfire_count']}；框(归一化 xc,yc,w,h)=("
            f"{xc:.4f}, {yc:.4f}, {bw:.4f}, {bh:.4f})；原图 {w}×{h}</div>"
            f"<div class='imgs'><figure><figcaption>框裁剪（外扩2倍）</figcaption>"
            f"<img src='crops/{crop_name}'></figure>"
            f"<figure><figcaption>整板（红=本单元，橙=该板其他 overfire）</figcaption>"
            f"<img src='plates/{thumb_path.name}'></figure></div></article>"
        )

    (OUT / "review.html").write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>C0 overfire 标签审计</title>"
        "<style>body{font-family:'Microsoft YaHei',sans-serif;background:#f2eee6;margin:0}"
        "header{padding:16px 32px;background:#fffdf8;border-bottom:1px solid #d8d0c2}"
        "main{max-width:1600px;margin:0 auto;padding:20px;display:grid;gap:16px}"
        ".card{background:#fffdf8;border:1px solid #d8d0c2;border-radius:10px;padding:12px}"
        ".card h2{font-size:16px;margin:0 0 4px}.heavy{color:#b34700}"
        ".meta{color:#5b6965;font-size:12px;margin-bottom:8px;font-variant-numeric:tabular-nums}"
        ".imgs{display:flex;gap:12px;align-items:flex-start}.imgs figure{margin:0}"
        ".imgs figcaption{font-size:12px;color:#5b6965;margin-bottom:4px}"
        ".imgs img{max-height:420px;border:1px solid #cfc7b9;display:block}</style></head><body>"
        "<header><h1>C0 overfire 标签审计（路线 C0.1）</h1>"
        f"<p>裁决三分类：<b>true_overfire</b>=业务认可的烧板标注 / <b>over_labeled</b>=过度标注（业务不认可） / "
        "<b>uncertain</b>=存疑。共 " + str(len(units)) + " 单元（重板全框 + 随机 40）。"
        "裁决方式：按单元号逐条给出 decision（可附 note），由 Agent 回填 decisions.jsonl。</p></header>"
        f"<main>{''.join(cards)}</main></body></html>", encoding="utf-8")

    with (OUT / "decisions.jsonl").open("w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps({"unit": u["unit"], "image": u["image"], "box_index": u["box_index"],
                                "box": [round(v, 6) for v in u["box"]],
                                "sampling": u["sampling"], "decision": "", "note": ""},
                               ensure_ascii=False) + "\n")

    (OUT / "manifest.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED, "random_n": RANDOM_N, "heavy_min_boxes": HEAVY_MIN,
        "source": {"labels": str(ROOT / "labels"), "dev": "splits/folds/fold_0{1,2,3}/{train,val}.txt"},
        "n_plates_with_overfire": len(plates), "n_overfire_boxes_dev": total_boxes,
        "n_heavy_plates": len(heavy), "n_units": len(units),
        "units": [{"unit": u["unit"], "image": u["image"], "box_index": u["box_index"],
                   "box": u["box"], "sampling": u["sampling"]} for u in units],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"DONE units={len(units)} -> {OUT / 'review.html'}", flush=True)


if __name__ == "__main__":
    main()
