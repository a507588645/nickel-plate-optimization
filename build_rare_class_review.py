#!/usr/bin/env python3
"""
L1 稀少类人工复核页生成：youpao(14) / heibian(15) / mijixiaokeng(16)。

数据源：
1. `dianjienie` 原始人工标签（17 类 id14/15/16，映射表见 `类别映射_dianjienie17_to_ryw18.md`）
2. `combined_all_pseudo_v2/labels_with_conf` 伪标签命中（18 类同 id，带置信度）

产物（写入 `E:\\PythonProject\\优化\\rare_class_review_v1\\`，不修改数据集）：
- `images/` 原图硬链接（失败则复制）
- `renders/` 画框渲染图（三类的框着色 + 类名标注）
- `review.html` 浏览器审查页（原图链接 + 渲染图 + 框明细表）
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_DIANJIENIE = Path(r"E:/PythonProject/datasets/dianjienie")
DEFAULT_COMBINED = Path(r"E:/PythonProject/datasets/combined_all_pseudo_v2")
DEFAULT_OUT = ROOT / "rare_class_review_v1"

TARGET = {14: "youpao", 15: "heibian", 16: "mijixiaokeng"}
COLORS = {14: (0, 200, 255), 15: (220, 0, 220), 16: (255, 160, 0)}  # BGR
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_RENDER_W = 1600


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="稀少类（14/15/16）图片+标注审查页生成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dianjienie", type=Path, default=DEFAULT_DIANJIENIE)
    p.add_argument("--combined", type=Path, default=DEFAULT_COMBINED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def read_gray_bgr(path: Path):
    """中文路径安全读图（BGR）。"""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def find_image(img_root: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_root / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        dst.hardlink_to(src)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def parse_yolo_line(parts: list[str]):
    if len(parts) < 5:
        return None
    try:
        cid = int(float(parts[0]))
        cx, cy, w, h = (float(parts[i]) for i in range(1, 5))
        conf = float(parts[5]) if len(parts) >= 6 else None
    except ValueError:
        return None
    return cid, cx, cy, w, h, conf


def render(img_path: Path, boxes: list[dict], out_path: Path) -> str | None:
    import cv2

    img = read_gray_bgr(img_path)
    if img is None:
        return None
    H, W = img.shape[:2]
    for b in boxes:
        x1 = int((b["cx"] - b["w"] / 2) * W)
        y1 = int((b["cy"] - b["h"] / 2) * H)
        x2 = int((b["cx"] + b["w"] / 2) * W)
        y2 = int((b["cy"] + b["h"] / 2) * H)
        color = COLORS[b["cid"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        tag = f'{b["name"]}' + (f' {b["conf"]:.2f}' if b.get("conf") is not None else "")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, tag, (x1 + 3, max(th, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    scale = 1.0
    if W > MAX_RENDER_W:
        scale = MAX_RENDER_W / W
        img = cv2.resize(img, (MAX_RENDER_W, int(H * scale)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    buf.tofile(str(out_path))  # 中文路径安全写图
    return f"{W}x{H}" + (f"→{MAX_RENDER_W}w" if scale < 1.0 else "")


def collect_dianjienie(root: Path) -> list[dict]:
    """扫描 17 类原始标签中的 id 14/15/16。"""
    hits: dict[str, dict] = {}
    for split in ("train", "valid", "test"):
        lab_dir = root / split / "labels"
        img_dir = root / split / "images"
        if not lab_dir.exists():
            continue
        for lf in lab_dir.glob("*.txt"):
            for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                parsed = parse_yolo_line(line.split())
                if not parsed:
                    continue
                cid, cx, cy, w, h, _ = parsed
                if cid not in TARGET:
                    continue
                key = f"{split}/{lf.stem}"
                entry = hits.setdefault(key, {"split": split, "stem": lf.stem, "img_dir": img_dir, "boxes": []})
                entry["boxes"].append(
                    {"cid": cid, "name": f"unknown_{cid}({TARGET[cid]})", "cx": cx, "cy": cy, "w": w, "h": h, "conf": None}
                )
    return list(hits.values())


def collect_combined(root: Path) -> list[dict]:
    """扫描伪标签（6 列）中的 id 14/15/16。"""
    lab_dir = root / "labels_with_conf"
    img_dir = root / "images_all"
    hits: dict[str, dict] = {}
    if not lab_dir.exists():
        return []
    for lf in lab_dir.glob("*.txt"):
        for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
            parsed = parse_yolo_line(line.split())
            if not parsed:
                continue
            cid, cx, cy, w, h, conf = parsed
            if cid not in TARGET:
                continue
            key = lf.stem
            entry = hits.setdefault(key, {"split": "pseudo", "stem": lf.stem, "img_dir": img_dir, "boxes": []})
            entry["boxes"].append(
                {"cid": cid, "name": TARGET[cid], "cx": cx, "cy": cy, "w": w, "h": h, "conf": conf}
            )
    return list(hits.values())


def build_html(sections: list[tuple[str, list[dict]]], out: Path, counts: dict) -> str:
    css = """
body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:24px;background:#fafafa;color:#222}
h1{font-size:22px} h2{font-size:18px;margin-top:32px;border-bottom:2px solid #999;padding-bottom:4px}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.card img{max-width:100%;height:auto;border:1px solid #ccc}
table{border-collapse:collapse;margin-top:8px;font-size:13px}
td,th{border:1px solid #ccc;padding:3px 10px}
th{background:#eee}
.legend span{display:inline-block;padding:2px 10px;margin-right:10px;border-radius:4px;color:#fff;font-size:13px}
.warn{background:#fff8e1;border:1px solid #e0c060;padding:8px 12px;border-radius:6px}
a{color:#06c}
"""
    html = [
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>",
        "<title>稀少类复核：youpao / heibian / mijixiaokeng</title>",
        f"<style>{css}</style></head><body>",
        "<h1>稀少类标注复核（youpao 14 / heibian 15 / mijixiaokeng 16）</h1>",
        f"<p>生成时间：{datetime.now():%Y-%m-%d %H:%M}　来源：dianjienie 原始人工标签 + combined_all_pseudo_v2 伪标签　"
        f"命中：<b>{counts['dianjienie']}</b> 张（人工）/ <b>{counts['combined']}</b> 张（伪标签）</p>",
        "<p class='legend'>图例：<span style='background:#00c8ff'>youpao 油泡</span>"
        "<span style='background:#dc00dc'>heibian 黑边</span>"
        "<span style='background:#ffa000'>mijixiaokeng 密集小坑</span></p>",
        "<p class='warn'>说明：人工标签以 <b>unknown_XX（映射名）</b> 双命名显示，无置信度列；伪标签框标注含置信度（&lt;0.40 为 rejected 带）。"
        "点击「原图」可查看未画框全分辨率图。<b>unknown_14（youpao 油泡）与 fanxitong 在全库 2694 个标签文件中零实例，无图可展示</b>；"
        "本页 4 张人工标签图即 unknown_15/16 的全部实例。</p>",
    ]
    for title, items in sections:
        html.append(f"<h2>{title}（{len(items)} 张）</h2>")
        for it in items:
            cells = []
            for b in it["boxes"]:
                conf_txt = "" if b["conf"] is None else f"{b['conf']:.3f}"
                cells.append(
                    f"<tr><td>{b['name']}</td><td>{conf_txt}</td>"
                    f"<td>{b['cx']:.4f}</td><td>{b['cy']:.4f}</td><td>{b['w']:.4f}</td><td>{b['h']:.4f}</td></tr>"
                )
            rows = "".join(cells)
            html.append(
                f"<div class='card'><b>{it['stem']}</b>（{it['split']}，{it.get('size','')}）　"
                f"<a href='images/{it['link_name']}' target='_blank'>原图</a><br>"
                f"<img src='renders/{it['render_name']}' loading='lazy'>"
                f"<table><tr><th>类</th><th>conf</th><th>cx</th><th>cy</th><th>w</th><th>h</th></tr>{rows}</table></div>"
            )
    html.append("</body></html>")
    text = "\n".join(html)
    (out / "review.html").write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=== 稀少类复核页生成 ===")
    if args.dry_run:
        print(f"[dry-run] dianjienie={args.dianjienie.exists()} combined={args.combined.exists()}")
        return 0

    dj = collect_dianjienie(args.dianjienie)
    cb = collect_combined(args.combined)
    print(f"人工标签命中 {len(dj)} 张；伪标签命中 {len(cb)} 张")

    for group in (dj, cb):
        for it in group:
            img_path = find_image(it["img_dir"], it["stem"])
            if img_path is None:
                it["error"] = "image_not_found"
                continue
            link_name = f"{it['split']}_{it['stem']}{img_path.suffix}"
            it["link_name"] = link_name
            it["link"] = link_or_copy(img_path, out / "images" / link_name)
            render_name = f"{it['split']}_{it['stem']}.jpg"
            it["render_name"] = render_name
            size = render(img_path, it["boxes"], out / "renders" / render_name)
            it["size"] = size or "render_failed"

    sections = [
        ("dianjienie 原始人工标签（映射：14=youpao 15=heibian 16=mijixiaokeng）", dj),
        ("combined_all_pseudo_v2 伪标签（带置信度）", cb),
    ]
    build_html(sections, out, {"dianjienie": len(dj), "combined": len(cb)})

    manifest = [
        {
            "stem": it["stem"], "split": it["split"],
            "n_boxes": len(it["boxes"]),
            "classes": sorted({b["name"] for b in it["boxes"]}),
            "link": it.get("link"), "render": it.get("size"), "error": it.get("error"),
        }
        for it in dj + cb
    ]
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "target_classes": TARGET,
                "items": manifest,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    with (out / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()) if manifest else ["stem"])
        w.writeheader()
        w.writerows(manifest)

    print(f"产物: {out}（review.html / images / renders / manifest.*）")
    missing = [m["stem"] for m in manifest if m.get("error")]
    if missing:
        print(f"缺图: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
