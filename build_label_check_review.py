#!/usr/bin/env python3
"""
L1 标签检查复核页：worst_images.csv TOP50 的原标签(GT) vs ryw.pt 预测对比。

绿色实线 = 原标签（人工/映射，真值候选）；红色细线 = ryw.pt 预测（conf≥0.25，标注类名+置信度）。
产物写入 `E:\\PythonProject\\优化\\label_check_ryw_v1\\review.html` 及 renders/。
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
UNION = Path(r"E:/PythonProject/datasets/union_original_labels_v1")
CHECK_DIR = ROOT / "label_check_ryw_v1"

CLASSES_18 = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]
GT_COLOR = (80, 200, 80)    # BGR 绿
PRED_COLOR = (60, 60, 230)  # BGR 红
MAX_W = 1600


def read_bgr(path: Path):
    import cv2

    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="标签检查 TOP50 复核页", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--union", type=Path, default=UNION)
    p.add_argument("--check-dir", type=Path, default=CHECK_DIR)
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    out = args.check_dir
    worst = list(csv.DictReader((out / "worst_images.csv").open(encoding="utf-8-sig")))[: args.top]
    preds: dict[str, list] = {}
    with (out / "predictions.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c = float(r["conf"])
            if c >= args.conf:
                preds.setdefault(r["union_name"], []).append(
                    (int(r["label"]), c, float(r["cx"]), float(r["cy"]), float(r["w"]), float(r["h"]))
                )
    print(f"复核图 {len(worst)} 张；预测读取 conf≥{args.conf}")
    if args.dry_run:
        return 0

    import cv2

    img_index = {p.stem: p for p in (args.union / "images").iterdir()}
    cards = []
    for w in worst:
        name = w["union_name"]
        ip = img_index.get(name)
        if ip is None:
            continue
        img = read_bgr(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = []
        lf = args.union / "labels" / f"{name}.txt"
        if lf.exists():
            for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                p_ = line.split()
                if len(p_) < 5:
                    continue
                gt.append((int(float(p_[0])), float(p_[1]), float(p_[2]), float(p_[3]), float(p_[4])))

        def draw(boxes, color, thick, with_conf):
            for b in boxes:
                cid, cx, cy, bw, bh = b[0], b[1], b[2], b[3], b[4]
                x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
                x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
                tag = CLASSES_18[cid] + (f" {b[5]:.2f}" if with_conf else "")
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                ty = y1 if not with_conf else y2
                cv2.rectangle(img, (x1, max(0, ty - th - 8)), (x1 + tw + 4, ty), color, -1)
                cv2.putText(img, tag, (x1 + 2, max(th, ty - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        draw([(c, x, y, ww, hh) for c, x, y, ww, hh in gt], GT_COLOR, 3, False)
        pr = preds.get(name, [])
        draw([(c, x, y, ww, hh, cf) for c, cf, x, y, ww, hh in pr], PRED_COLOR, 1, True)
        if W > MAX_W:
            img = cv2.resize(img, (MAX_W, int(H * MAX_W / W)))
        rn = f"{name}.jpg"
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            (out / "renders").mkdir(exist_ok=True)
            buf.tofile(str(out / "renders" / rn))
        cards.append((w, name, gt, pr, rn))

    css = "body{font-family:system-ui,'Microsoft YaHei';margin:24px;background:#fafafa}h2{font-size:18px}.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:14px 0}.card img{max-width:100%}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:2px 8px}.warn{background:#fff8e1;border:1px solid #e0c060;padding:8px 12px;border-radius:6px}"
    html = [
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'><title>标签检查 TOP50 复核</title>",
        f"<style>{css}</style></head><body>",
        f"<h1>原标签（绿）vs ryw.pt 预测（红, conf≥{args.conf}）——分歧 TOP{len(cards)}</h1>",
        f"<p>生成时间：{datetime.now():%Y-%m-%d %H:%M}　数据源：union_original_labels_v1 + label_check_ryw_v1/predictions.csv</p>",
        "<p class='warn'>绿框=原标签（真值候选），红框=ryw.pt 预测。score=漏检原框+0.5×额外框。"
        "请判断每张图是「原标签错/欠标」还是「模型错」；盲区类（yellowstain/fawu/NiOH/heibian/mijixiaokeng/overfire/greenstain）模型无检查能力。</p>",
    ]
    for w, name, gt, pr, rn in cards:
        def rows_of(items, with_conf):
            out_rows = []
            for b in items:
                conf_cell = f"<td>{b[5]:.2f}</td>" if with_conf else ""
                out_rows.append(
                    f"<tr><td>{CLASSES_18[b[0]]}</td>{conf_cell}<td>{b[1]:.3f}</td><td>{b[2]:.3f}</td><td>{b[3]:.3f}</td><td>{b[4]:.3f}</td></tr>"
                )
            return "".join(out_rows)

        gt_tbl = f"<h4>GT（{len(gt)}）</h4><table><tr><th>类</th><th>cx</th><th>cy</th><th>w</th><th>h</th></tr>{rows_of(gt, False)}</table>" if gt else "<p>无 GT</p>"
        pr_tbl = f"<h4>预测（{len(pr)}）</h4><table><tr><th>类</th><th>conf</th><th>cx</th><th>cy</th><th>w</th><th>h</th></tr>{rows_of(pr, True)}</table>" if pr else "<p>无预测</p>"
        html.append(
            f"<div class='card'><b>{name}</b>　GT={len(gt)} 框 / 预测={len(pr)} 框 / score={w['score']}<br>"
            f"<img src='renders/{rn}' loading='lazy'>{gt_tbl}{pr_tbl}</div>"
        )
    html.append("</body></html>")
    (out / "review.html").write_text("\n".join(html), encoding="utf-8")
    print(f"产物: {out / 'review.html'}（{len(cards)} 张卡）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
