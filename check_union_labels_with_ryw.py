#!/usr/bin/env python3
"""
L1：用 ryw.pt 检查 union_original_labels_v1 原标签准确性。

流程：
1. 对 union 中有标签的 1455 张图跑 ryw.pt（18 类，conf 下限 0.05 以便多阈值分析）。
2. 原标签框 vs 预测框做 IoU 匹配（同一类 id；类 id 体系一致，dianjienie 已按序映射）。
3. 输出：各类召回（多 conf×IoU 阈值）、类混淆对、额外预测、分歧最大图片清单。

运行环境：`E:\\PythonProject\\ImetricCodes\\JinchuanYolov5sTrain\\venv\\Scripts\\python.exe`
产物：`E:\\PythonProject\\优化\\label_check_ryw_v1\\`
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

UNION = Path(r"E:/PythonProject/datasets/union_original_labels_v1")
DEFAULT_OUT = Path(__file__).resolve().parent / "label_check_ryw_v1"
DEFAULT_WEIGHTS = Path(r"E:/PythonProject/Tools/ModelAutoLabelTool/weights/yolov5/ryw.pt")
DEFAULT_REPO = Path(r"E:/PythonProject/ImetricCodes/JinchuanYolov5sTrain")

CLASSES_18 = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]
CONF_THRS = [0.15, 0.25, 0.40, 0.70]
IOU_THRS = [0.1, 0.3, 0.5]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="ryw.pt 检查 union 原标签准确性",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--union", type=Path, default=UNION)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--yolov5-repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf-floor", type=float, default=0.05)
    p.add_argument("--iou-nms", type=float, default=0.45)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--analyze-only", action="store_true", help="跳过推理，只用已有 predictions.csv 分析")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def read_bgr(path: Path):
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_model(args):
    import torch

    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": k.get("weights_only", False)})
    device = "0" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    repo = str(args.yolov5_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    old = Path.cwd()
    os.chdir(repo)
    try:
        model = torch.hub.load(repo, "custom", path=str(args.weights), source="local",
                               device=device, autoshape=True, _verbose=False)
    finally:
        os.chdir(old)
    model.eval()
    model.conf = float(args.conf_floor)
    model.iou = float(args.iou_nms)
    print(f"device={device} cuda={torch.cuda.is_available()}")
    return model


def run_inference(args, labeled: list[dict[str, str]]) -> Path:
    model = load_model(args)
    out_csv = args.output_dir / "predictions.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["union_name", "label", "conf", "cx", "cy", "w", "h"])
        for i, row in enumerate(labeled):
            img_path = args.union / "images" / row["image_file"]
            img = read_bgr(img_path)
            if img is None:
                print(f"!! 不可读 {row['union_name']}", file=sys.stderr)
                continue
            H, W = img.shape[:2]
            res = model(img, size=args.imgsz)
            for *xyxy, conf, cls in res.xyxy[0].tolist():
                x1, y1, x2, y2 = xyxy
                w.writerow([
                    row["union_name"], int(cls), round(float(conf), 4),
                    round(((x1 + x2) / 2) / W, 6), round(((y1 + y2) / 2) / H, 6),
                    round((x2 - x1) / W, 6), round((y2 - y1) / H, 6),
                ])
            if (i + 1) % 100 == 0 or (i + 1) == len(labeled):
                rate = (i + 1) / (time.time() - t0)
                print(f"  [{i+1}/{len(labeled)}] {rate:.1f} img/s")
    return out_csv


def xywh_to_xyxy(b):
    cx, cy, w, h = b
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + bb - inter) if (aa + bb - inter) > 0 else 0.0


def load_labels(args, labeled: list[dict[str, str]]):
    """union_name -> [(cid, cx, cy, w, h)]"""
    out = {}
    for row in labeled:
        lf = args.union / "labels" / f"{row['union_name']}.txt"
        boxes = []
        if lf.exists():
            for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                try:
                    boxes.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
                except ValueError:
                    continue
        out[row["union_name"]] = boxes
    return out


def analyze(args, pred_csv: Path, labels: dict[str, list]) -> dict:
    preds: dict[str, list] = defaultdict(list)
    with pred_csv.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            preds[r["union_name"]].append(
                (int(r["label"]), float(r["conf"]), float(r["cx"]), float(r["cy"]), float(r["w"]), float(r["h"]))
            )

    # 各类召回：conf × IoU 网格
    recall = {c: {ct: {it: [0, 0] for it in IOU_THRS} for ct in CONF_THRS} for c in range(18)}
    confusion: dict[tuple[int, int], int] = defaultdict(int)  # (原类, 预测类 or -1)
    extra_pred: dict[int, int] = defaultdict(int)
    per_image = []

    for name, gt in labels.items():
        pr_all = preds.get(name, [])
        unmatched_gt = 0
        for (cid, cx, cy, w, h) in gt:
            box = (cx, cy, w, h)
            # 召回网格
            for ct in CONF_THRS:
                best = 0.0
                for (pc, pconf, px, py, pw, ph) in pr_all:
                    if pc != cid or pconf < ct:
                        continue
                    best = max(best, iou(box, (px, py, pw, ph)))
                for it in IOU_THRS:
                    recall[cid][ct][it][1] += 1
                    if best >= it:
                        recall[cid][ct][it][0] += 1
            # 混淆（conf≥0.25 任意类最佳 IoU）
            best_c, best_i = -1, 0.0
            for (pc, pconf, px, py, pw, ph) in pr_all:
                if pconf < 0.25:
                    continue
                v = iou(box, (px, py, pw, ph))
                if v > best_i:
                    best_c, best_i = pc, v
            confusion[(cid, best_c if best_i >= 0.3 else -1)] += 1
            if best_i < 0.3:
                unmatched_gt += 1
        # 额外预测（conf≥0.25，与任意原框 IoU<0.1）
        extra = 0
        for (pc, pconf, px, py, pw, ph) in pr_all:
            if pconf < 0.25:
                continue
            if all(iou((px, py, pw, ph), (cx, cy, w, h)) < 0.1 for (_, cx, cy, w, h) in gt):
                extra_pred[pc] += 1
                extra += 1
        per_image.append({
            "union_name": name, "gt_boxes": len(gt), "pred_boxes_ge25": sum(1 for p in pr_all if p[1] >= 0.25),
            "unmatched_gt_iou03": unmatched_gt, "extra_pred_ge25": extra,
            "score": unmatched_gt + 0.5 * extra,
        })

    return {"recall": recall, "confusion": confusion, "extra_pred": dict(extra_pred), "per_image": per_image}


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("=== ryw.pt 检查 union 原标签 ===")

    rows = list(csv.DictReader((args.union / "manifests" / "samples.csv").open(encoding="utf-8-sig")))
    if rows and "image_file" not in rows[0]:
        # samples.csv 字段为 union_name；推图像文件名
        img_index = {p.stem: p.name for p in (args.union / "images").iterdir()}
        for r in rows:
            r["image_file"] = img_index.get(r["union_name"], "")
    labeled = [r for r in rows if r.get("has_label") == "True"]
    if args.limit:
        labeled = labeled[: args.limit]
    print(f"有标签图: {len(labeled)}")
    if args.dry_run:
        print("[dry-run] 停止")
        return 0

    pred_csv = args.output_dir / "predictions.csv"
    if not args.analyze_only:
        pred_csv = run_inference(args, labeled)
    elif not pred_csv.exists():
        print(f"缺少 {pred_csv}", file=sys.stderr)
        return 2

    labels = load_labels(args, labeled)
    res = analyze(args, pred_csv, labels)

    # 写各类召回（conf=0.25 主表 + 全网格 JSON）
    recall_rows = []
    for cid in range(18):
        row = {"class": CLASSES_18[cid]}
        for ct in CONF_THRS:
            for it in IOU_THRS:
                m, n = res["recall"][cid][ct][it]
                row[f"r_c{ct}_i{it}"] = round(m / n, 4) if n else None
                row[f"n_c{ct}_i{it}"] = n
        recall_rows.append(row)
    with (args.output_dir / "per_class_recall.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recall_rows[0].keys()))
        w.writeheader()
        w.writerows(recall_rows)

    conf_rows = [
        {"orig_class": CLASSES_18[a], "pred_class": (CLASSES_18[b] if b >= 0 else "无预测"), "count": n}
        for (a, b), n in sorted(res["confusion"].items(), key=lambda x: -x[1])
    ]
    with (args.output_dir / "confusion_pairs.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["orig_class", "pred_class", "count"])
        w.writeheader()
        w.writerows(conf_rows)

    worst = sorted(res["per_image"], key=lambda x: -x["score"])[:50]
    with (args.output_dir / "worst_images.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(worst[0].keys()))
        w.writeheader()
        w.writerows(worst)

    # 主口径汇总：conf=0.25, IoU=0.3
    main_recall = {}
    for cid in range(18):
        m, n = res["recall"][cid][0.25][0.3]
        if n:
            main_recall[CLASSES_18[cid]] = {"matched": m, "total": n, "recall": round(m / n, 4)}
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "weights": str(args.weights), "imgsz": args.imgsz, "conf_floor": args.conf_floor,
        "labeled_images": len(labeled),
        "main_protocol": "conf>=0.25, IoU>=0.3, 同类匹配",
        "recall_main": main_recall,
        "extra_pred_ge25_per_class": {CLASSES_18[k]: v for k, v in sorted(res["extra_pred"].items())},
        "note": "recall=原标签框被 ryw.pt 同类预测覆盖的比例；低 recall 类=标签可疑或教师盲区，需人工复核 worst_images.csv",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n主口径（conf≥0.25, IoU≥0.3）各类召回：")
    for name, d in main_recall.items():
        flag = " ⚠️" if d["recall"] < 0.5 else ""
        print(f"  {name:14s} {d['matched']:5d}/{d['total']:5d} = {d['recall']:.3f}{flag}")
    print(f"\n产物: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
