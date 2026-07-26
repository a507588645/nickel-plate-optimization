#!/usr/bin/env python3
"""
L1：用 ryw.pt 为 union 中 1810 张无标签图生成候选伪标签（非真值）。

规范（AGENTS.md §6）：
- 六列（class cx cy w h confidence），conf 下限 0.15；
- 写入版本化目录 `优化\\label_check_ryw_v1\\pseudo_candidates\\labels_with_conf\\`，
  **不进数据集目录**；
- 三态统计 accepted(≥0.70)/uncertain(0.40~0.70)/rejected(0.15~0.40)；
- 训练使用前必须人工审查并另行生成五列标签。
- 已知盲区（label_check_ryw_v1/report.md）：yellowstain/fawu/NiOH/heibian/mijixiaokeng/overfire/greenstain
  该模型无法可靠生成，这些类需人工标注。

运行环境：JinchuanYolov5sTrain venv。
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
OUT_DEFAULT = Path(__file__).resolve().parent / "label_check_ryw_v1" / "pseudo_candidates"
DEFAULT_WEIGHTS = Path(r"E:/PythonProject/Tools/ModelAutoLabelTool/weights/yolov5/ryw.pt")
DEFAULT_REPO = Path(r"E:/PythonProject/ImetricCodes/JinchuanYolov5sTrain")

CLASSES_18 = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]
ACC_T, UNC_T, REJ_T = 0.70, 0.40, 0.15


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="union 无标签图 ryw.pt 候选标签生成（非真值）",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--union", type=Path, default=UNION)
    p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--yolov5-repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf-floor", type=float, default=0.15)
    p.add_argument("--iou-nms", type=float, default=0.45)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    labels_dir = out / "labels_with_conf"
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader((args.union / "manifests" / "samples.csv").open(encoding="utf-8-sig")))
    img_index = {p.stem: p for p in (args.union / "images").iterdir()}
    targets = [r for r in rows if r.get("has_label") != "True"]
    if args.limit:
        targets = targets[: args.limit]
    print(f"=== 无标签图候选标签生成 ===  targets={len(targets)}")
    if args.dry_run:
        return 0

    import torch

    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": k.get("weights_only", False)})
    device = "0" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"
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

    import cv2

    state_box = defaultdict(int)
    state_cls = {s: defaultdict(int) for s in ("accepted", "uncertain", "rejected")}
    per_image = []
    t0 = time.time()
    for i, r in enumerate(targets):
        ip = img_index.get(r["union_name"])
        img = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR) if ip else None
        if img is None:
            per_image.append({"union_name": r["union_name"], "n_boxes": -1})
            continue
        H, W = img.shape[:2]
        res = model(img, size=args.imgsz)
        n = 0
        lines = []
        for *xyxy, conf, cls in res.xyxy[0].tolist():
            x1, y1, x2, y2 = xyxy
            cx, cy = ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {float(conf):.4f}")
            st = "accepted" if conf >= ACC_T else ("uncertain" if conf >= UNC_T else "rejected")
            state_box[st] += 1
            state_cls[st][int(cls)] += 1
            n += 1
        if lines:
            (labels_dir / f"{r['union_name']}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        per_image.append({"union_name": r["union_name"], "n_boxes": n})
        if (i + 1) % 200 == 0 or (i + 1) == len(targets):
            print(f"  [{i+1}/{len(targets)}] {(i+1)/(time.time()-t0):.1f} img/s")

    with (out / "per_image_counts.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["union_name", "n_boxes"])
        w.writeheader()
        w.writerows(per_image)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "weights": str(args.weights), "imgsz": args.imgsz,
        "conf_floor": args.conf_floor,
        "thresholds": {"accepted": ACC_T, "uncertain": UNC_T, "rejected": REJ_T},
        "images_target": len(targets),
        "images_with_boxes": sum(1 for p in per_image if p["n_boxes"] > 0),
        "images_empty": sum(1 for p in per_image if p["n_boxes"] == 0),
        "images_unreadable": sum(1 for p in per_image if p["n_boxes"] < 0),
        "state_box_counts": dict(state_box),
        "state_class_counts": {s: {CLASSES_18[k]: v for k, v in sorted(d.items())} for s, d in state_cls.items()},
        "known_blind_classes": ["yellowstain", "fawu", "NiOH", "heibian", "mijixiaokeng", "overfire", "greenstain"],
        "status": "candidate_only_requires_human_review",
        "note": "六列候选标签，非真值；训练前必须人工审查并另生成五列标签；盲区类需人工标注",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# pseudo_candidates（ryw.pt 候选伪标签，非真值）\n\n"
        f"- 生成：{datetime.now():%Y-%m-%d %H:%M}，ryw.pt imgsz={args.imgsz} conf≥{args.conf_floor}\n"
        "- `labels_with_conf/` 六列（class cx cy w h confidence）\n"
        "- 三态：accepted≥0.70 / uncertain 0.40~0.70 / rejected 0.15~0.40，见 summary.json\n"
        "- **禁止直接用于训练**；人工审查后另行生成五列标签（AGENTS.md §6）\n"
        "- 盲区类（yellowstain/fawu/NiOH/heibian/mijixiaokeng/overfire/greenstain）需人工标注\n",
        encoding="utf-8",
    )
    print(f"三态框数: {dict(state_box)}")
    print(f"产物: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
