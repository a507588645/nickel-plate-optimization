# -*- coding: utf-8 -*-
"""overfire（cls 8）逐类置信度阈值分析：为候选 ONNX 臂选一个能杀误检、保真检的阈值。

数据源（均为已落盘产物，只读）：
- parity per_box.csv / per_image.csv：候选臂业务 overfire 框（无 conf）
- 同口径复评 preds_y11m_onnx_eval.jsonl：候选 ONNX conf=0.001/iou=0.7 全量预测池（取 conf）
- anchor_v1 fold_01 val 标签：判定业务 overfire 框是 TP（IoU>=0.5 命中 GT overfire）还是 FP

输出：阈值扫描表 + top20 审查图中 overfire 框的明细（回应用户对 #7/#20 的裁决）。
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
JSONL = Path("E:/PythonProject/优化/ryw_vs_y11m_same_protocol_20260904/outputs/preds_y11m_onnx_eval.jsonl")
LABELS_DIR = Path("E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1/labels")
OVERFIRE = 8


def iou_xyxy(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def to_xyxy(d: dict):
    return (d["cx"] - d["w"] / 2, d["cy"] - d["h"] / 2, d["cx"] + d["w"] / 2, d["cy"] + d["h"] / 2)


def main() -> None:
    # 1) 还原 top20 排名，定位用户裁决的 #7/#10/#16/#20
    rows = list(csv.DictReader((OUT / "per_image.csv").open(encoding="utf-8-sig")))
    for r in rows:
        for k in ("ok_cur", "ok_cand", "type_match", "nodule_match"):
            r[k] = int(r[k])
        for k in ("type_cur", "type_cand", "nodule_cur", "nodule_cand", "defects_cur", "defects_cand",
                  "matched", "only_cur", "only_cand"):
            r[k] = int(r[k])
    ranked = sorted(rows, key=lambda r: (0 if (r["type_match"] and r["nodule_match"]) else 1,
                                         -(r["only_cur"] + r["only_cand"]), r["image"]))
    flagged = {}
    for rank, r in enumerate(ranked[:20], 1):
        flagged[rank] = r["image"]
        if rank in (7, 10, 16, 20):
            print(f"#{rank:02d} = {r['image']}  type {r['type_cur']}→{r['type_cand']} "
                  f"defects {r['defects_cur']}→{r['defects_cand']} (only_cur {r['only_cur']}, only_cand {r['only_cand']})")

    # 2) 候选预测池（取 conf）
    pool: dict[str, list] = {}
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        pool[Path(rec["img"]).name] = [b for b in rec["boxes"] if b[0] == OVERFIRE]  # [cls, conf, x1,y1,x2,y2]

    # 3) GT overfire 框
    gt_cache: dict[str, list] = {}
    def gt_overfire(image_name: str) -> list:
        if image_name not in gt_cache:
            lbl = LABELS_DIR / (Path(image_name).stem + ".txt")
            boxes = []
            if lbl.exists():
                img_path = Path("E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1/images") / image_name
                with Image.open(img_path) as im:
                    w, h = im.size
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    p = line.split()
                    if len(p) != 5 or int(p[0]) != OVERFIRE:
                        continue
                    _, xc, yc, bw, bh = map(float, p)
                    boxes.append(((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
            gt_cache[image_name] = boxes
        return gt_cache[image_name]

    # 4) 候选臂业务 overfire 框 → conf + TP/FP
    items = []  # (image, conf, is_tp)
    unmatched = 0
    with (OUT / "per_box.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["arm"] != "candidate_onnx" or int(r["label"]) != OVERFIRE:
                continue
            xyxy = (float(r["cx"]) - float(r["w"]) / 2, float(r["cy"]) - float(r["h"]) / 2,
                    float(r["cx"]) + float(r["w"]) / 2, float(r["cy"]) + float(r["h"]) / 2)
            best, best_iou = None, 0.0
            for b in pool.get(r["image"], []):
                iou = iou_xyxy(xyxy, b[2:6])
                if iou > best_iou:
                    best_iou, best = iou, b[1]
            if best is None:
                unmatched += 1
                continue
            is_tp = any(iou_xyxy(xyxy, g) >= 0.5 for g in gt_overfire(r["image"]))
            items.append((r["image"], float(best), is_tp))

    confs = np.array([c for _, c, _ in items])
    tps = np.array([t for _, _, t in items])
    print(f"\n候选臂业务 overfire 框：{len(items)}（TP {int(tps.sum())} / FP {int((~tps).sum())}；"
          f"预测池未匹配 {unmatched}）")
    print(f"conf 分位：p10={np.percentile(confs,10):.3f} p25={np.percentile(confs,25):.3f} "
          f"p50={np.percentile(confs,50):.3f} p75={np.percentile(confs,75):.3f}")
    print(f"FP conf 分位：p50={np.percentile(confs[~tps],50):.3f} p90={np.percentile(confs[~tps],90):.3f}"
          if (~tps).any() else "无 FP")

    print("\n| 阈值 T | 保留框 | 保留TP | 杀掉FP | FP清除率 | TP损失 |")
    print("| --- | --- | --- | --- | --- | --- |")
    n_fp = int((~tps).sum())
    for T in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        keep = confs >= T
        kept_tp = int(tps[keep].sum())
        killed_fp = n_fp - int((~tps[keep]).sum())
        tp_lost = int(tps.sum()) - kept_tp
        print(f"| {T:.2f} | {int(keep.sum())} | {kept_tp} | {killed_fp} | "
              f"{killed_fp / n_fp:.1%} | {tp_lost} |")

    print("\n用户裁决图上的候选 overfire 框明细：")
    for rank in (7, 10, 16, 20):
        img = flagged[rank]
        boxes = [(c, t) for im, c, t in items if im == img]
        print(f"  #{rank:02d} {img}: {[(round(c,3), 'TP' if t else 'FP') for c, t in boxes] or '无 overfire 框'}")

    (OUT / "overfire_threshold_analysis.json").write_text(json.dumps({
        "n_business_overfire": len(items), "n_tp": int(tps.sum()), "n_fp": n_fp, "unmatched": unmatched,
        "sweep": {f"{T:.2f}": {"keep": int((confs >= T).sum()), "tp": int(tps[confs >= T].sum()),
                              "fp_killed": n_fp - int((~tps[confs >= T]).sum())}
                  for T in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)},
        "flagged": {f"#{k:02d}": flagged[k] for k in (7, 10, 16, 20)},
    }, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {OUT / 'overfire_threshold_analysis.json'}")


if __name__ == "__main__":
    main()
