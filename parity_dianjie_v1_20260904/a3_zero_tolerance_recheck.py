# -*- coding: utf-8 -*-
"""A3 零容差复核：优化分支 ce2f166 提交后的部署配置（dianjie=ultralytics ONNX）
全量重跑 376 图，与 parity 评测候选臂输出逐图逐框比对。

期望：100% 一致（同一检测器构造路径/ResNet/规则层，CPU 推理确定性）。
不一致即说明部署配置与 parity 候选臂存在实现差异，切换不得放行。
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLOV5_AUTOINSTALL", "false")

WORKTREE = Path("C:/Users/23916/AppData/Local/Temp/jcw_dianjie_switch_wt")
PARITY_OUT = Path("E:/PythonProject/优化/parity_dianjie_v1_20260904/outputs")
VAL_LIST = Path("E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1/splits/folds/fold_01/val.txt")
TOL = 1e-3


def main() -> None:
    os.chdir(WORKTREE)
    sys.path.insert(0, str(WORKTREE))

    from inference import models as model_state
    from inference.detectors import dianjie_detect

    model_state.load_resnet()
    model_state.load_dianjie_detector()
    cfg = model_state.MODELS_CONFIG["detectors"]["dianjie"]
    print(f"deployed dianjie config: kind={cfg['kind']} weights={cfg['weights']} "
          f"threshold={cfg.get('threshold')} options={cfg.get('options')}", flush=True)
    assert cfg["kind"] == "ultralytics" and "yolo11m" in cfg["weights"], "models.yaml 未按 ce2f166 切换"

    # parity 候选臂基准
    base_rows = {r["image"]: r for r in csv.DictReader((PARITY_OUT / "per_image.csv").open(encoding="utf-8-sig"))}
    base_boxes: dict[str, list[tuple]] = {}
    with (PARITY_OUT / "per_box.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["arm"] == "candidate_onnx":
                base_boxes.setdefault(r["image"], []).append(
                    (int(r["label"]), float(r["cx"]), float(r["cy"]), float(r["w"]), float(r["h"])))

    images = [Path(x.strip()) for x in VAL_LIST.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert set(i.name for i in images) == set(base_rows), "评测清单与 parity 基准不一致"

    n_img_ok = n_box_ok = 0
    mismatches = []
    t0 = time.perf_counter()
    for idx, img in enumerate(images, 1):
        res = dianjie_detect(str(img))
        d = res.get("data") or {}
        base = base_rows[img.name]
        cur_boxes = sorted((int(b["label"]), b["cx"], b["cy"], b["w"], b["h"]) for b in (d.get("defects") or []))
        ref_boxes = sorted(base_boxes.get(img.name, []))

        img_ok = (
            res.get("code") == 1
            and d.get("type") is not None and int(d["type"]) == int(base["type_cand"])
            and d.get("nodule") is not None and int(d["nodule"]) == int(base["nodule_cand"])
            and len(cur_boxes) == len(ref_boxes)
            and all(a[0] == b[0] and all(abs(x - y) <= TOL for x, y in zip(a[1:], b[1:]))
                    for a, b in zip(cur_boxes, ref_boxes))
        )
        n_img_ok += int(img_ok)
        n_box_ok += len(cur_boxes)
        if not img_ok:
            mismatches.append({
                "image": img.name,
                "type": (base["type_cand"], d.get("type")), "nodule": (base["nodule_cand"], d.get("nodule")),
                "n_boxes": (len(ref_boxes), len(cur_boxes)),
            })
        if idx == 1 or idx % 50 == 0 or idx == len(images):
            print(f"  [{idx}/{len(images)}] ok={n_img_ok} mismatch={len(mismatches)}", flush=True)

    wall = time.perf_counter() - t0
    rate = n_img_ok / len(images)
    report = {
        "commit": "ce2f166", "n_images": len(images), "exact_match": n_img_ok,
        "match_rate": round(rate, 6), "n_boxes_verified": n_box_ok, "wall_s": round(wall, 1),
        "mismatches": mismatches[:20],
    }
    (PARITY_OUT / "a3_zero_tolerance_recheck.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                               encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False)[:600], flush=True)
    if rate != 1.0:
        raise SystemExit(f"零容差复核未通过：{len(mismatches)} 张不一致，切换不得放行")


if __name__ == "__main__":
    main()
