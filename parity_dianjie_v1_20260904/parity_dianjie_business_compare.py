# -*- coding: utf-8 -*-
"""电解镍（dianjie）检测器切换 parity 评测工具（路线 A0/A1/A2）。

同一批图，分别以「现役 ryw.pt 检测器」与「候选 YOLO11m ONNX 检测器」运行完整
`inference.detectors._dianjie_detect` 业务管线（含 ryw.pth ResNet、findcent、
面积占比与决策树），对比业务输出 type / nodule / defects 的一致率。

- 现役臂：models.yaml `detectors.dianjie` 原配置（kind yolov5 / weights/ryw.pt /
  threshold 0.25 / AutoShape 640, iou 0.45, max_det 1000）。
- 候选臂：模拟真实切换——kind ultralytics / 冻结 ONNX（SHA-256=901ccf60…）/
  threshold 0.25 / imgsz 640 / iou=适配器默认 0.7（models.yaml 切换不配 iou 时的真实行为）。
- 两臂共用同一 ResNet 与规则层；业务输出 defects 无置信度（生产契约即如此）。

只读数据集；输出写入本目录 outputs/。冻结 198 图 test 不读取。

用法（必须在能写 的目录运行；webapi_opt 由 git archive 优化分支 解包）:
  python parity_dianjie_business_compare.py --dry-run     # 4 张图冒烟
  python parity_dianjie_business_compare.py --limit 32
  python parity_dianjie_business_compare.py               # 全量 376 张
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("YOLOV5_AUTOINSTALL", "false")  # 禁止 torch.hub 自动改 venv

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SCRATCH = HERE / "webapi_opt"
OUT = HERE / "outputs"
VAL_LIST = Path("E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1/splits/folds/fold_01/val.txt")
EXPECTED_ONNX_SHA = "901ccf6006f44689483aaf6f0d1d1b8c18132a85c7d881be7e62fcdccc255b21"

RYW18 = {0: "wuzi", 1: "waterstain", 2: "yellowstain", 3: "nodule", 4: "airhole", 5: "bubble",
         6: "bump", 7: "fawu", 8: "overfire", 9: "NiOH", 10: "greenstain", 11: "suanwuran",
         12: "quejiao", 13: "watermark", 14: "youpao", 15: "heibian", 16: "mijixiaokeng", 17: "fanxitong"}
TYPE_NAMES = {0: "无板", 1: "三号板", 2: "烫洗板", 3: "合格整版", 4: "合格600", 5: "合格不足600", 6: "二号板"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def match_boxes(boxes_a: list[dict], boxes_b: list[dict], iou_thr: float = 0.5):
    """同类 IoU>=0.5 贪心一对一匹配。返回 (matched, only_a, only_b)。"""
    a = [dict(b, used=False) for b in boxes_a]
    b = [dict(d, used=False) for d in boxes_b]
    pairs = []
    for i, ba in enumerate(a):
        for j, bb in enumerate(b):
            if ba["label"] != bb["label"] or b[j]["used"]:
                continue
            iou = _iou(ba, bb)
            if iou >= iou_thr:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    matched = []
    for iou, i, j in pairs:
        if a[i]["used"] or b[j]["used"]:
            continue
        a[i]["used"] = b[j]["used"] = True
        matched.append((i, j, iou))
    return matched, [x for x in a if not x["used"]], [x for x in b if not x["used"]]


def _iou(ba: dict, bb: dict) -> float:
    ax1, ay1, ax2, ay2 = ba["cx"] - ba["w"] / 2, ba["cy"] - ba["h"] / 2, ba["cx"] + ba["w"] / 2, ba["cy"] + ba["h"] / 2
    bx1, by1, bx2, by2 = bb["cx"] - bb["w"] / 2, bb["cy"] - bb["h"] / 2, bb["cx"] + bb["w"] / 2, bb["cy"] + bb["h"] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = ba["w"] * ba["h"] + bb["w"] * bb["h"] - inter
    return inter / union if union > 0 else 0.0


def _imwrite_cn(path: Path, img: np.ndarray) -> None:
    """cv2.imwrite 在本环境对含中文路径时好时坏（2026-09-04 实测），改 imencode+write_bytes。"""
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise ValueError(f"imencode 失败: {path}")
    path.write_bytes(buf.tobytes())


def render_panel(image_path: Path, defects: list[dict], title: str, max_width: int = 1400) -> np.ndarray:
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    scale = min(1.0, max_width / float(w))
    panel = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA) if scale != 1.0 else image.copy()
    header = 46
    canvas = np.full((panel.shape[0] + header, panel.shape[1], 3), 245, dtype=np.uint8)
    canvas[header:] = panel
    cv2.putText(canvas, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    def _class_color(cid: int) -> tuple:
        import colorsys
        r, g, b = colorsys.hsv_to_rgb((cid * 0.618033988749895) % 1.0, 0.85, 0.95)
        return (int(b * 255), int(g * 255), int(r * 255))  # 18 类黄金比例色相，避免 overfire/wuzi 同色误读

    for d in defects:
        x1 = int((d["cx"] - d["w"] / 2) * scale)
        y1 = int((d["cy"] - d["h"] / 2) * scale) + header
        x2 = int((d["cx"] + d["w"] / 2) * scale)
        y2 = int((d["cy"] + d["h"] / 2) * scale) + header
        c = _class_color(d["label"])
        cv2.rectangle(canvas, (max(0, x1), max(header, y1)), (min(canvas.shape[1] - 1, x2), min(canvas.shape[0] - 1, y2)), c, 3, cv2.LINE_AA)
        tag = f"{d['label']}:{RYW18.get(d['label'], '?')}"
        cv2.putText(canvas, tag, (max(0, x1), max(header + 16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="只跑前 4 张")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（0=全量）")
    ap.add_argument("--stride", type=int, default=1, help="每 N 张取 1 张")
    ap.add_argument("--val-list", type=Path, default=VAL_LIST)
    ap.add_argument("--top-review", type=int, default=20)
    args = ap.parse_args()

    if not SCRATCH.is_dir():
        raise FileNotFoundError(f"未找到优化分支解包目录: {SCRATCH}（先 git archive 优化分支）")
    os.chdir(SCRATCH)  # model_decryptor 的 torch.hub.load('.') 依赖 CWD
    if str(SCRATCH) not in sys.path:
        sys.path.insert(0, str(SCRATCH))

    onnx_path = SCRATCH / "weights" / "dianjienie_yolo11m_ryw18_640.onnx"
    onnx_sha = sha256(onnx_path)
    if onnx_sha != EXPECTED_ONNX_SHA:
        raise RuntimeError(f"ONNX SHA-256 与部署契约不一致: {onnx_sha}")
    ryw_sha = sha256(SCRATCH / "weights" / "ryw.pt")
    rywpth_sha = sha256(SCRATCH / "weights" / "ryw.pth")

    OUT.mkdir(exist_ok=True)
    image_paths = [Path(x.strip()) for x in args.val_list.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.dry_run:
        image_paths = image_paths[:4]
    elif args.limit:
        image_paths = image_paths[: args.limit]
    if args.stride > 1:
        image_paths = image_paths[:: args.stride]
    print(f"[1/5] 评测集 {len(image_paths)} 张", flush=True)

    print("[2/5] 加载共享 ResNet + 现役检测器 + 候选 ONNX ...", flush=True)
    from inference import models as model_state
    from inference.detectors import dianjie_detect
    from adapters.ultralytics import UltralyticsDetector

    model_state.load_resnet()           # 共享：ryw.pth bad/good
    model_state.load_dianjie_detector()  # 现役臂：models.yaml dianjie = yolov5/ryw.pt
    current_detector = model_state.yolo
    candidate_detector = UltralyticsDetector(
        weights=str(onnx_path), device="cpu", threshold=0.25, imgsz=640)  # iou=适配器默认 0.7=真实切换行为
    assert model_state.resnet is not None and current_detector is not None

    rows: list[dict] = []
    box_rows: list[dict] = []
    detections: dict[str, tuple] = {}
    t_all = time.perf_counter()
    fail_cur = fail_cand = 0
    for idx, img in enumerate(image_paths, 1):
        rec = {"image": img.name, "image_path": str(img)}

        t0 = time.perf_counter()
        r_cur = dianjie_detect(str(img))
        t_cur = time.perf_counter() - t0

        model_state.yolo = candidate_detector
        try:
            t0 = time.perf_counter()
            r_cand = dianjie_detect(str(img))
            t_cand = time.perf_counter() - t0
        finally:
            model_state.yolo = current_detector

        d_cur = r_cur.get("data") or {}
        d_cand = r_cand.get("data") or {}
        ok_cur, ok_cand = r_cur.get("code") == 1, r_cand.get("code") == 1
        fail_cur += 0 if ok_cur else 1
        fail_cand += 0 if ok_cand else 1
        defects_cur = d_cur.get("defects") or []
        defects_cand = d_cand.get("defects") or []
        matched, only_cur, only_cand = match_boxes(defects_cur, defects_cand)

        rec.update({
            "ok_cur": int(ok_cur), "ok_cand": int(ok_cand),
            "type_cur": d_cur.get("type"), "type_cand": d_cand.get("type"),
            "nodule_cur": d_cur.get("nodule"), "nodule_cand": d_cand.get("nodule"),
            "type_match": int(ok_cur and ok_cand and d_cur.get("type") == d_cand.get("type")),
            "nodule_match": int(ok_cur and ok_cand and d_cur.get("nodule") == d_cand.get("nodule")),
            "defects_cur": len(defects_cur), "defects_cand": len(defects_cand),
            "matched": len(matched), "only_cur": len(only_cur), "only_cand": len(only_cand),
            "t_cur_s": round(t_cur, 3), "t_cand_s": round(t_cand, 3),
        })
        rows.append(rec)
        detections[img.name] = (img, defects_cur, defects_cand, rec)
        for d in defects_cur:
            box_rows.append({"arm": "current_ryw", "image": img.name, **{k: d[k] for k in ("label", "cx", "cy", "w", "h")}})
        for d in defects_cand:
            box_rows.append({"arm": "candidate_onnx", "image": img.name, **{k: d[k] for k in ("label", "cx", "cy", "w", "h")}})
        if idx == 1 or idx % 10 == 0 or idx == len(image_paths):
            print(f"  [{idx}/{len(image_paths)}] {img.name} type {rec['type_cur']}→{rec['type_cand']} "
                  f"defects {rec['defects_cur']}→{rec['defects_cand']}", flush=True)

    wall = time.perf_counter() - t_all
    valid = [r for r in rows if r["ok_cur"] and r["ok_cand"]]
    n = len(valid)
    type_rate = sum(r["type_match"] for r in valid) / n if n else 0.0
    nodule_rate = sum(r["nodule_match"] for r in valid) / n if n else 0.0
    both_rate = sum(r["type_match"] and r["nodule_match"] for r in valid) / n if n else 0.0

    print("[3/5] 写 CSV 与逐框审计 ...", flush=True)
    img_fields = ["image", "image_path", "ok_cur", "ok_cand", "type_cur", "type_cand", "nodule_cur", "nodule_cand",
                  "type_match", "nodule_match", "defects_cur", "defects_cand", "matched", "only_cur", "only_cand",
                  "t_cur_s", "t_cand_s"]
    with (OUT / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=img_fields, extrasaction="ignore")
        wcsv.writeheader()
        wcsv.writerows(rows)
    with (OUT / "per_box.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["arm", "image", "label", "cx", "cy", "w", "h"], extrasaction="ignore")
        wcsv.writeheader()
        wcsv.writerows(box_rows)
    confusion = Counter((r["type_cur"], r["type_cand"]) for r in valid)
    (OUT / "type_confusion.json").write_text(json.dumps({f"{a}->{b}": c for (a, b), c in sorted(confusion.items())},
                                                        ensure_ascii=False, indent=2), encoding="utf-8")

    print("[4/5] 渲染 review.html ...", flush=True)
    def rank_key(rec):
        return (0 if not (rec["type_match"] and rec["nodule_match"]) else 1,
                -(rec["only_cur"] + rec["only_cand"]), rec["image"])
    top = sorted(rows, key=rank_key)[: args.top_review]
    review_dir = OUT / "review_images"
    review_dir.mkdir(exist_ok=True)
    cards = []
    for rank, rec in enumerate(top, 1):
        img, d_cur, d_cand, _ = detections[rec["image"]]
        name = f"{rank:02d}_{rec['image']}.png"
        left = render_panel(img, d_cur, f"CURRENT ryw.pt | type={rec['type_cur']} nodule={rec['nodule_cur']} | defects={rec['defects_cur']}")
        right = render_panel(img, d_cand, f"CANDIDATE y11m ONNX | type={rec['type_cand']} nodule={rec['nodule_cand']} | defects={rec['defects_cand']}")
        hmax = max(left.shape[0], right.shape[0])
        if left.shape[0] != hmax:
            left = cv2.copyMakeBorder(left, 0, hmax - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245,) * 3)
        if right.shape[0] != hmax:
            right = cv2.copyMakeBorder(right, 0, hmax - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245,) * 3)
        _imwrite_cn(review_dir / name, cv2.hconcat([left, right]))
        mismatch = "TYPE/NODULE MISMATCH" if not (rec["type_match"] and rec["nodule_match"]) else "box divergence"
        cards.append(f"<article class='card'><h2>#{rank:02d} {html.escape(rec['image'])}</h2>"
                     f"<div class='metrics'>{mismatch}; type {rec['type_cur']}→{rec['type_cand']}; "
                     f"nodule {rec['nodule_cur']}→{rec['nodule_cand']}; defects {rec['defects_cur']}→{rec['defects_cand']} "
                     f"(matched {rec['matched']}, only_cur {rec['only_cur']}, only_cand {rec['only_cand']})</div>"
                     f"<img src='review_images/{name}'></article>")
    (OUT / "review.html").write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>dianjie parity review</title>"
        "<style>body{font-family:'Microsoft YaHei',sans-serif;background:#f2eee6;margin:0}"
        "header{padding:18px 32px;background:#fffdf8;border-bottom:1px solid #d8d0c2}"
        "main{max-width:1900px;margin:0 auto;padding:20px;display:grid;gap:18px}"
        ".card{background:#fffdf8;border:1px solid #d8d0c2;border-radius:10px;padding:12px}"
        ".metrics{color:#5b6965;font-size:13px;margin-bottom:8px}"
        ".card img{width:100%;border:1px solid #cfc7b9}</style></head><body>"
        f"<header><h1>电解镍检测器切换 parity 审查（top {len(top)}）</h1>"
        f"<p>左=现役 ryw.pt，右=候选 YOLO11m ONNX；两臂同一 ResNet/规则层。全量 {len(rows)} 图。</p></header>"
        f"<main>{''.join(cards)}</main></body></html>", encoding="utf-8")

    print("[5/5] report.md ...", flush=True)
    cls_cur: Counter = Counter(); cls_cand: Counter = Counter(); cls_cur_only: Counter = Counter(); cls_cand_only: Counter = Counter()
    for r in valid:
        img, d_cur, d_cand, _ = detections[r["image"]]
        m, oc, oc2 = match_boxes(d_cur, d_cand)
        for _, j, _iou_v in m:
            cls_cand[d_cand[j]["label"]] += 1
        for x in oc:
            cls_cur_only[x["label"]] += 1
        for x in oc2:
            cls_cand_only[x["label"]] += 1
    by_class_cur = Counter(b["label"] for b in box_rows if b["arm"] == "current_ryw")
    by_class_cand = Counter(b["label"] for b in box_rows if b["arm"] == "candidate_onnx")

    if both_rate >= 0.95:
        gate = "≥95%：达标，可提请用户裁决切换（parity 0 容差复跑 + 三冒烟）"
    elif both_rate >= 0.85:
        gate = "85%~95%：输出分歧主因分析后再决定（微调规则阈值或转路线 C 蒸馏）"
    else:
        gate = "<85%：不建议直接切换；转路线 C 蒸馏或维持现状"

    lines = [
        "# 电解镍检测器切换 parity 评测（路线 A2）",
        "",
        f"- 运行时间：{datetime.now().isoformat(timespec='seconds')}；总墙钟 {wall:.0f}s；评测 {len(rows)} 图（有效 {n}，"
        f"现役臂失败 {fail_cur}，候选臂失败 {fail_cand}）。",
        f"- 现役臂：models.yaml `detectors.dianjie`（yolov5 / weights/ryw.pt / 0.25 / AutoShape 640, iou 0.45, max_det 1000），"
        f"ryw.pt SHA-256=`{ryw_sha[:16]}…`。",
        f"- 候选臂：ultralytics / 冻结 ONNX（SHA-256=`{onnx_sha[:16]}…` 校验一致）/ 0.25 / imgsz 640 / **iou=0.7（适配器默认，"
        "即 models.yaml 切换不配 iou 时的真实行为）**。",
        f"- 共享：ryw.pth ResNet（SHA-256=`{rywpth_sha[:16]}…`）与规则层完全一致；差异只来自检测器。",
        f"- 图像集：anchor_v1 fold_01 val {len(image_paths)} 张（冻结 198 图 test 未读取）。注意两权重都可能见过该批图，"
        "parity 量的是业务输出一致性，不是泛化精度。",
        "",
        "## 一致率与闸门",
        "",
        f"| 指标 | 一致率 |",
        f"| --- | --- |",
        f"| type 一致 | {type_rate:.4f}（{sum(r['type_match'] for r in valid)}/{n}） |",
        f"| nodule 一致 | {nodule_rate:.4f}（{sum(r['nodule_match'] for r in valid)}/{n}） |",
        f"| type+nodule 同时一致 | **{both_rate:.4f}** |",
        "",
        f"**闸门判定（路线 A2）：{gate}**",
        "",
        "## type 混淆（现役 → 候选）",
        "",
        "| 现役 → 候选 | 张数 |",
        "| --- | --- |",
    ]
    for (a, b), c in sorted(confusion.items(), key=lambda kv: -kv[1]):
        flag = "" if a == b else "  ← 不一致"
        lines.append(f"| {a}({TYPE_NAMES.get(a)}) → {b}({TYPE_NAMES.get(b)}) | {c}{flag} |")
    lines += [
        "",
        "## 框级差异（业务 defects 列表，同类 IoU≥0.5 一对一匹配）",
        "",
        f"- 现役总框 {sum(by_class_cur.values())}，候选总框 {sum(by_class_cand.values())}；"
        f"匹配 {sum(cls_cand.values())}，仅现役 {sum(cls_cur_only.values())}，仅候选 {sum(cls_cand_only.values())}。",
        "",
        "| cls | 名称 | 现役框 | 候选框 | 仅现役 | 仅候选 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in range(18):
        lines.append(f"| {c} | {RYW18[c]} | {by_class_cur.get(c, 0)} | {by_class_cand.get(c, 0)} | "
                     f"{cls_cur_only.get(c, 0)} | {cls_cand_only.get(c, 0)} |")
    t_cur = [r["t_cur_s"] for r in rows]
    t_cand = [r["t_cand_s"] for r in rows]
    lines += [
        "",
        "## 耗时（CPU，单图全管线均值）",
        "",
        f"- 现役臂 {np.mean(t_cur):.2f}s，候选臂 {np.mean(t_cand):.2f}s。",
        "",
        "## 限制",
        "",
        "- 业务 defects 列表无置信度（生产契约如此）；结瘤面积/长宽比过滤后的框会从 defects 中消失（规则行为，两臂一致）。",
        "- parity 高一致不代表两臂都正确——它只回答「切换是否改变业务输出」。正确性以人工审查 review.html 为准。",
        "- 评测图为开发集图片（两权重可能见过），结论用于切换决策，不得外推为泛化指标。",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "val_list": str(args.val_list), "n_images": len(image_paths),
        "images": [p.name for p in image_paths],
        "weights": {"ryw.pt": ryw_sha, "dianjienie_yolo11m_ryw18_640.onnx": onnx_sha, "ryw.pth": rywpth_sha},
        "arms": {"current": "models.yaml dianjie (yolov5/ryw.pt/0.25/AutoShape640/iou0.45/maxdet1000)",
                 "candidate": "ultralytics/onnx/0.25/640/iou0.7(adapter default)"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE both_rate={both_rate:.4f} type_rate={type_rate:.4f} nodule_rate={nodule_rate:.4f} wall={wall:.0f}s", flush=True)
    print(f"report -> {OUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
