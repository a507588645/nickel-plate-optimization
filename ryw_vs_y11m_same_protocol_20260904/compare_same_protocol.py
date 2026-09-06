# -*- coding: utf-8 -*-
"""ryw.pt vs 部署 YOLO11m ONNX 同口径复评对照工具。

同一评测集：dianjienie_ryw18_anchor_v1 fold_01 val（376 图，历史人工映射标签，RYW18 类别）。
同一把尺子：imgsz=640 letterbox、NMS IoU=0.7、conf=0.001、max_det=300、
标准 mAP@0.5 / mAP@0.5:0.95（ultralytics ap_per_class + 复刻 match_predictions 贪心匹配）。
另报生产运行点（conf=0.25 / iou=0.45）的框数与 P/R@0.5。

仅只读数据集；全部输出写入本目录 outputs/。冻结 198 图 test 不读取。

用法:
  python compare_same_protocol.py --dry-run          # 4 张图端到端冒烟
  python compare_same_protocol.py --limit 32         # 前 32 张
  python compare_same_protocol.py                    # 全量 376 张（含原生 val 交叉核对）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("YOLOV5_AUTOINSTALL", "false")  # 禁止 torch.hub 自动改 venv

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
YOLOV5_REPO = Path("E:/PythonProject/ImetricCodes/JinchuanYolov5sTrain")
RYW_PT = Path("E:/PythonProject/ImetricCodes/JinchuanPythonWebapi/ryw.pt")
DEPLOY_ONNX = HERE / "deployed_dianjienie_y11m_640.onnx"
FOLD01_PT = Path(
    "E:/PythonProject/models/runs/dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_anchorv1_fold01/weights/best.pt"
)
DATA_YAML = Path("E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1/data.yaml")
VAL_LIST = Path("E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1/splits/folds/fold_01/val.txt")

CONF_EVAL, IOU_EVAL, MAX_DET_EVAL = 0.001, 0.70, 300
CONF_PROD, IOU_PROD = 0.25, 0.45
MAX_DET_PROD = {"ryw": 1000, "y11m_onnx": 300}  # 与部署配置一致
IOUV = torch.linspace(0.5, 0.95, 10)

NAMES = {
    0: "wuzi", 1: "waterstain", 2: "yellowstain", 3: "nodule", 4: "airhole",
    5: "bubble", 6: "bump", 7: "fawu", 8: "overfire", 9: "NiOH",
    10: "greenstain", 11: "suanwuran", 12: "quejiao", 13: "watermark",
    14: "youpao", 15: "heibian", 16: "mijixiaokeng", 17: "fanxitong",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ground_truth(image_paths: list[Path]) -> dict[str, list[tuple[int, np.ndarray]]]:
    """返回 {image_path: [(cls, xyxy_abs), ...]}；标签为 YOLO 归一化 xcycwh。"""
    from PIL import Image

    gt: dict[str, list[tuple[int, np.ndarray]]] = {}
    for img_path in image_paths:
        lbl = Path(str(img_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        with Image.open(img_path) as im:
            w, h = im.size
        boxes = []
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                c, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
                x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
                boxes.append((c, np.array([x1, y1, x2, y2], dtype=np.float32)))
        gt[str(img_path)] = boxes
    return gt


def match_predictions(pred_classes: torch.Tensor, true_classes: torch.Tensor,
                      iou: torch.Tensor, iouv: torch.Tensor) -> torch.Tensor:
    """复刻 ultralytics DetectionValidator.match_predictions（8.4.66）的贪心匹配。"""
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes
    iou_np = (iou * correct_class).cpu().numpy()
    for i, threshold in enumerate(iouv.tolist()):
        matches = np.nonzero(iou_np >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool)


def evaluate_preds(per_image_preds: dict[str, list[tuple[int, float, np.ndarray]]],
                   gt: dict, names: dict) -> dict:
    """my-ruler 标准评测：mAP / 逐类 AP / maxF1 P-R / 生产点统计。"""
    from ultralytics.utils.metrics import ap_per_class, box_iou

    tp_all, conf_all, pcls_all, tcls_all = [], [], [], []
    per_image_prod: dict[str, dict] = {}
    for img, boxes in per_image_preds.items():
        g = gt[img]
        gt_cls = torch.tensor([c for c, _ in g], dtype=torch.float32)
        gt_xy = torch.tensor(np.stack([b for _, b in g])) if g else torch.zeros((0, 4))
        pred_cls = torch.tensor([c for c, _, _ in boxes], dtype=torch.float32)
        pred_conf = torch.tensor([cf for _, cf, _ in boxes], dtype=torch.float32)
        pred_xy = torch.tensor(np.stack([b for _, _, b in boxes])) if boxes else torch.zeros((0, 4))
        tcls_all.extend(int(c) for c, _ in g)

        if pred_xy.shape[0] and gt_xy.shape[0]:
            iou = box_iou(gt_xy, pred_xy)
            tp_all.append(match_predictions(pred_cls, gt_cls, iou, IOUV))
            conf_all.append(pred_conf)
            pcls_all.append(pred_cls)

        # 生产运行点（conf>=0.25）统计，IoU=0.5 一对一匹配
        keep = [k for k, (_, cf, _) in enumerate(boxes) if cf >= CONF_PROD]
        ntp = 0
        tp_flags = []
        if keep and gt_xy.shape[0]:
            sub_cls = pred_cls[keep]
            sub_xy = pred_xy[keep]
            tp50 = match_predictions(sub_cls, gt_cls, box_iou(gt_xy, sub_xy), torch.tensor([0.5]))[:, 0]
            tp_flags = tp50.tolist()
            ntp = int(tp50.sum())
        per_image_prod[img] = {
            "keep": keep,
            "tp_flags": tp_flags,
            "n_gt": int(gt_xy.shape[0]),
        }

    tp_cat = torch.cat(tp_all).numpy() if tp_all else np.zeros((0, 10))
    conf_cat = torch.cat(conf_all).numpy() if conf_all else np.zeros((0,))
    pcls_cat = torch.cat(pcls_all).numpy() if pcls_all else np.zeros((0,))
    tcls_arr = np.array(tcls_all)

    tp, fp, p, r, f1, ap, ap_cls, p_curve, r_curve, f1_curve, _x, _pv = ap_per_class(
        tp_cat, conf_cat, pcls_cat, tcls_arr, plot=False, names=names)
    # 8.4.66：p/r/f1 已是各类在 max-F1 点的一维取值
    out = {
        "mAP50": float(ap[:, 0].mean()),
        "mAP50-95": float(ap.mean()),
        "P_at_maxF1": float(p.mean()),
        "R_at_maxF1": float(r.mean()),
        "F1_at_maxF1": float(f1.mean()),
        "n_images": len(per_image_preds),
        "n_gt_boxes": int(tcls_arr.size),
        "n_pred_boxes_evalconf": int(conf_cat.size),
        "per_class": {},
    }

    gt_cls_count: dict[int, int] = {}
    for g in gt.values():
        for c, _ in g:
            gt_cls_count[c] = gt_cls_count.get(c, 0) + 1

    tp_by_cls: dict[int, int] = {}
    fp_by_cls: dict[int, int] = {}
    pred_by_cls: dict[int, int] = {}
    total = {"pred": 0, "tp": 0, "fp": 0}
    for img, s in per_image_prod.items():
        boxes = per_image_preds[img]
        for j, k in enumerate(s["keep"]):
            c = int(boxes[k][0])
            pred_by_cls[c] = pred_by_cls.get(c, 0) + 1
            total["pred"] += 1
            if j < len(s["tp_flags"]) and s["tp_flags"][j]:
                tp_by_cls[c] = tp_by_cls.get(c, 0) + 1
                total["tp"] += 1
            else:
                fp_by_cls[c] = fp_by_cls.get(c, 0) + 1
                total["fp"] += 1

    for c in sorted(set(list(gt_cls_count) + list(pred_by_cls))):
        pc = out["per_class"].setdefault(c, {})
        pc["name"] = NAMES.get(c, str(c))
        pc["gt"] = gt_cls_count.get(c, 0)
        pc["pred@0.25"] = pred_by_cls.get(c, 0)
        pc["tp@0.5"] = tp_by_cls.get(c, 0)
        pc["fp@0.5"] = fp_by_cls.get(c, 0)
        pc["recall@0.5"] = round(tp_by_cls.get(c, 0) / gt_cls_count[c], 5) if gt_cls_count.get(c) else None
    out["total@0.25"] = {
        "n_pred": total["pred"],
        "n_tp@0.5": total["tp"],
        "n_fp@0.5": total["fp"],
        "precision@0.5": round(total["tp"] / total["pred"], 5) if total["pred"] else None,
        "recall@0.5": round(total["tp"] / out["n_gt_boxes"], 5) if out["n_gt_boxes"] else None,
    }
    for row, c in zip(ap, ap_cls):
        out["per_class"][int(c)]["AP50"] = float(row[0])
        out["per_class"][int(c)]["AP50-95"] = float(row.mean())
    return out


def set_autoshape_params(model, conf: float, iou: float, max_det: int) -> None:
    for name, val in (("conf", conf), ("iou", iou), ("max_det", max_det),
                      ("multi_label", False), ("agnostic", False)):
        if hasattr(model, name):
            setattr(model, name, val)


def run_ryw(image_paths: list[Path], conf: float, iou: float, max_det: int) -> dict:
    orig_load = torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)  # 本地受信生产权重（sha256 已登记），绕过 torch2.12 默认
        return orig_load(*a, **k)

    torch.load = _load
    try:
        model = torch.hub.load(str(YOLOV5_REPO), "custom", path=str(RYW_PT), source="local",
                               device="cpu", _verbose=False)
    finally:
        torch.load = orig_load
    set_autoshape_params(model, conf, iou, max_det)
    print(f"      ryw AutoShape params: conf={getattr(model, 'conf', '?')} iou={getattr(model, 'iou', '?')} "
          f"max_det={getattr(model, 'max_det', '?')}", flush=True)
    preds: dict[str, list] = {}
    for i, p in enumerate(image_paths):
        det = model(str(p), size=640).xyxy[0].cpu().numpy()
        preds[str(p)] = [(int(d[5]), float(d[4]), d[:4].astype(np.float32)) for d in det]
        if (i + 1) % 50 == 0:
            print(f"  ryw {i + 1}/{len(image_paths)}", flush=True)
    return preds


def run_ultralytics_predict(model, image_paths: list[Path], conf: float, iou: float,
                            max_det: int, tag: str) -> dict:
    preds: dict[str, list] = {}
    for i, p in enumerate(image_paths):
        r = model.predict(str(p), imgsz=640, conf=conf, iou=iou, max_det=max_det,
                          device="cpu", verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            xy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            cf = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            preds[str(p)] = [(int(c), float(f), xy[j]) for j, (c, f) in enumerate(zip(cl, cf))]
        else:
            preds[str(p)] = []
        if (i + 1) % 50 == 0:
            print(f"  {tag} {i + 1}/{len(image_paths)}", flush=True)
    return preds


def dump_jsonl(preds: dict, path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps({"img": k, "boxes": [[int(c), float(f), *b.tolist()] for c, f, b in v]})
                  for k, v in preds.items()), encoding="utf-8")


def cls_row(d: dict, c: int) -> dict:
    e = d["per_class"].get(c, {})
    return {"gt": e.get("gt", 0), "pred": e.get("pred@0.25", 0), "tp": e.get("tp@0.5", 0),
            "fp": e.get("fp@0.5", 0), "ap95": e.get("AP50-95", "—")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="只跑 4 张图端到端冒烟（跳过原生 val）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（0=全量）")
    ap.add_argument("--skip-native-val", action="store_true", help="跳过 ultralytics 原生 val 交叉核对")
    args = ap.parse_args()
    if args.dry_run:
        args.skip_native_val = True

    OUT.mkdir(exist_ok=True)
    image_paths = [Path(x.strip()) for x in VAL_LIST.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.dry_run:
        image_paths = image_paths[:4]
    elif args.limit:
        image_paths = image_paths[: args.limit]
    print(f"[1/5] 评测集 {len(image_paths)} 张（fold_01 val）", flush=True)
    gt = load_ground_truth(image_paths)
    n_gt = sum(len(v) for v in gt.values())
    print(f"      GT 框 {n_gt}", flush=True)

    env = {
        "torch": torch.__version__, "numpy": np.__version__,
        "sha256": {"ryw.pt": sha256(RYW_PT), "deployed_onnx": sha256(DEPLOY_ONNX),
                   "fold01_best.pt": sha256(FOLD01_PT)},
        "eval_config": {"imgsz": 640, "conf": CONF_EVAL, "nms_iou": IOU_EVAL, "max_det": MAX_DET_EVAL},
        "prod_config": {"imgsz": 640, "conf": CONF_PROD, "nms_iou": IOU_PROD, "max_det": MAX_DET_PROD},
        "n_images": len(image_paths), "n_gt_boxes": n_gt,
    }
    print(f"      onnx sha256={env['sha256']['deployed_onnx'][:24]}…", flush=True)

    results: dict[str, dict] = {}

    print("[2/5] my-ruler 推理+评测（三模型同一评测器）", flush=True)
    t0 = time.time()
    ryw_eval = run_ryw(image_paths, CONF_EVAL, IOU_EVAL, MAX_DET_EVAL)
    results["ryw_eval"] = evaluate_preds(ryw_eval, gt, NAMES)
    dump_jsonl(ryw_eval, OUT / "preds_ryw_eval.jsonl")
    print(f"      ryw eval done {time.time() - t0:.0f}s mAP50-95={results['ryw_eval']['mAP50-95']:.5f}", flush=True)

    from ultralytics import YOLO
    m_onnx = YOLO(str(DEPLOY_ONNX), task="detect")
    m_pt = YOLO(str(FOLD01_PT))

    t0 = time.time()
    onnx_eval = run_ultralytics_predict(m_onnx, image_paths, CONF_EVAL, IOU_EVAL, MAX_DET_EVAL, "onnx-eval")
    results["y11m_onnx_deployed_eval"] = evaluate_preds(onnx_eval, gt, NAMES)
    dump_jsonl(onnx_eval, OUT / "preds_y11m_onnx_eval.jsonl")
    print(f"      onnx eval done {time.time() - t0:.0f}s mAP50-95={results['y11m_onnx_deployed_eval']['mAP50-95']:.5f}", flush=True)

    t0 = time.time()
    pt_eval = run_ultralytics_predict(m_pt, image_paths, CONF_EVAL, IOU_EVAL, MAX_DET_EVAL, "pt-eval")
    results["y11m_pt_fold01_eval"] = evaluate_preds(pt_eval, gt, NAMES)
    print(f"      pt fold01 eval done {time.time() - t0:.0f}s mAP50-95={results['y11m_pt_fold01_eval']['mAP50-95']:.5f}", flush=True)

    print("[3/5] 生产运行点（conf=0.25/iou=0.45）", flush=True)
    t0 = time.time()
    ryw_prod = run_ryw(image_paths, CONF_PROD, IOU_PROD, MAX_DET_PROD["ryw"])
    results["ryw_prod"] = evaluate_preds(ryw_prod, gt, NAMES)
    print(f"      ryw prod done {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    onnx_prod = run_ultralytics_predict(m_onnx, image_paths, CONF_PROD, IOU_PROD, MAX_DET_PROD["y11m_onnx"], "onnx-prod")
    results["y11m_onnx_deployed_prod"] = evaluate_preds(onnx_prod, gt, NAMES)
    print(f"      onnx prod done {time.time() - t0:.0f}s", flush=True)

    if not args.skip_native_val:
        print("[4/5] ultralytics 原生 val 交叉核对（376 张，PT fold01 + ONNX deployed）", flush=True)
        for tag, model in (("fold01_pt", m_pt), ("deployed_onnx", m_onnx)):
            t0 = time.time()
            m = model.val(data=str(DATA_YAML), imgsz=640, device="cpu", conf=CONF_EVAL,
                          iou=IOU_EVAL, max_det=MAX_DET_EVAL, plots=False, save_json=False,
                          workers=0, project=str(OUT / "ultralytics_val"), name=tag, exist_ok=True,
                          verbose=False)
            results[f"native_val_{tag}"] = {
                "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
                "P": float(m.box.mp), "R": float(m.box.mr),
            }
            print(f"      {tag} native val done {time.time() - t0:.0f}s", flush=True)

    (OUT / "metrics.json").write_text(json.dumps({"env": env, "results": results},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")

    print("[5/5] report.md", flush=True)
    rows = []
    for tag in ("ryw_eval", "y11m_pt_fold01_eval", "y11m_onnx_deployed_eval"):
        r = results[tag]
        rows.append(f"| {tag} | {r['mAP50']:.5f} | {r['mAP50-95']:.5f} | {r['P_at_maxF1']:.5f} | "
                    f"{r['R_at_maxF1']:.5f} | {r['total@0.25']['n_pred']} |")
    lines = [
        "# ryw.pt vs 部署 YOLO11m ONNX 同口径复评（fold_01 val 376 图）",
        "",
        f"- 评测口径：imgsz=640 letterbox、conf={CONF_EVAL}、NMS IoU={IOU_EVAL}、max_det={MAX_DET_EVAL}、标准 mAP@0.5:0.95",
        f"- 生产运行点：conf={CONF_PROD}、NMS IoU={IOU_PROD}（ryw max_det=1000 与部署一致；onnx max_det=300 与部署一致）",
        f"- ryw.pt SHA-256={env['sha256']['ryw.pt'][:16]}…；部署 ONNX SHA-256={env['sha256']['deployed_onnx'][:16]}…（=优化分支 weights/dianjienie_yolo11m_ryw18_640.onnx）",
        "- 注意：部署 ONNX 为全开发集(1127图)最终权重，fold_01 val 图片在其训练集内（对其有利偏差）；ryw.pt 对这批图的训练暴露未知。冻结 198 图 test 未读取。数字用于工程对照，不是泛化结论。",
        "",
        "## 主对照表（my-ruler，三模型同一评测器）",
        "",
        "| 模型 | mAP50 | mAP50-95 | P@maxF1 | R@maxF1 | 框数@conf0.25 |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
    ]
    if "native_val_fold01_pt" in results:
        nv = results["native_val_fold01_pt"]
        mine = results["y11m_pt_fold01_eval"]
        lines += [
            "## 评测器交叉核对",
            "",
            f"- fold01 PT 原生 ultralytics val：mAP50={nv['mAP50']:.5f}，mAP50-95={nv['mAP50-95']:.5f}（历史本机复评记录 0.27848/0.14971）",
            f"- my-ruler 同权重：mAP50={mine['mAP50']:.5f}，mAP50-95={mine['mAP50-95']:.5f}",
            f"- 差值：{mine['mAP50'] - nv['mAP50']:+.5f} / {mine['mAP50-95'] - nv['mAP50-95']:+.5f}（|Δ|<0.005 视为同一把尺）",
            "",
        ]
    lines += ["## 生产运行点（conf=0.25）检测数量与质量", ""]
    for tag in ("ryw_prod", "y11m_onnx_deployed_prod"):
        t = results[tag]["total@0.25"]
        lines.append(f"- **{tag}**：框 {t['n_pred']}，TP@0.5 {t['n_tp@0.5']}，FP@0.5 {t['n_fp@0.5']}，"
                     f"P={t['precision@0.5']}，R={t['recall@0.5']}（GT {results[tag]['n_gt_boxes']} 框）")
    lines += ["", "## 逐类明细（生产运行点计数 + my-ruler AP）", "",
              "| cls | name | GT | ryw pred/TP/FP@0.5 | onnx pred/TP/FP@0.5 | ryw AP50-95 | onnx AP50-95 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in range(18):
        a, b = cls_row(results["ryw_prod"], c), cls_row(results["y11m_onnx_deployed_prod"], c)
        ra, rb = cls_row(results["ryw_eval"], c), cls_row(results["y11m_onnx_deployed_eval"], c)
        lines.append(f"| {c} | {NAMES[c]} | {a['gt']} "
                     f"| {a['pred']}/{a['tp']}/{a['fp']} | {b['pred']}/{b['tp']}/{b['fp']} "
                     f"| {ra['ap95']} | {rb['ap95']} |")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE ->", OUT / "report.md", flush=True)


if __name__ == "__main__":
    main()
