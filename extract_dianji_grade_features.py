#!/usr/bin/env python3
"""
在 dianjinie_grade_classification_v1 视图上提取等级特征：
- v2a：join 既有 combined_all_pseudo_v2 samples.csv（1024/conf0.25 历史口径）
- v2b：ryw.pt 重推理 imgsz=1536，逐框写入 dianji_grade_v2/detections_1536/，
       并按 conf≥0.25 / 0.40 计算两套 43 列特征

派生产物只写 E:\\PythonProject\\优化\\dianji_grade_v2\\，不写数据集目录。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_VIEW = Path(r"E:/PythonProject/datasets/dianjinie_grade_classification_v1")
DEFAULT_SAMPLES_CSV = ROOT / "grade_feature_analysis" / "combined_all_pseudo_v2_grade" / "samples.csv"
DEFAULT_OUT = ROOT / "dianji_grade_v2"
DEFAULT_WEIGHTS = Path(r"E:/PythonProject/Tools/ModelAutoLabelTool/weights/yolov5/ryw.pt")
DEFAULT_YOLOV5_REPO = Path(r"E:/PythonProject/Tools/ModelAutoLabelTool/vendor/yolov5")
DEFAULT_PROJECT_ROOT = Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi")

CLASS_NAMES: list[str] = [
    "wuzi",
    "waterstain",
    "yellowstain",
    "nodule",
    "airhole",
    "bubble",
    "bump",
    "fawu",
    "overfire",
    "NiOH",
    "greenstain",
    "suanwuran",
    "quejiao",
    "watermark",
    "youpao",
    "heibian",
    "mijixiaokeng",
    "fanxitong",
]

A_LABELS = {0, 1}
B_LABELS = {2, 4, 5, 6, 10, 12, 13}
C_LABELS = {7, 8, 11}
AIRHOLE_LABELS = {4}
NODULE_LABEL = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="电积镍等级视图特征提取 v2a join + v2b ryw.pt 1536（未经企业确认）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--view-root", type=Path, default=DEFAULT_VIEW)
    p.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--yolov5-repo", type=Path, default=DEFAULT_YOLOV5_REPO)
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--img-size", type=int, default=1536)
    p.add_argument("--conf-thres", type=float, default=0.25)
    p.add_argument("--iou-thres", type=float, default=0.45)
    p.add_argument("--feature-conf-thresholds", type=float, nargs="+", default=[0.25, 0.40])
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-log-every", type=int, default=25)
    p.add_argument("--limit", type=int, default=0, help="调试：仅前 N 张；0=全部")
    p.add_argument("--skip-v2a", action="store_true")
    p.add_argument("--skip-v2b", action="store_true")
    p.add_argument("--resume-v2b", action="store_true", help="跳过已有 per_image 标签")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def read_image_bgr(path: Path) -> np.ndarray | None:
    import cv2

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def union_area(rectangles: Iterable[tuple[float, float, float, float]]) -> float:
    rects = [(x1, y1, x2, y2) for x1, y1, x2, y2 in rectangles if x2 > x1 and y2 > y1]
    if not rects:
        return 0.0
    xs = sorted({v for r in rects for v in (r[0], r[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted((y1, y2) for x1, y1, x2, y2 in rects if x1 < right and x2 > left)
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for ns, ne in intervals[1:]:
            if ns > end:
                covered += end - start
                start, end = ns, ne
            else:
                end = max(end, ne)
        covered += end - start
        total += (right - left) * covered
    return total


def normalized_rect(det: dict[str, Any]) -> tuple[float, float, float, float]:
    cx, cy, w, h = float(det["cx"]), float(det["cy"]), float(det["w"]), float(det["h"])
    return (
        max(0.0, cx - w / 2),
        max(0.0, cy - h / 2),
        min(1.0, cx + w / 2),
        min(1.0, cy + h / 2),
    )


def hardcoded_rect(det: dict[str, Any], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    cx = float(det["cx"]) * image_width
    cy = float(det["cy"]) * image_height
    width = float(det["w"]) * image_width
    height = float(det["h"]) * image_height
    x1 = max(0, int((cx - width / 2) / 4))
    y1 = max(0, int((cy - height / 2) / 4))
    x2 = min(4069, int((cx + width / 2) / 4))
    y2 = min(3554, int((cy + height / 2) / 4))
    return x1, y1, x2, y2


def compute_plate_area(image_bgr: np.ndarray, project_root: Path) -> tuple[float, bool]:
    import cv2

    h, w = image_bgr.shape[:2]
    fallback = float(w * h)
    try:
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from preprocess import findcent

        resized = cv2.resize(image_bgr, (1024, 888))
        _, pw, ph, _ = findcent(resized)
        area = float(pw * ph)
        if area <= 0:
            return fallback, True
        return area, False
    except Exception:
        return fallback, True


def features_from_detections(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    plate_area: float,
    conf_thres: float,
) -> dict[str, Any]:
    kept = [d for d in detections if float(d.get("conf", 0)) >= conf_thres]
    has_nodule = False
    for det in kept:
        if int(det["label"]) != NODULE_LABEL:
            continue
        width = float(det["w"]) * image_width
        height = float(det["h"]) * image_height
        aspect = max(width / height, height / width) if min(width, height) > 0 else 1.0
        if width * height >= 13000.0 or aspect >= 3.0:
            has_nodule = True

    def group_ratio(labels: set[int]) -> float:
        rects = [
            hardcoded_rect(det, image_width, image_height)
            for det in kept
            if int(det["label"]) in labels
        ]
        if plate_area <= 0:
            return 0.0
        area = union_area([(float(a), float(b), float(c), float(d)) for a, b, c, d in rects])
        return float(area / plate_area)

    all_norm = [normalized_rect(d) for d in kept]
    per_class_rects: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    per_class_counts: Counter[int] = Counter()
    for det in kept:
        cid = int(det["label"])
        per_class_counts[cid] += 1
        per_class_rects[cid].append(normalized_rect(det))

    out: dict[str, Any] = {
        "image_width": image_width,
        "image_height": image_height,
        "plate_area": plate_area,
        "detection_count": len(kept),
        "max_confidence": max((float(d.get("conf", 0)) for d in kept), default=0.0),
        "total_bbox_union_image_ratio": union_area(all_norm),
        "A_ratio": group_ratio(A_LABELS),
        "B_ratio": group_ratio(B_LABELS),
        "C_ratio": group_ratio(C_LABELS),
        "Airhole_ratio": group_ratio(AIRHOLE_LABELS),
        "has_nodule": int(has_nodule),
    }
    for i, name in enumerate(CLASS_NAMES):
        out[f"count__{name}"] = int(per_class_counts.get(i, 0))
        out[f"area_ratio__{name}"] = union_area(per_class_rects.get(i, []))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys: list[str] = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_manifest(view_root: Path) -> list[dict[str, str]]:
    path = view_root / "manifests" / "samples.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_image_path(view_root: Path, row: dict[str, str]) -> Path:
    rel = row.get("link_relpath", "")
    if rel:
        p = view_root / rel
        if p.exists():
            return p
    # fallback
    return Path(row.get("image_path", ""))


def join_v2a(manifest: list[dict[str, str]], samples_csv: Path, view_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    samples_by_name: dict[str, dict[str, str]] = {}
    samples_by_stem: dict[str, list[dict[str, str]]] = defaultdict(list)
    with samples_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("image_name", "")
            samples_by_name[name] = row
            samples_by_stem[Path(name).stem].append(row)

    joined: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for m in manifest:
        src = m["source_dataset"]
        grade = m["grade"]
        source_split = m.get("source_split", "")
        raw_name = m.get("image_name_raw", "")
        stem = m.get("image_stem", Path(raw_name).stem)
        hit: dict[str, str] | None = None

        if src == "classify_4types_dianji":
            for c in (
                f"classify_4types_dianji_{source_split}_{grade}_{raw_name}",
                f"classify_4types_dianji_train_{grade}_{raw_name}",
                f"classify_4types_dianji_valid_{grade}_{raw_name}",
            ):
                if c in samples_by_name:
                    hit = samples_by_name[c]
                    break
            if hit is None:
                for full_stem, cands in samples_by_stem.items():
                    if stem in full_stem and "classify_4types_dianji" in full_stem and grade in full_stem:
                        hit = cands[0]
                        break
        else:
            for c in (
                f"enterprise_dianji_0309_classification_images_{grade}_{raw_name}",
                f"enterprise_dianji_0309_classification_images_{grade}_{stem}.png",
            ):
                if c in samples_by_name:
                    hit = samples_by_name[c]
                    break
            if hit is None:
                token = f"images_{grade}_{stem}"
                for full_stem, cands in samples_by_stem.items():
                    if token in full_stem or (
                        full_stem.endswith(f"_{stem}") and "0309" in full_stem and grade in full_stem
                    ):
                        hit = cands[0]
                        break

        if hit is None:
            missing.append(dict(m))
            continue

        out: dict[str, Any] = {
            "image_name": m["image_name"],
            "image_stem": stem,
            "image_name_raw": raw_name,
            "image_path": str(resolve_image_path(view_root, m)),
            "source_dataset": src,
            "source_split": source_split,
            "grade": grade,
            "true_grade": grade,
            "split": m["split"],
            "dup_group_id": m.get("dup_group_id", ""),
            "sha256": m.get("sha256", ""),
            "feature_version": "v2a_join_samples_csv",
            "joined_image_name": hit.get("image_name", ""),
            "status": "unconfirmed_pending_enterprise",
        }
        for col in (
            "image_width",
            "image_height",
            "plate_area",
            "detection_count",
            "max_confidence",
            "total_bbox_union_image_ratio",
            "A_ratio",
            "B_ratio",
            "C_ratio",
            "Airhole_ratio",
            "has_nodule",
            "type_id_detection_only",
            "type_name_detection_only",
        ):
            out[col] = hit.get(col, "")
        for n in CLASS_NAMES:
            out[f"count__{n}"] = hit.get(f"count__{n}", "0")
            out[f"area_ratio__{n}"] = hit.get(f"area_ratio__{n}", "0")
        joined.append(out)

    meta = {
        "manifest_total": len(manifest),
        "joined": len(joined),
        "missing": len(missing),
        "coverage": len(joined) / len(manifest) if manifest else 0.0,
        "grade_counts_joined": dict(Counter(r["grade"] for r in joined)),
        "grade_counts_missing": dict(Counter(r["grade"] for r in missing)),
        "source_counts_joined": dict(Counter(r["source_dataset"] for r in joined)),
        "source_counts_missing": dict(Counter(r["source_dataset"] for r in missing)),
        "samples_csv": str(samples_csv.resolve()),
        "detection_params_note": "历史口径 ryw.pt conf=0.25 iou=0.45 imgsz=1024",
        "status": "unconfirmed_pending_enterprise",
    }
    return joined, meta, missing


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return requested


def load_yolov5(weights: Path, yolov5_repo: Path, device: str, conf: float, iou: float):
    import torch

    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _load  # type: ignore[assignment]

    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not (yolov5_repo / "hubconf.py").is_file():
        raise FileNotFoundError(yolov5_repo)

    yolo_str = str(yolov5_repo.resolve())
    if yolo_str not in sys.path:
        sys.path.insert(0, yolo_str)
    old = Path.cwd()
    os.chdir(yolo_str)
    try:
        model = torch.hub.load(
            yolo_str,
            "custom",
            path=str(weights),
            source="local",
            device=device,
            autoshape=True,
            _verbose=False,
        )
    finally:
        os.chdir(old)
    model.eval()
    if hasattr(model, "conf"):
        model.conf = float(conf)
    if hasattr(model, "iou"):
        model.iou = float(iou)
    return model


def run_v2b(
    manifest: list[dict[str, str]],
    view_root: Path,
    output_dir: Path,
    weights: Path,
    yolov5_repo: Path,
    project_root: Path,
    img_size: int,
    conf_thres: float,
    iou_thres: float,
    feature_conf_thresholds: list[float],
    device_req: str,
    batch_log_every: int,
    resume: bool,
    limit: int,
) -> dict[str, Any]:
    import torch

    device = resolve_device(device_req)
    det_dir = output_dir / "detections_1536"
    per_dir = det_dir / "per_image"
    det_dir.mkdir(parents=True, exist_ok=True)
    boxes_csv = det_dir / "all_boxes.csv"
    missing_path = det_dir / "missing_images.csv"

    print(f"[v2b] device={device} cuda={torch.cuda.is_available()} imgsz={img_size}")
    model = load_yolov5(weights, yolov5_repo, device, conf_thres, iou_thres)
    try:
        names = list(model.model.names.values()) if hasattr(model.model, "names") else list(model.names.values())
    except Exception:
        names = CLASS_NAMES

    rows = manifest[:limit] if limit and limit > 0 else list(manifest)

    # resume: 已有 per_image 文件
    done_keys: set[str] = set()
    if resume:
        for p in per_dir.rglob("*.txt"):
            # per_image/{split}/{grade}/{stem}.txt
            done_keys.add(p.stem)
        print(f"[v2b] resume: 已有 per_image {len(done_keys)} 个 stem")

    write_header = not (resume and boxes_csv.exists())
    box_fields = [
        "image_name",
        "image_stem",
        "source_dataset",
        "grade",
        "split",
        "dup_group_id",
        "label",
        "class_name",
        "conf",
        "cx",
        "cy",
        "w",
        "h",
        "image_width",
        "image_height",
        "infer_img_size",
    ]
    box_f = boxes_csv.open("a" if resume and boxes_csv.exists() else "w", encoding="utf-8-sig", newline="")
    box_writer = csv.DictWriter(box_f, fieldnames=box_fields)
    if write_header:
        box_writer.writeheader()

    missing: list[dict[str, Any]] = []
    # 收集每图 meta+dets
    image_store: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    processed = 0

    try:
        for m in rows:
            image_name = m["image_name"]
            stem = Path(image_name).stem
            key = image_name
            side = per_dir / m["split"] / m["grade"] / f"{stem}.txt"

            if resume and side.exists():
                # 读已有标签
                dets: list[dict[str, Any]] = []
                text = side.read_text(encoding="utf-8")
                for line in text.splitlines():
                    parts = line.split()
                    if len(parts) >= 6:
                        dets.append(
                            {
                                "label": int(float(parts[0])),
                                "cx": float(parts[1]),
                                "cy": float(parts[2]),
                                "w": float(parts[3]),
                                "h": float(parts[4]),
                                "conf": float(parts[5]),
                            }
                        )
                img_path = resolve_image_path(view_root, m)
                img = read_image_bgr(img_path)
                if img is None:
                    missing.append({**m, "reason": "unreadable_resume"})
                    continue
                h0, w0 = img.shape[:2]
                plate_area, plate_fb = compute_plate_area(img, project_root)
                image_store[key] = {
                    "meta": m,
                    "dets": dets,
                    "image_width": w0,
                    "image_height": h0,
                    "plate_area": plate_area,
                    "plate_area_fallback": int(plate_fb),
                    "image_path": str(img_path),
                }
                processed += 1
                continue

            img_path = resolve_image_path(view_root, m)
            img = read_image_bgr(img_path)
            if img is None:
                missing.append({**m, "reason": "unreadable"})
                continue
            h0, w0 = img.shape[:2]
            plate_area, plate_fb = compute_plate_area(img, project_root)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model(img, size=img_size)
            dets_xyxy = result.xyxy[0].cpu().numpy() if len(result.xyxy) else np.empty((0, 6))

            dets = []
            for det in dets_xyxy:
                x1, y1, x2, y2, conf, cls = det[:6]
                cls_id = int(cls)
                bw = max(0.0, float(x2 - x1))
                bh = max(0.0, float(y2 - y1))
                if bw <= 0 or bh <= 0:
                    continue
                cx = (float(x1) + float(x2)) / 2.0 / w0
                cy = (float(y1) + float(y2)) / 2.0 / h0
                wn = bw / w0
                hn = bh / h0
                d = {"label": cls_id, "conf": float(conf), "cx": cx, "cy": cy, "w": wn, "h": hn}
                dets.append(d)
                box_writer.writerow(
                    {
                        "image_name": image_name,
                        "image_stem": stem,
                        "source_dataset": m.get("source_dataset", ""),
                        "grade": m["grade"],
                        "split": m["split"],
                        "dup_group_id": m.get("dup_group_id", ""),
                        "label": cls_id,
                        "class_name": names[cls_id] if 0 <= cls_id < len(names) else str(cls_id),
                        "conf": f"{float(conf):.6f}",
                        "cx": f"{cx:.6f}",
                        "cy": f"{cy:.6f}",
                        "w": f"{wn:.6f}",
                        "h": f"{hn:.6f}",
                        "image_width": w0,
                        "image_height": h0,
                        "infer_img_size": img_size,
                    }
                )

            side.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                f"{d['label']} {d['cx']:.6f} {d['cy']:.6f} {d['w']:.6f} {d['h']:.6f} {d['conf']:.6f}"
                for d in dets
            ]
            side.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            image_store[key] = {
                "meta": m,
                "dets": dets,
                "image_width": w0,
                "image_height": h0,
                "plate_area": plate_area,
                "plate_area_fallback": int(plate_fb),
                "image_path": str(img_path),
            }
            processed += 1
            if processed % batch_log_every == 0 or processed == len(rows):
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  [v2b] {processed}/{len(rows)}  {rate:.2f} img/s  {elapsed/60:.1f} min")
            box_f.flush()
    finally:
        box_f.close()

    # 特征
    feature_paths: dict[str, str] = {}
    for thr in feature_conf_thresholds:
        feat_rows: list[dict[str, Any]] = []
        for key, pack in image_store.items():
            m = pack["meta"]
            feats = features_from_detections(
                pack["dets"],
                int(pack["image_width"]),
                int(pack["image_height"]),
                float(pack["plate_area"]),
                conf_thres=float(thr),
            )
            row = {
                "image_name": m["image_name"],
                "image_stem": Path(m["image_name"]).stem,
                "image_name_raw": m.get("image_name_raw", ""),
                "image_path": pack["image_path"],
                "source_dataset": m.get("source_dataset", ""),
                "source_split": m.get("source_split", ""),
                "grade": m["grade"],
                "true_grade": m["grade"],
                "split": m["split"],
                "dup_group_id": m.get("dup_group_id", ""),
                "sha256": m.get("sha256", ""),
                "feature_version": f"v2b_imgsz{img_size}_conf{thr:.2f}",
                "infer_img_size": img_size,
                "feature_conf_thres": thr,
                "detector_conf": conf_thres,
                "detector_iou": iou_thres,
                "plate_area_fallback": pack.get("plate_area_fallback", 0),
                "status": "unconfirmed_pending_enterprise",
                **feats,
            }
            feat_rows.append(row)
        thr_tag = f"{thr:.2f}".replace(".", "p")
        out_csv = output_dir / f"features_v2b_imgsz{img_size}_conf{thr_tag}.csv"
        write_csv(out_csv, feat_rows)
        feature_paths[f"conf_{thr_tag}"] = str(out_csv)
        print(f"[v2b] 特征 conf>={thr}: {len(feat_rows)} -> {out_csv.name}")

    write_csv(missing_path, missing)
    summary = {
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "weights": str(weights.resolve()),
        "img_size": img_size,
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
        "manifest_requested": len(rows),
        "images_with_features": len(image_store),
        "coverage": len(image_store) / len(rows) if rows else 0.0,
        "missing_count": len(missing),
        "missing_path": str(missing_path),
        "boxes_csv": str(boxes_csv),
        "feature_csvs": feature_paths,
        "class_names": names,
        "elapsed_sec": time.time() - t0,
        "status": "unconfirmed_pending_enterprise",
    }
    (det_dir / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print("=== 电积镍等级特征提取 ===")
    print(f"视图: {args.view_root}")
    print(f"输出: {out}")
    print("状态: unconfirmed_pending_enterprise")

    manifest = load_manifest(args.view_root)
    print(f"manifest 行数: {len(manifest)}")
    if args.dry_run:
        print("[dry-run] 跳过实际提取")
        return 0

    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "view_root": str(args.view_root.resolve()),
        "status": "unconfirmed_pending_enterprise",
        "manifest_rows": len(manifest),
    }

    if not args.skip_v2a:
        joined, v2a_meta, missing = join_v2a(manifest, args.samples_csv, args.view_root)
        write_csv(out / "features_v2a.csv", joined)
        write_csv(out / "features_v2a_missing.csv", missing)
        (out / "features_v2a_summary.json").write_text(
            json.dumps(v2a_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["v2a"] = v2a_meta
        print(f"[v2a] {v2a_meta['joined']}/{v2a_meta['manifest_total']} coverage={v2a_meta['coverage']:.2%}")

    if not args.skip_v2b:
        v2b = run_v2b(
            manifest=manifest,
            view_root=args.view_root,
            output_dir=out,
            weights=args.weights,
            yolov5_repo=args.yolov5_repo,
            project_root=args.project_root,
            img_size=args.img_size,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            feature_conf_thresholds=list(args.feature_conf_thresholds),
            device_req=args.device,
            batch_log_every=args.batch_log_every,
            resume=args.resume_v2b,
            limit=args.limit,
        )
        summary["v2b"] = v2b
        print(f"[v2b] coverage={v2b['coverage']:.2%} missing={v2b['missing_count']}")

    (out / "extract_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
