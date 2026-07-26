#!/usr/bin/env python3
"""
电积镍等级 v3：从 v2 逐框结果派生空间特征，从视图图像计算纹理特征。

- 空间：all_boxes.csv（归一化坐标），无需重新推理
- 纹理：视图硬链接图，短边缩至 ≤1024 后灰度 LBP/GLCM（numpy 手写，不依赖 skimage）
- 产物仅写入 dianji_grade_v3/，只读 v2 与数据集视图
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_V2 = ROOT / "dianji_grade_v2"
DEFAULT_VIEW = Path(r"E:/PythonProject/datasets/dianjinie_grade_classification_v1")
DEFAULT_OUT = ROOT / "dianji_grade_v3"
DEFAULT_SEED = 42

# 18 类 id（与 ryw.pt 一致）
CLASS_ID = {
    "nodule": 3,
    "overfire": 8,
    "quejiao": 12,
}

EDGE_CLASSES = ("quejiao", "nodule", "overfire")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="电积镍等级 v3：空间+纹理特征提取（未经企业确认）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--v2-dir", type=Path, default=DEFAULT_V2)
    p.add_argument("--view-root", type=Path, default=DEFAULT_VIEW)
    p.add_argument(
        "--v2-features",
        type=Path,
        default=None,
        help="默认 v2-dir/features_v2b_imgsz1536_conf0p25.csv",
    )
    p.add_argument(
        "--boxes-csv",
        type=Path,
        default=None,
        help="默认 v2-dir/detections_1536/all_boxes.csv",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--texture-short-side", type=int, default=1024, help="纹理计算前短边上限")
    p.add_argument("--edge-band", type=float, default=0.10, help="边缘带宽度（图边归一化比例）")
    p.add_argument("--conf-min", type=float, default=0.25, help="空间特征框置信度下限")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--limit", type=int, default=0, help="调试：仅前 N 张；0=全部")
    p.add_argument("--skip-texture", action="store_true", help="仅算空间特征")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def read_gray(path: Path) -> np.ndarray | None:
    import cv2

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return img


def resize_short_side(gray: np.ndarray, short_side: int) -> np.ndarray:
    import cv2

    h, w = gray.shape[:2]
    s = min(h, w)
    if s <= short_side:
        return gray
    scale = short_side / float(s)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)


def union_area_norm(rects: list[tuple[float, float, float, float]]) -> float:
    """归一化坐标系 [0,1]^2 上的矩形并集面积。"""
    cleaned = [(x1, y1, x2, y2) for x1, y1, x2, y2 in rects if x2 > x1 and y2 > y1]
    if not cleaned:
        return 0.0
    xs = sorted({v for r in cleaned for v in (r[0], r[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted((y1, y2) for x1, y1, x2, y2 in cleaned if x1 < right and x2 > left)
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
    return float(total)


def box_rect(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (
        max(0.0, cx - w / 2),
        max(0.0, cy - h / 2),
        min(1.0, cx + w / 2),
        min(1.0, cy + h / 2),
    )


def clip_rect_to_cell(
    rect: tuple[float, float, float, float],
    cell: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    x1 = max(rect[0], cell[0])
    y1 = max(rect[1], cell[1])
    x2 = min(rect[2], cell[2])
    y2 = min(rect[3], cell[3])
    if x2 > x1 and y2 > y1:
        return (x1, y1, x2, y2)
    return None


def spatial_features_from_boxes(
    boxes: list[dict[str, Any]],
    edge_band: float,
    conf_min: float,
) -> dict[str, float]:
    kept = [b for b in boxes if to_float(b.get("conf")) >= conf_min]
    feats: dict[str, float] = {}

    # 3x3 分区
    cells: list[tuple[int, int, tuple[float, float, float, float]]] = []
    for gy in range(3):
        for gx in range(3):
            cells.append(
                (
                    gy,
                    gx,
                    (gx / 3.0, gy / 3.0, (gx + 1) / 3.0, (gy + 1) / 3.0),
                )
            )

    rects: list[tuple[float, float, float, float]] = []
    areas: list[float] = []
    centers: list[tuple[float, float]] = []
    labels: list[int] = []
    for b in kept:
        cx, cy, w, h = to_float(b["cx"]), to_float(b["cy"]), to_float(b["w"]), to_float(b["h"])
        r = box_rect(cx, cy, w, h)
        a = max(0.0, (r[2] - r[0]) * (r[3] - r[1]))
        rects.append(r)
        areas.append(a)
        centers.append((cx, cy))
        labels.append(int(to_float(b.get("label"), -1)))

    for gy, gx, cell in cells:
        cell_rects = []
        cell_count = 0
        for r in rects:
            cr = clip_rect_to_cell(r, cell)
            if cr is not None:
                cell_count += 1
                cell_rects.append(cr)
        # 面积比相对整图
        feats[f"grid{gy}{gx}_count"] = float(cell_count)
        feats[f"grid{gy}{gx}_area_ratio"] = union_area_norm(cell_rects)

    # 边缘带：外环 [0,e]U[1-e,1]
    e = float(edge_band)
    edge_rects: list[tuple[float, float, float, float]] = []
    edge_count = 0
    edge_label_hit = {name: 0.0 for name in EDGE_CLASSES}
    for r, lab, (cx, cy) in zip(rects, labels, centers):
        on_edge = cx < e or cx > 1 - e or cy < e or cy > 1 - e
        # 也可用框与边缘带相交
        intersects_edge = (
            r[0] < e or r[2] > 1 - e or r[1] < e or r[3] > 1 - e
        )
        if intersects_edge or on_edge:
            edge_count += 1
            # 近似：整框面积计入边缘（简化，避免复杂环带裁剪）
            edge_rects.append(r)
            for name in EDGE_CLASSES:
                if lab == CLASS_ID[name]:
                    edge_label_hit[name] = 1.0
    feats["edge_count"] = float(edge_count)
    feats["edge_area_ratio"] = union_area_norm(edge_rects)
    feats["edge_box_frac"] = float(edge_count / len(rects)) if rects else 0.0
    for name in EDGE_CLASSES:
        feats[f"edge_has_{name}"] = edge_label_hit[name]

    if areas:
        arr = np.asarray(areas, dtype=np.float64)
        order = np.argsort(-arr)
        feats["max_box_area_ratio"] = float(arr[order[0]])
        topk = arr[order[: min(3, len(arr))]]
        feats["top3_box_area_sum_ratio"] = float(topk.sum())
        feats["box_area_mean"] = float(arr.mean())
        feats["box_area_var"] = float(arr.var())
    else:
        feats["max_box_area_ratio"] = 0.0
        feats["top3_box_area_sum_ratio"] = 0.0
        feats["box_area_mean"] = 0.0
        feats["box_area_var"] = 0.0

    # 最近邻中心距离均值（归一化欧氏）
    if len(centers) >= 2:
        pts = np.asarray(centers, dtype=np.float64)
        nn = []
        for i in range(len(pts)):
            d = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
            d[i] = 1e9
            nn.append(float(d.min()))
        feats["nn_center_dist_mean"] = float(np.mean(nn))
    else:
        feats["nn_center_dist_mean"] = 0.0

    feats["spatial_box_count"] = float(len(rects))
    return feats


def lbp_hist_10bin(gray: np.ndarray) -> np.ndarray:
    """半径1、8邻域 LBP，映射到 10 个均匀 bin 的归一化直方图。"""
    g = gray.astype(np.int16)
    h, w = g.shape
    if h < 3 or w < 3:
        return np.zeros(10, dtype=np.float64)
    c = g[1:-1, 1:-1]
    codes = np.zeros_like(c, dtype=np.uint8)
    neighbors = [
        (g[0:-2, 0:-2], 0),  # NW
        (g[0:-2, 1:-1], 1),  # N
        (g[0:-2, 2:], 2),  # NE
        (g[1:-1, 2:], 3),  # E
        (g[2:, 2:], 4),  # SE
        (g[2:, 1:-1], 5),  # S
        (g[2:, 0:-2], 6),  # SW
        (g[1:-1, 0:-2], 7),  # W
    ]
    for nb, bit in neighbors:
        codes |= ((nb >= c).astype(np.uint8) << bit)
    # 0..255 -> 10 bins
    hist, _ = np.histogram(codes.ravel(), bins=10, range=(0, 256), density=False)
    hist = hist.astype(np.float64)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def glcm_props(gray: np.ndarray, distances: list[int] | None = None) -> dict[str, float]:
    """
    简化 GLCM：灰度量化到 16 级；距离 1、2；方向 0/45/90/135 取均值。
    输出 contrast / dissimilarity / homogeneity / energy / correlation。
    """
    if distances is None:
        distances = [1, 2]
    levels = 16
    # quantize
    g = (gray.astype(np.float64) / 255.0 * (levels - 1)).astype(np.int32)
    g = np.clip(g, 0, levels - 1)
    h, w = g.shape
    angles = [(1, 0), (1, 1), (0, 1), (-1, 1)]  # dx,dy for 0,45,90,135
    props = {k: [] for k in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")}

    i_idx = np.arange(levels).reshape(-1, 1)
    j_idx = np.arange(levels).reshape(1, -1)
    ij_diff = i_idx - j_idx

    for dist in distances:
        for dx, dy in angles:
            # 构建共现
            x0 = max(0, -dx * dist)
            x1 = min(w, w - dx * dist)
            y0 = max(0, -dy * dist)
            y1 = min(h, h - dy * dist)
            if x1 <= x0 or y1 <= y0:
                continue
            a = g[y0:y1, x0:x1]
            b = g[y0 + dy * dist : y1 + dy * dist, x0 + dx * dist : x1 + dx * dist]
            if a.size == 0 or a.shape != b.shape:
                continue
            glcm = np.zeros((levels, levels), dtype=np.float64)
            np.add.at(glcm, (a.ravel(), b.ravel()), 1)
            # 对称化
            glcm = glcm + glcm.T
            s = glcm.sum()
            if s <= 0:
                continue
            p = glcm / s
            contrast = float((p * (ij_diff**2)).sum())
            dissim = float((p * np.abs(ij_diff)).sum())
            homo = float((p / (1.0 + np.abs(ij_diff))).sum())
            energy = float((p**2).sum())
            mu_i = float((p.sum(axis=1) * np.arange(levels)).sum())
            mu_j = float((p.sum(axis=0) * np.arange(levels)).sum())
            sig_i = float(np.sqrt(max((p.sum(axis=1) * (np.arange(levels) - mu_i) ** 2).sum(), 1e-12)))
            sig_j = float(np.sqrt(max((p.sum(axis=0) * (np.arange(levels) - mu_j) ** 2).sum(), 1e-12)))
            corr = float(
                ((i_idx - mu_i) * (j_idx - mu_j) * p).sum() / (sig_i * sig_j + 1e-12)
            )
            props["contrast"].append(contrast)
            props["dissimilarity"].append(dissim)
            props["homogeneity"].append(homo)
            props["energy"].append(energy)
            props["correlation"].append(corr)

    out = {}
    for k, vals in props.items():
        out[f"glcm_{k}"] = float(np.mean(vals)) if vals else 0.0
    return out


def texture_features(gray: np.ndarray, short_side: int) -> dict[str, float]:
    import cv2

    g = resize_short_side(gray, short_side)
    feats: dict[str, float] = {}
    gf = g.astype(np.float64)
    feats["gray_mean"] = float(gf.mean() / 255.0)
    feats["gray_var"] = float(gf.var() / (255.0**2))
    # 梯度幅值
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    feats["grad_mean"] = float(mag.mean() / 255.0)
    feats["grad_var"] = float(mag.var() / (255.0**2))
    feats["grad_p90"] = float(np.percentile(mag, 90) / 255.0)

    hist = lbp_hist_10bin(g)
    for i, v in enumerate(hist.tolist()):
        feats[f"lbp_bin{i:02d}"] = float(v)

    glcm = glcm_props(g)
    feats.update(glcm)
    return feats


def resolve_image_path(row: dict[str, str], view_root: Path) -> Path | None:
    for key in ("image_path_view", "image_path"):
        v = row.get(key) or ""
        if v:
            p = Path(v)
            if p.exists():
                return p
    rel = row.get("link_relpath") or ""
    if rel:
        p = view_root / rel
        if p.exists():
            return p
    # 视图 manifest 风格：按 stem+grade+split 搜索
    return None


def build_box_index(boxes: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    """key = (image_stem, source_norm, grade)"""
    idx: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for b in boxes:
        stem = b.get("image_stem") or Path(b.get("image_name", "")).stem
        src = b.get("source", "")
        if "0309" in src:
            src_n = "enterprise_dianji_0309_classification"
        else:
            src_n = "classify_4types_dianji"
        grade = b.get("grade", "")
        idx[(stem, src_n, grade)].append(b)
        # 也按 image_name 宽松索引
        idx[(b.get("image_name", ""), src_n, grade)].append(b)
    return idx


def normalize_source(row: dict[str, str]) -> str:
    src = row.get("source") or row.get("source_dataset") or ""
    if "0309" in src:
        return "enterprise_dianji_0309_classification"
    if "classify" in src or src == "classify_4types_dianji":
        return "classify_4types_dianji"
    return src


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    v2 = args.v2_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    feat_path = args.v2_features or (v2 / "features_v2b_imgsz1536_conf0p25.csv")
    boxes_path = args.boxes_csv or (v2 / "detections_1536" / "all_boxes.csv")
    if not feat_path.exists():
        print(f"缺少 v2 特征: {feat_path}", file=sys.stderr)
        return 2
    if not boxes_path.exists():
        print(f"缺少 boxes: {boxes_path}", file=sys.stderr)
        return 2

    print("=== dianji_grade_v3 特征提取 ===")
    print(f"v2 features: {feat_path}")
    print(f"boxes: {boxes_path}")
    print(f"output: {out}")
    print("status: unconfirmed_pending_enterprise")

    rows = read_csv(feat_path)
    boxes = read_csv(boxes_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    box_idx = build_box_index(boxes)
    print(f"样本 {len(rows)}，框记录 {len(boxes)}，索引键 {len(box_idx)}")

    if args.dry_run:
        print("[dry-run] 跳过计算")
        return 0

    spatial_rows: list[dict[str, Any]] = []
    texture_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    missing_img: list[dict[str, str]] = []
    t0 = time.time()

    for i, r in enumerate(rows):
        stem = r.get("image_stem") or Path(r.get("image_name", "")).stem
        src_n = normalize_source(r)
        grade = r.get("true_grade") or r.get("grade") or ""
        key1 = (stem, src_n, grade)
        key2 = (r.get("image_name", ""), src_n, grade)
        bxs = box_idx.get(key1) or box_idx.get(key2) or []

        sp = spatial_features_from_boxes(bxs, edge_band=args.edge_band, conf_min=args.conf_min)
        meta = {
            "image_name": r.get("image_name"),
            "image_stem": stem,
            "source": r.get("source") or r.get("source_dataset"),
            "source_dataset": src_n,
            "grade": grade,
            "true_grade": grade,
            "split": r.get("split"),
            "dup_group_id": r.get("dup_group_id"),
            "sha256": r.get("sha256", ""),
            "status": "unconfirmed_pending_enterprise",
            "feature_version": "v3_spatial",
            "n_boxes_used": len(bxs),
        }
        spatial_rows.append({**meta, **sp})

        tex: dict[str, float] = {}
        if not args.skip_texture:
            ip = resolve_image_path(r, args.view_root)
            if ip is None or not ip.exists():
                missing_img.append(
                    {
                        "image_name": r.get("image_name", ""),
                        "reason": "image_not_found",
                        "image_path": r.get("image_path", ""),
                        "image_path_view": r.get("image_path_view", ""),
                        "link_relpath": r.get("link_relpath", ""),
                    }
                )
                # 纹理填 0
                tex = {k: 0.0 for k in [
                    "gray_mean", "gray_var", "grad_mean", "grad_var", "grad_p90",
                    *[f"lbp_bin{j:02d}" for j in range(10)],
                    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
                    "glcm_energy", "glcm_correlation",
                ]}
                tex["texture_ok"] = 0.0
            else:
                gray = read_gray(ip)
                if gray is None:
                    missing_img.append(
                        {
                            "image_name": r.get("image_name", ""),
                            "reason": "unreadable",
                            "image_path": str(ip),
                        }
                    )
                    tex = {k: 0.0 for k in [
                        "gray_mean", "gray_var", "grad_mean", "grad_var", "grad_p90",
                        *[f"lbp_bin{j:02d}" for j in range(10)],
                        "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
                        "glcm_energy", "glcm_correlation",
                    ]}
                    tex["texture_ok"] = 0.0
                else:
                    tex = texture_features(gray, args.texture_short_side)
                    tex["texture_ok"] = 1.0
            texture_rows.append(
                {
                    **{k: meta[k] for k in (
                        "image_name", "image_stem", "source", "source_dataset",
                        "grade", "true_grade", "split", "dup_group_id", "sha256", "status",
                    )},
                    "feature_version": "v3_texture",
                    "texture_short_side": args.texture_short_side,
                    **tex,
                }
            )

        # 合并：v2 43 列 + 空间 + 纹理
        full = dict(r)
        full["status"] = "unconfirmed_pending_enterprise"
        full["feature_version"] = "v3_v2b025_spatial_texture"
        for k, v in sp.items():
            full[f"sp__{k}"] = v
        for k, v in tex.items():
            full[f"tx__{k}"] = v
        full_rows.append(full)

        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(rows)}] {rate:.2f} img/s  missing_tex={len(missing_img)}")

    write_csv(out / "features_spatial.csv", spatial_rows)
    if not args.skip_texture:
        write_csv(out / "features_texture.csv", texture_rows)
    write_csv(out / "features_v3_full.csv", full_rows)
    write_csv(out / "texture_missing_images.csv", missing_img)

    # 仅空间 / 仅纹理 合并表（含标签与 split）便于消融
    only_sp = []
    only_tx = []
    for r, sp_r in zip(rows, spatial_rows):
        o = {
            "image_name": r.get("image_name"),
            "true_grade": r.get("true_grade") or r.get("grade"),
            "grade": r.get("true_grade") or r.get("grade"),
            "split": r.get("split"),
            "dup_group_id": r.get("dup_group_id"),
            "status": "unconfirmed_pending_enterprise",
            "feature_version": "v3_spatial_only",
        }
        for k, v in sp_r.items():
            if k.startswith("grid") or k.startswith("edge") or k.startswith("max_") or k.startswith("top3") or k.startswith("box_") or k.startswith("nn_") or k == "spatial_box_count":
                o[k] = v
        only_sp.append(o)
    write_csv(out / "features_spatial_only.csv", only_sp)

    if texture_rows:
        only_tx = []
        for r, tx_r in zip(rows, texture_rows):
            o = {
                "image_name": r.get("image_name"),
                "true_grade": r.get("true_grade") or r.get("grade"),
                "grade": r.get("true_grade") or r.get("grade"),
                "split": r.get("split"),
                "dup_group_id": r.get("dup_group_id"),
                "status": "unconfirmed_pending_enterprise",
                "feature_version": "v3_texture_only",
            }
            for k, v in tx_r.items():
                if k.startswith("gray_") or k.startswith("grad_") or k.startswith("lbp_") or k.startswith("glcm_") or k == "texture_ok":
                    o[k] = v
            only_tx.append(o)
        write_csv(out / "features_texture_only.csv", only_tx)

    # NaN/inf 检查
    def check_finite(rows_in: list[dict[str, Any]], name: str) -> dict[str, Any]:
        bad = 0
        for r in rows_in:
            for k, v in r.items():
                if isinstance(v, (int, float, np.floating)):
                    if not math.isfinite(float(v)):
                        bad += 1
                else:
                    try:
                        fv = float(v)
                        if not math.isfinite(fv):
                            bad += 1
                    except (TypeError, ValueError):
                        pass
        return {"file": name, "rows": len(rows_in), "nonfinite_values": bad}

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "unconfirmed_pending_enterprise",
        "n_samples": len(rows),
        "n_boxes_csv": len(boxes),
        "images_with_any_box_index_hit": sum(1 for r in spatial_rows if r.get("n_boxes_used", 0) > 0),
        "texture_missing": len(missing_img),
        "texture_short_side": args.texture_short_side,
        "edge_band": args.edge_band,
        "conf_min": args.conf_min,
        "v2_features": str(feat_path.resolve()),
        "boxes_csv": str(boxes_path.resolve()),
        "view_root": str(args.view_root.resolve()),
        "outputs": {
            "features_v3_full": str((out / "features_v3_full.csv").resolve()),
            "features_spatial": str((out / "features_spatial.csv").resolve()),
            "features_texture": str((out / "features_texture.csv").resolve()) if not args.skip_texture else None,
        },
        "finite_checks": [
            check_finite(spatial_rows, "spatial"),
            check_finite(full_rows, "full"),
        ],
        "elapsed_sec": time.time() - t0,
        "note": "空间特征来自 v2 detections_1536，未重跑检测；纹理为整板灰度统计",
    }
    (out / "extract_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
