# -*- coding: utf-8 -*-
"""C2.2 电解镍 63 列特征提取（路线文档 §4.2，弱标签蒸馏第二步）。

用 A3 部署 ONNX（901ccf60，18 类 RYW）对 C2.1 弱标签同批 1,127 图提取与电积镍
LGBM 完全同构的 63 列特征（43 BASE + 20 纹理）。

口径原则「生产实现即来源」：直接复用优化分支 5a7bb5c 的 inference 生产代码路径——
`_predict_dianji_yolo_raw`（显示检测器 raw predict，640/0.25/0.45，与 v1 等级路径一致）
+ `_predict_dianji_grade_detections`（归一化+类别名校验，激活合同临时指回 v1 使其分派
ryw18 schema）+ `extract_grade_features`（63 列聚合）。不做平行重实现。

产物只写入本目录（版本化，不覆盖既有产物）；数据集只读。
用法：
  python extract_c2p2_features_y11m640.py --dry-run     # 只校验清单与 schema
  python extract_c2p2_features_y11m640.py --limit 4     # 冒烟
  python extract_c2p2_features_y11m640.py               # 全量 1,127 图（CPU 约 25 分钟）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
WORKTREE = Path("C:/Users/23916/AppData/Local/Temp/jcw_b3clean_wt")
WEAKLABEL_CSV = HERE / "weaklabels_summary.csv"
ONNX_PATH = WORKTREE / "weights" / "dianjienie_yolo11m_ryw18_640.onnx"
EXPECTED_ONNX_SHA256 = "901ccf6006f44689483aaf6f0d1d1b8c18132a85c7d881be7e62fcdccc255b21"
V1_CONTRACT = WORKTREE / "weights" / "dianji_grade_lgbm_v1_contract.json"
OUT_FEATURES = HERE / "features_c2p2_y11m640.csv"
OUT_PERBOX = HERE / "per_box_c2p2_y11m640.csv"
OUT_SUMMARY = HERE / "extraction_summary_c2p2.json"
FEATURE_VERSION = "c2p2_y11m640_v1"
METADATA_FIELDS = [
    "sample_id", "image_name", "image_path", "group_id", "fold", "split", "source_split",
    "type_weak", "nodule_weak", "n_defects_weak", "feature_version",
    "image_width", "image_height", "plate_area", "plate_area_fallback", "max_confidence",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_pair(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    image_gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image_bgr is None or image_gray is None:
        raise ValueError(f"无法解码图像: {image_path}")
    return image_bgr, image_gray


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="只校验清单与 schema，不加载模型")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全量）")
    parser.add_argument("--force", action="store_true", help="允许覆盖同名产物")
    args = parser.parse_args()

    for out in (OUT_FEATURES, OUT_PERBOX, OUT_SUMMARY):
        if out.exists() and not args.force:
            raise RuntimeError(f"产物已存在（用 --force 覆盖）: {out}")

    with WEAKLABEL_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1127:
        raise ValueError(f"弱标签清单行数 {len(rows)} != 1127")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("sample_id 不唯一")
    missing_images = [r["image_path"] for r in rows if not Path(r["image_path"]).is_file()]
    if missing_images:
        raise FileNotFoundError(f"缺少 {len(missing_images)} 张图片，首例: {missing_images[0]}")
    if args.dry_run:
        print(json.dumps({"rows": len(rows), "onnx_sha_prefix": file_sha256(ONNX_PATH)[:16],
                          "expected_prefix": EXPECTED_ONNX_SHA256[:16]}, ensure_ascii=False))
        print("[dry-run] 清单校验通过")
        return 0
    if args.limit:
        rows = rows[: args.limit]

    onnx_sha = file_sha256(ONNX_PATH)
    if onnx_sha != EXPECTED_ONNX_SHA256:
        raise RuntimeError(f"部署 ONNX SHA-256 不匹配: {onnx_sha}")
    with V1_CONTRACT.open("r", encoding="utf-8") as handle:
        v1 = json.load(handle)
    feature_names = list(v1["feature_names"])
    assert len(feature_names) == 63

    sys.path.insert(0, str(WORKTREE))
    os.chdir(WORKTREE)
    from inference import models as model_state  # noqa: E402
    from inference.detectors import (  # noqa: E402
        _predict_dianji_grade_detections,
        _predict_dianji_yolo_raw,
    )
    from inference.grade_features import (  # noqa: E402
        CLASS_NAMES,
        compute_plate_area,
        extract_grade_features,
    )

    # 激活合同临时指回 v1（63 列 / 18 类），使 _predict_dianji_grade_detections 走 ryw18 生产分支；
    # 仅影响本进程，不改 worktree 配置。
    model_state.dianji_grade_contract = v1
    model_state.dianji_grade_feature_names = tuple(feature_names)
    model_state.load_dianji_detector()
    assert model_state.slf_yolo is not None
    actual_names = [str(n) for n in
                    (model_state.slf_yolo.names.values() if isinstance(model_state.slf_yolo.names, dict)
                     else model_state.slf_yolo.names)]
    assert actual_names == list(CLASS_NAMES), f"检测器类别顺序不符: {actual_names}"
    extraction = model_state.MODELS_CONFIG["detectors"]["dianji"]
    print(f"[1/3] ONNX sha={onnx_sha[:12]}… 名称校验通过；dianji 配置 kind={extraction['kind']} "
          f"threshold={extraction.get('threshold')} options={extraction.get('options')}", flush=True)

    feature_rows: list[dict[str, Any]] = []
    per_box_rows: list[dict[str, Any]] = []
    fallback_count = 0
    t0 = time.perf_counter()
    for index, row in enumerate(rows, 1):
        image_path = Path(row["image_path"])
        image_bgr, image_gray = read_image_pair(image_path)
        image_height, image_width = image_bgr.shape[:2]
        plate_area = compute_plate_area(image_bgr)
        fallback = plate_area <= 0
        fallback_count += int(fallback)

        raw_detections = _predict_dianji_yolo_raw(image_bgr)
        detections = _predict_dianji_grade_detections(image_bgr, raw_detections)
        features = extract_grade_features(image_bgr, detections, feature_names, image_gray=image_gray)

        out_row: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "image_name": image_path.name,
            "image_path": str(image_path),
            "group_id": row["group_id"],
            "fold": row["fold"],
            "split": row["split"],
            "source_split": row["source_split"],
            "type_weak": int(row["type"]),
            "nodule_weak": int(row["nodule"]),
            "n_defects_weak": int(row["n_defects"]),
            "feature_version": FEATURE_VERSION,
            "image_width": image_width,
            "image_height": image_height,
            "plate_area": f"{plate_area:.2f}",
            "plate_area_fallback": int(fallback),
            "max_confidence": f"{max((d['conf'] for d in raw_detections), default=0.0):.6f}",
        }
        for name in feature_names:
            out_row[name] = f"{features[name]:.8f}"
        feature_rows.append(out_row)
        for det in detections:
            per_box_rows.append({
                "sample_id": row["sample_id"],
                "label": det["label"],
                "class_name": CLASS_NAMES[int(det["label"])],
                "conf": f"{det['conf']:.6f}",
                "cx": f"{det['cx']:.6f}",
                "cy": f"{det['cy']:.6f}",
                "w": f"{det['w']:.6f}",
                "h": f"{det['h']:.6f}",
            })
        if index == 1 or index % 50 == 0 or index == len(rows):
            rate = index / (time.perf_counter() - t0)
            eta_min = (len(rows) - index) / rate / 60 if rate else 0
            print(f"  [{index}/{len(rows)}] {rate:.2f} img/s, ETA {eta_min:.0f} min", flush=True)

    with OUT_FEATURES.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS + feature_names)
        writer.writeheader()
        writer.writerows(feature_rows)
    with OUT_PERBOX.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "sample_id", "label", "class_name", "conf", "cx", "cy", "w", "h"])
        writer.writeheader()
        writer.writerows(per_box_rows)

    counts = [float(r["detection_count"]) for r in feature_rows]
    class_counter: dict[str, int] = {}
    for box in per_box_rows:
        class_counter[box["class_name"]] = class_counter.get(box["class_name"], 0) + 1
    summary = {
        "task": "C2.2 电解镍 63 列特征提取（A3 部署 ONNX，弱标签同批 1,127 图）",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature_version": FEATURE_VERSION,
        "images": len(feature_rows),
        "boxes": len(per_box_rows),
        "detection_count": {"min": min(counts), "max": max(counts),
                            "mean": sum(counts) / len(counts)},
        "boxes_per_class": class_counter,
        "plate_area_fallbacks": fallback_count,
        "detector": {
            "onnx": str(ONNX_PATH), "sha256": onnx_sha,
            "models_yaml_entry": "detectors.dianji (同 A3 dianjie 权重 901ccf60)",
            "extraction_params": {"imgsz": 640, "conf": 0.25, "iou": 0.45, "device": "cpu"},
            "note": "口径与 v1 生产等级路径 _predict_dianji_yolo_raw 一致；显示路径 iou 0.7 与特征无关",
        },
        "features": {"count": len(feature_names), "schema": "43 BASE + 20 纹理（63 列，与电积镍 LGBM 同构）",
                     "implementation": "inference/grade_features.py extract_grade_features（5a7bb5c）"},
        "weaklabels_source": str(WEAKLABEL_CSV),
        "outputs": {"features_csv": str(OUT_FEATURES), "per_box_csv": str(OUT_PERBOX)},
        "provenance": {"worktree": str(WORKTREE), "commit": "5a7bb5c",
                       "python_env": "jcw_b3clean_wt/.venv (torch 2.12.1+cpu, ultralytics 8.4.66)"},
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("images", "boxes", "detection_count", "boxes_per_class",
                                              "plate_area_fallbacks")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
