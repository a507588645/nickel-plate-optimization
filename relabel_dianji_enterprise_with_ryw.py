#!/usr/bin/env python3
"""用电解镍 ryw.pt 重标企业 0309 图片，并生成电解镍规则辅助分类。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from dianjie_grade_rule import DianjieGradeRule, dianjie_grade_from_xyxy


_ORIGINAL_PROJECT = Path(r"E:/PythonProject/ImetricCodes/JinchuanPythonWebapi")
_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args: Any, **kwargs: Any) -> Any:
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 ryw.pt 重标企业 0309 数据并执行电解镍七类规则",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/dianji_enterprise_0309_v1"),
        help="已按企业目录名重建划分的独立数据集",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/dianji_enterprise_0309_dianjie_v2"),
        help="版本化输出目录",
    )
    parser.add_argument("--weights", type=Path, default=_ORIGINAL_PROJECT / "ryw.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--resize", type=int, nargs=2, default=[1024, 888], metavar=("W", "H"))
    return parser.parse_args()


def symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        raise FileExistsError(dst)
    os.symlink(src.resolve(), dst)


def load_model(weights: Path, device: str, conf: float, iou: float) -> Any:
    sys.path.insert(0, str(_ORIGINAL_PROJECT))
    os.chdir(_ORIGINAL_PROJECT)
    from model_decryptor import load_yolov5_with_encryption

    model = load_yolov5_with_encryption(str(weights), device=device, autoshape=True)
    model.conf = conf
    model.iou = iou
    return model


def plate_area(resized: np.ndarray) -> float:
    from preprocess import findcent

    try:
        _, width, height, _ = findcent(resized)
        return float(width * height)
    except Exception:
        return float(resized.shape[0] * resized.shape[1])


def main() -> None:
    args = parse_args()
    manifest_path = args.dataset_root / "manifests" / "samples.csv"
    samples = pd.read_csv(manifest_path)
    model = load_model(args.weights, args.device, args.conf_thres, args.iou_thres)
    names = list(model.model.names.values())
    rule = DianjieGradeRule()

    records: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    type_counts: Counter[int] = Counter()
    class_counts: Counter[int] = Counter()

    for index, row in samples.iterrows():
        source = Path(row["image_path"])
        split = row["split"]
        enterprise_class = row["class_name"]
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"无法读取图片: {source}")
        resized = cv2.resize(image, tuple(args.resize))
        area = plate_area(resized)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model(resized)
        dets = result.xyxy[0].cpu().numpy() if len(result.xyxy) else np.empty((0, 6))
        grade = dianjie_grade_from_xyxy(dets, area, rule=rule)
        type_counts[grade.type_id] += 1

        image_name = source.name
        stem = source.stem
        symlink(source, args.output_root / "images" / split / image_name)
        symlink(source, args.output_root / "classification_truth" / split / enterprise_class / image_name)
        symlink(source, args.output_root / "dianjie_rule_aux" / split / str(grade.type_id) / image_name)

        raw_lines: list[str] = []
        train_lines: list[str] = []
        height, width = resized.shape[:2]
        for det_index, det in enumerate(dets):
            x1, y1, x2, y2, conf, class_id_raw = det
            class_id = int(class_id_raw)
            class_counts[class_id] += 1
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            raw_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}")
            train_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            prediction_rows.append({
                "image_path": str(source),
                "split": split,
                "enterprise_class": enterprise_class,
                "detection_index": det_index,
                "class_id": class_id,
                "class_name": names[class_id] if 0 <= class_id < len(names) else str(class_id),
                "confidence": float(conf),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            })

        raw_path = args.output_root / "labels_teacher_ryw_v1_raw" / split / f"{stem}.txt"
        train_path = args.output_root / "labels_teacher_ryw_v1" / split / f"{stem}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("\n".join(raw_lines) + ("\n" if raw_lines else ""), encoding="utf-8")
        train_path.write_text("\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8")

        records.append({
            "image_path": str(source),
            "split": split,
            "enterprise_class": enterprise_class,
            "business_subtype": row["business_subtype"],
            "teacher_model": str(args.weights),
            "prediction_count": len(dets),
            "dianjie_type": grade.type_id,
            "dianjie_type_name": grade.type_name,
            "has_nodule": grade.has_nodule,
            **grade.ratios,
        })
        if (index + 1) % 25 == 0:
            print(f"已处理 {index + 1}/{len(samples)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_root / "samples_with_dianjie_rule.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(args.output_root / "predictions_ryw.csv", index=False, encoding="utf-8-sig")
    summary = {
        "version": "v2",
        "classification_truth_source": "enterprise folder names",
        "original_manual_defect_labels_used": False,
        "teacher_model": str(args.weights.resolve()),
        "conf_thres": args.conf_thres,
        "iou_thres": args.iou_thres,
        "resize": args.resize,
        "samples": len(samples),
        "enterprise_distribution": samples["class_name"].value_counts().to_dict(),
        "dianjie_rule_distribution": {str(k): v for k, v in sorted(type_counts.items())},
        "teacher_classes": names,
        "prediction_class_distribution": {
            names[k] if 0 <= k < len(names) else str(k): v for k, v in sorted(class_counts.items())
        },
        "warning": "电解镍规则结果仅为辅助特征，不覆盖企业3#/9950分类真值。",
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
