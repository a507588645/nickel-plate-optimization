#!/usr/bin/env python3
"""按明确目录名和原电积镍四分类模型构建企业 0309 等级 v3。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from grade_classifier_unified_api_v2 import UnifiedGradeAPI


EXPLICIT_GRADES = {
    "9950板（2#）": "9950",
    "不锈钢专用镍（3#）": "3#",
}


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        raise FileExistsError(dst)
    os.symlink(src.resolve(), dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="重建企业 0309 原电积镍四分类等级")
    parser.add_argument("--source-root", type=Path, default=Path(r"E:/PythonProject/datasets/dianji_enterprise_0309_v1"))
    parser.add_argument("--ryw-root", type=Path, default=Path(r"E:/PythonProject/datasets/dianji_enterprise_0309_dianjie_v2"))
    parser.add_argument("--output-root", type=Path, default=Path(r"E:/PythonProject/datasets/dianji_enterprise_0309_grade_v3"))
    args = parser.parse_args()

    samples = pd.read_csv(args.source_root / "manifests" / "samples.csv")
    api = UnifiedGradeAPI(device="cpu")
    records = []
    for index, row in samples.iterrows():
        image = Path(row.image_path)
        subtype = row.business_subtype
        split = row.split
        inference_image = args.output_root / "images" / split / image.name
        link(image, inference_image)
        explicit = EXPLICIT_GRADES.get(subtype)
        if explicit:
            grade = explicit
            source = "explicit_folder_name"
            confidence = 1.0
            probabilities = {grade: 1.0}
        else:
            result = api.predict_dianji(inference_image)["data"]
            grade = result["type_name"]
            source = "original_dianji_classifier"
            confidence = result["confidence"]
            probabilities = result["probabilities"]

        link(image, args.output_root / "classification" / split / grade / image.name)
        for label_dir in ("labels_teacher_ryw_v1", "labels_teacher_ryw_v1_raw"):
            src_label = args.ryw_root / label_dir / split / f"{image.stem}.txt"
            link(src_label, args.output_root / label_dir / split / src_label.name)
        records.append({
            "image_path": str(image), "split": split, "business_subtype": subtype,
            "grade": grade, "grade_source": source, "confidence": confidence,
            "probabilities": json.dumps(probabilities, ensure_ascii=False),
        })
        if (index + 1) % 10 == 0:
            print(f"已处理 {index + 1}/{len(samples)}")

    frame = pd.DataFrame(records)
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_root / "grade_assignments.csv", index=False, encoding="utf-8-sig")
    summary = {
        "version": "v3",
        "samples": len(frame),
        "explicit_folder_rules": EXPLICIT_GRADES,
        "explicit_count": int((frame.grade_source == "explicit_folder_name").sum()),
        "model_assigned_count": int((frame.grade_source == "original_dianji_classifier").sum()),
        "grade_distribution": frame.grade.value_counts().to_dict(),
        "split_distribution": {f"{s}/{g}": int(n) for (s, g), n in frame.groupby(["split", "grade"]).size().items()},
        "defect_labels": "ryw.pt v1 from dianji_enterprise_0309_dianjie_v2",
        "manual_defect_labels_used": False,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
