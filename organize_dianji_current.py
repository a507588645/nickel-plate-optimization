#!/usr/bin/env python3
"""建立电积镍数据唯一入口，不移动或删除任何历史目录。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


LINKS = {
    "01_detection_original": Path(r"E:/PythonProject/datasets/dianjienie"),
    "02_classification_original_trusted": Path(r"E:/PythonProject/datasets/classify_4types_dianji"),
    "03_enterprise_defect_labels_ryw": Path(r"E:/PythonProject/datasets/dianji_enterprise_0309_dianjie_v2"),
    "04_enterprise_grade_candidate_review": Path(r"E:/PythonProject/datasets/dianji_enterprise_0309_grade_v3"),
    "05_features_original": Path(r"E:/PythonProject/优化/classify_from_detection_v1/dianji"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="建立电积镍数据集唯一入口")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/dianji_current"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, target in LINKS.items():
        if not target.exists():
            raise FileNotFoundError(target)
        link = args.output_root / name
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == target.resolve():
                continue
            raise FileExistsError(link)
        os.symlink(target.resolve(), link, target_is_directory=True)
        print(f"{name} -> {target}")


if __name__ == "__main__":
    main()
