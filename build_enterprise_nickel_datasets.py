#!/usr/bin/env python3
"""用企业原图构建无版本号的硬链接数据集。"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DIANJI_GRADE_FOLDERS = {
    "9950板（2#）": "9950",
    "不锈钢专用镍（3#）": "3#",
}
IGNORED_DIY_FOLDERS = {"classify", "defect"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建企业电解镍/电积镍硬链接数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dianjie-source",
        type=Path,
        default=Path(r"E:/Users/23916/Pictures/label/电解/image-0306"),
    )
    parser.add_argument(
        "--dianji-source",
        type=Path,
        default=Path(r"E:/Users/23916/Pictures/label/电积镍板0309"),
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets"),
    )
    parser.add_argument(
        "--records-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/enterprise_dataset_records"),
        help="构建清单放在执行目录，不写入原始数据集目录",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def hardlink(source: Path, target: Path, dry_run: bool) -> str:
    if source.drive.lower() != target.drive.lower():
        raise ValueError(f"硬链接要求同一磁盘卷: {source} -> {target}")
    if dry_run:
        return "dry_run"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return "existing"
        raise FileExistsError(f"目标已存在且不是同一文件: {target}")
    os.link(source, target)
    if not os.path.samefile(source, target):
        raise OSError(f"硬链接校验失败: {target}")
    return "created"


def ensure_clean_destination(path: Path, dry_run: bool) -> None:
    if path.exists() and any(path.rglob("*")):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {path}")
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def write_records(path: Path, rows: list[dict[str, str]], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dianjie_output = args.datasets_root / "enterprise_dianjie_0306_detection_unlabeled"
    dianji_cls_output = args.datasets_root / "enterprise_dianji_0309_classification"
    dianji_det_output = args.datasets_root / "enterprise_dianji_0309_detection_unlabeled"
    for output in (dianjie_output, dianji_cls_output, dianji_det_output):
        ensure_clean_destination(output, args.dry_run)

    dianjie_rows: list[dict[str, str]] = []
    for source in image_files(args.dianjie_source):
        target = dianjie_output / "images" / source.name
        status = hardlink(source, target, args.dry_run)
        dianjie_rows.append({
            "source": str(source.resolve()),
            "hardlink": str(target),
            "status": status,
            "label_status": "unlabeled",
        })

    dianji_cls_rows: list[dict[str, str]] = []
    dianji_det_rows: list[dict[str, str]] = []
    for folder in sorted(p for p in args.dianji_source.iterdir() if p.is_dir()):
        if folder.name in IGNORED_DIY_FOLDERS:
            continue
        sources = image_files(folder)
        if folder.name in DIANJI_GRADE_FOLDERS:
            grade = DIANJI_GRADE_FOLDERS[folder.name]
            for source in sources:
                target = dianji_cls_output / "images" / grade / source.name
                status = hardlink(source, target, args.dry_run)
                dianji_cls_rows.append({
                    "source": str(source.resolve()),
                    "hardlink": str(target),
                    "grade": grade,
                    "grade_source": folder.name,
                    "status": status,
                })
        else:
            safe_hint = folder.name.replace("/", "_").replace("\\", "_")
            for source in sources:
                target = dianji_det_output / "images" / safe_hint / source.name
                status = hardlink(source, target, args.dry_run)
                dianji_det_rows.append({
                    "source": str(source.resolve()),
                    "hardlink": str(target),
                    "defect_hint": folder.name,
                    "status": status,
                    "label_status": "unlabeled",
                })

    if not dianjie_rows or not dianji_cls_rows or not dianji_det_rows:
        raise ValueError("源数据为空或目录规则未匹配，拒绝生成不完整数据集")

    write_records(args.records_root / "enterprise_dianjie_0306_detection.csv", dianjie_rows, args.dry_run)
    write_records(args.records_root / "enterprise_dianji_0309_classification.csv", dianji_cls_rows, args.dry_run)
    write_records(args.records_root / "enterprise_dianji_0309_detection.csv", dianji_det_rows, args.dry_run)

    if not args.dry_run:
        (dianjie_output / "README.md").write_text(
            "# 企业电解镍 0306 待标注检测集\n\n68 张原图硬链接。当前无可靠检测标签，不得直接训练。\n",
            encoding="utf-8",
        )
        (dianji_cls_output / "README.md").write_text(
            "# 企业电积镍 0309 明确等级分类集\n\n只包含目录名明确的 9950板（2#）和不锈钢专用镍（3#）；未创建 train/valid 划分。\n",
            encoding="utf-8",
        )
        (dianji_det_output / "README.md").write_text(
            "# 企业电积镍 0309 待标注检测集\n\n包含缺角、边部结瘤、电击板、氢氧化镍、烧板等缺陷提示目录，共 37 张。源 classify/defect 自制结果未使用。\n",
            encoding="utf-8",
        )

    print(f"企业电解镍待标注检测图片: {len(dianjie_rows)}")
    print(f"企业电积镍明确等级分类图片: {len(dianji_cls_rows)}")
    print(f"企业电积镍待标注检测图片: {len(dianji_det_rows)}")
    print(f"构建记录: {args.records_root}")


if __name__ == "__main__":
    main()
