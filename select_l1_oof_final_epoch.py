#!/usr/bin/env python3
"""Select a fixed final-training epoch from completed OOF model folds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUNS_ROOT = Path(r"E:/PythonProject/models/runs")
DATA_VERSION = "oofconsensusv1"
MODEL_OPTIONS = ("yolo11m.pt", "yolo26m.pt")
DEFAULT_MODEL = "yolo11m.pt"
FOLDS = (1, 2, 3)
DEFAULT_OUTPUT_ROOTS = {
    "yolo11m.pt": ROOT / "l1_final_epoch_selection_oofconsensusv1_yolo11m_20260809",
    "yolo26m.pt": ROOT / "l1_final_epoch_selection_oofconsensusv1_yolo26m_20260815",
}


def model_label(model: str) -> str:
    return model.removesuffix(".pt").replace("yolo", "YOLO", 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从完成的 OOF 三折训练曲线选择全开发集固定训练轮数；"
            "只读取归档 results.csv 和 requested_config.json。"
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=RUNS_ROOT,
        help="已归档 L1 run 的根目录。",
    )
    parser.add_argument("--model", choices=MODEL_OPTIONS, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, help="新建的选择报告目录；已存在时拒绝覆盖。")
    parser.add_argument("--dry-run", action="store_true", help="只读取和打印摘要，不写文件。")
    return parser.parse_args()


def run_dir(runs_root: Path, model: str, fold: int) -> Path:
    return runs_root / f"dianjienie_medium_{model.removesuffix('.pt')}_640_kaggle_seed0_cm0_{DATA_VERSION}_fold{fold:02d}"


def read_curve(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少训练曲线：{path}")
    rows: list[dict[str, float | int]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        metric_key = "metrics/mAP50-95(B)"
        if reader.fieldnames is None or metric_key not in reader.fieldnames or "epoch" not in reader.fieldnames:
            raise ValueError(f"results.csv 不含预期 epoch/mAP50-95 字段：{path}")
        for row in reader:
            try:
                epoch = int(row["epoch"])
                map50_95 = float(row[metric_key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"results.csv 存在无法解析的行：{path}") from error
            rows.append({"epoch": epoch, "map50_95": map50_95})
    if not rows:
        raise ValueError(f"results.csv 为空：{path}")
    if len({int(row['epoch']) for row in rows}) != len(rows):
        raise ValueError(f"results.csv 存在重复 epoch：{path}")
    return rows


def validate_requested_config(path: Path, model: str, fold: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少归档训练配置：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model": model,
        "data_version": DATA_VERSION,
        "development_only": True,
        "fold": fold,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"归档训练配置不匹配 {key}={payload.get(key)!r}，预期 {expected!r}: {path}")
    training = payload.get("training")
    if not isinstance(training, dict) or training.get("imgsz") != 640:
        raise ValueError(f"归档训练配置不是预期 640 OOF 基线：{path}")
    return payload


def format_row(row: dict[str, float | int]) -> str:
    return (
        f"| {row['epoch']} | {float(row['mean_map50_95']):.5f} | "
        f"{float(row['sample_std']):.5f} | {float(row['fold_01']):.5f} | "
        f"{float(row['fold_02']):.5f} | {float(row['fold_03']):.5f} |"
    )


def make_report(summary: dict[str, Any]) -> str:
    selected = summary["selected"]
    top = summary["top_epochs"]
    label = model_label(str(summary["model"]))
    lines = [
        f"# L1 OOF {label} 最终固定训练轮数选择",
        "",
        "## 结论",
        "",
        f"- 固定训练轮数：`{selected['epoch']}`。它是三个 OOF Fold 同时存在的 {summary['common_epoch_count']} 个 epoch 中，训练期平均 mAP50-95 最高的点。",
        f"- 该 epoch 的三折训练期平均 mAP50-95 为 `{selected['mean_map50_95']:.5f}`，样本标准差为 `{selected['sample_std']:.5f}`。",
        "- 全开发集训练使用固定轮数、`val=false`、`patience=0`，并使用 `last.pt` 的副本作为候选权重；Ultralytics 仍可能记录最后一个 epoch 的框架收尾验证，但它不参与 epoch 或发布权重选择。",
        "- 此选择只用于固定训练时长，不是标签质量或架构优劣证明；标签版本的自动安全判断见 `../l1_auto_gate_oofconsensusv1_20260809/report.md`。",
        "",
        "## 选择范围",
        "",
        f"- 仅使用 `{summary['model']}`、`oofconsensusv1`、`development_only=true` 的已归档三折 `results.csv`。",
        "- Fold 2 最先早停，因此只比较三个 Fold 都具有记录的共同 epoch；最后共同 epoch 为 "
        f"`{summary['last_common_epoch']}`。",
        "- 训练期指标来自各 Fold 的独立验证折，但不与历史锚点跨标签直接排名。",
        "",
        "## 排名前 15 的共同 Epoch",
        "",
        "| epoch | 平均 mAP50-95 | 样本标准差 | Fold 1 | Fold 2 | Fold 3 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(format_row(row) for row in top)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    curves: dict[int, dict[int, float]] = {}
    sources: dict[str, str] = {}
    for fold in FOLDS:
        directory = run_dir(runs_root, args.model, fold)
        config_path = directory / "requested_config.json"
        curve_path = directory / "results.csv"
        validate_requested_config(config_path, args.model, fold)
        curves[fold] = {int(row["epoch"]): float(row["map50_95"]) for row in read_curve(curve_path)}
        sources[str(fold)] = str(curve_path)

    common_epochs = sorted(set.intersection(*(set(curve) for curve in curves.values())))
    if not common_epochs:
        raise ValueError("三个 Fold 没有共同训练 epoch，不能选择固定训练轮数")
    rows = []
    for epoch in common_epochs:
        values = [curves[fold][epoch] for fold in FOLDS]
        rows.append(
            {
                "epoch": epoch,
                "mean_map50_95": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                **{f"fold_{fold:02d}": curves[fold][epoch] for fold in FOLDS},
            }
        )
    top = sorted(rows, key=lambda row: (-float(row["mean_map50_95"]), int(row["epoch"])))[:15]
    selected = top[0]
    summary = {
        "purpose": "fixed epoch selection for no-manual L1 final development training",
        "model": args.model,
        "data_version": DATA_VERSION,
        "development_only": True,
        "metric": "metrics/mAP50-95(B)",
        "sources": sources,
        "common_epoch_count": len(common_epochs),
        "last_common_epoch": common_epochs[-1],
        "selected": selected,
        "top_epochs": top,
        "final_training_policy": {
            "epochs": selected["epoch"],
            "validation": False,
            "patience": 0,
            "weight_for_release_validation": "weights/final_fixed_epoch.pt copied from weights/last.pt",
        },
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    output_root = (args.output_root or DEFAULT_OUTPUT_ROOTS[args.model]).resolve()
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有输出目录：{output_root}")
    output_root.mkdir(parents=True)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "report.md").write_text(make_report(summary), encoding="utf-8")
    print(f"[l1-final-epoch] selected_epoch={selected['epoch']}")
    print(f"[l1-final-epoch] output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
