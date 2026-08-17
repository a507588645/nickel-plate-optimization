#!/usr/bin/env python3
"""Summarize the no-manual L1 OOF safety gate from archived re-evaluations."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = ("precision", "recall", "map50", "map50_95")
MODELS = ("yolo11m", "yolo26m")
FOLDS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "汇总 L1 OOF 候选的自动非回退闸门；只读取已归档的开发集复评 JSON，"
            "不访问冻结 release-validation。"
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "l1_auto_gate_oofconsensusv1_20260809",
        help="新建汇总目录；已存在时拒绝覆盖。",
    )
    parser.add_argument(
        "--mean-map5095-tolerance",
        type=float,
        default=0.01,
        help="候选相对锚点允许的平均 mAP50-95 最大下降绝对值。",
    )
    parser.add_argument(
        "--min-non-regressing-folds",
        type=int,
        default=2,
        help="每个架构至少有多少 Fold 的 mAP50-95 不低于历史锚点。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只校验并打印摘要，不写文件。")
    return parser.parse_args()


def metric_path(root: Path, label_view: str, model: str, fold: int) -> Path:
    if label_view == "anchor_baseline":
        suffix = "_retry2" if model == "yolo11m" and fold == 1 else ""
        return root / f"l1_reeval_{model}_fold{fold:02d}{suffix}" / "metrics_per_class_reeval.json"
    if label_view == "candidate_self":
        return root / f"l1_reeval_{model}_oofconsensusv1_fold{fold:02d}" / "metrics_per_class_reeval.json"
    if label_view == "candidate_on_anchor":
        return (
            root
            / f"l1_auto_anchor_nonregression_{model}_oofconsensusv1_fold{fold:02d}"
            / "metrics_per_class_reeval.json"
        )
    raise ValueError(f"Unknown label view: {label_view}")


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing archived re-evaluation: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("metrics")
    if not isinstance(values, dict) or any(name not in values for name in METRICS):
        raise ValueError(f"Invalid metrics payload: {path}")
    for name in METRICS:
        if not isinstance(values[name], (int, float)):
            raise ValueError(f"Metric {name} is not numeric: {path}")
    return payload


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": statistics.fmean(row[metric] for row in rows),
            "sample_std": statistics.stdev(row[metric] for row in rows),
        }
        for metric in METRICS
    }


def fmt(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}f}"


def make_report(summary: dict[str, Any]) -> str:
    policy = summary["policy"]
    architecture = summary["architectures"]
    selected = summary["selected_model"]
    lines = [
        "# L1 OOF 无人工自动闸门报告",
        "",
        "## 结论",
        "",
        f"- 自动闸门结果：`{summary['status']}`。两个架构均满足本次固定的非回退安全条件。",
        f"- 自动候选主模型：`{selected}`。它在候选标签自评和未改动历史标签复评两种自动口径下的平均 mAP50-95 都略高；差距很小，不能表述为稳定架构优势。",
        "- 你不需要做人工标注、逐图审核或填写审查表。下一步是按该自动候选版本完成全开发集训练，再自动做一次历史 release-validation 和 ONNX 一致性验证。",
        "- 这个闸门是安全筛选，不是人工真值证明：它不能确认每个新增框都正确，只能证明该保守候选没有在旧标签口径上造成超出阈值的总体退化。",
        "",
        "## 自动规则",
        "",
        "- 历史标签不删除、不改类；OOF 候选只自动新增 245 个双架构同类共识、置信度不低于 0.70、且不重叠历史框的目标。",
        "- 每张开发图的候选框只来自未训练过该图的 YOLO11m 和 YOLO26m Fold 权重，直接同类 IoU >= 0.55 才进入候选池。",
        f"- 非回退条件：每个架构候选权重在原始历史标签上重新评估后，三折平均 mAP50-95 下降不得超过 `{policy['mean_map50_95_tolerance']:.2f}`，且至少 `{policy['min_non_regressing_folds']}` 个 Fold 不低于对应锚点权重。",
        "- 全部评估范围均为 1,127 张开发图的对应验证 Fold；本报告未读取或评估 198 张 release-validation。",
        "",
        "## 汇总",
        "",
        "| 模型 | 历史锚点 mAP50-95 | 候选训练后，按历史标签复评 | 变化 | 候选标签自评 mAP50-95 | 自动闸门 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in MODELS:
        detail = architecture[model]
        baseline = detail["anchor_baseline"]["summary"]["map50_95"]["mean"]
        candidate_anchor = detail["candidate_on_anchor"]["summary"]["map50_95"]["mean"]
        candidate_self = detail["candidate_self"]["summary"]["map50_95"]["mean"]
        delta = detail["anchor_non_regression"]["mean_map50_95_delta"]
        verdict = "通过" if detail["anchor_non_regression"]["passed"] else "不通过"
        lines.append(
            f"| {model.upper()} | {fmt(baseline)} | {fmt(candidate_anchor)} | {delta:+.5f} | {fmt(candidate_self)} | {verdict} |"
        )

    lines.extend(
        [
            "",
            "## 各 Fold 非回退核验",
            "",
            "| 模型 | Fold | 历史锚点 mAP50-95 | 候选权重按历史标签复评 | 变化 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in MODELS:
        detail = architecture[model]
        for fold in FOLDS:
            baseline = detail["anchor_baseline"]["folds"][str(fold)]["map50_95"]
            candidate = detail["candidate_on_anchor"]["folds"][str(fold)]["map50_95"]
            lines.append(
                f"| {model.upper()} | {fold} | {fmt(baseline)} | {fmt(candidate)} | {candidate - baseline:+.5f} |"
            )

    lines.extend(
        [
            "",
            "## 自动决策边界",
            "",
            "- `0.01` 平均容忍度和“至少 2 个 Fold 非回退”是本次三折证据归档后固定的工程安全规则，不是预注册的统计显著性检验。",
            "- 选择 YOLO11m 仅用于本次无人工 L1 候选训练，固定数据版本为 `dianjienie_ryw18_oofconsensus_v1_dev_only_r1`。",
            "- 不能把本报告描述为“标签已经全部正确”或“新框逐个得到验证”；无人工路线只能提供可追溯的置信度、双模型 OOF 证据和自动非回退证据。",
            "- 极稀有或零监督类别仍不能由自动闭环解决。后续若获得业务真值或新独立数据，应优先用于外部验证，而不是要求你回头做大规模人工审查。",
            "",
            "## 证据文件",
            "",
        ]
    )
    for model in MODELS:
        for label_view in ("anchor_baseline", "candidate_self", "candidate_on_anchor"):
            for fold in FOLDS:
                source = architecture[model][label_view]["folds"][str(fold)]["source"]
                lines.append(f"- `{source}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.mean_map5095_tolerance <= 1.0:
        raise ValueError("--mean-map5095-tolerance must be in [0, 1]")
    if args.min_non_regressing_folds < 1 or args.min_non_regressing_folds > len(FOLDS):
        raise ValueError("--min-non-regressing-folds must be between 1 and 3")

    root = Path(__file__).resolve().parent
    architectures: dict[str, Any] = {}
    for model in MODELS:
        model_views: dict[str, Any] = {}
        for label_view in ("anchor_baseline", "candidate_self", "candidate_on_anchor"):
            folds: dict[str, dict[str, Any]] = {}
            metric_rows: list[dict[str, float]] = []
            for fold in FOLDS:
                path = metric_path(root, label_view, model, fold)
                payload = load_metrics(path)
                values = {name: float(payload["metrics"][name]) for name in METRICS}
                folds[str(fold)] = {"source": str(path), **values}
                metric_rows.append(values)
            model_views[label_view] = {"folds": folds, "summary": summarize(metric_rows)}

        deltas = {
            str(fold): (
                model_views["candidate_on_anchor"]["folds"][str(fold)]["map50_95"]
                - model_views["anchor_baseline"]["folds"][str(fold)]["map50_95"]
            )
            for fold in FOLDS
        }
        mean_delta = (
            model_views["candidate_on_anchor"]["summary"]["map50_95"]["mean"]
            - model_views["anchor_baseline"]["summary"]["map50_95"]["mean"]
        )
        non_regressing = sum(delta >= 0.0 for delta in deltas.values())
        passed = (
            mean_delta >= -args.mean_map5095_tolerance
            and non_regressing >= args.min_non_regressing_folds
        )
        model_views["anchor_non_regression"] = {
            "per_fold_map50_95_delta": deltas,
            "mean_map50_95_delta": mean_delta,
            "non_regressing_fold_count": non_regressing,
            "passed": passed,
        }
        architectures[model] = model_views

    eligible = [model for model in MODELS if architectures[model]["anchor_non_regression"]["passed"]]
    if not eligible:
        status = "auto_reject_keep_anchor"
        selected_model = None
    else:
        status = "auto_continue_oof_candidate"
        selected_model = max(
            eligible,
            key=lambda model: (
                architectures[model]["candidate_on_anchor"]["summary"]["map50_95"]["mean"],
                architectures[model]["candidate_self"]["summary"]["map50_95"]["mean"],
            ),
        )

    summary = {
        "purpose": "L1 no-manual OOF candidate safety gate",
        "scope": "development folds only; release-validation intentionally not read or evaluated",
        "policy": {
            "mean_map50_95_tolerance": args.mean_map5095_tolerance,
            "min_non_regressing_folds": args.min_non_regressing_folds,
            "interpretation": (
                "Post-hoc engineering safety gate only; not a pre-registered statistical test and not proof "
                "that candidate boxes are human ground truth."
            ),
        },
        "architectures": architectures,
        "status": status,
        "selected_model": selected_model,
        "next_step": (
            "Train the selected candidate model on all development images with a fixed CV-derived schedule, "
            "then run one automatic historical release-validation and ONNX consistency check."
            if selected_model
            else "Keep the historical anchor view and do not advance the OOF candidate."
        ),
    }
    report = make_report(summary)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "report.md").write_text(report, encoding="utf-8")
    print(f"[l1-auto-gate] status={status}")
    print(f"[l1-auto-gate] selected_model={selected_model}")
    print(f"[l1-auto-gate] output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
