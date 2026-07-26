#!/usr/bin/env python3
"""可视化汇总伪标签数据集的电解镍等级特征与规则划分效果。

输出：
- 各等级分布图
- 关键特征（A/B/C/Airhole 比例、plate_area、检测框并集面积比）按等级的箱线图/小提琴图
- 硬编码阈值在 A/B/C 比例空间中的划分边界散点图
- 各缺陷类别面积比按等级的箱线图
- 一份 summary_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", font_scale=1.1)
# 在 set_theme 之后重新设置中文字体，避免被覆盖
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

TYPE_ID_ORDER = [3, 4, 1, 6, 2, 5, 0]
TYPE_NAMES = {
    0: "无板",
    1: "3号板",
    2: "烫洗板",
    3: "合格整版",
    4: "合格600",
    5: "合格不足600",
    6: "2号板",
}
TYPE_COLORS_BY_ID = {
    0: "#999999",
    1: "#e41a1c",
    2: "#ff7f00",
    3: "#4daf4a",
    4: "#377eb8",
    5: "#984ea3",
    6: "#f781bf",
}
TYPE_COLORS = {TYPE_NAMES[k]: v for k, v in TYPE_COLORS_BY_ID.items()}

# 与 dianjie_grade_rule.py 默认阈值一致，用于在图上画参考线
RULE_THRESHOLDS = {
    "good_a_ratio": 0.2,
    "good_b_ratio": 0.1,
    "good_airhole_ratio": 0.01,
    "good_c_ratio": 0.03,
    "wash_b_ratio": 0.05,
    "wash_c_ratio": 0.05,
    "c_severe_ratio": 0.5,
    "b_severe_ratio": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="可视化电解镍等级特征与规则划分",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=Path(r"E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v2_grade/samples.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(r"E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v2_grade/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:/PythonProject/优化/grade_feature_analysis/combined_all_pseudo_v2_grade/visualizations"),
    )
    parser.add_argument("--max-scatter", type=int, default=2000, help="散点图最多采样点数")
    return parser.parse_args()


def load_data(samples_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(samples_csv, encoding="utf-8-sig")
    # 处理找不到图片的 fallback 行：plate_area 为默认值，可能导致比例极小/极大，单独标记
    df["findcent_fallback"] = df["findcent_fallback"].astype(bool)
    df["type_id_detection_only"] = df["type_id_detection_only"].astype(int)
    df["type_name"] = df["type_id_detection_only"].map(TYPE_NAMES)
    df["type_name"] = pd.Categorical(df["type_name"], categories=[TYPE_NAMES[i] for i in TYPE_ID_ORDER], ordered=True)
    return df


def savefig(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    counts = df["type_name"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index.astype(str), counts.values, color=[TYPE_COLORS_BY_ID[k] for k in TYPE_ID_ORDER if k in TYPE_NAMES])
    ax.set_ylabel("图片数")
    ax.set_title(f"等级分布（detection_only，n={len(df)}）")
    ax.tick_params(axis="x", rotation=30)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}\n({height / len(df) * 100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return savefig(fig, output_dir, "01_grade_distribution.png")


def plot_ratio_box(df: pd.DataFrame, output_dir: Path) -> Path:
    ratio_cols = ["A_ratio", "B_ratio", "C_ratio", "Airhole_ratio"]
    melted = df.melt(id_vars=["type_name"], value_vars=ratio_cols, var_name="ratio_kind", value_name="ratio_value")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, col in enumerate(ratio_cols):
        ax = axes[idx]
        sns.boxplot(data=melted[melted["ratio_kind"] == col], x="type_name", y="ratio_value", ax=ax, palette=TYPE_COLORS)
        ax.set_xlabel("")
        ax.set_ylabel(col)
        ax.set_title(f"{col} 按等级分布")
        ax.tick_params(axis="x", rotation=30)
        # 画硬编码阈值参考线
        threshold_key = {
            "A_ratio": "good_a_ratio",
            "B_ratio": "good_b_ratio",
            "C_ratio": "good_c_ratio",
            "Airhole_ratio": "good_airhole_ratio",
        }[col]
        ax.axhline(RULE_THRESHOLDS[threshold_key], color="red", linestyle="--", linewidth=1.2, label=f"阈值={RULE_THRESHOLDS[threshold_key]}")
        ax.legend(loc="upper right", fontsize=8)
    return savefig(fig, output_dir, "02_ratio_boxplots.png")


def plot_ratio_violin(df: pd.DataFrame, output_dir: Path) -> Path:
    ratio_cols = ["A_ratio", "B_ratio", "C_ratio", "Airhole_ratio"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, col in enumerate(ratio_cols):
        ax = axes[idx]
        sns.violinplot(data=df, x="type_name", y=col, ax=ax, palette=TYPE_COLORS, inner="box", cut=0)
        ax.set_xlabel("")
        ax.set_ylabel(col)
        ax.set_title(f"{col} 按等级小提琴图")
        ax.tick_params(axis="x", rotation=30)
        threshold_key = {
            "A_ratio": "good_a_ratio",
            "B_ratio": "good_b_ratio",
            "C_ratio": "good_c_ratio",
            "Airhole_ratio": "good_airhole_ratio",
        }[col]
        ax.axhline(RULE_THRESHOLDS[threshold_key], color="red", linestyle="--", linewidth=1.2)
    return savefig(fig, output_dir, "03_ratio_violins.png")


def plot_scatter_with_thresholds(df: pd.DataFrame, output_dir: Path, max_points: int = 2000) -> Path:
    """在 B_ratio/C_ratio 等二维空间画散点，并标出规则边界。"""
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # B vs C，标出 2号板/3号板 的硬编码边界
    ax = axes[0]
    for tid in TYPE_ID_ORDER:
        sub = df[df["type_id_detection_only"] == tid]
        if sub.empty:
            continue
        ax.scatter(sub["B_ratio"], sub["C_ratio"], c=TYPE_COLORS_BY_ID[tid], label=TYPE_NAMES[tid], alpha=0.5, s=15)
    ax.axvline(RULE_THRESHOLDS["b_severe_ratio"], color="black", linestyle="--", linewidth=1)
    ax.axhline(RULE_THRESHOLDS["c_severe_ratio"], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("B_ratio")
    ax.set_ylabel("C_ratio")
    ax.set_title("B_ratio vs C_ratio（硬编码：B>0.5且C<0.5→2号板；C>0.5→3号板）")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, min(1.0, df["B_ratio"].quantile(0.999) * 1.1))
    ax.set_ylim(0, min(1.0, df["C_ratio"].quantile(0.999) * 1.1))

    # A vs B，标出 合格整版 / 烫洗板 边界
    ax = axes[1]
    for tid in TYPE_ID_ORDER:
        sub = df[df["type_id_detection_only"] == tid]
        if sub.empty:
            continue
        ax.scatter(sub["A_ratio"], sub["B_ratio"], c=TYPE_COLORS_BY_ID[tid], label=TYPE_NAMES[tid], alpha=0.5, s=15)
    ax.axvline(RULE_THRESHOLDS["good_a_ratio"], color="black", linestyle="--", linewidth=1)
    ax.axhline(RULE_THRESHOLDS["good_b_ratio"], color="black", linestyle="--", linewidth=1)
    ax.axhline(RULE_THRESHOLDS["wash_b_ratio"], color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("A_ratio")
    ax.set_ylabel("B_ratio")
    ax.set_title("A_ratio vs B_ratio（硬编码：A<0.2,B<0.1→合格整版；A>=0.2,B<0.05→烫洗板）")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, min(1.0, df["A_ratio"].quantile(0.999) * 1.1))
    ax.set_ylim(0, min(1.0, df["B_ratio"].quantile(0.999) * 1.1))

    return savefig(fig, output_dir, "04_scatter_thresholds.png")


def plot_plate_and_union_area(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sns.boxplot(data=df, x="type_name", y="plate_area", ax=ax, palette=TYPE_COLORS)
    ax.set_xlabel("")
    ax.set_ylabel("plate_area")
    ax.set_title("plate_area 按等级分布")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1]
    sns.boxplot(data=df, x="type_name", y="total_bbox_union_image_ratio", ax=ax, palette=TYPE_COLORS)
    ax.set_xlabel("")
    ax.set_ylabel("total_bbox_union_image_ratio")
    ax.set_title("检测框并集面积比 按等级分布")
    ax.tick_params(axis="x", rotation=30)

    return savefig(fig, output_dir, "05_plate_and_union_area.png")


def plot_defect_area_ratios(df: pd.DataFrame, output_dir: Path) -> Path:
    area_cols = [c for c in df.columns if c.startswith("area_ratio__")]
    if not area_cols:
        return None

    # 只保留有缺陷的类别
    active_cols = [c for c in area_cols if df[c].max() > 0]
    n_cols = len(active_cols)
    if n_cols == 0:
        return None

    n_rows = math.ceil(n_cols / 3)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]

    for idx, col in enumerate(active_cols):
        ax = axes[idx]
        sns.boxplot(data=df, x="type_name", y=col, ax=ax, palette=TYPE_COLORS)
        ax.set_xlabel("")
        ax.set_ylabel(col.replace("area_ratio__", ""))
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=30)

    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    return savefig(fig, output_dir, "06_defect_area_ratios.png")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    numeric_cols = ["A_ratio", "B_ratio", "C_ratio", "Airhole_ratio", "plate_area", "detection_count", "total_bbox_union_image_ratio"]
    area_cols = [c for c in df.columns if c.startswith("area_ratio__")]
    numeric_cols += area_cols
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("数值特征相关性")
    return savefig(fig, output_dir, "07_correlation_heatmap.png")


def generate_report(df: pd.DataFrame, summary: dict[str, Any], output_dir: Path, paths: dict[str, Path]) -> Path:
    report_path = output_dir / "summary_report.md"
    combined_root = summary.get("combined_root", "")
    dataset_name = Path(str(combined_root)).name or "unknown_dataset"

    dist = df["type_name"].value_counts().sort_index()
    lines = [
        f"# {dataset_name} 电解镍等级特征可视化报告",
        "",
        f"- 数据集：`{combined_root or 'unknown'}`",
        f"- 样本总数：{len(df)}",
        f"- 置信度阈值：{summary.get('conf_threshold', 'unknown')}",
        f"- 不可读/回退图片数：{summary.get('audit', {}).get('findcent_fallbacks', 'unknown')}",
        f"- 过滤后无框图片数：{summary.get('audit', {}).get('empty_after_filter', 'unknown')}",
        "- 限制：本报告基于模型伪标签和硬编码规则，不是人工确认的企业等级真值评测。",
        "- 限制：当前规则没有返回 type 5（合格不足600）的分支；该类为 0 不能解释为数据中不存在该等级。",
        "",
        "## 等级分布（detection_only）",
        "",
        "| 等级 | 数量 | 占比 |",
        "| --- | --- | --- |",
    ]
    for name, count in dist.items():
        lines.append(f"| {name} | {count} | {count / len(df) * 100:.2f}% |")

    lines += [
        "",
        "## 关键特征按等级统计",
        "",
        "| 等级 | A_ratio 均值 | B_ratio 均值 | C_ratio 均值 | Airhole_ratio 均值 | plate_area 均值 | detection_count 均值 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    stats = df.groupby("type_name")[
        ["A_ratio", "B_ratio", "C_ratio", "Airhole_ratio", "plate_area", "detection_count"]
    ].mean().reindex(dist.index)
    for name, row in stats.iterrows():
        lines.append(
            f"| {name} | {row['A_ratio']:.4f} | {row['B_ratio']:.4f} | {row['C_ratio']:.4f} | "
            f"{row['Airhole_ratio']:.4f} | {row['plate_area']:.1f} | {row['detection_count']:.1f} |"
        )

    lines += [
        "",
        "## 硬编码规则阈值",
        "",
        "| 阈值 | 数值 | 含义 |",
        "| --- | --- | --- |",
        f"| good_a_ratio | {RULE_THRESHOLDS['good_a_ratio']} | 合格整版要求 A 类面积比 < 此值 |",
        f"| good_b_ratio | {RULE_THRESHOLDS['good_b_ratio']} | 合格整版要求 B 类面积比 < 此值 |",
        f"| good_airhole_ratio | {RULE_THRESHOLDS['good_airhole_ratio']} | 合格整版要求气孔面积比 < 此值 |",
        f"| good_c_ratio | {RULE_THRESHOLDS['good_c_ratio']} | 合格整版要求 C 类面积比 < 此值 |",
        f"| wash_b_ratio | {RULE_THRESHOLDS['wash_b_ratio']} | 烫洗板要求 B 类面积比 < 此值 |",
        f"| wash_c_ratio | {RULE_THRESHOLDS['wash_c_ratio']} | 烫洗板要求 C 类面积比 < 此值 |",
        f"| b_severe_ratio | {RULE_THRESHOLDS['b_severe_ratio']} | B 类面积比 > 此值 → 2号板 |",
        f"| c_severe_ratio | {RULE_THRESHOLDS['c_severe_ratio']} | C 类面积比 > 此值 → 3号板 |",
        "",
        "## 生成图表",
        "",
    ]
    for title, path in paths.items():
        lines.append(f"- [{title}]({path.name})")

    lines += [
        "",
        "## 观察结论（自动生成，需人工复核）",
        "",
    ]

    # 自动检查：合格整版是否大部分在低比例区域
    good = df[df["type_id_detection_only"] == 3]
    if len(good) > 0:
        lines.append(
            f"- 合格整版 {len(good)} 张，A_ratio 均值 {good['A_ratio'].mean():.4f}，"
            f"B_ratio 均值 {good['B_ratio'].mean():.4f}，C_ratio 均值 {good['C_ratio'].mean():.4f}。"
        )

    rare = df[df["type_id_detection_only"].isin([1, 2, 6])]
    if len(rare) > 0:
        lines.append(
            f"- 少数等级（3号板/烫洗板/2号板）共 {len(rare)} 张，样本极少，散点图可能被合格整版/合格600淹没。"
        )

    fallback = df[df["findcent_fallback"]]
    if len(fallback) > 0:
        lines.append(
            f"- 有 {len(fallback)} 张图片使用默认 plate_area 回退，其 plate_area 值集中在 {fallback['plate_area'].unique()}，"
            "可能使比例计算失真。"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.samples_csv)
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))

    paths: dict[str, Path] = {}
    paths["等级分布"] = plot_distribution(df, output_dir)
    paths["A/B/C/Airhole 比例箱线图"] = plot_ratio_box(df, output_dir)
    paths["A/B/C/Airhole 比例小提琴图"] = plot_ratio_violin(df, output_dir)
    paths["硬编码阈值散点图"] = plot_scatter_with_thresholds(df, output_dir, args.max_scatter)
    paths["plate_area 与检测框并集面积比"] = plot_plate_and_union_area(df, output_dir)

    defect_path = plot_defect_area_ratios(df, output_dir)
    if defect_path:
        paths["各类缺陷面积比"] = defect_path

    paths["特征相关性热力图"] = plot_correlation_heatmap(df, output_dir)

    report_path = generate_report(df, summary, output_dir, paths)

    print(f"可视化输出目录: {output_dir}")
    for title, path in paths.items():
        print(f"  {title}: {path}")
    print(f"  报告: {report_path}")


if __name__ == "__main__":
    main()
