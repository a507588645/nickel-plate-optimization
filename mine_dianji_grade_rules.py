#!/usr/bin/env python3
"""
电积镍 0309 企业已判级样本 → 浅层候选规则挖掘。

输入：企业 0309 分类目录（目录名=等级）+ combined_all_pseudo_v2 特征 CSV
输出：dianji_grade_rule_v1/ 下
  - grade_features_0309.csv
  - candidate_rules.json（status=unconfirmed_pending_enterprise）
  - report.md（含小样本警告与未确认横幅）
  - grade_results_0309.csv（引擎批量回判）
  - README.md

约束：
- 禁止把挖掘阈值表述为企业标准；
- 仅 3# / 9950 二分类（无 9996/bad 样本）；
- 优先 sklearn DecisionTree(max_depth=3)；不可用时退化为 numpy 网格搜索；
- 禁止全局 pip install。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 路径默认值
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DEFAULT_ENTERPRISE = Path(r"E:/PythonProject/datasets/enterprise_dianji_0309_classification")
DEFAULT_SAMPLES_CSV = ROOT / "grade_feature_analysis" / "combined_all_pseudo_v2_grade" / "samples.csv"
DEFAULT_OUT = ROOT / "dianji_grade_rule_v1"
DEFAULT_SEED = 42

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

GROUPS: dict[str, list[int]] = {
    "A": [0, 1],
    "B": [2, 4, 5, 6, 10, 12, 13],
    "C": [7, 8, 11],
    "AIRHOLE": [4],
    "NODULE": [3],
}

FEATURE_COLUMNS: list[str] = (
    ["A_ratio", "B_ratio", "C_ratio", "Airhole_ratio", "has_nodule", "total_bbox_union_image_ratio", "detection_count"]
    + [f"count__{n}" for n in CLASS_NAMES]
    + [f"area_ratio__{n}" for n in CLASS_NAMES]
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="从企业 0309 已判级样本挖掘电积镍浅层候选规则（未经企业确认）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--enterprise-root", type=Path, default=DEFAULT_ENTERPRISE)
    p.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES_CSV, help="combined_all_pseudo_v2 特征表")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--n-splits", type=int, default=5, help="分层 K 折；少数类不足时自动退化为 LOO")
    p.add_argument("--dry-run", action="store_true", help="只组装特征表并打印覆盖率，不挖掘/不写规则")
    p.add_argument(
        "--skip-engine-apply",
        action="store_true",
        help="跳过用引擎回写 grade_results_0309.csv",
    )
    return p.parse_args(argv)


def list_enterprise_images(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    images_root = root / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"企业分类 images 目录不存在: {images_root}")
    for path in sorted(images_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        grade = path.parent.name
        rows.append(
            {
                "image_stem": path.stem,
                "image_name_raw": path.name,
                "true_grade": grade,
                "image_path_enterprise": str(path.resolve()),
            }
        )
    return rows


def match_prefixed_name(image_name: str, stem: str, grade: str) -> bool:
    """匹配 combined 中的命名：enterprise_dianji_0309_classification_images_{grade}_{stem}.png"""
    stem_name = Path(image_name).stem
    token = f"images_{grade}_{stem}"
    if stem_name.endswith(token) or token in stem_name:
        return True
    # 宽松：以 _{stem} 结尾且含 0309 与 grade
    if stem_name.endswith(f"_{stem}") and "0309" in stem_name and grade in stem_name:
        return True
    return False


def join_features(
    enterprise_rows: list[dict[str, str]],
    samples_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not samples_csv.exists():
        raise FileNotFoundError(f"特征 CSV 不存在: {samples_csv}")

    samples_by_key: dict[tuple[str, str], dict[str, str]] = {}
    samples_by_stem: dict[str, list[dict[str, str]]] = {}
    with samples_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("image_name", "")
            samples_by_stem.setdefault(Path(name).stem, []).append(row)
            # 预索引 0309 前缀
            for g in ("3#", "9950", "9996", "bad"):
                token = f"images_{g}_"
                stem_full = Path(name).stem
                if token in stem_full:
                    s = stem_full.split(token, 1)[1]
                    samples_by_key[(s, g)] = row

    joined: list[dict[str, Any]] = []
    missing: list[str] = []
    for er in enterprise_rows:
        stem = er["image_stem"]
        grade = er["true_grade"]
        row = samples_by_key.get((stem, grade))
        if row is None:
            # 二次扫描
            for cand in samples_by_stem.get(stem, []):
                if match_prefixed_name(cand["image_name"], stem, grade):
                    row = cand
                    break
            if row is None:
                # 任意含 0309 + stem 的行
                for full_stem, cands in samples_by_stem.items():
                    if full_stem.endswith(f"_{stem}") and "0309" in full_stem:
                        for cand in cands:
                            if match_prefixed_name(cand["image_name"], stem, grade):
                                row = cand
                                break
                    if row is not None:
                        break
        if row is None:
            missing.append(f"{grade}/{er['image_name_raw']}")
            continue
        out: dict[str, Any] = {
            "image_stem": stem,
            "image_name": row.get("image_name", ""),
            "image_name_raw": er["image_name_raw"],
            "true_grade": grade,
            "image_path_enterprise": er["image_path_enterprise"],
            "image_path": row.get("image_path", ""),
            "label_path": row.get("label_path", ""),
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
        ):
            out[col] = row.get(col, "")
        for name in CLASS_NAMES:
            out[f"count__{name}"] = row.get(f"count__{name}", "0")
            out[f"area_ratio__{name}"] = row.get(f"area_ratio__{name}", "0")
        joined.append(out)

    meta = {
        "enterprise_total": len(enterprise_rows),
        "joined": len(joined),
        "missing": len(missing),
        "missing_list": missing,
        "coverage": len(joined) / len(enterprise_rows) if enterprise_rows else 0.0,
        "grade_counts_enterprise": dict(Counter(r["true_grade"] for r in enterprise_rows)),
        "grade_counts_joined": dict(Counter(r["true_grade"] for r in joined)),
        "features_source": str(samples_csv.resolve()),
        "join_method": "image_name prefix join with combined_all_pseudo_v2 samples.csv (no re-inference)",
        "detection_params_note": "特征来自既有 combined_all_pseudo_v2 / ryw.pt 伪标签派生；历史口径 conf=0.25,iou=0.45,img-size=1024",
        "weight_path": r"E:\PythonProject\Tools\ModelAutoLabelTool\weights\yolov5\ryw.pt",
    }
    return joined, meta


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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def matrix_from_rows(rows: list[dict[str, Any]], feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """X, y(0=9950,1=3#), labels 字符串。"""
    x_list = []
    y_list = []
    labels = []
    for r in rows:
        g = r["true_grade"]
        if g not in {"3#", "9950"}:
            continue
        vec = []
        for c in feature_cols:
            v = r.get(c, 0)
            if v is None or v == "":
                vec.append(0.0)
            else:
                try:
                    vec.append(float(v))
                except (TypeError, ValueError):
                    # bool 字符串
                    s = str(v).strip().lower()
                    vec.append(1.0 if s in {"1", "true", "yes"} else 0.0)
        x_list.append(vec)
        y_list.append(1 if g == "3#" else 0)
        labels.append(g)
    return np.asarray(x_list, dtype=np.float64), np.asarray(y_list, dtype=np.int64), labels


def try_import_sklearn():
    try:
        from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        return {
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "StratifiedKFold": StratifiedKFold,
            "LeaveOneOut": LeaveOneOut,
            "cross_val_predict": cross_val_predict,
            "export_text": export_text,
            "accuracy_score": accuracy_score,
            "confusion_matrix": confusion_matrix,
            "classification_report": classification_report,
        }
    except ImportError:
        return None


def tree_to_rules(
    clf: Any,
    feature_cols: list[str],
    class_names_y: list[str],
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """
    把决策树叶子路径导出为有序规则列表。
    3# 路径优先，再 9950；引擎按序首条命中。

    support 优先用未加权真实样本计数（对 X/y 应用路径条件），
    避免 class_weight='balanced' 时 tree.value 浮点加权被 int() 截断失真。
    """
    tree = clf.tree_
    rules_3: list[dict[str, Any]] = []
    rules_9950: list[dict[str, Any]] = []

    def apply_conditions(conditions: list[dict[str, Any]]) -> np.ndarray | None:
        if X is None or y is None:
            return None
        mask = np.ones(len(X), dtype=bool)
        name_to_idx = {n: i for i, n in enumerate(feature_cols)}
        for cond in conditions:
            j = name_to_idx[cond["feature"]]
            thr = float(cond["threshold"])
            op = cond["op"]
            if op == "<=":
                mask &= X[:, j] <= thr
            elif op == ">":
                mask &= X[:, j] > thr
            elif op == "<":
                mask &= X[:, j] < thr
            elif op == ">=":
                mask &= X[:, j] >= thr
            elif op == "==":
                mask &= X[:, j] == thr
        return mask

    def recurse(node: int, conditions: list[dict[str, Any]]) -> None:
        left = tree.children_left[node]
        right = tree.children_right[node]
        if left == right:  # leaf
            values = tree.value[node][0]
            pred_idx = int(np.argmax(values))
            grade = class_names_y[pred_idx]
            # 未加权真实支持
            mask = apply_conditions(conditions)
            if mask is not None:
                n_samples = int(mask.sum())
                # y: 0=9950, 1=3#
                support = {
                    "3#": int(((y == 1) & mask).sum()),
                    "9950": int(((y == 0) & mask).sum()),
                }
            else:
                n_samples = int(tree.n_node_samples[node])
                support = {
                    class_names_y[i]: int(round(float(values[i])))
                    for i in range(len(class_names_y))
                }
            rule = {
                "name": f"tree_leaf_{'3sharp' if grade == '3#' else grade}_{len(rules_3) + len(rules_9950)}",
                "grade": grade,
                "conditions": list(conditions),
                "support_samples": n_samples,
                "support_by_class": support,
                "source": "DecisionTreeClassifier",
            }
            if grade == "3#":
                rules_3.append(rule)
            else:
                rules_9950.append(rule)
            return
        feat_id = int(tree.feature[node])
        thr = float(tree.threshold[node])
        fname = feature_cols[feat_id]
        # sklearn: feature <= thr → left; feature > thr → right
        recurse(left, conditions + [{"feature": fname, "op": "<=", "threshold": thr}])
        recurse(right, conditions + [{"feature": fname, "op": ">", "threshold": thr}])

    recurse(0, [])
    rules_3.sort(key=lambda r: r.get("support_samples", 0), reverse=True)
    rules_9950.sort(key=lambda r: r.get("support_samples", 0), reverse=True)
    return rules_3 + rules_9950


def numpy_grid_search_rules(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    无 sklearn 时：单特征阈值 + 至多两条 AND 组合的网格搜索。
    目标：最大化平衡准确率（3#/9950 各 0.5）。
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # 候选阈值：分位数
    candidates: list[tuple[int, float]] = []
    for j in range(d):
        col = X[:, j]
        qs = np.unique(np.quantile(col, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
        for q in qs:
            candidates.append((j, float(q)))
    rng.shuffle(candidates)
    candidates = candidates[: min(len(candidates), 200)]

    def bal_acc(mask_pos: np.ndarray) -> tuple[float, dict[str, int]]:
        # mask_pos 预测为 3#
        tp = int(((y == 1) & mask_pos).sum())
        fn = int(((y == 1) & ~mask_pos).sum())
        tn = int(((y == 0) & ~mask_pos).sum())
        fp = int(((y == 0) & mask_pos).sum())
        rec1 = tp / (tp + fn) if (tp + fn) else 0.0
        rec0 = tn / (tn + fp) if (tn + fp) else 0.0
        return 0.5 * (rec0 + rec1), {"tp": tp, "fn": fn, "tn": tn, "fp": fp}

    best_single = None
    best_score = -1.0
    for j, thr in candidates:
        for op in (">", "<="):
            if op == ">":
                mask = X[:, j] > thr
            else:
                mask = X[:, j] <= thr
            score, cm = bal_acc(mask)
            if score > best_score:
                best_score = score
                best_single = (j, thr, op, mask, cm)

    best_and = None
    best_and_score = -1.0
    # 限制组合数量
    top = candidates[:80]
    for i1, (j1, t1) in enumerate(top):
        for j2, t2 in top[i1 + 1 :]:
            if j1 == j2:
                continue
            for op1 in (">", "<="):
                m1 = X[:, j1] > t1 if op1 == ">" else X[:, j1] <= t1
                for op2 in (">", "<="):
                    m2 = X[:, j2] > t2 if op2 == ">" else X[:, j2] <= t2
                    mask = m1 & m2
                    if mask.sum() < 3:
                        continue
                    score, cm = bal_acc(mask)
                    if score > best_and_score:
                        best_and_score = score
                        best_and = (j1, t1, op1, j2, t2, op2, mask, cm)

    rules: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"method": "numpy_grid_search", "best_single_score": best_score, "best_and_score": best_and_score}

    def add_rule(name: str, conds: list[dict[str, Any]], mask: np.ndarray, cm: dict[str, int]) -> None:
        support_3 = int(((y == 1) & mask).sum())
        support_9950 = int(((y == 0) & mask).sum())
        # 叶子多数类
        grade = "3#" if support_3 >= support_9950 else "9950"
        rules.append(
            {
                "name": name,
                "grade": grade,
                "conditions": conds,
                "support_samples": int(mask.sum()),
                "support_by_class": {"3#": support_3, "9950": support_9950},
                "train_confusion_if_pos_is_3sharp": cm,
                "source": "numpy_grid_search",
            }
        )

    if best_and is not None and best_and_score >= best_score:
        j1, t1, op1, j2, t2, op2, mask, cm = best_and
        add_rule(
            "grid_and_primary",
            [
                {"feature": feature_cols[j1], "op": op1, "threshold": float(t1)},
                {"feature": feature_cols[j2], "op": op2, "threshold": float(t2)},
            ],
            mask,
            cm,
        )
        # 互补：不满足 AND → 另一类
        rules.append(
            {
                "name": "grid_and_complement",
                "grade": "9950" if rules[0]["grade"] == "3#" else "3#",
                "conditions": [
                    # 用“非第一条”无法直接表达；改为低缺陷兜底
                    {"feature": feature_cols[j1], "op": "<=" if op1 == ">" else ">", "threshold": float(t1)},
                ],
                "support_samples": int((~mask).sum()),
                "support_by_class": {
                    "3#": int(((y == 1) & ~mask).sum()),
                    "9950": int(((y == 0) & ~mask).sum()),
                },
                "source": "numpy_grid_search",
            }
        )
    elif best_single is not None:
        j, thr, op, mask, cm = best_single
        add_rule(
            "grid_single_primary",
            [{"feature": feature_cols[j], "op": op, "threshold": float(thr)}],
            mask,
            cm,
        )
        rules.append(
            {
                "name": "grid_single_complement",
                "grade": "9950" if rules[0]["grade"] == "3#" else "3#",
                "conditions": [
                    {"feature": feature_cols[j], "op": "<=" if op == ">" else ">", "threshold": float(thr)},
                ],
                "support_samples": int((~mask).sum()),
                "support_by_class": {
                    "3#": int(((y == 1) & ~mask).sum()),
                    "9950": int(((y == 0) & ~mask).sum()),
                },
                "source": "numpy_grid_search",
            }
        )
    return rules, meta


def evaluate_rules_on_data(
    rules: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    """用与引擎相同的首条命中逻辑评估。"""
    from dianji_grade_rule import DianjiGradeRule

    engine = DianjiGradeRule(
        rules_data={
            "status": "unconfirmed_pending_enterprise",
            "generated_at": "",
            "sample_count": len(rows),
            "class_names": CLASS_NAMES,
            "groups": GROUPS,
            "rules": rules,
        }
    )
    results = []
    for r in rows:
        feat = engine.features_from_row(r)
        # 只保留数值特征列，features_from_row 已含全部
        res = engine.classify_features(feat)
        results.append(
            {
                "image_name": r.get("image_name", ""),
                "image_stem": r.get("image_stem", ""),
                "true_grade": r.get("true_grade", ""),
                "pred_grade": res.grade,
                "hit_rule": res.hit_rule,
                "status": res.status,
                "message": res.message,
                **{c: feat.get(c, "") for c in feature_cols if c in feat},
            }
        )
    return results


def rule_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [r["true_grade"] for r in results]
    y_pred = [r["pred_grade"] for r in results]
    labels = ["3#", "9950"]
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t not in labels or p not in labels:
            continue
        cm[labels.index(t), labels.index(p)] += 1
    total = int(cm.sum())
    acc = float(np.trace(cm) / total) if total else 0.0
    # 逐规则
    by_rule: dict[str, dict[str, int]] = {}
    for r in results:
        name = r["hit_rule"] or "(none)"
        bucket = by_rule.setdefault(name, {"n": 0, "correct": 0})
        bucket["n"] += 1
        if r["true_grade"] == r["pred_grade"]:
            bucket["correct"] += 1
    return {
        "accuracy": acc,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "manual_review_count": sum(1 for r in results if r["pred_grade"] == "manual_review"),
        "by_rule": {
            k: {
                "n": v["n"],
                "correct": v["correct"],
                "accuracy": v["correct"] / v["n"] if v["n"] else 0.0,
            }
            for k, v in by_rule.items()
        },
    }


def build_report(
    meta_join: dict[str, Any],
    mining_meta: dict[str, Any],
    rules: list[dict[str, Any]],
    cv_info: dict[str, Any],
    fit_metrics: dict[str, Any],
    generated_at: str,
) -> str:
    lines = [
        "# 电积镍 0309 浅层候选规则挖掘报告",
        "",
        "> ## ⚠️ 未经企业确认 / UNCONFIRMED",
        ">",
        "> 本报告全部阈值与规则均来自 **86 张企业 0309 已判级样本的统计挖掘**，",
        "> **不是**企业书面标准，**不得**直接上线生产。",
        "> 规则表 `status` 固定为 `unconfirmed_pending_enterprise`。",
        "",
        "> ## ⚠️ 小样本警告",
        ">",
        f"> - 总样本：{meta_join.get('enterprise_total', '?')} 张",
        f"> - 3#：{meta_join.get('grade_counts_enterprise', {}).get('3#', 0)} 张（极少）",
        f"> - 9950：{meta_join.get('grade_counts_enterprise', {}).get('9950', 0)} 张",
        "> - 无 9996 / bad 样本 → **禁止**为这两类编造规则；未命中 → `manual_review`",
        "> - 交叉验证分数波动大，仅作候选线索，需企业现场确认后才能固化。",
        "",
        "## 1. 数据与 join",
        "",
        f"- 生成时间：`{generated_at}`",
        f"- 企业目录：`E:/PythonProject/datasets/enterprise_dianji_0309_classification`",
        f"- 特征来源：`{meta_join.get('features_source')}`",
        f"- Join 方法：{meta_join.get('join_method')}",
        f"- Join 覆盖：`{meta_join.get('joined')}/{meta_join.get('enterprise_total')}` "
        f"= **{meta_join.get('coverage', 0)*100:.1f}%**",
        f"- 缺失：{meta_join.get('missing')} 张",
        f"- 检测权重（历史）：`{meta_join.get('weight_path')}`",
        f"- 检测参数说明：{meta_join.get('detection_params_note')}",
        "",
        "企业等级分布（join 后）：",
        "",
    ]
    for g, c in sorted(meta_join.get("grade_counts_joined", {}).items()):
        lines.append(f"- `{g}`: {c}")
    lines.extend(
        [
            "",
            "## 2. 挖掘方法",
            "",
            f"- 方法：`{mining_meta.get('method')}`",
            f"- 随机种子：`{mining_meta.get('seed')}`",
            f"- 特征维数：`{mining_meta.get('n_features')}`",
            f"- 可用特征列：`{', '.join(mining_meta.get('feature_columns', [])[:12])}...`",
            "",
            "### 交叉验证",
            "",
            f"- 策略：{cv_info.get('strategy')}",
            f"- 准确率：`{cv_info.get('accuracy', float('nan')):.4f}`",
            f"- 说明：{cv_info.get('note', '')}",
            "",
        ]
    )
    cm = cv_info.get("confusion_matrix")
    if cm:
        lines.append("混淆矩阵（行=真值，列=预测）：")
        lines.append("")
        labels = cm.get("labels", [])
        mat = cm.get("matrix", [])
        lines.append("|  | " + " | ".join(labels) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(labels)) + " |")
        for lab, row in zip(labels, mat):
            lines.append("| " + lab + " | " + " | ".join(str(x) for x in row) + " |")
        lines.append("")

    lines.extend(["## 3. 候选规则（有序，首条命中）", ""])
    if not rules:
        lines.append("*无规则导出*")
    for i, rule in enumerate(rules, 1):
        conds = rule.get("conditions") or []
        cond_txt = " AND ".join(
            f"`{c['feature']}` {c['op']} {c['threshold']:.6g}" for c in conds
        ) or "(空条件)"
        support = rule.get("support_by_class", {})
        lines.extend(
            [
                f"### 规则 {i}: `{rule.get('name')}` → **{rule.get('grade')}**",
                "",
                f"- 条件：{cond_txt}",
                f"- 训练叶节点支持：{rule.get('support_samples')} "
                f"（3#={support.get('3#', support.get(1, '?'))}, "
                f"9950={support.get('9950', support.get(0, '?'))}）",
                f"- 来源：{rule.get('source', '?')}",
                f"- **状态：未经企业确认**",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. 训练集上回判（非独立测试，仅描述拟合）",
            "",
            f"- 准确率：`{fit_metrics.get('accuracy', 0):.4f}`",
            f"- manual_review 数：{fit_metrics.get('manual_review_count', 0)}",
            "",
            "逐规则：",
            "",
        ]
    )
    for name, info in fit_metrics.get("by_rule", {}).items():
        lines.append(
            f"- `{name}`: n={info['n']}, correct={info['correct']}, acc={info['accuracy']:.3f}"
        )

    fit_cm = fit_metrics.get("confusion_matrix", {})
    if fit_cm:
        lines.extend(["", "回判混淆矩阵：", ""])
        labels = fit_cm.get("labels", [])
        mat = fit_cm.get("matrix", [])
        lines.append("|  | " + " | ".join(labels) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(labels)) + " |")
        for lab, row in zip(labels, mat):
            lines.append("| " + lab + " | " + " | ".join(str(x) for x in row) + " |")

    lines.extend(
        [
            "",
            "## 5. 规则含义（业务解读草稿，非确认）",
            "",
            "- 规则基于 18 类 `count__*` / `area_ratio__*` 与 A/B/C/Airhole 组面积比；",
            "  A/B/C 分组沿用电解镍初始假设，**不是**企业电积镍正式分组。",
            "- 决策树/网格搜索找到的阈值反映的是本批 0309 目录中 3# 与 9950 的可分性，",
            "  可能混入批次、光照、伪标签噪声，不能外推。",
            "- 上线前必须：① 企业书面标准对照；② 独立批次复核；③ 人工审查伪标签框质量。",
            "",
            "## 6. 复现命令",
            "",
            "```bash",
            r"cd /d E:\PythonProject\优化",
            r".\.venv_classify\Scripts\python.exe mine_dianji_grade_rules.py",
            r".\.venv_classify\Scripts\python.exe dianji_grade_rule.py --features dianji_grade_rule_v1\grade_features_0309.csv --rules dianji_grade_rule_v1\candidate_rules.json --output dianji_grade_rule_v1\grade_results_0309.csv",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_readme(generated_at: str, meta_join: dict[str, Any]) -> str:
    return f"""# dianji_grade_rule_v1 — 电积镍规则判级第一版骨架产物

> **status: `unconfirmed_pending_enterprise`**
>
> 全部候选规则与阈值来自 0309 小样本统计挖掘，**未经企业确认**，禁止当作生产标准。

## 生成时间

`{generated_at}`

## 文件清单

| 文件 | 说明 |
| --- | --- |
| `grade_features_0309.csv` | 86 张企业等级样本的 18 类特征 + 真值标签 |
| `candidate_rules.json` | 有序候选规则表（引擎直接加载） |
| `report.md` | 挖掘报告（含小样本警告与未确认横幅） |
| `grade_results_0309.csv` | 引擎对 0309 特征的逐图回判 |
| `README.md` | 本说明 |

## 数据来源

- 等级真值：`E:/PythonProject/datasets/enterprise_dianji_0309_classification/images/{{3#,9950}}/`
- 特征：`grade_feature_analysis/combined_all_pseudo_v2_grade/samples.csv`
- Join 覆盖：{meta_join.get('joined')}/{meta_join.get('enterprise_total')} = {meta_join.get('coverage', 0)*100:.1f}%
- 检测权重（历史伪标签）：`E:/PythonProject/Tools/ModelAutoLabelTool/weights/yolov5/ryw.pt`
- 检测参数说明：{meta_join.get('detection_params_note')}

## 复现

```bash
cd /d E:\\PythonProject\\优化
.\\.venv_classify\\Scripts\\python.exe mine_dianji_grade_rules.py
.\\.venv_classify\\Scripts\\python.exe dianji_grade_rule.py --self-test
.\\.venv_classify\\Scripts\\python.exe dianji_grade_rule.py --features dianji_grade_rule_v1\\grade_features_0309.csv --rules dianji_grade_rule_v1\\candidate_rules.json --output dianji_grade_rule_v1\\grade_results_0309.csv
.\\.venv_classify\\Scripts\\python.exe dianji_grade_rule.py --features dianji_grade_rule_v1\\grade_features_0309.csv --dry-run --output dianji_grade_rule_v1\\grade_features_dryrun.csv
```

## 边界

- 仅覆盖 `3#` / `9950`；`9996` / `bad` 无样本 → 引擎输出 `manual_review`。
- 规则表顶层固定 `status=unconfirmed_pending_enterprise`。
- 派生产物仅在本目录；未改动任何数据集原件。
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now().isoformat(timespec="seconds")
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    enterprise_rows = list_enterprise_images(args.enterprise_root.resolve())
    print(f"企业样本: {len(enterprise_rows)}  分布: {dict(Counter(r['true_grade'] for r in enterprise_rows))}")

    joined, meta_join = join_features(enterprise_rows, args.samples_csv.resolve())
    print(
        f"Join 覆盖: {meta_join['joined']}/{meta_join['enterprise_total']} "
        f"= {meta_join['coverage']*100:.1f}%  缺失={meta_join['missing']}"
    )
    if meta_join["missing"]:
        print("缺失样例:", meta_join["missing_list"][:10])

    features_path = out_dir / "grade_features_0309.csv"
    write_csv(features_path, joined)
    print(f"已写特征表: {features_path}")

    if args.dry_run:
        print("dry-run：跳过挖掘与回判")
        return 0

    if meta_join["joined"] == 0:
        print("无 join 样本，无法挖掘", file=sys.stderr)
        return 1

    X, y, _ = matrix_from_rows(joined, FEATURE_COLUMNS)
    print(f"矩阵: X={X.shape}  3#={int((y==1).sum())}  9950={int((y==0).sum())}")

    sk = try_import_sklearn()
    rules: list[dict[str, Any]] = []
    mining_meta: dict[str, Any] = {
        "seed": args.seed,
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
    }
    cv_info: dict[str, Any] = {}

    class_names_y = ["9950", "3#"]  # index 0 / 1

    if sk is not None:
        DecisionTreeClassifier = sk["DecisionTreeClassifier"]
        StratifiedKFold = sk["StratifiedKFold"]
        LeaveOneOut = sk["LeaveOneOut"]
        cross_val_predict = sk["cross_val_predict"]
        export_text = sk["export_text"]
        accuracy_score = sk["accuracy_score"]
        confusion_matrix = sk["confusion_matrix"]

        clf = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            class_weight="balanced",
        )
        # CV
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        min_class = min(n_pos, n_neg)
        if min_class < args.n_splits:
            cv = LeaveOneOut()
            strategy = f"LeaveOneOut（少数类={min_class} < n_splits={args.n_splits}）"
            note = "3# 样本过少，分层 K 折不可用，已退化为 LeaveOneOut"
        else:
            cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
            strategy = f"StratifiedKFold(n_splits={args.n_splits}, shuffle=True, seed={args.seed})"
            note = "分层 5 折交叉验证"
        y_pred_cv = cross_val_predict(clf, X, y, cv=cv)
        acc = float(accuracy_score(y, y_pred_cv))
        cm = confusion_matrix(y, y_pred_cv, labels=[1, 0])  # 3#, 9950
        cv_info = {
            "strategy": strategy,
            "note": note,
            "accuracy": acc,
            "confusion_matrix": {
                "labels": ["3#", "9950"],
                "matrix": cm.tolist(),
            },
        }
        # 全量拟合导出规则
        clf.fit(X, y)
        tree_txt = export_text(clf, feature_names=FEATURE_COLUMNS, show_weights=True)
        mining_meta["method"] = "sklearn.DecisionTreeClassifier"
        mining_meta["max_depth"] = args.max_depth
        mining_meta["min_samples_leaf"] = args.min_samples_leaf
        mining_meta["class_weight"] = "balanced"
        mining_meta["tree_text"] = tree_txt
        rules = tree_to_rules(clf, FEATURE_COLUMNS, class_names_y, X=X, y=y)
        print(f"sklearn 树导出规则 {len(rules)} 条；CV acc={acc:.4f}")
        print(tree_txt)
    else:
        print("未找到 sklearn，使用 numpy 网格搜索替代")
        rules, grid_meta = numpy_grid_search_rules(X, y, FEATURE_COLUMNS, seed=args.seed)
        mining_meta["method"] = "numpy_grid_search"
        mining_meta.update(grid_meta)
        # 简易 hold-in 评估当 CV
        fit_results = evaluate_rules_on_data(rules, joined, FEATURE_COLUMNS)
        fit_m = rule_metrics(fit_results)
        cv_info = {
            "strategy": "无 sklearn：训练集回判近似（非真实 CV）",
            "note": "环境无 sklearn，未做独立交叉验证",
            "accuracy": fit_m["accuracy"],
            "confusion_matrix": fit_m["confusion_matrix"],
        }

    # 清理 rules 中不可 JSON 的内容，保留引擎需要的字段
    export_rules = []
    for r in rules:
        export_rules.append(
            {
                "name": r["name"],
                "grade": r["grade"],
                "conditions": r.get("conditions") or [],
                "support_samples": r.get("support_samples"),
                "support_by_class": r.get("support_by_class"),
                "source": r.get("source"),
                "unconfirmed": True,
            }
        )

    rules_payload = {
        "status": "unconfirmed_pending_enterprise",
        "generated_at": generated_at,
        "sample_count": int(meta_join["joined"]),
        "grades_present": sorted(meta_join.get("grade_counts_joined", {}).keys()),
        "grades_absent": ["9996", "bad"],
        "class_names": CLASS_NAMES,
        "groups": GROUPS,
        "feature_columns": FEATURE_COLUMNS,
        "mining": {
            "method": mining_meta.get("method"),
            "seed": args.seed,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "cv": {
                "strategy": cv_info.get("strategy"),
                "accuracy": cv_info.get("accuracy"),
                "note": cv_info.get("note"),
            },
        },
        "join": {
            "coverage": meta_join["coverage"],
            "joined": meta_join["joined"],
            "enterprise_total": meta_join["enterprise_total"],
            "features_source": meta_join["features_source"],
            "weight_path": meta_join["weight_path"],
        },
        "disclaimer": "候选规则由小样本挖掘得到，未经企业确认，禁止当作生产标准。",
        "rules": export_rules,
    }
    rules_path = out_dir / "candidate_rules.json"
    rules_path.write_text(json.dumps(rules_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写规则表: {rules_path}")

    # 回判
    results = evaluate_rules_on_data(export_rules, joined, FEATURE_COLUMNS)
    fit_metrics = rule_metrics(results)
    results_path = out_dir / "grade_results_0309.csv"
    if not args.skip_engine_apply:
        write_csv(results_path, results)
        print(f"已写回判: {results_path}  fit_acc={fit_metrics['accuracy']:.4f}")

    report = build_report(meta_join, mining_meta, export_rules, cv_info, fit_metrics, generated_at)
    # 附加树文本
    if mining_meta.get("tree_text"):
        report += "\n## 附录：决策树文本\n\n```\n" + mining_meta["tree_text"] + "\n```\n"
    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"已写报告: {report_path}")

    readme_path = out_dir / "README.md"
    readme_path.write_text(build_readme(generated_at, meta_join), encoding="utf-8")
    print(f"已写 README: {readme_path}")

    # 再调用引擎 CLI 风格确认
    if not args.skip_engine_apply:
        from dianji_grade_rule import DianjiGradeRule

        engine = DianjiGradeRule(rules_path=rules_path)
        n_hit = sum(1 for r in results if r["hit_rule"])
        print(
            f"引擎加载规则 {len(engine.rules)} 条；命中 {n_hit}/{len(results)}；"
            f"manual_review={fit_metrics['manual_review_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
