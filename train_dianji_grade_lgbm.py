#!/usr/bin/env python3
"""
电积镍等级四分类模型对比（严格 train/valid/test）：
- train 训练；valid 早停与选模；test 仅最终一次
- 多数类 / DianjieGradeRule 迁移 / 浅层决策树 / LightGBM
- 产物写入 dianji_grade_v2/，全部 unconfirmed_pending_enterprise
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

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "dianji_grade_v2"
DEFAULT_VIEW_SUMMARY = Path(
    r"E:/PythonProject/datasets/dianjinie_grade_classification_v1/dataset_summary.json"
)
DEFAULT_SEED = 42

CLASS_NAMES_18 = [
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

FEATURE_COLUMNS = (
    [
        "A_ratio",
        "B_ratio",
        "C_ratio",
        "Airhole_ratio",
        "has_nodule",
        "total_bbox_union_image_ratio",
        "detection_count",
    ]
    + [f"count__{n}" for n in CLASS_NAMES_18]
    + [f"area_ratio__{n}" for n in CLASS_NAMES_18]
)

GRADES = ["3#", "9950", "9996", "bad"]
GRADE_TO_ID = {g: i for i, g in enumerate(GRADES)}
ID_TO_GRADE = {i: g for g, i in GRADE_TO_ID.items()}
DIANJIE_MAP = {1: "3#", 6: "9950"}  # type_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="电积镍等级四分类：train/valid 选模，test 最终评估（未经企业确认）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--view-summary", type=Path, default=DEFAULT_VIEW_SUMMARY)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--feature-csvs", type=str, nargs="*", default=None)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--lgbm-num-leaves", type=int, default=8)
    p.add_argument("--lgbm-max-depth", type=int, default=4)
    p.add_argument("--lgbm-min-data", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys = fields or list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def to_float(v: Any) -> float:
    if v is None or v == "":
        return 0.0
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    s = str(v).strip().lower()
    if s in {"true", "yes"}:
        return 1.0
    if s in {"false", "no"}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def matrix_from_rows(rows: list[dict[str, Any]]):
    xs, ys, kept = [], [], []
    for r in rows:
        g = r.get("true_grade") or r.get("grade")
        if g not in GRADE_TO_ID:
            continue
        xs.append([to_float(r.get(c, 0)) for c in FEATURE_COLUMNS])
        ys.append(GRADE_TO_ID[g])
        kept.append(r)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64), kept


def metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    labels = list(range(len(GRADES)))
    # 将非法 pred（other=-1）强制错分
    y_pred2 = y_pred.copy()
    other_mask = y_pred2 < 0
    if other_mask.any():
        y_pred2[other_mask] = (y_true[other_mask] + 1) % len(GRADES)

    cm = confusion_matrix(y_true, y_pred2, labels=labels)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred2, labels=labels, zero_division=0
    )
    per_class = {
        g: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, g in enumerate(GRADES)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred2)),
        "macro_f1": float(f1_score(y_true, y_pred2, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred2, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": GRADES,
        "other_rate": float(other_mask.mean()) if len(other_mask) else 0.0,
        "other_count": int(other_mask.sum()),
        "classification_report": classification_report(
            y_true, y_pred2, labels=labels, target_names=GRADES, zero_division=0
        ),
    }


def majority_predict(y_train: np.ndarray, n: int) -> np.ndarray:
    maj = int(Counter(y_train.tolist()).most_common(1)[0][0])
    return np.full(n, maj, dtype=np.int64)


def dianjie_rule_predict(rows: list[dict[str, Any]]) -> np.ndarray:
    from dianjie_grade_rule import DianjieGradeRule

    rule = DianjieGradeRule()
    preds: list[int] = []
    for r in rows:
        tid = None
        if r.get("type_id_detection_only") not in (None, ""):
            try:
                tid = int(float(r["type_id_detection_only"]))
            except ValueError:
                tid = None
        if tid is None:
            a = to_float(r.get("A_ratio"))
            b = to_float(r.get("B_ratio"))
            c = to_float(r.get("C_ratio"))
            air = to_float(r.get("Airhole_ratio"))
            has_nodule = to_float(r.get("has_nodule")) >= 0.5
            type_id = 4
            if (
                not has_nodule
                and a < rule.good_a_ratio
                and b < rule.good_b_ratio
                and air < rule.good_airhole_ratio
                and c < rule.good_c_ratio
            ):
                type_id = 3
            elif (
                not has_nodule
                and a >= rule.good_a_ratio
                and b < rule.wash_b_ratio
                and c < rule.wash_c_ratio
            ):
                type_id = 2
            elif c > rule.c_severe_ratio:
                type_id = 1
            elif b > rule.b_severe_ratio and c < rule.c_severe_ratio:
                type_id = 6
            tid = type_id
        mapped = DIANJIE_MAP.get(tid)
        preds.append(GRADE_TO_ID[mapped] if mapped else -1)
    return np.asarray(preds, dtype=np.int64)


def train_tree(X_tr, y_tr, seed, max_depth, min_leaf):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    return clf


def train_lgbm(X_tr, y_tr, X_va, y_va, seed, num_leaves, max_depth, min_data):
    try:
        import lightgbm as lgb

        counts = Counter(y_tr.tolist())
        n = len(y_tr)
        k = len(GRADES)
        weights = {i: n / (k * max(counts.get(i, 1), 1)) for i in range(k)}
        sw = np.asarray([weights[int(y)] for y in y_tr], dtype=np.float64)
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw, feature_name=list(FEATURE_COLUMNS), free_raw_data=False)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=list(FEATURE_COLUMNS), free_raw_data=False)
        params = {
            "objective": "multiclass",
            "num_class": k,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_data_in_leaf": min_data,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": seed,
            "deterministic": True,
        }
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=400,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
        )
        imp = booster.feature_importance(importance_type="gain")
        top15 = sorted(zip(FEATURE_COLUMNS, [float(x) for x in imp]), key=lambda t: t[1], reverse=True)[:15]
        return {
            "backend": "lightgbm",
            "model": booster,
            "best_iteration": booster.best_iteration,
            "top15": top15,
        }
    except Exception as exc:
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_depth=max_depth,
            max_leaf_nodes=num_leaves,
            min_samples_leaf=min_data,
            learning_rate=0.05,
            max_iter=300,
            random_state=seed,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
        clf.fit(X_tr, y_tr)
        return {
            "backend": f"hist_gradient_boosting_fallback:{type(exc).__name__}",
            "model": clf,
            "best_iteration": getattr(clf, "n_iter_", None),
            "top15": [],
            "error": str(exc),
        }


def predict_model(bundle: dict[str, Any] | Any, X: np.ndarray, kind: str) -> np.ndarray:
    if kind == "tree":
        return bundle.predict(X)
    if kind == "lgbm":
        model = bundle["model"]
        if bundle["backend"] == "lightgbm":
            proba = model.predict(X, num_iteration=bundle.get("best_iteration"))
            return np.argmax(proba, axis=1)
        return model.predict(X)
    raise ValueError(kind)


def pair_9950_9996(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    i0, i1 = GRADE_TO_ID["9950"], GRADE_TO_ID["9996"]
    mask = np.isin(y_true, [i0, i1])
    if not mask.any():
        return {"n": 0}
    yt, yp = y_true[mask], y_pred[mask]
    yp2 = yp.copy()
    yp2[yp2 < 0] = -9
    return {
        "n": int(mask.sum()),
        "accuracy_on_subset": float((yt == yp2).mean()),
        "9950_as_9996": int(((yt == i0) & (yp2 == i1)).sum()),
        "9996_as_9950": int(((yt == i1) & (yp2 == i0)).sum()),
        "9950_support": int((yt == i0).sum()),
        "9996_support": int((yt == i1).sum()),
    }


def evaluate_one(
    path: Path,
    seed: int,
    max_depth: int,
    min_leaf: int,
    lgbm_num_leaves: int,
    lgbm_max_depth: int,
    lgbm_min_data: int,
) -> dict[str, Any]:
    rows = read_csv(path)
    X, y, kept = matrix_from_rows(rows)
    splits = np.asarray([r.get("split", "") for r in kept])
    tr = splits == "train"
    va = splits == "valid"
    te = splits == "test"
    if tr.sum() == 0 or va.sum() == 0:
        raise RuntimeError(f"{path.name}: 需要 train 与 valid，实际 train={tr.sum()} valid={va.sum()}")

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]
    rows_va = [kept[i] for i, f in enumerate(va) if f]
    rows_te = [kept[i] for i, f in enumerate(te) if f]

    feature_version = kept[0].get("feature_version", path.stem) if kept else path.stem
    result: dict[str, Any] = {
        "feature_csv": str(path),
        "feature_version": feature_version,
        "n_total": int(len(kept)),
        "n_train": int(tr.sum()),
        "n_valid": int(va.sum()),
        "n_test": int(te.sum()),
        "grade_counts": {
            "train": dict(Counter(ID_TO_GRADE[int(i)] for i in y_tr)),
            "valid": dict(Counter(ID_TO_GRADE[int(i)] for i in y_va)),
            "test": dict(Counter(ID_TO_GRADE[int(i)] for i in y_te)) if te.sum() else {},
        },
        "models": {"valid": {}, "test": {}},
        "status": "unconfirmed_pending_enterprise",
        "protocol": "train_fit / valid_select / test_final_once",
    }

    # 拟合可学习模型（仅 train；lgbm 用 valid 早停）
    tree = train_tree(X_tr, y_tr, seed, max_depth, min_leaf)
    lgbm = train_lgbm(X_tr, y_tr, X_va, y_va, seed, lgbm_num_leaves, lgbm_max_depth, lgbm_min_data)

    def run_split(split_name: str, X_ev, y_ev, rows_ev) -> dict[str, np.ndarray]:
        preds: dict[str, np.ndarray] = {}
        # majority
        p = majority_predict(y_tr, len(y_ev))
        preds["majority"] = p
        result["models"][split_name]["majority"] = metrics_report(y_ev, p)
        # rule
        p = dianjie_rule_predict(rows_ev)
        preds["dianjie_grade_rule"] = p
        m = metrics_report(y_ev, p)
        m["note"] = "仅映射 type1→3# / type6→9950，其余 other 计错"
        result["models"][split_name]["dianjie_grade_rule"] = m
        # tree
        p = predict_model(tree, X_ev, "tree")
        preds["decision_tree"] = p
        m = metrics_report(y_ev, p)
        m["params"] = {"max_depth": max_depth, "min_samples_leaf": min_leaf, "seed": seed}
        result["models"][split_name]["decision_tree"] = m
        # lgbm
        p = predict_model(lgbm, X_ev, "lgbm")
        preds["lightgbm"] = p
        m = metrics_report(y_ev, p)
        m["backend"] = lgbm["backend"]
        m["best_iteration"] = lgbm.get("best_iteration")
        m["feature_importance_top15"] = [{"feature": a, "gain": b} for a, b in lgbm.get("top15", [])]
        m["params"] = {
            "num_leaves": lgbm_num_leaves,
            "max_depth": lgbm_max_depth,
            "min_data_in_leaf": lgbm_min_data,
            "seed": seed,
        }
        m["9950_vs_9996"] = pair_9950_9996(y_ev, p)
        result["models"][split_name]["lightgbm"] = m
        return preds

    preds_va = run_split("valid", X_va, y_va, rows_va)
    # 选模：valid macro_f1
    best_name, best_score = "lightgbm", -1.0
    for name, m in result["models"]["valid"].items():
        if m.get("macro_f1", -1) > best_score:
            best_score = m["macro_f1"]
            best_name = name
    result["best_model_on_valid"] = {"name": best_name, "macro_f1": best_score}

    preds_te: dict[str, np.ndarray] = {}
    if te.sum() > 0:
        preds_te = run_split("test", X_te, y_te, rows_te)
        result["final_test_selected"] = {
            "model": best_name,
            "metrics": result["models"]["test"][best_name],
            "note": "test 未参与调参/早停/阈值选择",
        }
        # disagreements：选定模型
        best_pred = preds_te[best_name]
        disag = []
        for i, r in enumerate(rows_te):
            yt = int(y_te[i])
            yp = int(best_pred[i])
            true_g = ID_TO_GRADE[yt]
            pred_g = "other" if yp < 0 else ID_TO_GRADE.get(yp, str(yp))
            if pred_g != true_g:
                row = {
                    "image_name": r.get("image_name", ""),
                    "source_dataset": r.get("source_dataset", r.get("source", "")),
                    "split": "test",
                    "true_grade": true_g,
                    "pred_grade": pred_g,
                    "model": best_name,
                    "feature_version": feature_version,
                    "pair": f"{true_g}->{pred_g}",
                    "status": "unconfirmed_pending_enterprise",
                }
                for c in FEATURE_COLUMNS:
                    row[c] = r.get(c, "")
                disag.append(row)
        result["disagreements"] = disag
        cm = np.array(result["models"]["test"][best_name]["confusion_matrix"])
        off = int(cm.sum() - np.trace(cm))
        result["disagreement_count"] = len(disag)
        result["disagreement_vs_cm_offdiag"] = {
            "disagreements": len(disag),
            "cm_offdiag": off,
            "match": len(disag) == off,
        }
    else:
        result["disagreements"] = []
        result["disagreement_count"] = 0

    # 持久化模型元数据（不强制 dump booster 二进制）
    models_dir = path.parent / "models" / path.stem
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "lgbm_meta.json").write_text(
        json.dumps(
            {
                "backend": lgbm["backend"],
                "best_iteration": lgbm.get("best_iteration"),
                "top15": lgbm.get("top15"),
                "status": "unconfirmed_pending_enterprise",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def build_report(
    all_results: list[dict[str, Any]],
    view_summary: dict[str, Any] | None,
    extract_summary: dict[str, Any] | None,
    out_dir: Path,
) -> str:
    lines: list[str] = [
        "# 电积镍等级分类 v2 报告",
        "",
        "> ## ⚠️ 未经企业确认",
        ">",
        "> 本报告全部模型、规则与阈值均为实验性产物，**不得**作为企业正式判级标准或上线依据。",
        "> 准确率如实呈现；禁止调参过拟合后宣称可用。",
        "",
        "> ## ⚠️ 小样本警告",
        ">",
        "> `3#` 合计 46 张（含 0309 的 12 张）、`bad` 合计 28 张，类别极不平衡，少数类指标方差大。",
        "> 用户任务中 4types 口径 3#=34、bad=28；合并 0309 后 3#=46。",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 产物目录：`{out_dir}`",
        f"- 状态：`unconfirmed_pending_enterprise`",
        f"- 协议：train 训练 / valid 选模与早停 / **test 仅最终一次**",
        "",
        "## 1. 数据与去重 / 硬链接视图",
        "",
    ]
    if view_summary:
        lines += [
            f"- 视图：`{view_summary.get('output_root')}`",
            f"- 原始：4types={view_summary.get('source_counts_raw', {}).get('classify_4types_dianji')} + "
            f"0309={view_summary.get('source_counts_raw', {}).get('enterprise_dianji_0309_classification')}",
            f"- 完整重复组：{view_summary.get('dedup', {}).get('exact_duplicate_groups')}，移除 {view_summary.get('dedup', {}).get('exact_removed')}",
            f"- 近重复：方法={view_summary.get('near', {}).get('hash_method')} thr≤{view_summary.get('near', {}).get('phash_threshold')}，"
            f"多成员组={view_summary.get('near', {}).get('near_duplicate_groups')}，"
            f"涉及图={view_summary.get('near', {}).get('near_duplicate_images_in_multi_groups')}",
            f"- 去重后：{view_summary.get('kept_images')}；等级：`{view_summary.get('grade_counts')}`",
            f"- 划分 seed={view_summary.get('split', {}).get('seed')}，图片比例：`{view_summary.get('split', {}).get('image_ratio_by_split')}`",
            f"- 各 split×等级：`{view_summary.get('split', {}).get('images_by_split_grade')}`",
            f"- 链接：`{view_summary.get('link_mode_counts')}`（硬链接优先）",
            f"- **不沿用** 4types 原 train/valid；原 split 记在 manifest `source_split`",
            "",
        ]
    else:
        lines += ["- （未读到 dataset_summary.json）", ""]

    lines += ["## 2. 特征版本", ""]
    if extract_summary:
        v2a = extract_summary.get("v2a", {})
        v2b = extract_summary.get("v2b", {})
        if v2a:
            lines.append(
                f"- **v2a** join：{v2a.get('joined')}/{v2a.get('manifest_total')} = {float(v2a.get('coverage') or 0):.2%}；"
                f"缺失 {v2a.get('missing')}（`features_v2a_missing.csv`）。"
                f" 口径：{v2a.get('detection_params_note')}"
            )
            lines.append(
                "  - 说明：combined 伪标签集仅覆盖部分 4types 样本，故 v2a 仅作对照，**主结论以 v2b 为准**。"
            )
        if v2b:
            lines.append(
                f"- **v2b** 主特征：覆盖 {v2b.get('images_with_features')}/{v2b.get('manifest_requested')} = "
                f"{float(v2b.get('coverage') or 0):.2%}；device={v2b.get('device')} cuda={v2b.get('cuda_available')} "
                f"imgsz={v2b.get('img_size')} conf={v2b.get('conf_thres')} iou={v2b.get('iou_thres')}"
            )
            lines.append(f"  - 逐框：`{v2b.get('boxes_csv')}`；特征：`{v2b.get('feature_csvs')}`")
        lines.append("")
    else:
        lines += ["- （未读到 extract_summary.json）", ""]

    lines += [
        "## 3. 模型对比",
        "",
        "四行基线：majority / DianjieGradeRule / decision_tree / lightgbm。",
        "valid 用于选模；test 表为最终对照（**未参与选模**）。",
        "",
    ]

    for res in all_results:
        lines.append(f"### 特征 `{res.get('feature_version')}`")
        lines.append("")
        lines.append(
            f"- n: train={res.get('n_train')} valid={res.get('n_valid')} test={res.get('n_test')}；"
            f"分布 train=`{res.get('grade_counts', {}).get('train')}`"
        )
        lines.append("")
        for split in ("valid", "test"):
            rows_md = []
            for model in ("majority", "dianjie_grade_rule", "decision_tree", "lightgbm"):
                m = res.get("models", {}).get(split, {}).get(model)
                if not m:
                    continue
                rows_md.append(
                    [
                        model,
                        f"{m.get('accuracy', 0):.4f}",
                        f"{m.get('macro_f1', 0):.4f}",
                        f"{m.get('per_class', {}).get('3#', {}).get('f1', 0):.4f}",
                        f"{m.get('per_class', {}).get('9950', {}).get('f1', 0):.4f}",
                        f"{m.get('per_class', {}).get('9996', {}).get('f1', 0):.4f}",
                        f"{m.get('per_class', {}).get('bad', {}).get('f1', 0):.4f}",
                        f"{m.get('other_rate', 0):.3f}" if model == "dianjie_grade_rule" else "-",
                    ]
                )
            if rows_md:
                title = "Valid（选模）" if split == "valid" else "Test（最终，未参与调参）"
                lines.append(f"**{title}**")
                lines.append("")
                lines.append(
                    md_table(
                        ["模型", "Acc", "Macro-F1", "F1_3#", "F1_9950", "F1_9996", "F1_bad", "other_rate"],
                        rows_md,
                    )
                )
                lines.append("")
        best = res.get("best_model_on_valid", {})
        lines.append(f"- valid 最优：`{best.get('name')}` macro_f1={best.get('macro_f1')}")
        final = res.get("final_test_selected", {})
        if final:
            fm = final.get("metrics", {})
            lines.append(
                f"- 选定模型 test：`{final.get('model')}` acc={fm.get('accuracy')} macro_f1={fm.get('macro_f1')}"
            )
        lgb = res.get("models", {}).get("test", {}).get("lightgbm") or res.get("models", {}).get("valid", {}).get("lightgbm", {})
        if lgb.get("feature_importance_top15"):
            lines.append("")
            lines.append("LightGBM 特征重要性 top15（gain）：")
            lines.append("")
            for item in lgb["feature_importance_top15"]:
                lines.append(f"- `{item['feature']}`: {item['gain']:.4f}")
        if lgb.get("confusion_matrix"):
            lines.append("")
            lines.append("LightGBM 混淆矩阵（优先 test；行=真，列=预测，3#/9950/9996/bad）：")
            lines.append("")
            lines.append("```")
            lines.append(str(np.array(lgb["confusion_matrix"])))
            lines.append("```")
        if lgb.get("9950_vs_9996"):
            lines.append(f"- 9950 vs 9996：`{lgb['9950_vs_9996']}`")
        lines.append(
            f"- 分歧：{res.get('disagreement_count')}；与 cm off-diag：`{res.get('disagreement_vs_cm_offdiag')}`"
        )
        lines.append("")

    # 全局最优：在覆盖完整的 v2b 上按 valid lgbm macro_f1
    full = [r for r in all_results if (r.get("n_total") or 0) >= 1000]
    pool = full if full else all_results
    best_overall = max(
        pool,
        key=lambda r: r.get("models", {}).get("valid", {}).get("lightgbm", {}).get("macro_f1", -1),
    )

    lines += [
        "## 4. 一致性与分歧",
        "",
        f"- 主推特征（完整覆盖优先 + valid LGBM macro_f1）：`{best_overall.get('feature_version')}`",
        f"- 分歧清单：`disagreements.csv`（该特征 test 上选定模型）",
        "",
        "### 9950 vs 9996 可分性",
        "",
        "若企业语义上两者主要差在化验成分而非外观，则图像缺陷面积特征**原则上难以稳定区分**。",
        f"本次：`{best_overall.get('models', {}).get('test', {}).get('lightgbm', {}).get('9950_vs_9996')}`。",
        "若互相混淆高，应视为特征上限，而非继续堆模型。",
        "",
    ]
    pair_c = Counter(d["pair"] for d in best_overall.get("disagreements", []))
    if pair_c:
        lines.append("主要分歧对（test）：")
        lines.append("")
        for k, v in pair_c.most_common(15):
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    lines += [
        "## 5. 与侍掠飞论文三分类的定性对照",
        "",
        "论文 taxonomy：**合格 / 烫洗 / 不合格**（`师兄论文/2021141128_cglw_侍掠飞.txt` 约 819–820、944–946 行）。",
        "本实验四类：**3# / 9950 / 9996 / bad**，**无「烫洗」对应类**，禁止强行映射报准确率。",
        "",
        "特征画像（未验证假设）：",
        "",
        "- 「烫洗」≈ 高 `waterstain`/`watermark`/`A_ratio`，且 overfire/suanwuran/C 组不突出的样本。",
        "- 「不合格」≈ `bad` 与部分 `3#`（严重外观/C 组或大面积 B 组）。",
        "- 「合格」≈ 部分 `9996`/`9950`；但牌号化验差无法从图像复现。",
        "",
        "本实验回答的是「企业四类目录标签 vs 检测框面积特征」一致性，**不是**论文三类复现。",
        "",
        "## 6. 三个业务问题",
        "",
        "### ① LightGBM 四分类表现与最优特征版本",
        "",
    ]
    lv = best_overall.get("models", {}).get("valid", {}).get("lightgbm", {})
    lt = best_overall.get("models", {}).get("test", {}).get("lightgbm", {})
    lines += [
        f"- 主推特征：`{best_overall.get('feature_version')}`",
        f"- Valid LGBM：acc={lv.get('accuracy')} macro_f1={lv.get('macro_f1')} per-class={lv.get('per_class')}",
        f"- Test LGBM：acc={lt.get('accuracy')} macro_f1={lt.get('macro_f1')} per-class={lt.get('per_class')}",
        f"- backend={lt.get('backend') or lv.get('backend')} best_iteration={lt.get('best_iteration') or lv.get('best_iteration')}",
        "",
        "### ② 与原标签一致率及分歧模式",
        "",
        "- 见表中 Acc/混淆矩阵与 `disagreements.csv`。",
        "- 重点：9950↔9996；3#/bad 少数类召回不稳定。",
        "",
        "### ③ 与论文标准的定性符合度",
        "",
        "- taxonomy 不同，仅特征画像对照，无准确率映射。见第 5 节。",
        "",
        "## 7. 若准确率不足：备选方案",
        "",
        "1. 更丰富空间/纹理特征（分区覆盖、LBP/GLCM）。",
        "2. 检测层提精度（更高 imgsz、电积镍专用检测权重、人工审查伪标签）。",
        "3. 端到端图像分类（ResNet/ViT），注意小样本与不平衡。",
        "4. 双阶段：先合格/不合格（或论文三类）再细分牌号；牌号融合化验。",
        "5. 补充 3# / bad / 9950–9996 边界样本。",
        "6. `manual_review` 拒判：低置信度送人工。",
        "",
        "## 8. 复现命令",
        "",
        "```text",
        r"cd /d E:\PythonProject\优化",
        r".\.venv_classify\Scripts\python.exe build_dianjinie_grade_view.py --force",
        r"E:\PythonProject\ImetricCodes\JinchuanYolov5sTrain\venv\Scripts\python.exe extract_dianji_grade_features.py",
        r".\.venv_classify\Scripts\python.exe train_dianji_grade_lgbm.py",
        "```",
        "",
        "## 9. 先前任务取舍",
        "",
        "- 上一版仅计划 `dianji_grade_v2` 内合并 manifest；用户修订后改为 **硬链接视图 + 70/15/15 重划分**。",
        "- 中断前草稿 `build_dianji_grade_v2_dataset.py` 保留但不作为入口。",
        "- 流水线入口：`build_dianjinie_grade_view.py` → `extract_dianji_grade_features.py` → `train_dianji_grade_lgbm.py`。",
        "- `dianji_grade_rule_v1/` 与规则引擎未改动。",
        "",
        "## 10. 近重复说明",
        "",
        "- 初版 aHash thr=6 在镍板大图上假近重复极多（曾并成巨组导致划分倒置），已改为 **dHash + 同等级才并组 + thr=5**。",
        "- 完整重复复核：0 组（与 2026-07-13 审计一致）。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.feature_csvs:
        csv_paths = [Path(p) for p in args.feature_csvs]
    else:
        csv_paths = sorted(
            p
            for p in out.glob("features_*.csv")
            if "missing" not in p.name
        )

    print("特征文件:", [p.name for p in csv_paths])
    if args.dry_run:
        return 0
    if not csv_paths:
        print("未找到特征 CSV", file=sys.stderr)
        return 2

    all_results: list[dict[str, Any]] = []
    for p in csv_paths:
        print(f"=== 评估 {p.name} ===")
        res = evaluate_one(
            p,
            seed=args.seed,
            max_depth=args.max_depth,
            min_leaf=args.min_samples_leaf,
            lgbm_num_leaves=args.lgbm_num_leaves,
            lgbm_max_depth=args.lgbm_max_depth,
            lgbm_min_data=args.lgbm_min_data,
        )
        all_results.append(res)
        (out / f"metrics_{p.stem}.json").write_text(
            json.dumps({k: v for k, v in res.items() if k != "disagreements"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"  valid-lgbm macroF1={res['models']['valid'].get('lightgbm', {}).get('macro_f1')} "
            f"test-lgbm macroF1={res['models'].get('test', {}).get('lightgbm', {}).get('macro_f1')} "
            f"best={res.get('best_model_on_valid')}"
        )

    full = [r for r in all_results if (r.get("n_total") or 0) >= 1000]
    pool = full if full else all_results
    best = max(
        pool,
        key=lambda r: r.get("models", {}).get("valid", {}).get("lightgbm", {}).get("macro_f1", -1),
    )
    write_csv(out / "disagreements.csv", best.get("disagreements", []))
    pair_rows = [
        {"pair": k, "count": v, "feature_version": best.get("feature_version"), "status": "unconfirmed_pending_enterprise"}
        for k, v in Counter(d["pair"] for d in best.get("disagreements", [])).most_common()
    ]
    write_csv(out / "disagreement_pairs.csv", pair_rows)

    comparison_rows = []
    for res in all_results:
        for split in ("valid", "test"):
            for model in ("majority", "dianjie_grade_rule", "decision_tree", "lightgbm"):
                m = res.get("models", {}).get(split, {}).get(model)
                if not m:
                    continue
                comparison_rows.append(
                    {
                        "feature_version": res.get("feature_version"),
                        "split": split,
                        "model": model,
                        "accuracy": m.get("accuracy"),
                        "macro_f1": m.get("macro_f1"),
                        "weighted_f1": m.get("weighted_f1"),
                        "f1_3#": m.get("per_class", {}).get("3#", {}).get("f1"),
                        "f1_9950": m.get("per_class", {}).get("9950", {}).get("f1"),
                        "f1_9996": m.get("per_class", {}).get("9996", {}).get("f1"),
                        "f1_bad": m.get("per_class", {}).get("bad", {}).get("f1"),
                        "other_rate": m.get("other_rate"),
                        "status": "unconfirmed_pending_enterprise",
                    }
                )
    write_csv(out / "model_comparison.csv", comparison_rows)

    view_summary = None
    if args.view_summary.exists():
        view_summary = json.loads(args.view_summary.read_text(encoding="utf-8"))
    extract_summary = None
    if (out / "extract_summary.json").exists():
        extract_summary = json.loads((out / "extract_summary.json").read_text(encoding="utf-8"))

    report = build_report(all_results, view_summary, extract_summary, out)
    (out / "report.md").write_text(report, encoding="utf-8")
    (out / "README.md").write_text(
        "\n".join(
            [
                "# dianji_grade_v2",
                "",
                "> **未经企业确认**。特征与模型均为实验产物，不得上线。",
                "",
                "## 内容",
                "",
                "- `features_v2a.csv`：历史 samples.csv join（覆盖率低，仅对照）",
                "- `features_v2b_imgsz1536_conf0p25.csv` / `conf0p40.csv`：主特征",
                "- `detections_1536/`：逐框原始输出",
                "- `model_comparison.csv` / `metrics_*.json`",
                "- `disagreements.csv`：主推特征 test 分歧",
                "- `report.md`",
                "",
                "## 视图",
                "",
                r"- `E:\PythonProject\datasets\dianjinie_grade_classification_v1`",
                "",
                "## 小样本",
                "",
                "- 3#=46（合并后）、bad=28。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out / "train_summary.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "feature_csvs": [str(p) for p in csv_paths],
                "best_feature_version": best.get("feature_version"),
                "best_model_on_valid": best.get("best_model_on_valid"),
                "final_test_selected": best.get("final_test_selected"),
                "n_disagreements": best.get("disagreement_count"),
                "disagreement_vs_cm": best.get("disagreement_vs_cm_offdiag"),
                "status": "unconfirmed_pending_enterprise",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("完成。最优特征:", best.get("feature_version"), best.get("best_model_on_valid"))
    print("报告:", out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
