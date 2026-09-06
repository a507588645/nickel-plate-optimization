# -*- coding: utf-8 -*-
"""C2.3 电解镍判级方案匹配：LightGBM 首选 + 浅层树对比（路线文档 §4.2）。

在 C2.1 弱标签（type 1/2/3/4/6、nodule 0/2/3）× C2.2 63 列特征（A3 ONNX 901ccf60
生产口径）上训练判级模型，评测与弱标签（=规则树输出）的一致率/混淆矩阵。
协议逐字复用 `train_dianji_grade_v3.py`：seed42、multiclass（num_leaves=8、
max_depth=4、min_data_in_leaf=10、lr=0.05、ff/bf=0.9、bagging_freq=1、
deterministic、400 轮 + valid 早停 40）、**类平衡权重 w[i]=n/(k·count[i]) 必须保留**、
train 训练 / val 选模早停 / test 只评一次。电解镍 type 0-6 与电积镍等级语义不同，
只借方法不借模型。

臂：type_lgbm（主）/ type_tree（浅层树对比，depth3/leaf5/balanced）/ nodule_lgbm（副）
/ majority（多数类基线）。产出 per-image 预测、指标 JSON、覆盖率-一致率曲线、booster。
产物只写本目录；booster 落盘走 ASCII 临时路径 + move（LightGBM 中文路径坑）。
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
FEATURES_CSV = HERE / "features_c2p2_y11m640.csv"
METADATA_FIELDS = {
    "sample_id", "image_name", "image_path", "group_id", "fold", "split", "source_split",
    "type_weak", "nodule_weak", "n_defects_weak", "feature_version",
    "image_width", "image_height", "plate_area", "plate_area_fallback", "max_confidence",
}
TYPE_CLASSES = [1, 2, 3, 4, 6]
NODULE_CLASSES = [0, 2, 3]
TYPE_TO_DERIVED_NODULE = {3: 0, 2: 2, 1: 3, 4: 3, 6: 3}
COVERAGE_LEVELS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def load_rows() -> tuple[list[dict[str, str]], list[str]]:
    with FEATURES_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1127:
        raise ValueError(f"特征行数 {len(rows)} != 1127")
    columns = [c for c in rows[0] if c not in METADATA_FIELDS]
    if len(columns) != 63:
        raise ValueError(f"特征列数 {len(columns)} != 63")
    return rows, columns


def build_xy(rows: list[dict[str, str]], columns: list[str], label_key: str,
             classes: list[int]) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    label_to_id = {c: i for i, c in enumerate(classes)}
    xs, ys, kept = [], [], []
    for r in rows:
        label = int(float(r[label_key]))
        if label not in label_to_id:
            raise ValueError(f"未知标签 {label}（{label_key}）")
        xs.append([to_float(r.get(c, 0)) for c in columns])
        ys.append(label_to_id[label])
        kept.append(r)
    X = np.nan_to_num(np.asarray(xs, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.asarray(ys, dtype=np.int64), kept


def split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {"train": [], "valid": [], "test": []}
    for r in rows:
        s = r["split"]
        key = "valid" if s == "val" else s
        if key not in out:
            raise ValueError(f"未知 split: {s}")
        out[key].append(r)
    return out


def train_lgbm_c2(X_tr, y_tr, X_va, y_va, k: int, seed: int, num_leaves: int,
                  max_depth: int, min_data: int, columns: list[str]):
    """逐字镜像 train_dianji_grade_v3.train_lgbm 协议，仅类数参数化。"""
    import lightgbm as lgb

    counts = Counter(y_tr.tolist())
    n = len(y_tr)
    weights = {i: n / (k * max(counts.get(i, 1), 1)) for i in range(k)}
    sw = np.asarray([weights[int(y)] for y in y_tr], dtype=np.float64)
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw, feature_name=columns, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=columns, free_raw_data=False)
    params = {
        "objective": "multiclass", "num_class": k,
        "num_leaves": num_leaves, "max_depth": max_depth,
        "min_data_in_leaf": min_data, "learning_rate": 0.05,
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
        "verbosity": -1, "seed": seed, "deterministic": True,
    }
    booster = lgb.train(
        params, dtrain, num_boost_round=400, valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    imp = booster.feature_importance(importance_type="gain")
    top20 = sorted(zip(columns, [float(x) for x in imp]), key=lambda t: t[1], reverse=True)[:20]
    return {"model": booster, "best_iteration": booster.best_iteration, "top20": top20}


def train_tree_c2(X_tr, y_tr, seed: int):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5,
                                 class_weight="balanced", random_state=seed)
    clf.fit(X_tr, y_tr)
    return clf


def metrics_report(y_true: np.ndarray, y_pred: np.ndarray, classes: list[int]) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score,
        precision_recall_fscore_support,
    )

    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            str(classes[i]): {"precision": float(p[i]), "recall": float(r[i]),
                              "f1": float(f1[i]), "support": int(support[i])}
            for i in range(len(classes))
        },
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
    }


def save_booster(booster, out_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="c2p3_lgbm_") as tmp:
        tmp_path = Path(tmp) / "booster.txt"
        booster.save_model(str(tmp_path))
        shutil.move(str(tmp_path), str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_metrics = HERE / "c2p3_metrics.json"
    out_per_image = HERE / "c2p3_per_image.csv"
    out_curve = HERE / "c2p3_agreement_coverage.csv"
    out_type_booster = HERE / "c2p3_type_booster.txt"
    out_nodule_booster = HERE / "c2p3_nodule_booster.txt"
    for out in (out_metrics, out_per_image, out_curve, out_type_booster, out_nodule_booster):
        if out.exists() and not args.force:
            raise RuntimeError(f"产物已存在（用 --force 覆盖）: {out}")

    rows, columns = load_rows()
    splits = split_rows(rows)
    split_sizes = {k: len(v) for k, v in splits.items()}
    type_dist = {s: dict(Counter(int(float(r["type_weak"])) for r in splits[s])) for s in splits}
    nodule_dist = {s: dict(Counter(int(float(r["nodule_weak"])) for r in splits[s])) for s in splits}
    print(json.dumps({"split_sizes": split_sizes, "type_dist": type_dist, "nodule_dist": nodule_dist},
                     ensure_ascii=False))
    if args.dry_run:
        print("[dry-run] schema 校验通过")
        return 0

    seed = args.seed
    results: dict[str, Any] = {
        "task": "C2.3 电解镍判级方案匹配（弱标签蒸馏，63 列特征）",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "protocol": "train_dianji_grade_v3 逐字镜像（类平衡权重 w=n/(k*count)、早停 40@valid、test 只评一次）",
        "data": {"features_csv": str(FEATURES_CSV), "split_sizes": split_sizes,
                 "type_dist": type_dist, "nodule_dist": nodule_dist},
        "arms": {},
    }

    # —— 主臂：type 5 类 LGBM ——
    X_tr, y_tr, rows_tr = build_xy(splits["train"], columns, "type_weak", TYPE_CLASSES)
    X_va, y_va, rows_va = build_xy(splits["valid"], columns, "type_weak", TYPE_CLASSES)
    X_te, y_te, rows_te = build_xy(splits["test"], columns, "type_weak", TYPE_CLASSES)
    t0 = time.perf_counter()
    type_bundle = train_lgbm_c2(X_tr, y_tr, X_va, y_va, len(TYPE_CLASSES), seed,
                                num_leaves=8, max_depth=4, min_data=10, columns=columns)
    print(f"type_lgbm trained best_iter={type_bundle['best_iteration']} "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)
    proba_te = type_bundle["model"].predict(X_te, num_iteration=type_bundle["best_iteration"])
    pred_te = proba_te.argmax(axis=1)
    proba_va = type_bundle["model"].predict(X_va, num_iteration=type_bundle["best_iteration"])
    pred_va = proba_va.argmax(axis=1)

    results["arms"]["type_lgbm"] = {
        "best_iteration": type_bundle["best_iteration"],
        "top20_gain": type_bundle["top20"],
        "metrics_valid": metrics_report(y_va, pred_va, TYPE_CLASSES),
        "metrics_test": metrics_report(y_te, pred_te, TYPE_CLASSES),
    }
    save_booster(type_bundle["model"], out_type_booster)

    # —— 对比臂：浅层决策树（v3 默认 depth3/leaf5/balanced） ——
    tree = train_tree_c2(X_tr, y_tr, seed)
    tree_pred_te = tree.predict(X_te)
    tree_pred_va = tree.predict(X_va)
    results["arms"]["type_tree"] = {
        "metrics_valid": metrics_report(y_va, tree_pred_va, TYPE_CLASSES),
        "metrics_test": metrics_report(y_te, tree_pred_te, TYPE_CLASSES),
    }

    # —— 副臂：nodule 3 类 LGBM ——
    N_tr, ny_tr, _ = build_xy(splits["train"], columns, "nodule_weak", NODULE_CLASSES)
    N_va, ny_va, _ = build_xy(splits["valid"], columns, "nodule_weak", NODULE_CLASSES)
    N_te, ny_te, _ = build_xy(splits["test"], columns, "nodule_weak", NODULE_CLASSES)
    nodule_bundle = train_lgbm_c2(N_tr, ny_tr, N_va, ny_va, len(NODULE_CLASSES), seed,
                                  num_leaves=8, max_depth=4, min_data=10, columns=columns)
    nodule_pred_te = nodule_bundle["model"].predict(
        N_te, num_iteration=nodule_bundle["best_iteration"]).argmax(axis=1)
    nodule_pred_va = nodule_bundle["model"].predict(
        N_va, num_iteration=nodule_bundle["best_iteration"]).argmax(axis=1)
    results["arms"]["nodule_lgbm"] = {
        "best_iteration": nodule_bundle["best_iteration"],
        "metrics_valid": metrics_report(ny_va, nodule_pred_va, NODULE_CLASSES),
        "metrics_test": metrics_report(ny_te, nodule_pred_te, NODULE_CLASSES),
    }
    save_booster(nodule_bundle["model"], out_nodule_booster)

    # —— 基线：多数类（train 众数直接套用） ——
    majority_type = Counter(y_tr.tolist()).most_common(1)[0][0]
    majority_nodule = Counter(ny_tr.tolist()).most_common(1)[0][0]
    results["arms"]["majority_baseline"] = {
        "train_majority_type": TYPE_CLASSES[majority_type],
        "metrics_test": metrics_report(y_te, np.full_like(y_te, majority_type), TYPE_CLASSES),
    }

    # —— type 预测推导 nodule 的一致率（部署时 nodule 可由 type 直接得到） ——
    derived = np.asarray([TYPE_TO_DERIVED_NODULE[int(TYPE_CLASSES[i])] for i in pred_te])
    ny_te_values = np.asarray([NODULE_CLASSES[i] for i in ny_te])
    results["type_lgbm_derived_nodule"] = {
        "mapping": {str(k): v for k, v in TYPE_TO_DERIVED_NODULE.items()},
        "test_agreement_with_nodule_weak": float((derived == ny_te_values).mean()),
    }

    # —— 覆盖率-一致率曲线（test，按 max prob 降序） ——
    conf = proba_te.max(axis=1)
    order = np.argsort(-conf)
    correct = (pred_te == y_te)
    curve_rows = []
    for cov in COVERAGE_LEVELS:
        k_keep = max(1, int(round(cov * len(y_te))))
        kept_idx = order[:k_keep]
        curve_rows.append({
            "coverage": cov,
            "n": int(k_keep),
            "agreement_within": float(correct[kept_idx].mean()),
            "min_confidence": float(conf[kept_idx].min()),
        })
    with out_curve.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["coverage", "n", "agreement_within", "min_confidence"])
        writer.writeheader()
        writer.writerows(curve_rows)
    results["agreement_coverage_test"] = curve_rows

    # —— per-image 预测（train/valid/test 全量，type probs 全列） ——
    all_rows = [("train", rows_tr, X_tr, y_tr), ("valid", rows_va, X_va, y_va), ("test", rows_te, X_te, y_te)]
    with out_per_image.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["split", "sample_id", "group_id", "type_weak", "type_pred",
                      "type_conf", "type_correct"] + [f"p_type_{c}" for c in TYPE_CLASSES] + \
                     ["nodule_weak", "nodule_pred", "nodule_from_type_pred"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, split_row_list, Xs, ys in all_rows:
            probs = type_bundle["model"].predict(Xs, num_iteration=type_bundle["best_iteration"])
            preds = probs.argmax(axis=1)
            nodule_preds = nodule_bundle["model"].predict(Xs, num_iteration=nodule_bundle["best_iteration"]).argmax(axis=1)
            for r, pred_i, prob_vec, nod_i in zip(split_row_list, preds, probs, nodule_preds):
                writer.writerow({
                    "split": split_name,
                    "sample_id": r["sample_id"],
                    "group_id": r["group_id"],
                    "type_weak": int(float(r["type_weak"])),
                    "type_pred": TYPE_CLASSES[int(pred_i)],
                    "type_conf": f"{float(prob_vec[pred_i]):.6f}",
                    "type_correct": int(TYPE_CLASSES[int(pred_i)] == int(float(r["type_weak"]))),
                    **{f"p_type_{c}": f"{float(prob_vec[i]):.6f}" for i, c in enumerate(TYPE_CLASSES)},
                    "nodule_weak": int(float(r["nodule_weak"])),
                    "nodule_pred": NODULE_CLASSES[int(nod_i)],
                    "nodule_from_type_pred": TYPE_TO_DERIVED_NODULE[TYPE_CLASSES[int(pred_i)]],
                })

    out_metrics.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "type_lgbm_test": results["arms"]["type_lgbm"]["metrics_test"]["accuracy"],
        "type_lgbm_test_macro_f1": results["arms"]["type_lgbm"]["metrics_test"]["macro_f1"],
        "type_lgbm_valid": results["arms"]["type_lgbm"]["metrics_valid"]["accuracy"],
        "type_tree_test": results["arms"]["type_tree"]["metrics_test"]["accuracy"],
        "nodule_lgbm_test": results["arms"]["nodule_lgbm"]["metrics_test"]["accuracy"],
        "majority_test": results["arms"]["majority_baseline"]["metrics_test"]["accuracy"],
        "derived_nodule_agreement_test": results["type_lgbm_derived_nodule"]["test_agreement_with_nodule_weak"],
        "coverage_curve": curve_rows,
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
