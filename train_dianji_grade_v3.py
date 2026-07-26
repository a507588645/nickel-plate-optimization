#!/usr/bin/env python3
"""
电积镍等级 v3：空间/纹理特征增量评估 + 拒判分析（方案 4 + 方案 5）。

- 特征集对照（同 train/valid/test 协议，test 只评一次，不参与任何选择）：
    v2b_base43      = v2 的 43 列（重训基线，对齐 03:00 条目 0.640/0.413）
    v3_spatial_only = 43 列 + 空间特征（sp__*）
    v3_texture_only = 43 列 + 纹理特征（tx__*）
    v3_full         = 43 列 + 空间 + 纹理
- 模型：majority / decision_tree / lightgbm（参数纪律与 v2 相同）。
- 拒判（方案 5）：valid 选最优配置，按 valid 置信度分位定阈值（覆盖≈90/80/70/60%），
  test 上一次性报告各覆盖点指标；拒判样本业务上记 manual_review。
- 全部产物写入 dianji_grade_v3/，标记 unconfirmed_pending_enterprise。
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
DEFAULT_OUT = ROOT / "dianji_grade_v3"
DEFAULT_FEATURES = ROOT / "dianji_grade_v3" / "features_v3_full.csv"
DEFAULT_SEED = 42

CLASS_NAMES_18 = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]

BASE_COLUMNS = (
    ["A_ratio", "B_ratio", "C_ratio", "Airhole_ratio", "has_nodule",
     "total_bbox_union_image_ratio", "detection_count"]
    + [f"count__{n}" for n in CLASS_NAMES_18]
    + [f"area_ratio__{n}" for n in CLASS_NAMES_18]
)

GRADES = ["3#", "9950", "9996", "bad"]
GRADE_TO_ID = {g: i for i, g in enumerate(GRADES)}

# 与 03:00 条目对齐的 v2 基线（v2b conf0.25, LGBM, test）
V2_BASELINE = {"accuracy": 0.640, "macro_f1": 0.413}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="电积镍等级 v3：空间/纹理特征对比 + 拒判分析（未经企业确认）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
    s = str(v).strip().lower()
    if s in {"true", "yes"}:
        return 1.0
    if s in {"false", "no"}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def matrix_from_rows(rows: list[dict[str, Any]], columns: list[str]):
    xs, ys, kept = [], [], []
    for r in rows:
        g = r.get("true_grade") or r.get("grade")
        if g not in GRADE_TO_ID:
            continue
        xs.append([to_float(r.get(c, 0)) for c in columns])
        ys.append(GRADE_TO_ID[g])
        kept.append(r)
    X = np.nan_to_num(np.asarray(xs, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.asarray(ys, dtype=np.int64), kept


def split_rows(rows: list[dict[str, Any]]):
    out = {"train": [], "valid": [], "test": []}
    for r in rows:
        s = (r.get("split") or "").strip()
        if s in out:
            out[s].append(r)
    return out


def metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score,
        precision_recall_fscore_support,
    )

    labels = list(range(len(GRADES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            g: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "support": int(support[i])}
            for i, g in enumerate(GRADES)
        },
        "confusion_matrix": cm.tolist(),
    }


def train_tree(X_tr, y_tr, seed, max_depth, min_leaf):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_leaf,
        class_weight="balanced", random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    return clf


def train_lgbm(X_tr, y_tr, X_va, y_va, seed, num_leaves, max_depth, min_data, columns):
    import lightgbm as lgb

    counts = Counter(y_tr.tolist())
    n = len(y_tr)
    k = len(GRADES)
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


def lgbm_proba(bundle, X):
    return bundle["model"].predict(X, num_iteration=bundle.get("best_iteration"))


def eval_feature_set(name: str, columns: list[str], splits, args) -> dict[str, Any]:
    X_tr, y_tr, _ = matrix_from_rows(splits["train"], columns)
    X_va, y_va, _ = matrix_from_rows(splits["valid"], columns)
    X_te, y_te, _ = matrix_from_rows(splits["test"], columns)

    result: dict[str, Any] = {
        "feature_set": name, "n_features": len(columns),
        "n_train": len(y_tr), "n_valid": len(y_va), "n_test": len(y_te),
        "models": {},
    }

    # majority
    maj = int(Counter(y_tr.tolist()).most_common(1)[0][0])
    for split, X, y in (("valid", X_va, y_va), ("test", X_te, y_te)):
        m = metrics_report(y, np.full(len(y), maj, dtype=np.int64))
        result["models"].setdefault("majority", {})[split] = m

    # decision tree
    tree = train_tree(X_tr, y_tr, args.seed, args.max_depth, args.min_samples_leaf)
    for split, X, y in (("valid", X_va, y_va), ("test", X_te, y_te)):
        result["models"].setdefault("decision_tree", {})[split] = metrics_report(y, tree.predict(X))

    # lightgbm
    bundle = train_lgbm(X_tr, y_tr, X_va, y_va, args.seed,
                        args.lgbm_num_leaves, args.lgbm_max_depth, args.lgbm_min_data, columns)
    for split, X, y in (("valid", X_va, y_va), ("test", X_te, y_te)):
        pred = np.argmax(lgbm_proba(bundle, X), axis=1)
        result["models"].setdefault("lightgbm", {})[split] = metrics_report(y, pred)
    result["lightgbm_top20"] = bundle["top20"]
    result["lightgbm_best_iteration"] = bundle["best_iteration"]

    # 供拒判分析复用
    result["_proba_valid"] = lgbm_proba(bundle, X_va)
    result["_proba_test"] = lgbm_proba(bundle, X_te)
    result["_y_valid"] = y_va
    result["_y_test"] = y_te
    return result


def reject_curve(best: dict[str, Any]) -> list[dict[str, Any]]:
    """阈值只用 valid 定；test 一次性报告。"""
    pv, yv = best["_proba_valid"], best["_y_valid"]
    pt, yt = best["_proba_test"], best["_y_test"]
    conf_v = pv.max(axis=1)
    pred_v = pv.argmax(axis=1)
    conf_t = pt.max(axis=1)
    pred_t = pt.argmax(axis=1)

    rows = []
    for target in (1.0, 0.9, 0.8, 0.7, 0.6):
        if target >= 1.0:
            thr = 0.0
        else:
            thr = float(np.quantile(conf_v, 1.0 - target))
        acc_v_mask = conf_v >= thr
        acc_t_mask = conf_t >= thr
        cov_t = float(acc_t_mask.mean())
        row: dict[str, Any] = {
            "coverage_target": target,
            "threshold_from_valid": round(thr, 4),
            "valid_coverage": float(acc_v_mask.mean()),
            "valid_accuracy": float((pred_v[acc_v_mask] == yv[acc_v_mask]).mean()) if acc_v_mask.any() else None,
            "test_coverage": cov_t,
            "test_n_accepted": int(acc_t_mask.sum()),
        }
        if acc_t_mask.any():
            m = metrics_report(yt[acc_t_mask], pred_t[acc_t_mask])
            row["test_accuracy"] = m["accuracy"]
            row["test_macro_f1"] = m["macro_f1"]
        rows.append(row)
    return rows


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=== dianji_grade_v3 训练对比（方案4+5）===")
    print(f"features: {args.features}")
    print("status: unconfirmed_pending_enterprise")

    if not args.features.exists():
        print(f"缺少特征文件: {args.features}", file=sys.stderr)
        return 2
    rows = read_csv(args.features)
    if args.dry_run:
        print(f"[dry-run] 样本 {len(rows)}，跳过训练")
        return 0

    header = rows[0].keys()
    sp_cols = [c for c in header if c.startswith("sp__")]
    tx_cols = [c for c in header if c.startswith("tx__") and c != "tx__texture_ok"]
    feature_sets = {
        "v2b_base43": BASE_COLUMNS,
        "v3_spatial_only": BASE_COLUMNS + sp_cols,
        "v3_texture_only": BASE_COLUMNS + tx_cols,
        "v3_full": BASE_COLUMNS + sp_cols + tx_cols,
    }
    splits = split_rows(rows)
    print(f"split: train={len(splits['train'])} valid={len(splits['valid'])} test={len(splits['test'])}")
    print(f"特征: base={len(BASE_COLUMNS)} sp={len(sp_cols)} tx={len(tx_cols)}")

    results = []
    for name, cols in feature_sets.items():
        print(f"--- {name} ({len(cols)} 列) ---")
        results.append(eval_feature_set(name, cols, splits, args))

    # 对照表
    comp_rows = []
    for res in results:
        for model in ("majority", "decision_tree", "lightgbm"):
            for split in ("valid", "test"):
                m = res["models"][model][split]
                comp_rows.append({
                    "feature_set": res["feature_set"], "split": split, "model": model,
                    "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4),
                    "f1_3#": round(m["per_class"]["3#"]["f1"], 4),
                    "f1_9950": round(m["per_class"]["9950"]["f1"], 4),
                    "f1_9996": round(m["per_class"]["9996"]["f1"], 4),
                    "f1_bad": round(m["per_class"]["bad"]["f1"], 4),
                    "status": "unconfirmed_pending_enterprise",
                })
    write_csv(out / "model_comparison.csv", comp_rows)

    # 选最优（只用 valid macro_f1，lightgbm 范围内）
    best = max(results, key=lambda r: r["models"]["lightgbm"]["valid"]["macro_f1"])
    base = next(r for r in results if r["feature_set"] == "v2b_base43")
    base_lgbm_test = base["models"]["lightgbm"]["test"]
    best_lgbm_test = best["models"]["lightgbm"]["test"]

    # 拒判曲线
    rc_rows = reject_curve(best)
    write_csv(out / "reject_curve.csv", [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rc_rows
    ])

    # metrics JSON
    for res in results:
        clean = {k: v for k, v in res.items() if not k.startswith("_")}
        (out / f"metrics_{res['feature_set']}.json").write_text(
            json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 报告
    base_acc, base_f1 = base_lgbm_test["accuracy"], base_lgbm_test["macro_f1"]
    best_acc, best_f1 = best_lgbm_test["accuracy"], best_lgbm_test["macro_f1"]
    delta_acc, delta_f1 = best_acc - base_acc, best_f1 - base_f1
    base_drift = base_acc - V2_BASELINE["accuracy"]

    tbl_rows = []
    for res in results:
        m = res["models"]["lightgbm"]["test"]
        tbl_rows.append([
            res["feature_set"], str(res["n_features"]),
            f"{res['models']['lightgbm']['valid']['macro_f1']:.4f}",
            f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}",
            f"{m['per_class']['3#']['f1']:.3f}", f"{m['per_class']['9996']['f1']:.3f}",
        ])
    rc_tbl = [[
        f"{r['coverage_target']:.2f}", f"{r['threshold_from_valid']:.4f}",
        f"{r['test_coverage']:.3f}", f"{r.get('test_accuracy') or 0:.4f}",
        f"{r.get('test_macro_f1') or 0:.4f}",
    ] for r in rc_rows]
    acc85 = next((r for r in rc_rows if (r.get("test_accuracy") or 0) >= 0.85), None)

    report = f"""# 电积镍等级 v3：空间/纹理特征 + 拒判分析报告

> ## ⚠️ 未经企业确认 / UNCONFIRMED
> 全部结果 `unconfirmed_pending_enterprise`，企业书面标准确认前不得上线。
> 小样本警告：3#=46、bad=28（test 各 7/4 张），少数类指标不稳。

- 生成时间：`{datetime.now().isoformat(timespec="seconds")}`
- 特征来源：`dianji_grade_v3\\features_v3_full.csv`（v2b conf0.25 43 列 + 空间 + 纹理）
- 协议：train 训练、valid 选模/早停/定拒判阈值、test 仅最终一次；seed={args.seed}
- v2 基线重训行（v2b_base43）test LGBM：acc={base_acc:.4f} / macroF1={base_f1:.4f}
  （03:00 条目参考值 0.640/0.413，漂移 {base_drift:+.4f}，±0.01 内视为自洽）

## 1. 方案 4：特征增量对照（LightGBM, test）

{md_table(["特征集", "列数", "valid macroF1", "test acc", "test macroF1", "F1_3#", "F1_9996"], tbl_rows)}

- 最优特征集：**{best['feature_set']}**；相对 v2 基线重训 Δacc={delta_acc:+.4f}、ΔmacroF1={delta_f1:+.4f}。
- 全模型对照见 `model_comparison.csv`（majority/decision_tree/lightgbm × valid/test）。
- 最优特征集 LightGBM 特征重要性 top20：

{chr(10).join(f"- `{k}`: {v:.4f}" for k, v in best["lightgbm_top20"])}

## 2. 方案 5：拒判分析（最优配置 = {best['feature_set']} + LightGBM）

阈值仅由 valid 置信度分位确定；test 一次性报告。拒判样本业务上记 `manual_review`。

{md_table(["目标覆盖率", "阈值(valid)", "test 实际覆盖", "test acc", "test macroF1"], rc_tbl)}

- acc≥0.85 的覆盖点：{"达到（覆盖率≈" + f"{acc85['test_coverage']:.0%}）" if acc85 else "未达到"}。

## 3. 结论与局限

- 空间/纹理特征对 test 的增量：Δacc={delta_acc:+.4f}、ΔmacroF1={delta_f1:+.4f}（<0.01 视为无实质提升）。
- 3#/bad 小样本约束不变；9950↔9996 混对是否缓解见 `metrics_v3_full.json` 混淆矩阵。
- 拒判可作为部署保底策略，但阈值与覆盖率均为未确认候选，需企业确认业务可接受精度。

## 4. 复现

```text
cd /d E:\\PythonProject\\优化
.\\.venv_classify\\Scripts\\python.exe extract_dianji_grade_v3_features.py
.\\.venv_classify\\Scripts\\python.exe train_dianji_grade_v3.py
```
"""
    (out / "report.md").write_text(report, encoding="utf-8")

    readme = f"""# dianji_grade_v3 产物说明

- 状态：**unconfirmed_pending_enterprise**，企业确认前不得用于生产。
- 内容：方案 4（空间/纹理特征）与方案 5（拒判分析）对 v2 基线的对照。
- 文件：
  - `features_v3_full.csv` / `features_spatial.csv` / `features_texture.csv`：特征（提取脚本 `extract_dianji_grade_v3_features.py`）
  - `model_comparison.csv`：三模型 × 四特征集 × valid/test
  - `metrics_*.json`：各特征集详细指标与混淆矩阵
  - `reject_curve.csv`：拒判曲线（阈值仅来自 valid）
  - `report.md`：结论报告
- 基线对齐：v2b_base43 重训 test LGBM acc={base_acc:.4f} / macroF1={base_f1:.4f}（03:00 条目 0.640/0.413）
- 生成时间：{datetime.now().isoformat(timespec="seconds")}
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    print(f"\n基线重训: acc={base_acc:.4f} macroF1={base_f1:.4f}")
    print(f"最优({best['feature_set']}): acc={best_acc:.4f} macroF1={best_f1:.4f} "
          f"(Δ{best_acc - base_acc:+.4f}/{best_f1 - base_f1:+.4f})")
    print(f"产物: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
