#!/usr/bin/env python3
"""B3 泄漏隔离评估：在 B3 检测器从未见过的 grade-test 子集上对比 B0/B3/arm3。

泄漏判定（保守并集，任一命中即视为泄漏）：
1. SHA-256 命中：等级视图行 sha256 ∈ B3 训练 990 图哈希集（config 包 image_hashes.json）；
2. stem 命中：image_stem / image_name_raw ∈ B3 train/val 清单 stem 集；
3. 近重复组传染：该行的 dup_group_id 内任一成员命中 1 或 2。

随后在 grade-test 全集（161）与洁净子集上分别评估三个特征血缘的 LGBM
（协议与 train_dianji_grade_v3.py 完全一致：train 训练、valid 早停、seed=42）：
- B0   = dianji_grade_y11m640_v1（18 类 ONNX 跨域，43+20=63 列）
- B3   = dianji_grade_b3_y11m640_v1（5 类域内 ultralytics，13+20=33 列）
- arm3 = dianji_grade_detector_compare_v1/arm3_slf640（5 类域内 YOLOv5，13+20=33 列）

产物：dianji_grade_b3_y11m640_v1/leakage_isolated_eval.json + .md
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train_dianji_grade_v3 import (  # noqa: E402
    BASE_COLUMNS,
    SLF_BASE_COLUMNS,
    GRADES,
    lgbm_proba,
    matrix_from_rows,
    metrics_report,
    read_csv,
    split_rows,
    train_lgbm,
)

VIEW_MANIFEST = Path(r"E:/PythonProject/archives/dianjinie_grade_classification_v1/manifests/samples.csv")
B3_TASK = ROOT / "b3_dianjinie0211_y11m640_task_v1"
B3_IMAGE_HASHES = B3_TASK / "kaggle_config_upload_v1" / "image_hashes.json"
B3_TRAIN_TXT = B3_TASK / "splits" / "train_local.txt"
B3_VAL_TXT = B3_TASK / "splits" / "val_local.txt"

ARMS = {
    "b0_y11m18_onnx640": {
        "csv": ROOT / "dianji_grade_y11m640_v1" / "features_y11m640_v1.csv",
        "columns": list(BASE_COLUMNS),
    },
    "b3_d0211y11m640": {
        "csv": ROOT / "dianji_grade_b3_y11m640_v1" / "features_b3y11m640_v1.csv",
        "columns": list(SLF_BASE_COLUMNS),
    },
    "arm3_slf640": {
        "csv": ROOT / "dianji_grade_detector_compare_v1" / "arm3_slf640" / "features_slf640_v1.csv",
        "columns": list(SLF_BASE_COLUMNS),
    },
}

OUT_JSON = ROOT / "dianji_grade_b3_y11m640_v1" / "leakage_isolated_eval.json"
OUT_MD = ROOT / "dianji_grade_b3_y11m640_v1" / "leakage_isolated_eval.md"
SEED = 42


def load_b3_identities() -> dict[str, set[str]]:
    hashes = json.loads(B3_IMAGE_HASHES.read_text(encoding="utf-8"))
    sha_train = {v for k, v in hashes.items() if k.startswith("images/train/")}
    sha_val = {v for k, v in hashes.items() if k.startswith("images/val/")}
    stem_train = {Path(line.strip()).stem for line in B3_TRAIN_TXT.read_text(encoding="utf-8").splitlines() if line.strip()}
    stem_val = {Path(line.strip()).stem for line in B3_VAL_TXT.read_text(encoding="utf-8").splitlines() if line.strip()}
    return {
        "sha_train": sha_train,
        "sha_val": sha_val,
        "stem_train": stem_train,
        "stem_val": stem_val,
    }


def row_keys(row: dict[str, str]) -> set[str]:
    keys = {row["image_stem"].strip()}
    if row.get("image_name_raw"):
        keys.add(row["image_name_raw"].strip())
    return keys


def classify_leakage(rows: list[dict[str, str]], ident: dict[str, set[str]]) -> dict[str, dict[str, str]]:
    """返回 image_name -> {sha_hit, stem_hit, group_hit, leaked}"""
    sha_all = ident["sha_train"] | ident["sha_val"]
    stem_all = ident["stem_train"] | ident["stem_val"]

    direct: dict[str, dict[str, bool]] = {}
    for row in rows:
        sha_hit = row["sha256"].strip().lower() in sha_all
        stem_hit = bool(row_keys(row) & stem_all)
        direct[row["image_name"]] = {"sha_hit": sha_hit, "stem_hit": stem_hit}

    group_members: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group_members[row.get("dup_group_id", "") or row["image_name"]].append(row["image_name"])
    group_leaked = {
        gid: any(direct[n]["sha_hit"] or direct[n]["stem_hit"] for n in members)
        for gid, members in group_members.items()
    }

    out: dict[str, dict[str, str]] = {}
    for row in rows:
        gid = row.get("dup_group_id", "") or row["image_name"]
        d = direct[row["image_name"]]
        sha_where = []
        sha_l = row["sha256"].strip().lower()
        if sha_l in ident["sha_train"]:
            sha_where.append("train")
        if sha_l in ident["sha_val"]:
            sha_where.append("val")
        out[row["image_name"]] = {
            "sha_hit": d["sha_hit"],
            "stem_hit": d["stem_hit"],
            "group_hit": (not d["sha_hit"] and not d["stem_hit"]) and group_leaked.get(gid, False),
            "leaked": d["sha_hit"] or d["stem_hit"] or group_leaked.get(gid, False),
            "sha_in": "+".join(sha_where),
        }
    return out


def tx_columns(header: list[str]) -> list[str]:
    cols = [c for c in header if c.startswith("tx__") and c != "tx__texture_ok"]
    if len(cols) != 20:
        raise ValueError(f"tx 列数不是 20: {len(cols)}")
    return cols


def eval_arm(name: str, cfg: dict[str, object], clean_test_names: set[str]) -> dict[str, object]:
    rows = read_csv(Path(cfg["csv"]))
    header = list(rows[0])
    cols = list(cfg["columns"]) + tx_columns(header)
    missing = [c for c in cols if c not in header]
    if missing:
        raise ValueError(f"{name} 缺列: {missing[:5]}")
    splits = split_rows(rows)

    X_tr, y_tr, _ = matrix_from_rows(splits["train"], cols)
    X_va, y_va, _ = matrix_from_rows(splits["valid"], cols)
    bundle = train_lgbm(X_tr, y_tr, X_va, y_va, SEED, 8, 4, 10, cols)

    test_rows = splits["test"]
    X_te, y_te, kept_te = matrix_from_rows(test_rows, cols)
    pred_te = lgbm_proba(bundle, X_te).argmax(axis=1)

    clean_idx = [i for i, r in enumerate(kept_te) if r["image_name"] in clean_test_names]
    result: dict[str, object] = {
        "n_features": len(cols),
        "best_iteration": bundle["best_iteration"],
        "test_full": metrics_report(y_te, pred_te),
        "clean_subset_n": len(clean_idx),
    }
    if clean_idx:
        import numpy as np

        y_sub = y_te[clean_idx]
        p_sub = pred_te[clean_idx]
        result["clean_subset"] = metrics_report(y_sub, p_sub)
        result["clean_subset_grade_support"] = dict(
            Counter(GRADES[int(y)] for y in y_sub.tolist())
        )
        result["clean_subset_correct"] = int((y_sub == p_sub).sum())
    return result


def main() -> int:
    rows = read_csv(VIEW_MANIFEST)
    if len(rows) != 1072:
        raise ValueError(f"manifest 行数不是 1072: {len(rows)}")
    ident = load_b3_identities()
    leak = classify_leakage(rows, ident)

    per_split: dict[str, Counter[str]] = {}
    clean_test_names: set[str] = set()
    clean_test_detail: list[dict[str, str]] = []
    for row in rows:
        sp = row["split"]
        per_split.setdefault(sp, Counter())
        info = leak[row["image_name"]]
        per_split[sp]["total"] += 1
        if info["sha_hit"]:
            per_split[sp]["sha_hit"] += 1
        if info["stem_hit"]:
            per_split[sp]["stem_hit"] += 1
        if info["group_hit"]:
            per_split[sp]["group_hit_only"] += 1
        if info["leaked"]:
            per_split[sp]["leaked_union"] += 1
        elif sp == "test":
            clean_test_names.add(row["image_name"])
            clean_test_detail.append(
                {
                    "image_name": row["image_name"],
                    "grade": row["grade"],
                    "source_dataset": row.get("source_dataset", ""),
                    "dup_group_id": row.get("dup_group_id", ""),
                }
            )

    print("=== B3 泄漏复核（SHA + stem + dup_group 并集） ===")
    for sp in ("train", "valid", "test"):
        c = per_split[sp]
        print(
            f"grade-{sp}: total={c['total']} sha={c['sha_hit']} stem={c['stem_hit']} "
            f"group_only={c['group_hit_only']} union={c['leaked_union']} "
            f"clean={c['total'] - c['leaked_union']}"
        )
    print(f"grade-test 洁净子集: {len(clean_test_names)} 张")
    for d in clean_test_detail:
        print(f"  {d['image_name']} grade={d['grade']} src={d['source_dataset']} grp={d['dup_group_id']}")

    results = {name: eval_arm(name, cfg, clean_test_names) for name, cfg in ARMS.items()}

    print("\n=== LGBM（seed=42，协议同 train_dianji_grade_v3） ===")
    for name, res in results.items():
        full = res["test_full"]
        print(f"{name}: test161 acc={full['accuracy']:.4f} macroF1={full['macro_f1']:.4f}", end="")
        if "clean_subset" in res:
            sub = res["clean_subset"]
            print(
                f" | clean({res['clean_subset_n']}) acc={sub['accuracy']:.4f} macroF1={sub['macro_f1']:.4f}"
                f" support={res['clean_subset_grade_support']}"
            )
        else:
            print(" | clean 子集为空")

    payload = {
        "seed": SEED,
        "leakage_method": "sha256 ∪ stem(image_stem/image_name_raw) ∪ dup_group 传染，对 B3 990 图（train774/val216）",
        "per_grade_split": {k: dict(v) for k, v in per_split.items()},
        "clean_test_names": sorted(clean_test_names),
        "clean_test_detail": clean_test_detail,
        "arms": results,
        "caveat": "clean 子集 n 很小，macroF1 逐类支持度极低，仅作方向性证据；全部 unconfirmed_pending_enterprise",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# B3 泄漏隔离评估（grade-test 洁净子集）",
        "",
        "> 全部结果 `unconfirmed_pending_enterprise`；洁净子集样本量极小，仅作方向性证据。",
        "",
        "## 泄漏复核（SHA-256 ∪ stem ∪ dup_group 传染，对 B3 990 图）",
        "",
        "| grade split | total | sha 命中 | stem 命中 | 仅组传染 | 并集泄漏 | 洁净 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for sp in ("train", "valid", "test"):
        c = per_split[sp]
        lines.append(
            f"| {sp} | {c['total']} | {c['sha_hit']} | {c['stem_hit']} "
            f"| {c['group_hit_only']} | {c['leaked_union']} | {c['total'] - c['leaked_union']} |"
        )
    lines += [
        "",
        f"grade-test 洁净子集 **{len(clean_test_names)} 张**：",
        "",
    ]
    lines += [f"- `{d['image_name']}` grade={d['grade']} src={d['source_dataset']}" for d in clean_test_detail]
    lines += [
        "",
        "## LGBM 对比（seed=42，test161 全集 vs 洁净子集）",
        "",
        "| 血缘 | 列数 | test161 acc | test161 macroF1 | 洁净 n | 洁净 acc | 洁净 macroF1 | 洁净等级分布 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, res in results.items():
        full = res["test_full"]
        if "clean_subset" in res:
            sub = res["clean_subset"]
            lines.append(
                f"| {name} | {res['n_features']} | {full['accuracy']:.4f} | {full['macro_f1']:.4f} "
                f"| {res['clean_subset_n']} | {sub['accuracy']:.4f} | {sub['macro_f1']:.4f} "
                f"| {res['clean_subset_grade_support']} |"
            )
        else:
            lines.append(
                f"| {name} | {res['n_features']} | {full['accuracy']:.4f} | {full['macro_f1']:.4f} | 0 | — | — | — |"
            )
    lines += [
        "",
        "解读约束：洁净子集 macroF1 的单样本波动可达 ±0.1 以上（bad/3# 支持度个位数），",
        "只能判断「B3 在未见图上是否崩塌」，不能给出精确排名。",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n产物: {OUT_JSON} / {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
