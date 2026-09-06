# -*- coding: utf-8 -*-
"""C2.5 闸门机械证据：3 seed 复验 + 拒判阈值标定（路线文档 §4.2）。

- 3 seed（42/43/44）重训 type_lgbm（协议逐字镜像 train_dianji_grade_v3），
  报告 test 一致率 mean±std（AGENTS §8：3 种子才能宣称稳定）；
- 拒判阈值只在 VALID 上标定（目标覆盖 0.70，协议同 B3clean 契约「valid 分位数标定、
  test 仅报告一次」）：valid 覆盖率扫描 → 选定阈值 → test 用该阈值报告一次；
- 产出 c2p5_gate_evidence.json。不切换任何生产配置。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train_c2p3_type_lgbm as t3  # noqa: E402

SEEDS = [42, 43, 44]
TARGET_COVERAGE = 0.70
SWEEP_COVERAGES = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]


def main() -> int:
    rows, columns = t3.load_rows()
    splits = t3.split_rows(rows)
    X_tr, y_tr, _ = t3.build_xy(splits["train"], columns, "type_weak", t3.TYPE_CLASSES)
    X_va, y_va, _ = t3.build_xy(splits["valid"], columns, "type_weak", t3.TYPE_CLASSES)
    X_te, y_te, _ = t3.build_xy(splits["test"], columns, "type_weak", t3.TYPE_CLASSES)

    per_seed: list[dict[str, Any]] = []
    primary_bundle = None
    primary_proba_va = None
    for seed in SEEDS:
        bundle = t3.train_lgbm_c2(X_tr, y_tr, X_va, y_va, len(t3.TYPE_CLASSES), seed,
                                  num_leaves=8, max_depth=4, min_data=10, columns=columns)
        pred_te = bundle["model"].predict(X_te, num_iteration=bundle["best_iteration"]).argmax(axis=1)
        proba_va = bundle["model"].predict(X_va, num_iteration=bundle["best_iteration"])
        pred_va = proba_va.argmax(axis=1)
        entry = {
            "seed": seed,
            "best_iteration": bundle["best_iteration"],
            "valid_agreement": float((pred_va == y_va).mean()),
            "test_agreement": float((pred_te == y_te).mean()),
            "test_macro_f1": t3.metrics_report(y_te, pred_te, t3.TYPE_CLASSES)["macro_f1"],
        }
        per_seed.append(entry)
        if seed == SEEDS[0]:
            primary_bundle, primary_proba_va = bundle, proba_va
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    agreements = [e["test_agreement"] for e in per_seed]
    mean, std = float(np.mean(agreements)), float(np.std(agreements))

    # —— VALID 覆盖率扫描（仅 valid，不碰 test） ——
    conf_va = primary_proba_va.max(axis=1)
    correct_va = (primary_proba_va.argmax(axis=1) == y_va)
    sweep = []
    for cov in SWEEP_COVERAGES:
        thr = float(np.quantile(conf_va, 1.0 - cov))
        acc = float(correct_va[conf_va >= thr].mean())
        sweep.append({"coverage": cov, "threshold": round(thr, 4),
                      "val_agreement_within": acc})
        print(f"cov={cov:.2f} thr={thr:.4f} val_acc@={acc:.4f}", flush=True)

    chosen = next(s for s in sweep if s["coverage"] == TARGET_COVERAGE)
    threshold = chosen["threshold"]

    # —— test 用该阈值报告一次（协议同 B3clean 契约） ——
    proba_te = primary_bundle["model"].predict(X_te, num_iteration=primary_bundle["best_iteration"])
    conf_te = proba_te.max(axis=1)
    correct_te = (proba_te.argmax(axis=1) == y_te)
    accepted = conf_te >= threshold
    test_at_threshold = {
        "threshold": threshold,
        "coverage": float(accepted.mean()),
        "agreement_within": float(correct_te[accepted].mean()),
        "manual_review_rate": float((~accepted).mean()),
        "note": "test 以 valid 标定阈值报告一次（B3clean 契约同协议），此后不得再读 test",
    }

    evidence = {
        "task": "C2.5 闸门机械证据（3 seed 复验 + valid 拒判阈值标定）",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "train_dianji_grade_v3 逐字镜像；阈值 valid 分位数标定（目标覆盖 0.70）；test 报告一次后冻结",
        "seeds": per_seed,
        "test_agreement_mean": mean,
        "test_agreement_std": std,
        "valid_coverage_sweep": sweep,
        "chosen_threshold": chosen,
        "test_at_threshold": test_at_threshold,
        "gate_interpretation": {
            "一致率≥95%（无拒判）": f"未达（3 seed test mean={mean:.4f}）",
            "一致率≥95%（带拒判）": f"valid 覆盖 {TARGET_COVERAGE} 阈值 {threshold} 下 test 接收区一致率 "
                                    f"{test_at_threshold['agreement_within']:.4f}",
            "复核集不劣于旧方案": "C2.4 人工复核未执行（用户 2026-09-06 反馈无法按 type 语义复核），"
                                 "该条件转由现场监控替代验证，需用户确认此口径",
        },
    }
    (HERE / "c2p5_gate_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"seeds_test_mean": mean, "seeds_test_std": std,
                      "threshold": threshold,
                      "test_agreement_within": test_at_threshold["agreement_within"],
                      "test_coverage": test_at_threshold["coverage"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
