# -*- coding: utf-8 -*-
"""C2.5 部署件打包：电解镍学习型判级 booster + 契约 JSON（用户 2026-09-06 已确认改道口径）。

- 从 C2.3/C2.5 产物组装部署契约（63 列特征、5 类 type 1/2/3/4/6、拒判阈值 0.8525、
  type→nodule 映射、provenance 全记录）；
- booster 拷入 worktree weights/（.gitattributes weights/** -text 防 CRLF）；
- 回读验证：ASCII 临时路径加载 booster，特征名与契约一致。
"""
from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import lightgbm as lgb

HERE = Path(__file__).resolve().parent
WEAKLABEL_DIR = HERE.parent / "dianjie_grade_weaklabel_v1_20260906"
WORKTREE = Path("C:/Users/23916/AppData/Local/Temp/jcw_b3clean_wt")
SRC_BOOSTER = WEAKLABEL_DIR / "c2p3_type_booster.txt"
V1_CONTRACT = WORKTREE / "weights" / "dianji_grade_lgbm_v1_contract.json"
OUT_CONTRACT_WT = WORKTREE / "weights" / "dianjie_grade_lgbm_c2p3_v1_contract.json"
OUT_BOOSTER_WT = WORKTREE / "weights" / "dianjie_grade_lgbm_c2p3_v1.txt"


def file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    metrics = json.loads((WEAKLABEL_DIR / "c2p3_metrics.json").read_text(encoding="utf-8"))
    evidence = json.loads((WEAKLABEL_DIR / "c2p5_gate_evidence.json").read_text(encoding="utf-8"))
    v1 = json.loads(V1_CONTRACT.read_text(encoding="utf-8"))
    feature_names = list(v1["feature_names"])
    assert len(feature_names) == 63

    arm = metrics["arms"]["type_lgbm"]
    best_iteration = int(arm["best_iteration"])
    threshold = float(evidence["chosen_threshold"]["threshold"])
    assert abs(threshold - 0.8525) < 1e-9

    shutil.copy2(SRC_BOOSTER, OUT_BOOSTER_WT)
    booster_sha = file_sha256(OUT_BOOSTER_WT)
    onnx_sha = file_sha256(WORKTREE / "weights" / "dianjienie_yolo11m_ryw18_640.onnx")
    assert onnx_sha.startswith("901ccf60")

    with tempfile.TemporaryDirectory(prefix="dianjie_lgbm_") as tmp:
        tmp_model = Path(tmp) / "b.txt"
        shutil.copy2(OUT_BOOSTER_WT, tmp_model)
        booster = lgb.Booster(model_file=str(tmp_model))
        assert tuple(booster.feature_name()) == tuple(feature_names), "特征名不一致"
        assert booster.num_model_per_iteration() == 5, "num_class 非 5"
        round_trip = True

    contract = {
        "contract_version": "dianjie_grade_c2p3_v1",
        "status": "weaklabel_distilled_20260906_user_approved_field_monitoring",
        "feature_names": feature_names,
        "feature_count": 63,
        "grades": ["1", "2", "3", "4", "6"],
        "grade_to_id": {"1": 0, "2": 1, "3": 2, "4": 3, "6": 4},
        "id_to_grade": {"0": "1", "1": "2", "2": "3", "3": "4", "4": "6"},
        "type_to_nodule": {"1": 3, "2": 2, "3": 0, "4": 3, "6": 3},
        "model": {
            "file": "dianjie_grade_lgbm_c2p3_v1.txt",
            "format": "lightgbm_booster_text",
            "best_iteration": best_iteration,
            "num_class": 5,
            "sha256": booster_sha,
            "feature_names_in_model": feature_names,
        },
        "rejection": {
            "selected_target_coverage": 0.70,
            "threshold": threshold,
            "calibrated_on": "valid only (quantile of max prob)",
            "test_coverage_reference": evidence["test_at_threshold"]["coverage"],
            "test_agreement_within_reference": evidence["test_at_threshold"]["agreement_within"],
            "manual_review_when_confidence_below_threshold": True,
        },
        "extraction": {
            "weight_path": str(WORKTREE / "weights" / "dianjienie_yolo11m_ryw18_640.onnx"),
            "weight_sha256": onnx_sha,
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.45,
            "device": "cpu",
            "class_names": list(v1["extraction"]["class_names"]),
            "aggregation_reference": "inference/grade_features.py extract_grade_features（63 列生产口径，5a7bb5c）",
        },
        "training": {
            "seed": 42,
            "protocol": "train_dianji_grade_v3 逐字镜像（类平衡权重 w=n/(k*count)，早停 40@valid，test 只评一次）",
            "weak_label_source": "优化/dianjie_grade_weaklabel_v1_20260906/weaklabels.jsonl（c5055bc 旧方案管线，企业认可配置）",
            "features_csv": "优化/dianjie_grade_weaklabel_v1_20260906/features_c2p2_y11m640.csv",
            "splits": {"train": 376, "valid": 375, "test": 376, "grouping": "板级 group，组无跨划分"},
            "seeds_test_agreement": [e["test_agreement"] for e in evidence["seeds"]],
        },
        "metrics": {
            "test_agreement": arm["metrics_test"]["accuracy"],
            "test_macro_f1": arm["metrics_test"]["macro_f1"],
            "three_seed_mean": evidence["test_agreement_mean"],
            "three_seed_std": evidence["test_agreement_std"],
        },
        "gate": {
            "一致率≥95%（带拒判）": "PASS（0.9529 @ 阈值 0.8525，test 报告一次）",
            "复核集不劣于旧方案": "未执行——用户确认 C2.4 改道：信任弱标签+拒判部署+现场监控替代验证",
        },
        "round_trip": {"verified": round_trip},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    OUT_CONTRACT_WT.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    HERE.joinpath("dianjie_grade_lgbm_c2p3_v1_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"booster_sha": booster_sha[:16], "best_iteration": best_iteration,
                      "threshold": threshold, "feature_count": len(feature_names)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
