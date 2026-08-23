"""Synthetic end-to-end coverage for the E2 paired archive verifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("e2_verifier", ROOT / "verify_l1_e2_mosaic_pair.py")
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_archive(root: Path, role: str, map50_95: float, recall_offset: float) -> Path:
    run_name = VERIFIER.EXPECTED_RUN_NAMES[role]
    run = root / run_name
    best_bytes = f"synthetic-best-{role}".encode("ascii")
    best_hash = hashlib.sha256(best_bytes).hexdigest()
    common = {
        "development_only": True,
        "frozen_test_set": None,
        "fold": 1,
        "split_counts": {"train": 751, "val": 376, "test_frozen": None},
        "base_checkpoint": {"sha256": VERIFIER.EXPECTED_BASE_CHECKPOINT_SHA256},
        "runtime_pins": {
            "torch": "2.6.0+cu124",
            "torchvision": "0.21.0",
            "numpy": "1.26.4",
            "ultralytics": "8.4.66",
        },
        "evaluation": {
            "entry": "l1_tile_pipeline.py --mode evaluate",
            "pipeline_sha256": VERIFIER.EXPECTED_PIPELINE_SHA256,
            "model_weight": "best.pt",
            "tile_size": 4096,
            "overlap": 0.0,
            "fixed_baseline_full_mAP50_95": VERIFIER.EXPECTED_BASELINE_MAP50_95,
        },
        "training": {
            "epochs": 30,
            "imgsz": 640,
            "batch": 8,
            "optimizer": "SGD",
            "seed": 0,
            "deterministic": True,
            "resume": False,
            "close_mosaic": 0,
            "patience": 100,
            "save_period": 1,
        },
    }
    common_hash = hashlib.sha256(VERIFIER.canonical_json(common).encode("utf-8")).hexdigest()
    mosaic = VERIFIER.EXPECTED_MOSAIC[role]
    sibling = "candidate" if role == "control" else "control"
    contract = {
        "pair_id": "synthetic-e2-pair",
        "common_contract_sha256": common_hash,
        "common_contract": common,
        "variant": {"role": role, "mosaic": mosaic, "run_name": run_name},
        "expected_sibling": {
            "role": sibling,
            "mosaic": VERIFIER.EXPECTED_MOSAIC[sibling],
            "run_name": VERIFIER.EXPECTED_RUN_NAMES[sibling],
        },
    }
    requested_training = {
        "epochs": 30,
        "imgsz": 640,
        "batch": 8,
        "optimizer": "SGD",
        "seed": 0,
        "deterministic": True,
        "resume": False,
        "close_mosaic": 0,
        "patience": 100,
        "save_period": 1,
    }
    write_json(
        run / "requested_config.json",
        {
            "run_name": run_name,
            "pair_id": contract["pair_id"],
            "role": role,
            "mosaic": mosaic,
            "development_only": True,
            "frozen_test_set": None,
            "base_checkpoint_sha256": VERIFIER.EXPECTED_BASE_CHECKPOINT_SHA256,
            "training": requested_training,
        },
    )
    write_json(run / "e2_pair_contract.json", contract)
    write_json(
        run / "environment.json",
        {"torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124", "numpy": "1.26.4", "ultralytics": "8.4.66"},
    )
    (run / "weights").mkdir(parents=True, exist_ok=True)
    (run / "weights" / "best.pt").write_bytes(best_bytes)
    (run / "weights" / "last.pt").write_bytes(b"synthetic-last")
    (run / "results.csv").write_text("epoch,metrics/mAP50-95(B)\n0,0.1\n", encoding="utf-8")
    (run / "README.md").write_text("synthetic\n", encoding="utf-8")
    (run / "dataset_config").mkdir(parents=True, exist_ok=True)
    (run / "dataset_config" / "data.yaml").write_text("nc: 18\n", encoding="utf-8")
    train = [f"/synthetic/images/train_{index:04d}.jpg" for index in range(751)]
    val = [f"/synthetic/images/val_{index:04d}.jpg" for index in range(376)]
    (run / "dataset_config" / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (run / "dataset_config" / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    args = {
        "name": run_name,
        "epochs": 30,
        "imgsz": 640,
        "batch": 8,
        "optimizer": "SGD",
        "seed": 0,
        "deterministic": True,
        "resume": False,
        "close_mosaic": 0,
        "patience": 100,
        "save_period": 1,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "auto_augment": "randaugment",
        "erasing": 0.4,
        "mosaic": mosaic,
    }
    (run / "args.yaml").write_text("\n".join(f"{key}: {str(value).lower() if isinstance(value, bool) else value}" for key, value in args.items()) + "\n", encoding="utf-8")
    recalls = {name: 0.50 + recall_offset for name in VERIFIER.FOCUS_CLASSES}
    report = {"mAP50_95": map50_95, "model_sha256": best_hash, "per_class": {name: {"recall_conf025_iou50": value} for name, value in recalls.items()}}
    evaluation_root = run / "evaluation" / "tile_evaluation"
    write_json(evaluation_root / "baseline_full.json", {"mAP50_95": VERIFIER.EXPECTED_BASELINE_MAP50_95, "model_sha256": VERIFIER.EXPECTED_BASE_CHECKPOINT_SHA256})
    write_json(evaluation_root / "tile_model_full.json", report)
    write_json(
        run / "evaluation" / "evaluation_summary.json",
        {
            "role": role,
            "mosaic": mosaic,
            "candidate_best_model_sha256": best_hash,
            "candidate_best_mAP50_95": map50_95,
            "baseline_model_sha256": VERIFIER.EXPECTED_BASE_CHECKPOINT_SHA256,
        },
    )
    archive = root / f"{role}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(run.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(root))
    return archive


class VerifyE2MosaicPairTests(unittest.TestCase):
    def test_accepts_paired_archives_and_writes_pass_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = write_archive(root, "control", 0.190, 0.00)
            candidate = write_archive(root, "candidate", 0.205, 0.01)
            output = root / "verification"
            result = VERIFIER.main(
                [
                    "--control-archive", str(control),
                    "--candidate-archive", str(candidate),
                    "--output-root", str(output),
                ]
            )
            self.assertEqual(result, 0)
            decision = json.loads((output / "e2_pair_gate_decision.json").read_text(encoding="utf-8"))
            self.assertEqual(decision["status"], "pass_expand_eligible")
            self.assertAlmostEqual(decision["candidate_minus_control_mAP50_95"], 0.015)
            self.assertTrue((output / "control" / "weights" / "best.pt").is_file())
            self.assertTrue((output / "candidate" / "evaluation" / "tile_evaluation" / "tile_model_full.json").is_file())

    def test_focus_recall_decline_blocks_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = write_archive(root, "control", 0.190, 0.00)
            candidate = write_archive(root, "candidate", 0.205, -0.01)
            verified_control = VERIFIER.validate_archive(
                control, "control", VERIFIER.EXPECTED_BASE_CHECKPOINT_SHA256
            )
            verified_candidate = VERIFIER.validate_archive(
                candidate, "candidate", VERIFIER.EXPECTED_BASE_CHECKPOINT_SHA256
            )
            decision = VERIFIER.gate_decision(verified_control, verified_candidate)
            self.assertEqual(decision["status"], "not_passed_do_not_expand")
            self.assertFalse(decision["checks"]["focus_recall_non_degradation_pass"])


if __name__ == "__main__":
    unittest.main()
