#!/usr/bin/env python3
"""Verify the B3-clean run archive: weight SHA-256, best epoch, evidence set."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

RUN = Path(r"E:/PythonProject/models/runs/dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211clean1")

REQUIRED = [
    "weights/best.pt", "weights/last.pt", "args.yaml", "requested_config.json",
    "environment.json", "results.csv", "README.md",
    "dataset_config/data.yaml", "dataset_config/train.txt", "dataset_config/val.txt",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [rel for rel in REQUIRED if not (RUN / rel).is_file()]
    print(f"[evidence] missing={missing}")

    rows = list(csv.DictReader((RUN / "results.csv").read_text(encoding="utf-8").splitlines()))
    print(f"[results] epochs={len(rows)}")
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    print(f"[results] best epoch by mAP50-95: {best['epoch']} "
          f"mAP50={float(best['metrics/mAP50(B)']):.5f} mAP50-95={float(best['metrics/mAP50-95(B)']):.5f}")
    epochs_after_best = int(rows[-1]["epoch"]) - int(best["epoch"])
    print(f"[results] epochs after best: {epochs_after_best} (patience=100)")

    hashes = {rel: sha256_file(RUN / rel) for rel in ("weights/best.pt", "weights/last.pt")}
    (RUN / "weights_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for rel, digest in hashes.items():
        print(f"[sha256] {rel} = {digest}")

    prefix = "/kaggle/input/datasets/lly3323w/dianjinie/"
    train_txt = [line[len(prefix):] if line.startswith(prefix) else line
                 for line in (RUN / "dataset_config" / "train.txt").read_text(encoding="utf-8").split()]
    val_txt = [line[len(prefix):] if line.startswith(prefix) else line
               for line in (RUN / "dataset_config" / "val.txt").read_text(encoding="utf-8").split()]
    local_train = set(Path(r"E:/PythonProject/优化/b3clean_dianjinie0211_y11m640_task_v1/splits/train_local.txt").read_text(encoding="utf-8").split())
    local_val = set(Path(r"E:/PythonProject/优化/b3clean_dianjinie0211_y11m640_task_v1/splits/val_local.txt").read_text(encoding="utf-8").split())
    print(f"[split] run train={len(train_txt)} val={len(val_txt)}; "
          f"train match={set(train_txt) == local_train} val match={set(val_txt) == local_val}")


if __name__ == "__main__":
    main()
