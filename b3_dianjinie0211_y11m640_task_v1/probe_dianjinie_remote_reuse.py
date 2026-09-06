#!/usr/bin/env python3
"""Probe lly3323w/dianjinie: can it serve as the B3 image dataset?

Downloads the remote file listing (all pages), then verifies:
1. every B3 included image exists remotely with an identical size,
2. labels/train + labels/val directories exist remotely,
3. remote data.yaml matches the local one.
Read-only; writes a verdict JSON into this task directory.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent
DATASET_REF = "lly3323w/dianjinie"
LOCAL_ROOT = Path(r"E:/PythonProject/datasets/Dianjinie0211")
EXPECTED_CLASSES = ["waterstain", "waterblock", "colorstain", "nodule", "bubble"]
SPLITS = {
    "train": ROOT / "splits" / "train_local.txt",
    "val": ROOT / "splits" / "val_local.txt",
}


def main() -> int:
    api = KaggleApi()
    api.authenticate()

    files: dict[str, dict] = {}
    token = None
    while True:
        result = api.dataset_list_files(DATASET_REF, page_token=token)
        for item in result.files or []:
            files[item.name] = {"size": item.total_bytes}
        token = getattr(result, "next_page_token", None)
        print(f"[list] fetched {len(files)} files (token={bool(token)})", flush=True)
        if not token:
            break

    verdict: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_ref": DATASET_REF,
        "remote_file_count": len(files),
        "images_checked": 0,
        "size_mismatches": [],
        "missing_images": [],
        "missing_label_dirs": [],
        "label_size_mismatches": [],
        "label_sample_downloaded": 0,
        "label_sample_byte_mismatches": [],
        "data_yaml_match": None,
    }
    (ROOT / "remote_file_listing.json").write_text(
        json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    expected_labels = set()
    label_pairs: list[tuple[Path, str]] = []
    for split, list_path in SPLITS.items():
        for relative in list_path.read_text(encoding="utf-8").split():
            name = relative.split("/", 1)[1]
            remote_name = f"images/{name}"
            entry = files.get(remote_name)
            local_path = LOCAL_ROOT / relative
            if entry is None:
                verdict["missing_images"].append(remote_name)
                continue
            verdict["images_checked"] += 1
            if int(entry["size"]) != local_path.stat().st_size:
                verdict["size_mismatches"].append(
                    {"remote": remote_name, "remote_size": entry["size"], "local_size": local_path.stat().st_size}
                )
            label_local = LOCAL_ROOT / "labels" / split / f"{local_path.stem}.txt"
            remote_label = f"labels/{split}/{local_path.stem}.txt"
            expected_labels.add(remote_label)
            label_entry = files.get(remote_label)
            if label_entry is not None and int(label_entry["size"]) != label_local.stat().st_size:
                verdict["label_size_mismatches"].append(
                    {"remote": remote_label, "remote_size": label_entry["size"], "local_size": label_local.stat().st_size}
                )
            label_pairs.append((label_local, remote_label))

    for label_prefix in ("labels/train/", "labels/val/"):
        if not any(name.startswith(label_prefix) for name in files):
            verdict["missing_label_dirs"].append(label_prefix)

    # Byte-compare a fixed-seed sample of 40 labels across both splits.
    import random

    random.seed(0)
    download_dir = ROOT / "probe_downloads"
    download_dir.mkdir(exist_ok=True)
    for label_local, remote_label in random.sample(label_pairs, min(40, len(label_pairs))):
        try:
            api.dataset_download_file(DATASET_REF, remote_label, path=str(download_dir))
        except Exception as error:  # noqa: BLE001 - record any download failure
            verdict["label_sample_byte_mismatches"].append({"remote": remote_label, "error": f"download_failed:{error}"})
            continue
        downloaded = download_dir / remote_label.split("/", 1)[1]
        if not downloaded.is_file():
            verdict["label_sample_byte_mismatches"].append({"remote": remote_label, "error": "download_missing"})
            continue
        verdict["label_sample_downloaded"] += 1
        if downloaded.read_bytes() != label_local.read_bytes():
            verdict["label_sample_byte_mismatches"].append({"remote": remote_label, "error": "content_differs"})
        downloaded.unlink(missing_ok=True)

    label_missing = sorted(expected_labels - set(files))
    verdict["missing_labels"] = label_missing
    verdict["missing_labels_count"] = len(label_missing)

    remote_data_yaml = files.get("data.yaml")
    verdict["remote_data_yaml_present"] = remote_data_yaml is not None
    if remote_data_yaml is not None:
        # Semantic comparison only: the remote copy intentionally uses
        # ``path: .`` instead of the local absolute path.
        remote_yaml_text = (ROOT / "probe_downloads" / "data.yaml").read_text(encoding="utf-8")
        verdict["data_yaml_match"] = (
            "nc: 5" in remote_yaml_text
            and all(f"{index}: {name}" in remote_yaml_text for index, name in enumerate(EXPECTED_CLASSES))
        )

    verdict["verdict"] = (
        "REUSE_OK"
        if (
            not verdict["missing_images"]
            and not verdict["size_mismatches"]
            and not verdict["missing_label_dirs"]
            and not verdict["label_size_mismatches"]
            and not verdict["label_sample_byte_mismatches"]
            and not label_missing
            and verdict["data_yaml_match"]
        )
        else "REUSE_FAILED"
    )
    out = ROOT / "dianjinie_remote_reuse_probe.json"
    out.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in verdict.items() if not isinstance(v, list)}, ensure_ascii=False, indent=2))
    if verdict["missing_images"][:5]:
        print("missing_images sample:", verdict["missing_images"][:5])
    if verdict["size_mismatches"][:3]:
        print("size_mismatch sample:", verdict["size_mismatches"][:3])
    if label_missing[:5]:
        print("missing_labels sample:", label_missing[:5])
    print(f"[out] {out}")
    return 0 if verdict["verdict"] == "REUSE_OK" else 1


if __name__ == "__main__":
    sys.exit(main())
