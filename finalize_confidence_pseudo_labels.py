#!/usr/bin/env python3
"""验证六列伪标签，并生成 accepted/uncertain/rejected 三状态清单。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="验证置信度伪标签并生成三状态 JSONL")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--accepted-threshold", type=float, default=0.70)
    parser.add_argument("--uncertain-threshold", type=float, default=0.40)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    label_root = root / "labels_with_conf"
    states_root = root / "confidence_states"
    states_root.mkdir(parents=True, exist_ok=True)
    handles = {
        state: (states_root / f"{state}.jsonl").open("w", encoding="utf-8")
        for state in ("accepted", "uncertain", "rejected")
    }
    counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[int]] = {state: Counter() for state in handles}
    invalid = []
    try:
        labels = sorted(path for path in label_root.glob("*.txt") if path.stem != "classes")
        for label in labels:
            for index, line in enumerate(label.read_text(encoding="utf-8").splitlines()):
                fields = line.split()
                if len(fields) != 6:
                    if line.strip():
                        invalid.append({"label": str(label), "line": index + 1, "reason": f"fields={len(fields)}"})
                    continue
                class_id = int(fields[0])
                xywh = [float(v) for v in fields[1:5]]
                confidence = float(fields[5])
                state = (
                    "accepted" if confidence >= args.accepted_threshold
                    else "uncertain" if confidence >= args.uncertain_threshold
                    else "rejected"
                )
                record = {
                    "image_stem": label.stem,
                    "box_index": index,
                    "class_id": class_id,
                    "xywh": xywh,
                    "confidence": confidence,
                    "default_status": state,
                    "teacher_model": "ryw.pt",
                }
                handles[state].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[state] += 1
                class_counts[state][class_id] += 1
    finally:
        for handle in handles.values():
            handle.close()
    image_count = sum(
        1 for path in (root / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    label_count = len(labels)
    summary = {
        "teacher_model": "ryw.pt",
        "candidate_threshold": 0.15,
        "accepted_threshold": args.accepted_threshold,
        "uncertain_threshold": args.uncertain_threshold,
        "images": image_count,
        "labels_with_conf": label_count,
        "state_box_counts": dict(counts),
        "state_class_counts": {state: dict(counter) for state, counter in class_counts.items()},
        "invalid_entries": invalid,
    }
    (states_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path = root / "dataset_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata_path.exists()
        else {"dataset_name": root.name, "dataset_root": str(root)}
    )
    metadata.update({
        "confidence_labels": "labels_with_conf",
        "confidence_candidate_threshold": 0.15,
        "confidence_states": {"accepted": ">=0.70", "uncertain": "0.40-0.70", "rejected": "0.15-0.40"},
    })
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
