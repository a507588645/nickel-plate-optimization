#!/usr/bin/env python3
"""Build the B3-clean split: B3 split minus every capture key seen in grade-test.

Leakage rule (decided 2026-09-05, option A authorized by user):
the LGBM grade model evaluates on the grade view's test split, so the B3
detector must never have trained/validated on the same physical captures.
A grade-test row counts as leaked when it hits the B3 990-image pool by
SHA-256, by stem (image_stem / image_name_raw), or by dup_group contagion
(same logic as analyze_b3_leakage_isolated_eval.py).  Every leaked row
contributes capture keys; any B3 train/val image whose own capture key is in
that set is excluded.

Capture key families (990 B3 stems: 853 B + 105 A + 32 OTHER):
- B family  ``^(\\d{17})_\\d+_\\d+$``          key = 17-digit timestamp
- A family  ``^(\\d{6})(bottom|top)_\\d+$``    key = 6-digit prefix
- OTHER     anything else (``20260212-2169``)  key = full stem
Keys are namespaced (``B:``/``A:``/``O:``) so families can never collide.

Only grade-test keys are excluded.  grade-train/valid rows also overlap B3
heavily, but LGBM trains on those rows, so detector-side overlap there does
not inflate LGBM *test* metrics; removing them would gut the detector
(764/774 train images covered).  The residual skew is recorded, not removed.

Expected reconstruction (probe numbers, asserted below):
grade-test leaked = 146 rows -> 145 keys; excluded train = 121 images
(2,506 boxes) -> 653 remain; excluded val = 38 -> 178 remain.

Outputs (under ``splits/``): train_local.txt, val_local.txt,
excluded_images.csv, split_summary.json.  Use --dry-run to audit only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
B3_TASK = ROOT.parent / "b3_dianjinie0211_y11m640_task_v1"
B3_SPLITS = B3_TASK / "splits"
B3_IMAGE_HASHES = B3_TASK / "kaggle_config_upload_v1" / "image_hashes.json"
VIEW_MANIFEST = Path(r"E:/PythonProject/archives/dianjinie_grade_classification_v1/manifests/samples.csv")
DEFAULT_DATASET_ROOT = Path(r"E:/PythonProject/datasets/Dianjinie0211")

RE_B = re.compile(r"^(\d{17})_\d+_\d+$")
RE_A = re.compile(r"^(\d{6})(?:bottom|top)_\d+$")

EXPECT = {
    "grade_test_total": 161,
    "grade_test_leaked": 146,
    "leaked_keys": 145,
    "train_before": 774,
    "val_before": 216,
    "train_excluded": 121,
    "val_excluded": 38,
    "train_box_excluded": 2506,
}


def capture_key(stem: str) -> str:
    match = RE_B.match(stem)
    if match:
        return f"B:{match.group(1)}"
    match = RE_A.match(stem)
    if match:
        return f"A:{match.group(1)}"
    return f"O:{stem}"


def read_split_list(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"划分清单为空：{path}")
    if len(set(lines)) != len(lines):
        raise ValueError(f"划分清单存在重复条目：{path}")
    return lines


def read_view_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1072:
        raise ValueError(f"等级视图 manifest 行数不是 1072: {len(rows)}")
    return rows


def classify_leakage(rows: list[dict[str, str]]) -> dict[str, dict[str, bool]]:
    """Same union rule as analyze_b3_leakage_isolated_eval.py: sha ∪ stem ∪ dup_group."""
    hashes = json.loads(B3_IMAGE_HASHES.read_text(encoding="utf-8"))
    sha_all = {value for value in hashes.values()}
    stem_all: set[str] = set()
    for name in ("train_local.txt", "val_local.txt"):
        stem_all |= {Path(line).stem for line in read_split_list(B3_SPLITS / name)}

    direct: dict[str, bool] = {}
    for row in rows:
        keys = {row["image_stem"].strip()}
        if row.get("image_name_raw"):
            keys.add(Path(row["image_name_raw"].strip()).stem)
        direct[row["image_name"]] = (row["sha256"].strip().lower() in sha_all) or bool(keys & stem_all)

    group_members: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group_members[row.get("dup_group_id", "") or row["image_name"]].append(row["image_name"])
    group_leaked = {gid: any(direct[name] for name in members) for gid, members in group_members.items()}

    out: dict[str, dict[str, bool]] = {}
    for row in rows:
        gid = row.get("dup_group_id", "") or row["image_name"]
        out[row["image_name"]] = {
            "direct": direct[row["image_name"]],
            "group": (not direct[row["image_name"]]) and group_leaked.get(gid, False),
            "leaked": direct[row["image_name"]] or group_leaked.get(gid, False),
        }
    return out


def count_boxes(dataset_root: Path, relatives: list[str], n_classes: int) -> tuple[int, Counter[str], dict[str, int]]:
    """Return (total_boxes, per-class-name Counter, per-relative box count)."""
    classes = json.loads((B3_SPLITS / "split_summary.json").read_text(encoding="utf-8"))["classes"]
    if len(classes) != n_classes:
        raise ValueError("classes 数与预期不符")
    per_class: Counter[str] = Counter()
    per_image: dict[str, int] = {}
    total = 0
    for relative in relatives:
        label_path = (dataset_root / relative.replace("images/", "labels/", 1)).with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"缺少标签文件：{label_path}")
        count = 0
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            per_class[classes[int(parts[0])]] += 1
            count += 1
        per_image[relative] = count
        total += count
    return total, per_class, per_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 B3-clean 划分：按采集键排除 grade-test 泄漏图")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "splits")
    parser.add_argument("--dry-run", action="store_true", help="只审计打印，不写文件")
    parser.add_argument("--no-expect-check", action="store_true", help="不校验 2026-09-05 探针复算数字")
    args = parser.parse_args(argv)

    dataset_root = args.dataset_root.resolve()
    train_before = read_split_list(B3_SPLITS / "train_local.txt")
    val_before = read_split_list(B3_SPLITS / "val_local.txt")
    summary_b3 = json.loads((B3_SPLITS / "split_summary.json").read_text(encoding="utf-8"))
    classes = summary_b3["classes"]

    families = Counter(capture_key(Path(line).stem).split(":", 1)[0] for line in [*train_before, *val_before])
    print(f"[families] 990 stems: B={families['B']} A={families['A']} O={families['O']}")

    rows = read_view_rows(VIEW_MANIFEST)
    leak = classify_leakage(rows)
    test_rows = [row for row in rows if row["split"] == "test"]
    leaked_test = [row for row in test_rows if leak[row["image_name"]]["leaked"]]
    print(f"[leak] grade-test total={len(test_rows)} leaked={len(leaked_test)} "
          f"(direct={sum(1 for r in leaked_test if leak[r['image_name']]['direct'])} "
          f"group_only={sum(1 for r in leaked_test if leak[r['image_name']]['group'])})")

    key_to_grade_rows: dict[str, list[str]] = defaultdict(list)
    for row in leaked_test:
        stems = {row["image_stem"].strip()}
        if row.get("image_name_raw"):
            stems.add(Path(row["image_name_raw"].strip()).stem)
        for stem in stems:
            key_to_grade_rows[capture_key(stem)].append(row["image_name"])
    leaked_keys = set(key_to_grade_rows)
    print(f"[keys] {len(leaked_test)} leaked rows -> {len(leaked_keys)} capture keys")

    excluded: list[dict[str, object]] = []
    train_after: list[str] = []
    val_after: list[str] = []
    for split_name, before, after in (("train", train_before, train_after), ("val", val_before, val_after)):
        for relative in before:
            key = capture_key(Path(relative).stem)
            if key in leaked_keys:
                excluded.append({
                    "relative": relative,
                    "split": split_name,
                    "key": key,
                    "grade_test_rows": sorted(set(key_to_grade_rows[key])),
                })
            else:
                after.append(relative)
    print(f"[exclude] train {len(train_before)} -> {len(train_after)} (-{len(train_before) - len(train_after)})")
    print(f"[exclude] val   {len(val_before)} -> {len(val_after)} (-{len(val_before) - len(val_after)})")

    n_classes = len(classes)
    train_box_before, train_cls_before, per_image_boxes = count_boxes(dataset_root, train_before, n_classes)
    val_box_before, val_cls_before, val_image_boxes = count_boxes(dataset_root, val_before, n_classes)
    per_image_boxes = {**per_image_boxes, **val_image_boxes}
    ex_train_cls: Counter[str] = Counter()
    ex_val_cls: Counter[str] = Counter()
    for item in excluded:
        n_boxes = per_image_boxes[str(item["relative"])]
        target = ex_train_cls if item["split"] == "train" else ex_val_cls
        label_path = (dataset_root / str(item["relative"]).replace("images/", "labels/", 1)).with_suffix(".txt")
        counted = 0
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                target[classes[int(parts[0])]] += 1
                counted += 1
        if counted != n_boxes:
            raise ValueError(f"框数前后不一致：{item['relative']}")
    train_box_after = train_box_before - sum(ex_train_cls.values())
    val_box_after = val_box_before - sum(ex_val_cls.values())
    train_cls_after = train_cls_before - ex_train_cls + Counter()
    val_cls_after = val_cls_before - ex_val_cls + Counter()
    print(f"[boxes] train {train_box_before} -> {train_box_after} (-{train_box_before - train_box_after})")
    print(f"[boxes] val   {val_box_before} -> {val_box_after} (-{val_box_before - val_box_after})")
    print(f"[boxes] train excluded per-class: {dict(ex_train_cls)}")
    print(f"[boxes] train after   per-class: {dict(train_cls_after)}")
    nodule_after = train_cls_after.get("nodule", 0)
    print(f"[nodule] train nodule boxes: {train_cls_before.get('nodule', 0)} -> {nodule_after}")
    if nodule_after == 0:
        print("[nodule][WARN] nodule 训练框被排光，该类将完全不可学", flush=True)

    checks = {
        "grade_test_total": len(test_rows),
        "grade_test_leaked": len(leaked_test),
        "leaked_keys": len(leaked_keys),
        "train_before": len(train_before),
        "val_before": len(val_before),
        "train_excluded": len(train_before) - len(train_after),
        "val_excluded": len(val_before) - len(val_after),
        "train_box_excluded": train_box_before - train_box_after,
    }
    print("[expect] " + json.dumps(checks, ensure_ascii=False))
    if not args.no_expect_check:
        mismatches = {k: (EXPECT[k], v) for k, v in checks.items() if EXPECT[k] != v}
        if mismatches:
            raise AssertionError(f"复算数字与 2026-09-05 探针不一致：{mismatches}")
        print("[expect] 全部与探针数字一致 ✓")

    if args.dry_run:
        print("[dry-run] 不写文件")
        return 0

    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"splits 目录已存在，拒绝覆盖（如需重建请人工删除）：{output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "train_local.txt").write_text("\n".join(train_after) + "\n", encoding="utf-8")
    (output_dir / "val_local.txt").write_text("\n".join(val_after) + "\n", encoding="utf-8")

    with (output_dir / "excluded_images.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative", "split", "capture_key", "n_boxes", "matched_grade_test_images"])
        for item in excluded:
            writer.writerow([
                item["relative"], item["split"], item["key"],
                per_image_boxes[str(item["relative"])],
                "|".join(item["grade_test_rows"]),
            ])

    summary = {
        "schema": "b3clean_split_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_split": {"schema": "b3_split_v1", "path": str(B3_SPLITS), "train": len(train_before), "val": len(val_before)},
        "dataset_root": str(dataset_root),
        "classes": classes,
        "exclusion_rule": (
            "排除采集键命中 grade-test 泄漏行的 B3 图；泄漏判定 = sha256 ∪ stem(image_stem/image_name_raw) "
            "∪ dup_group 传染（同 analyze_b3_leakage_isolated_eval.py）；只排 grade-test，"
            "grade-train/valid 重叠保留（LGBM 训练侧，见任务包 README）"
        ),
        "key_families": {"B": families["B"], "A": families["A"], "O": families["O"]},
        "grade_test": {
            "total": len(test_rows),
            "leaked": len(leaked_test),
            "clean": len(test_rows) - len(leaked_test),
            "leaked_capture_keys": len(leaked_keys),
            "leaked_rows": [
                {
                    "image_name": row["image_name"],
                    "grade": row["grade"],
                    "source_dataset": row.get("source_dataset", ""),
                    "keys": sorted({capture_key(row["image_stem"].strip())} | (
                        {capture_key(Path(row["image_name_raw"].strip()).stem)} if row.get("image_name_raw") else set()
                    )),
                }
                for row in leaked_test
            ],
        },
        "excluded": [
            {"relative": item["relative"], "split": item["split"], "capture_key": item["key"],
             "matched_grade_test_images": item["grade_test_rows"]}
            for item in excluded
        ],
        "train": {
            "before": len(train_before), "after": len(train_after),
            "box_count_before": train_box_before, "box_count_after": train_box_after,
            "class_distribution_before": dict(train_cls_before),
            "class_distribution_after": dict(train_cls_after),
        },
        "val": {
            "before": len(val_before), "after": len(val_after),
            "box_count_before": val_box_before, "box_count_after": val_box_after,
            "class_distribution_before": dict(val_cls_before),
            "class_distribution_after": dict(val_cls_after),
        },
        "expect_check": checks,
        "status": "unconfirmed_pending_enterprise",
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[splits] ready: {output_dir} (train={len(train_after)}, val={len(val_after)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
