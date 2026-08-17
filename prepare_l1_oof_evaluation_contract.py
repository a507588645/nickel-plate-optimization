#!/usr/bin/env python3
"""Freeze a reproducible, development-only L1 OOF label-quality review sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OOF_ROOT = ROOT / "l1_oof_consensus_20260802_232333_v1"
DEFAULT_DEV_ROOT = ROOT / "dianjienie_ryw18_oofconsensus_v1_dev_only_r1"
DEFAULT_OUTPUT_ROOT = ROOT / "l1_oof_evaluation_contract_v1"
FOCUS_ORDER = ("accepted_addition", "class_conflict", "supported_control")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a grouped, development-only review sample for the L1 OOF consensus view. "
            "It never reads the frozen test split and never writes dataset labels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--oof-root", type=Path, default=DEFAULT_OOF_ROOT)
    parser.add_argument("--dev-root", type=Path, default=DEFAULT_DEV_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-groups", type=int, default=200)
    parser.add_argument("--seed", default="20260808")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the selection plan only.")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")


def ensure_workspace_output(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Output must stay under the optimization workspace: {resolved}")
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    require_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_source_manifest(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    require_file(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Source manifest has no header")
        required = {"anchor_image", "source_image", "group_id"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"Source manifest is missing columns: {sorted(required - set(reader.fieldnames))}")
        rows = list(reader)

    by_image: dict[str, dict[str, str]] = {}
    group_members: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        image_name = Path(row["anchor_image"]).name
        group_id = row["group_id"].strip()
        if not image_name or not group_id:
            raise ValueError(f"Invalid source-manifest row: {row}")
        if image_name in by_image:
            raise ValueError(f"Duplicate development image in source manifest: {image_name}")
        by_image[image_name] = dict(row)
        group_members[group_id].append(image_name)

    for members in group_members.values():
        members.sort()
    return by_image, dict(group_members)


def read_decisions(path: Path, source_rows: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    require_file(path)
    decisions: list[dict[str, Any]] = []
    required = {
        "image",
        "fold",
        "class_id",
        "class_name",
        "xyxy_norm",
        "confidence",
        "status",
        "historical_relation",
        "max_historical_iou",
        "action",
        "evidence",
    }
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict) or not required.issubset(value):
                raise ValueError(f"Invalid decision at {path}:{line_number}")
            image_name = value["image"]
            if not isinstance(image_name, str) or image_name not in source_rows:
                raise ValueError(f"Decision references a non-development image at {path}:{line_number}")
            if value["fold"] not in (1, 2, 3):
                raise ValueError(f"Decision has an invalid OOF fold at {path}:{line_number}")
            if not isinstance(value["class_id"], int) or not isinstance(value["class_name"], str):
                raise ValueError(f"Decision has an invalid class at {path}:{line_number}")
            if not isinstance(value["xyxy_norm"], list) or len(value["xyxy_norm"]) != 4:
                raise ValueError(f"Decision has an invalid box at {path}:{line_number}")
            decisions.append(value)
    if not decisions:
        raise ValueError("No OOF decisions were found")
    return decisions


def focus_for_record(record: dict[str, Any]) -> str | None:
    if record["action"] == "add_to_training_candidate_view":
        return "accepted_addition"
    if record["action"] == "record_conflict_no_auto_reclass":
        return "class_conflict"
    if record["historical_relation"] == "existing_supported" and record["status"] == "accepted":
        return "supported_control"
    return None


def stable_rank(seed: str, focus: str, record: dict[str, Any]) -> str:
    payload = {
        "seed": seed,
        "focus": focus,
        "image": record["image"],
        "class_id": record["class_id"],
        "xyxy_norm": record["xyxy_norm"],
        "confidence": record["confidence"],
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def select_focus(
    candidates: list[dict[str, Any]],
    budget: int,
    excluded_groups: set[str],
    seed: str,
    focus: str,
) -> list[dict[str, Any]]:
    if budget <= 0:
        return []

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate["group_id"] not in excluded_groups:
            by_class[candidate["record"]["class_name"]].append(candidate)
    for class_name, items in by_class.items():
        items.sort(key=lambda item: (stable_rank(seed, focus, item["record"]), item["group_id"], item["record"]["image"]))

    selected: list[dict[str, Any]] = []
    positions = {class_name: 0 for class_name in by_class}
    while len(selected) < budget:
        progressed = False
        for class_name in sorted(by_class):
            items = by_class[class_name]
            while positions[class_name] < len(items) and items[positions[class_name]]["group_id"] in excluded_groups:
                positions[class_name] += 1
            if positions[class_name] >= len(items):
                continue
            candidate = items[positions[class_name]]
            positions[class_name] += 1
            excluded_groups.add(candidate["group_id"])
            selected.append(candidate)
            progressed = True
            if len(selected) >= budget:
                break
        if not progressed:
            break
    return selected


def allocate_budgets(target_groups: int) -> dict[str, int]:
    if target_groups < 3:
        raise ValueError("--target-groups must be at least 3")
    addition = target_groups // 2
    conflict = target_groups // 4
    control = target_groups - addition - conflict
    return {
        "accepted_addition": addition,
        "class_conflict": conflict,
        "supported_control": control,
    }


def build_selection(
    decisions: list[dict[str, Any]],
    source_rows: dict[str, dict[str, str]],
    group_members: dict[str, list[str]],
    seed: str,
    target_groups: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    candidates_by_focus: dict[str, list[dict[str, Any]]] = {focus: [] for focus in FOCUS_ORDER}
    records_by_group_focus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in decisions:
        focus = focus_for_record(record)
        if focus is None:
            continue
        group_id = source_rows[record["image"]]["group_id"]
        candidate = {"focus": focus, "group_id": group_id, "record": record}
        candidates_by_focus[focus].append(candidate)
        records_by_group_focus[f"{group_id}\u0000{focus}"].append(record)

    excluded_groups: set[str] = set()
    selected: list[dict[str, Any]] = []
    budgets = allocate_budgets(target_groups)
    for focus in FOCUS_ORDER:
        selected.extend(select_focus(candidates_by_focus[focus], budgets[focus], excluded_groups, seed, focus))

    if len(selected) < target_groups:
        remaining = target_groups - len(selected)
        for focus in FOCUS_ORDER:
            if remaining <= 0:
                break
            extra = select_focus(candidates_by_focus[focus], remaining, excluded_groups, seed, focus)
            selected.extend(extra)
            remaining -= len(extra)

    selected.sort(
        key=lambda item: (
            FOCUS_ORDER.index(item["focus"]),
            item["group_id"],
            item["record"]["image"],
            item["record"]["class_id"],
        )
    )
    return selected, candidates_by_focus, records_by_group_focus


def record_to_unit(
    review_id: str,
    focus: str,
    record: dict[str, Any],
    group_id: str,
    source_rows: dict[str, dict[str, str]],
    group_members: dict[str, list[str]],
) -> dict[str, Any]:
    source = source_rows[record["image"]]
    return {
        "review_id": review_id,
        "review_focus": focus,
        "group_id": group_id,
        "group_member_images": group_members[group_id],
        "image": record["image"],
        "image_path": source["anchor_image"],
        "source_image": source["source_image"],
        "fold": record["fold"],
        "class_id": record["class_id"],
        "class_name": record["class_name"],
        "candidate_xyxy_norm": record["xyxy_norm"],
        "candidate_confidence": record["confidence"],
        "candidate_status": record["status"],
        "historical_relation": record["historical_relation"],
        "historical_overlap_iou": record["max_historical_iou"],
        "suggested_action": record["action"],
        "evidence": record["evidence"],
        "human_decision": "",
        "reviewer_id": "",
        "reviewed_at": "",
        "notes": "",
    }


def build_review_units(
    selected: list[dict[str, Any]],
    records_by_group_focus: dict[str, list[dict[str, Any]]],
    source_rows: dict[str, dict[str, str]],
    group_members: dict[str, list[str]],
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    review_number = 1
    for item in selected:
        focus = item["focus"]
        group_id = item["group_id"]
        group_records = records_by_group_focus[f"{group_id}\u0000{focus}"]
        if focus == "supported_control":
            group_records = [item["record"]]
        else:
            group_records = sorted(
                group_records,
                key=lambda record: (record["image"], record["class_id"], record["xyxy_norm"], record["confidence"]),
            )
        for record in group_records:
            review_id = f"L1OOF-R{review_number:04d}"
            units.append(record_to_unit(review_id, focus, record, group_id, source_rows, group_members))
            review_number += 1
    return units


def count_by_focus(items: list[dict[str, Any]], field: str = "review_focus") -> dict[str, int]:
    return dict(sorted(Counter(str(item[field]) for item in items).items()))


def count_selected_classes(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(item["class_name"]) for item in items).items()))


def format_box(box: list[float]) -> str:
    return " ".join(f"{float(value):.8f}" for value in box)


def write_csv(path: Path, units: list[dict[str, Any]], blind: bool) -> None:
    common = [
        "review_id",
        "group_id",
        "group_member_images",
        "image",
        "image_path",
        "source_image",
        "fold",
        "class_id",
        "class_name",
        "candidate_xyxy_norm",
    ]
    focus_column = ["review_focus"] if not blind else []
    audit_only = [
        "candidate_confidence",
        "candidate_status",
        "historical_relation",
        "historical_overlap_iou",
        "suggested_action",
    ]
    reviewer_columns = ["human_decision", "reviewer_id", "reviewed_at", "notes"]
    fields = common + focus_column + ([] if blind else audit_only) + reviewer_columns
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for unit in units:
            row = {
                "review_id": unit["review_id"],
                "group_id": unit["group_id"],
                "group_member_images": ";".join(unit["group_member_images"]),
                "image": unit["image"],
                "image_path": unit["image_path"],
                "source_image": unit["source_image"],
                "fold": unit["fold"],
                "class_id": unit["class_id"],
                "class_name": unit["class_name"],
                "candidate_xyxy_norm": format_box(unit["candidate_xyxy_norm"]),
                "human_decision": "",
                "reviewer_id": "",
                "reviewed_at": "",
                "notes": "",
            }
            if not blind:
                row.update(
                    {
                        "review_focus": unit["review_focus"],
                        "candidate_confidence": f"{float(unit['candidate_confidence']):.8f}",
                        "candidate_status": unit["candidate_status"],
                        "historical_relation": unit["historical_relation"],
                        "historical_overlap_iou": f"{float(unit['historical_overlap_iou']):.8f}",
                        "suggested_action": unit["suggested_action"],
                    }
                )
            writer.writerow(row)


def render_protocol(manifest: dict[str, Any]) -> str:
    return f"""# L1 OOF 固定人工评测契约 v1

## 用途与边界

- 目的：在固定的开发集样本上核验 OOF 自动共识候选的新增框、类别冲突与正常对照，避免仅用候选五列标签自身的验证指标作结论。
- 数据范围：仅 `dianjienie_ryw18_oofconsensus_v1_dev_only_r1` 的开发集。冻结的 198 张 `release-validation` 候选不在本目录、不读取、不推理、不作为本次抽样来源。
- 这是一份审查任务契约和空白记录表，不是人工真值，也不会修改原始标签、候选标签或数据集。
- 本次冻结了 {manifest['counts']['selected_groups']} 个物理分组、{manifest['counts']['review_units']} 个框级审查单元。抽样种子为 `{manifest['selection']['seed']}`。

## 审查顺序

1. 先使用 `review_queue_blind.csv` 审查，避免看到模型置信度、历史标签关系或自动动作。
2. 每一行只判断该候选框：是否为真实缺陷、类别是否正确、框是否可用。填写 `human_decision`、`reviewer_id`、`reviewed_at` 和 `notes`。
3. 合法的 `human_decision`：`keep`、`delete_false_positive`、`change_class`、`box_adjust_needed`、`uncertain`、`not_reviewable`。
4. 完成首轮后，才可用 `review_queue_audit.csv` / `review_units_audit.jsonl` 查看置信度、历史关系和双模型证据，核对是否存在系统性偏差。
5. 审查结果必须导出到新的版本化结果目录；不得覆盖本契约、`labels_with_conf/`、历史人工标签或 OOF 候选五列标签。

## 通过规则

- `accepted_addition` 是主判定样本：重点统计真实新增框比例、错误类别比例与不可判比例。
- `class_conflict` 是风险样本：禁止依据本次审查把历史标签静默改类；任何改类建议都应独立记录并等待业务确认。
- `supported_control` 只用于检查双模型一致且历史支持的情形，不能替代新增框的人工判断。
- 该样本不含所有稀有类或所有候选框。若关键类别缺少可审实例，必须把该类标记为“不可评”，不能用总体比例代替逐类结论。
- 在未填写审查结果前，不得把本目录当作“已完成人工评测”的证据，也不得据此选择最终模型或最终训练视图。

## 可追溯性

- 源 OOF `summary.json` SHA-256：`{manifest['sources']['oof_summary_sha256']}`
- 源 OOF `decisions.jsonl` SHA-256：`{manifest['sources']['oof_decisions_sha256']}`
- 候选开发集 `view_manifest.json` SHA-256：`{manifest['sources']['view_manifest_sha256']}`
- 开发集 `source_manifest.csv` SHA-256：`{manifest['sources']['source_manifest_sha256']}`
"""


def write_outputs(output_root: Path, units: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    staging_root = output_root.parent / f".{output_root.name}.building_{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_root}")
    staging_root.mkdir(parents=True)
    try:
        (staging_root / "selection_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (staging_root / "review_units_audit.jsonl").open("w", encoding="utf-8") as handle:
            for unit in units:
                handle.write(json.dumps(unit, ensure_ascii=False) + "\n")
        write_csv(staging_root / "review_queue_blind.csv", units, blind=True)
        write_csv(staging_root / "review_queue_audit.csv", units, blind=False)
        (staging_root / "review_protocol.md").write_text(render_protocol(manifest), encoding="utf-8")
        staging_root.replace(output_root)
    except Exception:
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.target_groups < 3:
        raise ValueError("--target-groups must be at least 3")

    oof_root = args.oof_root.resolve()
    dev_root = args.dev_root.resolve()
    output_root = ensure_workspace_output(args.output_root)
    summary_path = oof_root / "summary.json"
    decisions_path = oof_root / "decisions.jsonl"
    view_manifest_path = dev_root / "view_manifest.json"
    source_manifest_path = dev_root / "manifests" / "source_manifest.csv"
    summary = read_json(summary_path)
    view_manifest = read_json(view_manifest_path)
    if view_manifest.get("mode") != "development_only":
        raise ValueError("The supplied view is not development-only")
    if view_manifest.get("frozen_test_access") != "not read; not linked; not included":
        raise ValueError("The supplied view does not have the expected frozen-test boundary")
    if view_manifest.get("source_oof_summary_sha256") != sha256_file(summary_path):
        raise ValueError("OOF summary hash does not match the development view manifest")
    if view_manifest.get("source_oof_decisions_sha256") != sha256_file(decisions_path):
        raise ValueError("OOF decision hash does not match the development view manifest")

    source_rows, group_members = read_source_manifest(source_manifest_path)
    decisions = read_decisions(decisions_path, source_rows)
    expected_images = int(summary.get("images", -1))
    if len(source_rows) != expected_images:
        raise ValueError(f"Development image count mismatch: manifest={len(source_rows)}, summary={expected_images}")
    expected_classes = summary.get("class_names")
    if not isinstance(expected_classes, list) or not expected_classes:
        raise ValueError("OOF summary does not contain class names")
    if view_manifest.get("classes") != expected_classes:
        raise ValueError("Development-view class order differs from OOF summary")

    selected, candidates_by_focus, records_by_group_focus = build_selection(
        decisions,
        source_rows,
        group_members,
        args.seed,
        args.target_groups,
    )
    units = build_review_units(selected, records_by_group_focus, source_rows, group_members)
    manifest = {
        "schema_version": 1,
        "purpose": "L1 OOF consensus development-only label-quality review contract",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection": {
            "seed": args.seed,
            "target_groups": args.target_groups,
            "group_deduplicated": True,
            "focus_order": list(FOCUS_ORDER),
            "budgets": allocate_budgets(args.target_groups),
            "method": "category then round-robin class selection with deterministic SHA-256 ranking",
        },
        "boundaries": {
            "data_scope": "development_only",
            "frozen_test_access": "not read; not linked; not included",
            "label_mutation": "none",
            "model_selection": "prohibited until human_decision fields are completed and analyzed",
        },
        "sources": {
            "oof_root": str(oof_root),
            "development_view_root": str(dev_root),
            "oof_summary_sha256": sha256_file(summary_path),
            "oof_decisions_sha256": sha256_file(decisions_path),
            "view_manifest_sha256": sha256_file(view_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
        },
        "counts": {
            "development_images": len(source_rows),
            "development_groups": len(group_members),
            "all_oof_decisions": len(decisions),
            "candidate_groups_by_focus": {
                focus: len({item['group_id'] for item in candidates_by_focus[focus]}) for focus in FOCUS_ORDER
            },
            "selected_groups": len(selected),
            "review_units": len(units),
            "selected_groups_by_focus": count_by_focus(selected, field="focus"),
            "review_units_by_focus": count_by_focus(units),
            "review_units_by_class": count_selected_classes(units),
        },
        "review_unit_ids": [unit["review_id"] for unit in units],
    }
    display_manifest = {key: value for key, value in manifest.items() if key != "review_unit_ids"}
    display_manifest["review_unit_id_count"] = len(units)
    print(json.dumps(display_manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    write_outputs(output_root, units, manifest)
    print(f"[created] {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
