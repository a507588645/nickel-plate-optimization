#!/usr/bin/env python3
"""
电积镍等级判定规则引擎（配置驱动，第一版骨架）。

设计对齐 `dianjie_grade_rule.py`：
- dataclass 结果
- 画布/并集面积特征
- 有序分支判级

与电解镍的关键差异：
- 阈值**不**硬编码进类构造默认值，一律从外部 JSON 规则表加载；
- 规则表默认标记为 `unconfirmed_pending_enterprise`（未经企业确认）；
- 无规则命中时输出 `manual_review`；
- 不为 9996 / bad 编造规则（无样本）。

18 类缺陷体系（YOLO id → 名称）与 A/B/C 分组见 `DEFAULT_CLASS_NAMES` / `DEFAULT_GROUPS`。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 18 类与分组（初始假设，与电解镍 A/B/C 同构；仅作特征工程，非企业确认语义）
# ---------------------------------------------------------------------------

DEFAULT_CLASS_NAMES: list[str] = [
    "wuzi",
    "waterstain",
    "yellowstain",
    "nodule",
    "airhole",
    "bubble",
    "bump",
    "fawu",
    "overfire",
    "NiOH",
    "greenstain",
    "suanwuran",
    "quejiao",
    "watermark",
    "youpao",
    "heibian",
    "mijixiaokeng",
    "fanxitong",
]

# A: 水印/污渍  B: 黄渍/气孔/结粒/疙瘩/绿色结晶/缺角/严重水渍  C: 发乌/烧板/酸污染
DEFAULT_GROUPS: dict[str, list[int]] = {
    "A": [0, 1],
    "B": [2, 4, 5, 6, 10, 12, 13],
    "C": [7, 8, 11],
    "AIRHOLE": [4],
    "NODULE": [3],
}

VALID_GRADES = frozenset({"3#", "9950", "9996", "bad", "manual_review"})
VALID_OPS = frozenset({"<", "<=", ">", ">=", "=="})

# 电解镍原项目坐标系（用于 A/B/C 画布近似，与 dianjie_grade_rule 一致）
CANVAS_H = 4069
CANVAS_W = 3554
NODULE_AREA_THRESHOLD = 13000.0
NODULE_ASPECT_THRESHOLD = 3.0

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "dianji_grade_rule_v1" / "candidate_rules.json"


@dataclass
class DianjiGradeResult:
    """单张图判级结果。"""

    grade: str
    hit_rule: str
    features: dict[str, Any] = field(default_factory=dict)
    status: str = "unconfirmed_pending_enterprise"
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compare(value: float, op: str, threshold: float) -> bool:
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "==":
        return float(value) == float(threshold)
    raise ValueError(f"不支持的比较符: {op}")


def _to_box_xyxy_norm(det: dict[str, Any]) -> tuple[float, float, float, float]:
    """中心宽高（归一化 0-1 或像素）→ xyxy。调用方保证单位一致。"""
    cx = float(det["cx"])
    cy = float(det["cy"])
    w = float(det["w"])
    h = float(det["h"])
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _union_area_normalized(rects: list[tuple[float, float, float, float]]) -> float:
    """归一化坐标系下的矩形并集面积（扫描线）。"""
    cleaned = [(x1, y1, x2, y2) for x1, y1, x2, y2 in rects if x2 > x1 and y2 > y1]
    if not cleaned:
        return 0.0
    xs = sorted({v for r in cleaned for v in (r[0], r[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted((y1, y2) for x1, y1, x2, y2 in cleaned if x1 < right and x2 > left)
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for ns, ne in intervals[1:]:
            if ns > end:
                covered += end - start
                start, end = ns, ne
            else:
                end = max(end, ne)
        covered += end - start
        total += (right - left) * covered
    return float(total)


def _scalculate_canvas(array: list[list[int]]) -> int:
    """缺陷像素面积（去重），与电解镍原项目画布近似一致。"""
    if not array:
        return 0
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    for x1, y1, x2, y2 in array:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(CANVAS_H, x2), min(CANVAS_W, y2)
        if x2 > x1 and y2 > y1:
            canvas[x1:x2, y1:y2] = 1
    return int(canvas.sum())


class DianjiGradeRule:
    """
    电积镍配置驱动规则引擎。

    规则表为有序列表：首条全部条件满足即返回对应 grade；
    无命中 → manual_review。
    """

    def __init__(
        self,
        rules_path: Path | str | None = None,
        rules_data: dict[str, Any] | None = None,
        class_names: list[str] | None = None,
        groups: dict[str, list[int]] | None = None,
    ):
        if rules_data is None:
            path = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
            if not path.exists():
                # 允许无规则表时仅做特征提取 / dry-run
                rules_data = {
                    "status": "unconfirmed_pending_enterprise",
                    "generated_at": "",
                    "sample_count": 0,
                    "rules": [],
                    "class_names": class_names or DEFAULT_CLASS_NAMES,
                    "groups": groups or DEFAULT_GROUPS,
                    "note": "规则表不存在，仅可做特征提取",
                }
            else:
                rules_data = json.loads(path.read_text(encoding="utf-8-sig"))

        self.rules_data = rules_data
        self.status = str(rules_data.get("status", "unconfirmed_pending_enterprise"))
        self.class_names = list(rules_data.get("class_names") or class_names or DEFAULT_CLASS_NAMES)
        raw_groups = rules_data.get("groups") or groups or DEFAULT_GROUPS
        self.groups: dict[str, set[int]] = {
            k: set(int(x) for x in v) for k, v in raw_groups.items()
        }
        self.name_to_id = {n: i for i, n in enumerate(self.class_names)}
        self.rules = list(rules_data.get("rules") or [])
        self._validate_rules()

    def _validate_rules(self) -> None:
        for i, rule in enumerate(self.rules):
            if "name" not in rule:
                raise ValueError(f"规则[{i}] 缺少 name")
            grade = rule.get("grade")
            if grade not in VALID_GRADES:
                raise ValueError(f"规则[{i}] grade 非法: {grade}")
            for j, cond in enumerate(rule.get("conditions") or []):
                op = cond.get("op")
                if op not in VALID_OPS:
                    raise ValueError(f"规则[{i}].conditions[{j}] op 非法: {op}")
                if "feature" not in cond or "threshold" not in cond:
                    raise ValueError(f"规则[{i}].conditions[{j}] 缺少 feature/threshold")

    # ------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------

    def compute_features_from_detections(
        self,
        detections: list[dict[str, Any]],
        plate_area: float,
        image_width: float = 1024.0,
        image_height: float = 888.0,
        coords: str = "pixel",
    ) -> dict[str, Any]:
        """
        从检测框计算规则可用特征。

        Args:
            detections: 每项含 label/class_id, cx, cy, w, h；可选 confidence。
            plate_area: 板面面积（与电解镍一致，常用 1024×888 下 findcent 结果）。
            image_width / image_height: 图像尺寸。
            coords: "pixel" 表示 cx/cy/w/h 为像素；"normalized" 表示 0-1。
        """
        dets: list[dict[str, Any]] = []
        for d in detections:
            label = int(d.get("label", d.get("class_id", -1)))
            cx, cy, w, h = float(d["cx"]), float(d["cy"]), float(d["w"]), float(d["h"])
            if coords == "normalized":
                cx_p, cy_p, w_p, h_p = cx * image_width, cy * image_height, w * image_width, h * image_height
                cx_n, cy_n, w_n, h_n = cx, cy, w, h
            else:
                cx_p, cy_p, w_p, h_p = cx, cy, w, h
                cx_n = cx / image_width if image_width else 0.0
                cy_n = cy / image_height if image_height else 0.0
                w_n = w / image_width if image_width else 0.0
                h_n = h / image_height if image_height else 0.0
            conf = float(d.get("confidence", d.get("conf", 1.0)))
            dets.append(
                {
                    "label": label,
                    "cx": cx_p,
                    "cy": cy_p,
                    "w": w_p,
                    "h": h_p,
                    "cx_n": cx_n,
                    "cy_n": cy_n,
                    "w_n": w_n,
                    "h_n": h_n,
                    "confidence": conf,
                }
            )

        has_nodule = False
        filtered = []
        for det in dets:
            if det["label"] == 3:  # nodule
                aspect = max(det["h"] / det["w"], det["w"] / det["h"]) if min(det["w"], det["h"]) > 0 else 1.0
                area = det["w"] * det["h"]
                if area < NODULE_AREA_THRESHOLD and aspect < NODULE_ASPECT_THRESHOLD:
                    continue
                has_nodule = True
            filtered.append(det)

        # 逐类 count / 归一化并集面积比
        features: dict[str, Any] = {}
        all_norm_rects: list[tuple[float, float, float, float]] = []
        for class_id, name in enumerate(self.class_names):
            class_dets = [d for d in filtered if d["label"] == class_id]
            features[f"count__{name}"] = len(class_dets)
            rects = []
            for d in class_dets:
                x1 = max(0.0, d["cx_n"] - d["w_n"] / 2)
                y1 = max(0.0, d["cy_n"] - d["h_n"] / 2)
                x2 = min(1.0, d["cx_n"] + d["w_n"] / 2)
                y2 = min(1.0, d["cy_n"] + d["h_n"] / 2)
                rects.append((x1, y1, x2, y2))
                all_norm_rects.append((x1, y1, x2, y2))
            features[f"area_ratio__{name}"] = _union_area_normalized(rects)

        # A/B/C/Airhole：像素 /4 画布并集 / plate_area（对齐电解镍）
        def group_canvas_ratio(labels: set[int]) -> float:
            boxes: list[list[int]] = []
            for d in filtered:
                if d["label"] not in labels:
                    continue
                x1 = int((d["cx"] - d["w"] / 2) / 4)
                y1 = int((d["cy"] - d["h"] / 2) / 4)
                x2 = int((d["cx"] + d["w"] / 2) / 4)
                y2 = int((d["cy"] + d["h"] / 2) / 4)
                boxes.append([x1, y1, x2, y2])
            area = _scalculate_canvas(boxes)
            return area / plate_area if plate_area > 0 else 0.0

        features["A_ratio"] = group_canvas_ratio(self.groups.get("A", set()))
        features["B_ratio"] = group_canvas_ratio(self.groups.get("B", set()))
        features["C_ratio"] = group_canvas_ratio(self.groups.get("C", set()))
        features["Airhole_ratio"] = group_canvas_ratio(self.groups.get("AIRHOLE", set()))
        features["has_nodule"] = int(has_nodule)
        features["detection_count"] = len(filtered)
        features["max_confidence"] = max((d["confidence"] for d in filtered), default=0.0)
        features["total_bbox_union_image_ratio"] = _union_area_normalized(all_norm_rects)
        features["plate_area"] = float(plate_area)
        features["image_width"] = float(image_width)
        features["image_height"] = float(image_height)
        return features

    def features_from_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """从特征 CSV 行构造特征 dict（字符串数值转 float/int）。"""
        features: dict[str, Any] = {}
        for name in self.class_names:
            ck, ak = f"count__{name}", f"area_ratio__{name}"
            if ck in row and row[ck] != "":
                features[ck] = int(float(row[ck]))
            else:
                features[ck] = 0
            if ak in row and row[ak] != "":
                features[ak] = float(row[ak])
            else:
                features[ak] = 0.0
        for key in (
            "A_ratio",
            "B_ratio",
            "C_ratio",
            "Airhole_ratio",
            "total_bbox_union_image_ratio",
            "max_confidence",
            "plate_area",
            "image_width",
            "image_height",
        ):
            if key in row and row[key] != "":
                features[key] = float(row[key])
            else:
                features[key] = 0.0
        if "detection_count" in row and row["detection_count"] != "":
            features["detection_count"] = int(float(row["detection_count"]))
        else:
            features["detection_count"] = 0
        if "has_nodule" in row and row["has_nodule"] != "":
            val = row["has_nodule"]
            if isinstance(val, str):
                features["has_nodule"] = 1 if val.strip().lower() in {"1", "true", "yes"} else int(float(val)) if val.replace(".", "", 1).isdigit() else 0
            else:
                features["has_nodule"] = int(bool(val))
        else:
            features["has_nodule"] = 0
        return features

    # ------------------------------------------------------------------
    # 判级
    # ------------------------------------------------------------------

    def evaluate_rule(self, rule: dict[str, Any], features: dict[str, Any]) -> bool:
        conditions = rule.get("conditions") or []
        if not conditions:
            return False
        for cond in conditions:
            feat = cond["feature"]
            if feat not in features:
                return False
            value = features[feat]
            if isinstance(value, bool):
                value = int(value)
            if not _compare(float(value), cond["op"], float(cond["threshold"])):
                return False
        return True

    def classify_features(self, features: dict[str, Any]) -> DianjiGradeResult:
        """对已算好的特征 dict 执行有序规则。"""
        for rule in self.rules:
            if self.evaluate_rule(rule, features):
                return DianjiGradeResult(
                    grade=str(rule["grade"]),
                    hit_rule=str(rule["name"]),
                    features=features,
                    status=self.status,
                    message="命中候选规则（未经企业确认）",
                )
        return DianjiGradeResult(
            grade="manual_review",
            hit_rule="",
            features=features,
            status=self.status,
            message="无规则命中，需人工复核",
        )

    def classify(
        self,
        detections: list[dict[str, Any]] | None = None,
        plate_area: float = 0.0,
        features: dict[str, Any] | None = None,
        image_width: float = 1024.0,
        image_height: float = 888.0,
        coords: str = "pixel",
        dry_run: bool = False,
    ) -> DianjiGradeResult:
        """
        判级入口。

        优先使用 features；否则从 detections 计算。
        dry_run=True 时只返回特征，grade=manual_review，hit_rule=dry_run。
        """
        if features is None:
            if detections is None:
                detections = []
            features = self.compute_features_from_detections(
                detections,
                plate_area=plate_area,
                image_width=image_width,
                image_height=image_height,
                coords=coords,
            )
        if dry_run:
            return DianjiGradeResult(
                grade="manual_review",
                hit_rule="dry_run",
                features=features,
                status=self.status,
                message="dry-run：仅输出特征，不判级",
            )
        return self.classify_features(features)


def load_yolo_labels(
    label_path: Path,
    image_width: float = 1024.0,
    image_height: float = 888.0,
    conf_thres: float = 0.0,
) -> list[dict[str, Any]]:
    """读取 YOLO 标签（5 列或 6 列 conf），输出像素坐标检测框。"""
    if not label_path.exists():
        return []
    dets: list[dict[str, Any]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) not in (5, 6):
            continue
        cls = int(float(fields[0]))
        cx_n, cy_n, w_n, h_n = map(float, fields[1:5])
        conf = float(fields[5]) if len(fields) == 6 else 1.0
        if conf < conf_thres:
            continue
        dets.append(
            {
                "label": cls,
                "cx": cx_n * image_width,
                "cy": cy_n * image_height,
                "w": w_n * image_width,
                "h": h_n * image_height,
                "confidence": conf,
            }
        )
    return dets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="电积镍 18 类缺陷种类+面积 配置驱动规则判级引擎（候选规则未经企业确认）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=DEFAULT_RULES_PATH,
        help="外部 JSON 规则表路径",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="特征 CSV，批量判级（需含 count__/area_ratio__ 等列）",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="YOLO 标签目录（*.txt），批量从框计算特征并判级",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="批量结果 CSV 输出路径",
    )
    parser.add_argument(
        "--plate-area",
        type=float,
        default=1024.0 * 888.0,
        help="标签目录批量时的默认板面面积",
    )
    parser.add_argument("--image-width", type=float, default=1024.0)
    parser.add_argument("--image-height", type=float, default=888.0)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出特征，不执行规则判级",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="运行内置边界自测后退出",
    )
    return parser.parse_args(argv)


def run_self_test() -> int:
    """最小边界自测：空检测 → manual_review；合成命中首条规则。"""
    print("=== dianji_grade_rule 自测 ===")
    # 空规则表
    engine_empty = DianjiGradeRule(
        rules_data={
            "status": "unconfirmed_pending_enterprise",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "sample_count": 0,
            "class_names": DEFAULT_CLASS_NAMES,
            "groups": DEFAULT_GROUPS,
            "rules": [],
        }
    )
    r0 = engine_empty.classify(detections=[], plate_area=100000)
    assert r0.grade == "manual_review", r0
    assert r0.hit_rule == "", r0
    print("[OK] 空检测 + 空规则 → manual_review")

    # 合成规则：B_ratio > 0.1 → 3#
    rules = {
        "status": "unconfirmed_pending_enterprise",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_count": 0,
        "class_names": DEFAULT_CLASS_NAMES,
        "groups": DEFAULT_GROUPS,
        "rules": [
            {
                "name": "synthetic_b_high_to_3sharp",
                "grade": "3#",
                "conditions": [{"feature": "B_ratio", "op": ">", "threshold": 0.1}],
            },
            {
                "name": "synthetic_low_defect_to_9950",
                "grade": "9950",
                "conditions": [
                    {"feature": "B_ratio", "op": "<=", "threshold": 0.1},
                    {"feature": "C_ratio", "op": "<=", "threshold": 0.05},
                ],
            },
        ],
    }
    engine = DianjiGradeRule(rules_data=rules)

    # 用特征 dict 强制命中首条
    feat_hit = engine.compute_features_from_detections([], plate_area=100000)
    feat_hit["B_ratio"] = 0.5
    r1 = engine.classify_features(feat_hit)
    assert r1.grade == "3#", r1
    assert r1.hit_rule == "synthetic_b_high_to_3sharp", r1
    print("[OK] 合成特征 B_ratio=0.5 → 命中首条 3#")

    feat_low = engine.compute_features_from_detections([], plate_area=100000)
    feat_low["B_ratio"] = 0.01
    feat_low["C_ratio"] = 0.01
    r2 = engine.classify_features(feat_low)
    assert r2.grade == "9950", r2
    print("[OK] 合成特征低缺陷 → 命中 9950")

    r3 = engine.classify(detections=[], plate_area=100000, dry_run=True)
    assert r3.hit_rule == "dry_run", r3
    print("[OK] dry-run 仅输出特征")
    print("=== 全部自测通过 ===")
    return 0


def _batch_features_csv(engine: DianjiGradeRule, path: Path, dry_run: bool) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            features = engine.features_from_row(row)
            result = engine.classify(features=features, dry_run=dry_run)
            out = {
                "image_name": row.get("image_name", ""),
                "image_stem": row.get("image_stem", ""),
                "true_grade": row.get("true_grade", row.get("original_grade", "")),
                "pred_grade": result.grade,
                "hit_rule": result.hit_rule,
                "status": result.status,
                "message": result.message,
            }
            # 关键特征
            for k in (
                "detection_count",
                "A_ratio",
                "B_ratio",
                "C_ratio",
                "Airhole_ratio",
                "has_nodule",
                "total_bbox_union_image_ratio",
            ):
                out[k] = features.get(k, "")
            for name in engine.class_names:
                out[f"count__{name}"] = features.get(f"count__{name}", 0)
                out[f"area_ratio__{name}"] = features.get(f"area_ratio__{name}", 0.0)
            rows_out.append(out)
    return rows_out


def _batch_labels_dir(engine: DianjiGradeRule, path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for label_path in sorted(path.glob("*.txt")):
        if label_path.name == "classes.txt":
            continue
        dets = load_yolo_labels(
            label_path,
            image_width=args.image_width,
            image_height=args.image_height,
            conf_thres=args.conf_thres,
        )
        result = engine.classify(
            detections=dets,
            plate_area=args.plate_area,
            image_width=args.image_width,
            image_height=args.image_height,
            coords="pixel",
            dry_run=args.dry_run,
        )
        out = {
            "image_name": label_path.stem,
            "true_grade": "",
            "pred_grade": result.grade,
            "hit_rule": result.hit_rule,
            "status": result.status,
            "message": result.message,
        }
        for k, v in result.features.items():
            out[k] = v
        rows_out.append(out)
    return rows_out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    # 稳定列序：元信息在前
    preferred = [
        "image_name",
        "image_stem",
        "true_grade",
        "pred_grade",
        "hit_rule",
        "status",
        "message",
        "detection_count",
        "A_ratio",
        "B_ratio",
        "C_ratio",
        "Airhole_ratio",
        "has_nodule",
        "total_bbox_union_image_ratio",
    ]
    keys: list[str] = []
    seen: set[str] = set()
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k)
            seen.add(k)
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # 显式自测，或无任何批量输入时默认跑自测
    if args.self_test or (args.features is None and args.labels is None):
        return run_self_test()

    engine = DianjiGradeRule(rules_path=args.rules)
    rows: list[dict[str, Any]] = []
    if args.features is not None:
        if not args.features.exists():
            print(f"特征 CSV 不存在: {args.features}", file=sys.stderr)
            return 1
        rows = _batch_features_csv(engine, args.features, dry_run=args.dry_run)
    else:
        if not args.labels.exists():
            print(f"标签目录不存在: {args.labels}", file=sys.stderr)
            return 1
        rows = _batch_labels_dir(engine, args.labels, args)

    if args.output is None:
        for row in rows[:5]:
            print(
                f"{row.get('image_name')}: true={row.get('true_grade')} "
                f"pred={row.get('pred_grade')} rule={row.get('hit_rule')}"
            )
        if len(rows) > 5:
            print(f"... 共 {len(rows)} 行")
        print(f"status={engine.status} rules={len(engine.rules)}")
    else:
        _write_csv(args.output, rows)
        print(f"已写入 {args.output} （{len(rows)} 行），status={engine.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
