# -*- coding: utf-8 -*-
"""B3-clean 判级切换 D4 式业务输出对比页（§7「B3-clean 部署」步骤③用户裁决材料）。

在同一批企业真值图（grade-test 161 为主表、grade-valid 161 为参照）上对比两臂：
- 旧臂（现役 v1）：显示同源 18 类检测器特征 → 63 列 → LGBM v1（拒判 0.7362）；
- 新臂（B3-clean 集成管线）：detectors.dianji_detect 端到端（显示 18 类不动 +
  判级专用 5 类检测器 → 33 列 → LGBM b3clean，拒判 0.7819）。

产出：compare_summary.json / compare_per_image.csv / review.html（分歧图卡片）。
旧臂 test acc/macroF1 必须复现 v1 记录值 0.74534/0.67970，作为对照 harness 正确性锚点。
只读评测：不改任何配置、不写数据集目录。
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

WORKTREE = Path("C:/Users/23916/AppData/Local/Temp/jcw_b3clean_wt")
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
MANIFEST = Path(r"E:/PythonProject/archives/dianjinie_grade_classification_v1/manifests/samples.csv")
VIEW_ROOT = Path(r"E:/PythonProject/archives/dianjinie_grade_classification_v1")
GRADES = ["3#", "9950", "9996", "bad"]
GRADE_TO_TYPE = {"3#": 4, "9950": 3, "9996": 2, "bad": 1}
V1_RECORDED_TEST = {"acc": 0.74534, "macro_f1": 0.67970}
V1_ANCHOR_TOL = 0.02
THUMB_WIDTH = 420


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def macro_f1(per_class: dict[str, dict[str, float]]) -> float:
    values = [per_class[g]["f1"] for g in GRADES]
    return float(np.mean(values))


def prf(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for grade in GRADES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == grade and p == grade)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != grade and p == grade)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == grade and p != grade)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[grade] = {"precision": precision, "recall": recall, "f1": f1,
                         "support": sum(1 for t in y_true if t == grade)}
    return report


def read_image_pair(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    image_gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image_bgr is None or image_gray is None:
        raise ValueError(f"无法解码图像: {image_path}")
    return image_bgr, image_gray


def thumbnail_base64(image_bgr: np.ndarray, boxes: list[dict[str, Any]],
                     names: dict[int, str], highlight: list[tuple[int, int, int]]) -> str:
    canvas = image_bgr.copy()
    for box in boxes:
        x1, y1 = int(box["cx"] - box["w"] / 2), int(box["cy"] - box["h"] / 2)
        x2, y2 = int(box["cx"] + box["w"] / 2), int(box["cy"] + box["h"] / 2)
        label = int(box["label"])
        color = highlight[label % len(highlight)] if highlight else (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        name = names.get(label, str(label))
        cv2.putText(canvas, name, (max(0, x1), max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    height, width = canvas.shape[:2]
    scale = THUMB_WIDTH / width
    canvas = cv2.resize(canvas, (THUMB_WIDTH, int(height * scale)))
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(WORKTREE))
    os.chdir(WORKTREE)

    import lightgbm as lgb

    from inference import models as model_state
    from inference.detectors import _predict_dianji_yolo_raw, dianji_detect
    from inference.grade_features import extract_grade_features

    # 服务臂模型栈（新臂）
    model_state.load_dianji_grade_model()
    model_state.load_dianji_grade_detector()
    model_state.load_dianji_detector()
    model_state.load_resnet()
    assert model_state.dianji_grade_model is not None
    new_contract = model_state.dianji_grade_contract
    new_threshold = float(model_state.dianji_grade_threshold)
    new_names = list(model_state.dianji_grade_feature_names)
    new_booster = model_state.dianji_grade_model
    new_best_iter = int(model_state.dianji_grade_best_iteration)

    # 旧臂模型栈（v1，63 列，显示同源特征）
    with (WORKTREE / "weights" / "dianji_grade_lgbm_v1_contract.json").open(encoding="utf-8") as handle:
        old_contract = json.load(handle)
    old_names = list(old_contract["feature_names"])
    assert len(old_names) == 63
    old_threshold = float(old_contract["rejection"]["threshold"])
    old_best_iter = int(old_contract["model"]["best_iteration"])
    old_booster = lgb.Booster(model_file=str(WORKTREE / "weights" / "dianji_grade_lgbm_v1.txt"))
    assert tuple(old_booster.feature_name()) == tuple(old_names)
    display_names = {int(k): str(v) for k, v in model_state.slf_yolo.names.items()}
    assert len(display_names) == 18

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    per_image: list[dict[str, Any]] = []
    palette = [(60, 76, 231), (94, 177, 191), (0, 200, 255), (0, 170, 255), (180, 105, 255),
               (80, 200, 120), (0, 120, 255), (200, 160, 60), (160, 60, 220), (255, 90, 90)]
    t0 = time.perf_counter()

    for split in ("test", "valid"):
        rows = sorted((r for r in manifest if r["split"] == split), key=lambda r: r["image_name"])
        expected = 161
        assert len(rows) == expected, f"{split} 行数 {len(rows)} != {expected}"
        for index, row in enumerate(rows, 1):
            image_path = VIEW_ROOT / row["link_relpath"]
            image_bgr, image_gray = read_image_pair(image_path)
            image_height, image_width = image_bgr.shape[:2]

            # —— 旧臂：显示检测器原始框 → 63 列 → v1 booster ——
            raw = _predict_dianji_yolo_raw(image_bgr)
            detections63 = []
            for det in raw:
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                detections63.append({
                    "label": int(det["label"]), "conf": float(det["conf"]),
                    "cx": (x1 + x2) / 2.0 / image_width, "cy": (y1 + y2) / 2.0 / image_height,
                    "w": w / image_width, "h": h / image_height,
                })
            old_features = extract_grade_features(image_bgr, detections63, old_names, image_gray=image_gray)
            old_row = np.asarray([[old_features[name] for name in old_names]], dtype=np.float64)
            old_probs = np.asarray(old_booster.predict(old_row, num_iteration=old_best_iter), dtype=np.float64).reshape(-1)
            old_grade_id = int(np.argmax(old_probs))
            old_grade = GRADES[old_grade_id]
            old_conf = float(old_probs[old_grade_id])

            # —— 新臂：端到端集成管线（与 HTTP 同一函数） ——
            response = dianji_detect(str(image_path))
            data = response.get("data") or {}
            new_grade = str(data.get("grade"))
            new_conf = float(data.get("confidence"))
            display_boxes = data.get("defects") or []

            per_image.append({
                "split": split,
                "image_name": row["image_name"],
                "true_grade": row["grade"],
                "old_grade": old_grade,
                "old_confidence": old_conf,
                "old_manual_review": old_conf < old_threshold,
                "new_grade": new_grade,
                "new_confidence": new_conf,
                "new_manual_review": bool(data.get("manual_review")),
                "grade_changed": old_grade != new_grade,
                "review_changed": (old_conf < old_threshold) != bool(data.get("manual_review")),
                "old_correct": old_grade == row["grade"],
                "new_correct": new_grade == row["grade"],
                "e2e_code": response.get("code"),
            })
            if index == 1 or index % 20 == 0 or index == len(rows):
                rate = index / (time.perf_counter() - t0)
                print(f"  [{split} {index}/{len(rows)}] {rate:.2f} img/s", flush=True)

    # —— 汇总指标 ——
    summary: dict[str, Any] = {"arms": {}, "anchors": {}, "agreement": {}, "shas": {}}
    review_cards: list[dict[str, Any]] = []
    for split in ("test", "valid"):
        rows = [r for r in per_image if r["split"] == split]
        y_true = [r["true_grade"] for r in rows]
        arms = {}
        for arm, grade_key, conf_key in (("old_v1", "old_grade", "old_confidence"),
                                          ("new_b3clean", "new_grade", "new_confidence")):
            y_pred = [r[grade_key] for r in rows]
            per_class = prf(y_true, y_pred)
            confs = np.asarray([r[conf_key] for r in rows], dtype=np.float64)
            threshold = old_threshold if arm == "old_v1" else new_threshold
            accepted = confs >= threshold
            acc_accepted = float((np.asarray([1.0 if r[grade_key] == r["true_grade"] else 0.0
                                              for r in rows])[accepted]).mean()) if accepted.any() else 0.0
            arms[arm] = {
                "threshold": threshold,
                "accuracy": float(np.mean([1.0 if p == t else 0.0 for p, t in zip(y_pred, y_true)])),
                "macro_f1": macro_f1(per_class),
                "per_class": per_class,
                "manual_review_rate": float((~accepted).mean()),
                "accepted_coverage": float(accepted.mean()),
                "acc_accepted": acc_accepted,
            }
        changed = [r for r in rows if r["grade_changed"]]
        summary["arms"][split] = arms
        summary["agreement"][split] = {
            "n": len(rows),
            "grade_changed": len(changed),
            "grade_change_rate": len(changed) / len(rows),
            "review_changed": sum(1 for r in rows if r["review_changed"]),
            "old_to_new": {
                f"{o}->{n}": sum(1 for r in rows if r["old_grade"] == o and r["new_grade"] == n)
                for o in GRADES for n in GRADES
                if sum(1 for r in rows if r["old_grade"] == o and r["new_grade"] == n) > 0
            },
        }

    old_test = summary["arms"]["test"]["old_v1"]
    summary["anchors"]["old_v1_test_recorded"] = V1_RECORDED_TEST
    summary["anchors"]["old_v1_test_reproduced"] = {
        "acc": old_test["accuracy"], "macro_f1": old_test["macro_f1"]}
    summary["anchors"]["within_tolerance"] = bool(
        abs(old_test["accuracy"] - V1_RECORDED_TEST["acc"]) <= V1_ANCHOR_TOL
        and abs(old_test["macro_f1"] - V1_RECORDED_TEST["macro_f1"]) <= V1_ANCHOR_TOL
    )

    summary["shas"] = {
        "display_onnx": file_sha256(WORKTREE / "weights" / "dianjienie_yolo11m_ryw18_640.onnx")[:16],
        "grade_onnx": file_sha256(WORKTREE / "weights" / "dianji0211_y11m640_b3clean.onnx")[:16],
        "lgbm_v1": file_sha256(WORKTREE / "weights" / "dianji_grade_lgbm_v1.txt")[:16],
        "lgbm_b3clean": file_sha256(WORKTREE / "weights" / "dianji_grade_lgbm_b3clean_v1.txt")[:16],
    }

    # —— review.html：先 test 后 valid 的分歧图卡片（缩略图 + 两臂输出 + 真值） ——
    disagree = [r for r in per_image if r["grade_changed"]]
    print(f"grade_changed total={len(disagree)}，渲染缩略图…", flush=True)
    manifest_index = {r["image_name"]: r for r in manifest}
    for row in disagree:
        rel = manifest_index[row["image_name"]]["link_relpath"]
        image_bgr, _ = read_image_pair(VIEW_ROOT / rel)
        response = dianji_detect(str(VIEW_ROOT / rel))
        boxes = (response.get("data") or {}).get("defects") or []
        boxes = [{"cx": b["cx"], "cy": b["cy"], "w": b["w"], "h": b["h"], "label": b["label"]} for b in boxes]
        thumb = thumbnail_base64(image_bgr, boxes, display_names, palette)
        verdict = "both_wrong" if (not row["old_correct"] and not row["new_correct"]) else (
            "new_fix" if row["new_correct"] else "old_fix")
        review_cards.append({
            "row": row,
            "thumb": thumb,
            "verdict": verdict,
        })

    cards_html = []
    verdict_label = {"new_fix": "新臂修正", "old_fix": "旧臂正确/新臂改错", "both_wrong": "两臂皆错"}
    verdict_color = {"new_fix": "#1a7f37", "old_fix": "#cf222e", "both_wrong": "#9a6700"}
    for card in review_cards:
        r = card["row"]
        cards_html.append(f"""
        <div class="card">
          <img src="data:image/jpeg;base64,{card['thumb']}" alt="{r['image_name']}"/>
          <div class="meta">
            <div class="img">{r['image_name']} <span class="split">{r['split']}</span></div>
            <div><b>企业真值</b>：{r['true_grade']}</div>
            <div><b>旧臂 v1</b>：{r['old_grade']}（conf {r['old_confidence']:.4f}，
                 {'需人工' if r['old_manual_review'] else '自动'}）</div>
            <div><b>新臂 B3-clean</b>：{r['new_grade']}（conf {r['new_confidence']:.4f}，
                 {'需人工' if r['new_manual_review'] else '自动'}）</div>
            <div class="verdict" style="color:{verdict_color[card['verdict']]}">
              {verdict_label[card['verdict']]}</div>
          </div>
        </div>""")

    def arm_table(arms: dict) -> str:
        rows_html = ""
        for arm, label in (("old_v1", "旧臂 v1（63 列，拒判 {:.4f}）"),
                           ("new_b3clean", "新臂 B3clean（33 列，拒判 {:.4f}）")):
            a = arms[arm]
            rows_html += ("<tr><td>" + label.format(a["threshold"]) + "</td>"
                          + f"<td>{a['accuracy']:.5f}</td><td>{a['macro_f1']:.5f}</td>"
                          + f"<td>{a['manual_review_rate']:.2%}</td>"
                          + f"<td>{a['accepted_coverage']:.2%}</td>"
                          + f"<td>{a['acc_accepted']:.5f}</td>"
                          + "".join(f"<td>{a['per_class'][g]['f1']:.3f}</td>" for g in GRADES)
                          + "</tr>")
        return ("<table><tr><th>臂</th><th>acc</th><th>macroF1</th><th>人工复核率</th>"
                "<th>接收覆盖</th><th>接收区acc</th>"
                + "".join(f"<th>F1 {g}</th>" for g in GRADES) + "</tr>" + rows_html + "</table>")

    def agree_table(split: str) -> str:
        a = summary["agreement"][split]
        cells = "".join(f"<td>{k}：{v}</td>" for k, v in a["old_to_new"].items())
        return (f"<p><b>{split}</b>：n={a['n']}，等级改变 {a['grade_changed']} 张"
                f"（{a['grade_change_rate']:.2%}），人工复核状态改变 {a['review_changed']} 张。"
                f"迁移明细：{cells or '无'}</p>")

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>B3-clean 判级切换对比（D4 式）</title><style>
body{{font-family:"Microsoft YaHei",sans-serif;margin:24px;color:#1f2328}}
table{{border-collapse:collapse;margin:8px 0}}
td,th{{border:1px solid #d0d7de;padding:4px 10px;font-size:13px}}
.grid{{display:flex;flex-wrap:wrap;gap:14px}}
.card{{width:450px;border:1px solid #d0d7de;border-radius:8px;padding:10px}}
.card img{{width:100%;border-radius:4px}}
.meta{{font-size:13px;line-height:1.7}}
.img{{font-size:11px;color:#57606a;word-break:break-all}}
.split{{background:#ddf4ff;border-radius:4px;padding:0 6px}}
.verdict{{font-weight:600}}
h2{{border-bottom:1px solid #d0d7de;padding-bottom:6px}}
</style></head><body>
<h1>B3-clean 判级切换对比（A 方案显示/判级分离）</h1>
<p>显示路径不变（18 类 ONNX {summary['shas']['display_onnx']}…）；判级路径由 v1（63 列）
切换为 B3-clean（33 列，判级专用检测器 {summary['shas']['grade_onnx']}…）。
真值口径：企业划分等级（用户 2026-09-04 确认）。</p>
<h2>主表：grade-test 161</h2>{arm_table(summary['arms']['test'])}{agree_table('test')}
<h2>参照：grade-valid 161</h2>{arm_table(summary['arms']['valid'])}{agree_table('valid')}
<p>锚点校验：旧臂 test 复现值 acc={old_test['accuracy']:.5f}/macroF1={old_test['macro_f1']:.5f}
 vs v1 记录 0.74534/0.67970 → {'一致，harness 可信' if summary['anchors']['within_tolerance'] else '<b>超出容差，先查 harness</b>'}</p>
<h2>分歧图裁决（{len(review_cards)} 张，等级 old≠new）</h2>
<div class="grid">{''.join(cards_html)}</div>
</body></html>"""
    (OUT_DIR / "review.html").write_text(html, encoding="utf-8")

    (OUT_DIR / "compare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "compare_per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)
    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arms": {"old": "lgbm v1 (63col, thr 0.7362)", "new": "lgbm b3clean_v1 (33col, thr 0.7819)"},
        "splits": {"test": 161, "valid": 161},
        "manifest": str(MANIFEST),
        "shas": summary["shas"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary["arms"]["test"],
                      "anchor_ok": summary["anchors"]["within_tolerance"],
                      "agreement": summary["agreement"]["test"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
