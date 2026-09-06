# -*- coding: utf-8 -*-
"""C2.4 电解镍判级人工复核页生成（路线文档 §4.2，防循环论证强制闸门）。

抽样（预注册，抽样后不得更改）：
- A 层：val+test 中新方案（LGBM）与弱标签（规则树）分歧的**全部**图（104 张）；
- B 层：val+test 一致图中均匀随机 36 张（seed42，按 sample_id 排序后抽取）；
- train 分歧不入复核集（模型见过，避免记忆干扰「新不劣于旧」比较）。
共 140 张 ≥ 100 达标。

复核任务：逐张判断「判级」对错——A 层新旧输出不同，四选一：旧对/新对/都错/都难说；
B 层两臂输出一致，二选一：对/错。用户以编号在对话中批量裁决，Agent 转写进 decisions.jsonl。
产物：review.html（编号卡片，含旧方案检测框渲染）+ manifest.json + 空白 decisions.jsonl。
"""
from __future__ import annotations

import base64
import colorsys
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "c2p4_review"
WEAKLABELS = HERE / "weaklabels.jsonl"
PER_IMAGE = HERE / "c2p3_per_image.csv"
SEED = 42
N_AGREE_SAMPLE = 36
THUMB_WIDTH = 880
RYW18 = ("wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble", "bump", "fawu",
         "overfire", "NiOH", "greenstain", "suanwuran", "quejiao", "watermark", "youpiao",
         "heibian", "mijixiaokeng", "fanxitong")
TYPE_RULE_MEANING = {
    "3": "合格板（无结瘤且 A/B/C/气孔占比均低于阈值）",
    "2": "烫洗（A 类占比高、B/C 低）",
    "1": "C 类占比 &gt; 50%",
    "6": "B 类占比 &gt; 50% 且 C 类 &lt; 50%",
    "4": "其他（以上均不满足，等待处理）",
}


def thumbnail(image_path: Path, defects: list[dict[str, Any]]) -> str:
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码: {image_path}")
    h, w = image.shape[:2]
    scale = min(1.0, THUMB_WIDTH / float(w))
    panel = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale != 1.0 else image.copy()
    for d in defects:
        x1 = int((float(d["cx"]) - float(d["w"]) / 2) * scale)
        y1 = int((float(d["cy"]) - float(d["h"]) / 2) * scale)
        x2 = int((float(d["cx"]) + float(d["w"]) / 2) * scale)
        y2 = int((float(d["cy"]) + float(d["h"]) / 2) * scale)
        cid = int(d["label"]) % 18
        r, g, b = colorsys.hsv_to_rgb((cid * 0.618033988749895) % 1.0, 0.85, 0.95)
        color = (int(b * 255), int(g * 255), int(r * 255))
        cv2.rectangle(panel, (max(0, x1), max(0, y1)), (min(panel.shape[1] - 1, x2), min(panel.shape[0] - 1, y2)), color, 2, cv2.LINE_AA)
        cv2.putText(panel, f"{cid}:{RYW18[cid]}", (max(0, x1), max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError(f"编码失败: {image_path}")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weak = {}
    with WEAKLABELS.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            weak[row["sample_id"]] = row
    with PER_IMAGE.open("r", encoding="utf-8-sig", newline="") as handle:
        per_image = [r for r in csv.DictReader(handle) if r["split"] in ("valid", "test")]
    assert len(per_image) == 751, f"val+test 行数 {len(per_image)} != 751"

    disagree = sorted((r for r in per_image if r["type_correct"] == "0"), key=lambda r: r["sample_id"])
    agree = sorted((r for r in per_image if r["type_correct"] == "1"), key=lambda r: r["sample_id"])
    rng = random.Random(SEED)
    agree_sample = sorted(rng.sample(agree, N_AGREE_SAMPLE), key=lambda r: r["sample_id"])
    selected = [(r, "disagree") for r in disagree] + [(r, "agree") for r in agree_sample]
    assert len(selected) >= 100, f"复核单元 {len(selected)} < 100"
    print(f"stratum A (disagree)={len(disagree)}, stratum B (agree sample)={len(agree_sample)}, total={len(selected)}")

    cards = []
    manifest_rows = []
    for idx, (row, stratum) in enumerate(selected, 1):
        wid = weak[row["sample_id"]]
        image_path = Path(wid["image_path"])
        defects = json.loads(wid["defects"]) if wid["defects"] else []
        old_type, old_nodule = int(wid["type"]), int(wid["nodule"])
        new_type, new_conf = int(row["type_pred"]), float(row["type_conf"])
        thumb = thumbnail(image_path, defects)
        same = str(old_type) == str(new_type)
        cards.append({
            "idx": idx, "sample_id": row["sample_id"], "split": row["split"],
            "group_id": row["group_id"], "image": image_path.name, "thumb": thumb,
            "old_type": old_type, "old_nodule": old_nodule, "new_type": new_type,
            "new_conf": new_conf, "same": same, "n_defects": len(defects),
            "stratum": stratum,
        })
        manifest_rows.append({
            "idx": idx, "sample_id": row["sample_id"], "stratum": stratum,
            "old_type": old_type, "new_type": new_type, "new_conf": new_conf,
            "split": row["split"], "group_id": row["group_id"],
        })

    cards_html = []
    for c in cards:
        if c["same"]:
            verdict_ui = ("两方案一致：<b>type {}</b>。请判：<b>对 / 错</b>（若错请给出正确 type）。"
                          .format(c["old_type"]))
            old_ui = f"<b>type {c['old_type']}</b>（nodule {c['old_nodule']}）"
            new_ui = f"<b>type {c['new_type']}</b>（conf {c['new_conf']:.3f}，与旧方案一致）"
        else:
            verdict_ui = ("两方案分歧。请判四选一：<b>旧对 / 新对 / 都错（请给正确 type）/ 都难说</b>。")
            old_ui = f"<b>type {c['old_type']}</b>（nodule {c['old_nodule']}）"
            new_ui = f"<b>type {c['new_type']}</b>（conf {c['new_conf']:.3f}）"
        cards_html.append(f"""
<div class="card" id="unit{c['idx']}">
  <div class="head">#{c['idx']:03d} · {c['image']} · {c['split']} · {c['stratum']}</div>
  <img src="data:image/jpeg;base64,{c['thumb']}" alt="#{c['idx']:03d}"/>
  <div class="meta">
    <div><b>旧方案（规则树）</b>：{old_ui}</div>
    <div><b>新方案（LGBM）</b>：{new_ui}</div>
    <div class="verdict">{verdict_ui}</div>
  </div>
</div>""")

    legend = "".join(f"<li>type {k} = {v}</li>" for k, v in TYPE_RULE_MEANING.items())
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>C2.4 电解镍判级人工复核（{len(cards)} 单元）</title><style>
body{{font-family:"Microsoft YaHei",sans-serif;margin:20px;color:#1f2328}}
.grid{{display:flex;flex-wrap:wrap;gap:16px}}
.card{{width:900px;border:1px solid #d0d7de;border-radius:8px;padding:10px}}
.card img{{width:100%;border-radius:4px}}
.head{{font-size:13px;color:#57606a;margin-bottom:6px}}
.meta{{font-size:14px;line-height:1.8}}
.verdict{{color:#9a6700;font-weight:600}}
h2{{border-bottom:1px solid #d0d7de;padding-bottom:6px}}
</style></head><body>
<h1>C2.4 电解镍判级人工复核（{len(cards)} 单元 ≥ 100 达标）</h1>
<p>目的：人工判断「判级」本身的对错，比较<b>旧方案（规则树）</b>与<b>新方案（LGBM，63 列特征）</b>，
作为 C2.5 切换裁决的外部验收锚（防循环论证）。</p>
<ul><li><b>A 层（{sum(1 for c in cards if not c['same'])} 张）</b>：新旧输出分歧图——四选一：旧对 / 新对 / 都错（给正确 type）/ 都难说。</li>
<li><b>B 层（{sum(1 for c in cards if c['same'])} 张）</b>：两臂一致图——对 / 错（若错给正确 type）。</li>
<li>框为旧方案检测输出（18 类，标签=class_id:类名）。type 含义（来自规则树分支）：{legend}</li>
<li>裁决方式：按编号回复，例如「#001 对，#002 新对，#003 都错（真值 4）」；可批量（如「#010-#020 全部旧对」）。</li></ul>
<h2>复核单元</h2>
<div class="grid">{''.join(cards_html)}</div>
</body></html>"""
    (OUT_DIR / "review.html").write_text(html, encoding="utf-8")

    manifest = {
        "task": "C2.4 电解镍判级人工复核（防循环论证外部验收锚）",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sampling": {
            "stratum_A": "val+test 中新旧判级分歧的全部图（预注册：全部纳入）",
            "stratum_B": f"val+test 一致图均匀随机 {N_AGREE_SAMPLE} 张（seed={SEED}，按 sample_id 排序后抽样）",
            "excluded": "train 分歧不入复核集（模型见过，避免记忆干扰）",
            "total": len(cards),
            "seed": SEED,
        },
        "verdict_schema": {
            "A 层（分歧）": "old_right | new_right | both_wrong | uncertain（both_wrong 建议附 true_type）",
            "B 层（一致）": "correct | incorrect（incorrect 建议附 true_type）",
        },
        "inputs": {"weaklabels": str(WEAKLABELS), "per_image": str(PER_IMAGE)},
        "units": manifest_rows,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps({
                "idx": row["idx"], "sample_id": row["sample_id"], "stratum": row["stratum"],
                "old_type": row["old_type"], "new_type": row["new_type"],
                "verdict": None, "true_type": None, "note": "",
            }, ensure_ascii=False) + "\n")
    print(json.dumps({"units": len(cards), "out": str(OUT_DIR)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
