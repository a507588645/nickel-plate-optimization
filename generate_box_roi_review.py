#!/usr/bin/env python3
"""按「框」审查：低置信度框 ROI 放大 + 多处理拼接，而不是看整图。

用途：全图增强后仍看不清时，把每个难框裁成局部大图，并排 raw / Lab /
灰阶 / Gamma / top-hat / black-hat / 边缘 / 局部强对比，并给出结构分启发式。

输出：
  <out>/rois/*.jpg     每个框一张 montage
  <out>/review.html    按框翻页审查（保留/删除/存疑）
  <out>/boxes.jsonl    框元数据
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from review_enhance_utils import (
    IMAGE_SUFFIXES,
    BoxRec,
    build_roi_panels,
    crop_box_roi,
    draw_box_on_roi,
    imread_unicode,
    imwrite_unicode,
    make_montage,
    read_boxes,
    structure_score,
)


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{TITLE}}</title>
<style>
:root{--ink:#18201c;--paper:#f3efe4;--panel:#fffdf7;--line:#c8c0ae;--green:#2f6f4e;--red:#a33b32;--amber:#b07a18;--blue:#27658a}
*{box-sizing:border-box}body{margin:0;font-family:"Microsoft YaHei",sans-serif;background:var(--paper);color:var(--ink)}
header{padding:10px 16px;background:#1d2923;color:#fff;display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}
h1{margin:0;font-size:16px}.stats{font-size:12px;color:#d9e2dc}
.app{display:grid;grid-template-columns:300px 1fr;height:calc(100vh - 56px)}
aside{border-right:1px solid var(--line);background:var(--panel);padding:10px;overflow:auto}
.filters{display:grid;gap:6px;margin-bottom:8px}input,select,button,textarea{font:inherit}
input,select,textarea{width:100%;border:1px solid var(--line);padding:6px;background:#fff}
.list{display:grid;gap:4px}.item{padding:6px;font-size:11px;background:#f8f5ec;cursor:pointer;border-left:4px solid transparent;word-break:break-all}
.item.active{outline:1px solid var(--blue);background:#e7f1f5}
.item.keep{border-left-color:var(--green)}.item.remove{border-left-color:var(--red)}.item.uncertain{border-left-color:var(--amber)}
main{display:grid;grid-template-rows:auto 1fr auto;min-width:0;background:#111}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--line)}
.btn{border:0;padding:8px 12px;color:#fff;cursor:pointer}.keepBtn{background:var(--green)}.removeBtn{background:var(--red)}.uncertainBtn{background:var(--amber)}.neutral{background:#59645d}.exportBtn{background:var(--blue)}
.viewer{overflow:auto;display:flex;align-items:flex-start;justify-content:center;padding:12px}
.viewer img{max-width:100%;height:auto;background:#000;box-shadow:0 8px 24px #0008}
.meta{padding:10px 12px;background:var(--panel);border-top:1px solid var(--line);display:grid;gap:6px}
.meta .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.hint{font-size:12px;color:#625f57}.filename{font-weight:700;font-size:13px}
.score{font-family:Consolas,monospace;font-size:12px;background:#eef6f0;padding:6px 8px;border:1px solid #c5d6cb}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;color:#fff}
.badge.low{background:#a33b32}.badge.mid{background:#b07a18}.badge.ok{background:#2f6f4e}
textarea{min-height:40px}
</style></head><body>
<header>
  <h1>{{TITLE}}</h1>
  <div class="stats" id="stats"></div>
  <div>
    <button class="btn exportBtn" onclick="exportReview()">导出 JSONL</button>
    <button class="btn neutral" onclick="clearReview()">清空进度</button>
  </div>
</header>
<div class="app">
<aside>
  <div class="filters">
    <input id="search" placeholder="搜索图片名/类别" oninput="renderList()">
    <select id="statusFilter" onchange="renderList()">
      <option value="all">全部状态</option>
      <option value="pending" selected>未审查</option>
      <option value="keep">保留</option>
      <option value="remove">删除</option>
      <option value="uncertain">存疑</option>
    </select>
    <select id="structFilter" onchange="renderList()">
      <option value="all" selected>全部结构分</option>
      <option value="very_low">结构很低 (&lt;0.15)</option>
      <option value="low">结构偏低 (&lt;0.30)</option>
      <option value="ok">结构尚可 (≥0.30)</option>
    </select>
    <select id="sortBy" onchange="renderList()">
      <option value="structure_asc" selected>排序：结构分升序（更难先看）</option>
      <option value="conf_asc">排序：置信度升序</option>
      <option value="area_asc">排序：框面积升序</option>
      <option value="name">排序：文件名</option>
    </select>
  </div>
  <div class="list" id="list"></div>
</aside>
<main>
  <div class="toolbar">
    <button class="btn neutral" onclick="move(-1)">上一框 A</button>
    <button class="btn neutral" onclick="move(1)">下一框 D</button>
    <button class="btn keepBtn" onclick="mark('keep')">保留 K</button>
    <button class="btn removeBtn" onclick="mark('remove')">删除 R</button>
    <button class="btn uncertainBtn" onclick="mark('uncertain')">存疑 U</button>
    <button class="btn neutral" onclick="mark('remove');">空框一键删（同删除）</button>
    <span class="hint">建议：montage 中框内若各面板都像均匀噪声/无边缘 → 删；有连贯结构 → 保留；拿不准 → 存疑。默认自动跳下一未审框。</span>
  </div>
  <div class="viewer"><img id="roiImg" alt="roi"></div>
  <div class="meta">
    <div class="filename" id="title"></div>
    <div class="row">
      <span id="badge" class="badge"></span>
      <span class="score" id="score"></span>
    </div>
    <div class="hint" id="advice"></div>
    <textarea id="note" placeholder="备注（假纹理/框偏了/边缘在框外…）" oninput="saveNote()"></textarea>
  </div>
</main>
</div>
<script>
const ITEMS={{ITEMS}};
const KEY={{STORAGE_KEY}};
let states=JSON.parse(localStorage.getItem(KEY)||'{}');
let filtered=[]; let current=0;
function stateFor(id){return states[id]||(states[id]={decision:'',note:''})}
function persist(){localStorage.setItem(KEY,JSON.stringify(states));updateStats()}
function decisionOf(it){return stateFor(it.id).decision||'pending'}
function renderList(){
  const q=document.getElementById('search').value.toLowerCase();
  const f=document.getElementById('statusFilter').value;
  const sf=document.getElementById('structFilter').value;
  const sort=document.getElementById('sortBy').value;
  filtered=ITEMS.filter(it=>{
    const st=decisionOf(it);
    const text=(it.image_name+' '+it.class_name+' '+it.id).toLowerCase();
    if(q && !text.includes(q)) return false;
    if(f!=='all' && st!==f) return false;
    const s=it.structure||0;
    if(sf==='very_low' && s>=0.15) return false;
    if(sf==='low' && s>=0.30) return false;
    if(sf==='ok' && s<0.30) return false;
    return true;
  });
  filtered.sort((a,b)=>{
    if(sort==='structure_asc') return (a.structure||0)-(b.structure||0);
    if(sort==='conf_asc') return (a.confidence||0)-(b.confidence||0);
    if(sort==='area_asc') return (a.area||0)-(b.area||0);
    return a.image_name.localeCompare(b.image_name)||(a.box_index-b.box_index);
  });
  const list=document.getElementById('list'); list.innerHTML='';
  filtered.forEach((it,i)=>{
    const d=document.createElement('div');
    const st=decisionOf(it);
    d.className='item '+st+(i===current?' active':'');
    d.textContent=`${i+1}. ${it.image_name} #${it.box_index} ${it.class_name} c=${it.confidence?.toFixed?.(3)??it.confidence} s=${(it.structure||0).toFixed(2)}`;
    d.onclick=()=>{current=i;show()};
    list.appendChild(d);
  });
  if(current>=filtered.length) current=Math.max(0,filtered.length-1);
  show();
}
function adviceFor(it){
  const s=it.structure||0;
  if(s<0.12) return '启发式：结构分很低，框内可能几乎无缺陷结构 → 优先考虑删除（仍需人眼确认 montage）。';
  if(s<0.25) return '启发式：结构分偏低，仔细看 top-hat/black-hat/边缘 面板是否有连贯结构。';
  return '启发式：结构分尚可，检查结构是否落在黄框内；若结构在框外则位置偏了，应删除。';
}
function show(){
  if(!filtered.length){document.getElementById('title').textContent='无匹配框';document.getElementById('roiImg').removeAttribute('src');return}
  const it=filtered[current];
  const st=stateFor(it.id);
  document.getElementById('title').textContent=`${current+1}/${filtered.length} | ${it.image_name} | 框#${it.box_index} | ${it.class_name} | conf=${it.confidence}`;
  document.getElementById('roiImg').src=it.roi_rel;
  document.getElementById('note').value=st.note||'';
  const s=it.structure||0;
  const badge=document.getElementById('badge');
  badge.textContent=s<0.15?'结构很低':(s<0.30?'结构偏低':'结构尚可');
  badge.className='badge '+(s<0.15?'low':(s<0.30?'mid':'ok'));
  document.getElementById('score').textContent=`structure=${s.toFixed(3)} | edge=${it.edge_mean} | std=${it.std} | highfreq=${it.highfreq} | area=${it.area?.toFixed?.(6)} | 当前=${decisionOf(it)}`;
  document.getElementById('advice').textContent=adviceFor(it);
  [...document.querySelectorAll('.item')].forEach((e,i)=>e.classList.toggle('active',i===current));
  updateStats();
}
function mark(v){
  if(!filtered.length)return;
  const it=filtered[current];
  stateFor(it.id).decision=v;
  persist();
  // jump to next pending in full list order of current filter
  const start=current;
  for(let step=1; step<=filtered.length; step++){
    const j=(start+step)%filtered.length;
    if(decisionOf(filtered[j])==='pending'){current=j;renderList();return}
  }
  renderList();
}
function saveNote(){if(!filtered.length)return;stateFor(filtered[current].id).note=document.getElementById('note').value;persist()}
function move(d){if(!filtered.length)return;current=Math.max(0,Math.min(filtered.length-1,current+d));show()}
function updateStats(){
  let k=0,r=0,u=0,p=0;
  ITEMS.forEach(it=>{const d=decisionOf(it);if(d==='keep')k++;else if(d==='remove')r++;else if(d==='uncertain')u++;else p++});
  document.getElementById('stats').textContent=`框 ${ITEMS.length} | 未审 ${p} | 保留 ${k} | 删除 ${r} | 存疑 ${u}`;
}
function exportReview(){
  const lines=ITEMS.map(it=>{
    const s=stateFor(it.id);
    return JSON.stringify({
      id:it.id,
      image_name:it.image_name,
      box_index:it.box_index,
      class_id:it.class_id,
      class_name:it.class_name,
      confidence:it.confidence,
      xywh:it.xywh,
      structure:it.structure,
      decision:s.decision||'pending',
      note:s.note||'',
      scores:{edge_mean:it.edge_mean,std:it.std,highfreq:it.highfreq}
    });
  }).join('\n');
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([lines],{type:'application/jsonl'}));
  a.download='{{EXPORT_NAME}}';
  a.click();
}
function clearReview(){if(confirm('清空本页全部框审查进度？')){states={};persist();renderList()}}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  const k=e.key.toLowerCase();
  if(k==='a')move(-1); if(k==='d')move(1);
  if(k==='k')mark('keep'); if(k==='r')mark('remove'); if(k==='u')mark('uncertain');
});
renderList();
</script></body></html>'''


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成低置信度「按框 ROI」审查页")
    p.add_argument(
        "dataset_root",
        type=Path,
        help="含 images 或 images_raw + labels_with_conf 的数据集/抽样集",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/box_roi_reviews"),
    )
    p.add_argument("--conf-max", type=float, default=0.40, help="只审查 conf < 该阈值的框")
    p.add_argument("--pad-ratio", type=float, default=1.8, help="ROI 相对框的外扩比例")
    p.add_argument("--min-crop", type=int, default=128, help="ROI 最小边（像素）")
    p.add_argument("--cell-size", type=int, default=360, help="montage 每个面板边长")
    p.add_argument("--max-boxes", type=int, default=0, help="限制框数，0=全部")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prefer-low-structure",
        action="store_true",
        help="若设置 max-boxes，优先保留结构分更低的框",
    )
    return p.parse_args()


def find_images_dir(dataset: Path) -> Path:
    for cand in (dataset / "images_raw", dataset / "images"):
        if cand.is_dir():
            return cand
    raise NotADirectoryError(f"images_raw/images not found under {dataset}")


def load_classes(dataset: Path) -> list[str]:
    path = dataset / "classes.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8").splitlines()
    return []


def main() -> None:
    args = parse_args()
    dataset = args.dataset_root.resolve()
    images_dir = find_images_dir(dataset)
    labels_dir = dataset / "labels_with_conf"
    if not labels_dir.is_dir():
        labels_dir = dataset / "labels"
    classes = load_classes(dataset)

    out = args.output_root / f"{dataset.name}_boxroi"
    rois_dir = out / "rois"
    rois_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )

    items: list[dict] = []
    for img_path in image_paths:
        boxes = read_boxes(labels_dir / f"{img_path.stem}.txt")
        if not boxes:
            continue
        image = imread_unicode(img_path)
        if image is None:
            print(f"[warn] unreadable {img_path}")
            continue
        for bi, box in enumerate(boxes):
            conf = box.confidence
            if conf is None or conf >= args.conf_max:
                continue
            roi, box_xyxy, crop_xyxy = crop_box_roi(
                image, box, pad_ratio=args.pad_ratio, min_size=args.min_crop
            )
            # structure on inner box crop (tighter), not full pad
            bx1, by1, bx2, by2 = box_xyxy
            # expand slightly for score
            h, w = image.shape[:2]
            sx1 = max(0, bx1 - 2)
            sy1 = max(0, by1 - 2)
            sx2 = min(w, bx2 + 2)
            sy2 = min(h, by2 + 2)
            core = image[sy1:sy2, sx1:sx2]
            scores = structure_score(core if core.size else roi)

            panels = build_roi_panels(roi)
            # draw box on raw panel only for spatial reference
            panels["raw"] = draw_box_on_roi(panels["raw"], box_xyxy, crop_xyxy, (0, 255, 255), 2)
            for key in ("lab", "gray", "local_eq"):
                if key in panels:
                    panels[key] = draw_box_on_roi(panels[key], box_xyxy, crop_xyxy, (0, 255, 255), 2)

            labels = {
                "raw": "raw+box",
                "lab": "Lab强对比",
                "gray": "灰阶CLAHE",
                "gamma": "Gamma提暗",
                "tophat": "亮残差",
                "blackhat": "暗残差",
                "edges": "边缘",
                "local_eq": "局部强对比",
            }
            montage = make_montage(
                panels,
                panel_order=["raw", "lab", "gray", "gamma", "tophat", "blackhat", "edges", "local_eq"],
                cell_size=args.cell_size,
                labels=labels,
                cols=4,
            )
            roi_name = f"{img_path.stem}__b{bi:03d}.jpg"
            roi_path = rois_dir / roi_name
            imwrite_unicode(roi_path, montage)

            cls_name = (
                classes[box.class_id]
                if 0 <= box.class_id < len(classes)
                else str(box.class_id)
            )
            item = {
                "id": f"{img_path.name}#{bi}",
                "image_name": img_path.name,
                "box_index": bi,
                "class_id": box.class_id,
                "class_name": cls_name,
                "confidence": conf,
                "xywh": [box.cx, box.cy, box.w, box.h],
                "area": box.area,
                "structure": scores["structure"],
                "edge_mean": scores["edge_mean"],
                "std": scores["std"],
                "highfreq": scores["highfreq"],
                "roi_rel": Path(os.path.relpath(roi_path, out)).as_posix(),
                "box_xyxy": list(box_xyxy),
                "crop_xyxy": list(crop_xyxy),
            }
            items.append(item)

    if args.max_boxes and len(items) > args.max_boxes:
        import random

        rng = random.Random(args.seed)
        if args.prefer_low_structure:
            items.sort(key=lambda x: x["structure"])
            items = items[: args.max_boxes]
        else:
            items = rng.sample(items, args.max_boxes)
        items.sort(key=lambda x: (x["image_name"], x["box_index"]))

    # write jsonl meta
    meta_path = out / "boxes.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    title = f"按框 ROI 审查 - {dataset.name} (conf<{args.conf_max})"
    html = HTML.replace("{{TITLE}}", title)
    html = html.replace("{{ITEMS}}", json.dumps(items, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("{{STORAGE_KEY}}", json.dumps(f"boxroi::{dataset.name}"))
    html = html.replace("{{EXPORT_NAME}}", f"{dataset.name}_boxroi_review.jsonl")
    (out / "review.html").write_text(html, encoding="utf-8")

    summary = {
        "dataset": str(dataset),
        "output": str(out),
        "review_html": str(out / "review.html"),
        "n_boxes": len(items),
        "conf_max": args.conf_max,
        "structure_mean": (
            round(sum(i["structure"] for i in items) / len(items), 4) if items else None
        ),
        "structure_lt_0.15": sum(1 for i in items if i["structure"] < 0.15),
        "structure_lt_0.30": sum(1 for i in items if i["structure"] < 0.30),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = f"""# 按框 ROI 审查

全图增强仍看不清时，用本页**逐框**看局部放大 montage。

- 源: `{dataset}`
- 框数: {len(items)}（conf < {args.conf_max}）
- 结构分 <0.15: {summary['structure_lt_0.15']}
- 结构分 <0.30: {summary['structure_lt_0.30']}

## 怎么判

1. 看黄框内在 8 个面板里是否出现**连贯**结构（边缘/残差/对比）
2. 各面板都是均匀噪声、无边缘 → **删除**
3. 有结构但在框外 → 位置偏了 → **删除**
4. 有结构在框内 → **保留**
5. 仍拿不准 → **存疑**

结构分只是启发式，不能替代人眼。
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
