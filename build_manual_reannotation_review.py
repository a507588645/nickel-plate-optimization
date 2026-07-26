#!/usr/bin/env python3
"""Build a browser-based manual reannotation page for the L1 union dataset.

The page deliberately does not display ``ryw.pt`` predictions. Existing labels
are only a starting point: annotators can draw, resize, move, reclassify,
delete, restore, and add boxes. Exported JSONL is a human decision record that
``apply_manual_reannotation.py`` converts into a new five-column label version.

No source images or source labels are changed by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_UNION = Path(r"E:/PythonProject/datasets/union_original_labels_v1")
DEFAULT_AUDIT = ROOT / "label_quality_audit_v1" / "review_priority.csv"
DEFAULT_OUTPUT = ROOT / "manual_reannotation_v1"

CLASSES = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]
CLASSES_ZH = [
    "污渍", "水渍", "黄渍", "边部结瘤", "气孔", "结粒", "疙瘩", "发乌", "烧板",
    "氢氧化镍", "绿色结晶", "酸污染", "缺角", "水印", "油泡", "黑边", "密集小坑", "返析铜",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 L1 人工重标审查页（不显示旧模型候选、不修改原标签）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--union", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--audit-priority", type=Path, default=DEFAULT_AUDIT,
                        help="可选的 review_priority.csv；存在时按人工复核优先级排序")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0,
                        help="仅生成优先级前 N 张，0 表示全部。适合先做试标批次")
    parser.add_argument("--include-unlabelled", action="store_true",
                        help="将当前无标签图也纳入；默认只审已有原标签的图片")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def read_boxes(path: Path) -> list[list[float]]:
    if not path.is_file():
        return []
    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            class_id = int(fields[0])
            coords = [float(value) for value in fields[1:]]
        except ValueError:
            continue
        if 0 <= class_id < len(CLASSES) and len(coords) == 4:
            boxes.append([class_id, *coords])
    return boxes


def load_priority(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["union_name"]: row for row in csv.DictReader(handle) if row.get("union_name")}


def annotation_reason(priority_row: dict[str, str]) -> str:
    """Remove model-derived hints before a human annotator sees the image."""
    reasons = priority_row.get("reasons", "")
    kept = [part.strip() for part in reasons.split("|") if "旧模型分歧" not in part]
    return " | ".join(kept) or "常规人工复核"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    union = args.union.resolve()
    manifest_path = union / "manifests" / "samples.csv"
    images_dir, labels_dir = union / "images", union / "labels"
    if not manifest_path.is_file() or not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("union 必须包含 manifests/samples.csv、images/ 和 labels/")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    image_index = {
        path.stem: path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    priority = load_priority(args.audit_priority)
    items: list[dict] = []
    for row in rows:
        name = row["union_name"]
        if row.get("has_label") != "True" and not args.include_unlabelled:
            continue
        image = image_index.get(name)
        if image is None:
            continue
        priority_row = priority.get(name, {})
        items.append({
            "name": name,
            # Relative URLs work when the page is opened directly and when a
            # local server is rooted at E:/PythonProject for browser QA.
            "src": Path(os.path.relpath(image, args.output_dir)).as_posix(),
            "source": row.get("source", ""),
            "source_split": row.get("source_split", ""),
            "has_original_label": row.get("has_label") == "True",
            "original_boxes": read_boxes(labels_dir / f"{name}.txt"),
            "priority_rank": int(priority_row.get("priority_rank") or 999999),
            "priority_score": float(priority_row.get("priority_score") or 0),
            "priority_reasons": annotation_reason(priority_row),
        })
    items.sort(key=lambda item: (item["priority_rank"], item["name"]))
    if args.limit:
        items = items[:args.limit]

    print(f"人工重标页：{len(items)} 张；已有原标签 {sum(item['has_original_label'] for item in items)} 张")
    if args.dry_run:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "classes": CLASSES,
        "classes_zh": CLASSES_ZH,
        "items": items,
    }
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("__GENERATED__", datetime.now().strftime("%Y-%m-%d %H:%M"))
    (args.output_dir / "review.html").write_text(html, encoding="utf-8")
    summary = {
        "generated_at": payload["generated_at"],
        "union": str(union),
        "images": len(items),
        "with_original_labels": sum(item["has_original_label"] for item in items),
        "include_unlabelled": args.include_unlabelled,
        "priority_source": str(args.audit_priority) if args.audit_priority.is_file() else None,
        "review_html": str(args.output_dir / "review.html"),
        "review_scope": str(args.output_dir / "review_scope.txt"),
        "export_format": "manual_reannotation_result.jsonl",
    }
    (args.output_dir / "review_scope.txt").write_text(
        "\n".join(item["name"] for item in items) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    instructions = """# L1 人工重标页使用说明

本页只显示原始人工标签，**不显示 `ryw.pt` 候选**。原框只是起点，最终以人工观察为准。

1. 选择一张图；优先级排序来自标签质量审计，不代表旧模型裁决。
2. 点击框选择。拖动框内部移动；拖动边或角调整大小；类别下拉框可改类别；删除可移除误标；“恢复原框”撤销单框改动。
3. 点“新增框”后，在图像空白位置拖出框；用类别下拉框选定正确类别；不需要的新增框删除即可。
4. 人工确认该图已完整重标后，点“标为已完成”。没有缺陷时，删除所有框再标为已完成，导出器会写同名空标签。
5. 绑定保存文件后每次改动自动写入 JSONL；也可最后点“导出 JSONL”。
6. 不能确定时写备注并保持“未完成”，不要把不确定图导入冻结测试集。

导出后运行：

```powershell
python apply_manual_reannotation.py `
  --review-jsonl <导出的 JSONL> `
  --scope-file E:/PythonProject/优化/manual_reannotation_v1/review_scope.txt `
  --require-completed
```

脚本只导出 `completed=true` 的图片。带上 `--scope-file --require-completed` 时，必须本批次
所有图片都已经完成，避免漏审图片被悄悄遗漏。

结果只会写入 `E:\PythonProject\优化\manual_reannotation_v1\labels_human_reviewed_v1`，不会覆盖原始数据集标签。
"""
    (args.output_dir / "README.md").write_text(instructions, encoding="utf-8")
    print(f"产物: {args.output_dir / 'review.html'}")
    return 0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>L1 人工重标</title>
<style>
:root{--ink:#202428;--paper:#f4f0e8;--panel:#fffdf8;--line:#c9c1b4;--accent:#0c6b67;--warn:#b95a12;--danger:#b53737;--selected:#ffd447}*{box-sizing:border-box}body{margin:0;background:linear-gradient(120deg,#edf1ea,#e7ddd0);color:var(--ink);font-family:"Microsoft YaHei",sans-serif;height:100vh;overflow:hidden}.app{display:grid;grid-template-columns:290px minmax(0,1fr);height:100%}aside{background:#203530;color:#f9f4e9;padding:14px;overflow:auto}h1{font-family:Georgia,"Times New Roman",serif;font-size:23px;margin:0 0 7px}.subtitle{font-size:12px;line-height:1.45;color:#d7e3d4;margin-bottom:12px}.search,.filter,select,textarea{width:100%;font:inherit}.search,.filter,select{padding:7px;border:1px solid var(--line);border-radius:5px;background:#fff;color:var(--ink)}.filter{margin-top:7px}.list{margin-top:11px;display:grid;gap:3px}.item{padding:7px 8px;border-left:3px solid transparent;cursor:pointer;font-size:12px;line-height:1.35;background:#2d4841}.item:hover{background:#3a5b52}.item.active{border-left-color:#f4bd3d;background:#476b5f}.item.done::after{content:" 已完成";color:#caefab;font-weight:bold}.rank{opacity:.78;font-variant-numeric:tabular-nums}main{display:grid;grid-template-rows:auto minmax(0,1fr) auto;background:var(--paper)}header{padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;align-items:center;gap:8px;flex-wrap:wrap}.status{font-size:12px;color:#52615e}.btn{border:1px solid #9d9589;border-radius:5px;background:#fffdf8;padding:6px 9px;font:inherit;font-size:12px;cursor:pointer}.btn:hover{background:#f4ead7}.btn.primary{background:var(--accent);color:#fff;border-color:var(--accent)}.btn.warn{background:#f3a344;color:#2a1c06;border-color:#d98326}.btn.danger{background:var(--danger);color:#fff;border-color:var(--danger)}.workspace{min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 310px}.viewer{position:relative;overflow:auto;background:#161c1a;min-height:0}.canvas-wrap{position:relative;display:inline-block;min-width:100%;min-height:100%}canvas{display:block;background:#161c1a;cursor:crosshair}.hint{position:sticky;left:9px;bottom:9px;margin:9px;background:#16251fdd;color:#eef5ed;padding:6px 8px;border-radius:4px;font-size:11px;max-width:580px;pointer-events:none}.sidepanel{padding:12px;background:var(--panel);border-left:1px solid var(--line);overflow:auto}.meta{font-size:12px;line-height:1.55;padding:8px;background:#edf3ec;border-left:3px solid var(--accent);margin-bottom:10px}.field{margin:10px 0}.field label{display:block;font-size:12px;font-weight:bold;margin-bottom:4px}.row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.boxlist{max-height:36vh;overflow:auto;border:1px solid var(--line);border-radius:5px;background:#fff}.boxrow{padding:7px;border-bottom:1px solid #e5ded3;cursor:pointer;font-size:12px;line-height:1.35}.boxrow:last-child{border-bottom:0}.boxrow.selected{background:#fff1b8}.boxrow.deleted{opacity:.5;text-decoration:line-through}.boxrow.added{border-left:3px solid var(--accent)}textarea{min-height:64px;padding:7px;border:1px solid var(--line);border-radius:5px;resize:vertical}.legend{font-size:11px;color:#5b564e;line-height:1.5}.hotkey{font-family:Consolas,monospace;background:#eee7db;padding:1px 4px;border-radius:3px}.banner{font-size:12px;background:#fff0d2;border:1px solid #e6b55e;padding:6px;border-radius:4px;margin-bottom:9px}.modal{position:fixed;inset:0;background:#0008;display:none;align-items:center;justify-content:center;z-index:20}.modal.show{display:flex}.modal-card{width:min(520px,92vw);background:#fffdf8;padding:18px;border-radius:8px;box-shadow:0 18px 70px #0008}.modal-card h2{margin-top:0;font-family:Georgia,"Times New Roman",serif}.modal-card p{font-size:13px;line-height:1.6}@media(max-width:900px){.app{grid-template-columns:200px minmax(0,1fr)}.workspace{grid-template-columns:minmax(0,1fr)}.sidepanel{display:none}}
</style>
</head>
<body>
<div class="app">
  <aside>
    <h1>人工重标</h1>
    <div class="subtitle">原标签只是起点。这里不显示旧模型候选，不让模型影响人工判断。</div>
    <input id="search" class="search" placeholder="搜索文件名" oninput="renderList()">
    <select id="filter" class="filter" onchange="renderList()"><option value="all">全部</option><option value="todo">未完成</option><option value="done">已完成</option><option value="original">仅有原标签</option><option value="empty">原本无标签</option></select>
    <div id="list" class="list"></div>
  </aside>
  <main>
    <header>
      <button class="btn" onclick="navigate(-1)">上一张 A</button><button class="btn" onclick="navigate(1)">下一张 D</button>
      <button class="btn" onclick="fitImage()">适应窗口</button><button class="btn" onclick="zoomBy(1.2)">放大 +</button><button class="btn" onclick="zoomBy(1/1.2)">缩小 -</button>
      <button id="drawBtn" class="btn primary" onclick="toggleDraw()">新增框 N</button><button class="btn danger" onclick="deleteSelected()">删除选中 Del</button><button class="btn" onclick="restoreSelected()">恢复原框</button>
      <button class="btn warn" onclick="toggleCompleted()">标为已完成 C</button><button class="btn" onclick="bindSaveFile()">绑定保存文件</button><span id="saveState" class="status">未绑定：进度只在浏览器内</span>
      <button class="btn" onclick="exportJSONL()">导出 JSONL</button><button class="btn" onclick="document.getElementById('importFile').click()">导入进度</button><input id="importFile" type="file" accept=".jsonl" hidden onchange="importJSONL(event)">
      <button class="btn" onclick="showHelp()">使用说明</button><span id="counter" class="status"></span>
    </header>
    <section class="workspace">
      <div id="viewer" class="viewer"><div class="canvas-wrap"><canvas id="canvas"></canvas><div class="hint">普通模式：拖动框内部移动、拖边/角缩放、点击选中。新增框模式：拖拽画框，类别由右侧下拉框决定。按 N / Del / C / A / D。</div></div></div>
      <div class="sidepanel">
        <div id="meta" class="meta"></div>
        <div id="drawBanner" class="banner" hidden>新增框模式已开启：在图像上拖拽画框。按 Esc 或再点“新增框”退出。</div>
        <div class="field"><label for="classSelect">选中/新增框类别</label><select id="classSelect" onchange="changeSelectedClass()"></select></div>
        <div class="field"><div class="row"><button class="btn" onclick="markNoDefect()">确认无缺陷</button><button class="btn" onclick="clearAllBoxes()">删除全部框</button></div><div class="legend">“确认无缺陷”会删掉全部框并标记完成。仅在人工确认后使用。</div></div>
        <div class="field"><label>当前图标注框</label><div id="boxList" class="boxlist"></div></div>
        <div class="field"><label for="note">人工备注</label><textarea id="note" placeholder="例如：原标漏掉右下角气孔；该图无缺陷；类别边界待业务确认" oninput="saveNote()"></textarea></div>
        <div class="legend">绿色实线=保留/新增框；灰色虚线=已删除原框。只有标为“已完成”的图才建议导出为训练或测试标签。</div>
      </div>
    </section>
  </main>
</div>
<div id="help" class="modal"><div class="modal-card"><h2>人工重标规则</h2><p>按目视和业务类别定义重标，不依据旧模型框。先修正漏标、错类和框边界；无法判断就留下备注且不要标为已完成。</p><p>导出的 JSONL 仅是人工操作记录。随后使用 `apply_manual_reannotation.py --scope-file review_scope.txt --require-completed` 生成独立五列标签版本，确保当前批次没有漏审。原始 `labels/` 不会被覆盖。</p><div class="row"><button class="btn primary" onclick="hideHelp()">关闭</button></div></div></div>
<script>
const DATA=__PAYLOAD__;const ITEMS=DATA.items;const CL=DATA.classes;const CLZ=DATA.classes_zh;const KEY='manual_reannotation_v1';let state=JSON.parse(localStorage.getItem(KEY)||'{}');let filtered=[];let current=0;let img=null;let imageLoading=false;let imageError='';let imageRequest=0;let scale=1;let selected=null;let drawMode=false;let pointer=null;let saveHandle=null;let saveTimer=null;let fitAfterLoad=true;
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),viewer=document.getElementById('viewer');
function item(){return filtered[current]||null}function blankState(){return{boxes:null,deleted:{},completed:false,note:''}}function st(name){if(!state[name])state[name]=blankState();return state[name]}function originalBoxes(it){return it.original_boxes.map((b,i)=>({id:'o'+i,class_id:b[0],xywh:b.slice(1),origin:'original'}))}function boxes(it){const s=st(it.name);if(s.boxes===null)s.boxes=originalBoxes(it);return s.boxes}function activeBoxes(it){return boxes(it).filter(b=>!st(it.name).deleted[b.id])}function save(){localStorage.setItem(KEY,JSON.stringify(state));scheduleSave();updateHeader()}function completedCount(){return ITEMS.filter(it=>st(it.name).completed).length}
function classOptions(){const sel=document.getElementById('classSelect');sel.innerHTML='';CL.forEach((name,i)=>{const op=document.createElement('option');op.value=i;op.textContent=`${i} ${name} / ${CLZ[i]}`;sel.appendChild(op)})}function setClassSelect(){const it=item();const box=it&&selected?boxes(it).find(b=>b.id===selected):null;if(box)document.getElementById('classSelect').value=box.class_id}
function renderList(){const q=document.getElementById('search').value.toLowerCase(),f=document.getElementById('filter').value;filtered=ITEMS.filter(it=>{const s=st(it.name);return it.name.toLowerCase().includes(q)&&(f==='all'||(f==='todo'&&!s.completed)||(f==='done'&&s.completed)||(f==='original'&&it.has_original_label)||(f==='empty'&&!it.has_original_label))});if(current>=filtered.length)current=Math.max(0,filtered.length-1);const root=document.getElementById('list');root.innerHTML='';filtered.forEach((it,i)=>{const s=st(it.name),d=document.createElement('div');d.className='item'+(i===current?' active':'')+(s.completed?' done':'');d.innerHTML=`<span class="rank">#${it.priority_rank===999999?'-':it.priority_rank}</span> ${it.name}<br><small>${activeBoxes(it).length} 框 · ${it.priority_reasons}</small>`;d.onclick=()=>{current=i;selected=null;loadImage(true)};root.appendChild(d)});loadImage(false)}
function navigate(delta){if(!filtered.length)return;current=Math.max(0,Math.min(filtered.length-1,current+delta));selected=null;loadImage(true)}function imagePos(e){const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left)*(canvas.width/r.width),y:(e.clientY-r.top)*(canvas.height/r.height)}}function fromRect(x1,y1,x2,y2){const left=Math.max(0,Math.min(x1,x2)),top=Math.max(0,Math.min(y1,y2)),right=Math.min(canvas.width,Math.max(x1,x2)),bottom=Math.min(canvas.height,Math.max(y1,y2));return[(left+right)/2/canvas.width,(top+bottom)/2/canvas.height,(right-left)/canvas.width,(bottom-top)/canvas.height]}
function rect(box){const[cx,cy,w,h]=box.xywh;return{x:(cx-w/2)*canvas.width,y:(cy-h/2)*canvas.height,w:w*canvas.width,h:h*canvas.height}}function hitBox(p){const it=item();if(!it)return null;let best=null,area=Infinity;for(const b of activeBoxes(it)){const r=rect(b);if(p.x>=r.x&&p.x<=r.x+r.w&&p.y>=r.y&&p.y<=r.y+r.h&&r.w*r.h<area){best=b;area=r.w*r.h}}return best}function handleAt(box,p){const r=rect(box),tol=Math.max(7,8/scale);const points={nw:[r.x,r.y],n:[r.x+r.w/2,r.y],ne:[r.x+r.w,r.y],e:[r.x+r.w,r.y+r.h/2],se:[r.x+r.w,r.y+r.h],s:[r.x+r.w/2,r.y+r.h],sw:[r.x,r.y+r.h],w:[r.x,r.y+r.h/2]};for(const[k,v]of Object.entries(points))if(Math.abs(p.x-v[0])<=tol&&Math.abs(p.y-v[1])<=tol)return k;return null}
function setMeta(it){const status=imageLoading?'状态：图片加载中':imageError?`状态：${imageError}`:'';document.getElementById('meta').textContent=`${it.name}\n来源：${it.source}/${it.source_split}\n优先级：${it.priority_rank===999999?'未排序':it.priority_rank}，${it.priority_reasons}${status?'\n'+status:''}`}
function loadImage(reset){const it=item(),request=++imageRequest;img=null;imageLoading=Boolean(it);imageError='';if(!it){imageLoading=false;draw();renderBoxes();updateHeader();return}setMeta(it);document.getElementById('note').value=st(it.name).note||'';draw();renderBoxes();updateHeader();const requestedName=it.name,nextImage=new Image();nextImage.onload=()=>{if(request!==imageRequest||item()?.name!==requestedName)return;img=nextImage;imageLoading=false;canvas.width=nextImage.naturalWidth;canvas.height=nextImage.naturalHeight;if(reset||fitAfterLoad){fitImage();fitAfterLoad=false}else draw();setMeta(item());renderBoxes();updateHeader()};nextImage.onerror=()=>{if(request!==imageRequest||item()?.name!==requestedName)return;imageLoading=false;imageError='图片加载失败，请确认 review.html 没有被移出 manual_reannotation_v1 目录。';setMeta(item());draw();renderBoxes();updateHeader()};nextImage.src=it.src;renderListOnly()}function renderListOnly(){document.querySelectorAll('.item').forEach((node,i)=>node.classList.toggle('active',i===current))}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);if(!img){ctx.fillStyle='#161c1a';ctx.fillRect(0,0,canvas.width,canvas.height);if(imageLoading||imageError){ctx.fillStyle=imageError?'#ffb6a8':'#d8e6df';ctx.font='16px sans-serif';ctx.fillText(imageError||'图片加载中…',18,32)}return}ctx.drawImage(img,0,0);const it=item();if(!it)return;for(const b of boxes(it)){const r=rect(b),deleted=st(it.name).deleted[b.id];ctx.save();ctx.lineWidth=Math.max(2,2/scale);ctx.setLineDash(deleted?[8,5]:[]);ctx.strokeStyle=deleted?'#929292':'#00c98d';ctx.fillStyle=deleted?'#777':'#00c98d';if(selected===b.id){ctx.strokeStyle='#ffd447';ctx.lineWidth=Math.max(3,3/scale)}ctx.strokeRect(r.x,r.y,r.w,r.h);ctx.setLineDash([]);ctx.font=`${Math.max(14,16/scale)}px sans-serif`;ctx.fillText(`${CL[b.class_id]} / ${CLZ[b.class_id]}${b.origin==='added'?' +':''}`,r.x,Math.max(16/scale,r.y-4/scale));if(selected===b.id&&!deleted){for(const p of [[r.x,r.y],[r.x+r.w/2,r.y],[r.x+r.w,r.y],[r.x+r.w,r.y+r.h/2],[r.x+r.w,r.y+r.h],[r.x+r.w/2,r.y+r.h],[r.x,r.y+r.h],[r.x,r.y+r.h/2]])ctx.fillRect(p[0]-4/scale,p[1]-4/scale,8/scale,8/scale)}ctx.restore()}if(pointer&&pointer.kind==='draw'){const r=fromRect(pointer.start.x,pointer.start.y,pointer.now.x,pointer.now.y);const temp={xywh:r};const px=rect(temp);ctx.save();ctx.strokeStyle='#ffb000';ctx.lineWidth=Math.max(2,2/scale);ctx.setLineDash([8/scale,5/scale]);ctx.strokeRect(px.x,px.y,px.w,px.h);ctx.restore()}}
function renderBoxes(){const root=document.getElementById('boxList'),it=item();if(!it){root.innerHTML='';return}root.innerHTML='';boxes(it).forEach((b,i)=>{const deleted=st(it.name).deleted[b.id],div=document.createElement('div');div.className='boxrow'+(selected===b.id?' selected':'')+(deleted?' deleted':'')+(b.origin==='added'?' added':'');div.innerHTML=`#${i+1} ${CL[b.class_id]} / ${CLZ[b.class_id]}${b.origin==='added'?'（新增）':''}<br><small>x=${b.xywh[0].toFixed(3)} y=${b.xywh[1].toFixed(3)} w=${b.xywh[2].toFixed(3)} h=${b.xywh[3].toFixed(3)}</small>`;div.onclick=()=>{selected=b.id;setClassSelect();draw();renderBoxes()};root.appendChild(div)});setClassSelect()}
function setCanvasScale(){canvas.style.width=Math.round(canvas.width*scale)+'px';canvas.style.height=Math.round(canvas.height*scale)+'px'}function fitImage(){if(!img)return;const fit=Math.min((viewer.clientWidth-20)/canvas.width,(viewer.clientHeight-20)/canvas.height);scale=Math.min(1,Math.max(.08,fit));setCanvasScale();draw()}function zoomBy(factor){scale=Math.min(12,Math.max(.05,scale*factor));setCanvasScale();draw()}
function toggleDraw(){drawMode=!drawMode;document.getElementById('drawBtn').textContent=drawMode?'退出新增框 Esc':'新增框 N';document.getElementById('drawBanner').hidden=!drawMode;canvas.style.cursor=drawMode?'crosshair':'default'}function deleteSelected(){const it=item();if(!it||!selected)return;st(it.name).deleted[selected]=true;save();draw();renderBoxes()}function restoreSelected(){const it=item();if(!it||!selected)return;const b=boxes(it).find(b=>b.id===selected);if(!b)return;if(b.origin==='added')st(it.name).boxes=boxes(it).filter(x=>x.id!==b.id);else delete st(it.name).deleted[b.id];selected=null;save();draw();renderBoxes()}function changeSelectedClass(){const it=item(),b=it&&selected?boxes(it).find(b=>b.id===selected):null;if(!b)return;b.class_id=Number(document.getElementById('classSelect').value);save();draw();renderBoxes()}function clearAllBoxes(){const it=item();if(!it||!confirm('确认删除当前图所有框？'))return false;for(const b of boxes(it))st(it.name).deleted[b.id]=true;selected=null;save();draw();renderBoxes();return true}function markNoDefect(){const it=item();if(!it||!confirm('确认当前图人工检查后无任何缺陷？这会删除全部框并标为已完成。'))return;for(const b of boxes(it))st(it.name).deleted[b.id]=true;selected=null;st(it.name).completed=true;save();draw();renderBoxes();renderList()}function toggleCompleted(){const it=item();if(!it)return;st(it.name).completed=!st(it.name).completed;save();renderList()}function saveNote(){const it=item();if(!it)return;st(it.name).note=document.getElementById('note').value;save()}
canvas.addEventListener('pointerdown',e=>{const it=item();if(!it||!img)return;canvas.setPointerCapture(e.pointerId);const p=imagePos(e);if(drawMode){pointer={kind:'draw',start:p,now:p};draw();return}const hit=hitBox(p);if(!hit){selected=null;draw();renderBoxes();return}selected=hit.id;const h=handleAt(hit,p);pointer={kind:'edit',id:hit.id,handle:h||'move',start:p,original:[...hit.xywh]};setClassSelect();draw();renderBoxes()});canvas.addEventListener('pointermove',e=>{if(!pointer)return;const it=item(),p=imagePos(e);if(pointer.kind==='draw'){pointer.now=p;draw();return}const b=boxes(it).find(b=>b.id===pointer.id);if(!b)return;const r0=rect({xywh:pointer.original});let x1=r0.x,y1=r0.y,x2=r0.x+r0.w,y2=r0.y+r0.h,dx=p.x-pointer.start.x,dy=p.y-pointer.start.y,h=pointer.handle;if(h==='move'){x1=Math.max(0,Math.min(canvas.width-r0.w,x1+dx));y1=Math.max(0,Math.min(canvas.height-r0.h,y1+dy));x2=x1+r0.w;y2=y1+r0.h}else{if(h.includes('w'))x1=Math.max(0,Math.min(x2-4,x1+dx));if(h.includes('e'))x2=Math.min(canvas.width,Math.max(x1+4,x2+dx));if(h.includes('n'))y1=Math.max(0,Math.min(y2-4,y1+dy));if(h.includes('s'))y2=Math.min(canvas.height,Math.max(y1+4,y2+dy))}b.xywh=fromRect(x1,y1,x2,y2);draw()});canvas.addEventListener('pointerup',e=>{if(!pointer)return;const it=item();if(pointer.kind==='draw'){const xywh=fromRect(pointer.start.x,pointer.start.y,pointer.now.x,pointer.now.y);if(xywh[2]*canvas.width>=4&&xywh[3]*canvas.height>=4){const id='a'+Date.now()+'_'+Math.random().toString(16).slice(2);boxes(it).push({id,class_id:Number(document.getElementById('classSelect').value),xywh,origin:'added'});selected=id}}pointer=null;save();draw();renderBoxes()});
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='a'||e.key==='A')navigate(-1);if(e.key==='d'||e.key==='D')navigate(1);if(e.key==='n'||e.key==='N')toggleDraw();if(e.key==='Delete'||e.key==='Backspace')deleteSelected();if(e.key==='c'||e.key==='C')toggleCompleted();if(e.key==='Escape'&&drawMode)toggleDraw()});viewer.addEventListener('wheel',e=>{if(!e.ctrlKey)return;e.preventDefault();zoomBy(e.deltaY<0?1.15:1/1.15)},{passive:false});
function updateHeader(){const it=item(),loadStatus=imageLoading?' · 图片加载中':imageError?' · 图片加载失败':'';document.getElementById('counter').textContent=`${current+1}/${filtered.length||0} · 已完成 ${completedCount()}/${ITEMS.length}${it?' · '+activeBoxes(it).length+' 框':''}${loadStatus}`}
function serialize(){const lines=[];for(const it of ITEMS){const s=st(it.name),all=boxes(it);const pack=b=>({id:b.id,class_id:b.class_id,class_name:CL[b.class_id],xywh:b.xywh,origin:b.origin});lines.push(JSON.stringify({image_name:it.name,completed:Boolean(s.completed),note:s.note||'',boxes:activeBoxes(it).map(pack),all_boxes:all.map(pack),deleted_box_ids:Object.keys(s.deleted),source:it.source,has_original_label:it.has_original_label}))}return lines.join('\n')+'\n'}function exportJSONL(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([serialize()],{type:'application/jsonl'}));a.download='manual_reannotation_result.jsonl';a.click()}
function importJSONL(e){const file=e.target.files[0];if(!file)return;file.text().then(text=>{let loaded=0;text.split(/\r?\n/).forEach(line=>{if(!line.trim())return;const r=JSON.parse(line),it=ITEMS.find(x=>x.name===r.image_name);if(!it)return;const savedBoxes=r.all_boxes||r.boxes||[];state[it.name]={boxes:savedBoxes.map(b=>({id:b.id||'i'+Math.random(),class_id:Number(b.class_id),xywh:b.xywh,origin:b.origin||'imported'})),deleted:Object.fromEntries((r.deleted_box_ids||[]).map(id=>[id,true])),completed:Boolean(r.completed),note:r.note||''};loaded++});save();renderList();alert('已导入 '+loaded+' 张的进度')});e.target.value=''}
function idb(){return new Promise((resolve,reject)=>{const req=indexedDB.open('manual_reannotation_db',1);req.onupgradeneeded=()=>req.result.createObjectStore('kv');req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error)})}async function bindSaveFile(){try{saveHandle=await window.showSaveFilePicker({suggestedName:'manual_reannotation_progress.jsonl',types:[{description:'JSONL',accept:{'application/jsonl':['.jsonl']}}]});const db=await idb();const tx=db.transaction('kv','readwrite');tx.objectStore('kv').put(saveHandle,'handle');document.getElementById('saveState').textContent='已绑定，自动保存中';await writeFile()}catch(err){if(err.name!=='AbortError')alert('绑定失败：'+err)}}async function writeFile(){if(!saveHandle)return;try{const w=await saveHandle.createWritable();await w.write(serialize());await w.close();document.getElementById('saveState').textContent='已自动保存 '+new Date().toLocaleTimeString()}catch(err){document.getElementById('saveState').textContent='自动保存失败，请手动导出'}}function scheduleSave(){clearTimeout(saveTimer);saveTimer=setTimeout(writeFile,700)}async function restoreHandle(){try{const db=await idb(),tx=db.transaction('kv','readonly'),req=tx.objectStore('kv').get('handle');req.onsuccess=async()=>{if(req.result){saveHandle=req.result;document.getElementById('saveState').textContent='已恢复保存绑定';await writeFile()}}}catch(_){}}function showHelp(){document.getElementById('help').classList.add('show')}function hideHelp(){document.getElementById('help').classList.remove('show')}
classOptions();renderList();restoreHandle();
</script>
</body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
