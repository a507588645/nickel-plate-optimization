#!/usr/bin/env python3
"""为「原图 vs 增强图」对照标注生成单一 HTML 审查页。

左右两侧同步显示同一张图的原图和增强图，各自叠加对应伪标签框。
增强侧支持人工决策（保留/删除/存疑）并导出 JSONL；原图侧仅作参照。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{TITLE}}</title><style>
:root{--ink:#18201c;--paper:#f3efe4;--panel:#fffdf7;--line:#c8c0ae;--green:#2f6f4e;--red:#a33b32;--amber:#b07a18;--blue:#27658a}
*{box-sizing:border-box}body{margin:0;font-family:"Microsoft YaHei",sans-serif;background:var(--paper);color:var(--ink)}
header{height:64px;padding:10px 18px;background:#1d2923;color:white;display:flex;align-items:center;justify-content:space-between;gap:12px}
h1{font-size:17px;margin:0}.stats{font-size:13px;color:#d9e2dc}.app{display:grid;grid-template-rows:auto 1fr;height:calc(100vh - 64px)}
.toolbar{display:flex;gap:10px;align-items:center;padding:10px 14px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}
.toolbar input,.toolbar select,.toolbar button{font:inherit}
.toolbar input,.toolbar select{width:220px;border:1px solid var(--line);background:white;padding:6px}
main{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:#111;overflow:hidden}
.side{display:flex;flex-direction:column;background:#c9c4b8;min-width:0;overflow:hidden}
.side-label{flex:0 0 auto;padding:4px 8px;background:#1d2923;color:white;font-size:12px;text-align:center}
.viewer{flex:1 1 0;position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden;cursor:grab;background:#111;min-height:0}
.viewer:active{cursor:grabbing}
canvas{max-width:100%;max-height:100%;background:#111;box-shadow:0 8px 30px #0005;display:block;transition:transform .08s linear;transform-origin:center center}
.zoom-hint{position:absolute;bottom:8px;left:8px;background:rgba(0,0,0,.6);color:#fff;padding:4px 8px;border-radius:3px;font-size:11px;pointer-events:none}
.side-bottom{flex:0 0 220px;background:var(--panel);border-top:1px solid var(--line);padding:8px 10px;display:flex;flex-direction:column;gap:6px;overflow:hidden;min-height:0}
.filename{font-weight:700;font-size:13px;word-break:break-all;flex:0 0 auto}
.row{display:flex;gap:6px;align-items:center;flex-wrap:wrap;flex:0 0 auto}
.btn{border:0;padding:6px 10px;color:white;cursor:pointer;font-size:12px}.neutral{background:#59645d}.keepBtn{background:var(--green)}.removeBtn{background:var(--red)}.uncertainBtn{background:var(--amber)}.exportBtn{background:var(--blue)}
.boxes{flex:1 1 0;overflow:auto;border:1px solid var(--line);min-height:0}
.boxrow{display:grid;grid-template-columns:38px 1fr 180px;gap:6px;align-items:center;padding:4px 6px;border-bottom:1px solid #e4ded0;font-size:11px}.boxrow:last-child{border-bottom:0}
.boxactions button{padding:3px 5px;border:1px solid var(--line);background:white;cursor:pointer;font-size:10px}.boxactions button.on{color:white}.boxactions .k.on{background:var(--green)}.boxactions .r.on{background:var(--red)}.boxactions .u.on{background:var(--amber)}
.hint{font-size:11px;color:#625f57}
.box-popup{position:fixed;background:var(--panel);border:1px solid var(--line);padding:8px;border-radius:4px;box-shadow:0 4px 12px #0003;z-index:100;display:grid;gap:6px;min-width:150px}.box-popup.hidden{display:none}.box-popup-title{font-size:12px;font-weight:700}.box-popup-actions{display:flex;gap:4px;flex-wrap:wrap}.box-popup-actions button{font-size:11px;padding:5px 8px}
</style></head><body><header><h1>{{TITLE}}</h1><div class="stats" id="stats"></div><div class="row"><button class="btn exportBtn" onclick="exportReview()">导出 JSONL（增强侧）</button><button class="btn neutral" onclick="clearReview()">清空进度</button></div></header>
<div class="app">
<div class="toolbar">
<input id="search" placeholder="搜索文件名" oninput="renderList()">
<select id="statusFilter" onchange="renderList()"><option value="all" selected>全部状态</option><option value="pending">未审查</option><option value="keep">保留</option><option value="remove">删除</option><option value="uncertain">存疑</option></select>
<select id="confPreset" onchange="renderList()"><option value="all" selected>全部置信度</option><option value="high">高置信度 (≥0.70)</option><option value="medium">中置信度 (0.40~0.70)</option><option value="low">低置信度 (<0.40)</option></select>
<button class="btn neutral" onclick="move(-1)">上一张 A</button>
<button class="btn neutral" onclick="move(1)">下一张 D</button>
<button class="btn keepBtn" onclick="markImage('keep')">整图保留 K</button>
<button class="btn removeBtn" onclick="markImage('remove')">整图删除 R</button>
<button class="btn uncertainBtn" onclick="markImage('uncertain')">整图存疑 U</button>
<span class="hint">左侧=原图（只读参考）；右侧=增强图（可决策）</span>
</div>
<main>
<div class="side"><div class="side-label">原图 · 现有伪标签（只读）</div><div class="viewer" id="viewer-raw"><canvas id="canvas-raw"></canvas><div class="zoom-hint">滚轮缩放 / 拖拽平移 / 双击重置</div></div><div class="side-bottom"><div class="filename" id="filename-raw"></div><div class="row"><button class="btn neutral" onclick="zoomOut('raw')">- 缩小</button><button class="btn neutral" onclick="zoomReset('raw')">重置 0</button><button class="btn neutral" onclick="zoomIn('raw')">+ 放大</button><span id="zoomInfo-raw" style="font-size:12px;color:#625f57;min-width:40px;text-align:center">100%</span></div><div class="boxes" id="boxes-raw"></div></div></div>
<div class="side"><div class="side-label">增强图 · 新伪标签（可决策）</div><div class="viewer" id="viewer-enhanced"><canvas id="canvas-enhanced"></canvas><div class="zoom-hint">滚轮缩放 / 拖拽平移 / 双击重置</div></div><div class="side-bottom"><div class="filename" id="filename-enhanced"></div><div class="row"><button class="btn neutral" onclick="zoomOut('enhanced')">- 缩小</button><button class="btn neutral" onclick="zoomReset('enhanced')">重置 0</button><button class="btn neutral" onclick="zoomIn('enhanced')">+ 放大</button><span id="zoomInfo-enhanced" style="font-size:12px;color:#625f57;min-width:40px;text-align:center">100%</span></div><div class="boxes" id="boxes-enhanced"></div></div></div>
</main>
</div>
<div id="boxPopup" class="box-popup hidden"><div id="popupTitle" class="box-popup-title"></div><div class="box-popup-actions" id="popupActions"></div></div>
<script>
const ITEMS={{ITEMS}};const CLASSES={{CLASSES}};const KEY={{STORAGE_KEY}};let states=JSON.parse(localStorage.getItem(KEY)||'{}');let filtered=[];let current=0;const colors=['#ff5252','#4fc3f7','#ffd54f','#81c784','#ba68c8','#ff8a65','#90a4ae','#f06292','#aed581','#4db6ac','#7986cb','#dce775','#fff176','#a1887f','#64b5f6','#e57373','#9575cd','#4dd0e1'];
function stateFor(name){return states[name]||(states[name]={imageStatus:'',boxes:{},note:''})}function persist(){localStorage.setItem(KEY,JSON.stringify(states));updateStats()}
function effectiveRaw(item,i){return item.raw_boxes[i][6]||'pending'}
function effectiveEnhanced(item,i){const s=stateFor(item.name);return s.boxes[i]||s.imageStatus||item.enhanced_boxes[i][6]||'pending'}
function boxMatches(side,item,i){const b=side==='raw'?item.raw_boxes[i]:item.enhanced_boxes[i];const preset=document.getElementById('confPreset').value;const conf=b[5]===null?0:b[5];if(preset==='high'&&conf<0.70)return false;if(preset==='medium'&&(conf<0.40||conf>=0.70))return false;if(preset==='low'&&conf>=0.40)return false;return true}
function getVisibleBoxes(side,item){const arr=[];const boxes=side==='raw'?item.raw_boxes:item.enhanced_boxes;for(let i=0;i<boxes.length;i++)if(boxMatches(side,item,i))arr.push(i);return arr}
function renderList(){const q=document.getElementById('search').value.toLowerCase(),f=document.getElementById('statusFilter').value;filtered=ITEMS.filter(x=>{const rawMatch=x.raw_boxes.some((_,i)=>boxMatches('raw',x,i));const enhMatch=x.enhanced_boxes.some((_,i)=>boxMatches('enhanced',x,i));return x.name.toLowerCase().includes(q)&&(f==='all'||(stateFor(x.name).imageStatus||'pending')===f)&&(document.getElementById('confPreset').value==='all'||rawMatch||enhMatch)});if(current>=filtered.length)current=Math.max(0,filtered.length-1);show()}
function show(){if(!filtered.length){document.getElementById('filename-raw').textContent='无匹配图片';document.getElementById('filename-enhanced').textContent='';return}const item=filtered[current];document.getElementById('filename-raw').textContent=`${current+1}/${filtered.length} ${item.name} [原图框 ${item.raw_boxes.length}]`;document.getElementById('filename-enhanced').textContent=`[增强图框 ${item.enhanced_boxes.length}]`;drawSide('raw',item);drawSide('enhanced',item);updateStats()}
function drawSide(side,item){const c=document.getElementById('canvas-'+side);const img=new Image();img.onload=()=>{const ctx=c.getContext('2d');c.width=img.naturalWidth;c.height=img.naturalHeight;ctx.drawImage(img,0,0);ctx.lineWidth=Math.max(2,img.naturalWidth/1000*2);ctx.font=`${Math.max(16,img.naturalWidth/120)}px sans-serif`;let visible=getVisibleBoxes(side,item);const cp=document.getElementById('confPreset').value;if(cp==='low'&&side==='enhanced')visible.sort((a,b)=>(item.enhanced_boxes[a][5]??0)-(item.enhanced_boxes[b][5]??0));visible.forEach(i=>{const b=side==='raw'?item.raw_boxes[i]:item.enhanced_boxes[i],cls=b[0],cx=b[1],cy=b[2],w=b[3],h=b[4],x=(cx-w/2)*c.width,y=(cy-h/2)*c.height,bw=w*c.width,bh=h*c.height,decision=side==='raw'?effectiveRaw(item,i):effectiveEnhanced(item,i);let stroke;if(cp==='low')stroke='#ff2a2a';else if(decision==='remove'||decision==='rejected')stroke='#666';else if(decision==='uncertain')stroke='#ffb300';else stroke=colors[cls%colors.length];ctx.strokeStyle=stroke;ctx.strokeRect(x,y,bw,bh);ctx.fillStyle=stroke;ctx.fillText(`${i}:${CLASSES[cls]||cls}`,x,Math.max(18,y-4))});resetZoom(side)};img.src=side==='raw'?item.raw_image:item.enhanced_image;renderBoxes(side,item)}
function renderBoxes(side,item){const root=document.getElementById('boxes-'+side);root.innerHTML='';const visible=getVisibleBoxes(side,item);const readOnly=side==='raw';if(!visible.length){root.innerHTML='<div class="boxrow"><span></span><span>当前筛选条件下无框</span></div>';return}visible.forEach(i=>{const b=side==='raw'?item.raw_boxes[i]:item.enhanced_boxes[i];const d=document.createElement('div');d.className='boxrow';const cur=readOnly?effectiveRaw(item,i):effectiveEnhanced(item,i),conf=b[5];let actions='';if(!readOnly){actions=`<span class="boxactions"><button class="k ${cur==='keep'||cur==='accepted'?'on':''}" onclick="markBox(${i},'keep')">保留</button><button class="r ${cur==='remove'||cur==='rejected'?'on':''}" onclick="markBox(${i},'remove')">删除</button><button class="u ${cur==='uncertain'?'on':''}" onclick="markBox(${i},'uncertain')">存疑</button></span>`}d.innerHTML=`<span>#${i}</span><span>${CLASSES[b[0]]||b[0]} | conf=${conf===null?'无':conf.toFixed(3)} | ${cur}</span>${actions}`;root.appendChild(d)})}
function markImage(v){if(!filtered.length)return;stateFor(filtered[current].name).imageStatus=v;persist();show();move(1)}
function markBox(i,v){if(!filtered.length)return;const item=filtered[current];if(v==='')delete stateFor(item.name).boxes[i];else stateFor(item.name).boxes[i]=v;persist();show()}
function move(delta){if(!filtered.length)return;current=Math.max(0,Math.min(filtered.length-1,current+delta));show()}
function updateStats(){let reviewed=0,k=0,r=0,u=0;ITEMS.forEach(x=>{const s=states[x.name];if(s&&s.imageStatus){reviewed++;if(s.imageStatus==='keep')k++;if(s.imageStatus==='remove')r++;if(s.imageStatus==='uncertain')u++}});document.getElementById('stats').textContent=`已审查 ${reviewed}/${ITEMS.length} | 保留 ${k} 删除 ${r} 存疑 ${u}`}
let popupBoxIdx=null,popupSide=null;
function clickToBox(e,side){const c=document.getElementById('canvas-'+side);const rect=c.getBoundingClientRect();if(!rect.width||!c.width)return -1;const x=(e.clientX-rect.left)*(c.width/rect.width);const y=(e.clientY-rect.top)*(c.height/rect.height);const item=filtered[current];const boxes=side==='raw'?item.raw_boxes:item.enhanced_boxes;const visible=getVisibleBoxes(side,item);let best=-1,bestArea=Infinity;for(let vi=visible.length-1;vi>=0;vi--){const i=visible[vi];const b=boxes[i];const x1=(b[1]-b[3]/2)*c.width,y1=(b[2]-b[4]/2)*c.height,x2=(b[1]+b[3]/2)*c.width,y2=(b[2]+b[4]/2)*c.height;if(x>=x1&&x<=x2&&y>=y1&&y<=y2){const area=(x2-x1)*(y2-y1);if(area<bestArea){bestArea=area;best=i;}}}return best;}
function openBoxPopup(e,side){const i=clickToBox(e,side);if(i<0){closeBoxPopup();return}popupBoxIdx=i;popupSide=side;const item=filtered[current],b=side==='raw'?item.raw_boxes[i]:item.enhanced_boxes[i];const p=document.getElementById('boxPopup');document.getElementById('popupTitle').textContent=`${side==='raw'?'原图':'增强'} #${i} ${CLASSES[b[0]]||b[0]} conf=${b[5]===null?'无':b[5].toFixed(3)}`;const actions=document.getElementById('popupActions');actions.innerHTML='';if(side==='enhanced'){actions.innerHTML='<button class="btn keepBtn" onclick="popupMark(\'keep\')">保留</button><button class="btn removeBtn" onclick="popupMark(\'remove\')">删除</button><button class="btn uncertainBtn" onclick="popupMark(\'uncertain\')">存疑</button><button class="btn neutral" onclick="popupMark(\'\')">清除</button>';}else{actions.innerHTML='<span class="hint">原图侧只读</span>';}p.classList.remove('hidden');let px=e.clientX+10,py=e.clientY+10;const pw=p.offsetWidth||150,ph=p.offsetHeight||80;const ww=window.innerWidth,wh=window.innerHeight;if(px+pw>ww)px=ww-pw-10;if(py+ph>wh)py=wh-ph-10;p.style.left=px+'px';p.style.top=py+'px';}
function popupMark(v){if(popupBoxIdx===null||popupSide!=='enhanced')return;markBox(popupBoxIdx,v);closeBoxPopup();}
function closeBoxPopup(){const p=document.getElementById('boxPopup');if(p){p.classList.add('hidden');popupBoxIdx=null;popupSide=null;}}
function exportReview(){const lines=ITEMS.map(item=>{const s=stateFor(item.name);return JSON.stringify({image_name:item.name,image_status:s.imageStatus||'pending',box_decisions:item.enhanced_boxes.map((b,i)=>({index:i,class_id:b[0],class_name:CLASSES[b[0]]||String(b[0]),confidence:b[5],default_status:b[6],decision:effectiveEnhanced(item,i),xywh:b.slice(1,5)})),note:s.note||''})}).join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines],{type:'application/jsonl'}));a.download='{{EXPORT_NAME}}';a.click()}
function clearReview(){if(confirm('确认清空本浏览器中的全部审查进度？')){states={};persist();renderList()}}
const zoom={raw:{scale:1,tx:0,ty:0,dragging:false,hasDragged:false},enhanced:{scale:1,tx:0,ty:0,dragging:false,hasDragged:false}};
function applyZoom(side){const c=document.getElementById('canvas-'+side);if(!c)return;c.style.transform='translate('+zoom[side].tx+'px,'+zoom[side].ty+'px) scale('+zoom[side].scale+')';const z=document.getElementById('zoomInfo-'+side);if(z)z.textContent=Math.round(zoom[side].scale*100)+'%'}function resetZoom(side){zoom[side].scale=1;zoom[side].tx=0;zoom[side].ty=0;applyZoom(side)}function zoomIn(side){zoom[side].scale=Math.min(zoom[side].scale*1.25,10);applyZoom(side)}function zoomOut(side){zoom[side].scale=Math.max(zoom[side].scale/1.25,0.2);if(zoom[side].scale<=1){zoom[side].tx=0;zoom[side].ty=0}applyZoom(side)}
function setupZoom(side){const v=document.getElementById('viewer-'+side);if(!v)return;v.addEventListener('wheel',e=>{e.preventDefault();e.deltaY<0?zoomIn(side):zoomOut(side)},{passive:false});v.addEventListener('mousedown',e=>{if(e.button!==0)return;zoom[side].dragging=true;zoom[side].hasDragged=false;zoom[side].dragX=e.clientX;zoom[side].dragY=e.clientY;zoom[side].dragTx=zoom[side].tx;zoom[side].dragTy=zoom[side].ty;v.style.cursor='grabbing'});window.addEventListener('mousemove',e=>{if(!zoom[side].dragging)return;const dx=e.clientX-zoom[side].dragX,dy=e.clientY-zoom[side].dragY;if(Math.abs(dx)>2||Math.abs(dy)>2)zoom[side].hasDragged=true;zoom[side].tx=zoom[side].dragTx+dx;zoom[side].ty=zoom[side].dragTy+dy;applyZoom(side)});window.addEventListener('mouseup',()=>{if(!zoom[side].dragging)return;zoom[side].dragging=false;v.style.cursor='grab'});v.addEventListener('dblclick',()=>resetZoom(side));const c=document.getElementById('canvas-'+side);if(c)c.addEventListener('click',e=>{if(zoom[side].hasDragged)return;openBoxPopup(e,side)});}
document.addEventListener('click',e=>{const p=document.getElementById('boxPopup');if(p&&!p.classList.contains('hidden')&&!p.contains(e.target)&&!e.target.id.startsWith('canvas'))closeBoxPopup()});
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;const k=e.key.toLowerCase();if(k==='a')move(-1);if(k==='d')move(1);if(k==='k')markImage('keep');if(k==='r')markImage('remove');if(k==='u')markImage('uncertain');});
setupZoom('raw');setupZoom('enhanced');renderList();
</script></body></html>'''


def read_boxes(label_path: Path) -> list[list]:
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) in (5, 6):
            confidence = float(fields[5]) if len(fields) == 6 else None
            status = (
                "accepted" if confidence is not None and confidence >= 0.70
                else "uncertain" if confidence is not None and confidence >= 0.40
                else "rejected" if confidence is not None
                else "pending"
            )
            boxes.append([int(fields[0]), *[float(v) for v in fields[1:5]], confidence, status])
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="生成「原图 vs 增强图」对照审查页")
    parser.add_argument("--raw-root", type=Path, required=True, help="原图数据集根目录")
    parser.add_argument("--enhanced-root", type=Path, required=True, help="增强图数据集根目录")
    parser.add_argument("--output-root", type=Path, default=Path(r"E:/PythonProject/优化/pseudo_label_reviews"))
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    enhanced_root = args.enhanced_root.resolve()
    output = args.output_root / f"paired_{raw_root.name}_vs_{enhanced_root.name}"
    output.mkdir(parents=True, exist_ok=True)

    classes_src = enhanced_root / "classes.txt" if (enhanced_root / "classes.txt").is_file() else raw_root / "classes.txt"
    classes = classes_src.read_text(encoding="utf-8").splitlines() if classes_src.is_file() else []

    raw_images = {p.name: p for p in sorted((raw_root / "images").iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    enhanced_images = {p.name: p for p in sorted((enhanced_root / "images").iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    common_names = sorted(set(raw_images) & set(enhanced_images))

    items = []
    for name in common_names:
        raw_label = raw_root / "labels_with_conf" / f"{Path(name).stem}.txt"
        enhanced_label = enhanced_root / "labels_with_conf" / f"{Path(name).stem}.txt"
        raw_img = raw_images[name]
        enh_img = enhanced_images[name]
        items.append({
            "name": name,
            "raw_image": Path(os.path.relpath(raw_img, output)).as_posix(),
            "enhanced_image": Path(os.path.relpath(enh_img, output)).as_posix(),
            "raw_boxes": read_boxes(raw_label),
            "enhanced_boxes": read_boxes(enhanced_label),
        })

    title = f"对照审查：{raw_root.name} vs {enhanced_root.name}"
    html = HTML.replace("{{TITLE}}", title)
    html = html.replace("{{ITEMS}}", json.dumps(items, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("{{CLASSES}}", json.dumps(classes, ensure_ascii=False))
    html = html.replace("{{STORAGE_KEY}}", json.dumps(f"paired-review::{enhanced_root.name}"))
    html = html.replace("{{EXPORT_NAME}}", f"{enhanced_root.name}_paired_review_result.jsonl")
    (output / "review.html").write_text(html, encoding="utf-8")

    summary = {
        "raw_root": str(raw_root),
        "enhanced_root": str(enhanced_root),
        "images": len(items),
        "raw_boxes": sum(len(x["raw_boxes"]) for x in items),
        "enhanced_boxes": sum(len(x["enhanced_boxes"]) for x in items),
        "review_html": str(output / "review.html"),
    }
    (output / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
