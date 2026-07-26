#!/usr/bin/env python3
"""为扁平 YOLO 伪标签数据集生成浏览器可视化审查页。

支持可选多视图（原图 / Lab / HSV / 灰阶 / top-hat / black-hat 等），
框始终来自 labels_with_conf，不因视图切换改变标签。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from review_enhance_utils import DEFAULT_VIEW_KEYS, VIEW_CATALOG, discover_view_dirs


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{TITLE}}</title><style>
:root{--ink:#18201c;--paper:#f3efe4;--panel:#fffdf7;--line:#c8c0ae;--green:#2f6f4e;--red:#a33b32;--amber:#b07a18;--blue:#27658a}
*{box-sizing:border-box}body{margin:0;font-family:"Microsoft YaHei",sans-serif;background:var(--paper);color:var(--ink)}
header{height:64px;padding:10px 18px;background:#1d2923;color:white;display:flex;align-items:center;justify-content:space-between;gap:12px}
h1{font-size:16px;margin:0;max-width:42vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.stats{font-size:12px;color:#d9e2dc;max-width:40vw}.app{display:grid;grid-template-columns:320px 1fr;grid-template-rows:minmax(0,1fr);height:calc(100vh - 64px)}
aside{background:var(--panel);border-right:1px solid var(--line);padding:12px;display:flex;flex-direction:column;overflow:hidden}.filters{display:grid;gap:8px;margin-bottom:10px}
input,select,button,textarea{font:inherit}input,select,textarea{width:100%;border:1px solid var(--line);background:white;padding:7px}
.list{display:grid;gap:5px;overflow:auto;flex:1;min-height:80px;align-content:start;border-top:1px solid var(--line);padding-top:8px;margin-top:4px}.item{border:1px solid transparent;padding:7px;cursor:pointer;background:#f8f5ec;font-size:12px;word-break:break-all}
.item.active{border-color:var(--blue);background:#e7f1f5}.item.keep{border-left:5px solid var(--green)}.item.remove{border-left:5px solid var(--red)}.item.uncertain{border-left:5px solid var(--amber)}
main{display:grid;grid-template-rows:minmax(0,1fr) minmax(0,auto);background:#c9c4b8;min-height:0;overflow:hidden}.viewer{position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden;cursor:grab;background:#111;min-height:0;min-width:0}.viewer:active{cursor:grabbing}
canvas{max-width:100%;max-height:100%;min-width:0;min-height:0;background:#111;box-shadow:0 8px 30px #0005;display:block;transition:transform .08s linear;transform-origin:center center}.zoom-hint{position:absolute;bottom:8px;left:8px;background:rgba(0,0,0,.6);color:#fff;padding:4px 8px;border-radius:3px;font-size:11px;pointer-events:none;max-width:70%}.bottom{background:var(--panel);border-top:1px solid var(--line);padding:6px 10px;display:grid;gap:5px;overflow:auto;max-height:46vh}
.row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.btn{border:0;padding:7px 10px;color:white;cursor:pointer;font-size:12px}.neutral{background:#59645d}.keepBtn{background:var(--green)}.removeBtn{background:var(--red)}.uncertainBtn{background:var(--amber)}.exportBtn{background:var(--blue)}.viewBtn{background:#3d4f46;padding:5px 8px;font-size:11px}.viewBtn.on{background:var(--blue);outline:2px solid #9fd0ff}
.boxes{max-height:26vh;min-height:120px;overflow:auto;border:1px solid var(--line)}.boxrow{display:grid;grid-template-columns:44px 1fr 320px;gap:6px;align-items:center;padding:4px 8px;border-bottom:1px solid #e4ded0;font-size:12px;cursor:pointer}.boxrow:last-child{border-bottom:0}.boxrow.focused{background:#e7f1f5}
.boxactions button{padding:4px 6px;border:1px solid var(--line);background:white;cursor:pointer;font-size:11px}.boxactions button.on{color:white}.boxactions .k.on{background:var(--green)}.boxactions .r.on{background:var(--red)}.boxactions .u.on{background:var(--amber)}.boxactions .f{background:#eef6ff}.boxactions .o{background:#fff3e0}.editedTag{color:#00b0ff;font-weight:700;margin-left:3px}#editBtn.on{background:var(--blue)}
.filename{font-weight:700;word-break:break-all}.hint{font-size:12px;color:#625f57}textarea{min-height:28px;max-height:54px;resize:vertical}
.box-popup{position:fixed;background:var(--panel);border:1px solid var(--line);padding:8px;border-radius:4px;box-shadow:0 4px 12px #0003;z-index:100;display:grid;gap:6px;min-width:160px}.box-popup.hidden{display:none}.box-popup-title{font-size:12px;font-weight:700}.box-popup-actions{display:flex;gap:4px;flex-wrap:wrap}.box-popup-actions button{font-size:11px;padding:5px 8px}
.visibility-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;border:1px solid var(--line);padding:8px;background:#f8f5ec}
.visibility-grid label{display:grid;gap:2px;font-size:11px}.visibility-grid select{padding:4px;font-size:12px}
.meta{font-size:11px;color:#625f57;line-height:1.4}.view-help{font-size:11px;color:#3d4f46;background:#eef6f0;border:1px solid #c5d6cb;padding:6px;line-height:1.35}
#viewSwitch{max-height:120px;overflow:auto;align-content:flex-start}
</style></head><body><header><h1>{{TITLE}}</h1><div class="stats" id="stats"></div><div class="row"><button class="btn exportBtn" onclick="exportReview()">导出 JSONL</button><button class="btn neutral" onclick="document.getElementById('importFile').click()">导入进度</button><input type="file" id="importFile" accept=".jsonl,.json" style="display:none" onchange="importReview(event)"><button class="btn neutral" onclick="clearReview()">清空进度</button></div></header>
<div class="app"><aside><div class="filters"><input id="search" placeholder="搜索文件名" oninput="renderList()"><select id="statusFilter" onchange="renderList()"><option value="all" selected>全部状态</option><option value="pending">未审查</option><option value="keep">保留</option><option value="remove">删除</option><option value="uncertain">存疑</option></select><select id="confPreset" onchange="renderList()"><option value="all" selected>全部置信度</option><option value="high">高置信度 (≥0.70)</option><option value="medium">中置信度 (0.40~0.70)</option><option value="low">低置信度 (<0.40)</option></select>
<div id="viewPanel" style="display:none">
<label class="hint">显示视图（框不变）</label>
<select id="viewSelect" onchange="setView(this.value)"></select>
<div id="viewSwitch" class="row"></div>
<div class="view-help" id="viewHelp"></div>
<div class="hint">快捷键：Q/E 切换视图 · F 聚焦当前框 · 1-9 跳转前 9 个视图</div>
</div>
<div id="visibilityPanel" class="visibility-grid" style="display:none">
<label>原图能否判定<select id="visRaw" onchange="saveVisibility()"><option value="">未填</option><option value="yes">能</option><option value="no">不能</option></select></label>
<label>最佳增强能否判定<select id="visBest" onchange="saveVisibility()"><option value="">未填</option><option value="yes">能</option><option value="no">不能</option></select></label>
<label>框位置是否合理<select id="visAlign" onchange="saveVisibility()"><option value="">未填</option><option value="yes">对准</option><option value="no">偏了/空框</option><option value="unsure">不确定</option></select></label>
<label>最有用视图<select id="visBestView" onchange="saveVisibility()"><option value="">未填</option></select></label>
</div>
<div class="meta" id="imageMeta"></div>
</div><div class="list" id="list"></div></aside>
<main><div class="viewer" id="viewer"><canvas id="canvas"></canvas><div class="zoom-hint">滚轮缩放 / 拖拽平移 / 双击重置 · 点框行自动聚焦 · 按 B 改框：框内拖动=移动，拖角/拖边=缩放</div></div><div class="bottom"><div class="filename" id="filename"></div><div class="row"><button class="btn neutral" onclick="move(-1)">上一张 A</button><button class="btn neutral" onclick="move(1)">下一张 D</button><button class="btn keepBtn" onclick="markImage('keep')">整图保留 K</button><button class="btn removeBtn" onclick="markImage('remove')">整图删除 R</button><button class="btn uncertainBtn" onclick="markImage('uncertain')">整图存疑 U</button><span class="hint">框来自 labels_with_conf；位置错按 B 进改框模式拖拽修正，修正坐标随导出保存。</span></div><div class="row"><button class="btn neutral" onclick="zoomOut()">- 缩小</button><button class="btn neutral" onclick="zoomReset()">重置 0</button><button class="btn neutral" onclick="zoomIn()">+ 放大</button><button class="btn neutral" onclick="focusCurrentBox()">聚焦框 F</button><button class="btn neutral" id="editBtn" onclick="toggleEdit()">改框 B</button><span id="zoomInfo" style="font-size:13px;color:#625f57;min-width:45px;text-align:center">100%</span><span class="hint" id="viewHint"></span></div><div class="boxes" id="boxes"></div><textarea id="note" placeholder="审查备注（如：假纹理 / 框偏右 / 边缘可见）" oninput="saveNote()"></textarea></div></main><div id="boxPopup" class="box-popup hidden"><div id="popupTitle" class="box-popup-title"></div><div class="box-popup-actions"><button class="btn keepBtn" onclick="popupMark('keep')">保留</button><button class="btn removeBtn" onclick="popupMark('remove')">删除</button><button class="btn uncertainBtn" onclick="popupMark('uncertain')">存疑</button><button class="btn neutral" onclick="focusPopupBox()">聚焦</button><button class="btn neutral" onclick="popupMark('')">清除</button></div></div></div>
<script>
const ITEMS={{ITEMS}};const CLASSES={{CLASSES}};const KEY={{STORAGE_KEY}};const HAS_VIEWS={{HAS_VIEWS}};const VIEW_DEFS={{VIEW_DEFS}};
let states=JSON.parse(localStorage.getItem(KEY)||'{}');let filtered=[];let current=0;let currentView='raw';let focusedBox=null;let visibleBoxes=[];let popupBoxIdx=null;
const colors=['#ff5252','#4fc3f7','#ffd54f','#81c784','#ba68c8','#ff8a65','#90a4ae','#f06292','#aed581','#4db6ac','#7986cb','#dce775','#fff176','#a1887f','#64b5f6','#e57373','#9575cd','#4dd0e1'];
function viewLabel(k){const d=VIEW_DEFS.find(x=>x.key===k);return d?d.label:k}
function viewHelp(k){const d=VIEW_DEFS.find(x=>x.key===k);return d?d.help:''}
function stateFor(name){if(!states[name])states[name]={imageStatus:'',boxes:{},note:'',visibility:{raw:'',best:'',align:'',best_view:''},boxCoords:{}};if(!states[name].visibility)states[name].visibility={raw:'',best:'',align:'',best_view:''};if(!states[name].boxCoords)states[name].boxCoords={};return states[name]}
function boxXYWH(item,i){const e=stateFor(item.name).boxCoords[i];return e?e:item.boxes[i].slice(1,5)}
function persist(){localStorage.setItem(KEY,JSON.stringify(states));updateStats()}
function effective(item,i){const s=stateFor(item.name);return s.boxes[i]||s.imageStatus||item.boxes[i][6]||'pending'}
function imageSrc(item){if(!HAS_VIEWS)return item.image;const v=item.views||{};if(v[currentView])return v[currentView];if(currentView==='raw')return item.image||v.raw;return item.image||v.raw||Object.values(v)[0]}
function setupViews(){if(!HAS_VIEWS||!VIEW_DEFS.length)return;document.getElementById('viewPanel').style.display='grid';document.getElementById('visibilityPanel').style.display='grid';const sel=document.getElementById('viewSelect');const sw=document.getElementById('viewSwitch');const best=document.getElementById('visBestView');sel.innerHTML='';sw.innerHTML='';best.innerHTML='<option value="">未填</option>';VIEW_DEFS.forEach((d,i)=>{const o=document.createElement('option');o.value=d.key;o.textContent=`${i+1}. ${d.label}`;sel.appendChild(o);const b=document.createElement('button');b.className='btn viewBtn'+(i===0?' on':'');b.dataset.view=d.key;b.textContent=d.label;b.title=d.help||'';b.onclick=()=>setView(d.key);sw.appendChild(b);const bo=document.createElement('option');bo.value=d.key;bo.textContent=d.label;best.appendChild(bo)});currentView=VIEW_DEFS[0].key;document.getElementById('viewHelp').textContent=viewHelp(currentView);document.getElementById('viewHint').textContent='当前视图：'+viewLabel(currentView)}
function setView(v){if(!HAS_VIEWS)return;if(!VIEW_DEFS.some(d=>d.key===v))return;currentView=v;document.getElementById('viewSelect').value=v;document.querySelectorAll('#viewSwitch .viewBtn').forEach(b=>b.classList.toggle('on',b.dataset.view===v));document.getElementById('viewHelp').textContent=viewHelp(v);document.getElementById('viewHint').textContent='当前视图：'+viewLabel(v);show(false)}
function cycleView(delta){if(!HAS_VIEWS||!VIEW_DEFS.length)return;const i=VIEW_DEFS.findIndex(d=>d.key===currentView);const n=(i+delta+VIEW_DEFS.length)%VIEW_DEFS.length;setView(VIEW_DEFS[n].key)}
function renderList(){const q=document.getElementById('search').value.toLowerCase(),f=document.getElementById('statusFilter').value,cp=document.getElementById('confPreset').value;filtered=ITEMS.filter(x=>{const st=stateFor(x.name).imageStatus||'pending';return x.name.toLowerCase().includes(q)&&(f==='all'||st===f)&&(cp==='all'||x.boxes.some((_,i)=>boxMatches(x,i)))});const list=document.getElementById('list');list.innerHTML='';filtered.forEach((x,i)=>{const st=stateFor(x.name).imageStatus||'pending';const d=document.createElement('div');d.className='item '+st+(i===current?' active':'');d.textContent=`${i+1}. ${x.name} [${x.boxes.length}]`;d.onclick=()=>{current=i;focusedBox=null;show()};list.appendChild(d)});if(current>=filtered.length)current=Math.max(0,filtered.length-1);show()}
let currentImg=null;
function drawScene(){const img=currentImg;if(!img||!filtered.length)return;const item=filtered[current];const c=document.getElementById('canvas'),ctx=c.getContext('2d');c.width=img.naturalWidth;c.height=img.naturalHeight;ctx.drawImage(img,0,0);ctx.lineWidth=Math.max(2,img.naturalWidth/1000*2);ctx.font=`${Math.max(18,img.naturalWidth/100)}px sans-serif`;const cp=document.getElementById('confPreset').value;const rc0=c.getBoundingClientRect(),invK=rc0.width?c.width/rc0.width:1;visibleBoxes.forEach(i=>{const b=item.boxes[i],cls=b[0],[cx,cy,w,h]=boxXYWH(item,i),x=(cx-w/2)*c.width,y=(cy-h/2)*c.height,bw=w*c.width,bh=h*c.height,col=colors[cls%colors.length],decision=effective(item,i),edited=Boolean(stateFor(item.name).boxCoords[i]);let stroke=cp==='low'?'#ff2a2a':(decision==='remove'?'#666':decision==='uncertain'?'#ffb300':col);if(focusedBox===i){ctx.lineWidth=Math.max(1,1.5*invK);stroke='#00e5ff'}else{ctx.lineWidth=Math.max(2,img.naturalWidth/1000*2)}ctx.strokeStyle=stroke;ctx.setLineDash(edited?[8,5]:[]);ctx.strokeRect(x,y,bw,bh);ctx.setLineDash([]);ctx.fillStyle=stroke;ctx.fillText(`${i}:${CLASSES[cls]||cls}${edited?'*':''}`,x,Math.max(20,y-4));if(editMode&&focusedBox===i){const hs=[[x,y],[x+bw/2,y],[x+bw,y],[x+bw,y+bh/2],[x+bw,y+bh],[x+bw/2,y+bh],[x,y+bh],[x,y+bh/2]];const r=5.5*invK;ctx.fillStyle='#00e5ff';ctx.strokeStyle='#083038';ctx.lineWidth=Math.max(1,1.2*invK);hs.forEach(p=>{ctx.fillRect(p[0]-r,p[1]-r,2*r,2*r);ctx.strokeRect(p[0]-r,p[1]-r,2*r,2*r)})}})}
function show(resetZ=true){if(!filtered.length){document.getElementById('filename').textContent='无匹配图片';return}const item=filtered[current],s=stateFor(item.name);document.getElementById('filename').textContent=`${current+1}/${filtered.length} ${item.name}`;document.getElementById('note').value=s.note||'';if(HAS_VIEWS){document.getElementById('visRaw').value=s.visibility.raw||'';document.getElementById('visBest').value=s.visibility.best||'';document.getElementById('visAlign').value=s.visibility.align||'';document.getElementById('visBestView').value=s.visibility.best_view||'';const meta=[];if(item.mean_brightness!=null)meta.push('亮度均值 '+item.mean_brightness);if(item.brightness_bucket)meta.push('亮度档 '+item.brightness_bucket);if(item.conf_bucket)meta.push('低conf档 '+item.conf_bucket);if(item.area_bucket)meta.push('框面积档 '+item.area_bucket);if(item.stratum)meta.push('分层 '+item.stratum);if(item.low_box_count!=null)meta.push('低conf框 '+item.low_box_count);document.getElementById('imageMeta').textContent=meta.join(' · ')}
visibleBoxes=getVisibleBoxes(item);const cp=document.getElementById('confPreset').value;if(cp==='low')visibleBoxes.sort((a,b)=>(item.boxes[a][5]??0)-(item.boxes[b][5]??0));if(focusedBox!=null&&!visibleBoxes.includes(focusedBox))focusedBox=visibleBoxes[0]??null;const img=new Image();img.onload=()=>{currentImg=img;drawScene();if(resetZ){if(focusedBox!=null)focusBox(focusedBox,false);else resetZoom()}else applyZoom()};img.onerror=()=>{document.getElementById('filename').textContent+=` [视图加载失败: ${currentView}]`};img.src=imageSrc(item);renderBoxes(item);renderListOnly();updateStats()}
function renderListOnly(){[...document.querySelectorAll('.item')].forEach((e,i)=>e.classList.toggle('active',i===current))}
function renderBoxes(item){const root=document.getElementById('boxes');root.innerHTML='';if(!visibleBoxes.length){root.innerHTML='<div class="boxrow"><span></span><span>当前筛选条件下无框，请调整左侧筛选条件</span></div>';return}visibleBoxes.forEach(i=>{const b=item.boxes[i];const d=document.createElement('div');d.className='boxrow'+(focusedBox===i?' focused':'');d.onclick=e=>{if(e.target.tagName!=='BUTTON')focusBox(i)};const cur=effective(item,i),conf=b[5],[bx,by,bw2,bh2]=boxXYWH(item,i),edited=Boolean(stateFor(item.name).boxCoords[i]);d.innerHTML=`<span>#${i}${edited?'<span class="editedTag">改</span>':''}</span><span>${CLASSES[b[0]]||b[0]} | conf=${conf===null?'无':conf.toFixed(3)} | 默认=${b[6]} | x=${bx.toFixed(3)} y=${by.toFixed(3)} w=${bw2.toFixed(3)} h=${bh2.toFixed(3)}</span><span class="boxactions"><button class="f" onclick="focusBox(${i})">聚焦</button><button class="k ${cur==='keep'||cur==='accepted'?'on':''}" onclick="markBox(${i},'keep')">保留</button><button class="r ${cur==='remove'||cur==='rejected'?'on':''}" onclick="markBox(${i},'remove')">删除</button><button class="u ${cur==='uncertain'?'on':''}" onclick="markBox(${i},'uncertain')">存疑</button>${edited?`<button class="o" title="恢复原始坐标" onclick="resetBoxCoords(${i})">复原</button>`:''}</span>`;root.appendChild(d)})}
function markImage(v){if(!filtered.length)return;stateFor(filtered[current].name).imageStatus=v;persist();show();move(1)}
function markBox(i,v){const item=filtered[current];if(v==='')delete stateFor(item.name).boxes[i];else stateFor(item.name).boxes[i]=v;persist();show(false)}
function saveNote(){if(!filtered.length)return;stateFor(filtered[current].name).note=document.getElementById('note').value;persist()}
function saveVisibility(){if(!filtered.length||!HAS_VIEWS)return;const s=stateFor(filtered[current].name);s.visibility={raw:document.getElementById('visRaw').value,best:document.getElementById('visBest').value,align:document.getElementById('visAlign').value,best_view:document.getElementById('visBestView').value};persist()}
function boxMatches(item,i){const b=item.boxes[i];const preset=document.getElementById('confPreset').value;const conf=b[5]===null?0:b[5];if(preset==='high'&&conf<0.70)return false;if(preset==='medium'&&(conf<0.40||conf>=0.70))return false;if(preset==='low'&&conf>=0.40)return false;return true}
function getVisibleBoxes(item){const arr=[];for(let i=0;i<item.boxes.length;i++)if(boxMatches(item,i))arr.push(i);return arr}
function clickToBox(e){const c=document.getElementById('canvas');const rect=c.getBoundingClientRect();if(!rect.width||!c.width)return -1;const x=(e.clientX-rect.left)*(c.width/rect.width);const y=(e.clientY-rect.top)*(c.height/rect.height);const item=filtered[current];let best=-1,bestArea=Infinity;for(let vi=visibleBoxes.length-1;vi>=0;vi--){const i=visibleBoxes[vi];const [bcx,bcy,bbw,bbh]=boxXYWH(item,i);const x1=(bcx-bbw/2)*c.width,y1=(bcy-bbh/2)*c.height,x2=(bcx+bbw/2)*c.width,y2=(bcy+bbh/2)*c.height;if(x>=x1&&x<=x2&&y>=y1&&y<=y2){const area=(x2-x1)*(y2-y1);if(area<bestArea){bestArea=area;best=i;}}}return best;}
function openBoxPopup(e){const i=clickToBox(e);if(i<0){closeBoxPopup();return}popupBoxIdx=i;focusedBox=i;const item=filtered[current],b=item.boxes[i];const p=document.getElementById('boxPopup');document.getElementById('popupTitle').textContent=`#${i} ${CLASSES[b[0]]||b[0]} conf=${b[5]===null?'无':b[5].toFixed(3)}`;p.classList.remove('hidden');let px=e.clientX+10,py=e.clientY+10;const pw=p.offsetWidth||150,ph=p.offsetHeight||80;const ww=window.innerWidth,wh=window.innerHeight;if(px+pw>ww)px=ww-pw-10;if(py+ph>wh)py=wh-ph-10;p.style.left=px+'px';p.style.top=py+'px';show(false)}
function popupMark(v){if(popupBoxIdx===null)return;markBox(popupBoxIdx,v);closeBoxPopup()}
function focusPopupBox(){if(popupBoxIdx===null)return;focusBox(popupBoxIdx);closeBoxPopup()}
function closeBoxPopup(){const p=document.getElementById('boxPopup');if(p){p.classList.add('hidden');popupBoxIdx=null}}
let editMode=false,editDrag=null;
function toggleEdit(){editMode=!editMode;editDrag=null;document.getElementById('editBtn').classList.toggle('on',editMode);const v=document.getElementById('viewer');v.style.cursor=editMode?'crosshair':'grab';show(false)}
function canvasPos(e){const c=document.getElementById('canvas');const rect=c.getBoundingClientRect();return{x:(e.clientX-rect.left)*(c.width/rect.width),y:(e.clientY-rect.top)*(c.height/rect.height),tol:10*(c.width/rect.width)}}
function boxRectPx(item,i){const [cx,cy,w,h]=boxXYWH(item,i);const c=document.getElementById('canvas');return{x1:(cx-w/2)*c.width,y1:(cy-h/2)*c.height,x2:(cx+w/2)*c.width,y2:(cy+h/2)*c.height}}
function hitHandle(i,p){const r=boxRectPx(filtered[current],i);const hs={nw:[r.x1,r.y1],n:[(r.x1+r.x2)/2,r.y1],ne:[r.x2,r.y1],e:[r.x2,(r.y1+r.y2)/2],se:[r.x2,r.y2],s:[(r.x1+r.x2)/2,r.y2],sw:[r.x1,r.y2],w:[r.x1,(r.y1+r.y2)/2]};for(const k in hs)if(Math.abs(p.x-hs[k][0])<=p.tol&&Math.abs(p.y-hs[k][1])<=p.tol)return k;return null}
function startEditDrag(e){const p=canvasPos(e);let target=null,mode='move';if(focusedBox!=null){const h=hitHandle(focusedBox,p);if(h){target=focusedBox;mode=h}}if(target==null){const hit=clickToBox(e);if(hit>=0){target=hit;focusedBox=hit}}if(target==null)return false;editDrag={i:target,mode,startX:p.x,startY:p.y,orig:boxXYWH(filtered[current],target).slice()};return true}
function applyEditDrag(e){const d=editDrag;if(!d)return;const p=canvasPos(e);const c=document.getElementById('canvas');const dx=p.x-d.startX,dy=p.y-d.startY;if(Math.abs(dx)<1&&Math.abs(dy)<1)return;const [cx,cy,w,h]=d.orig;let x1=(cx-w/2)*c.width,y1=(cy-h/2)*c.height,x2=(cx+w/2)*c.width,y2=(cy+h/2)*c.height;const m=d.mode;if(m==='move'){const bw=x2-x1,bh=y2-y1;x1=Math.max(0,Math.min(c.width-bw,x1+dx));y1=Math.max(0,Math.min(c.height-bh,y1+dy));x2=x1+bw;y2=y1+bh}else{if(m.includes('w'))x1=Math.max(0,Math.min(x2-8,x1+dx));if(m.includes('e'))x2=Math.min(c.width,Math.max(x1+8,x2+dx));if(m.includes('n'))y1=Math.max(0,Math.min(y2-8,y1+dy));if(m.includes('s'))y2=Math.min(c.height,Math.max(y1+8,y2+dy))}stateFor(filtered[current].name).boxCoords[d.i]=[+((x1+x2)/2/c.width).toFixed(6),+((y1+y2)/2/c.height).toFixed(6),+((x2-x1)/c.width).toFixed(6),+((y2-y1)/c.height).toFixed(6)];drawScene()}
function resetBoxCoords(i){delete stateFor(filtered[current].name).boxCoords[i];persist();show(false)}
function move(delta){if(!filtered.length)return;current=Math.max(0,Math.min(filtered.length-1,current+delta));focusedBox=null;show()}
function focusBox(i,redraw=true){if(!filtered.length)return;focusedBox=i;const item=filtered[current];const [fcx,fcy,fbw0,fbh0]=boxXYWH(item,i);const c=document.getElementById('canvas');const v=document.getElementById('viewer');if(!c.width||!v)return;const bw=Math.max(fbw0*c.width,8),bh=Math.max(fbh0*c.height,8);const pad=3.0;const need=Math.min(c.width/(bw*pad),c.height/(bh*pad));zoomScale=Math.max(1.8,Math.min(need,12));const scaleFit=Math.min(v.clientWidth/c.width,v.clientHeight/c.height);const drawW=c.width*scaleFit,drawH=c.height*scaleFit;zoomTx=(0.5-fcx)*drawW*zoomScale;zoomTy=(0.5-fcy)*drawH*zoomScale;applyZoom();if(redraw)show(false)}
function focusCurrentBox(){if(!visibleBoxes.length)return;if(focusedBox==null)focusedBox=visibleBoxes[0];else{const p=visibleBoxes.indexOf(focusedBox);focusedBox=visibleBoxes[(p+1)%visibleBoxes.length]}focusBox(focusedBox)}
function updateStats(){let reviewed=0,k=0,r=0,u=0,visFilled=0,rawNo=0,bestYesFromNo=0,alignNo=0;const bestViewCnt={};ITEMS.forEach(x=>{const s=states[x.name];if(s&&s.imageStatus){reviewed++;if(s.imageStatus==='keep')k++;if(s.imageStatus==='remove')r++;if(s.imageStatus==='uncertain')u++}if(s&&s.visibility&&(s.visibility.raw||s.visibility.best||s.visibility.align)){visFilled++;if(s.visibility.raw==='no'){rawNo++;if(s.visibility.best==='yes')bestYesFromNo++}if(s.visibility.align==='no')alignNo++;if(s.visibility.best_view)bestViewCnt[s.visibility.best_view]=(bestViewCnt[s.visibility.best_view]||0)+1}});let extra='';if(HAS_VIEWS&&visFilled){const lift=rawNo?Math.round(100*bestYesFromNo/rawNo):0;const top=Object.entries(bestViewCnt).sort((a,b)=>b[1]-a[1])[0];extra=` | 可见性已填 ${visFilled} | 原图不可审 ${rawNo} | 增强后可审 ${bestYesFromNo}(${lift}%) | 框位不对 ${alignNo}`+(top?` | 最常用视图 ${viewLabel(top[0])}×${top[1]}`:'')}let edited=0;Object.values(states).forEach(s=>{if(s&&s.boxCoords)edited+=Object.keys(s.boxCoords).length});document.getElementById('stats').textContent=`已审查 ${reviewed}/${ITEMS.length} | 保留 ${k} 删除 ${r} 存疑 ${u}`+(edited?` | 改坐标 ${edited}`:'')+extra}
function exportReview(){const lines=ITEMS.map(item=>{const s=stateFor(item.name);return JSON.stringify({image_name:item.name,image_status:s.imageStatus||'pending',visibility:s.visibility||{},box_decisions:item.boxes.map((b,i)=>({index:i,class_id:b[0],class_name:CLASSES[b[0]]||String(b[0]),confidence:b[5],default_status:b[6],decision:effective(item,i),xywh:boxXYWH(item,i),edited:Boolean(s.boxCoords&&s.boxCoords[i])})),note:s.note||'',mean_brightness:item.mean_brightness??null,stratum:item.stratum??null})}).join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines],{type:'application/jsonl'}));a.download='{{EXPORT_NAME}}';a.click()}
function importReview(e){const file=e.target.files[0];if(!file)return;const reader=new FileReader();reader.onload=()=>{let ok=0,skip=0;try{String(reader.result).split(/\r?\n/).forEach(line=>{line=line.trim();if(!line)return;const rec=JSON.parse(line);const name=rec.image_name;if(!name||!ITEMS.some(x=>x.name===name)){skip++;return}const s=stateFor(name);if(rec.image_status&&rec.image_status!=='pending')s.imageStatus=rec.image_status;if(rec.note)s.note=rec.note;if(rec.visibility&&typeof rec.visibility==='object')s.visibility={raw:rec.visibility.raw||'',best:rec.visibility.best||'',align:rec.visibility.align||'',best_view:rec.visibility.best_view||''};(rec.box_decisions||[]).forEach(bd=>{if(bd.decision&&['keep','remove','uncertain'].includes(bd.decision)&&bd.decision!==bd.default_status)s.boxes[bd.index]=bd.decision;if(bd.edited&&Array.isArray(bd.xywh)&&bd.xywh.length===4)s.boxCoords[bd.index]=bd.xywh});ok++});persist();renderList();alert(`导入完成：恢复 ${ok} 张图片的审查记录`+(skip?`，跳过 ${skip} 条不匹配记录`:''))}catch(err){alert('导入失败：文件格式不正确\n'+err)}};reader.readAsText(file);e.target.value=''}
function clearReview(){if(confirm('确认清空本浏览器中的全部审查进度？')){states={};persist();renderList()}}
let zoomScale=1,zoomTx=0,zoomTy=0,zoomDragging=false,hasDragged=false,zoomDragX=0,zoomDragY=0,zoomDragTx=0,zoomDragTy=0;
function applyZoom(){const c=document.getElementById('canvas');if(!c)return;c.style.transform='translate('+zoomTx+'px,'+zoomTy+'px) scale('+zoomScale+')';const z=document.getElementById('zoomInfo');if(z)z.textContent=Math.round(zoomScale*100)+'%'}
function resetZoom(){zoomScale=1;zoomTx=0;zoomTy=0;applyZoom()}function zoomIn(){zoomScale=Math.min(zoomScale*1.25,12);applyZoom()}function zoomOut(){zoomScale=Math.max(zoomScale/1.25,0.2);if(zoomScale<=1){zoomTx=0;zoomTy=0}applyZoom()}function zoomReset(){resetZoom()}
function setupZoom(){const v=document.getElementById('viewer');if(!v)return;v.addEventListener('wheel',e=>{e.preventDefault();e.deltaY<0?zoomIn():zoomOut()},{passive:false});v.addEventListener('mousedown',e=>{if(e.button!==0)return;if(editMode){if(startEditDrag(e)){e.preventDefault();hasDragged=false;return}}zoomDragging=true;hasDragged=false;zoomDragX=e.clientX;zoomDragY=e.clientY;zoomDragTx=zoomTx;zoomDragTy=zoomTy;v.style.cursor='grabbing'});window.addEventListener('mousemove',e=>{if(editDrag){applyEditDrag(e);hasDragged=true;return}if(!zoomDragging)return;const dx=e.clientX-zoomDragX,dy=e.clientY-zoomDragY;if(Math.abs(dx)>2||Math.abs(dy)>2)hasDragged=true;zoomTx=zoomDragTx+dx;zoomTy=zoomDragTy+dy;applyZoom()});window.addEventListener('mouseup',()=>{if(editDrag){editDrag=null;persist();show(false);return}if(!zoomDragging)return;zoomDragging=false;v.style.cursor=editMode?'crosshair':'grab'});v.addEventListener('dblclick',()=>{focusedBox=null;resetZoom();show(false)});const c=document.getElementById('canvas');if(c)c.addEventListener('click',e=>{if(hasDragged||editMode)return;openBoxPopup(e)})}
document.addEventListener('click',e=>{const p=document.getElementById('boxPopup');if(p&&!p.classList.contains('hidden')&&!p.contains(e.target)&&e.target.id!=='canvas')closeBoxPopup()});
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;const k=e.key.toLowerCase();if(k==='a')move(-1);if(k==='d')move(1);if(k==='k')markImage('keep');if(k==='r')markImage('remove');if(k==='u')markImage('uncertain');if(k==='q')cycleView(-1);if(k==='e')cycleView(1);if(k==='f')focusCurrentBox();if(k==='b')toggleEdit();if(k==='escape'&&editMode)toggleEdit();if(k>='1'&&k<='9'){const idx=parseInt(k,10)-1;if(HAS_VIEWS&&VIEW_DEFS[idx])setView(VIEW_DEFS[idx].key)}if(k==='+'||k==='=')zoomIn();if(k==='-'||k==='_')zoomOut();if(k==='0')zoomReset()});
setupViews();setupZoom();renderList();
</script></body></html>'''


def conf_status(confidence: float | None) -> str:
    if confidence is None:
        return "pending"
    if confidence >= 0.70:
        return "accepted"
    if confidence >= 0.40:
        return "uncertain"
    return "rejected"


def read_boxes(label_path: Path) -> list[list]:
    if not label_path.is_file():
        return []
    boxes: list[list] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) not in (5, 6):
            continue
        confidence = float(fields[5]) if len(fields) == 6 else None
        boxes.append(
            [int(fields[0]), *[float(v) for v in fields[1:5]], confidence, conf_status(confidence)]
        )
    return boxes


def find_image(images_dir: Path, stem: str) -> Path | None:
    if not images_dir.is_dir():
        return None
    for path in images_dir.iterdir():
        if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def load_meta(dataset: Path) -> dict[str, dict]:
    meta_path = dataset / "sample_meta.json"
    if not meta_path.is_file():
        return {}
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "images" in data:
        images = data["images"]
        if isinstance(images, list):
            return {row["name"]: row for row in images if "name" in row}
        if isinstance(images, dict):
            return images
    if isinstance(data, list):
        return {row["name"]: row for row in data if "name" in row}
    return {}


def build_view_defs(view_keys: list[str]) -> list[dict]:
    defs = []
    for key in view_keys:
        if key in VIEW_CATALOG:
            _suffix, label, help_text = VIEW_CATALOG[key]
        else:
            label, help_text = key, key
        defs.append({"key": key, "label": label, "help": help_text})
    return defs


def main() -> None:
    parser = argparse.ArgumentParser(description="生成扁平 YOLO 伪标签可视化审查页")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/pseudo_label_reviews"),
    )
    parser.add_argument(
        "--multi-view",
        action="store_true",
        help="启用 images_* 多视图切换（框仍用 labels_with_conf）",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="使用 file:// 绝对路径引用图片，避免符号链接在浏览器中加载失败",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=None,
        help="可选的类别显示名文件（每行一个，可带中文）；优先级高于数据集内 classes.txt",
    )
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    output = args.output_root / dataset.name
    output.mkdir(parents=True, exist_ok=True)

    use_absolute = args.absolute

    def read_classes(path: Path) -> list[str]:
        if not path.is_file():
            return []
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    classes: list[str] = []
    if args.classes_file is not None:
        classes = read_classes(args.classes_file)
    if not classes:
        for candidate in (
            dataset / "classes.txt",
            dataset / "labels_with_conf" / "classes.txt",
            dataset / "labels" / "classes.txt",
        ):
            classes = read_classes(candidate)
            if classes:
                break
    if not classes:
        yaml_path = dataset / "data.yaml"
        if yaml_path.is_file():
            match = re.search(r"^names:\s*\[(?P<names>[^\]]*)\]", yaml_path.read_text(encoding="utf-8"), re.MULTILINE)
            if match:
                classes = [name.strip().strip("'\"") for name in match.group("names").split(",") if name.strip()]
    if not classes:
        print("[warn] 未找到类别名文件，审查页将只显示类别数字 ID")

    def make_image_url(image_path: Path) -> str:
        if use_absolute:
            return image_path.resolve().as_uri()
        return Path(os.path.relpath(image_path, output)).as_posix()

    confidence_root = dataset / "labels_with_conf"
    labels_root = confidence_root if confidence_root.exists() else dataset / "labels"
    meta = load_meta(dataset)

    discovered = [k for k in discover_view_dirs(dataset) if k != "all"]
    multi_view = bool(discovered) and (
        args.multi_view
        or len(discovered) >= 2
        or (dataset / "images_raw").is_dir()
    )
    view_keys = discovered if multi_view else []

    items = []
    if multi_view:
        # 源图目录：images_raw → images/（扁平）→ images_all → 第一个视图目录；取第一个含图片文件的
        first_suffix = VIEW_CATALOG.get(view_keys[0], (view_keys[0], "", ""))[0]
        candidates = [
            dataset / "images_raw",
            dataset / "images",
            dataset / "images_all",
            dataset / f"images_{first_suffix}",
        ]
        source_dir = next(
            (
                d
                for d in candidates
                if d.is_dir()
                and any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in d.iterdir())
            ),
            None,
        )
        if source_dir is None:
            raise FileNotFoundError(f"多视图模式下找不到源图目录: {dataset}")
        image_files = sorted(
            p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        for image in image_files:
            label = labels_root / f"{image.stem}.txt"
            boxes = read_boxes(label)
            views: dict[str, str] = {}
            for key in view_keys:
                suffix = VIEW_CATALOG.get(key, (key, key, ""))[0]
                found = find_image(dataset / f"images_{suffix}", image.stem)
                if found is not None:
                    views[key] = make_image_url(found)
            # ensure raw fallback
            if "raw" not in views:
                views["raw"] = make_image_url(image)
            row_meta = meta.get(image.name, meta.get(image.stem, {}))
            items.append(
                {
                    "name": image.name,
                    "image": views.get("raw") or make_image_url(image),
                    "views": views,
                    "boxes": boxes,
                    "mean_brightness": row_meta.get("mean_brightness"),
                    "brightness_bucket": row_meta.get("brightness_bucket"),
                    "conf_bucket": row_meta.get("conf_bucket"),
                    "area_bucket": row_meta.get("area_bucket"),
                    "stratum": row_meta.get("stratum"),
                    "low_box_count": row_meta.get("low_box_count"),
                }
            )
        # keep only keys that exist for majority
        present = set()
        for it in items:
            present.update(it["views"].keys())
        view_keys = [k for k in (DEFAULT_VIEW_KEYS if not view_keys else view_keys) if k in present]
        # append extras
        for k in sorted(present):
            if k not in view_keys:
                view_keys.append(k)
    else:
        # 单视图：优先使用 images/ 根目录；若为空（划分子目录场景），回退到 images_all/
        images_dir = dataset / "images"
        image_files = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_files and (dataset / "images_all").is_dir():
            images_dir = dataset / "images_all"
            image_files = sorted(
                p for p in images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
        for image in image_files:
            label = labels_root / f"{image.stem}.txt"
            boxes = read_boxes(label)
            relative_image = make_image_url(image)
            items.append({"name": image.name, "image": relative_image, "boxes": boxes})

    view_defs = build_view_defs(view_keys) if multi_view else []
    title = f"伪标签可视化审查 - {dataset.name}"
    if multi_view:
        title = f"多视图审查 - {dataset.name}"
    html = HTML.replace("{{TITLE}}", title)
    html = html.replace("{{ITEMS}}", json.dumps(items, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("{{CLASSES}}", json.dumps(classes, ensure_ascii=False))
    html = html.replace("{{STORAGE_KEY}}", json.dumps(f"review::{dataset.name}"))
    html = html.replace("{{EXPORT_NAME}}", f"{dataset.name}_review_result.jsonl")
    html = html.replace("{{HAS_VIEWS}}", "true" if multi_view else "false")
    html = html.replace("{{VIEW_DEFS}}", json.dumps(view_defs, ensure_ascii=False, separators=(",", ":")))
    (output / "review.html").write_text(html, encoding="utf-8")
    summary = {
        "dataset": str(dataset),
        "images": len(items),
        "boxes": sum(len(x["boxes"]) for x in items),
        "multi_view": multi_view,
        "views": view_keys,
        "review_html": str(output / "review.html"),
    }
    (output / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
