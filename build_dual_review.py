#!/usr/bin/env python3
"""
L1 单页双源对比审查页：GT（绿框，无置信度） vs ryw.pt 候选（红框，真实 conf）。

- 点击表格行或图上的框 → 自动聚焦居中并高亮（旧审查页的聚焦行为）。
- 自动保存：File System Access API 绑定本地 JSONL 文件后，每次判决自动写盘
  （句柄存 IndexedDB，重开页面一键恢复；不支持的浏览器退回 localStorage + 手动导出）。
- 每框独立判决：GT → 保留/删除/存疑；候选 → 补入/删除/存疑。
- 产物：`E:\\PythonProject\\优化\\label_review_v1\\dual\\review.html`（自包含单文件）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UNION = Path(r"E:/PythonProject/datasets/union_original_labels_v1")
CHECK = ROOT / "label_check_ryw_v1"
OUT = ROOT / "label_review_v1" / "dual"

CLASSES_18 = [
    "wuzi", "waterstain", "yellowstain", "nodule", "airhole", "bubble",
    "bump", "fawu", "overfire", "NiOH", "greenstain", "suanwuran",
    "quejiao", "watermark", "youpao", "heibian", "mijixiaokeng", "fanxitong",
]
CLASSES_ZH = [
    "污渍", "水渍", "黄渍", "边部结瘤", "气孔", "结粒", "疙瘩", "发乌", "烧板",
    "氢氧化镍", "绿色结晶", "酸污染", "缺角", "水印", "油泡", "黑边", "密集小坑", "返析铜",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="单页双源对比审查页生成（聚焦+自动保存）", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--union", type=Path, default=UNION)
    p.add_argument("--check-dir", type=Path, default=CHECK)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--conf-min", type=float, default=0.15)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = list(csv.DictReader((args.union / "manifests" / "samples.csv").open(encoding="utf-8-sig")))
    img_index = {p.stem: p for p in (args.union / "images").iterdir()}

    gt = {}
    for r in rows:
        if r.get("has_label") != "True":
            continue
        boxes = []
        lf = args.union / "labels" / f"{r['union_name']}.txt"
        if lf.exists():
            for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = line.split()
                if len(p) >= 5:
                    boxes.append([int(float(p[0]))] + [round(float(x), 5) for x in p[1:5]])
        gt[r["union_name"]] = boxes

    cand = defaultdict(list)
    with (args.check_dir / "predictions.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c = float(r["conf"])
            if c >= args.conf_min:
                cand[r["union_name"]].append(
                    [int(r["label"]), round(c, 3)] + [round(float(r[k]), 5) for k in ("cx", "cy", "w", "h")]
                )
    pc_dir = args.check_dir / "pseudo_candidates" / "labels_with_conf"
    for r in rows:
        if r.get("has_label") == "True":
            continue
        lf = pc_dir / f"{r['union_name']}.txt"
        if lf.exists():
            for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = line.split()
                if len(p) >= 6:
                    c = float(p[5])
                    if c >= args.conf_min:
                        cand[r["union_name"]].append(
                            [int(float(p[0])), round(c, 3)] + [round(float(x), 5) for x in p[1:5]]
                        )

    images = []
    for r in rows:
        name = r["union_name"]
        ip = img_index.get(name)
        if ip is None:
            continue
        images.append({
            "name": name,
            "src": ip.as_uri(),
            "gt": gt.get(name, []),
            "cand": sorted(cand.get(name, []), key=lambda b: -b[1]),
        })
    n_gt = sum(len(i["gt"]) for i in images)
    n_cand = sum(len(i["cand"]) for i in images)
    print(f"图 {len(images)}；GT 框 {n_gt}；候选框 {n_cand}")
    if args.dry_run:
        return 0

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "classes": CLASSES_18,
        "classes_zh": CLASSES_ZH,
        "images": images,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json).replace(
        "__GENERATED__", datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "review.html").write_text(html, encoding="utf-8")
    print(f"产物: {args.output_dir / 'review.html'}（{len(html)//1024} KB）")
    return 0


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>L1 双源对比审查（GT vs ryw.pt 候选）</title>
<style>
body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:0;display:flex;height:100vh;overflow:hidden}
#side{width:260px;min-width:260px;overflow-y:auto;background:#f4f4f4;border-right:1px solid #ccc;padding:8px}
#side .item{padding:2px 6px;cursor:pointer;border-radius:4px;font-size:12px;line-height:1.4;word-break:break-all}
#side .item:hover{background:#e0e8f0}#side .item.cur{background:#2c5f8a;color:#fff}
#side .item.done::after{content:' ✓';color:#2a7d2a;font-weight:bold}
#side .item.done.cur::after{color:#9f9}
#main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-height:0}
#topbar{background:#2c5f8a;color:#fff;padding:4px 10px;display:flex;gap:7px;align-items:center;flex-wrap:wrap;flex:none;font-size:13px}
#topbar button{background:#fff;border:0;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:12px}
#savestat{font-size:12px;opacity:.9}
#stage{flex:1;overflow:auto;background:#222;position:relative;min-height:0;text-align:left}
#wrap{position:relative;display:inline-block;margin-bottom:36px}
#plate{display:block;max-width:none}
.box{position:absolute;box-sizing:border-box;cursor:pointer}
.box.gt{border:3px solid #3fbf3f}
.box.cand{border:2px dashed #ff5050}
.box .tag{position:absolute;font-size:11px;white-space:nowrap;padding:0 3px;border-radius:2px;color:#000;z-index:4}
.box.gt .tag{background:#3fbf3f;top:-18px;left:-2px}
.box.cand .tag{background:#ff5050;top:0;left:0;opacity:.92}
.box.sel{outline:4px solid #ffd400;z-index:5}
#wrap.pick-gt .box.gt{outline:3px solid #ffd400;cursor:copy}
#wrap.pick-gt .box.cand{opacity:.35}
#replacebar{display:none;background:#6a1b9a;color:#fff;padding:4px 10px;border-radius:4px;font-size:13px;margin:4px 0}
#replacebar button{background:#fff;border:0;border-radius:3px;margin-left:8px;cursor:pointer}
#panel{border-top:1px solid #ccc;background:#fafafa;height:34vh;min-height:120px;overflow:auto;padding:8px;position:relative;flex:none}
#divider{height:6px;background:#d5dde5;cursor:ns-resize;flex:none}
#divider:hover{background:#2c5f8a}
#maxbtn{float:right}
table{border-collapse:collapse;font-size:12px;min-width:100%;margin-bottom:26px}
td,th{border:1px solid #ccc;padding:2px 5px;white-space:nowrap}
th{position:sticky;top:0;background:#eee;z-index:2}
tr.brow{cursor:pointer}tr.brow:hover{background:#eef4fa}
tr.brow.cur{background:#fff3bf}
button.b{border:0;border-radius:4px;padding:2px 9px;cursor:pointer;margin-right:4px;font-size:12px}
.bk{background:#4caf50;color:#fff}.bd{background:#f44336;color:#fff}.bu{background:#ff9800;color:#fff}.ba{background:#2196f3;color:#fff}.br{background:#9c27b0;color:#fff}
.d-keep{background:#e4f4e4}.d-remove{background:#fbe0e0}.d-uncertain{background:#fff3d9}.d-add{background:#e0ecf9}.d-replace{background:#f0e0f5}
.warn{background:#fff8e1;border:1px solid #e0c060;padding:6px 10px;border-radius:6px;font-size:13px}
select,input[type=text]{font-size:13px}
</style>
</head>
<body>
<div id="side">
  <input type="text" id="search" placeholder="搜索文件名" style="width:98%" oninput="renderList()">
  <select id="fstate" onchange="renderList()" style="width:100%;margin:4px 0">
    <option value="">全部（3265）</option><option value="todo">未审</option><option value="done">已审</option>
    <option value="hasgt">仅有 GT 标签（1455）</option><option value="nogt">仅无 GT（候选图，1810）</option>
  </select>
  <div id="list"></div>
</div>
<div id="main">
  <div id="topbar">
    <b id="counter"></b>
    <span id="fname"></span>
    <button onclick="nav(-1)">上一张 A</button><button onclick="nav(1)">下一张 D</button>
    <button onclick="zoom(0.8)">缩小</button><button onclick="zoom(1.25)">放大</button><button onclick="fitWidth()">适应窗口</button><button onclick="setZoom(1)">100%</button>
    <select id="fclass" onchange="render()"><option value="-1">全部类</option></select>
    <label>候选conf≥<input id="fconf" type="text" value="0.15" size="4" onchange="render()"></label>
    <label title="密集区域可关掉框上标签名"><input type="checkbox" id="hidetag" onchange="drawBoxes()">隐藏标签名</label>
    <button onclick="bindSaveFile()">绑定保存文件</button><span id="savestat">未绑定（仅 localStorage）</span>
    <button onclick="exportJSONL()">导出 JSONL</button>
    <button onclick="document.getElementById('imp').click()">导入进度</button>
    <input type="file" id="imp" accept=".jsonl" style="display:none" onchange="importJSONL(event)">
    <button onclick="clearProgress()">清空进度</button>
  </div>
  <div id="stage"><div id="wrap"><img id="plate" onload="drawBoxes()"></div></div>
  <div id="divider" title="拖拽调整表格区高度"></div>
  <div id="panel">
    <div class="warn"><button id="maxbtn" class="b ba" onclick="toggleMax()">表格最大化</button>绿实线=原标签 GT（无置信度，判决：保留/删除/存疑）；红虚线=ryw.pt 候选（真实 conf，判决：补入/替换GT/删除/存疑）。<b>滚轮缩放（光标为中心）、拖拽平移</b>。
    <b>替换GT</b>：点候选的「替换GT」→ 再点要替换的 GT 绿框（图上或表格 GT 行）→ 候选标替换、该 GT <b>联动标删除</b>；应用时按你的选择换框，未选择时按同类最大重叠自动配对。
    <b>判定原则：漏实例 &gt; 框松</b>——候选更准且不漏掉 GT 里的实例 → 替换GT；若 GT 里还有候选没盖住的缺陷（如图角多出一个），则保留 GT 或对那个实例单独补入/补画，宁框松勿漏标。
    低 conf（如 0.17）只代表教师不确定，框对不对以你眼睛为准。
    <b>点表格行或图上的框自动聚焦</b>；「绑定保存文件」= 浏览器要求你亲手选一个进度文件授权后，每次判决自动写盘（不绑定则仅存浏览器本地，最后手动导出）。
    <div id="replacebar">替换模式：请点击要被替换的 GT 绿框（图上或表格 GT 行）<button onclick="cancelReplace()">取消 Esc</button></div>
    整图：<button class="b bk" onclick="bulk('gt','keep')">GT全留 K</button><button class="b bd" onclick="bulk('gt','remove')">GT全删</button>
    <button class="b ba" onclick="bulk('cand','add')">候选全补</button><button class="b bd" onclick="bulk('cand','remove')">候选全删</button>
    <span id="bulkinfo"></span></div>
    <table id="boxes"></table>
  </div>
</div>
<script>
const DATA = __DATA_JSON__;
const CL = DATA.classes, CLZ = DATA.classes_zh;
let cur = 0, z = 1, selBox = '';
let decisions = JSON.parse(localStorage.getItem('dual_review_v1') || '{}');
function save(){localStorage.setItem('dual_review_v1', JSON.stringify(decisions));scheduleSave();}
function key(){return DATA.images[cur].name;}
function reviewed(name){const d=decisions[name];return d && Object.keys(d).length>0;}

/* ---------- 自动保存到文件（File System Access API + IndexedDB 句柄） ---------- */
let saveHandle = null, saveTimer = null;
const FS_OK = 'showSaveFilePicker' in window;
function idb(){return new Promise((res,rej)=>{const r=indexedDB.open('dual_review_db',1);
  r.onupgradeneeded=()=>r.result.createObjectStore('kv');r.onsuccess=()=>res(r.result);r.onerror=()=>rej(r.error);});}
async function idbSet(k,v){const db=await idb();return new Promise((res,rej)=>{const t=db.transaction('kv','readwrite');t.objectStore('kv').put(v,k);t.oncomplete=res;t.onerror=()=>rej(t.error);});}
async function idbGet(k){const db=await idb();return new Promise((res,rej)=>{const t=db.transaction('kv','readonly');const q=t.objectStore('kv').get(k);q.onsuccess=()=>res(q.result);q.onerror=()=>rej(q.error);});}
function serializeJSONL(){
  const lines=[];
  for(const name in decisions){const d=decisions[name];
    const bds=Object.values(d).map(v=>({source:v.s,decision:v.d,class_id:v.c,xywh:v.x,...(v.cf!==undefined?{conf:v.cf}:{}),...(v.gt?{replace_gt:v.gt}:{})}));
    if(bds.length)lines.push(JSON.stringify({image_name:name,box_decisions:bds}));}
  return lines.join('\n')+'\n';
}
async function writeToFile(){
  if(!saveHandle)return;
  try{const w=await saveHandle.createWritable();await w.write(serializeJSONL());await w.close();
    document.getElementById('savestat').textContent=`已自动保存 ${saveHandle.name} ${new Date().toLocaleTimeString()}`;
  }catch(e){document.getElementById('savestat').textContent='写盘失败：'+e.message;}
}
function scheduleSave(){if(!saveHandle)return;clearTimeout(saveTimer);saveTimer=setTimeout(writeToFile,800);}
async function bindSaveFile(){
  if(!FS_OK){alert('当前浏览器不支持文件自动保存（Chrome/Edge 支持），将仅使用 localStorage + 手动导出');return;}
  try{
    saveHandle=await showSaveFilePicker({suggestedName:'dual_review_progress.jsonl',
      types:[{description:'JSONL',accept:{'text/plain':['.jsonl']}}]});
    await idbSet('save_handle',saveHandle);
    await writeToFile();
  }catch(e){if(e.name!=='AbortError')alert('绑定失败：'+e.message);}
}
async function restoreHandle(){
  if(!FS_OK)return;
  try{const h=await idbGet('save_handle');if(!h)return;
    if((await h.queryPermission({mode:'readwrite'}))==='granted'){
      saveHandle=h;
      const f=await h.getFile();const txt=await f.text();
      mergeProgressText(txt);
      document.getElementById('savestat').textContent=`已恢复绑定 ${h.name}`;
    }else{
      document.getElementById('savestat').textContent='点击「绑定保存文件」恢复自动保存';
    }
  }catch(e){/* 忽略 */}
}
/* ---------- /自动保存 ---------- */

function renderList(){
  const q = document.getElementById('search').value.toLowerCase();
  const fs = document.getElementById('fstate').value;
  const el = document.getElementById('list'); el.innerHTML='';
  DATA.images.forEach((im,i)=>{
    if(q && !im.name.toLowerCase().includes(q)) return;
    const done = reviewed(im.name);
    if(fs==='todo' && done) return;
    if(fs==='done' && !done) return;
    if(fs==='hasgt' && im.gt.length===0) return;
    if(fs==='nogt' && im.gt.length>0) return;
    const d = document.createElement('div');
    d.className = 'item'+(i===cur?' cur':'')+(done?' done':'');
    d.textContent = `${i+1}. ${im.name} [GT${im.gt.length}/候${im.cand.length}]`;
    d.onclick = ()=>{cur=i;selBox='';render();renderList();};
    el.appendChild(d);
  });
}
function nav(d){cur=Math.min(Math.max(cur+d,0),DATA.images.length-1);selBox='';render();renderList();}
function zoom(f){z=Math.min(Math.max(z*f,0.1),8);applyZoom();}
function setZoom(v){z=v;applyZoom();}
function fitWidth(){const img=document.getElementById('plate'),stage=document.getElementById('stage');
  if(img.naturalWidth){z=Math.max((stage.clientWidth-24)/img.naturalWidth,0.02);}applyZoom();}
function applyZoom(){const img=document.getElementById('plate');if(img.naturalWidth)img.width=img.naturalWidth*z;drawBoxes();}

/* ---------- 替换GT 两步显式配对 ---------- */
let replaceMode = null; // 候选框 idx
function startReplace(i){
  replaceMode = (replaceMode === i ? null : i);
  document.getElementById('replacebar').style.display = replaceMode === null ? 'none' : 'block';
  document.getElementById('wrap').classList.toggle('pick-gt', replaceMode !== null);
}
function cancelReplace(){replaceMode=null;
  document.getElementById('replacebar').style.display='none';
  document.getElementById('wrap').classList.remove('pick-gt');}
function pickGt(i){
  if(replaceMode===null)return false;
  setDec('cand',replaceMode,'replace');
  const im=DATA.images[cur];const g=im.gt[i];
  decisions[key()]['cand'+replaceMode].gt=[g[1],g[2],g[3],g[4]]; // 显式配对坐标
  setDec('gt',i,'remove');                                       // 联动：被替换 GT 标删除
  cancelReplace();
  save();updateCounter();drawBoxes();renderList();
  return true;
}
/* ---------- /替换GT ---------- */

/* ---------- 聚焦 ---------- */
function focusBox(src,idx){
  selBox = src+idx;
  const im = DATA.images[cur];
  const b = src==='gt'?im.gt[idx]:im.cand[idx];
  if(!b)return;
  const [x,y,w,h] = src==='gt'?[b[1],b[2],b[3],b[4]]:[b[2],b[3],b[4],b[5]];
  const stage = document.getElementById('stage');
  const targetZ = Math.min(Math.max(0.35/Math.max(w,h,0.01), 0.5), 8);
  z = targetZ; applyZoom();
  const img = document.getElementById('plate');
  stage.scrollLeft = x*img.width - stage.clientWidth/2;
  stage.scrollTop = y*img.height - stage.clientHeight/2;
  drawBoxes();
  const row = document.getElementById('row_'+src+idx);
  if(row){row.scrollIntoView({block:'center'});}
}
/* ---------- /聚焦 ---------- */

function render(){
  const im = DATA.images[cur];
  document.getElementById('fname').textContent = `${cur+1}/${DATA.images.length} ${im.name}`;
  const img=document.getElementById('plate');
  img.onload=()=>{fitWidth();};
  img.src = im.src;
  updateCounter();
}
function updateCounter(){
  let done=0,gk=0,gd=0,ca=0,cd=0,rp=0;
  for(const n in decisions){const d=decisions[n];if(Object.keys(d).length)done++;
    for(const k in d){const v=d[k];if(v.s==='gt'&&v.d==='keep')gk++;if(v.s==='gt'&&v.d==='remove')gd++;
      if(v.s==='cand'&&v.d==='add')ca++;if(v.s==='cand'&&v.d==='remove')cd++;if(v.s==='cand'&&v.d==='replace')rp++;}}
  document.getElementById('counter').textContent=`已审 ${done}/${DATA.images.length}｜GT留${gk} 删${gd}｜候补${ca} 替${rp} 删${cd}`;
}
function visible(box,src){
  const fc=+document.getElementById('fclass').value;
  if(fc>=0 && box[0]!==fc) return false;
  if(src==='cand'){const cf=+document.getElementById('fconf').value||0;if(box[1]<cf)return false;}
  return true;
}
function drawBoxes(){
  const im=DATA.images[cur], wrap=document.getElementById('wrap'), img=document.getElementById('plate');
  wrap.querySelectorAll('.box').forEach(e=>e.remove());
  const W=img.width,H=img.height;if(!W)return;
  const mk=(b,src,idx)=>{
    if(!visible(b,src))return;
    const [cid,x,y,w,h]=src==='gt'?b:[b[0],b[2],b[3],b[4],b[5]];
    const d=document.createElement('div');
    d.className='box '+src+(selBox===src+idx?' sel':'');
    d.style.left=(x-w/2)*W+'px';d.style.top=(y-h/2)*H+'px';d.style.width=w*W+'px';d.style.height=h*H+'px';
    d.onclick=(ev)=>{ev.stopPropagation();
      if(suppressPanClick())return;
      if(src==='gt'&&replaceMode!==null){pickGt(idx);return;}
      focusBox(src,idx);};
    if(!document.getElementById('hidetag').checked){
      const t=document.createElement('span');t.className='tag';
      t.textContent=src==='gt'?CL[cid]:`${CL[cid]} ${b[1].toFixed(2)}`;
      d.appendChild(t);
    }
    wrap.appendChild(d);
  };
  im.gt.forEach((b,i)=>mk(b,'gt',i));im.cand.forEach((b,i)=>mk(b,'cand',i));
  renderTable();
}
function decisionOf(src,idx){const d=decisions[key()];return d&&d[src+idx]?d[src+idx].d:'';}
function setDec(src,idx,val){
  if(!decisions[key()])decisions[key()]={};
  const im=DATA.images[cur];
  const b=src==='gt'?im.gt[idx]:im.cand[idx];
  const xywh=src==='gt'?[b[1],b[2],b[3],b[4]]:[b[2],b[3],b[4],b[5]];
  const rec={s:src,d:val,c:b[0],x:xywh};
  if(src==='cand')rec.cf=b[1];
  decisions[key()][src+idx]=rec;
  save();updateCounter();drawBoxes();renderList();
}
function bulk(src,val){
  const im=DATA.images[cur];
  const n=src==='gt'?im.gt.length:im.cand.length;
  let cnt=0;
  for(let i=0;i<n;i++){if(visible(src==='gt'?im.gt[i]:im.cand[i],src)){setDecQuiet(src,i,val);cnt++;}}
  save();updateCounter();drawBoxes();renderList();
  document.getElementById('bulkinfo').textContent=` 已对${cnt}框执行`;
}
function setDecQuiet(src,idx,val){
  if(!decisions[key()])decisions[key()]={};
  const im=DATA.images[cur];const b=src==='gt'?im.gt[idx]:im.cand[idx];
  const xywh=src==='gt'?[b[1],b[2],b[3],b[4]]:[b[2],b[3],b[4],b[5]];
  const rec={s:src,d:val,c:b[0],x:xywh};if(src==='cand')rec.cf=b[1];
  decisions[key()][src+idx]=rec;
}
function rowClick(src,i){
  if(src==='gt'&&replaceMode!==null){pickGt(i);return;}
  focusBox(src,i);
}
function renderTable(){
  const im=DATA.images[cur],t=document.getElementById('boxes');
  let h='<tr><th>#</th><th>源</th><th>类</th><th>conf</th><th>判决操作</th><th>cx</th><th>cy</th><th>w</th><th>h</th><th>状态</th></tr>';
  const row=(src,i,b)=>{
    const cid=b[0],cf=src==='gt'?'—':b[1].toFixed(2);
    const xywh=src==='gt'?[b[1],b[2],b[3],b[4]]:[b[2],b[3],b[4],b[5]];
    const dv=decisionOf(src,i);
    const acts=src==='gt'
      ?`<button class="b bk" onclick="event.stopPropagation();setDec('gt',${i},'keep')">保留</button><button class="b bd" onclick="event.stopPropagation();setDec('gt',${i},'remove')">删除</button><button class="b bu" onclick="event.stopPropagation();setDec('gt',${i},'uncertain')">存疑</button>`
      :`<button class="b ba" onclick="event.stopPropagation();setDec('cand',${i},'add')">补入</button><button class="b br" onclick="event.stopPropagation();startReplace(${i})">替换GT</button><button class="b bd" onclick="event.stopPropagation();setDec('cand',${i},'remove')">删除</button><button class="b bu" onclick="event.stopPropagation();setDec('cand',${i},'uncertain')">存疑</button>`;
    const clr=dv?('d-'+dv):'';
    h+=`<tr id="row_${src}${i}" class="brow ${clr} ${selBox===src+i?'cur':''}" onclick="rowClick('${src}',${i})">`+
       `<td>${src[0]}${i}</td><td>${src==='gt'?'GT':'候选'}</td><td>${CL[cid]} ${CLZ[cid]}</td><td>${cf}</td><td>${acts}</td>`+
       xywh.map(v=>`<td>${v.toFixed(2)}</td>`).join('')+`<td>${dv||'未判'}</td></tr>`;
  };
  im.gt.forEach((b,i)=>{if(visible(b,'gt'))row('gt',i,b);});
  im.cand.forEach((b,i)=>{if(visible(b,'cand'))row('cand',i,b);});
  t.innerHTML=h;
}
function exportJSONL(){
  const blob=new Blob([serializeJSONL()],{type:'text/plain'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='dual_review_result.jsonl';a.click();
}
function mergeProgressText(txt){
  txt.split('\n').forEach(line=>{line=line.trim();if(!line)return;
    let rec;try{rec=JSON.parse(line);}catch(e){return;}
    const im=DATA.images.find(i=>i.name===rec.image_name);if(!im)return;
    const d={};
    (rec.box_decisions||[]).forEach(b=>{
      const arr=b.source==='gt'?im.gt:im.cand;
      let idx=-1;
      for(let i=0;i<arr.length;i++){const bb=arr[i];
        const xywh=b.source==='gt'?[bb[1],bb[2],bb[3],bb[4]]:[bb[2],bb[3],bb[4],bb[5]];
        if(bb[0]===b.class_id&&Math.abs(xywh[0]-b.xywh[0])<1e-4&&Math.abs(xywh[1]-b.xywh[1])<1e-4&&Math.abs(xywh[2]-b.xywh[2])<1e-4&&Math.abs(xywh[3]-b.xywh[3])<1e-4){idx=i;break;}}
      if(idx>=0)d[b.source+idx]={s:b.source,d:b.decision,c:b.class_id,x:b.xywh,...(b.conf!==undefined?{cf:b.conf}:{})};});
    if(Object.keys(d).length)decisions[rec.image_name]=d;});
  localStorage.setItem('dual_review_v1', JSON.stringify(decisions));
}
function importJSONL(ev){
  const f=ev.target.files[0];if(!f)return;
  f.text().then(txt=>{mergeProgressText(txt);scheduleSave();updateCounter();drawBoxes();renderList();alert('进度已导入');});
}
function clearProgress(){if(confirm('清空全部审查进度？')){decisions={};save();updateCounter();drawBoxes();renderList();}}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(e.key==='Escape'){cancelReplace();return;}
  if(e.key==='a'||e.key==='A')nav(-1);if(e.key==='d'||e.key==='D')nav(1);
  if(e.key==='k'||e.key==='K')bulk('gt','keep');
});

/* ---------- 滚轮缩放（光标为中心） + 拖拽平移 ---------- */
let panState=null, panned=false;
function suppressPanClick(){return panned;}
(function(){
  const stage=document.getElementById('stage');
  stage.addEventListener('wheel',e=>{
    e.preventDefault();
    const rect=stage.getBoundingClientRect();
    const cx=e.clientX-rect.left, cy=e.clientY-rect.top;
    const oldZ=z;
    zoom(e.deltaY<0?1.15:1/1.15);
    const r=z/oldZ;
    stage.scrollLeft=(stage.scrollLeft+cx)*r-cx;
    stage.scrollTop=(stage.scrollTop+cy)*r-cy;
  },{passive:false});
  stage.addEventListener('mousedown',e=>{
    panState={x:e.clientX,y:e.clientY,sl:stage.scrollLeft,st:stage.scrollTop};
    stage.style.cursor='grabbing';
  });
  window.addEventListener('mousemove',e=>{
    if(!panState)return;
    const dx=e.clientX-panState.x, dy=e.clientY-panState.y;
    if(Math.abs(dx)+Math.abs(dy)>4){panned=true;
      stage.scrollLeft=panState.sl-dx;stage.scrollTop=panState.st-dy;}
  });
  window.addEventListener('mouseup',()=>{
    if(panState){stage.style.cursor='';panState=null;setTimeout(()=>{panned=false;},60);}
  });
})();
/* ---------- /缩放平移 ---------- */
const fcsel=document.getElementById('fclass');
CL.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${c} ${CLZ[i]}`;fcsel.appendChild(o);});

/* 表格区拖拽调高 + 最大化 */
(function(){
  const div=document.getElementById('divider'),panel=document.getElementById('panel'),stage=document.getElementById('stage');
  let drag=false;
  div.addEventListener('mousedown',e=>{drag=true;e.preventDefault();document.body.style.userSelect='none';});
  window.addEventListener('mousemove',e=>{if(!drag)return;
    const h=window.innerHeight-e.clientY;
    panel.style.height=Math.min(Math.max(h,120),window.innerHeight*0.85)+'px';});
  window.addEventListener('mouseup',()=>{drag=false;document.body.style.userSelect='';});
})();
let maximized=false, savedH='';
function toggleMax(){
  const panel=document.getElementById('panel'),stage=document.getElementById('stage'),div=document.getElementById('divider'),btn=document.getElementById('maxbtn');
  maximized=!maximized;
  if(maximized){savedH=panel.style.height;stage.style.display='none';div.style.display='none';panel.style.height='calc(100vh - 150px)';btn.textContent='还原图像';}
  else{stage.style.display='';div.style.display='';panel.style.height=savedH||'34vh';btn.textContent='表格最大化';}
}

renderList();render();restoreHandle();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
