"""
生成 pseudo-label v1 综合审查界面（accepted / uncertain / rejected 三标签页）。
读取三个状态的 manifests 和对应可视化图，生成一个自包含 HTML 文件。
"""

import json
import re
from collections import defaultdict
from pathlib import Path


_STATUS_RE = re.compile(r"^(?P<dataset>[^_]+)_(?P<status>accepted|uncertain|rejected)_(?P<idx>\d+)_(?P<stem>.+)\.jpg$")


def load_manifest(manifest_path: Path):
    """读取 jsonl，按 image_stem 聚合所有框。"""
    groups = defaultdict(list)
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            groups[rec["image_stem"]].append(rec)
    return groups


def collect_visualizations(vis_root: Path):
    """扫描三个状态的可视化目录，返回 [{dataset, status, idx, stem, filename, rel_dir}, ...]。"""
    items = []
    for status in ["accepted", "uncertain", "rejected"]:
        vis_dir = vis_root / f"pseudo_v1_{status}_only"
        if not vis_dir.exists():
            continue
        for p in sorted(vis_dir.glob("*.jpg")):
            m = _STATUS_RE.match(p.name)
            if not m:
                continue
            items.append({
                "dataset": m.group("dataset"),
                "status": m.group("status"),
                "idx": int(m.group("idx")),
                "stem": m.group("stem"),
                "filename": p.name,
                "rel_dir": f"pseudo_v1_{status}_only",
            })
    return items


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pseudo-Label v1 综合审查界面</title>
    <style>
        :root { --bg:#f5f5f5; --panel:#fff; --border:#ddd; --accent:#2b6cb0; --danger:#c53030; --warn:#d69e2e; --ok:#38a169; }
        * { box-sizing: border-box; }
        body { margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background:var(--bg); color:#333; }
        header { background:#1a202c; color:#fff; padding:12px 20px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; }
        header h1 { margin:0; font-size:18px; }
        header .stats { font-size:13px; color:#cbd5e0; }
        .container { display:flex; height:calc(100vh - 56px); }
        .sidebar { width:320px; background:var(--panel); border-right:1px solid var(--border); overflow-y:auto; padding:15px; }
        .main { flex:1; display:flex; flex-direction:column; overflow:hidden; }
        .image-wrap { flex:1; display:flex; align-items:center; justify-content:center; background:#e2e8f0; overflow:hidden; position:relative; cursor:grab; }
        .image-wrap:active { cursor:grabbing; }
        .image-wrap .empty { position:absolute; color:#718096; }
        .image-viewer { position:absolute; top:0; left:0; width:100%; height:100%; display:flex; align-items:center; justify-content:center; transform-origin:center center; }
        .image-viewer img { max-width:100%; max-height:100%; object-fit:contain; box-shadow:0 2px 10px rgba(0,0,0,.15); pointer-events:none; user-select:none; }
        .zoom-hint { position:absolute; bottom:8px; left:8px; background:rgba(0,0,0,.6); color:#fff; padding:4px 8px; border-radius:3px; font-size:11px; pointer-events:none; }
        .controls { background:var(--panel); border-top:1px solid var(--border); padding:12px 20px; display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }
        .btn { padding:8px 14px; border:0; border-radius:4px; cursor:pointer; font-size:14px; background:#e2e8f0; color:#1a202c; }
        .btn:hover { filter:brightness(.95); }
        .btn-primary { background:var(--accent); color:#fff; }
        .btn-success { background:var(--ok); color:#fff; }
        .btn-danger { background:var(--danger); color:#fff; }
        .btn-warning { background:var(--warn); color:#fff; }
        .tabs { display:flex; gap:4px; margin-bottom:12px; }
        .tab { flex:1; padding:8px; border:1px solid var(--border); background:#edf2f7; cursor:pointer; text-align:center; font-size:13px; border-radius:4px; }
        .tab.active { background:var(--accent); color:#fff; border-color:var(--accent); }
        .tab-accepted.active { background:var(--ok); }
        .tab-uncertain.active { background:var(--warn); }
        .tab-rejected.active { background:var(--danger); }
        .filter-group { display:flex; gap:6px; align-items:center; margin-bottom:12px; flex-wrap:wrap; }
        .filter-group label { font-size:13px; color:#555; }
        .thumb-list { display:flex; flex-direction:column; gap:6px; }
        .thumb { display:flex; align-items:center; gap:10px; padding:8px; border-radius:4px; cursor:pointer; border:1px solid transparent; }
        .thumb:hover { background:#edf2f7; }
        .thumb.active { background:#ebf8ff; border-color:var(--accent); }
        .thumb img { width:48px; height:48px; object-fit:cover; border-radius:3px; background:#cbd5e0; }
        .thumb .info { flex:1; min-width:0; }
        .thumb .title { font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .thumb .badge { font-size:11px; padding:1px 5px; border-radius:3px; display:inline-block; margin-top:2px; }
        .badge-pending { background:#edf2f7; color:#4a5568; }
        .badge-ok { background:#c6f6d5; color:#22543d; }
        .badge-reject { background:#fed7d7; color:#742a2a; }
        .badge-review { background:#fefcbf; color:#744210; }
        .box-list { margin-top:15px; }
        .box-item { display:flex; align-items:center; justify-content:space-between; padding:8px; border-bottom:1px solid #edf2f7; font-size:13px; }
        .box-item .cls { font-weight:600; min-width:90px; }
        .box-item .conf { color:#718096; }
        .box-actions { display:flex; gap:4px; }
        .box-actions button { font-size:11px; padding:3px 8px; border-radius:3px; border:1px solid #cbd5e0; background:#fff; cursor:pointer; }
        .box-actions button.active-keep { background:var(--ok); color:#fff; border-color:var(--ok); }
        .box-actions button.active-remove { background:var(--danger); color:#fff; border-color:var(--danger); }
        .box-actions button.active-uncertain { background:var(--warn); color:#fff; border-color:var(--warn); }
        .global-actions { display:flex; gap:8px; margin-bottom:12px; }
        .global-actions button { flex:1; font-size:12px; }
        .progress { height:4px; background:#e2e8f0; border-radius:2px; margin-bottom:12px; }
        .progress-bar { height:100%; background:var(--accent); border-radius:2px; transition:width .2s; }
        .note { width:100%; margin-top:8px; }
        .note textarea { width:100%; min-height:60px; padding:8px; border:1px solid var(--border); border-radius:4px; font-size:13px; resize:vertical; }
        .empty { text-align:center; color:#718096; padding:40px; }
        .status-guide { font-size:12px; color:#4a5568; background:#f7fafc; padding:8px; border-radius:4px; margin-bottom:12px; border:1px solid var(--border); }
        .status-guide span { font-weight:600; }
        .status-guide .accepted { color:var(--ok); }
        .status-guide .uncertain { color:var(--warn); }
        .status-guide .rejected { color:var(--danger); }
    </style>
</head>
<body>
<header>
    <h1>Pseudo-Label v1 综合审查界面</h1>
    <div class="stats">
        已审查 <span id="reviewedCount">0</span> / <span id="totalCount">0</span>
        | 保留 <span id="keepCount">0</span>
        | 删除 <span id="removeCount">0</span>
        | 存疑 <span id="uncertainCount">0</span>
    </div>
    <div style="display:flex;gap:8px;">
        <button class="btn" onclick="clearState()">🗑️ 清除进度</button>
        <button class="btn" onclick="downloadState()">💾 保存进度</button>
        <button class="btn btn-primary" onclick="exportFinal()">📥 导出结果</button>
        <label class="btn" style="position:relative;">
            📂 加载进度
            <input type="file" id="loadFile" accept=".json" style="position:absolute;left:0;top:0;width:100%;height:100%;opacity:0;cursor:pointer;" onchange="loadStateFile(event)">
        </label>
    </div>
</header>
<div class="container">
    <aside class="sidebar">
        <div class="progress"><div class="progress-bar" id="progressBar"></div></div>
        
        <div class="tabs" id="statusTabs">
            <div class="tab tab-accepted active" data-status="accepted" onclick="switchStatus('accepted')">已采纳 (≥0.70)</div>
            <div class="tab tab-uncertain" data-status="uncertain" onclick="switchStatus('uncertain')">不确定 (0.40-0.69)</div>
            <div class="tab tab-rejected" data-status="rejected" onclick="switchStatus('rejected')">已拒绝 (<0.40)</div>
        </div>
        
        <div class="status-guide" id="statusGuide"></div>
        
        <div class="filter-group">
            <label>数据集:</label>
            <select id="datasetFilter" onchange="renderThumbs(); showCurrent()">
                <option value="all">全部</option>
                <option value="Dianjinie0211">Dianjinie0211</option>
                <option value="dianjienie">dianjienie</option>
            </select>
        </div>
        <div class="filter-group">
            <label>审查状态:</label>
            <select id="reviewStatusFilter" onchange="renderThumbs()">
                <option value="all">全部</option>
                <option value="pending">未审查</option>
                <option value="done">已审查</option>
            </select>
        </div>
        <div class="global-actions">
            <button class="btn btn-success" onclick="markAllBoxes('keep')">全保留</button>
            <button class="btn btn-danger" onclick="markAllBoxes('remove')">全删除</button>
            <button class="btn btn-warning" onclick="markAllBoxes('uncertain')">全存疑</button>
        </div>
        <div class="thumb-list" id="thumbList"></div>
    </aside>
    <section class="main">
        <div class="image-wrap" id="imageWrap">
            <div class="empty">请选择左侧图片开始审查</div>
        </div>
        <div class="controls">
            <div style="display:flex;gap:8px;align-items:center;">
                <button class="btn" onclick="prev()">⬅ 上一张</button>
                <button class="btn" onclick="next()">下一张 ➡</button>
                <span id="navInfo" style="font-size:13px;color:#555;"></span>
            </div>
            <div style="display:flex;gap:8px;align-items:center;">
                <button class="btn" onclick="zoomOut()">➖ 缩小</button>
                <button class="btn" onclick="zoomReset()">🔍 重置</button>
                <button class="btn" onclick="zoomIn()">➕ 放大</button>
                <span id="zoomInfo" style="font-size:13px;color:#555;min-width:55px;text-align:center;">100%</span>
            </div>
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                <button class="btn btn-success" onclick="markImage('keep')">整图保留</button>
                <button class="btn btn-danger" onclick="markImage('remove')">整图删除</button>
                <button class="btn btn-warning" onclick="markImage('uncertain')">整图存疑</button>
            </div>
        </div>
        <div class="controls" style="border-top:0;flex-direction:column;align-items:stretch;">
            <div class="box-list" id="boxList"></div>
            <div class="note">
                <label style="font-size:13px;color:#555;">备注:</label>
                <textarea id="imageNote" placeholder="输入关于本图的备注..." oninput="saveNote()"></textarea>
            </div>
        </div>
    </section>
</div>

<script>
const IMAGES = {{IMAGES_JSON}};
const BOXES = {{BOXES_JSON}};

const STATUS_GUIDES = {
    accepted: '<span class="accepted">已采纳（≥0.70）</span>：这些框已写入伪标签。审查重点是发现高置信度假阳性错误，把错框标为“删除”。',
    uncertain: '<span class="uncertain">不确定（0.40-0.69）</span>：模型拿不准的框，审查优先级最高。请逐框决定“保留”或“删除”。',
    rejected: '<span class="rejected">已拒绝（<0.40）</span>：置信度过低被过滤的框。可抽样检查，把被误删的真缺陷标为“保留”。'
};

let currentStatus = 'accepted';
let currentIndex = 0;
let filtered = [];
let reviewState = loadLocalState() || {};

function loadLocalState() {
    try { return JSON.parse(localStorage.getItem('pseudo_v1_review_state_v2')); }
    catch (e) { return null; }
}
function saveLocalState() {
    localStorage.setItem('pseudo_v1_review_state_v2', JSON.stringify(reviewState));
    updateStats();
}

function getKey(img) { return img.dataset + '/' + img.stem; }

function ensureState(img) {
    const key = getKey(img);
    if (!reviewState[key]) {
        reviewState[key] = { boxes: {}, note: '', imageStatus: '' };
    }
    return reviewState[key];
}

function switchStatus(status) {
    currentStatus = status;
    currentIndex = 0;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.tab[data-status="${status}"]`).classList.add('active');
    document.getElementById('statusGuide').innerHTML = STATUS_GUIDES[status];
    renderThumbs();
    showCurrent();
}

function renderThumbs() {
    const ds = document.getElementById('datasetFilter').value;
    const st = document.getElementById('reviewStatusFilter').value;
    filtered = IMAGES.filter(img => {
        if (img.status !== currentStatus) return false;
        if (ds !== 'all' && img.dataset !== ds) return false;
        const state = reviewState[getKey(img)];
        if (st === 'pending' && state && (state.imageStatus || Object.keys(state.boxes).length)) return false;
        if (st === 'done' && (!state || (!state.imageStatus && !Object.keys(state.boxes).length))) return false;
        return true;
    });
    const list = document.getElementById('thumbList');
    list.innerHTML = '';
    filtered.forEach((img, idx) => {
        const state = reviewState[getKey(img)] || {};
        let statusClass = 'badge-pending', statusText = '未审查';
        if (state.imageStatus) {
            if (state.imageStatus === 'keep') { statusClass = 'badge-ok'; statusText = '保留'; }
            else if (state.imageStatus === 'remove') { statusClass = 'badge-reject'; statusText = '删除'; }
            else if (state.imageStatus === 'uncertain') { statusClass = 'badge-review'; statusText = '存疑'; }
        } else if (Object.keys(state.boxes || {}).length) {
            const vals = Object.values(state.boxes);
            if (vals.every(v => v === 'keep')) { statusClass = 'badge-ok'; statusText = '保留'; }
            else if (vals.every(v => v === 'remove')) { statusClass = 'badge-reject'; statusText = '删除'; }
            else { statusClass = 'badge-review'; statusText = '部分'; }
        }
        const div = document.createElement('div');
        div.className = 'thumb' + (idx === currentIndex ? ' active' : '');
        div.onclick = () => { currentIndex = idx; showCurrent(); };
        div.innerHTML = `<img src="${img.rel_dir}/${img.filename}" loading="lazy" alt=""><div class="info"><div class="title">${img.dataset}: ${img.stem}</div><span class="badge ${statusClass}">${statusText}</span></div>`;
        list.appendChild(div);
    });
    updateStats();
}

function showCurrent() {
    resetZoom();
    if (!filtered.length) {
        document.getElementById('imageWrap').innerHTML = '<div class="empty">该状态下无匹配图片</div>';
        document.getElementById('boxList').innerHTML = '';
        document.getElementById('navInfo').textContent = '';
        document.getElementById('imageNote').value = '';
        return;
    }
    if (currentIndex >= filtered.length) currentIndex = filtered.length - 1;
    const img = filtered[currentIndex];
    document.getElementById('imageWrap').innerHTML = `
        <div class="image-viewer" id="imageViewer">
            <img src="${img.rel_dir}/${img.filename}" alt="" id="mainImage">
        </div>
        <div class="zoom-hint">滚轮缩放 / 拖拽平移 / 双击重置</div>
    `;
    document.getElementById('navInfo').textContent = `${currentIndex + 1} / ${filtered.length}`;

    const key = getKey(img);
    const state = ensureState(img);
    const boxes = BOXES[key + '/' + img.status] || [];
    const boxList = document.getElementById('boxList');
    boxList.innerHTML = '';
    boxes.forEach((box, i) => {
        const current = state.boxes[i] || '';
        const div = document.createElement('div');
        div.className = 'box-item';
        div.innerHTML = `
            <span class="cls">${box.class_name}</span>
            <span class="conf">conf=${box.confidence.toFixed(2)}</span>
            <div class="box-actions">
                <button class="${current === 'keep' ? 'active-keep' : ''}" onclick="markBox(${i}, 'keep')">保留</button>
                <button class="${current === 'remove' ? 'active-remove' : ''}" onclick="markBox(${i}, 'remove')">删除</button>
                <button class="${current === 'uncertain' ? 'active-uncertain' : ''}" onclick="markBox(${i}, 'uncertain')">存疑</button>
            </div>
        `;
        boxList.appendChild(div);
    });
    document.getElementById('imageNote').value = state.note || '';
    renderThumbs();
    const active = document.querySelector('.thumb.active');
    if (active) active.scrollIntoView({ block: 'nearest' });
}

function markBox(idx, decision) {
    const img = filtered[currentIndex];
    const state = ensureState(img);
    if (state.boxes[idx] === decision) delete state.boxes[idx];
    else state.boxes[idx] = decision;
    state.imageStatus = '';
    saveLocalState();
    showCurrent();
}

function markAllBoxes(decision) {
    const img = filtered[currentIndex];
    const state = ensureState(img);
    const boxes = BOXES[getKey(img) + '/' + img.status] || [];
    boxes.forEach((_, i) => state.boxes[i] = decision);
    state.imageStatus = '';
    saveLocalState();
    showCurrent();
}

function markImage(decision) {
    const img = filtered[currentIndex];
    const state = ensureState(img);
    state.imageStatus = decision;
    state.boxes = {};
    saveLocalState();
    showCurrent();
}

function saveNote() {
    const img = filtered[currentIndex];
    const state = ensureState(img);
    state.note = document.getElementById('imageNote').value;
    saveLocalState();
}

function prev() { if (currentIndex > 0) { currentIndex--; showCurrent(); } }
function next() { if (currentIndex < filtered.length - 1) { currentIndex++; showCurrent(); } }

function updateStats() {
    let reviewed = 0, keep = 0, remove = 0, uncertain = 0;
    IMAGES.forEach(img => {
        const state = reviewState[getKey(img)];
        if (!state) return;
        let status = state.imageStatus;
        if (!status && Object.keys(state.boxes || {}).length) {
            const vals = Object.values(state.boxes);
            if (vals.every(v => v === 'keep')) status = 'keep';
            else if (vals.every(v => v === 'remove')) status = 'remove';
            else status = 'uncertain';
        }
        if (status) {
            reviewed++;
            if (status === 'keep') keep++;
            else if (status === 'remove') remove++;
            else if (status === 'uncertain') uncertain++;
        }
    });
    document.getElementById('reviewedCount').textContent = reviewed;
    document.getElementById('totalCount').textContent = IMAGES.length;
    document.getElementById('keepCount').textContent = keep;
    document.getElementById('removeCount').textContent = remove;
    document.getElementById('uncertainCount').textContent = uncertain;
    document.getElementById('progressBar').style.width = (IMAGES.length ? (reviewed / IMAGES.length * 100) : 0) + '%';
}

function downloadState() {
    const blob = new Blob([JSON.stringify(reviewState, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'pseudo_v1_review_state_v2.json';
    a.click();
}

function loadStateFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
        try {
            reviewState = JSON.parse(e.target.result);
            saveLocalState();
            renderThumbs();
            showCurrent();
            alert('审查进度已加载');
        } catch (err) { alert('文件解析失败: ' + err.message); }
    };
    reader.readAsText(file);
}

function exportFinal() {
    const lines = [];
    IMAGES.forEach(img => {
        const key = getKey(img);
        const state = reviewState[key];
        const boxes = BOXES[key + '/' + img.status] || [];
        if (!state) return;
        let imageStatus = state.imageStatus;
        boxes.forEach((box, i) => {
            let decision = state.boxes[i] || imageStatus;
            if (!decision) {
                // 未标记时使用状态默认
                if (img.status === 'accepted') decision = 'keep';
                else if (img.status === 'rejected') decision = 'remove';
                else decision = 'uncertain';
            }
            if (decision === 'uncertain') decision = 'review';
            lines.push({
                dataset: img.dataset,
                status: img.status,
                image_stem: img.stem,
                box_index: i,
                class_name: box.class_name,
                confidence: box.confidence,
                decision: decision,
                note: state.note || ''
            });
        });
    });
    const blob = new Blob([lines.map(l => JSON.stringify(l)).join('\n')], { type: 'application/jsonl' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'pseudo_v1_review_result.jsonl';
    a.click();
}

function clearState() {
    if (!confirm('确定要清除所有审查进度吗？')) return;
    reviewState = {};
    saveLocalState();
    renderThumbs();
    showCurrent();
}

// ================== 图片缩放与拖拽平移 ==================
let zoomScale = 1;
let zoomTranslateX = 0;
let zoomTranslateY = 0;
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let dragStartTranslateX = 0;
let dragStartTranslateY = 0;

function applyZoom() {
    const viewer = document.getElementById('imageViewer');
    if (!viewer) return;
    viewer.style.transform = `translate(${zoomTranslateX}px, ${zoomTranslateY}px) scale(${zoomScale})`;
    document.getElementById('zoomInfo').textContent = Math.round(zoomScale * 100) + '%';
}

function resetZoom() {
    zoomScale = 1;
    zoomTranslateX = 0;
    zoomTranslateY = 0;
    applyZoom();
}

function zoomIn() {
    zoomScale = Math.min(zoomScale * 1.25, 10);
    applyZoom();
}

function zoomOut() {
    zoomScale = Math.max(zoomScale / 1.25, 0.2);
    if (zoomScale <= 1) { zoomTranslateX = 0; zoomTranslateY = 0; }
    applyZoom();
}

function zoomReset() {
    resetZoom();
}

function setupZoomEvents() {
    const wrap = document.getElementById('imageWrap');
    if (!wrap) return;

    wrap.addEventListener('wheel', e => {
        e.preventDefault();
        if (e.deltaY < 0) zoomIn();
        else zoomOut();
    }, { passive: false });

    wrap.addEventListener('mousedown', e => {
        if (e.button !== 0) return;
        isDragging = true;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        dragStartTranslateX = zoomTranslateX;
        dragStartTranslateY = zoomTranslateY;
        wrap.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', e => {
        if (!isDragging) return;
        const dx = e.clientX - dragStartX;
        const dy = e.clientY - dragStartY;
        zoomTranslateX = dragStartTranslateX + dx;
        zoomTranslateY = dragStartTranslateY + dy;
        applyZoom();
    });

    window.addEventListener('mouseup', () => {
        if (!isDragging) return;
        isDragging = false;
        const wrap = document.getElementById('imageWrap');
        if (wrap) wrap.style.cursor = 'grab';
    });

    wrap.addEventListener('dblclick', () => {
        zoomReset();
    });
}

document.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft') prev();
    if (e.key === 'ArrowRight') next();
});

switchStatus('accepted');
setupZoomEvents();
</script>
</body>
</html>
"""


def main():
    project_root = Path("E:/PythonProject/优化")
    vis_root = project_root / "visualizations"
    out_html = vis_root / "pseudo_v1_comprehensive_review.html"
    manifest_root = Path("E:/PythonProject/datasets")

    images = collect_visualizations(vis_root)
    print(f"找到 {len(images)} 张三状态可视化图")

    # 加载所有 manifests，按 dataset/stem/status 分别存储
    boxes = {}
    for ds in ["Dianjinie0211", "dianjienie"]:
        for status in ["accepted", "uncertain", "rejected"]:
            manifest = manifest_root / ds / "manifests" / f"{status}.jsonl"
            if not manifest.exists():
                continue
            groups = load_manifest(manifest)
            for stem, box_list in groups.items():
                boxes[f"{ds}/{stem}/{status}"] = box_list

    # 只保留有 box 信息的图片
    images = [img for img in images if f"{img['dataset']}/{img['stem']}/{img['status']}" in boxes]
    print(f"其中 {len(images)} 张可匹配到 manifest 框信息")

    html = HTML_TEMPLATE
    html = html.replace("{{IMAGES_JSON}}", json.dumps(images, ensure_ascii=False))
    html = html.replace("{{BOXES_JSON}}", json.dumps(boxes, ensure_ascii=False))

    out_html.write_text(html, encoding="utf-8")
    print(f"已生成综合审查界面: {out_html}")
    print("打开方式: 用浏览器直接打开该 HTML 文件即可开始审查")


if __name__ == "__main__":
    main()
