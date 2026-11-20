'use strict';

// 与后端 sam3label/postprocess.py 保持一致
const PALETTE = [
  [255, 59, 48], [52, 199, 89], [0, 122, 255], [255, 149, 0],
  [175, 82, 222], [255, 204, 0], [90, 200, 250], [255, 45, 85],
  [88, 86, 214], [0, 199, 190], [142, 142, 147], [162, 132, 94],
  [50, 173, 230], [255, 105, 180], [60, 179, 113], [218, 112, 214],
];

const state = {
  projectClasses: [],   // 项目级类别（配色 + 批量默认）
  settings: {},
  images: [],
  filter: 'all',
  q: '',
  current: null,        // {id, ann, img, ov, editMask}
  selected: null,
  hovered: null,
  mode: 'select',       // select | box | brush
  brush: { size: 18, erase: false },
  edit: null,           // {instId, add, sub, strokes:[], erase}
  view: { scale: 1, tx: 0, ty: 0, fitted: true },
  jobId: null,
  engine: null,
  _drag: null,          // 框选橡皮筋（图像坐标）
  _pan: null,
  _spaceDown: false,
  _paint: null,
  _ovToken: 0,
  showMasks: true,      // 显示掩码
  showBoxes: true,      // 显示方框
  hideClasses: new Set(), // 视图层隐藏的类别（跨图片保持）
  hideInst: new Set(),    // 视图层隐藏的实例（按图片切换时清空）
  collapsed: new Set(),        // 折叠起来的类别（右侧实例列表）
  collapsedGroups: new Set(),  // 折叠起来的缩略图分组（左侧）
  order: [],                   // 左侧实际渲染顺序（键盘导航用这个）
  page: { offset: 0, size: 400, total: 0, groups: [], loading: false },
  importSel: new Set(),   // 「添加图片」里勾选的单个文件
  selImages: new Set(),   // 左侧列表里勾选、待批量移除的图片
};

const PREF_KEY = 'sam3label.prefs';
function loadPrefs() {
  try {
    const p = JSON.parse(localStorage.getItem(PREF_KEY) || '{}');
    if (typeof p.showMasks === 'boolean') state.showMasks = p.showMasks;
    if (typeof p.showBoxes === 'boolean') state.showBoxes = p.showBoxes;
  } catch (e) { /* ignore */ }
}
function savePrefs() {
  try {
    localStorage.setItem(PREF_KEY, JSON.stringify({ showMasks: state.showMasks, showBoxes: state.showBoxes }));
  } catch (e) { /* ignore */ }
}

// 掩码叠加层缓存：key = 图片id|标注版本|高亮实例
const ovCache = new Map();
// 命中测试用的标签图缓存（像素值 = 实例 id）
const labelCache = new Map();
const LABEL_MAX = 2400;

function visibilityParams() {
  const params = new URLSearchParams();
  params.set('hi', state.selected ?? '');
  for (const id of state.hideInst) params.append('hide', String(id));
  for (const cls of state.hideClasses) params.append('hcls', cls);
  return params;
}

function overlayKey(ann) {
  const hid = [...state.hideInst].sort((a, b) => a - b).join(',');
  const hcls = [...state.hideClasses].sort().join(',');
  return `${IMAGE_ID_OF(ann)}|${ann.updated || ''}|${state.selected ?? ''}|${hid}|${hcls}`;
}

function overlayUrl(ann) {
  const params = visibilityParams();
  params.set('v', ann.updated || '');
  return `/media/overlay/${IMAGE_ID_OF(ann)}?${params.toString()}`;
}

// 实例是否在"视图"上可见（kept 是语义排除，hideXxx 只是显示过滤）
function isVisible(inst) {
  return inst.kept
    && !state.hideInst.has(inst.id)
    && !state.hideClasses.has(inst.class_name);
}
function IMAGE_ID_OF(ann) { return ann.image_id || (state.current && state.current.id) || ''; }

const el = (id) => document.getElementById(id);
const $canvas = el('canvas');
const $ctx = $canvas.getContext('2d');

// ------------------------------------------------------------------ 网络
async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('json') ? res.json() : res;
}
const GET = (p) => api(p);
const POST = (p, body) => api(p, { method: 'POST', body: JSON.stringify(body || {}) });
const DELETE = (p) => api(p, { method: 'DELETE' });

function toast(msg, isErr = false) {
  const t = el('toast');
  t.textContent = msg;
  t.className = 'on' + (isErr ? ' err' : '');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.className = ''; }, isErr ? 6000 : 3000);
}
function busy(on) { el('loading').classList.toggle('on', !!on); }

// ------------------------------------------------------------------ 颜色
function classColor(name) {
  const custom = (state.classColors || {})[name];
  if (custom && /^#[0-9a-fA-F]{6}$/.test(custom)) return custom;
  const all = Array.from(new Set([...state.projectClasses, ...currentClasses()]));
  const i = all.indexOf(name);
  const c = PALETTE[(i >= 0 ? i : all.length) % PALETTE.length];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function classColorHex(name) {
  const c = classColor(name);
  if (c.startsWith('#')) return c;
  const m = c.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (!m) return '#888888';
  return '#' + [1, 2, 3].map((k) => Number(m[k]).toString(16).padStart(2, '0')).join('');
}

// ------------------------------------------------------------------ 初始化
async function boot() {
  try {
    const s = await GET('/api/state');
    state.projectClasses = s.classes || [];
    state.classColors = s.class_colors || {};
    state.settings = s.settings || {};
    state.engine = s.engine;
    el('projName').textContent = s.project.name || '';
    const verMeta = (document.querySelector('meta[name="app-ver"]') || {}).content || '?';
    el('verBadge').textContent = `前端 v${verMeta}`;
    el('conf').value = state.settings.confidence ?? 0.5;
    el('confVal').textContent = el('conf').value;
    loadPrefs();
    setShowMasks(state.showMasks);
    setShowBoxes(state.showBoxes);
    el('exportPath').value = `${s.project.root}/exports/export-${stamp()}`;
    renderStats(s.project);
    renderEngine(s);
    if (s.load_error) toast('模型加载失败：' + s.load_error, true);
    await loadImages();
    if (state.images.length) await openImage(state.images[0].id);
    else showClasses(state.projectClasses);
    // 刷新页面后自动恢复显示正在跑的任务
    try {
      const r = await GET('/api/jobs');
      const live = (r.jobs || []).find((j) => ['running', 'paused', 'canceling'].includes(j.state));
      if (live) { renderJobBar(live); pollJob(live.id); }
    } catch (e) { /* ignore */ }
  } catch (e) {
    toast('初始化失败：' + e.message, true);
  }
}

function stamp() {
  const d = new Date(), p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

async function refreshStats() {
  try { renderStats(await GET('/api/stats')); } catch (e) { /* ignore */ }
}

function renderStats(p) {
  const miss = p.missing ? ` · <span style="color:#f59e0b">缺失 <b>${p.missing}</b></span>` : '';
  el('stats').innerHTML =
    `图片 <b>${p.images}</b> · 已标 <b>${p.done}</b> · 已复核 <b>${p.reviewed}</b> · ` +
    `待处理 <b>${p.pending}</b> · 实例 <b>${p.instances}</b>${miss}`;
  state.lastStats = p;
}

function renderEngine(s) {
  const e = s.engine;
  const box = el('engineInfo');
  if (!e) { box.textContent = '模型未加载'; return; }
  const gpu = e.gpu ? `${e.gpu.name} ${e.gpu.allocated_gb}/${e.gpu.total_gb}GB` : e.device;
  const ver = (document.querySelector('meta[name="app-ver"]') || {}).content || '?';
  box.textContent = `${gpu} · 分辨率 ${e.resolution} · 加载 ${e.load_seconds}s · 前端 v${ver}`;
  setTimeout(async () => {
    try { const x = await GET('/api/state'); state.engine = x.engine; renderEngine(x); } catch (err) {}
  }, 15000);
}

// ------------------------------------------------------------------ 类别输入（跟随图片）
function currentClasses() {
  const c = state.current;
  if (c && c.ann && Array.isArray(c.ann.classes) && c.ann.classes.length) return c.ann.classes;
  return [];
}

function showClasses(list) {
  el('classesInput').value = (list || []).join(',');
}

function updateClassInput(force = false) {
  if (state._classesDirty && !force) return;
  if (currentClasses().length) {
    showClasses(currentClasses());
    el('classesInput').placeholder = '当前图片的类别提示词，逗号分隔';
  } else {
    showClasses(state.projectClasses);
    el('classesInput').placeholder = '项目类别（本图还没标注，可直接用这批类别跑）';
  }
}

function inputClasses() {
  return el('classesInput').value.split(/[,，]/).map((s) => s.trim()).filter(Boolean);
}

// ------------------------------------------------------------------ 缩略图
async function loadImages(reset = true) {
  if (state.page.loading) return;
  state.page.loading = true;
  if (reset) {
    state.images = [];
    state.page.offset = 0;
  }
  const q = encodeURIComponent(state.q);
  try {
    const data = await GET(
      `/api/images?status=${state.filter}&q=${q}&offset=${state.page.offset}&limit=${state.page.size}`
    );
    state.page.total = data.total || 0;
    state.page.groups = data.groups || [];
    state.images = reset ? (data.items || []) : state.images.concat(data.items || []);
    state.page.offset = state.images.length;
    renderThumbs();
  } catch (e) {
    toast('加载图片失败：' + e.message, true);
  } finally {
    state.page.loading = false;
    updatePageBar();
  }
}

async function loadMoreImages() {
  if (state.page.loading || state.images.length >= state.page.total) return;
  await loadImages(false);
}

function thumbEl(it) {
  const d = document.createElement('div');
  d.className = 'thumb' + (state.current && state.current.id === it.id ? ' sel' : '');
  d.dataset.id = it.id;
  d.title = it.relpath;

  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.className = 'tcb';
  cb.title = '勾选后可批量移除';
  cb.checked = state.selImages.has(it.id);
  if (cb.checked) d.classList.add('picked');
  cb.onclick = (ev) => ev.stopPropagation();
  cb.onchange = (ev) => {
    ev.stopPropagation();
    if (cb.checked) state.selImages.add(it.id);
    else state.selImages.delete(it.id);
    d.classList.toggle('picked', cb.checked);
    renderSelBar();
    renderGroupChecks();
  };

  const img = document.createElement('img');
  img.src = it.thumb;
  img.loading = 'lazy';
  img.onerror = () => { d.style.opacity = .35; img.removeAttribute('src'); };
  d.appendChild(img);

  const b = document.createElement('div');
  b.className = 'badge ' + it.status;
  b.textContent = it.status === 'pending' ? '待标' : it.status === 'done' ? '已标' : '复核';
  d.appendChild(b);
  if (it.n_instances) {
    const c = document.createElement('div');
    c.className = 'cnt';
    c.textContent = it.n_instances;
    d.appendChild(c);
  }
  d.appendChild(cb);  // 放在最后，保证盖在图片上面可点击
  d.onclick = () => openImage(it.id);
  return d;
}

function groupBlock(g) {
  const box = document.createElement('div');
  box.className = 'tgroup';
  const collapsed = state.collapsedGroups.has(g.key);

  const head = document.createElement('div');
  head.className = 'tgroup-head';

  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.className = 'vis gcb';
  cb.dataset.group = g.key;
  cb.title = '全选 / 取消全选该组';
  const ids = g.items.map((i) => i.id);
  const picked = ids.filter((id) => state.selImages.has(id)).length;
  cb.checked = picked > 0 && picked === ids.length;
  cb.indeterminate = picked > 0 && picked < ids.length;
  cb.onclick = (ev) => ev.stopPropagation();
  cb.onchange = (ev) => {
    ev.stopPropagation();
    if (cb.checked) ids.forEach((id) => state.selImages.add(id));
    else ids.forEach((id) => state.selImages.delete(id));
    renderThumbs();
    renderSelBar();
  };
  head.appendChild(cb);

  const icon = document.createElement('span');
  icon.textContent = g.kind === 'video' ? '🎬' : '📁';
  head.appendChild(icon);

  const name = document.createElement('span');
  name.className = 'gname';
  name.textContent = g.name;
  name.title = g.name;
  head.appendChild(name);

  const n = document.createElement('span');
  n.className = 'gn';
  const totalOfGroup = (state.page.groups.find((x) => x.key === g.key) || {}).count;
  n.textContent = totalOfGroup && totalOfGroup > g.items.length
    ? `${g.items.length}/${totalOfGroup}`
    : String(g.items.length);
  n.title = '已显示 / 该组总数';
  head.appendChild(n);

  const arrow = document.createElement('span');
  arrow.className = 'arrow';
  arrow.textContent = collapsed ? '▸' : '▾';
  head.appendChild(arrow);

  const del = document.createElement('button');
  del.className = 'gdel';
  del.textContent = '🗑';
  del.title = '删除整组（含磁盘上的抽帧文件）';
  del.onclick = async (ev) => {
    ev.stopPropagation();
    ids.forEach((id) => state.selImages.add(id));
    renderThumbs();
    renderSelBar();
    await removeSelected(true, { groupName: g.name, count: ids.length });
  };
  head.appendChild(del);

  head.onclick = (ev) => {
    if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'BUTTON') return;
    if (collapsed) state.collapsedGroups.delete(g.key);
    else state.collapsedGroups.add(g.key);
    renderThumbs();
  };
  box.appendChild(head);

  if (!collapsed) {
    const body = document.createElement('div');
    body.className = 'tgroup-body';
    const grid = document.createElement('div');
    grid.className = 'tgrid';
    for (const it of g.items) grid.appendChild(thumbEl(it));
    body.appendChild(grid);
    box.appendChild(body);
  }
  return box;
}

let thumbObserver = null;

function updatePageBar() {
  const shown = state.images.length;
  const total = state.page.total || shown;
  const bar = el('pageBar');
  const need = shown < total;
  bar.classList.toggle('hidden', !need);
  if (need) {
    el('pageText').textContent = `已显示 ${shown} / 共 ${total} 张`;
    el('pageMore').textContent = state.page.loading ? '加载中…' : '加载更多';
    el('pageMore').disabled = state.page.loading;
  }
}

function setupInfiniteScroll() {
  if (thumbObserver) { thumbObserver.disconnect(); thumbObserver = null; }
  const box = el('thumbs');
  const sentinel = box.querySelector('.load-sentinel');
  if (!sentinel || typeof IntersectionObserver === 'undefined') return;
  thumbObserver = new IntersectionObserver(
    (entries) => {
      if (entries.some((e) => e.isIntersecting) && !state.page.loading) loadMoreImages();
    },
    { root: box, rootMargin: '800px' }
  );
  thumbObserver.observe(sentinel);
}

el('pageMore').onclick = () => loadMoreImages();

function renderGroupChecks() {
  for (const cb of document.querySelectorAll('#thumbs .gcb')) {
    const key = cb.dataset.group;
    const grp = document.querySelector(`.tgroup[data-key="${CSS.escape(key)}"]`);
    if (!grp) continue;
    const ids = [...grp.querySelectorAll('.thumb')].map((t) => t.dataset.id);
    const picked = ids.filter((id) => state.selImages.has(id)).length;
    cb.checked = picked > 0 && picked === ids.length;
    cb.indeterminate = picked > 0 && picked < ids.length;
  }
}

function renderThumbs() {
  const wrap = el('thumbs');
  wrap.innerHTML = '';
  state.order = [];
  const groups = new Map();
  const loose = [];
  for (const it of state.images) {
    if (it.group_path) {
      if (!groups.has(it.group_path)) {
        groups.set(it.group_path, {
          key: it.group_path,
          name: it.group || it.group_path,
          kind: it.group_kind || (it.group_path.startsWith('video:') ? 'video' : 'folder'),
          items: [],
        });
      }
      groups.get(it.group_path).items.push(it);
    } else {
      loose.push(it);
    }
  }
  for (const g of groups.values()) {
    const box = groupBlock(g);
    box.dataset.key = g.key;
    wrap.appendChild(box);
    for (const it of g.items) state.order.push(it.id);
  }
  if (loose.length) {
    if (groups.size) {
      const lbl = document.createElement('div');
      lbl.className = 'tgroup-label';
      lbl.textContent = '未分组';
      wrap.appendChild(lbl);
    }
    const grid = document.createElement('div');
    grid.className = 'tgrid';
    for (const it of loose) {
      grid.appendChild(thumbEl(it));
      state.order.push(it.id);
    }
    wrap.appendChild(grid);
  }
  if (state.images.length < state.page.total) {
    const more = document.createElement('button');
    more.className = 'load-more';
    more.textContent =
      `加载更多（已显示 ${state.images.length} / 共 ${state.page.total} 张）`;
    more.onclick = () => loadMoreImages();
    wrap.appendChild(more);
    const sentinel = document.createElement('div');
    sentinel.className = 'load-sentinel';
    wrap.appendChild(sentinel);
  }
  updatePageBar();
  setupInfiniteScroll();
  if (!state.images.length) {
    wrap.innerHTML = '<div style="color:#8b95a7;padding:8px;font-size:12px">没有符合条件的图片</div>';
  }
  const visible = new Set(state.images.map((i) => i.id));
  for (const id of [...state.selImages]) if (!visible.has(id)) state.selImages.delete(id);
  renderSelBar();
}

function renderSelBar() {
  const n = state.selImages.size;
  el('selBar').classList.remove('hidden');  // 常驻，别让人找不到
  el('selCount').textContent = String(n);
  el('selHint').textContent = n === 0 ? '勾选缩略图右上角方框可批量移除' : '';
  for (const id of ['selRemove', 'selRemoveFiles']) el(id).disabled = n === 0;
  for (const id of ['selClear', 'selAll']) el(id).disabled = state.images.length === 0;
}

el('selAll').onclick = () => {
  for (const it of state.images) state.selImages.add(it.id);
  renderThumbs();
  renderSelBar();
};
el('selClear').onclick = () => {
  state.selImages.clear();
  renderThumbs();
  renderSelBar();
};
el('selRemove').onclick = () => removeSelected(false);
el('selRemoveFiles').onclick = () => removeSelected(true);

async function removeSelected(deleteFiles, opts = {}) {
  const ids = [...state.selImages];
  if (!ids.length) { toast('先勾选要移除的图片', true); return; }
  const what = opts.groupName ? `整组「${opts.groupName}」共 ${ids.length} 张` : `${ids.length} 张图片`;
  const msg = deleteFiles
    ? `移除${what}，并删除项目内的文件（抽帧图片会真删）？\n（项目外引用的原图不会被删除，只解除引用）`
    : `从工作区移除${what}？\n（只删清单和标注，磁盘文件保留）`;
  if (!confirm(msg)) return;
  try {
    busy(true);
    const r = await POST('/api/images/remove', { ids, delete_files: deleteFiles });
    let msg2 = `已移除 ${r.removed.length} 张`;
    if (r.deleted_files.length) msg2 += `，删文件 ${r.deleted_files.length} 个`;
    if (r.kept_external.length) msg2 += `，外部原图保留 ${r.kept_external.length} 个`;
    toast(msg2 + `；剩余 ${r.total} 张`);
    const removedCurrent = state.current && ids.includes(state.current.id);
    state.selImages.clear();
    if (removedCurrent) closeCurrent();
    await loadImages();
    await refreshStats();
    renderSelBar();
  } catch (e) {
    toast('移除失败：' + e.message, true);
  } finally { busy(false); }
}

function closeCurrent() {
  state.current = null;
  state.selected = null;
  state.hovered = null;
  state.edit = null;
  el('editBanner').classList.add('hidden');
  el('instList').innerHTML = '';
  el('instCount').textContent = '';
  el('hint').textContent = '';
  const dpr = window.devicePixelRatio || 1;
  $ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  $ctx.clearRect(0, 0, $canvas.width / dpr, $canvas.height / dpr);
  el('empty').style.display = '';
  ovCache.clear();
  labelCache.clear();
}

function markThumb(id) {
  for (const d of el('thumbs').children) {
    if (!d.classList) continue;
    d.classList.toggle('sel', d.dataset && d.dataset.id === id);
  }
}

// ------------------------------------------------------------------ 视图（缩放/平移）
function fitScale() {
  const c = state.current;
  if (!c) return 1;
  const w = $canvas.clientWidth || 1, h = $canvas.clientHeight || 1;
  return Math.min(w / c.W, h / c.H) * 0.98;
}

function fitView() {
  const c = state.current;
  if (!c) return;
  const w = $canvas.clientWidth || 1, h = $canvas.clientHeight || 1;
  const s = fitScale();
  state.view = { scale: s, tx: (w - c.W * s) / 2, ty: (h - c.H * s) / 2, fitted: true };
  updateZoomLabel();
}

function updateZoomLabel() {
  el('zoomLabel').textContent = Math.round(state.view.scale * 100) + '%';
}

function zoomAt(factor, cx, cy) {
  const v = state.view;
  const s0 = v.scale;
  const s1 = Math.min(Math.max(s0 * factor, fitScale() * 0.2), 40);
  v.tx = cx - (cx - v.tx) * (s1 / s0);
  v.ty = cy - (cy - v.ty) * (s1 / s0);
  v.scale = s1;
  v.fitted = false;
  updateZoomLabel();
  draw();
}

function screenToImage(ev) {
  const r = $canvas.getBoundingClientRect();
  const cx = ev.clientX - r.left, cy = ev.clientY - r.top;
  return { x: (cx - state.view.tx) / state.view.scale, y: (cy - state.view.ty) / state.view.scale };
}

// ------------------------------------------------------------------ 打开图片
function loadImg(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error('图片加载失败: ' + url));
    im.src = url;
  });
}

async function openImage(id) {
  try {
    busy(true);
    resetEditMode();
    const ann = await GET(`/api/images/${id}`);
    const img = await loadImg(`/media/image/${id}?max_size=3000`);
    ovCache.clear();
    state.current = {
      id, ann, img, ov: null,
      W: ann.width || img.naturalWidth,
      H: ann.height || img.naturalHeight,
    };
    state.selected = null;
    state.hovered = null;
    state.hideInst = new Set();
    state.selImages.delete(id);
    renderSelBar();
    state._classesDirty = false;
    markThumb(id);
    // 类别跟随当前图片（没有标注过就退回项目类别）
    updateClassInput();
    resizeCanvas();
    fitView();
    draw();
    renderInstances();
    await refreshOverlay();
    refreshLabelMap();
    closeCtx();
    el('empty').style.display = 'none';
    el('hint').textContent =
      `${ann.relpath} · ${ann.width}×${ann.height} · ${ann.instances.filter((i) => i.kept).length} 实例` +
      (ann.params && ann.params.confidence != null ? ` · 阈值 ${ann.params.confidence}` : '');
  } catch (e) {
    toast('打开失败：' + e.message, true);
  } finally {
    busy(false);
  }
}

// ------------------------------------------------------------------ 画布尺寸
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, $canvas.clientWidth), h = Math.max(1, $canvas.clientHeight);
  $canvas.width = Math.round(w * dpr);
  $canvas.height = Math.round(h * dpr);
}

window.addEventListener('resize', () => {
  const wasFit = state.view.fitted;
  resizeCanvas();
  if (wasFit) fitView();
  draw();
});

// ------------------------------------------------------------------ 绘制
function draw() {
  const dpr = window.devicePixelRatio || 1;
  const v = state.view;
  const cw = $canvas.width / dpr, ch = $canvas.height / dpr;
  $ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  $ctx.clearRect(0, 0, cw, ch);

  const c = state.current;
  if (!c) { el('empty').style.display = ''; return; }
  el('empty').style.display = 'none';

  const s = v.scale;
  $ctx.setTransform(dpr * s, 0, 0, dpr * s, dpr * v.tx, dpr * v.ty);
  const w = c.W, h = c.H;

  // 底图（按标注坐标系铺满，原图比 max_size 大时会自动缩放）
  $ctx.drawImage(c.img, 0, 0, w, h);
  if (state.showMasks && c.ov) $ctx.drawImage(c.ov, 0, 0, w, h);

  // 画笔精修图层
  const e = state.edit;
  if (e) {
    const aw = c.ann.width || w, ah = c.ann.height || h;
    if (e.maskImg) {
      $ctx.globalAlpha = 0.32;
      $ctx.drawImage(e.maskImg, 0, 0, aw, ah);
      $ctx.globalAlpha = 1;
    }
    $ctx.globalAlpha = 0.72;
    $ctx.drawImage(e.add, 0, 0, aw, ah);
    $ctx.drawImage(e.sub, 0, 0, aw, ah);
    $ctx.globalAlpha = 1;
    $ctx.strokeStyle = 'rgba(255,255,255,.55)';
    $ctx.lineWidth = 1 / s;
    $ctx.strokeRect(0, 0, aw, ah);
  }

  const base = Math.max(1, 1 / s);
  $ctx.textBaseline = 'top';
  $ctx.font = `${Math.max(11, 12 / s)}px -apple-system, "Noto Sans CJK SC", sans-serif`;
  for (const inst of c.ann.instances) {
    if (!state.showBoxes || !isVisible(inst)) continue;
    const [x0, y0, x1, y1] = inst.box;
    const sel = inst.id === state.selected;
    const hov = inst.id === state.hovered;
    $ctx.strokeStyle = sel ? '#ffffff' : hov ? 'rgba(255,255,255,.85)' : classColor(inst.class_name);
    $ctx.lineWidth = sel ? base * 2.6 : hov ? base * 1.8 : base;
    $ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    if (sel || hov) {
      const label = `${inst.class_name} ${(inst.score ?? 0).toFixed(2)}`;
      const tw = $ctx.measureText(label).width + 10 / s;
      const lh = Math.max(15, 16 / s);
      const ly = Math.max(0, y0 - lh);
      $ctx.fillStyle = sel ? 'rgba(255,255,255,.94)' : 'rgba(15,18,26,.88)';
      $ctx.fillRect(x0, ly, tw, lh);
      $ctx.fillStyle = sel ? '#10131a' : '#ffffff';
      $ctx.fillText(label, x0 + 4 / s, ly + 2 / s);
    }
  }

  // 框选橡皮筋
  if (state._drag) {
    const d = state._drag;
    $ctx.setLineDash([6 / s, 4 / s]);
    $ctx.strokeStyle = '#f59e0b';
    $ctx.lineWidth = 2.5 / s;
    $ctx.strokeRect(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0));
    $ctx.setLineDash([]);
  }

  // 画笔光标
  if (state.mode === 'brush' && state._brushPos) {
    const p = state._brushPos;
    const r = state.brush.size / 2 / s;
    $ctx.beginPath();
    $ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    $ctx.strokeStyle = state.brush.erase ? 'rgba(255,90,90,.95)' : 'rgba(120,255,150,.95)';
    $ctx.lineWidth = 1.5 / s;
    $ctx.stroke();
  }
}

// ------------------------------------------------------------------ 掩码叠加层
async function refreshOverlay() {
  const c = state.current;
  if (!c) return;
  if (!state.showMasks) { c.ov = null; draw(); return; }
  const key = overlayKey(c.ann);
  const hit = ovCache.get(key);
  if (hit) { c.ov = hit; draw(); return; }
  const token = ++state._ovToken;
  try {
    const img = await loadImg(overlayUrl(c.ann));
    ovCache.set(key, img);
    while (ovCache.size > 24) ovCache.delete(ovCache.keys().next().value);
    if (token === state._ovToken && state.current === c) { c.ov = img; draw(); }
  } catch (e) {
    if (token === state._ovToken && state.current === c) { c.ov = null; draw(); }
  }
}

async function refreshLabelMap() {
  const c = state.current;
  if (!c) return;
  const hid = [...state.hideInst].sort((a, b) => a - b).join(',');
  const hcls = [...state.hideClasses].sort().join(',');
  const key = `lm|${IMAGE_ID_OF(c.ann)}|${c.ann.updated || ''}|${hid}|${hcls}|${LABEL_MAX}`;
  const hit = labelCache.get(key);
  if (hit) { c.label = hit; return; }
  const params = visibilityParams();
  params.delete('hi');            // 标签图和高亮无关
  params.set('max_size', String(LABEL_MAX));
  try {
    const img = await loadImg(`/api/images/${c.id}/labelmap?${params.toString()}`);
    const cv = document.createElement('canvas');
    cv.width = img.naturalWidth;
    cv.height = img.naturalHeight;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    const rec = { w: cv.width, h: cv.height, data: ctx.getImageData(0, 0, cv.width, cv.height).data };
    labelCache.set(key, rec);
    while (labelCache.size > 8) labelCache.delete(labelCache.keys().next().value);
    if (state.current === c) c.label = rec;
  } catch (e) {
    if (state.current === c) c.label = null;
  }
}

function labelIdAt(pt) {
  const c = state.current;
  const L = c && c.label;
  if (!L) return null;
  const x = Math.floor((pt.x / c.W) * L.w);
  const y = Math.floor((pt.y / c.H) * L.h);
  if (x < 0 || y < 0 || x >= L.w || y >= L.h) return null;
  const i = (y * L.w + x) * 4;
  const id = L.data[i] | (L.data[i + 1] << 8) | (L.data[i + 2] << 16);
  return id || null;
}

function setSelected(id) {
  if (state.selected === id) return;
  state.selected = id;
  if (state.edit && state.edit.instId !== id) exitBrush(false);
  draw();
  renderInstances();
  refreshOverlay();
}

// ------------------------------------------------------------------ 命中测试
function hitTest(pt) {
  const c = state.current;
  if (!c) return null;
  // 首选：按掩码像素命中（点在哪个实例的掩码里就选哪个）
  const id = labelIdAt(pt);
  if (id != null) {
    const inst = c.ann.instances.find((i) => i.id === id);
    if (inst && isVisible(inst)) return inst;
  }
  if (c.label) return null;  // 标签图可用时严格按掩码，落在背景/空洞就不选
  // 标签图没加载出来时退回外框命中，保证还能操作
  let best = null;
  for (const inst of c.ann.instances) {
    if (!isVisible(inst)) continue;
    const [x0, y0, x1, y1] = inst.box;
    if (pt.x >= x0 && pt.x <= x1 && pt.y >= y0 && pt.y <= y1) {
      if (!best || inst.area < best.area) best = inst;
    }
  }
  return best;
}

// ------------------------------------------------------------------ 交互
$canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  if (!state.current) return;
  const r = $canvas.getBoundingClientRect();
  const factor = Math.exp(-ev.deltaY * 0.0015);
  zoomAt(factor, ev.clientX - r.left, ev.clientY - r.top);
}, { passive: false });

$canvas.addEventListener('contextmenu', (ev) => {
  ev.preventDefault();
  if (!state.current) return;
  const hit = hitTest(screenToImage(ev));
  if (hit) {
    setSelected(hit.id);
    showCtx(ev.clientX, ev.clientY, hit);
  }
});

$canvas.addEventListener('mousedown', (ev) => {
  if (!state.current) return;
  closeCtx();
  const pt = screenToImage(ev);
  const panning = ev.button === 1 || state._spaceDown ||
    (ev.button === 0 && state.mode === 'select' && !hitTest(pt));
  if (panning) {
    state._pan = { x: ev.clientX, y: ev.clientY, tx: state.view.tx, ty: state.view.ty };
    state._panMoved = 0;
    $canvas.style.cursor = 'grabbing';
    ev.preventDefault();
    return;
  }
  if (ev.button !== 0) return;

  if (state.mode === 'box') {
    state._drag = { x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y };
    draw();
    return;
  }
  if (state.mode === 'brush') {
    if (!state.edit) { toast('先在右侧实例上点「编辑掩码」选中一个实例', true); return; }
    state._paint = { erase: state.brush.erase, pts: [[pt.x, pt.y]] };
    paintDot(pt);
    return;
  }
  state._lastClick = pt;
  const hit = hitTest(pt);
  setSelected(hit ? hit.id : null);
  if (hit) showCtx(ev.clientX, ev.clientY, hit);
});

$canvas.addEventListener('mousemove', (ev) => {
  if (state._pan) {
    const v = state.view;
    const dx = ev.clientX - state._pan.x, dy = ev.clientY - state._pan.y;
    state._panMoved = (state._panMoved || 0) + Math.abs(dx) + Math.abs(dy);
    v.tx = state._pan.tx + dx;
    v.ty = state._pan.ty + dy;
    v.fitted = false;
    draw();
    return;
  }
  if (!state.current) return;
  const pt = screenToImage(ev);
  if (state._drag) {
    state._drag.x1 = pt.x;
    state._drag.y1 = pt.y;
    draw();
    return;
  }
  if (state._paint) {
    state._paint.pts.push([pt.x, pt.y]);
    paintDot(pt);
    return;
  }
  if (state.mode === 'brush') {
    state._brushPos = pt;
    draw();
    return;
  }
  const hit = hitTest(pt);
  const id = hit ? hit.id : null;
  if (id !== state.hovered) {
    state.hovered = id;
    $canvas.style.cursor = id ? 'pointer' : 'default';
    draw();
  }
});

$canvas.addEventListener('mouseleave', () => {
  state._brushPos = null;
  if (state.hovered != null) { state.hovered = null; draw(); }
});

$canvas.addEventListener('mouseup', async (ev) => {
  if (state._pan) {
    const moved = state._panMoved || 0;
    state._pan = null;
    $canvas.style.cursor = state.mode === 'brush' ? 'none' : 'crosshair';
    // 点在空白处（没有拖动）＝ 取消选中，高亮随之消失
    if (moved < 4 && ev.button === 0 && state.selected != null) setSelected(null);
    return;
  }
  if (state._paint) {
    const strokes = state._paint.pts;
    state._paint = null;
    if (state.edit) {
      state.edit.strokes.push({
        erase: state.edit._eraseStroke, w: state.brush.size / state.view.scale,
        pts: strokes, add: !state.edit._eraseStroke,
      });
    }
    return;
  }
  const d = state._drag;
  if (!d) return;
  state._drag = null;
  const c = state.current;
  const x0 = Math.min(d.x0, d.x1), x1 = Math.max(d.x0, d.x1);
  const y0 = Math.min(d.y0, d.y1), y1 = Math.max(d.y0, d.y1);
  if (x1 - x0 < 4 || y1 - y0 < 4) { draw(); return; }
  const hint = el('promptInput').value.trim() || (currentClasses()[0] || 'object');
  try {
    busy(true);
    const r = await POST(`/api/images/${c.id}/box_prompt`, {
      box: [x0, y0, x1, y1], class_name: hint, text_hint: hint,
      confidence: Number(el('conf').value),
    });
    toast(`框选补标：新增 ${r.added} 个实例`);
    await refreshCurrent();
  } catch (e) {
    toast('补标失败：' + e.message, true);
    draw();
  } finally { busy(false); }
});

// 空格临时平移 / 快捷键
// 只保留三个快捷键：a=上一张，d=下一张，Enter=已完成
// 监听放在 window 的「捕获阶段」，比扩展常挂在 document 上的冒泡监听更早拿到事件
let lastKeydownAt = 0;
let lastHandledKey = '';

function navStep(dir) {
  const c = state.current;
  if (!c) { toast('还没有打开图片', true); return false; }
  const order = (state.order && state.order.length) ? state.order : state.images.map((i) => i.id);
  const idx = order.findIndex((id) => id === c.id);
  if (idx < 0) {
    toast('当前图片不在左侧列表里（可能被搜索/筛选过滤了）', true);
    return false;
  }
  if (dir > 0) {
    if (idx < order.length - 1) { openImage(order[idx + 1]); return true; }
    loadMoreImages().then(() => {
      const o2 = (state.order && state.order.length) ? state.order : state.images.map((i) => i.id);
      if (idx < o2.length - 1) openImage(o2[idx + 1]);
      else toast('已经是最后一张了');
    });
    return true;
  }
  if (idx > 0) { openImage(order[idx - 1]); return true; }
  toast('已经是第一张了');
  return false;
}

function markReviewed() { el('btnReviewed').click(); }
el('btnPrev').onclick = () => navStep(-1);
el('btnNext').onclick = () => navStep(1);
window.__nav = { next: () => navStep(1), prev: () => navStep(-1), reviewed: markReviewed };

function keyInfo(ev) {
  return { code: ev.code || '', key: ev.key || '', composing: !!ev.isComposing, tag: (ev.target.tagName || '').toLowerCase() };
}

function showKey(ev, action) {
  const el2 = el('keyDebug');
  if (!el2) return;
  const k = keyInfo(ev);
  el2.textContent = `键 ${k.code || k.key}${k.composing ? '（输入法）' : ''} → ${action}`;
  el2.className = 'keydbg ' + (action.startsWith('✗') ? 'bad' : 'ok');
}

function handleKey(ev, viaKeyup) {
  const k = keyInfo(ev);
  const isA = k.code === 'KeyA' || k.key === 'a' || k.key === 'A';
  const isD = k.code === 'KeyD' || k.key === 'd' || k.key === 'D';
  const isEnter = k.code === 'Enter' || k.code === 'NumpadEnter' || k.key === 'Enter';
  if (!isA && !isD && !isEnter) { lastHandledKey = ''; return; }

  if (k.tag === 'input' || k.tag === 'textarea' || k.tag === 'select') {
    showKey(ev, '✗ 焦点在输入框');
    const now = Date.now();
    if (!window.__fieldHintAt || now - window.__fieldHintAt > 4000) {
      window.__fieldHintAt = now;
      toast('焦点在输入框里，先点一下图片区域再按 a/d', true);
    }
    lastHandledKey = '';
    return;
  }
  const sameKey = (k.code || k.key);
  if (viaKeyup && lastHandledKey === sameKey && Date.now() - lastKeydownAt < 600) {
    lastHandledKey = '';
    return;  // keydown 已经处理过，避免 keyup 兜底重复触发
  }
  lastHandledKey = sameKey;
  lastKeydownAt = Date.now();
  ev.preventDefault();
  if (isEnter) { showKey(ev, '✓ 已复核'); markReviewed(); return; }
  const ok = navStep(isD ? 1 : -1);
  showKey(ev, ok ? (viaKeyup ? '✓ 翻页(兜底)' : '✓ 翻页') : '✗ 到头了');
}

// Delete 删除选中实例，Ctrl+Z 撤销
document.addEventListener('keydown', (ev) => {
  const tag = (ev.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
  const isDel = ev.code === 'Delete' || ev.code === 'Backspace' || ev.key === 'Delete';
  const isUndo = (ev.ctrlKey || ev.metaKey) && (ev.code === 'KeyZ' || ev.key === 'z' || ev.key === 'Z');

  if (isUndo) {
    ev.preventDefault();
    undoDelete();
    return;
  }
  if (!isDel) return;
  ev.preventDefault();
  const c = state.current;
  if (!c) { toast('还没有打开图片', true); return; }
  if (state.selected == null) {
    toast('先点选一个实例，再按 Delete 删除', true);
    el('keyDebug').textContent = '键 Delete → ✗ 未选中实例';
    el('keyDebug').className = 'keydbg bad';
    return;
  }
  const id = state.selected;
  el('keyDebug').textContent = `键 Delete → ✓ 删除 #${id}`;
  el('keyDebug').className = 'keydbg ok';
  deleteInstance(id, true);
}, true);

async function undoDelete() {
  const c = state.current;
  if (!c) return;
  try {
    const r = await POST(`/api/images/${c.id}/undelete`, {});
    toast(`已撤销删除，恢复为 #${r.id}`);
    await refreshCurrent();
  } catch (e) {
    toast('没有可撤销的删除', true);
  }
}

window.addEventListener('keydown', (ev) => handleKey(ev, false), true);
window.addEventListener('keyup', (ev) => handleKey(ev, true), true);

// 页面里也能看到扩展是否拦了键：先按一下 a/d，看工具栏那行"键 …"有没有反应
document.addEventListener('keyup', (ev) => {
  if (ev.code === 'Space') {
    state._spaceDown = false;
    if (!state._pan) $canvas.style.cursor = state.mode === 'brush' ? 'none' : 'crosshair';
  }
});

// ------------------------------------------------------------------ 模式
function setMode(mode) {
  state.mode = mode;
  for (const [id, m] of [['modeSelect', 'select'], ['modeBox', 'box'], ['modeBrush', 'brush']]) {
    el(id).classList.toggle('active', mode === m);
  }
  el('brushOpts').classList.toggle('hidden', mode !== 'brush');
  $canvas.style.cursor = mode === 'brush' ? 'none' : 'crosshair';
  if (mode === 'brush' && !state.edit) {
    const sel = state.current && state.current.ann.instances.find((i) => i.id === state.selected);
    if (sel) startBrush(sel);
    else toast('提示：先在右侧点实例的「编辑」按钮，再画');
  }
  draw();
}
el('modeSelect').onclick = () => setMode('select');
el('modeBox').onclick = () => setMode('box');
el('modeBrush').onclick = () => setMode('brush');

function setBrushSize(v) {
  state.brush.size = Math.min(90, Math.max(2, v));
  el('brushSize').value = state.brush.size;
  el('brushSizeVal').textContent = state.brush.size;
  draw();
}
function setBrushErase(on) {
  state.brush.erase = on;
  el('brushAdd').classList.toggle('active', !on);
  el('brushErase').classList.toggle('active', on);
  draw();
}
el('brushSize').addEventListener('input', (e) => setBrushSize(Number(e.target.value)));
el('brushAdd').onclick = () => setBrushErase(false);
el('brushErase').onclick = () => setBrushErase(true);
el('brushUndo').onclick = () => undoStroke();
el('brushClear').onclick = () => { if (state.edit) { state.edit.strokes = []; repaintEditLayers(); } };
el('brushSave').onclick = () => saveBrush();

// ------------------------------------------------------------------ 画笔精修
function resetEditMode() {
  state.edit = null;
  state._paint = null;
  state._brushPos = null;
  el('editBanner').classList.add('hidden');
  if (state.mode === 'brush') setMode('select');
}

async function startBrush(inst) {
  const c = state.current;
  if (!c) return;
  try {
    busy(true);
    const aw = c.ann.width || c.img.naturalWidth;
    const ah = c.ann.height || c.img.naturalHeight;
    const maskImg = await loadImg(
      `/api/images/${c.id}/instances/${inst.id}/mask?color=255,255,255&alpha=255&max_size=2000`
    ).catch(() => null);
    const mk = () => {
      const cv = document.createElement('canvas');
      cv.width = aw; cv.height = ah;
      return cv;
    };
    state.edit = { instId: inst.id, add: mk(), sub: mk(), strokes: [], maskImg };
    state.selected = inst.id;
    el('editWho').textContent = `#${inst.id} ${inst.class_name}`;
    el('editBanner').classList.remove('hidden');
    setMode('brush');
    renderInstances();
    draw();
  } catch (e) {
    toast('进入精修失败：' + e.message, true);
  } finally { busy(false); }
}

function paintDot(pt) {
  const e = state.edit;
  if (!e) return;
  const erase = state.brush.erase;
  e._eraseStroke = erase;
  const ctx = (erase ? e.sub : e.add).getContext('2d');
  const r = state.brush.size / 2 / state.view.scale;
  ctx.fillStyle = erase ? 'rgb(255,60,60)' : 'rgb(60,255,120)';
  ctx.beginPath();
  ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
  ctx.fill();
  // 用线段连接保证连笔
  const pts = state._paint ? state._paint.pts : null;
  if (pts && pts.length >= 2) {
    const [x0, y0] = pts[pts.length - 2];
    ctx.strokeStyle = ctx.fillStyle;
    ctx.lineWidth = r * 2;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(pt.x, pt.y);
    ctx.stroke();
  }
  draw();
}

function repaintEditLayers() {
  const e = state.edit;
  if (!e) return;
  const addCtx = e.add.getContext('2d');
  const subCtx = e.sub.getContext('2d');
  addCtx.clearRect(0, 0, e.add.width, e.add.height);
  subCtx.clearRect(0, 0, e.sub.width, e.sub.height);
  for (const st of e.strokes) {
    const ctx = (st.add ? addCtx : subCtx);
    ctx.strokeStyle = st.add ? 'rgb(60,255,120)' : 'rgb(255,60,60)';
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = st.w;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    if (st.pts.length === 1) {
      const [x, y] = st.pts[0];
      ctx.arc(x, y, st.w / 2, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.moveTo(st.pts[0][0], st.pts[0][1]);
      for (let i = 1; i < st.pts.length; i++) ctx.lineTo(st.pts[i][0], st.pts[i][1]);
      ctx.stroke();
    }
  }
  draw();
}

function undoStroke() {
  if (!state.edit || !state.edit.strokes.length) return;
  state.edit.strokes.pop();
  repaintEditLayers();
}

async function saveBrush() {
  const e = state.edit;
  const c = state.current;
  if (!e || !c) return;
  if (!e.strokes.length) { toast('还没有涂抹内容'); return; }
  try {
    busy(true);
    const hasAdd = e.strokes.some((s) => s.add);
    const hasSub = e.strokes.some((s) => !s.add);
    const body = {
      w: e.add.width, h: e.add.height,
      add: hasAdd ? e.add.toDataURL('image/png') : null,
      sub: hasSub ? e.sub.toDataURL('image/png') : null,
    };
    const r = await POST(`/api/images/${c.id}/instances/${e.instId}/mask`, body);
    toast(`掩码已更新：#${e.instId} 面积 ${r.area}`);
    exitBrush(false);
    await refreshCurrent();
  } catch (err) {
    toast('保存掩码失败：' + err.message, true);
  } finally { busy(false); }
}

function exitBrush(reopen = true) {
  state.edit = null;
  state._paint = null;
  state._brushPos = null;
  el('editBanner').classList.add('hidden');
  if (reopen && state.mode === 'brush') setMode('select');
  draw();
  renderInstances();
}

// ------------------------------------------------------------------ 实例面板
function renderInstances() {
  const c = state.current;
  const list = el('instList');
  list.innerHTML = '';
  if (!c) { el('instCount').textContent = ''; return; }
  const insts = c.ann.instances || [];
  const kept = insts.filter((i) => i.kept);
  const visible = kept.filter(isVisible).length;
  const hiddenView = kept.length - visible;
  el('instCount').textContent = hiddenView
    ? `${visible}/${kept.length} · 视图隐藏 ${hiddenView}`
    : `${visible}/${kept.length}`;
  if (!insts.length) {
    list.innerHTML = '<div style="color:#8b95a7;padding:8px;font-size:12px">还没有实例。点「按类别重跑」或输入提示词。</div>';
    return;
  }

  // 按类别分组（顺序：项目类别 → 标注里的类别 → 其余按名字）
  const orderList = [...state.projectClasses, ...(c.ann.classes || [])];
  const groups = new Map();
  for (const inst of insts) {
    const key = inst.class_name || '?';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(inst);
  }
  const names = [...groups.keys()].sort((a, b) => {
    const ia = orderList.indexOf(a), ib = orderList.indexOf(b);
    const va = ia < 0 ? 1e9 : ia, vb = ib < 0 ? 1e9 : ib;
    return va - vb || a.localeCompare(b);
  });

  for (const name of names) {
    const items = groups.get(name);
    const nKept = items.filter((i) => i.kept).length;
    const nVisible = items.filter(isVisible).length;
    const classHidden = state.hideClasses.has(name);
    const collapsed = state.collapsed.has(name);

    const grp = document.createElement('div');
    grp.className = 'grp' + (classHidden ? ' grp-off' : '');

    const head = document.createElement('div');
    head.className = 'grp-head';

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.className = 'vis';
    cb.checked = !classHidden;
    cb.title = '显示 / 隐藏整个类别（只影响视图，不影响导出）';
    cb.onchange = (ev) => { ev.stopPropagation(); toggleClassView(name, !cb.checked); };
    head.appendChild(cb);

    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.style.background = classColor(name);
    head.appendChild(dot);

    const label = document.createElement('span');
    label.className = 'grp-name';
    label.textContent = name;
    head.appendChild(label);

    const cnt = document.createElement('span');
    cnt.className = 'grp-n';
    cnt.textContent = classHidden ? `${nKept} 已隐藏` : `${nVisible}/${nKept}`;
    head.appendChild(cnt);

    const arrow = document.createElement('span');
    arrow.className = 'arrow';
    arrow.textContent = collapsed ? '▸' : '▾';
    head.appendChild(arrow);

    head.onclick = (ev) => {
      if (ev.target === cb) return;
      toggleCollapse(name);
    };
    grp.appendChild(head);

    if (!collapsed) {
      const body = document.createElement('div');
      body.className = 'grp-body';
      for (const inst of items) body.appendChild(instanceRow(c, inst));
      grp.appendChild(body);
    }
    list.appendChild(grp);
  }
}

function instanceRow(c, inst) {
  const shown = isVisible(inst);
  const row = document.createElement('div');
  row.className = 'inst' + (inst.id === state.selected ? ' sel' : '')
    + (inst.kept ? '' : ' off') + (shown ? '' : ' hidden-view');

  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.className = 'vis';
  cb.checked = shown;
  cb.title = '显示 / 隐藏这个实例（只影响视图，不影响导出）';
  cb.onchange = (ev) => { ev.stopPropagation(); toggleInstView(inst.id, !cb.checked); };
  cb.onclick = (ev) => ev.stopPropagation();
  row.appendChild(cb);

  const dot = document.createElement('div');
  dot.className = 'dot';
  dot.style.background = classColor(inst.class_name);
  row.appendChild(dot);

  const meta = document.createElement('div');
  meta.className = 'meta';
  const cls = document.createElement('div');
  cls.className = 'cls';
  cls.textContent = inst.class_name;
  cls.title = '双击可改名';
  cls.ondblclick = async (ev) => { ev.stopPropagation(); await renameInstance(inst); };
  const sub = document.createElement('div');
  sub.className = 'sub';
  const flags = (inst.edited ? ' · 已精修' : '') + (inst.kept ? '' : ' · 已排除');
  sub.textContent = `#${inst.id} · ${(inst.score ?? 0).toFixed(2)} · ${inst.area}px² · ${inst.source}${flags}`;
  meta.appendChild(cls);
  meta.appendChild(sub);
  row.appendChild(meta);

  const acts = document.createElement('div');
  acts.className = 'acts';
  const edit = document.createElement('button');
  edit.textContent = '编辑';
  edit.title = '画笔精修这个实例的掩码';
  edit.onclick = async (ev) => { ev.stopPropagation(); await startBrush(inst); };
  const excl = document.createElement('button');
  excl.textContent = inst.kept ? '排除' : '恢复';
  excl.title = '从导出结果里排除（语义操作，和上面的显示勾选框不同）';
  excl.onclick = async (ev) => {
    ev.stopPropagation();
    await POST(`/api/images/${c.id}/instances/${inst.id}`, { kept: !inst.kept });
    await refreshCurrent();
  };
  const del = document.createElement('button');
  del.textContent = '删除';
  del.className = 'danger';
  del.onclick = async (ev) => { ev.stopPropagation(); await deleteInstance(inst.id); };
  acts.appendChild(edit);
  acts.appendChild(excl);
  acts.appendChild(del);
  row.appendChild(acts);

  row.onclick = (ev) => {
    if (ev.target.tagName === 'INPUT') return;
    setSelected(state.selected === inst.id ? null : inst.id);
  };
  row.onmouseenter = () => { state.hovered = inst.id; draw(); };
  row.onmouseleave = () => { state.hovered = null; draw(); };
  return row;
}

// ------------------------------------------------------------------ 视图开关
function afterViewChange() {
  const c = state.current;
  if (c && state.selected != null) {
    const sel = c.ann.instances.find((i) => i.id === state.selected);
    if (!sel || !isVisible(sel)) state.selected = null;
  }
  renderInstances();
  draw();
  refreshOverlay();
  refreshLabelMap();
}

function toggleClassView(cls, hidden) {
  if (hidden) state.hideClasses.add(cls); else state.hideClasses.delete(cls);
  afterViewChange();
}

function toggleInstView(id, hidden) {
  if (hidden) state.hideInst.add(id); else state.hideInst.delete(id);
  afterViewChange();
}

function toggleCollapse(cls) {
  if (state.collapsed.has(cls)) state.collapsed.delete(cls);
  else state.collapsed.add(cls);
  renderInstances();
}

function setShowMasks(v) {
  state.showMasks = !!v;
  savePrefs();
  el('btnMasks').classList.toggle('active', state.showMasks);
  el('btnMasks').textContent = state.showMasks ? '掩码 开' : '掩码 关';
  if (!state.showMasks && state.current) state.current.ov = null;
  draw();
  if (state.showMasks) refreshOverlay();
}

function setShowBoxes(v) {
  state.showBoxes = !!v;
  savePrefs();
  el('btnBoxes').classList.toggle('active', state.showBoxes);
  el('btnBoxes').textContent = state.showBoxes ? '方框 开' : '方框 关';
  draw();
}

function setAllViewVisible() {
  state.hideClasses.clear();
  state.hideInst.clear();
  afterViewChange();
}

async function deleteInstance(id, viaKey = false) {
  const c = state.current;
  if (!c || id == null) return;
  try {
    await DELETE(`/api/images/${c.id}/instances/${id}`);
  } catch (e) {
    toast('删除失败：' + e.message, true);
    return;
  }
  if (state.edit && state.edit.instId === id) exitBrush(false);
  if (state.selected === id) state.selected = null;
  toast(`已删除实例 #${id}${viaKey ? '（Ctrl+Z 撤销）' : ''}`);
  await refreshCurrent();
}

let renameTarget = null;

function openRename(inst) {
  const c = state.current;
  if (!c) return;
  renameTarget = { instId: inst.id, current: inst.class_name };
  const names = [];
  for (const n of [...state.projectClasses, ...currentClasses(), inst.class_name]) {
    if (n && !names.includes(n)) names.push(n);
  }
  const sel = el('renameSelect');
  sel.innerHTML = '';
  for (const n of names) {
    const o = document.createElement('option');
    o.value = n;
    o.textContent = `${n}${n === inst.class_name ? '（当前）' : ''}`;
    sel.appendChild(o);
  }
  sel.value = inst.class_name;
  el('renameCustom').value = '';
  el('renameInfo').textContent = `实例 #${inst.id} · 分数 ${(inst.score ?? 0).toFixed(2)} · 面积 ${inst.area}px²`;
  openModal('modalRename');
  setTimeout(() => sel.focus(), 50);
}

async function doRename() {
  const c = state.current;
  if (!c || !renameTarget) return;
  const custom = el('renameCustom').value.trim();
  const name = custom || el('renameSelect').value;
  closeModal('modalRename');
  if (!name || name === renameTarget.current) return;
  try {
    await POST(`/api/images/${c.id}/instances/${renameTarget.instId}`, { class_name: name });
    toast(`#${renameTarget.instId} 类别改为「${name}」`);
    await refreshCurrent();
  } catch (e) {
    toast('改名失败：' + e.message, true);
  } finally {
    renameTarget = null;
  }
}

async function renameInstance(inst) {
  openRename(inst);
}

el('renameCancel').onclick = () => { closeModal('modalRename'); renameTarget = null; };
el('renameOk').onclick = () => doRename();
el('renameSelect').ondblclick = () => doRename();
el('renameCustom').onkeydown = (ev) => { if (ev.key === 'Enter') { ev.preventDefault(); doRename(); } };

// ------------------------------------------------------------------ 浮动菜单
function showCtx(x, y, inst) {
  const menu = el('ctxmenu');
  el('ctxTitle').textContent = `#${inst.id} ${inst.class_name} · ${(inst.score ?? 0).toFixed(2)}`;
  menu.dataset.instId = inst.id;
  menu.classList.remove('hidden');
  const w = menu.offsetWidth || 180, h = menu.offsetHeight || 180;
  const left = Math.min(x, window.innerWidth - w - 8);
  const top = Math.min(y, window.innerHeight - h - 8);
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
}

function closeCtx() { el('ctxmenu').classList.add('hidden'); }

el('ctxmenu').addEventListener('click', async (ev) => {
  const act = ev.target.dataset && ev.target.dataset.act;
  if (!act) return;
  const id = Number(el('ctxmenu').dataset.instId);
  const c = state.current;
  const inst = c && c.ann.instances.find((i) => i.id === id);
  closeCtx();
  if (!inst) return;
  if (act === 'rename') await renameInstance(inst);
  else if (act === 'brush') await startBrush(inst);
  else if (act === 'toggle') {
    await POST(`/api/images/${c.id}/instances/${id}`, { kept: !inst.kept });
    await refreshCurrent();
  } else if (act === 'hideclass') {
    toggleClassView(inst.class_name, true);
  } else if (act === 'hideinst') {
    toggleInstView(inst.id, true);
  } else if (act === 'delete') await deleteInstance(id);
});
document.addEventListener('mousedown', (ev) => {
  if (!el('ctxmenu').contains(ev.target)) closeCtx();
}, true);

// ------------------------------------------------------------------ 刷新当前图
async function refreshCurrent() {
  const c = state.current;
  if (!c) return;
  const ann = await GET(`/api/images/${c.id}`);
  c.ann = ann;
  if (!ann.instances.some((i) => i.id === state.selected && i.kept)) state.selected = null;
  ovCache.clear();
  updateClassInput();
  draw();
  renderInstances();
  await refreshOverlay();
  await refreshLabelMap();
  const st = await GET('/api/stats');
  renderStats(st);
  const item = state.images.find((i) => i.id === c.id);
  if (item) { item.status = ann.status; item.n_instances = ann.instances.filter((i) => i.kept).length; }
  renderThumbs();
  markThumb(c.id);
}

// ------------------------------------------------------------------ 工具栏动作
el('conf').addEventListener('input', () => { el('confVal').textContent = el('conf').value; });
el('classesInput').addEventListener('input', () => { state._classesDirty = true; });

el('saveClasses').onclick = async () => {
  const list = inputClasses();
  if (!list.length) { toast('类别不能为空', true); return; }
  const r = await POST('/api/classes', { classes: list });
  state.projectClasses = r.classes;
  const c = state.current;
  if (c) {
    await POST(`/api/images/${c.id}/classes`, { classes: list });
    await refreshCurrent();
  }
  state._classesDirty = false;
  toast('类别已保存：' + state.projectClasses.join(', '));
};

el('btnRerun').onclick = async () => {
  const c = state.current;
  if (!c) return;
  const list = inputClasses();
  if (!list.length) { toast('先在顶部填写类别提示词', true); return; }
  try {
    busy(true);
    exitBrush(false);
    const r = await POST(`/api/images/${c.id}/segment`, {
      classes: list, confidence: Number(el('conf').value), replace: true,
    });
    state.projectClasses = Array.from(new Set([...state.projectClasses, ...list]));
    state._classesDirty = false;
    toast(`重跑完成：${r.added} 个实例`);
    await refreshCurrent();
  } catch (e) { toast('重跑失败：' + e.message, true); }
  finally { busy(false); }
};

el('btnPrompt').onclick = async () => {
  const c = state.current;
  const text = el('promptInput').value.trim();
  if (!c || !text) { toast('先输入提示词', true); return; }
  try {
    busy(true);
    const r = await POST(`/api/images/${c.id}/prompt`, { text, confidence: Number(el('conf').value) });
    if (!state.projectClasses.includes(text)) state.projectClasses.push(text);
    toast(`“${text}” 新增 ${r.added} 个实例`);
    await refreshCurrent();
  } catch (e) { toast('提示词失败：' + e.message, true); }
  finally { busy(false); }
};

el('btnDone').onclick = async () => {
  const c = state.current; if (!c) return;
  await POST(`/api/images/${c.id}/status`, { status: 'done' });
  toast('已标记完成');
  await refreshCurrent();
};
el('btnReviewed').onclick = async () => {
  const c = state.current; if (!c) return;
  await POST(`/api/images/${c.id}/status`, { status: 'reviewed' });
  toast('已标记已复核');
  await refreshCurrent();
};
el('btnClear').onclick = async () => {
  const c = state.current; if (!c || !confirm('清空这张图的全部实例？')) return;
  exitBrush(false);
  await POST(`/api/images/${c.id}/clear`);
  await refreshCurrent();
};

el('statusFilter').onchange = async (e) => { state.filter = e.target.value; await loadImages(); };
let searchTimer = null;
el('search').oninput = (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => { state.q = e.target.value.trim(); await loadImages(); }, 250);
};

el('btnMasks').onclick = () => setShowMasks(!state.showMasks);
el('btnBoxes').onclick = () => setShowBoxes(!state.showBoxes);
el('btnShowAll').onclick = () => setAllViewVisible();

el('zoomIn').onclick = () => zoomAt(1.25, $canvas.clientWidth / 2, $canvas.clientHeight / 2);
el('zoomOut').onclick = () => zoomAt(0.8, $canvas.clientWidth / 2, $canvas.clientHeight / 2);
el('zoomFit').onclick = () => { fitView(); draw(); };

// ------------------------------------------------------------------ 批量标注
el('btnBatch').onclick = async () => {
  const busyJob = await anyJobRunning();
  if (busyJob) {
    toast(`已有任务在跑（${busyJob.name}，${Math.round((busyJob.progress || 0) * 100)}%），等它结束或先取消再提交`, true);
    return;
  }
  const list = inputClasses().length ? inputClasses() : state.projectClasses;
  if (!list.length) { toast('先在顶部填写类别提示词', true); return; }
  try {
    const r = await POST('/api/annotate', {
      classes: list, confidence: Number(el('conf').value), overwrite: false, status: 'pending',
    });
    state.jobId = r.job_id;
    state._jobSettled = null;
    el('jobBar').classList.remove('hidden');
    toast(`已启动批量标注，共 ${r.total} 张`);
    pollJob(r.job_id);
  } catch (e) { toast('启动失败：' + e.message, true); }
};

function jobStateLabel(st) {
  return { running: '进行中', paused: '已暂停', canceling: '取消中', finished: '已完成',
           failed: '失败', canceled: '已取消' }[st] || st;
}

function renderJobBar(j) {
  state.job = j;
  const running = ['running', 'paused', 'canceling'].includes(j.state);
  const fill = el('jobBar').querySelector('.fill');
  const pct = Math.round((j.progress || 0) * 100);
  if (running && !j.progress) {
    fill.classList.add('indet');
    fill.style.width = '100%';
  } else {
    fill.classList.remove('indet');
    fill.style.width = pct + '%';
  }
  el('jobBar').querySelector('span').textContent =
    `${j.name} · ${jobStateLabel(j.state)} · ${pct}% · ${j.message} · ${j.elapsed}s`;
  el('jobPause').textContent = j.state === 'paused' ? '继续' : '暂停';
  el('jobPause').disabled = !(j.state === 'running' || j.state === 'paused');
  el('jobCancel').disabled = !running;
  el('jobClose').disabled = running;
  el('jobBar').classList.remove('hidden');
}

async function pollJob(jobId) {
  let j;
  try {
    j = await GET(`/api/jobs/${jobId}`);
  } catch (e) {
    setTimeout(() => pollJob(jobId), 1500);
    return;
  }
  renderJobBar(j);
  const running = ['running', 'paused', 'canceling'].includes(j.state);
  if (running) {
    setTimeout(() => pollJob(jobId), 800);
    return;
  }
  // 终态：如果还有别的任务在跑，切过去继续显示
  if (!state._jobSettled || state._jobSettled !== jobId) {
    const others = (await GET('/api/jobs').catch(() => ({ jobs: [] }))).jobs || [];
    const next = others.find((x) => ['running', 'paused', 'canceling'].includes(x.state));
    if (next) {
      state._jobSettled = jobId;
      renderJobBar(next);
      setTimeout(() => pollJob(next.id), 800);
      return;
    }
    state._jobSettled = jobId;
    if (j.state === 'finished') {
      const errs = (j.results && j.results[0] && j.results[0].chunk_errors) || [];
      if (errs.length) toast('任务完成，但有 ' + errs.length + ' 段被跳过：' + errs[0], true);
      else toast('任务完成：' + j.message);
    }
    else if (j.state === 'canceled') toast('任务已取消：' + j.message);
    else toast('任务失败：' + j.message, true);
    await loadImages();
    await refreshStats();
    if (state.current) await refreshCurrent();
  }
}

el('jobPause').onclick = async () => {
  const j = state.job;
  if (!j) return;
  try {
    const next = j.state === 'paused' ? 'resume' : 'pause';
    const r = await POST(`/api/jobs/${j.id}/${next}`, {});
    renderJobBar(r);
    pollJob(j.id);
  } catch (e) { toast('操作失败：' + e.message, true); }
};

el('jobCancel').onclick = async () => {
  const j = state.job;
  if (!j) return;
  if (!confirm('取消这个任务？已经完成的部分会保留。')) return;
  try {
    const r = await POST(`/api/jobs/${j.id}/cancel`, {});
    renderJobBar(r);
    pollJob(j.id);
  } catch (e) { toast('取消失败：' + e.message, true); }
};

el('jobClose').onclick = async () => {
  const j = state.job;
  if (!j) { el('jobBar').classList.add('hidden'); return; }
  try { await DELETE(`/api/jobs/${j.id}`); } catch (e) { /* 忽略 */ }
  state.job = null;
  state._jobSettled = null;
  el('jobBar').classList.add('hidden');
};

// ------------------------------------------------------------------ 项目切换
async function openProjectModal() {
  openModal('modalProject');
  const box = el('projectList');
  box.innerHTML = '<div class="hint" style="padding:6px">读取中…</div>';
  try {
    const r = await GET('/api/projects');
    box.innerHTML = '';
    for (const p of r.projects) {
      const row = document.createElement('div');
      row.className = 'pj-row' + (p.current ? ' cur' : '');
      const nm = document.createElement('div');
      nm.className = 'pj-name';
      nm.textContent = p.name;
      const meta = document.createElement('div');
      meta.className = 'pj-meta';
      meta.textContent = `${p.images} 张 · ${p.rel}`;
      row.appendChild(nm);
      row.appendChild(meta);
      if (p.current) {
        const tag = document.createElement('span');
        tag.className = 'pj-tag';
        tag.textContent = '当前';
        row.appendChild(tag);
      }
      row.onclick = () => switchProject(p);
      box.appendChild(row);
    }
  } catch (e) {
    box.innerHTML = `<div class="hint" style="padding:6px">读取失败：${e.message}</div>`;
  }
}

async function switchProject(p) {
  closeModal('modalProject');
  if (p.current) return;
  toast(`正在切换到「${p.name}」，服务重启约 10~20 秒…`);
  try {
    await POST('/api/project/switch', { path: p.rel });
  } catch (e) {
    toast('切换失败：' + e.message, true);
    return;
  }
  // 等服务重启并用新项目答话后自动刷新
  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    try {
      const s2 = await GET('/api/state');
      if ((s2.project && s2.project.name) === p.name) { location.reload(); return; }
    } catch (e) { /* 服务正在重启，继续等 */ }
  }
  toast('服务没有在 60 秒内恢复，请手动刷新页面', true);
}

el('btnProject').onclick = async () => {
  el('projectNewForm').classList.add('hidden');
  el('pjCreate').classList.add('hidden');
  el('pjInfo').textContent = '';
  await openProjectModal();
  // 预填当前类别与新项目名
  const base = (state.projectClasses && state.projectClasses.length) ? state.projectClasses : currentClasses();
  el('pjClasses').value = base.join(',');
};
el('projectCancel').onclick = () => closeModal('modalProject');

el('projectNewBtn').onclick = () => {
  const form = el('projectNewForm');
  const show = form.classList.contains('hidden');
  form.classList.toggle('hidden', !show);
  el('pjCreate').classList.toggle('hidden', !show);
  if (show) {
    el('pjName').value = '';
    el('pjDir').value = '';
    el('pjInfo').textContent = '';
    el('pjName').focus();
  }
};

el('pjCreate').onclick = async () => {
  const name = el('pjName').value.trim();
  const dir = el('pjDir').value.trim();
  const classes = el('pjClasses').value.split(/[,，]/).map((x) => x.trim()).filter(Boolean);
  if (!name) { toast('先填项目名', true); return; }
  try {
    el('pjInfo').textContent = '正在创建…';
    const r = await POST('/api/project/create', {
      name, source_dir: dir, classes, do_copy: el('pjCopy').checked, switch: true,
    });
    el('pjInfo').textContent = `已创建「${r.project}」：${r.images} 张（新导入 ${r.added} 张），正在切换…`;
    toast(`项目「${r.project}」已创建，服务切换中…`);
    const target = r.project;
    for (let i = 0; i < 60; i++) {
      await new Promise((res) => setTimeout(res, 1000));
      try {
        const s2 = await GET('/api/state');
        if (s2.project && s2.project.name === target) { location.reload(); return; }
      } catch (e) { /* 重启中 */ }
    }
    toast('服务没有在 60 秒内恢复，请手动刷新页面', true);
  } catch (e) {
    el('pjInfo').textContent = '创建失败：' + e.message;
    toast('创建失败：' + e.message, true);
  }
};

// ------------------------------------------------------------------ 类别阈值
let confClasses = [];

async function anyJobRunning() {
  try {
    const r = await GET('/api/jobs');
    return (r.jobs || []).find((j) => ['running', 'paused', 'canceling'].includes(j.state)) || null;
  } catch (e) { return null; }
}

async function openConfModal() {
  openModal('modalConf');
  el('confStats').textContent = '';
  let state_, stats;
  try {
    [state_, stats] = await Promise.all([GET('/api/state'), GET('/api/stats')]);
  } catch (e) { toast('读取失败：' + e.message, true); return; }
  const saved = (state_.settings && state_.settings.class_conf) || {};
  const globalConf = (state_.settings && state_.settings.confidence) ?? 0.5;
  el('confGlobal').textContent = `全局默认阈值 ${globalConf}`;
  el('confAll').value = globalConf;

  // 类别来源：项目类别 + 标注出现过的类别
  const names = [];
  for (const c of [...(state_.classes || []), ...Object.keys(stats.by_class || {}), ...Object.keys(saved)]) {
    if (c && !names.includes(c)) names.push(c);
  }
  confClasses = names;

  const box = el('confList');
  box.innerHTML = '';
  for (const name of names) {
    const row = document.createElement('div');
    row.className = 'conf-row';
    row.dataset.cls = name;

    const nm = document.createElement('div');
    nm.className = 'cname';
    nm.textContent = name;
    nm.title = name;
    row.appendChild(nm);

    const col = document.createElement('input');
    col.type = 'color';
    col.className = 'ccolor';
    col.dataset.cls = name;
    col.title = '该类别在掩码/方框/列表里的颜色';
    col.value = classColorHex(name);
    row.appendChild(col);

    const cnt = document.createElement('div');
    cnt.className = 'ccnt';
    cnt.textContent = `${(stats.by_class || {})[name] || 0} 个`;
    row.appendChild(cnt);

    const num = document.createElement('input');
    num.type = 'number';
    num.min = '0.05';
    num.max = '0.99';
    num.step = '0.01';
    num.value = saved[name] != null ? saved[name] : '';
    num.placeholder = String(globalConf);
    num.dataset.cls = name;
    row.appendChild(num);

    const rng = document.createElement('input');
    rng.type = 'range';
    rng.min = '0.05';
    rng.max = '0.99';
    rng.step = '0.01';
    rng.value = saved[name] != null ? saved[name] : globalConf;
    rng.dataset.cls = name;
    row.appendChild(rng);

    num.oninput = () => { if (num.value !== '') rng.value = num.value; };
    rng.oninput = () => { num.value = rng.value; };
    box.appendChild(row);
  }
}

function collectConf() {
  const out = {};
  for (const row of document.querySelectorAll('#confList .conf-row')) {
    const name = row.dataset.cls;
    const num = row.querySelector('input[type="number"]');
    const v = num.value.trim();
    if (v === '') continue;          // 空 = 用全局
    const f = Number(v);
    if (!Number.isFinite(f)) continue;
    out[name] = Math.min(0.99, Math.max(0.05, f));
  }
  return out;
}

function collectColors() {
  const out = {};
  for (const inp of document.querySelectorAll('#confList .ccolor')) {
    out[inp.dataset.cls] = inp.value;
  }
  return out;
}

function renderConfStats(stats) {
  const rows = [];
  let above = 0, below = 0, changed = 0;
  for (const [cls, v] of Object.entries(stats || {})) {
    above += v.above; below += v.below; changed += v.changed;
    rows.push(`${cls}：≥${v.threshold} 的 ${v.above} 个 / 低于 ${v.below} 个` +
      (v.manual ? `（人工决定 ${v.manual} 个不动）` : ''));
  }
  el('confStats').innerHTML =
    `合计 保留 ${above} / 排除 ${below}，本次会改变 <b>${changed}</b> 个实例` +
    (rows.length ? `<br>${rows.join('<br>')}` : '');
}

el('btnClassConf').onclick = () => openConfModal();
el('confCancel').onclick = () => closeModal('modalConf');

el('confApplyAll').onclick = () => {
  const v = Number(el('confAll').value);
  if (!Number.isFinite(v)) return;
  for (const row of document.querySelectorAll('#confList .conf-row')) {
    row.querySelector('input[type="number"]').value = v;
    row.querySelector('input[type="range"]').value = v;
  }
};

el('confResetAll').onclick = () => {
  for (const row of document.querySelectorAll('#confList .conf-row')) {
    row.querySelector('input[type="number"]').value = '';
  }
};

el('confPreview').onclick = async () => {
  try {
    busy(true);
    await POST('/api/thresholds', { conf: collectConf() });
    const r = await POST('/api/refilter', { apply: false });
    renderConfStats(r.classes);
    toast(`预览：会改变 ${r.changed} 个实例（未写入）`);
  } catch (e) { toast('预览失败：' + e.message, true); }
  finally { busy(false); }
};

el('confSaveOnly').onclick = async () => {
  try {
    const r = await POST('/api/thresholds', { conf: collectConf() });
    const cr = await POST('/api/colors', { colors: collectColors() });
    state.classColors = cr.class_colors || {};
    ovCache.clear();
    if (state.current) await refreshCurrent();
    toast('阈值已保存：' + (Object.keys(r.class_conf).length ? '已设 ' + Object.keys(r.class_conf).length + ' 个类别' : '全部用全局'));
    closeModal('modalConf');
  } catch (e) { toast('保存失败：' + e.message, true); }
};

el('confSave').onclick = async () => {
  try {
    busy(true);
    await POST('/api/thresholds', { conf: collectConf() });
    const cr = await POST('/api/colors', { colors: collectColors() });
    state.classColors = cr.class_colors || {};
    ovCache.clear();
    const r = await POST('/api/refilter', { apply: true });
    renderConfStats(r.classes);
    toast(`已重筛：改变 ${r.changed} 个实例`);
    closeModal('modalConf');
    await loadImages();
    await refreshStats();
    if (state.current) await refreshCurrent();
  } catch (e) { toast('重筛失败：' + e.message, true); }
  finally { busy(false); }
};

// ------------------------------------------------------------------ 添加图片
function openModal(id) { el(id).classList.remove('hidden'); }
function closeModal(id) { el(id).classList.add('hidden'); }

el('btnImport').onclick = async () => {
  openModal('modalImport');
  el('importPath').value = '';
  el('dirlist').classList.add('hidden');
  el('uploadInfo').textContent = '';
  el('videoClasses').value = (currentClasses().length ? currentClasses() : state.projectClasses).join(',');
  el('videoInfo').textContent = '';
  el('videoDirlist').classList.add('hidden');
  el('videoDirlist').innerHTML = '';
  state.importSel.clear();
  state.browsedDir = null;
  el('browseBar').classList.add('hidden');
  el('importDirToo').checked = false;
  renderPickBar();
  el('missingRow').classList.toggle('hidden', !(state.lastStats && state.lastStats.missing));
  if (state.lastStats) el('missingText').textContent = `有 ${state.lastStats.missing} 个条目的原图已丢失`;
};

el('pruneBtn').onclick = async () => {
  try {
    const r = await POST('/api/prune', {});
    toast(`已清理 ${r.removed} 个失效条目，剩 ${r.total} 张`);
    el('missingRow').classList.add('hidden');
    await loadImages();
    await refreshStats();
  } catch (e) { toast('清理失败：' + e.message, true); }
};
el('importCancel').onclick = () => closeModal('modalImport');
el('exportCancel').onclick = () => closeModal('modalExport');

document.querySelectorAll('#modalImport .tabs button').forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll('#modalImport .tabs button').forEach((x) => x.classList.toggle('active', x === b));
    const tab = b.dataset.tab;
    el('tabLocal').classList.toggle('hidden', tab !== 'local');
    el('tabUpload').classList.toggle('hidden', tab !== 'upload');
    el('tabVideo').classList.toggle('hidden', tab !== 'video');
    el('importOk').classList.toggle('hidden', tab === 'video');
    el('videoOk').classList.toggle('hidden', tab !== 'video');
  };
});

el('importBrowse').onclick = async () => {
  const start = el('importPath').value.trim() || '~';
  await browse(start);
};

async function browse(path) {
  try {
    const r = await GET(`/api/browse?dir=${encodeURIComponent(path)}`);
    state.browsedDir = r.path;
    el('browseBar').classList.remove('hidden');
    el('browsePath').textContent = r.path;
    const box = el('dirlist');
    box.classList.remove('hidden');
    box.innerHTML = '';

    const up = document.createElement('button');
    up.className = 'up';
    up.textContent = '⬆ ' + (r.parent || '/');
    up.onclick = () => browse(r.parent || '/');
    box.appendChild(up);

    // 先列本层图片（主要用途：单张/多张勾选），再列子目录
    if (r.images.length) {
      const sec = document.createElement('div');
      sec.className = 'sec';
      sec.textContent = `本层图片 ${r.n_images} 张（勾选后点“添加”只加勾选的）`;
      box.appendChild(sec);
      for (const f of r.images) {
        const row = document.createElement('label');
        row.className = 'file';
        row.dataset.path = f.path;
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = state.importSel.has(f.path);
        cb.onchange = () => {
          if (cb.checked) state.importSel.add(f.path);
          else state.importSel.delete(f.path);
          if (el('importPath').value.trim() === (state.browsedDir || '')) {
            el('importPath').value = '';
          }
          renderPickBar();
        };
        const name = document.createElement('span');
        name.className = 'fname';
        name.textContent = f.name;
        const size = document.createElement('span');
        size.className = 'fsize';
        size.textContent = (f.size / 1024).toFixed(0) + ' KB';
        row.appendChild(cb);
        row.appendChild(name);
        row.appendChild(size);
        box.appendChild(row);
      }
    } else {
      const info = document.createElement('div');
      info.className = 'sec';
      info.textContent = '本层没有图片文件（子目录里的图片不会被添加）';
      box.appendChild(info);
    }

    if (r.dirs.length) {
      const sec = document.createElement('div');
      sec.className = 'sec';
      sec.textContent = '子目录（点进去选）';
      box.appendChild(sec);
      for (const d of r.dirs) {
        const b = document.createElement('button');
        b.className = 'dir';
        b.textContent = '📁 ' + d.name;
        b.onclick = () => browse(d.path);
        box.appendChild(b);
      }
    }
    renderPickBar();
  } catch (e) {
    toast('浏览失败：' + e.message, true);
  }
}

function renderPickBar() {
  const bar = el('pickBar');
  const n = state.importSel.size;
  bar.classList.toggle('hidden', n === 0);
  el('pickCount').textContent = String(n);
}

el('pickClear').onclick = () => {
  state.importSel.clear();
  for (const cb of document.querySelectorAll('#dirlist .file input')) cb.checked = false;
  renderPickBar();
};

el('pickAll').onclick = () => {
  for (const cb of document.querySelectorAll('#dirlist .file input')) {
    cb.checked = true;
    const row = cb.closest('.file');
    if (row && row.dataset.path) state.importSel.add(row.dataset.path);
  }
  renderPickBar();
};

el('importOk').onclick = async () => {
  try {
    busy(true);
    const typed = el('importPath').value
      .split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
    const picked = [...state.importSel];
    const dirToo = el('importDirToo').checked && state.browsedDir ? [state.browsedDir] : [];
    const paths = [...new Set([...picked, ...typed, ...dirToo])];
    if (!paths.length) {
      toast('请填路径、或在列表勾选图片、或勾上"同时添加整个文件夹"', true);
      return;
    }
    const r = await POST('/api/import', {
      paths,
      do_copy: el('importCopy').checked,
      recursive: false,
    });
    if (r.added === 0) {
      toast(`没有新增（已存在 ${r.skipped} 张）; 缺失: ${r.missing.join(',') || '无'}`, true);
    } else {
      toast(`已添加 ${r.added} 张，共 ${r.total} 张`);
    }
    state.importSel.clear();
    state.browsedDir = null;
    el('importPath').value = '';
    el('browseBar').classList.add('hidden');
    el('importDirToo').checked = false;
    closeModal('modalImport');
    await loadImages();
    await refreshStats();
    const first = state.images.find((i) => i.status === 'pending');
    if (first) await openImage(first.id);
  } catch (e) {
    toast('添加失败：' + e.message, true);
  } finally { busy(false); }
};

// 视频页自己一套目录浏览：只列子目录 + 视频文件，点文件即填路径并读元信息
function dirOf(path) {
  const v = (path || '').trim();
  if (!v) return '~';
  if (v.endsWith('/')) return v;
  const i = v.lastIndexOf('/');
  return i > 0 ? v.slice(0, i) : (i === 0 ? '/' : '~');
}

async function videoBrowse(path) {
  try {
    const r = await GET(`/api/browse?dir=${encodeURIComponent(path)}`);
    const box = el('videoDirlist');
    box.classList.remove('hidden');
    box.innerHTML = '';

    const up = document.createElement('button');
    up.className = 'up';
    up.textContent = '⬆ ' + (r.parent || '/');
    up.onclick = () => videoBrowse(r.parent || '/');
    box.appendChild(up);

    if (r.videos && r.videos.length) {
      const sec = document.createElement('div');
      sec.className = 'sec';
      sec.textContent = `本层视频 ${r.n_videos} 个（点一下选中）`;
      box.appendChild(sec);
      for (const v of r.videos) {
        const row = document.createElement('button');
        row.className = 'dir vid';
        row.dataset.path = v.path;
        row.textContent = `🎬 ${v.name}  ·  ${(v.size / 1024 / 1024).toFixed(1)} MB`;
        row.onclick = () => pickVideo(v.path);
        box.appendChild(row);
      }
    } else {
      const sec = document.createElement('div');
      sec.className = 'sec';
      sec.textContent = '本层没有视频文件，进下面的子目录看看';
      box.appendChild(sec);
    }

    if (r.dirs.length) {
      const sec = document.createElement('div');
      sec.className = 'sec';
      sec.textContent = '子目录';
      box.appendChild(sec);
      for (const d of r.dirs) {
        const b = document.createElement('button');
        b.className = 'dir';
        b.textContent = '📁 ' + d.name;
        b.onclick = () => videoBrowse(d.path);
        box.appendChild(b);
      }
    }
    el('videoInfo').textContent = `当前目录：${r.path}`;
  } catch (e) {
    toast('浏览失败：' + e.message, true);
  }
}

async function pickVideo(path) {
  el('videoPath').value = path;
  el('videoInfo').textContent = '读取视频信息…';
  try {
    const info = await GET(`/api/video/probe?path=${encodeURIComponent(path)}`);
    el('videoInfo').textContent =
      `${path} ｜ ${info.width}×${info.height} · ${info.fps} fps · ${info.frames} 帧 · ${info.duration}s`;
  } catch (e) {
    el('videoInfo').textContent = '这个视频读不出来：' + e.message;
  }
}

el('videoBrowse').onclick = () => videoBrowse(dirOf(el('videoPath').value));

el('videoOk').onclick = async () => {
  const busyJob = await anyJobRunning();
  if (busyJob) {
    toast(`已有任务在跑（${busyJob.name}，${Math.round((busyJob.progress || 0) * 100)}%），等它结束或先取消再提交`, true);
    return;
  }
  const path = el('videoPath').value.trim();
  const classes = el('videoClasses').value.split(/[,，]/).map((x) => x.trim()).filter(Boolean);
  const annotate = el('videoAnnotate').checked;
  if (!path) { toast('请填写视频文件路径', true); return; }
  if (annotate && !classes.length) { toast('自动标注需要类别提示词', true); return; }
  try {
    busy(true);
    el('videoInfo').textContent = '提交中…';
    const r = await POST('/api/video', {
      path,
      classes,
      fps: Number(el('videoFps').value) || 0,
      max_frames: Number(el('videoMaxFrames').value) || 300,
      max_size: Number(el('videoMaxSize').value) || 1600,
      chunk_size: Number(el('videoChunk').value) || 0,
      mode: el('videoMode').value,
      annotate,
    });
    closeModal('modalImport');
    state.jobId = r.job_id;
    state._jobSettled = null;
    el('jobBar').classList.remove('hidden');
    toast('视频任务已启动，抽帧与标注可能需要几分钟');
    pollJob(r.job_id);
  } catch (e) {
    el('videoInfo').textContent = '失败：' + e.message;
    toast('视频导入失败：' + e.message, true);
  } finally { busy(false); }
};

el('pickDir').onclick = () => el('uploadDir').click();
el('pickFiles').onclick = () => el('uploadFiles').click();

async function doUpload(fileList) {
  if (!fileList || !fileList.length) return;
  const fd = new FormData();
  let n = 0;
  for (const f of fileList) {
    if (!/\.(jpe?g|png|bmp|webp|tiff?)$/i.test(f.name)) continue;
    fd.append('files', f, f.name);
    n += 1;
  }
  if (!n) { toast('没有可上传的图片', true); return; }
  el('uploadInfo').textContent = `上传中… ${n} 张`;
  try {
    busy(true);
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    const r = await res.json();
    el('uploadInfo').textContent = `已上传 ${r.added} 张，项目共 ${r.total} 张`;
    closeModal('modalImport');
    await loadImages();
    await refreshStats();
    const firstPending = state.images.find((i) => i.status === 'pending');
    if (firstPending) await openImage(firstPending.id);
  } catch (e) {
    toast('上传失败：' + e.message, true);
    el('uploadInfo').textContent = '上传失败：' + e.message;
  } finally { busy(false); }
}
el('uploadDir').onchange = (e) => doUpload(e.target.files);
el('uploadFiles').onchange = (e) => doUpload(e.target.files);

// ------------------------------------------------------------------ 导出
el('btnExport').onclick = async () => {
  openModal('modalExport');
  el('exportResult').textContent = '';
  el('exportScope').value = 'all';
  el('exportGroups').classList.add('hidden');
  // 填充分组清单（视频/文件夹）
  try {
    const r = await GET('/api/images?status=all&q=&limit=1');
    const groups = (r.groups || []).filter((g) => g.key);
    const box = el('exportGroups');
    box.innerHTML = '';
    if (!groups.length) {
      box.innerHTML = '<div class="hint" style="padding:6px">当前项目没有分组（视频抽帧/文件夹导入才会成组）</div>';
    }
    for (const g of groups) {
      const row = document.createElement('label');
      row.className = 'file';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.dataset.key = g.key;
      cb.checked = true;
      const name = document.createElement('span');
      name.className = 'fname';
      name.textContent = (g.kind === 'video' ? '🎬 ' : '📁 ') + g.name;
      const cnt = document.createElement('span');
      cnt.className = 'fsize';
      cnt.textContent = `${g.count} 张`;
      row.appendChild(cb); row.appendChild(name); row.appendChild(cnt);
      box.appendChild(row);
    }
  } catch (e) { /* ignore */ }
  try {
    const s = await GET('/api/state');
    if (!el('exportPath').value) el('exportPath').value = `${s.project.root}/exports/export-${stamp()}`;
  } catch (e) { /* ignore */ }
};

el('exportScope').onchange = () => {
  el('exportGroups').classList.toggle('hidden', el('exportScope').value !== 'groups');
};

el('exportDo').onclick = async () => {
  const formats = Array.from(document.querySelectorAll('#modalExport .fmts input:checked')).map((i) => i.value);
  if (!formats.length) { toast('至少选一种格式', true); return; }
  const scope = el('exportScope').value;
  const body = {
    formats,
    only_reviewed: el('exportReviewed').checked,
    out: el('exportPath').value.trim() || null,
  };
  if (scope === 'groups') {
    body.groups = [...document.querySelectorAll('#exportGroups input:checked')].map((i) => i.dataset.key);
    if (!body.groups.length) { toast('至少勾选一个分组', true); return; }
  } else if (scope === 'picked') {
    body.images = [...state.selImages];
    if (!body.images.length) { toast('左侧还没有勾选任何图片', true); return; }
  }
  try {
    busy(true);
    el('exportResult').textContent = '导出中…';
    const r = await POST('/api/export', body);
    el('exportResult').innerHTML = `已导出到：<code>${r.out}</code><br>` +
      Object.entries(r.written).map(([k, v]) => `${k}: ${v}`).join('<br>');
    toast('导出完成：' + r.out);
  } catch (e) {
    el('exportResult').textContent = '导出失败：' + e.message;
    toast('导出失败：' + e.message, true);
  } finally { busy(false); }
};

// 调试/自动化用：浏览器控制台里可以直接看状态
window.__state = state;
window.__debug = { hitTest, labelIdAt, screenToImage, isVisible };

// ------------------------------------------------------------------ 启动
boot();
