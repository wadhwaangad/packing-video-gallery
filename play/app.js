'use strict';
// Packing playground: one renderer for three sources of frames.
//   live      the simulator runs in a worker (Pyodide); you type as the assistant
//   replay    a symbolic episode exported by packing/export_web.py
//   rendered  a physics episode; the clock follows its video
// A frame is one action: [t0, t1] in simulated seconds, a walking path, the state after
// it, the turns spoken since the previous frame, and a private block (labels, events).

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const NS = 'http://www.w3.org/2000/svg';
function svg(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
const fmt = t => { t = Math.max(0, t); const m = Math.floor(t / 60), s = Math.floor(t % 60); return m + ':' + String(s).padStart(2, '0'); };
const lab = n => (W && W.items[n] ? W.items[n].label : n);
const pretty = s => String(s).replace(/_/g, ' ');

const ITEM_COL = {
  trousers: '#5b7bb3', tshirts: '#dfe3ea', sweater_pile: '#b9805a', striped_shirts: '#86a9cf', sweatshirts: '#9c90c7',
  towel: '#e7d6a2', towel_stack: '#bfe0e6', gym_clothes: '#8a95a3', toiletry_bag: '#3f8a6c', bottle: '#7fc39a',
  notebook: '#94603f', camera: '#c8c8c8',
};
const LABEL_COL = { nominal: '#3b4454', error: '#c07a22', recovery: '#3a8d58', guided: '#3a8d58', listen: '#555e6e',
  question: '#3a6f99', request: '#6b58a0', plan_change: '#2f7a72' };
const CH_COL = { error: '#ffc27a', question: '#9fd0ff', request: '#d2b6ff', plan_change: '#7fe0d0' };
const GOOD = new Set(['prevented', 'corrected', 'answered_correct', 'reminded', 'reminded_late', 'notified', 'acknowledged']);
const NEUTRAL = new Set(['self_recovered']);
const HANDLING = new Set(['pick_up', 'place', 'put_back']);
const WALK = new Set(['go_to', 'enter', 'look_at']);

let W = null;               // world.json
let mode = 'live';
let EP = null;              // the loaded episode
let clock = 0, playing = false, speed = 4, stopAt = null, lastTs = 0;
let reveal = false, revealTouched = false;
let heading = Math.PI / 2;
const dirty = { chat: true, log: true, timeline: true, events: true };

// ------------------------------------------------------------------ episode model
function newEpisode(kind, start, extra) {
  const ep = Object.assign({
    kind, case: start.case, privateCase: start.private_case || null, startState: start.state, startPlan: start.plan || null,
    frames: [], turns: [], events: start.events || [], ended: false, endT: null, summary: null, avail: start.t || 0,
    pending: [],
  }, extra || {});
  addTurns(ep, start.conv || []);
  return ep;
}
function addTurns(ep, turns) {
  for (const t of turns || []) ep.turns.push(t);
  dirty.chat = true; dirty.timeline = true;
}
function speechSeconds(text) { return Math.max(1.2, 0.36 * String(text).split(/\s+/).length); }
function lastTurnEnd(ep) { const t = ep.turns[ep.turns.length - 1]; return t ? t.t + speechSeconds(t.text) : 0; }
function episodeEnd(ep) {
  const f = ep.frames[ep.frames.length - 1];
  return Math.max(f ? f.t1 : 0, lastTurnEnd(ep), ep.avail || 0);
}
function loadStatic(kind, data) {
  const ep = newEpisode(kind, data.start, { videos: data.videos || null, assistant: data.assistant || '' });
  ep.privateCase = data.private_case;
  for (const f of data.frames) { ep.frames.push(f); addTurns(ep, f.conv); }
  addTurns(ep, data.end_conv || []);
  ep.summary = data.summary;
  ep.events = data.summary.events || [];
  ep.ended = true;
  ep.endT = episodeEnd(ep);
  ep.avail = ep.endT;
  return ep;
}

// index of the last frame with t0 <= t (-1 if none)
function frameAt(t) {
  const F = EP.frames; let lo = 0, hi = F.length - 1, k = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (F[m].t0 <= t + 1e-9) { k = m; lo = m + 1; } else hi = m - 1; }
  return k;
}
function pathPoint(path, p) {
  if (path.length === 1) return { x: path[0][0], y: path[0][1], dir: null };
  const seg = []; let L = 0;
  for (let i = 1; i < path.length; i++) { const d = Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]); seg.push(d); L += d; }
  let s = Math.min(1, Math.max(0, p)) * L;
  for (let i = 0; i < seg.length; i++) {
    if (s <= seg[i] || i === seg.length - 1) {
      const a = path[i], b = path[i + 1], u = seg[i] ? Math.min(1, s / seg[i]) : 1;
      return { x: a[0] + (b[0] - a[0]) * u, y: a[1] + (b[1] - a[1]) * u, dir: Math.atan2(b[1] - a[1], b[0] - a[0]) };
    }
    s -= seg[i];
  }
  const e = path[path.length - 1];
  return { x: e[0], y: e[1], dir: null };
}
// what the scene looks like at time t
function sceneAt(t) {
  const k = frameAt(t);
  if (k < 0) return { k, state: EP.startState, x: EP.startState.xy[0], y: EP.startState.xy[1], dir: null, action: null, goals: null };
  const f = EP.frames[k], prev = k ? EP.frames[k - 1].state : EP.startState;
  const goals = t >= f.t1 ? f.goals : (k ? EP.frames[k - 1].goals : null);
  if (t >= f.t1) { const e = f.path[f.path.length - 1]; return { k, state: f.state, x: e[0], y: e[1], dir: null, action: null, goals }; }
  const p = (t - f.t0) / Math.max(1e-6, f.t1 - f.t0);
  const walkP = WALK.has(f.verb) || f.path.length > 1 ? p : 1;
  const pt = pathPoint(f.path, walkP);
  const state = HANDLING.has(f.verb) && p >= 0.5 ? f.state : prev;
  return { k, state, x: pt.x, y: pt.y, dir: pt.dir, action: f, goals };
}

// ------------------------------------------------------------------ map
const sy = y => 6.4 - y;
const deg = r => r * 180 / Math.PI;
let G = {};
function buildMap() {
  const m = $('map'); m.innerHTML = '';
  const roomFill = { bedroom: '#1d222b', living: '#1b2029', hall: '#1e232c', bathroom: '#1a2129', kitchen: '#1a2129' };
  const gr = svg('g', {}, m);
  for (const [k, r] of Object.entries(W.rooms)) {
    svg('rect', { x: r.x[0], y: sy(r.y[1]), width: r.x[1] - r.x[0], height: r.y[1] - r.y[0], fill: roomFill[k] || '#1c2129' }, gr);
  }
  const gn = svg('g', { opacity: 0.5 }, m); G.nav = gn;
  for (const [a, b] of W.edges) {
    const p = W.nodes[a], q = W.nodes[b];
    svg('line', { x1: p[0], y1: sy(p[1]), x2: q[0], y2: sy(q[1]), stroke: '#3a4352', 'stroke-width': 0.015, 'stroke-dasharray': '0.05 0.05' }, gn);
  }
  const gf = svg('g', {}, m);
  const NAMES = { mattress: 'bed', dresser: 'dresser', nightstand: 'nightstand', vanity: 'vanity', kitchen_base: 'counter', sofa: 'sofa',
    coffee_table: 'coffee table', bookshelf: 'bookshelf', tv_stand: 'tv', towel_shelf: 'shelf' };
  for (const f of W.furniture) {
    if (f.name === 'bed_frame' || f.name.endsWith('_side') || f.name === 'vanity_top' || f.name === 'kitchen_top' || f.name === 'towel_shelf_lower') continue;
    const solid = f.solid;
    svg('rect', { x: f.c[0] - f.s[0] / 2, y: sy(f.c[1]) - f.s[1] / 2, width: f.s[0], height: f.s[1], rx: 0.03,
      fill: solid ? (f.name === 'mattress' ? '#2c3340' : '#283040') : '#20262f', stroke: solid ? '#3d4757' : '#2b323d',
      'stroke-width': 0.015, 'stroke-dasharray': solid ? '' : '0.04 0.03' }, gf);
    if (NAMES[f.name]) {
      const tx = svg('text', { x: f.c[0], y: sy(f.c[1]) + (f.name === 'mattress' ? 0.55 : 0.04), 'text-anchor': 'middle', 'font-size': 0.11, fill: '#7d8798' }, gf);
      tx.textContent = NAMES[f.name];
    }
  }
  // room names where furniture does not cover them
  const LABEL_AT = { living: [5.3, 5.2], bedroom: [3.0, 0.3], hall: [5.4, 3.05], bathroom: [4.7, 1.6], kitchen: [6.1, 1.95] };
  for (const [k, r] of Object.entries(W.rooms)) {
    const at = LABEL_AT[k] || [(r.x[0] + r.x[1]) / 2, (r.y[0] + r.y[1]) / 2];
    const tx = svg('text', { x: at[0], y: sy(at[1]), 'text-anchor': 'middle', 'font-size': 0.16, fill: '#6f798a', 'letter-spacing': 0.02 }, gf);
    tx.textContent = pretty(k).toUpperCase();
  }
  const gw = svg('g', {}, m);
  for (const w of W.walls) svg('rect', { x: w.c[0] - w.s[0] / 2, y: sy(w.c[1]) - w.s[1] / 2, width: w.s[0], height: w.s[1], fill: '#c4ccd8' }, gw);
  G.trail = svg('polyline', { fill: 'none', stroke: '#9fd0ff', 'stroke-width': 0.035, 'stroke-opacity': 0.35, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, m);
  G.caseG = svg('g', {}, m);
  G.items = svg('g', {}, m);
  G.person = svg('g', {}, m);
  svg('circle', { r: 0.19, fill: '#f0b36b', stroke: '#12141a', 'stroke-width': 0.03 }, G.person);
  svg('path', { d: 'M0.12,-0.09 L0.3,0 L0.12,0.09 Z', fill: '#f0b36b' }, G.person);
  G.held = svg('g', {}, m);
  G.bubble = svg('g', { class: 'bubble' }, m);
  G.caption = svg('g', { class: 'bubble' }, m);
}
function itemShape(n, parent, x, y, yaw, opts) {
  const it = W.items[n]; opts = opts || {};
  const g = svg('g', { transform: `translate(${x},${y}) rotate(${-deg(yaw)})` }, parent);
  const col = ITEM_COL[n] || '#aab';
  const onList = EP && EP.case && (curPack().includes(n));
  const stroke = opts.stroke || (onList ? '#ffffff' : '#12141a');
  const sw = onList ? 0.018 : 0.012;
  if (it.family === 'bottle' && !opts.lying) svg('circle', { r: it.size[0] / 2 + 0.01, fill: col, stroke, 'stroke-width': sw }, g);
  else {
    let w = it.size[0], h = it.size[1];
    if (opts.lying) { w = it.size[2]; h = it.size[0]; }
    w = Math.max(w, 0.08); h = Math.max(h, 0.08);
    svg('rect', { x: -w / 2, y: -h / 2, width: w, height: h, rx: 0.02, fill: col, stroke, 'stroke-width': sw, opacity: opts.opacity || 1 }, g);
  }
  const t = svg('title', {}, g); t.textContent = it.label + (onList ? ' (on the list)' : ' (not on the list)');
  return g;
}
function curPack(state) {
  const st = state || (EP && EP._state) || null;
  return (st && st.pack) || (EP ? EP.case.pack : []);
}
function drawItems(state, px, py) {
  G.items.innerHTML = ''; G.held.innerHTML = ''; G.caseG.innerHTML = '';
  EP._state = state;
  const c = W.case, by = {};
  // the suitcase
  if (state.case_pose === 'upright_floor') {
    const spot = c.floor_spots[state.case_at] || { case: c.pos };
    svg('rect', { x: spot.case[0] - c.outer[0] / 2, y: sy(spot.case[1]) - 0.16, width: c.outer[0], height: 0.32, rx: 0.05, fill: '#0c0d10', stroke: '#9fd0ff', 'stroke-width': 0.02 }, G.caseG);
    const n = Object.values(state.loc).filter(v => v === 'case').length;
    const tx = svg('text', { x: spot.case[0], y: sy(spot.case[1]) - 0.22, 'text-anchor': 'middle', 'font-size': 0.1, fill: '#9fd0ff' }, G.caseG);
    tx.textContent = `case · ${n} items`;
  } else {
    const g = svg('g', { transform: `translate(${c.pos[0]},${sy(c.pos[1])}) rotate(${-deg(c.yaw)})` }, G.caseG);
    const Wd = c.outer[0], L = c.outer[1];
    if (state.lid === 'open') svg('rect', { x: -Wd * 1.5, y: -L / 2, width: Wd, height: L, rx: 0.04, fill: '#16181d', stroke: '#454c58', 'stroke-width': 0.015 }, g);
    svg('rect', { x: -Wd / 2, y: -L / 2, width: Wd, height: L, rx: 0.04, fill: '#0c0d10', stroke: '#9fd0ff', 'stroke-width': 0.02 }, g);
    const inCase = Object.entries(state.place || {}).sort((a, b) => a[1].z - b[1].z);
    for (const [n, p] of inCase) {
      if (state.loc[n] !== 'case') continue;
      itemShape(n, g, p.x, -p.y, p.yaw, { lying: p.orient === 'lying' });
    }
    if (state.lid !== 'open') {
      svg('rect', { x: -Wd / 2, y: -L / 2, width: Wd, height: L, rx: 0.04, fill: '#1b1e24', stroke: '#9fd0ff', 'stroke-width': 0.02 }, g);
      const tx = svg('text', { x: 0, y: 0.03, 'text-anchor': 'middle', 'font-size': 0.09, fill: '#9fd0ff', transform: `rotate(${deg(c.yaw)})` }, g);
      tx.textContent = state.zipped ? 'zipped' : 'lid closed';
    }
  }
  // items in the rooms
  for (const [n, it] of Object.entries(W.items)) {
    const loc = state.loc[n];
    if (!loc || loc === 'case') continue;
    if (loc === 'held') {
      const a = heading;
      itemShape(n, G.held, px + Math.cos(a) * 0.3, sy(py + Math.sin(a) * 0.3), a + Math.PI / 2);
      continue;
    }
    const st = loc.slice(8);
    let x, y, yaw = it.yaw;
    if (st === it.station) { x = it.xy[0]; y = it.xy[1]; }
    else {
      const s = W.stations[st] || W.stations[it.station];
      const k = by[st] = (by[st] || 0) + 1;
      x = s.stand[0] + s.normal[0] * 0.5 + s.normal[1] * 0.18 * (k - 1); y = s.stand[1] + s.normal[1] * 0.5 + s.normal[0] * 0.18 * (k - 1);
    }
    itemShape(n, G.items, x, sy(y), yaw);
  }
}
function clearScene() {
  if (!G.items) return;
  for (const k of ['items', 'held', 'caseG', 'bubble', 'caption']) G[k].innerHTML = '';
  G.trail.setAttribute('points', '');
  G.person.setAttribute('transform', 'translate(-5,-5)');
  $('caseSvg').innerHTML = '';
}
function drawPerson(x, y) {
  G.person.setAttribute('transform', `translate(${x},${sy(y)}) rotate(${-deg(heading)})`);
}
function bubble(group, x, y, text, color) {
  group.innerHTML = '';
  if (!text) return;
  const words = text.split(/\s+/), lines = [];
  let cur = '';
  for (const w of words) { if ((cur + ' ' + w).trim().length > 34) { lines.push(cur); cur = w; } else cur = (cur + ' ' + w).trim(); }
  if (cur) lines.push(cur);
  const fs = 0.13, lh = 0.17, padX = 0.1, w = Math.min(3.2, Math.max(...lines.map(l => l.length)) * fs * 0.52 + padX * 2), h = lines.length * lh + 0.12;
  let bx = Math.min(7.1 - w, Math.max(0, x - w / 2)), by = sy(y) - 0.35 - h;
  if (by < 0) by = sy(y) + 0.35;
  svg('rect', { x: bx, y: by, width: w, height: h, rx: 0.08, fill: color, opacity: 0.95 }, group);
  lines.forEach((l, i) => { const t = svg('text', { x: bx + padX, y: by + 0.06 + (i + 0.8) * lh, 'font-size': fs, fill: '#0b1220' }, group); t.textContent = l; });
}

// ------------------------------------------------------------------ case inset
function drawInset(state) {
  const s = $('caseSvg'); s.innerHTML = '';
  const [X0, X1, Y0, Y1] = W.case.inner, pad = 0.02;
  s.setAttribute('viewBox', `${Y0 - pad} ${X0 - pad - 0.03} ${Y1 - Y0 + 2 * pad} ${X1 - X0 + 2 * pad + 0.06}`);
  const g = svg('g', { transform: 'matrix(0,1,1,0,0,0)' }, s);
  svg('rect', { x: X0, y: Y0, width: X1 - X0, height: Y1 - Y0, fill: '#0b0d12', stroke: '#9fd0ff', 'stroke-width': 0.004, rx: 0.01 }, g);
  for (const [k, a] of Object.entries(W.case.anchors)) {
    if (k === 'centre') continue;
    svg('circle', { cx: a.xy[0], cy: a.xy[1], r: 0.008, fill: '#3a4352' }, g);
  }
  const placed = Object.entries(state.place || {}).filter(([n]) => state.loc[n] === 'case').sort((a, b) => a[1].z - b[1].z);
  for (const [n, p] of placed) {
    const it = W.items[n];
    const gg = svg('g', { transform: `translate(${p.x},${p.y}) rotate(${deg(p.yaw)})` }, g);
    const col = ITEM_COL[n] || '#aab';
    if (it.family === 'bottle' && p.orient !== 'lying') svg('circle', { r: it.size[0] / 2, fill: col, stroke: '#000', 'stroke-width': 0.003 }, gg);
    else {
      const w = p.orient === 'lying' ? it.size[2] : it.size[0], h = p.orient === 'lying' ? it.size[0] : it.size[1];
      svg('rect', { x: -w / 2, y: -h / 2, width: w, height: h, rx: 0.006, fill: col, stroke: (p.on && p.on.length) ? '#ffc27a' : '#000', 'stroke-width': (p.on && p.on.length) ? 0.006 : 0.003 }, gg);
    }
    const tt = svg('title', {}, gg); tt.textContent = it.label + (p.on && p.on.length ? ' (on ' + p.on.map(lab).join(', ') + ')' : '');
  }
  const t1 = svg('text', { x: Y0, y: X0 - 0.012, 'font-size': 0.022, fill: '#7d8696' }, s); t1.textContent = 'wheel end';
  const t2 = svg('text', { x: Y1, y: X0 - 0.012, 'font-size': 0.022, fill: '#7d8696', 'text-anchor': 'end' }, s); t2.textContent = 'handle end';
  if (state.lid !== 'open') svg('rect', { x: Y0, y: X0, width: Y1 - Y0, height: X1 - X0, fill: '#1b1e24', opacity: 0.6 }, s);
  $('lidTxt').textContent = state.lid === 'open' ? 'lid open' : 'lid closed';
  $('poseTxt').textContent = state.case_pose === 'upright_floor' ? 'standing on the floor' : 'lying on the bed';
  const shut = 1 - (state.zip_open == null ? 1 : state.zip_open);
  $('zipTxt').textContent = state.zipped ? 'shut' + (state.locked ? ', locked' : '') : (shut > 0.01 ? Math.round(shut * 100) + '% closed' : 'open');
  $('zipBar').style.width = Math.round(shut * 100) + '%';
}

// ------------------------------------------------------------------ panels
const speakerName = () => (EP && EP.kind === 'live' && EP.assistantMode === 'human' ? 'You' : 'Assistant');
function visibleTurns(t) { return EP.turns.filter(x => x.t <= t + 1e-6); }
function renderChat(t) {
  const box = $('chat'), turns = visibleTurns(t);
  const key = turns.length + ':' + reveal + ':' + EP.pending.length;
  if (box.dataset.key === key && !dirty.chat) return;
  box.dataset.key = key; dirty.chat = false;
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.innerHTML = '';
  for (const c of turns) {
    const d = document.createElement('div');
    d.className = 'msg ' + c.speaker;
    let why = '';
    if (reveal && c.speaker === 'assistant' && c.understood) {
      const acts = c.understood.map(g => g.act + (g.item ? '(' + lab(g.item) + ')' : g.verb ? '(' + g.verb + (g.target ? ' ' + pretty(g.target) : '') + ')' : g.value != null ? '(' + (Array.isArray(g.value) ? g.value.map(lab).join(', ') : g.value) + ')' : ''));
      const res = (c.responses || []).filter(r => r.needed).map(r => (r.accepted ? 'took ' : 'declined ') + r.act);
      why = `<span class="why">read as ${esc(acts.join(', '))}${res.length ? ' · ' + esc(res.join(', ')) : ''}</span>`;
    }
    if (reveal && c.speaker === 'user' && c.event) why = `<span class="why">event ${esc(c.event)}${c.act ? ' · ' + esc(c.act) : ''}</span>`;
    d.innerHTML = `<span class="who">${c.speaker === 'user' ? 'Person' : speakerName()} · ${fmt(c.t)}</span>${esc(c.text)}${why}`;
    box.appendChild(d);
  }
  for (const p of EP.pending) {
    const d = document.createElement('div'); d.className = 'msg assistant pending';
    d.innerHTML = `<span class="who">You · said after this action</span>${esc(p)}`; box.appendChild(d);
  }
  if (!turns.length && !EP.pending.length) box.innerHTML = '<div class="empty">Nothing said yet.</div>';
  if (atBottom || turns.length < 4) box.scrollTop = box.scrollHeight;
}
function renderLog(k) {
  const box = $('log'), key = k + ':' + EP.frames.length + ':' + reveal;
  if (box.dataset.key === key && !dirty.log) return;
  box.dataset.key = key; dirty.log = false;
  const rows = [];
  EP.frames.forEach((f, i) => {
    if (i > k + 3) return;
    const P = f.private || {};
    const tag = reveal && P.label && P.label !== 'nominal' ? `<span class="pill ${P.label}">${esc(P.label)}${P.event ? ' ' + esc(P.event) : ''}${P.commit ? ' · commit' : ''}</span>` : '';
    const fb = !f.ok ? ` <span class="rej">refused: ${esc(f.feedback)}</span>` : '';
    rows.push(`<div class="${i === k ? 'now' : i > k ? 'fut' : ''}" data-i="${i}"><span class="t">${fmt(f.t0)}</span><span>${esc(f.text)}${fb}</span>${tag || '<span></span>'}</div>`);
  });
  box.innerHTML = rows.join('') || '<div class="empty" style="display:block">No actions yet.</div>';
  const now = box.querySelector('.now');
  if (now) box.scrollTop = Math.max(0, now.offsetTop - box.clientHeight + 40);
}
function renderGoals(goals, state) {
  const pack = state.pack || EP.case.pack;
  if (!goals) {
    goals = {};
    for (const n of pack) goals[`packed(${n})`] = state.loc[n] === 'case';
  }
  const out = Object.entries(goals).map(([k, v]) => `<span class="${v ? 'ok' : ''}">${esc(goalText(k))}</span>`);
  $('goals').innerHTML = out.join('');
  const prefs = (EP.case.preferences || []).map(([i, s]) => `${lab(i)} at the ${pretty(s)}`);
  $('taskTxt').innerHTML = `<b style="color:var(--ink)">${esc(EP.case.title)}</b> · ${esc(EP.case.persona)} persona` +
    (W.personas[EP.case.persona] ? ` (${esc(W.personas[EP.case.persona].label)})` : '') + (prefs.length ? `<br>Preference: ${esc(prefs.join('; '))}` : '');
}
function goalText(k) {
  const m = /^(\w+)\((\w+)(?:, (\w+))?\)$/.exec(k);
  if (m && m[1] === 'packed') return lab(m[2]);
  if (m && m[1] === 'heavy_on_floor') return lab(m[2]) + ' on the bottom';
  if (m && m[1] === 'nothing_on') return 'nothing on the ' + lab(m[2]);
  if (m && m[1] === 'pref') return lab(m[2]) + ' at the ' + pretty(m[3] || '');
  return { no_extra_items: 'nothing extra packed', bottle_upright: 'bottle upright', lid_closed: 'lid closed', zipped: 'zipped', case_upright: 'case standing', locked: 'locked' }[k] || pretty(k);
}
function eventStatus(e, t) {
  const T = e.t || {};
  if (T.fired == null || T.fired > t + 1e-6) return { s: 'scheduled', cls: 'hidden' };
  const end = T.resolved != null ? T.resolved : (EP.ended && t >= EP.endT - 1e-6 ? t : null);
  if (end != null && end <= t + 1e-6 && e.outcome) return { s: e.outcome, cls: GOOD.has(e.outcome) ? 'good' : NEUTRAL.has(e.outcome) ? '' : 'bad' };
  if (T.commit != null && T.commit <= t) return { s: 'committed', cls: 'error' };
  return { s: 'active', cls: e.channel };
}
function renderEvents(t, sc) {
  $('privatePanel').hidden = !reveal;
  if (!reveal) return;
  const evs = EP.events || [];
  $('events').innerHTML = evs.map(e => {
    const st = eventStatus(e, t);
    const who = [e.target && lab(e.target), e.other && (e.other.includes(':') ? e.other.replace(':', ' ').replace(/(\w+)$/, m => lab(m)) : (W.items[e.other] ? lab(e.other) : pretty(e.other)))].filter(Boolean).join(' / ');
    const tt = e.t && e.t.fired != null && e.t.fired <= t ? ` · fired ${fmt(e.t.fired)}` : '';
    return `<div class="ev ${st.cls === 'hidden' ? 'hidden' : ''}"><span class="pill ${e.channel}">${esc(e.id)} ${esc(pretty(e.type))}</span> <span class="pill ${st.cls}">${esc(pretty(st.s))}</span>${tt}<div class="d">${esc(e.description || '')}${who ? ' — ' + esc(who) : ''}</div></div>`;
  }).join('') || '<div class="status">No events.</div>';
  const f = sc.k >= 0 ? EP.frames[sc.k] : null, b = f && f.private && f.private.beliefs;
  $('beliefs').innerHTML = b ? `User's beliefs: ${[b.omitted.length && 'forgot ' + b.omitted.map(lab).join(', '), b.believed.length && 'thinks ' + b.believed.map(lab).join(', ') + ' belong', b.misplaced.length && 'misplacing ' + b.misplaced.map(lab).join(', '), b.thinks_zipped && 'thinks the zip is shut'].filter(Boolean).join('; ') || 'all correct'}` +
    (f && f.private ? ` · current step: ${esc(f.private.source || '')} (${esc(f.private.controller || '')})` : '') : '';
  const plan = f ? f.private && f.private.plan : EP.startPlan;
  $('planBox').hidden = !plan;
  if (plan) $('plan').innerHTML = plan.map(p => `<li class="${p.status}">${esc(p.step)}</li>`).join('');
}
function renderSummary(t) {
  const box = $('summary');
  const show = EP.ended && EP.summary && t >= EP.endT - 1e-6;
  box.classList.toggle('show', !!show);
  if (!show) return;
  if (box.dataset.ep === EP.id) return;
  box.dataset.ep = EP.id;
  const s = EP.summary;
  const evs = (s.events || EP.events || []).map(e => `<div class="ev"><span class="pill ${e.channel}">${esc(pretty(e.type))}</span> <span class="pill ${GOOD.has(e.outcome) ? 'good' : NEUTRAL.has(e.outcome) ? '' : 'bad'}">${esc(pretty(e.outcome || '–'))}</span> <span class="d">${esc(e.description || '')}</span></div>`).join('');
  const prefs = Object.entries(s.preferences || {}).map(([k, v]) => `${esc(goalText(k))} ${v ? 'met' : 'not met'}`).join(', ');
  box.innerHTML = `<h3>Episode over</h3><p style="margin:0 0 8px;color:var(--ink)">${s.goals_satisfied ? 'Everything on the list ended up packed, closed and standing.' : 'Not finished right: ' + esc(Object.entries(s.goals).filter(([, v]) => !v).map(([k]) => (k.startsWith('packed(') ? lab(k.slice(7, -1)) + ' not packed' : goalText(k) + ' failed')).join(', ')) + '.'} ${s.actions} actions, ${fmt(s.seconds)} of simulated time.${prefs ? ' Preferences: ' + prefs + '.' : ''}</p>${evs}`;
}
function renderTimeline() {
  const tl = $('timeline'), end = Math.max(1, EP.ended ? EP.endT : Math.max(episodeEnd(EP), clock) * 1.15 + 20);
  const key = EP.frames.length + ':' + EP.turns.length + ':' + reveal + ':' + Math.round(end);
  if (tl.dataset.key !== key || dirty.timeline) {
    tl.dataset.key = key; dirty.timeline = false;
    let h = '';
    for (const f of EP.frames) {
      const P = f.private || {}, col = reveal ? (LABEL_COL[P.label] || '#3b4454') : (f.ok ? '#3b4454' : '#7a5a32');
      h += `<span class="seg" style="left:${f.t0 / end * 100}%;width:${Math.max(0.15, (f.t1 - f.t0) / end * 100)}%;background:${col}"></span>`;
    }
    for (const c of EP.turns) h += `<span class="turn ${c.speaker === 'assistant' ? 'a' : ''}" style="left:${c.t / end * 100}%" title="${esc(c.text)}"></span>`;
    if (reveal) for (const e of EP.events || []) if (e.t && e.t.fired != null) h += `<span class="mark" style="left:${e.t.fired / end * 100}%;background:${CH_COL[e.channel] || '#fff'}" title="${esc(e.id + ' ' + e.type)}"></span>`;
    if (!EP.ended) h += `<span class="avail" style="left:${episodeEnd(EP) / end * 100}%"></span>`;
    h += '<span class="head" id="head"></span>';
    tl.innerHTML = h; tl.dataset.end = end;
  }
  $('head').style.left = `min(calc(100% - 2px), ${clock / Number(tl.dataset.end) * 100}%)`;
}

let lastFrameIdx = -2;
function render() {
  if (!EP || !W) return;
  const sc = sceneAt(clock);
  // walking: face the way they go; at a station: face what they work on
  if (sc.dir != null) heading = sc.dir;
  else if (W.stations[sc.state.station]) { const n = W.stations[sc.state.station].normal; heading = Math.atan2(n[1], n[0]); }
  drawItems(sc.state, sc.x, sc.y);
  drawPerson(sc.x, sc.y);
  if (sc.action && WALK.has(sc.action.verb) && sc.action.path.length > 1) G.trail.setAttribute('points', sc.action.path.map(p => p[0] + ',' + sy(p[1])).join(' '));
  else G.trail.setAttribute('points', '');
  // the person's words over their head while they speak (and a moment after)
  const turns = visibleTurns(clock), last = turns[turns.length - 1];
  const userLast = [...turns].reverse().find(c => c.speaker === 'user');
  if (userLast && clock - userLast.t < speechSeconds(userLast.text) + 3) bubble(G.bubble, sc.x, sc.y, userLast.text, '#f6e7c8');
  else G.bubble.innerHTML = '';
  if (last && last.speaker === 'assistant' && clock - last.t < speechSeconds(last.text) + 2) bubble(G.caption, 3.5, 0.15, speakerName() + ': ' + last.text, '#bfe0ff');
  else G.caption.innerHTML = '';
  drawInset(sc.state);
  const room = sc.state.room ? (W.rooms[sc.state.room] ? W.rooms[sc.state.room].label : sc.state.room) : '';
  const holding = sc.state.holding ? ' · holding the ' + lab(sc.state.holding) : '';
  $('hud').innerHTML = `<b>${fmt(clock)}</b> · ${esc(room)}${esc(holding)}${sc.action ? ' · <span style="color:var(--muted)">' + esc(sc.action.text) + '</span>' : ''}`;
  if (sc.k !== lastFrameIdx) { dirty.log = true; lastFrameIdx = sc.k; }
  renderChat(clock);
  renderLog(sc.k);
  renderGoals(sc.goals, sc.state);
  renderEvents(clock, sc);
  renderSummary(clock);
  renderTimeline();
}

// ------------------------------------------------------------------ the clock
function setPlaying(p) {
  playing = p;
  $('play').textContent = playing ? 'Pause' : (EP && EP.ended && clock >= EP.endT - 1e-6 ? 'Replay' : 'Play');
  if (mode === 'rendered' && EP && EP.videos) { const v = $('video'); if (p) v.play().catch(() => {}); else v.pause(); }
}
function loop(ts) {
  const dt = Math.min(0.1, (ts - lastTs) / 1000 || 0); lastTs = ts;
  if (EP) {
    if (mode === 'rendered' && EP.videos) {
      clock = videoSimTime();
      if (stopAt != null && clock >= stopAt) { stopAt = null; setPlaying(false); }
    }
    else if (playing) {
      clock += dt * speed;
      if (stopAt != null && clock >= stopAt) { clock = stopAt; stopAt = null; setPlaying(false); }
      if (mode === 'live') {
        if (clock >= EP.avail) {
          clock = EP.avail;
          if (EP.ended) setPlaying(false);
          else if (!busy) liveStep();
        }
      } else if (clock >= EP.endT) { clock = EP.endT; setPlaying(false); }
    }
    render();
  }
  requestAnimationFrame(loop);
}
function seek(t) {
  if (!EP) return;
  const max = mode === 'live' ? EP.avail : EP.endT;
  clock = Math.max(0, Math.min(max, t));
  stopAt = null;
  if (mode === 'rendered' && EP.videos) setVideoTime(clock);
  dirty.chat = true; dirty.log = true;
  render();
}

// ------------------------------------------------------------------ live (worker)
let worker = null, busy = false, pyReady = false, reqId = 0;
const waiting = new Map();
function call(cmd, args) {
  return new Promise((res, rej) => { const id = ++reqId; waiting.set(id, { res, rej }); worker.postMessage({ id, cmd, args }); });
}
function pyError(e) { const s = String(e); const lines = s.trim().split('\n'); return lines[lines.length - 1]; }
function startWorker() {
  try { worker = new Worker('worker.js'); } catch (e) { $('pystatus').textContent = 'This browser could not start a worker: ' + e; $('pystatus').className = 'status err'; return; }
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.progress) { $('pystatus').textContent = m.progress; return; }
    if (m.ready) { pyReady = true; $('start').disabled = false; $('start').textContent = 'Start episode'; $('pystatus').textContent = 'Ready. The simulator runs in this tab.'; return; }
    if (m.fatal) { $('pystatus').textContent = 'The simulator could not load: ' + m.fatal + '. The replays and rendered episodes still work.'; $('pystatus').className = 'status err'; $('start').textContent = 'Unavailable'; return; }
    const w = waiting.get(m.id); if (!w) return; waiting.delete(m.id);
    m.ok ? w.res(m.out) : w.rej(m.error);
  };
  worker.onerror = (e) => { $('pystatus').textContent = 'The simulator stopped: ' + (e.message || e); $('pystatus').className = 'status err'; };
}
async function startLive() {
  if (!pyReady || busy) return;
  const events = [...document.querySelectorAll('#evpick input:checked')].map(i => i.value);
  const args = { task: $('task').value, persona: $('persona').value, seed: Number($('seed').value) || 0, events, assistant: $('assistant').value };
  busy = true; $('start').disabled = true; $('pystatus').className = 'status'; $('pystatus').textContent = 'Sampling a case and checking it with a silent run and an oracle run (10–30 s in the browser)…';
  try {
    const s = await call('new', args);
    EP = newEpisode('live', s); EP.id = 'live' + Date.now(); EP.assistantMode = args.assistant;
    clock = s.t; lastFrameIdx = -2; Object.keys(dirty).forEach(k => dirty[k] = true);
    $('pystatus').textContent = `Seed ${args.seed}: ${s.case.title}, ${s.case.persona}, ${s.private_case.events.length} hidden event${s.private_case.events.length === 1 ? '' : 's'}.`;
    enableControls(true);
    buildChips();
    setPlaying(true);
  } catch (e) {
    $('pystatus').textContent = 'Could not start: ' + pyError(e); $('pystatus').className = 'status err';
  } finally { busy = false; $('start').disabled = false; }
}
async function liveStep() {
  if (!EP || EP.kind !== 'live' || busy || EP.ended) return;
  busy = true;
  const text = EP.pending.shift() || '';
  try {
    const r = await call('step', { text });
    if (r.done) {
      addTurns(EP, r.conv); EP.events = r.events || EP.events; EP.summary = r.summary; EP.ended = true;
      EP.avail = Math.max(EP.avail, r.t || 0, lastTurnEnd(EP)); EP.endT = EP.avail;
      $('pystatus').textContent = 'Episode over. Start another, or switch on "show what the assistant can\'t see" to review it.';
      $('say').disabled = true; $('sendBtn').disabled = true; $('hintBtn').disabled = true;
    } else {
      const f = r.frame;
      EP.frames.push(f); addTurns(EP, f.conv);
      if (f.private && f.private.events) EP.events = f.private.events;
      EP.avail = f.t1;
      if (stepOnce) { stopAt = f.t1; stepOnce = false; }
      // the person asked or told you something: stop after their words
      if ($('autopause').checked) {
        const ask = f.conv.find(c => c.speaker === 'user' && c.t >= clock - 1e-6 && (/\?\s*$/.test(c.text) || /^(ask_|request_|state_pref|announce|plan)/.test(c.act || '') || (c.event && c.act && c.act !== 'accept')));
        if (ask) stopAt = Math.min(stopAt == null ? Infinity : stopAt, ask.t + speechSeconds(ask.text));
      }
    }
    dirty.log = true; dirty.timeline = true; dirty.chat = true;
    $('hint').classList.remove('show');
    buildChips();
  } catch (e) {
    $('pystatus').textContent = 'Step failed: ' + pyError(e); $('pystatus').className = 'status err'; setPlaying(false);
  } finally { busy = false; }
}
let stepOnce = false;
function stepButton() {
  if (!EP) return;
  if (mode === 'live') {
    if (clock < EP.avail - 1e-6) { stopAt = EP.avail; setPlaying(true); return; }
    if (EP.ended) return;
    stepOnce = true; setPlaying(true);
  } else {
    const k = frameAt(clock), f = EP.frames[k + 1] || null;
    const cur = EP.frames[k];
    const target = cur && clock < cur.t1 - 1e-6 ? cur.t1 : f ? f.t1 : EP.endT;
    stopAt = target; setPlaying(true);
  }
}
async function send(text) {
  text = text.trim();
  if (!text || !EP || EP.kind !== 'live' || EP.ended) return;
  $('say').value = '';
  // between actions (paused at the end of what has run): say it now and hear the reply
  if (!busy && clock >= EP.avail - 1e-6 && !playing) {
    busy = true;
    try {
      const r = await call('say', { text });
      addTurns(EP, r.conv); if (r.events) EP.events = r.events;
      EP.avail = Math.max(EP.avail, r.t || 0); clock = EP.avail;
      buildChips();
    } catch (e) { $('pystatus').textContent = 'Could not say that: ' + pyError(e); }
    finally { busy = false; }
  } else {
    EP.pending.push(text); dirty.chat = true;     // heard before the next action
    if (!playing) setPlaying(true);
  }
}
async function hint() {
  if (!EP || EP.kind !== 'live' || busy) return;
  busy = true;
  try {
    const r = await call('hint', {});
    const h = $('hint');
    h.innerHTML = r.text ? `Oracle (reads the hidden schedule): “${esc(r.text)}” <button type="button" id="useHint">Use it</button>` : 'Oracle: nothing to say right now.';
    h.classList.add('show');
    const u = $('useHint'); if (u) u.onclick = () => { $('say').value = r.text; $('say').focus(); };
  } catch (e) { $('pystatus').textContent = 'Hint failed: ' + pyError(e); }
  finally { busy = false; }
}
function buildChips() {
  const box = $('chips'); box.innerHTML = '';
  if (!EP || EP.kind !== 'live' || EP.ended) return;
  const st = (EP.frames.length ? EP.frames[EP.frames.length - 1].state : EP.startState);
  const pack = st.pack || EP.case.pack;
  const left = pack.filter(n => st.loc[n] !== 'case'), packed = pack.filter(n => st.loc[n] === 'case');
  const lastUser = [...EP.turns].reverse().find(c => c.speaker === 'user');
  const asked = lastUser && /\?\s*$/.test(lastUser.text) && clock - lastUser.t < 30;
  const chips = [];
  if (asked) {
    if (/how long|when did|how much time/i.test(lastUser.text)) chips.push('About __ seconds.');
    else if (/what('s| is) left|what else|still need/i.test(lastUser.text)) chips.push(left.length ? 'You still need the ' + left.map(lab).join(' and the ') + '.' : "Nothing, that's everything.");
    else chips.push('Yes.', 'No.');
  }
  if (st.holding && !pack.includes(st.holding)) chips.push("That's not on your list.");
  if (left.length) chips.push('You still need the ' + lab(left[0]) + '.');
  if (packed.length) chips.push('You already packed the ' + lab(packed[packed.length - 1]) + '.');
  if (st.lid === 'open' && left.length) chips.push("Don't close the lid yet.");
  if (st.lid !== 'open' && !st.zipped) chips.push("The zip isn't closed all the way.");
  if (!left.length && st.lid === 'open') chips.push("That's everything on your list.");
  chips.push('Okay, noted.', 'Stop.');
  for (const c of chips.slice(0, 7)) {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = c;
    b.onclick = () => { if (c.includes('__')) { $('say').value = c; $('say').focus(); } else send(c); };
    box.appendChild(b);
  }
}

// ------------------------------------------------------------------ replays and rendered episodes
let INDEX = [];
async function loadIndex() {
  try { INDEX = await (await fetch('episodes/index.json', { cache: 'no-cache' })).json(); }
  catch (e) { INDEX = []; }
  const pairs = {};
  for (const r of INDEX.filter(r => r.kind === 'symbolic')) {
    const key = r.name.replace(/-(none|oracle)$/, '');
    (pairs[key] = pairs[key] || {})[r.assistant] = r;
  }
  const rc = $('replayCards'); rc.innerHTML = '';
  for (const [key, p] of Object.entries(pairs)) {
    const r = p.none || p.oracle;
    const b = document.createElement('button'); b.className = 'card'; b.dataset.key = key;
    const oc = run => run ? Object.values(run.outcomes).map(o => `<span class="pill ${GOOD.has(o) ? 'good' : NEUTRAL.has(o) ? '' : 'bad'}">${esc(pretty(o))}</span>`).join('') : '';
    b.innerHTML = `<b>${esc(r.title)}</b> · ${esc(r.persona)}<div class="pills">${r.events.map(e => `<span class="pill ${(W && W.event_types[e] || {}).channel || ''}">${esc(pretty(e))}</span>`).join('')}</div>` +
      `<div class="meta">Nobody speaks:</div><div class="pills">${oc(p.none)}</div><div class="meta">Oracle:</div><div class="pills">${oc(p.oracle)}</div>`;
    b.onclick = () => openReplay(p.none ? p.none.name : p.oracle.name, b);
    rc.appendChild(b);
  }
  if (!rc.children.length) rc.innerHTML = '<div class="empty">No replays exported yet (run packing/export_web.py).</div>';
  const rd = $('renderedCards'); rd.innerHTML = '';
  for (const r of INDEX.filter(r => r.kind === 'physics')) {
    const b = document.createElement('button'); b.className = 'card';
    b.innerHTML = `<img alt="" loading="lazy" src="${esc(r.folder)}/packing-overview.jpg"><b>${esc(r.title)}</b> · ${esc(r.persona)} · ${esc(r.caption || '')}` +
      `<div class="pills">${r.events.map(e => `<span class="pill ${(W && W.event_types[e] || {}).channel || ''}">${esc(pretty(e))}</span>`).join('')}${Object.values(r.outcomes).map(o => `<span class="pill ${GOOD.has(o) ? 'good' : NEUTRAL.has(o) ? '' : 'bad'}">${esc(pretty(o))}</span>`).join('')}</div>` +
      `<div class="meta">${r.actions} actions · ${fmt(r.seconds)} simulated · goals ${r.goals_satisfied ? 'met' : 'not met'}</div>`;
    b.onclick = () => openRendered(r, b);
    rd.appendChild(b);
  }
  if (!rd.children.length) rd.innerHTML = '<div class="empty">No rendered episodes yet.</div>';
}
function selectCard(container, card) { container.querySelectorAll('.card').forEach(c => c.classList.toggle('sel', c === card)); }
async function openReplay(name, card) {
  try {
    const data = await (await fetch('episodes/' + name + '.json')).json();
    const keep = EP && EP.kind === 'symbolic' && EP.name && EP.name.replace(/-(none|oracle)$/, '') === name.replace(/-(none|oracle)$/, '') ? clock : 0;
    EP = loadStatic('symbolic', data); EP.id = name; EP.name = name;
    if (card) selectCard($('replayCards'), card);
    lastFrameIdx = -2; Object.keys(dirty).forEach(k => dirty[k] = true);
    enableControls(false);
    const other = name.endsWith('-none') ? name.replace(/-none$/, '-oracle') : name.replace(/-oracle$/, '-none');
    const pb = $('pairBtn');
    pb.hidden = !INDEX.some(r => r.name === other);
    pb.textContent = name.endsWith('-none') ? 'Switch to the oracle run' : 'Switch to the silent run';
    pb.onclick = () => openReplay(other, card);
    $('chatTitle').textContent = 'Conversation · ' + (data.assistant === 'oracle' ? 'oracle assistant' : 'nobody speaks');
    clock = Math.min(keep, EP.endT); setPlaying(keep === 0);
  } catch (e) { $('chat').innerHTML = '<div class="empty">Could not load ' + esc(name) + ': ' + esc(e) + '</div>'; }
}
let curView = 'overview';
async function openRendered(row, card) {
  try {
    const data = await (await fetch(row.folder + '/episode.json')).json();
    EP = loadStatic('physics', data); EP.id = row.name; EP.folder = row.folder;
    selectCard($('renderedCards'), card);
    lastFrameIdx = -2; Object.keys(dirty).forEach(k => dirty[k] = true);
    enableControls(false);
    $('chatTitle').textContent = 'Conversation · ' + (row.assistant === 'oracle' ? 'oracle assistant' : row.assistant === 'none' ? 'nobody speaks' : 'assistant');
    $('videoBox').hidden = false;
    curView = EP.videos.some(v => v.view === curView) ? curView : EP.videos[0].view;
    $('viewBtn').hidden = EP.videos.length < 2;
    setView(curView, 0);
    clock = 0; setPlaying(false);
  } catch (e) { $('chat').innerHTML = '<div class="empty">Could not load ' + esc(row.name) + ': ' + esc(e) + '</div>'; }
}
function curVideo() { return EP && EP.videos ? EP.videos.find(v => v.view === curView) : null; }
function setView(view, t) {
  curView = view;
  const v = curVideo(), el = $('video');
  el.poster = EP.folder + '/' + v.poster;
  el.src = EP.folder + '/' + v.file;
  el.playbackRate = speedToRate(speed);
  el.addEventListener('loadedmetadata', () => { setVideoTime(t); }, { once: true });
  $('viewBtn').textContent = view === 'ego' ? 'Room cameras' : 'Head camera';
}
function videoSimTime() {
  const v = curVideo(), el = $('video');
  if (!v) return clock;
  const x = el.currentTime * v.fps, n = Math.floor(x), T = v.times;
  if (n >= T.length - 1) return Math.min(EP.endT, T[T.length - 1] + Math.max(0, x - (T.length - 1)) * v.speed / v.fps);
  return T[n] + (T[n + 1] - T[n]) * (x - n);
}
function setVideoTime(t) {
  const v = curVideo(); if (!v) return;
  const T = v.times; let lo = 0, hi = T.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (T[m] <= t) lo = m; else hi = m - 1; }
  const frac = lo < T.length - 1 && T[lo + 1] > T[lo] ? (t - T[lo]) / (T[lo + 1] - T[lo]) : 0;
  $('video').currentTime = (lo + Math.max(0, Math.min(1, frac))) / v.fps;
}
const speedToRate = s => Math.max(0.25, Math.min(2, s / 8));

// ------------------------------------------------------------------ wiring
function enableControls(live) {
  $('play').disabled = false; $('stepBtn').disabled = false;
  $('say').disabled = !live || (EP && EP.assistantMode === 'none'); $('sendBtn').disabled = $('say').disabled;
  $('hintBtn').disabled = !live;
  $('chips').hidden = !live; $('hint').classList.remove('show'); $('liveOnly').hidden = !live;
  $('autopause').parentElement.hidden = !live;
  if (!live) $('chips').innerHTML = '';
}
function setMode(m) {
  mode = m;
  document.querySelectorAll('.tabs button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.mode === m)));
  $('pane-live').hidden = m !== 'live'; $('pane-replay').hidden = m !== 'replay'; $('pane-rendered').hidden = m !== 'rendered';
  $('videoBox').hidden = true; $('pairBtn').hidden = true; $('viewBtn').hidden = true;
  const v = $('video'); v.pause(); v.removeAttribute('src'); v.load();
  setPlaying(false); EP = null; clearScene();
  $('chat').innerHTML = '<div class="empty">' + (m === 'live' ? 'Start an episode above.' : 'Pick an episode above.') + '</div>';
  $('log').innerHTML = ''; $('goals').innerHTML = ''; $('taskTxt').innerHTML = ''; $('timeline').innerHTML = ''; $('timeline').dataset.key = '';
  $('summary').classList.remove('show'); $('summary').dataset.ep = ''; $('privatePanel').hidden = true;
  $('hud').textContent = m === 'live' ? 'Start an episode' : 'Pick an episode';
  $('play').disabled = true; $('stepBtn').disabled = true; $('say').disabled = true; $('sendBtn').disabled = true; $('hintBtn').disabled = true;
  $('chips').innerHTML = ''; $('chatTitle').textContent = 'Conversation'; $('liveOnly').hidden = m !== 'live';
  $('autopause').parentElement.hidden = m !== 'live';
  if (!revealTouched) { reveal = m !== 'live'; $('reveal').checked = reveal; }
  speed = m === 'rendered' ? 8 : 4; buildSpeed();
  if (W) { G.items && (G.items.innerHTML = ''); }
}
function buildSpeed() {
  const box = $('speed'); box.innerHTML = '';
  const opts = mode === 'rendered' ? [[2, '¼×'], [4, '½×'], [8, '1×'], [16, '2×']] : [[1, '1×'], [2, '2×'], [4, '4×'], [8, '8×'], [16, '16×']];
  for (const [s, t] of opts) {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = t; b.className = s === speed ? 'on' : '';
    b.title = mode === 'rendered' ? 'video speed' : 'simulated seconds per second';
    b.onclick = () => { speed = s; if (mode === 'rendered') $('video').playbackRate = speedToRate(s); buildSpeed(); };
    box.appendChild(b);
  }
}
function timelineSeek(ev) {
  if (!EP) return;
  const r = $('timeline').getBoundingClientRect();
  seek((ev.clientX - r.left) / r.width * Number($('timeline').dataset.end || 1));
}
async function init() {
  buildSpeed();
  try { W = await (await fetch('world.json', { cache: 'no-cache' })).json(); }
  catch (e) { $('hud').textContent = 'Could not load world.json: ' + e; return; }
  buildMap();
  for (const [k, t] of Object.entries(W.tasks)) $('task').insertAdjacentHTML('beforeend', `<option value="${esc(k)}">${esc(t.title)}</option>`);
  for (const [k, p] of Object.entries(W.personas)) $('persona').insertAdjacentHTML('beforeend', `<option value="${esc(k)}">${esc(k)} — ${esc(p.label.split(':')[0])}</option>`);
  for (const [k, e] of Object.entries(W.event_types)) {
    if (!e.sampled) continue;
    $('evpick').insertAdjacentHTML('beforeend', `<label title="${esc(e.description)}"><input type="checkbox" value="${esc(k)}"><span style="color:${CH_COL[e.channel]}">●</span> ${esc(pretty(k))}</label>`);
  }
  $('evpick').addEventListener('change', (e) => {
    const on = [...document.querySelectorAll('#evpick input:checked')];
    if (on.length > 3) e.target.checked = false;
  });
  document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => setMode(b.dataset.mode));
  $('start').onclick = startLive;
  $('play').onclick = () => {
    if (!EP) return;
    if (!playing && EP.ended && clock >= EP.endT - 1e-6) seek(0);
    stopAt = null; setPlaying(!playing);
  };
  $('stepBtn').onclick = stepButton;
  $('compose').onsubmit = (e) => { e.preventDefault(); send($('say').value); };
  $('hintBtn').onclick = hint;
  $('reveal').onchange = () => { reveal = $('reveal').checked; revealTouched = true; Object.keys(dirty).forEach(k => dirty[k] = true); };
  $('viewBtn').onclick = () => { if (EP && EP.videos) { const t = clock, was = playing; setView(curView === 'ego' ? 'overview' : 'ego', t); if (was) $('video').play().catch(() => {}); } };
  $('video').addEventListener('ended', () => setPlaying(false));
  let dragging = false;
  $('timeline').addEventListener('pointerdown', e => { dragging = true; $('timeline').setPointerCapture(e.pointerId); timelineSeek(e); });
  $('timeline').addEventListener('pointermove', e => { if (dragging) timelineSeek(e); });
  $('timeline').addEventListener('pointerup', () => { dragging = false; });
  $('log').addEventListener('click', e => { const d = e.target.closest('[data-i]'); if (d && EP) seek(EP.frames[Number(d.dataset.i)].t0); });
  document.addEventListener('keydown', e => {
    if (e.target.matches('input,select,textarea')) return;
    if (e.key === ' ') { e.preventDefault(); $('play').click(); }
    else if (e.key === 'ArrowRight') stepButton();
  });
  setMode(location.hash === '#replays' ? 'replay' : location.hash === '#rendered' ? 'rendered' : 'live');
  await loadIndex();
  const want = new URLSearchParams(location.search).get('episode');
  if (want) {
    const r = INDEX.find(x => x.name === want);
    if (r && r.kind === 'physics') { setMode('rendered'); openRendered(r, null); }
    else if (r) { setMode('replay'); openReplay(r.name, null); }
  }
  startWorker();
  requestAnimationFrame(loop);
}
init();
