// Pack it yourself: you are the person packing, in a real-time 3D version of the apartment
// the Blender videos show. The scene is built from pack/world3d.json (scene.py's numbers:
// rooms, walls, furniture, items, the case and its hardware) with the same CC0 Poly Haven
// models and textures, the same sun, room lights and sky. The scanned garment piles and
// case are not redistributable, so their stand-ins are built here.
//
// The simulator (packing/game.py in Pyodide, via worker.js) decides what can be done from
// where you stand and what it does; this file walks, animates and draws.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { Reflector } from 'three/addons/objects/Reflector.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);          // z up, metres: the simulator's frame
const $ = id => document.getElementById(id);
const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, k) => a + (b - a) * k;
const ease = k => k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
const seg = (k, a, b) => clamp((k - a) / (b - a), 0, 1);
const angDiff = (a, b) => Math.atan2(Math.sin(b - a), Math.cos(b - a));
const fmtTime = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ------------------------------------------------------------------ the simulator (Pyodide worker)
const py = { worker: null, ready: false, calls: new Map(), n: 0, error: '' };
function startPython() {
  try { py.worker = new Worker('worker.js'); } catch (e) { py.error = String(e); pyStatus(); return; }
  py.worker.onmessage = e => {
    const m = e.data;
    if (m.progress) { pyProgress = m.progress; pyStatus(); return; }
    if (m.ready) { py.ready = true; pyStatus(); onPythonReady(); return; }
    if (m.fatal) { py.error = m.fatal; pyStatus(); return; }
    const c = py.calls.get(m.id); if (!c) return;
    py.calls.delete(m.id);
    m.ok ? c.resolve(m.out) : c.reject(new Error(m.error));
  };
}
function call(cmd, args = {}) {
  return new Promise((resolve, reject) => {
    const id = ++py.n;
    py.calls.set(id, { resolve, reject });
    py.worker.postMessage({ id, cmd, args });
  });
}
let pyProgress = 'Starting Python…';

// ------------------------------------------------------------------ renderer, scene, cameras
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.AgXToneMapping;      // the renders use Blender's AgX view transform
const TUNE = Object.fromEntries([...new URLSearchParams(location.search)].filter(([k]) => ['exp', 'env', 'area', 'sunI', 'hemi', 'ao'].includes(k)).map(([k, v]) => [k, Number(v)]));
renderer.toneMappingExposure = TUNE.exp ?? 0.56;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9fb4c8);
const camera = new THREE.PerspectiveCamera(60, 1, 0.03, 120);
RectAreaLightUniformsLib.init();
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const gtao = new GTAOPass(scene, camera, 16, 16);
gtao.updateGtaoMaterial({ radius: 0.6, distanceExponent: 1.5, thickness: 1.0, scale: 1.0, samples: 12 });
gtao.blendIntensity = TUNE.ao ?? 1.0;
composer.addPass(gtao);
composer.addPass(new OutputPass());

const manager = new THREE.LoadingManager();
let loaded3d = false;
manager.onProgress = (url, n, total) => { $('loadBar').style.width = `${Math.round(100 * n / total)}%`; };
const TL = new THREE.TextureLoader(manager);
const GL = new GLTFLoader(manager);
let W = null;                                       // pack/world3d.json

// ------------------------------------------------------------------ materials
const maxAniso = renderer.capabilities.getMaxAnisotropy();
const texCache = new Map();
function tex(url, srgb) {
  if (texCache.has(url)) return texCache.get(url);
  const t = TL.load(url);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.anisotropy = Math.min(8, maxAniso);
  texCache.set(url, t);
  return t;
}
const lin = c => new THREE.Color().setRGB(c[0], c[1], c[2], THREE.LinearSRGBColorSpace);
// A Poly Haven texture set (diff, normal, roughness). The scale (metres per tile) is baked
// into each mesh's box-projected UVs, as Blender's textured() maps object coordinates.
function pbr(name, { tint = [1, 1, 1], rough = 1, bump = 1, side = THREE.FrontSide } = {}) {
  const base = `pack/assets/textures/${name}/`;
  return new THREE.MeshStandardMaterial({
    map: tex(base + 'diff.jpg', true), normalMap: tex(base + 'nor_gl.jpg', false),
    roughnessMap: tex(base + 'rough.jpg', false), color: lin(tint), roughness: rough,
    normalScale: new THREE.Vector2(bump, bump), side,
  });
}
const plain = (c, rough = 0.5, metal = 0, extra = {}) =>
  new THREE.MeshStandardMaterial({ color: lin(c), roughness: rough, metalness: metal, ...extra });

let M = {};
function makeMaterials() {
  M = {
    oak: pbr('oak_wood_planks', { tint: [0.92, 0.85, 0.78], bump: 0.6 }),
    tiles: pbr('long_white_tiles', { bump: 0.8 }),
    plaster: new THREE.MeshStandardMaterial({ color: lin([0.86, 0.84, 0.80]), roughness: 0.85,
      normalMap: tex('pack/assets/textures/painted_plaster_wall/nor_gl.jpg', false), normalScale: new THREE.Vector2(0.25, 0.25) }),
    trim: plain([0.9, 0.9, 0.88], 0.35), ceiling: plain([0.93, 0.93, 0.92], 0.8),
    frame: plain([0.95, 0.95, 0.94], 0.3), lawn: plain([0.16, 0.22, 0.10], 0.9),
    glass: new THREE.MeshStandardMaterial({ color: lin([0.85, 0.92, 1.0]), roughness: 0.02, metalness: 0, transparent: true, opacity: 0.035, depthWrite: false }),
    casing: plain([0.92, 0.92, 0.9], 0.35), leaf: plain([0.88, 0.87, 0.84], 0.4),
    linen: pbr('cotton_jersey', { tint: [1.0, 0.97, 0.92], bump: 0.35 }),
    upholstery: pbr('wool_boucle', { tint: [0.55, 0.52, 0.47], bump: 0.8 }),
    walnut: plain([0.20, 0.11, 0.06], 0.45), lightOak: plain([0.55, 0.40, 0.25], 0.5),
    lacquer: plain([0.88, 0.88, 0.86], 0.25), sage: plain([0.43, 0.50, 0.42], 0.35),
    stone: plain([0.85, 0.84, 0.81], 0.25), rug: pbr('wool_boucle', { tint: [0.75, 0.70, 0.62], bump: 1.0 }),
    ceramic: plain([0.95, 0.95, 0.95], 0.08), chrome: plain([0.9, 0.9, 0.9], 0.12, 1, { envMapIntensity: 8 }),
    mirror: plain([0.95, 0.95, 0.95], 0.02, 1),
    // the case: black ABS shell, charcoal woven lining, zip tape, metal sliders
    shell: new THREE.MeshPhysicalMaterial({ color: lin([0.03, 0.03, 0.035]), roughness: 0.38, clearcoat: 0.35, clearcoatRoughness: 0.35 }),
    lining: pbr('cotton_jersey', { tint: [0.11, 0.115, 0.13], bump: 0.5, rough: 1, side: THREE.BackSide }),
    liningTop: pbr('cotton_jersey', { tint: [0.11, 0.115, 0.13], bump: 0.5 }),
    tape: plain([0.02, 0.02, 0.022], 0.55), slider: plain([0.62, 0.63, 0.65], 0.28, 1, { envMapIntensity: 8 }),
    pull: plain([0.025, 0.025, 0.027], 0.55), alu: plain([0.08, 0.08, 0.085], 0.35, 0.8, { envMapIntensity: 6 }),
    wheel: plain([0.02, 0.02, 0.02], 0.6), lockRed: plain([0.55, 0.06, 0.05], 0.4),
    // the person: the mannequin of the renders
    shirt: pbr('cotton_jersey', { tint: [0.36, 0.45, 0.40], bump: 0.6 }),
    pants: pbr('denim_fabric', { tint: [0.30, 0.30, 0.33], bump: 0.6 }),
    skin: new THREE.MeshPhysicalMaterial({ color: lin([0.56, 0.36, 0.26]), roughness: 0.45, sheen: 0.2, sheenColor: lin([0.8, 0.5, 0.4]) }),
    hair: plain([0.05, 0.035, 0.025], 0.6), shoe: plain([0.85, 0.85, 0.83], 0.5),
  };
  // Painted walls: the renders' walls fall off away from the ceiling panels and darken into
  // the floor (bounce light Cycles traces); a height gradient stands in for it.
  M.plaster.onBeforeCompile = sh => {
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying float vWz;')
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvWz = (modelMatrix * vec4(transformed, 1.0)).z;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying float vWz;')
      .replace('#include <color_fragment>', '#include <color_fragment>\ndiffuseColor.rgb *= mix(vec3(0.70, 0.66, 0.60), vec3(1.04, 1.0, 0.95), smoothstep(0.0, 2.5, vWz));');
  };
}

// ------------------------------------------------------------------ geometry helpers
// Box-projected UVs in metres (u, v from the two axes across the face's dominant normal).
function projectUV(geo, scale, off = V()) {
  const p = geo.attributes.position, n = geo.attributes.normal;
  const uv = new Float32Array(p.count * 2);
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i) + off.x, y = p.getY(i) + off.y, z = p.getZ(i) + off.z;
    const ax = Math.abs(n.getX(i)), ay = Math.abs(n.getY(i)), az = Math.abs(n.getZ(i));
    let u, v;
    if (ax >= ay && ax >= az) { u = y; v = z; } else if (ay >= az) { u = x; v = z; } else { u = x; v = y; }
    uv[2 * i] = u / scale; uv[2 * i + 1] = v / scale;
  }
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  return geo;
}
function boxGeo(s, bevel = 0, segs = 2) {
  if (bevel > 0) {
    const r = Math.min(bevel, s[0] / 2 - 1e-4, s[1] / 2 - 1e-4, s[2] / 2 - 1e-4);
    if (r > 1e-4) return new RoundedBoxGeometry(s[0], s[1], s[2], segs, r);
  }
  return new THREE.BoxGeometry(s[0], s[1], s[2]);
}
function mesh(geo, mat, pos, yaw = 0, parent = scene, shadow = true) {
  const m = new THREE.Mesh(geo, mat);
  if (pos) m.position.set(pos[0], pos[1], pos[2]);
  m.rotation.z = yaw;
  m.castShadow = shadow; m.receiveShadow = true;
  parent.add(m);
  return m;
}
// A textured box at a world position: UVs in world metres, so neighbours line up.
function tbox(s, mat, c, { bevel = 0, segs = 2, scale = 1, yaw = 0, parent = scene, shadow = true } = {}) {
  const g = boxGeo(s, bevel, segs);
  if (mat.map || mat.normalMap) projectUV(g, scale, yaw ? V() : V(c[0], c[1], c[2]));
  return mesh(g, mat, c, yaw, parent, shadow);
}
const occluders = [];                                  // what the follow camera must not pass through
const walkable = [];                                   // static obstacles (2D boxes) for walking

// ------------------------------------------------------------------ the apartment (render_blender.build_apartment)
function buildApartment() {
  for (const [name, r] of Object.entries(W.rooms)) {
    const mat = r.floor === 'long_white_tiles' ? M.tiles : M.oak;
    const s = [r.x[1] - r.x[0], r.y[1] - r.y[0], 0.02];
    tbox(s, mat, [(r.x[0] + r.x[1]) / 2, (r.y[0] + r.y[1]) / 2, -0.01], { scale: mat === M.tiles ? 1.2 : 2.0 });
  }
  for (const w of W.walls) {
    const m = tbox(w.s, M.plaster, w.c, { scale: 1.5 });
    occluders.push(m);
    const z0 = w.c[2] - w.s[2] / 2;
    if (z0 < 1e-6 && w.s[2] > 0.5) {
      const sx = w.s[0], sy = w.s[1];
      tbox([sx + (sx < sy ? 0.024 : 0), sy + (sy < sx ? 0.024 : 0), 0.085], M.trim, [w.c[0], w.c[1], 0.0425], { bevel: 0.003 });
      walkable.push(rect2(w.c, w.s));
    }
  }
  ceiling = tbox([7.2, 6.6, 0.05], M.ceiling, [3.5, 3.2, W.wall_h + 0.025], { shadow: false });
  ceiling.visible = false;
  for (const [wall, s0, s1, z0, z1] of W.windows) {
    const y = wall === 'wall_south' ? 0.0 : 6.4, cx = (s0 + s1) / 2;
    const g = tbox([s1 - s0, 0.01, z1 - z0], M.glass, [cx, y, (z0 + z1) / 2], { shadow: false });
    g.castShadow = false;
    for (const [sz, c] of [[[s1 - s0 + 0.1, 0.12, 0.05], [cx, y, z0 - 0.02]], [[s1 - s0 + 0.1, 0.12, 0.05], [cx, y, z1 + 0.02]],
      [[0.05, 0.12, z1 - z0], [s0, y, (z0 + z1) / 2]], [[0.05, 0.12, z1 - z0], [s1, y, (z0 + z1) / 2]],
      [[0.04, 0.10, z1 - z0], [cx, y, (z0 + z1) / 2]]]) tbox(sz, M.frame, c, { bevel: 0.004 });
    tbox([s1 - s0 + 0.2, 0.2, 0.025], M.trim, [cx, y + (wall === 'wall_south' ? 0.08 : -0.08), z0 - 0.05], { bevel: 0.003 });
  }
  mesh(new THREE.BoxGeometry(60, 60, 0.02), M.lawn, [3.5, 3.2, -3.0], 0, scene, false);
  for (const [s0, s1, at] of [[2.75, 3.65, 3.4], [4.15, 4.95, 2.2]]) {
    for (const s of [s0, s1]) tbox([0.07, 0.14, 2.08], M.casing, [s, at, 1.04], { bevel: 0.004 });
    tbox([s1 - s0 + 0.14, 0.14, 0.07], M.casing, [(s0 + s1) / 2, at, 2.085], { bevel: 0.004 });
  }
  tbox([0.84, 0.04, 2.02], M.leaf, [2.75 - 0.43, 3.4 - 0.06, 1.01], { bevel: 0.004 });
  tbox([0.76, 0.04, 2.02], M.leaf, [4.15 - 0.39, 2.2 - 0.06, 1.01], { bevel: 0.004 });
}
let ceiling = null;
const rect2 = (c, s, pad = 0) => ({ x0: c[0] - s[0] / 2 - pad, x1: c[0] + s[0] / 2 + pad, y0: c[1] - s[1] / 2 - pad, y1: c[1] + s[1] / 2 + pad });

// ------------------------------------------------------------------ furniture (build_furniture)
function buildFurniture() {
  const mats = { bed_frame: M.walnut, mattress: M.linen, vanity: M.lacquer, counter: M.stone, rug: M.rug };
  const texScale = { mattress: 0.35, rug: 0.5, headboard: 0.4 };
  for (const b of W.boxes) {
    let mat = mats[b.m] || M.lightOak;
    if (b.name === 'headboard') mat = M.upholstery;
    if (b.name.startsWith('kitchen_base')) mat = M.sage;
    if (b.name === 'kitchen_top') mat = M.lightOak;
    const bev = b.name === 'mattress' ? 0.04 : b.name === 'headboard' ? 0.03 : 0.006;
    const m = tbox(b.s, mat, b.c, { bevel: bev, segs: b.name === 'mattress' ? 5 : 2, scale: texScale[b.name] || texScale[b.m] || 1, yaw: b.yaw });
    if (b.solid) walkable.push(rect2(b.c, b.s));
    if (b.s[2] > 1.0) occluders.push(m);
  }
  for (const y of [1.35, 2.15]) tbox([0.36, 0.62, 0.14], M.linen, [0.30, y, W.bed.mattress_top + 0.10], { bevel: 0.06, segs: 5, scale: 0.35 });
  tbox([0.42, 0.32, 0.02], M.ceramic, [4.95, 0.27, 0.872], { bevel: 0.008 });
  tbox([0.03, 0.03, 0.16], M.chrome, [4.95, 0.09, 0.95], { bevel: 0.012 });
  tbox([0.82, 0.02, 0.72], M.chrome, [4.70, 0.06, 1.45], { bevel: 0.004 });
  // the mirror reflects the room (rendered only while it is in view)
  const mirror = new Reflector(new THREE.PlaneGeometry(0.78, 0.68), { textureWidth: 512, textureHeight: 448, color: 0xb8bcc0 });
  mirror.position.set(4.70, 0.0715, 1.45); mirror.rotation.x = -Math.PI / 2; mirror.rotation.order = 'ZXY';
  scene.add(mirror);
  tbox([0.02, 2.04, 0.55], M.tiles, [6.94, 1.10, 1.17], { scale: 0.8 });
  for (const a of W.assets) {
    for (const b of a.collision) walkable.push(rect2(b.c, b.s));
    loadModel(a.asset).then(g => {
      const holder = new THREE.Group();
      holder.position.set(...a.pos); holder.rotation.z = a.yaw; holder.scale.setScalar(a.scale ?? 1);
      holder.add(g); scene.add(holder);
      if (a.name === 'bookshelf') g.traverse(o => { if (o.isMesh) occluders.push(o); });
    });
  }
}
const modelCache = new Map();
function loadModel(name, keep = null) {
  if (!modelCache.has(name)) modelCache.set(name, GL.loadAsync(`pack/assets/models/${name}/${name}.gltf`));
  return modelCache.get(name).then(gltf => {
    const g = gltf.scene.clone(true);
    if (keep && keep.length) {
      const drop = [];
      g.traverse(o => { if (o.isMesh && !keep.includes(o.name.split('.')[0]) && !keep.includes(o.parent?.name)) drop.push(o); });
      drop.forEach(o => o.parent.remove(o));
    }
    g.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    g.rotation.x = Math.PI / 2;                     // glTF is y up; the scene is z up (as Blender imports it)
    return g;
  });
}

// ------------------------------------------------------------------ items
// Folded piles: stacked, slightly uneven folded layers in fabric scans, coloured like the
// scans they stand in for.
const PILES = {
  trousers: { tex: 'denim_fabric', scale: 0.18, colors: [[0.10, 0.16, 0.30], [0.08, 0.12, 0.24], [0.12, 0.18, 0.32]], bump: 0.9 },
  tshirts: { tex: 'cotton_jersey', scale: 0.12, colors: [[0.80, 0.80, 0.78], [0.08, 0.12, 0.24], [0.42, 0.44, 0.46], [0.30, 0.34, 0.20], [0.75, 0.42, 0.30]] },
  sweater_pile: { tex: 'cotton_jersey', scale: 0.05, colors: [[0.55, 0.47, 0.37], [0.30, 0.31, 0.33], [0.46, 0.20, 0.17]], bump: 1.6 },
  striped_shirts: { tex: 'stripes', scale: 0.11, colors: [[1, 1, 1], [0.92, 0.95, 1.0]] },
  sweatshirts: { tex: 'cotton_jersey', scale: 0.14, colors: [[0.35, 0.36, 0.38], [0.14, 0.15, 0.17], [0.55, 0.40, 0.42]] },
  towel: { tex: 'cotton_jersey', scale: 0.035, colors: [[0.86, 0.85, 0.82]], bump: 2.0 },
  towel_stack: { tex: 'cotton_jersey', scale: 0.035, colors: [[0.86, 0.85, 0.82], [0.43, 0.52, 0.44], [0.70, 0.62, 0.50]], bump: 2.0 },
  gym_clothes: { tex: 'cotton_jersey', scale: 0.12, colors: [[0.05, 0.05, 0.06], [0.30, 0.31, 0.33], [0.06, 0.07, 0.10]] },
};
let stripeTex = null;
function stripeMaterial(color) {
  if (!stripeTex) {
    const c = document.createElement('canvas'); c.width = 64; c.height = 64;
    const x = c.getContext('2d'); x.fillStyle = '#f2f2ee'; x.fillRect(0, 0, 64, 64);
    x.fillStyle = '#2f4f86'; for (let i = 0; i < 64; i += 16) x.fillRect(i, 0, 6, 64);
    stripeTex = new THREE.CanvasTexture(c); stripeTex.wrapS = stripeTex.wrapT = THREE.RepeatWrapping;
    stripeTex.colorSpace = THREE.SRGBColorSpace; stripeTex.anisotropy = 8;
  }
  return new THREE.MeshStandardMaterial({ map: stripeTex, color: lin(color), roughness: 0.85,
    normalMap: tex('pack/assets/textures/cotton_jersey/nor_gl.jpg', false), normalScale: new THREE.Vector2(0.4, 0.4) });
}
function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }
function buildPile(name, it) {
  const g = new THREE.Group();
  const p = PILES[name] || PILES.tshirts;
  const [sx, sy, sz] = it.size;
  const n = Math.max(1, Math.round(sz / (name.startsWith('towel') ? 0.045 : 0.026)));
  const th = sz / n, rand = rng(name.length * 977 + n);
  for (let i = 0; i < n; i++) {
    const c = p.colors[i % p.colors.length];
    const mat = p.tex === 'stripes' ? stripeMaterial(c) : pbr(p.tex, { tint: c, bump: p.bump || 0.7 });
    const s = [sx * (0.93 + 0.07 * rand()), sy * (0.93 + 0.07 * rand()), th * 1.02];
    const geo = new RoundedBoxGeometry(s[0], s[1], s[2], 4, Math.min(th * 0.48, 0.02));
    projectUV(geo, p.scale);
    const m = mesh(geo, mat, [(rand() - 0.5) * 0.012, (rand() - 0.5) * 0.012, -sz / 2 + th * (i + 0.5)], (rand() - 0.5) * 0.06, g);
    m.scale.z = 0.98;
  }
  if (name === 'gym_clothes') {                       // a pair of rolled socks on top
    const sm = pbr('cotton_jersey', { tint: [0.85, 0.85, 0.83] });
    for (const dx of [-0.035, 0.035]) {
      const s = mesh(new THREE.CapsuleGeometry(0.022, 0.05, 6, 12), sm, [dx, 0.04, sz / 2 + 0.012], 0, g);
      s.rotation.x = Math.PI / 2; s.scale.z = 0.8;
    }
  }
  return g;
}
function buildBag(it) {                                 // the toiletry bag: suede shell, zip, leather pull
  const g = new THREE.Group();
  const [L, Wd, H] = it.size;
  const shell = pbr('scuba_suede', { tint: [0.18, 0.36, 0.29], bump: 0.9 });
  const geo = new RoundedBoxGeometry(L - 0.004, Wd - 0.006, H - 0.004, 6, Math.min(0.03, H / 2 - 0.002));
  projectUV(geo, 0.10);
  mesh(geo, shell, [0, 0, 0], 0, g);
  mesh(new THREE.BoxGeometry(L - 0.04, 0.012, 0.004), M.tape, [0, 0, H / 2 - 0.001], 0, g);
  mesh(new RoundedBoxGeometry(0.022, 0.014, 0.008, 2, 0.003), M.slider, [L / 2 - 0.05, 0, H / 2 + 0.002], 0, g);
  const leather = pbr('brown_leather', { tint: [0.35, 0.18, 0.09], bump: 0.7 });
  mesh(new RoundedBoxGeometry(0.04, 0.012, 0.003, 2, 0.0014), leather, [L / 2 - 0.022, 0, H / 2 + 0.004], 0, g);
  return g;
}
function buildBottle(it) {                              // the water bottle: lathe-turned, as item_assets.water_bottle
  const g = new THREE.Group();
  const D = it.size[0], Hh = it.size[2], R = D / 2 - 0.0005, zb = -Hh / 2, zt = Hh / 2;
  const coat = plain([0.30, 0.40, 0.33], 0.62, 0, {});
  const steel = plain([0.80, 0.80, 0.80], 0.26, 1, { envMapIntensity: 8 });
  const cap = plain([0.045, 0.045, 0.05], 0.5);
  const lathe = (pts, mat) => { const geo = new THREE.LatheGeometry(pts.map(([r, z]) => new THREE.Vector2(r, z)), 48); geo.rotateX(Math.PI / 2); return mesh(geo, mat, null, 0, g); };
  lathe([[0.0001, zb], [R - 0.006, zb], [R - 0.0025, zb + 0.0008], [R - 0.0006, zb + 0.003], [R, zb + 0.006]], steel);
  const zs = zt - 0.060, prof = [[R, zb + 0.006], [R, zs - 0.01]];
  for (let i = 1; i <= 12; i++) { const f = i / 12; prof.push([R - (R - 0.0255) * (1 - Math.cos(f * Math.PI / 2)), zs - 0.01 + 0.03 * Math.sin(f * Math.PI / 2)]); }
  const zn = prof[prof.length - 1][1];
  prof.push([0.0255, zn + 0.004]);
  lathe(prof, coat);
  lathe([[0.0255, zn + 0.004], [0.0262, zn + 0.0045], [0.0264, zn + 0.009], [0.0258, zn + 0.0095], [0.020, zn + 0.0095]], steel);
  const zc0 = zn + 0.0095, zc1 = zt - 0.012, cp = [[0.0001, zc0], [0.0272, zc0], [0.0278, zc0 + 0.0015], [0.0278, zc1 - 0.0045]];
  for (let i = 1; i <= 8; i++) { const a = Math.PI / 2 * i / 8; cp.push([0.0278 - 0.0045 * (1 - Math.cos(a)), zc1 - 0.0045 + 0.0045 * Math.sin(a)]); }
  cp.push([0.0001, zc1]);
  lathe(cp, cap);
  const loop = mesh(new THREE.TorusGeometry(0.0125, 0.0028, 8, 24, Math.PI), cap, [0, 0, zc1 - 0.002], 0, g);
  loop.rotation.x = Math.PI / 2;
  return g;
}
const items = {};                                       // name -> holder group (centred on its collision box)
function buildItems() {
  for (const [name, it] of Object.entries(W.items)) {
    let g;
    if (it.kind === 'pile') g = buildPile(name, it);
    else if (name === 'toiletry_bag') g = buildBag(it);
    else if (name === 'bottle') g = buildBottle(it);
    else {
      g = new THREE.Group();
      const off = new THREE.Group(); off.position.set(...it.mesh_offset); g.add(off);
      loadModel(it.asset, it.mesh_objects).then(m => off.add(m));
    }
    g.name = name;
    g.userData = { it };
    scene.add(g);
    items[name] = g;
  }
}

// ------------------------------------------------------------------ the case
// A hard-shell spinner the size of the physics case: base and lid tubs with rounded
// corners and a rounded back edge, a fabric lining, zip tape and sliders on the front,
// the lock on the lid's front wall, four spinner wheels and the telescopic handle.
function rrect(w, l, r, n) {
  const pts = [], cs = [[w / 2 - r, l / 2 - r, 0], [-w / 2 + r, l / 2 - r, 1], [-w / 2 + r, -l / 2 + r, 2], [w / 2 - r, -l / 2 + r, 3]];
  for (const [cx, cy, k] of cs) for (let i = 0; i <= n; i++) {
    const a = (k + i / n) * Math.PI / 2; pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return pts;
}
function tubGeo(w, l, d, r, rb, n = 6, m = 6) {        // open-top tub, bottom at z 0, rim at z d
  const rings = [];
  for (let j = 0; j <= m; j++) { const t = Math.PI / 2 * j / m; rings.push([rb * (1 - Math.sin(t)), rb * (1 - Math.cos(t))]); }
  rings.push([0, d]);
  const pos = [], idx = [];
  const N = 4 * (n + 1);
  for (const [inset, z] of rings) for (const [x, y] of rrect(w - 2 * inset, l - 2 * inset, Math.max(r - inset, 0.003), n)) pos.push(x, y, z);
  const c = pos.length / 3; pos.push(0, 0, 0);
  for (let i = 0; i < N; i++) idx.push(c, (i + 1) % N, i);
  for (let j = 0; j < rings.length - 1; j++) for (let i = 0; i < N; i++) {
    const a = j * N + i, b = j * N + (i + 1) % N, a2 = a + N, b2 = b + N;
    idx.push(a, b, b2, a, b2, a2);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}
function ringGeo(w, l, r, t, z, n = 6) {                 // flat rim between the shell and the lining
  const o = rrect(w, l, r, n), i = rrect(w - 2 * t, l - 2 * t, Math.max(r - t, 0.003), n), N = o.length;
  const pos = [], idx = [];
  for (const [x, y] of o) pos.push(x, y, z);
  for (const [x, y] of i) pos.push(x, y, z);
  for (let k = 0; k < N; k++) { const a = k, b = (k + 1) % N; idx.push(a, b, N + b, a, N + b, N + a); }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
  return g;
}
function bandGeo(w, l, r, z0, z1, grow, n = 6) {          // a vertical band just outside the shell (zip tape)
  const o = rrect(w + 2 * grow, l + 2 * grow, r + grow, n), N = o.length, pos = [], idx = [];
  for (const z of [z0, z1]) for (const [x, y] of o) pos.push(x, y, z);
  for (let k = 0; k < N; k++) { const a = k, b = (k + 1) % N; idx.push(a, b, N + b, a, N + b, N + a); }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
  return g;
}
const CASE = {};
function buildCase() {
  const C = W.case, [w, l] = C.outer, D = C.base_depth, DL = C.lid_depth, h = C.hinge_h, t = C.shell, r = 0.05, rb = 0.03;
  Object.assign(CASE, { w, l, D, DL, h });
  const root = new THREE.Group(); scene.add(root); CASE.root = root;
  // base
  mesh(tubGeo(w, l, D, r, rb), M.shell, null, 0, root);
  const lb = mesh(projectUV(tubGeo(w - 2 * t, l - 2 * t, D - C.floor - 0.001, r - t, rb * 0.6), 0.08), M.lining, [0, 0, C.floor], 0, root, false);
  lb.material = M.lining;
  mesh(ringGeo(w, l, r, t, D), M.shell, null, 0, root);
  mesh(bandGeo(w, l, r, D - 0.010, D - 0.0005, 0.0012), M.tape, null, 0, root, false);
  // lid: a group pivoting on the hinge (local -x edge, at mid depth); open = lying beside the base
  const pivot = new THREE.Group(); pivot.position.set(-w / 2, 0, h); root.add(pivot); CASE.pivot = pivot;
  const lid = new THREE.Group(); lid.position.set(-w / 2, 0, -h); pivot.add(lid);
  mesh(tubGeo(w, l, DL, r, rb * 1.4), M.shell, null, 0, lid);
  mesh(projectUV(tubGeo(w - 2 * t - 0.016, l - 2 * t - 0.016, DL - C.lid_floor - 0.001, r - t, rb), 0.08), M.lining, [0, 0, C.lid_floor], 0, lid, false);
  mesh(ringGeo(w, l, r, t + 0.008, DL), M.shell, null, 0, lid);
  mesh(bandGeo(w, l, r, DL - 0.010, DL - 0.0005, 0.0012), M.tape, null, 0, lid, false);
  // things on the closed case, given in the base frame of the closed case, mounted on the lid
  const onLid = (obj, p) => { obj.position.set(-(p[0] + w / 2), p[1], -(p[2] - h)); obj.rotation.y = Math.PI; pivot.add(obj); return obj; };
  const hw = C.hardware;
  const plate = new THREE.Mesh(new RoundedBoxGeometry(0.012, 0.13, 0.05, 2, 0.004), M.alu);
  plate.castShadow = true; onLid(plate, [hw.dial_x - 0.002, 0.065, hw.dial_z + 0.004]);
  CASE.dials = [];
  for (let k = 0; k < 3; k++) {
    const d = new THREE.Mesh(new THREE.CylinderGeometry(hw.dial_r, hw.dial_r, hw.dial_w, 20), digitsMaterial());
    const holder = new THREE.Group(); holder.add(d);
    onLid(holder, [hw.dial_x + 0.005, hw.dial_y[k], hw.dial_z]);
    CASE.dials.push(d);
  }
  const btn = new THREE.Mesh(new RoundedBoxGeometry(0.008, 0.03, 0.016, 2, 0.002), M.lockRed);
  onLid(btn, [hw.dial_x + 0.004, hw.button.y, hw.button.z]); CASE.button = btn;
  // spinner wheels at the wheel end (two on each half)
  const wheel = () => {
    const g = new THREE.Group();
    const housing = new THREE.Mesh(new RoundedBoxGeometry(0.05, 0.05, 0.06, 2, 0.012), M.shell); housing.position.y = 0.02; g.add(housing);
    const wl = new THREE.Mesh(new THREE.CylinderGeometry(C.wheel_r, C.wheel_r, 0.025, 20), M.wheel); wl.rotation.z = Math.PI / 2; wl.position.y = -0.005; g.add(wl);
    g.traverse(o => { if (o.isMesh) o.castShadow = true; });
    return g;
  };
  for (const sx of [-C.wheel_x, C.wheel_x]) {
    const a = wheel(); a.position.set(sx, C.wheel_y + 0.01, C.wheel_z_base); root.add(a);
    const b = wheel(); onLid(b, [sx, C.wheel_y + 0.01, D + DL - C.wheel_z_base]);
  }
  // telescopic handle at the handle end: two tubes and the grip bar
  const bar = new THREE.Group(); root.add(bar); CASE.bar = bar;
  const grip = new THREE.Mesh(new THREE.CapsuleGeometry(C.bar.r, 2 * C.bar.half, 6, 16), M.shell);
  grip.rotation.z = Math.PI / 2; grip.position.set(C.bar.x, 0, 0); grip.castShadow = true; bar.add(grip);
  CASE.tubes = [];
  for (const sx of [C.bar.x - C.bar.half - 0.016, C.bar.x + C.bar.half + 0.016]) {
    const tb = new THREE.Mesh(new THREE.BoxGeometry(0.016, 1, 0.016), M.alu); tb.castShadow = true; root.add(tb); tb.userData.sx = sx; CASE.tubes.push(tb);
    const post = new THREE.Mesh(new RoundedBoxGeometry(0.03, 0.06, 0.03, 2, 0.008), M.shell); post.position.set(sx, -0.03, 0); post.castShadow = true; bar.add(post);
  }
  // zip sliders and their pulls on the front (+x) wall at the seam
  CASE.sliders = [];
  for (const k of [0, 1]) {
    const s = new THREE.Group();
    const body = new THREE.Mesh(new RoundedBoxGeometry(0.012, 0.024, 0.016, 2, 0.003), M.slider); body.castShadow = true; s.add(body);
    const tab = new THREE.Mesh(new RoundedBoxGeometry(0.003, 0.012, hw.tab.len - 0.004, 2, 0.0013), M.pull);
    tab.position.set(0.004, 0, -(hw.tab.len - 0.004) / 2); tab.castShadow = true; s.add(tab);
    s.userData.tab = tab;
    root.add(s); CASE.sliders.push(s);
  }
}
let digitsTex = null;
function digitsMaterial() {
  if (!digitsTex) {
    const c = document.createElement('canvas'); c.width = 256; c.height = 32;
    const x = c.getContext('2d'); x.fillStyle = '#2a2a2c'; x.fillRect(0, 0, 256, 32);
    x.fillStyle = '#f2f2f2'; x.font = 'bold 22px sans-serif'; x.textAlign = 'center'; x.textBaseline = 'middle';
    for (let i = 0; i < 10; i++) x.fillText(String(i), 12.8 + i * 25.6, 17);
    digitsTex = new THREE.CanvasTexture(c); digitsTex.colorSpace = THREE.SRGBColorSpace;
  }
  return new THREE.MeshStandardMaterial({ map: digitsTex, roughness: 0.4, metalness: 0.3 });
}
// Case poses: lying open on the bed (scene.SUITCASE), or standing on its wheels at a floor spot.
function lyingPose() {
  const C = W.case;
  return { p: V(C.pos[0], C.pos[1], C.pos[2]), q: new THREE.Quaternion().setFromAxisAngle(V(0, 0, 1), C.yaw) };
}
function uprightPose(at) {
  const C = W.case, spot = C.floor_spots[at] || C.floor_spots.bed_foot;
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, Math.PI / 2, 'ZYX'));
  const foot = V(0, C.wheel_y - C.wheel_r + 0.01, (CASE.D + CASE.DL) / 2).applyQuaternion(q);
  return { p: V(spot.case[0], spot.case[1], 0).sub(foot), q };
}
function setCaseVisual(s) {
  const C = W.case, hw = C.hardware;
  CASE.lidAngle = s.lid === 'open' ? 0 : Math.PI;
  CASE.zip = s.zip_open;
  CASE.trolley = [0, 0.2, 0.38][s.trolley] || 0;
  CASE.dialsV = s.dials.slice();
  CASE.docked = s.pulls === 'docked';
  const pose = s.case_pose === 'upright_floor' ? uprightPose(s.case_at) : lyingPose();
  CASE.root.position.copy(pose.p); CASE.root.quaternion.copy(pose.q);
  CASE.upright = s.case_pose === 'upright_floor';
  CASE.at = s.case_at;
  updateCaseParts();
}
function updateCaseParts() {
  const C = W.case, hw = C.hardware;
  CASE.pivot.rotation.y = CASE.lidAngle;
  // sliders: open near the ends of the front wall, shut at the lock
  const closed = 1 - CASE.zip;
  const ya = lerp(hw.slider.open_a, hw.dock_y - hw.slider.gap, closed), yb = lerp(hw.slider.open_b, hw.dock_y + hw.slider.gap, closed);
  CASE.sliders.forEach((s, k) => {
    s.position.set(CASE.w / 2 + 0.007, k ? yb : ya, CASE.D);
    s.visible = CASE.lidAngle > 2.6;           // they ride on the seam, which only exists with the lid down
    const tab = s.userData.tab;
    if (CASE.docked) { tab.rotation.y = Math.PI; tab.position.set(0.004, 0, (hw.tab.len - 0.004) / 2 - 0.004); }
    else { tab.rotation.y = 0; tab.position.set(0.004, 0, -(hw.tab.len - 0.004) / 2); }
  });
  CASE.dials.forEach((d, k) => { d.rotation.y = -(CASE.dialsV[k] / 10) * Math.PI * 2 + Math.PI / 2; });
  const ext = CASE.trolley;
  CASE.bar.position.set(0, C.bar.y + ext, C.bar.z);
  for (const tb of CASE.tubes) {
    const y0 = CASE.l / 2 - 0.01, y1 = C.bar.y - 0.05 + ext;
    tb.visible = ext > 0.004;
    tb.scale.y = Math.max(y1 - y0, 0.001); tb.position.set(tb.userData.sx, (y0 + y1) / 2, C.bar.z);
  }
}
// The case's footprint for walking, when it stands on the floor.
function caseObstacle() {
  if (!CASE.upright) return null;
  const c = CASE.root.localToWorld(V(0, 0, (CASE.D + CASE.DL) / 2));
  return { x0: c.x - 0.22, x1: c.x + 0.22, y0: c.y - 0.3, y1: c.y + 0.3 };
}

// ------------------------------------------------------------------ the person (render_blender.Body)
class Person {
  constructor() {
    this.x = 1.55; this.y = 0.64; this.heading = Math.PI / 2;
    this.phase = 0; this.speed = 0; this.crouch = 0; this.lean = 0;
    this.hand = { r: null, l: null }; this.handW = { r: 0, l: 0 };
    this.look = null;
    this.g = new THREE.Group(); scene.add(this.g);
    const seg = (r0, r1, mat) => { const geo = new THREE.CylinderGeometry(r1, r0, 1, 28, 1, false); geo.translate(0, 0.5, 0); return this.add(geo, mat); };
    const ball = (r, mat) => this.add(new THREE.SphereGeometry(r, 28, 14), mat);
    this.segs = {
      upper_r: seg(0.050, 0.040, M.shirt), upper_l: seg(0.050, 0.040, M.shirt),
      fore_r: seg(0.036, 0.028, M.skin), fore_l: seg(0.036, 0.028, M.skin),
      thigh_r: seg(0.078, 0.056, M.pants), thigh_l: seg(0.078, 0.056, M.pants),
      shin_r: seg(0.054, 0.040, M.pants), shin_l: seg(0.054, 0.040, M.pants),
      neck: seg(0.052, 0.048, M.skin), waist: seg(0.150, 0.150, M.shirt),
    };
    this.balls = {
      shoulder_r: ball(0.056, M.shirt), shoulder_l: ball(0.056, M.shirt), elbow_r: ball(0.040, M.skin), elbow_l: ball(0.040, M.skin),
      hip_r: ball(0.080, M.pants), hip_l: ball(0.080, M.pants), knee_r: ball(0.057, M.pants), knee_l: ball(0.057, M.pants),
    };
    this.chest = this.add(new RoundedBoxGeometry(0.22, 0.36, 0.34, 8, 0.10), M.shirt);
    this.hips = this.add(new RoundedBoxGeometry(0.21, 0.33, 0.20, 6, 0.08), M.pants);
    this.head = this.add(new RoundedBoxGeometry(0.175, 0.15, 0.215, 8, 0.068), M.skin);
    this.hair = new THREE.Mesh(new RoundedBoxGeometry(0.185, 0.16, 0.10, 6, 0.05), M.hair);
    this.hair.position.set(-0.014, 0, 0.062); this.hair.castShadow = true; this.head.add(this.hair);
    this.shoes = [0, 1].map(() => this.add(new RoundedBoxGeometry(0.27, 0.105, 0.075, 5, 0.035), M.shoe));
    this.hands = ['r', 'l'].map(() => {
      const h = new THREE.Group(); this.g.add(h);
      const palm = new THREE.Mesh(new RoundedBoxGeometry(0.10, 0.085, 0.032, 4, 0.014), M.skin); palm.position.x = 0.04; palm.castShadow = true; h.add(palm);
      const thumb = new THREE.Mesh(new THREE.CapsuleGeometry(0.012, 0.035, 4, 10), M.skin); thumb.rotation.z = Math.PI / 2.6; thumb.position.set(0.02, 0.045, 0.004); thumb.castShadow = true; h.add(thumb);
      return h;
    });
    this.hands[0].children.forEach((c, i) => { c.position.y *= -1; if (i) c.rotation.z *= -1; });   // thumbs forward
    this.pts = {};
    this.update(0);
  }
  add(geo, mat) { const m = new THREE.Mesh(geo, mat); m.castShadow = true; m.receiveShadow = true; this.g.add(m); return m; }
  frame() {
    const f = V(Math.cos(this.heading), Math.sin(this.heading), 0), l = V(-Math.sin(this.heading), Math.cos(this.heading), 0);
    return { f, l, u: V(0, 0, 1) };
  }
  static place(m, a, b) {                          // a cylinder from a to b
    const d = b.clone().sub(a), len = d.length();
    m.position.copy(a); m.scale.set(1, Math.max(len, 1e-4), 1);
    m.quaternion.setFromUnitVectors(V(0, 1, 0), d.divideScalar(Math.max(len, 1e-6)));
  }
  static ik(a, t, l1, l2, pole) {                  // two-bone IK: the middle joint
    const d = t.clone().sub(a), dist = clamp(d.length(), 1e-4, (l1 + l2) * 0.999);
    const dir = d.normalize(), x = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist), hgt = Math.sqrt(Math.max(l1 * l1 - x * x, 0));
    const pd = pole.clone().sub(a); pd.sub(dir.clone().multiplyScalar(pd.dot(dir))).normalize();
    return { mid: a.clone().add(dir.clone().multiplyScalar(x)).add(pd.multiplyScalar(hgt)), end: a.clone().add(dir.clone().multiplyScalar(dist)) };
  }
  basis(fwd, left, up) { return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(fwd, left, up)); }
  update(dt) {
    const { f, l, u } = this.frame();
    const s = clamp(this.speed / 1.4, 0, 1.4);
    this.phase += dt * this.speed * 4.2;
    const bob = s > 0.05 ? 0.018 * s * Math.abs(Math.sin(this.phase)) : 0;
    const c = this.crouch;
    const pelvis = V(this.x, this.y, 1.0 - 0.32 * c - bob);       // legs (0.45 + 0.43) nearly straight when standing
    const lean = 0.06 * s + 0.55 * c + this.lean;
    const tu = u.clone().multiplyScalar(Math.cos(lean)).add(f.clone().multiplyScalar(Math.sin(lean)));
    const tf = f.clone().multiplyScalar(Math.cos(lean)).sub(u.clone().multiplyScalar(Math.sin(lean)));
    const P = this.pts;
    P.pelvis = pelvis; P.f = f; P.l = l; P.tu = tu;
    this.chest.position.copy(pelvis.clone().add(tu.clone().multiplyScalar(0.31))); this.chest.quaternion.copy(this.basis(tf, l, tu));
    this.hips.position.copy(pelvis.clone().add(u.clone().multiplyScalar(0.02))); this.hips.quaternion.copy(this.basis(f, l, u));
    Person.place(this.segs.waist, pelvis.clone().add(tu.clone().multiplyScalar(0.06)), pelvis.clone().add(tu.clone().multiplyScalar(0.20)));
    const neckA = pelvis.clone().add(tu.clone().multiplyScalar(0.48)), neckB = pelvis.clone().add(tu.clone().multiplyScalar(0.60));
    Person.place(this.segs.neck, neckA, neckB);
    // head: looks at the target if any
    let hf = tf.clone();
    if (this.look) { const d = this.look.clone().sub(neckB); if (d.length() > 0.1) hf = d.normalize(); }
    const hl = u.clone().cross(hf).normalize(), hu = hf.clone().cross(hl).normalize();
    P.head = pelvis.clone().add(tu.clone().multiplyScalar(0.71));
    this.head.position.copy(P.head); this.head.quaternion.copy(this.basis(hf, hl, hu));
    P.eye = P.head.clone().add(hf.clone().multiplyScalar(0.07)).add(hu.clone().multiplyScalar(0.02)); P.hf = hf;
    // legs
    for (const [k, side] of [['r', -1], ['l', 1]]) {
      const hip = pelvis.clone().add(l.clone().multiplyScalar(0.095 * side)).add(V(0, 0, -0.06));
      const ph = this.phase + (side > 0 ? Math.PI : 0);
      const stride = 0.24 * Math.min(s, 1.2), lift = 0.08 * Math.min(s, 1) * Math.max(0, Math.cos(ph));
      const ankle = V(hip.x, hip.y, 0.075 + lift).add(f.clone().multiplyScalar(stride * Math.sin(ph) + 0.02 - 0.10 * c)).add(l.clone().multiplyScalar(0.01 * side));
      const { mid: knee, end } = Person.ik(hip, ankle, 0.45, 0.43, hip.clone().add(f.clone().multiplyScalar(1)));
      Person.place(this.segs['thigh_' + k], hip, knee); Person.place(this.segs['shin_' + k], knee, end.clone().add(V(0, 0, 0.02)));
      this.balls['hip_' + k].position.copy(hip); this.balls['knee_' + k].position.copy(knee);
      const shoe = this.shoes[k === 'r' ? 0 : 1];
      shoe.position.copy(V(end.x, end.y, end.z - 0.0375).add(f.clone().multiplyScalar(0.05)));
      shoe.quaternion.copy(this.basis(f, l, u));
    }
    // arms
    ['r', 'l'].forEach((k, i) => {
      const side = k === 'r' ? -1 : 1;
      const sh = pelvis.clone().add(tu.clone().multiplyScalar(0.45)).add(l.clone().multiplyScalar(0.19 * side));
      const swing = -0.16 * Math.min(s, 1) * Math.sin(this.phase + (side > 0 ? 0 : Math.PI));
      const rest = sh.clone().add(tu.clone().multiplyScalar(-0.52)).add(f.clone().multiplyScalar(0.06 + swing)).add(l.clone().multiplyScalar(0.03 * side));
      const w = this.handW[k], tgt = this.hand[k] ? rest.clone().lerp(this.hand[k], w) : rest;
      const pole = sh.clone().add(f.clone().multiplyScalar(-0.5)).add(u.clone().multiplyScalar(-0.3)).add(l.clone().multiplyScalar(0.45 * side));
      const { mid: elbow, end: wrist } = Person.ik(sh, tgt, 0.29, 0.30, pole);
      Person.place(this.segs['upper_' + k], sh, elbow); Person.place(this.segs['fore_' + k], elbow, wrist);
      this.balls['shoulder_' + k].position.copy(sh); this.balls['elbow_' + k].position.copy(elbow);
      const hd = wrist.clone().sub(elbow).normalize();
      let side2 = l.clone().multiplyScalar(side); side2.sub(hd.clone().multiplyScalar(side2.dot(hd))).normalize();
      const palmN = side2.clone().negate();               // palms face the middle (or the thing held)
      const hy = palmN.clone().cross(hd).normalize();
      const hand = this.hands[i];
      hand.position.copy(wrist.clone().sub(hd.clone().multiplyScalar(0.035)));
      hand.quaternion.copy(this.basis(hd, hy, palmN).normalize());
      P['wrist_' + k] = wrist; P['sh_' + k] = sh;
    });
  }
  setHeadVisible(v) { this.head.visible = v; this.segs.neck.visible = v; }
}

// ------------------------------------------------------------------ lights and sky (render_blender.setup_world)
let sun = null;
const roomLights = [];
function buildLights() {
  sun = new THREE.DirectionalLight(lin([1.0, 0.93, 0.84]), TUNE.sunI ?? 3.2);
  const dir = V(0, 0, -1).applyEuler(new THREE.Euler(58 * Math.PI / 180, 0, 165 * Math.PI / 180, 'ZYX'));
  sun.target.position.set(3.5, 3.2, 0);
  sun.position.copy(sun.target.position).sub(dir.multiplyScalar(18));
  sun.castShadow = true;
  const sc = sun.shadow.camera; sc.left = -7; sc.right = 7; sc.top = 7; sc.bottom = -7; sc.near = 2; sc.far = 40;
  sun.shadow.mapSize.set(4096, 4096); sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.02; sun.shadow.radius = 3;
  scene.add(sun, sun.target);
  for (const r of Object.values(W.rooms)) {
    const cx = (r.x[0] + r.x[1]) / 2, cy = (r.y[0] + r.y[1]) / 2, sx = r.x[1] - r.x[0], sy = r.y[1] - r.y[0];
    // Blender: 7 * area W over a panel of half the room's size -> radiance P / (pi * A)
    const L = new THREE.RectAreaLight(lin([1.0, 0.88, 0.74]), 7 * sx * sy / (Math.PI * 0.25 * sx * sy) * LIGHT_GAIN, sx * 0.5, sy * 0.5);
    L.position.set(cx, cy, W.wall_h - 0.02); L.up.set(0, 1, 0); L.lookAt(cx, cy, 0);
    scene.add(L); roomLights.push(L);
  }
  scene.add(new THREE.HemisphereLight(lin([0.85, 0.88, 0.95]), lin([0.45, 0.38, 0.30]), TUNE.hemi ?? 0.08));
  new RGBELoader(manager).load('pack/' + W.hdri, t => {
    t.mapping = THREE.EquirectangularReflectionMapping;
    scene.background = t; scene.environment = t;
    // the HDRI is y up; the scene is z up. Blender also turns it 200 degrees about z.
    scene.backgroundRotation.set(Math.PI / 2, 0, 0);
    scene.environmentRotation.set(Math.PI / 2, 0, 0);
    scene.environmentIntensity = TUNE.env ?? 0.08;
  });
}
const LIGHT_GAIN = TUNE.area ?? 1.1;

// ------------------------------------------------------------------ walking and collision
const RADIUS = 0.2;
function collide(x, y) {
  const boxes = walkable.concat(caseObstacle() ? [caseObstacle()] : []);
  for (let it = 0; it < 3; it++) {
    for (const b of boxes) {
      const cx = clamp(x, b.x0, b.x1), cy = clamp(y, b.y0, b.y1);
      const dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy);
      if (d < RADIUS) {
        if (d > 1e-6) { x = cx + dx / d * RADIUS; y = cy + dy / d * RADIUS; }
        else {                                       // centre inside the box: out the nearest side
          const o = [[x - b.x0, -1, 0], [b.x1 - x, 1, 0], [y - b.y0, 0, -1], [b.y1 - y, 0, 1]].sort((a, c) => a[0] - c[0])[0];
          x += o[1] * (o[0] + RADIUS); y += o[2] * (o[0] + RADIUS);
        }
      }
    }
  }
  return [clamp(x, 0.2, 6.8), clamp(y, 0.2, 6.2)];
}
function roomAt(x, y) {
  for (const [k, r] of Object.entries(W.rooms)) if (x >= r.x[0] && x <= r.x[1] && y >= r.y[0] && y <= r.y[1]) return k;
  return 'living';
}

// ------------------------------------------------------------------ cameras
const cam = { mode: 'follow', yaw: 0.9, pitch: 0.62, dist: 2.7, egoPitch: -0.25, room: 'bedroom', roomSince: 0, touched: -99 };
const MODES = ['follow', 'room', 'first person'];
const ray = new THREE.Raycaster();
function updateCamera(dt) {
  const P = person.pts;
  const aspect = innerWidth / innerHeight;
  if (cam.mode === 'room') {
    const room = roomAt(person.x, person.y);
    if (room !== cam.room && now() - cam.roomSince > 1.2) { cam.room = room; cam.roomSince = now(); }
    const [eye, tgt, lens] = W.overview[cam.room];
    const hfov = 2 * Math.atan(18 / lens);
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(hfov / 2) / aspect));
    camera.position.set(...eye); camera.up.set(0, 0, 1); camera.lookAt(...tgt);
  } else if (cam.mode === 'first person') {
    camera.fov = 75;
    const e = P.eye || V(person.x, person.y, 1.6);
    camera.position.copy(e);
    const d = V(Math.cos(person.heading) * Math.cos(cam.egoPitch), Math.sin(person.heading) * Math.cos(cam.egoPitch), Math.sin(cam.egoPitch));
    camera.up.set(0, 0, 1); camera.lookAt(e.clone().add(d));
  } else {
    camera.fov = 55;
    const tgt = V(person.x, person.y, 1.25 - 0.25 * person.crouch);
    const dir = V(Math.cos(cam.yaw) * Math.cos(cam.pitch), Math.sin(cam.yaw) * Math.cos(cam.pitch), Math.sin(cam.pitch));
    // walking: settle in behind the person (unless the player has just turned the camera)
    if (person.speed > 0.3 && now() - cam.touched > 2.5)
      cam.yaw += angDiff(cam.yaw, person.heading + Math.PI) * Math.min(1, dt * 0.9);
    const free = (yaw, pitch = cam.pitch) => {
      const d = V(Math.cos(yaw) * Math.cos(pitch), Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch));
      ray.set(tgt, d); ray.far = Math.min(cam.dist, (W.wall_h - 0.12 - tgt.z) / Math.max(Math.sin(pitch), 0.05));
      const h = ray.intersectObjects(occluders, false)[0];
      return h ? h.distance : ray.far;
    };
    // a wall in the way: look down from higher up first (there is no ceiling), then swing round
    let want = cam.pitch;
    for (let p = cam.pitch; p <= 1.3 && free(cam.yaw, p) < Math.min(1.5, cam.dist); p += 0.1) want = Math.min(1.3, p + 0.1);
    cam.pitchNow = lerp(cam.pitchNow ?? cam.pitch, want, Math.min(1, dt * 4));
    let dist = free(cam.yaw, cam.pitchNow);
    // too close to a wall: swing round to where there is room (unless the player is turning it)
    if (dist < Math.min(1.4, cam.dist) && !drag && !keys.has('q') && !keys.has('r')) {
      let best = cam.yaw, bestD = dist;
      for (const off of [0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4, 1.9, -1.9, 2.5, -2.5, Math.PI]) {
        const d = free(cam.yaw + off, cam.pitchNow) - Math.abs(off) * 0.15;
        if (d > bestD + 0.2) { best = cam.yaw + off; bestD = d; }
      }
      cam.yaw += angDiff(cam.yaw, best) * Math.min(1, dt * 2.5);
      dist = free(cam.yaw, cam.pitchNow);
    }
    dist = dist < cam.dist ? Math.max(0.3, dist - 0.12) : dist;
    cam.close = dist < 0.6;
    dir.set(Math.cos(cam.yaw) * Math.cos(cam.pitchNow), Math.sin(cam.yaw) * Math.cos(cam.pitchNow), Math.sin(cam.pitchNow));
    const pos = tgt.clone().add(dir.multiplyScalar(dist));
    camera.position.lerp(pos, cam.snap ? 1 : 1 - Math.exp(-dt * 12)); cam.snap = false;
    camera.up.set(0, 0, 1); camera.lookAt(tgt);
  }
  camera.aspect = aspect; camera.updateProjectionMatrix();
  person.setHeadVisible(cam.mode !== 'first person');
  person.g.visible = !(cam.mode === 'follow' && cam.close);   // the camera squeezed in against the body
  ceiling.visible = cam.mode === 'first person';
}
// direction of "forward" on the floor for the keys
function camForward() {
  if (cam.mode === 'first person') return V(Math.cos(person.heading), Math.sin(person.heading), 0);
  const d = V(); camera.getWorldDirection(d); d.z = 0;
  return d.lengthSq() < 1e-6 ? V(1, 0, 0) : d.normalize();
}

// ------------------------------------------------------------------ item and case placement in the world
function itemHome(name) {
  const it = W.items[name];
  return { p: V(it.xy[0], it.xy[1], it.top + it.size[2] / 2), q: new THREE.Quaternion().setFromAxisAngle(V(0, 0, 1), it.yaw) };
}
function itemInCase(name, place) {                     // place: x, y, z (bottom), yaw in the base frame
  const it = W.items[name];
  const local = V(place.x, place.y, place.z + it.size[2] / 2);
  const p = CASE.root.localToWorld(local.clone());
  const q = CASE.root.quaternion.clone().multiply(new THREE.Quaternion().setFromAxisAngle(V(0, 0, 1), place.yaw));
  return { p, q };
}
function carryPose(name) {
  const it = W.items[name], P = person.pts, { f, l } = person.frame();
  if (it.hands === 1) {
    const p = P.pelvis.clone().add(f.clone().multiplyScalar(0.30)).add(l.clone().multiplyScalar(-0.17)); p.z = 0.98 - 0.25 * person.crouch;
    return { p, q: new THREE.Quaternion().setFromAxisAngle(V(0, 0, 1), person.heading) };
  }
  const p = P.pelvis.clone().add(f.clone().multiplyScalar(0.34)); p.z = 1.02 + it.size[2] * 0.15 - 0.28 * person.crouch;
  return { p, q: new THREE.Quaternion().setFromAxisAngle(V(0, 0, 1), person.heading + Math.PI / 2) };
}
function setItemPose(name, pose) {
  const g = items[name];
  if (g.parent !== scene) scene.attach(g);
  g.position.copy(pose.p); g.quaternion.copy(pose.q);
}
function graspPoints(name, pose) {                      // where the hands go on an item
  const it = W.items[name];
  if (it.hands === 1) {
    const toward = V(person.x, person.y, pose.p.z).sub(pose.p); toward.z = 0; toward.normalize();
    return { r: pose.p.clone().add(toward.multiplyScalar(0.045)).add(V(0, 0, 0.02)), l: null };
  }
  const ax = V(1, 0, 0).applyQuaternion(pose.q).multiplyScalar(it.size[0] / 2 + 0.03);
  const a = pose.p.clone().add(ax), b = pose.p.clone().sub(ax);
  const { l } = person.frame();
  const aLeft = a.clone().sub(V(person.x, person.y, 0)).dot(l) > b.clone().sub(V(person.x, person.y, 0)).dot(l);
  return aLeft ? { l: a, r: b } : { l: b, r: a };
}

// ------------------------------------------------------------------ game state from the simulator
let S = null;               // last state from game.py
let started = false, startT = 0, finished = false, busy = false, anim = null;
let task = 'weekend_trip';
const now = () => performance.now() / 1000;
const gameTime = () => started ? (finished ? finishT : now() - startT) : 0;
let finishT = 0;
const path = [];
function applyState(s, { skipItems = [] } = {}) {
  S = s;
  setCaseVisual(s);
  for (const [name, d] of Object.entries(s.items)) {
    if (skipItems.includes(name)) continue;
    if (d.loc === 'held') continue;                    // posed every frame
    if (d.loc === 'case' && d.place) {
      const g = items[name];
      CASE.root.attach(g);
      g.position.set(d.place.x, d.place.y, d.place.z + W.items[name].size[2] / 2);
      g.quaternion.setFromAxisAngle(V(0, 0, 1), d.place.yaw);
    } else setItemPose(name, itemHome(name));
  }
  renderChecklist(s); renderConv(s.conv);
}
function renderChecklist(s) {
  const t = W.tasks[task];
  $('taskTitle').textContent = t.title;
  $('taskNote').textContent = t.note && t.lock ? 'Lock it before it leaves: pulls in the lock, dials scrambled.' : '';
  const todo = s ? s.checklist.filter(c => !c.rule) : t.pack.map(n => ({ text: 'Pack the ' + W.items[n].label, done: false }));
  const rules = s ? s.checklist.filter(c => c.rule) : [];
  $('todo').innerHTML = todo.map(c => `<li class="${c.done ? 'done' : ''}"><span class="box">${c.done ? '✓' : ''}</span><span class="t">${esc(c.text)}</span></li>`).join('');
  $('rulesHead').hidden = !rules.length;
  $('rules').innerHTML = rules.map(c => `<li class="${c.done ? 'done' : 'broken'}"><span class="box">${c.done ? '✓' : '!'}</span><span>${esc(c.text)}</span></li>`).join('');
  const packed = todo.filter(c => c.text.startsWith('Pack') && c.done).length, total = todo.filter(c => c.text.startsWith('Pack')).length;
  $('progress').textContent = s ? `${packed} of ${total} packed · ${s.checklist.filter(c => c.done).length}/${s.checklist.length} done` : '';
  $('clCount').textContent = s ? `${s.checklist.filter(c => c.done).length}/${s.checklist.length}` : '';
}
function renderConv(conv) {
  const box = $('conv');
  $('asCount').textContent = conv && conv.length ? String(conv.length) : '';
  if (!conv || !conv.length) { box.innerHTML = '<div class="empty">No messages yet. The assistant slot is empty, so nothing will come back.</div>'; return; }
  box.innerHTML = conv.map(c => `<div class="msg ${c.speaker}"><span class="who">${c.speaker === 'user' ? 'You' : 'Assistant'} · ${fmtTime(c.t)}</span>${esc(c.text)}</div>`).join('');
  box.scrollTop = box.scrollHeight;
}

// ------------------------------------------------------------------ what can be done here
let opts = { options: [], blocked: [], where: '' }, optKey = '', optPending = false, optAt = 0;
function wantOptions(force) {
  if (!py.ready || !S || busy || finished) return;
  const key = `${person.x.toFixed(2)},${person.y.toFixed(2)},${(person.heading).toFixed(1)},${S.holding},${S.lid},${S.zip_open},${S.case_pose}`;
  if (!force && (key === optKey || optPending || now() - optAt < 0.12)) return;
  optKey = key; optPending = true; optAt = now();
  call('game_options', { x: person.x, y: person.y, heading: person.heading }).then(o => {
    optPending = false;
    if (busy) return;
    opts = o; renderPrompt();
  }).catch(e => { optPending = false; console.error(e); });
}
function renderPrompt() {
  const head = $('promptHead'), list = $('opts'), bl = $('blocked');
  if (!py.ready) { head.textContent = py.error ? 'The simulator could not load: ' + py.error : `Simulator loading (${pyProgress}) — you can walk around.`; list.innerHTML = ''; bl.innerHTML = ''; return; }
  if (busy) { head.textContent = anim ? anim.label : '…'; list.innerHTML = ''; bl.innerHTML = ''; return; }
  const o = opts.options;
  head.textContent = o.length ? (S.holding ? `Holding the ${W.items[S.holding].label}` : 'Within reach') : (S && S.holding ? `Holding the ${W.items[S.holding].label}: walk to the case to pack it, or back to where it belongs` : 'Walk up to something on the checklist');
  $('promptHint').textContent = opts.where ? 'in ' + opts.where.replace(/^the /, '') : '';
  list.innerHTML = o.map((q, i) => `<button class="opt ${i ? '' : 'primary'}" data-i="${i}"><kbd>${i ? i + 1 : 'E'}</kbd><span>${esc(q.text)}</span></button>`).join('');
  bl.innerHTML = (opts.blocked || []).map(b => `<div class="blocked">${esc(b.text.replace(/^./, c => c.toUpperCase()))}: ${esc(b.why)}</div>`).join('');
}
function targetPoint(q) {
  if (!q) return null;
  if (q.facing && items[q.facing]) return items[q.facing].getWorldPosition(V()).add(V(0, 0, 0.12));
  const m = q.id.match(/^(?:pick_up|put_back)\((\w+)\)/);
  if (m && items[m[1]]) return items[m[1]].getWorldPosition(V()).add(V(0, 0, 0.12));
  return CASE.root.localToWorld(V(0, 0, CASE.D + 0.1));
}

// ------------------------------------------------------------------ actions: ask the simulator, then animate
async function doOption(i) {
  const q = opts.options[i];
  if (!q || busy || !started || finished) return;
  busy = true; renderPrompt(); $('tag').hidden = true;
  let r;
  try {
    r = await call('game_act', { id: q.id, station: q.station, facing: q.facing, t: gameTime(), x: person.x, y: person.y });
  } catch (e) { toast(String(e.message || e), true); busy = false; wantOptions(true); return; }
  if (!r.ok) { toast(r.feedback || 'That did not work.', true); busy = false; wantOptions(true); return; }
  if (r.feedback) toast(r.feedback);
  await animate(q, r.state);
  applyState(r.state);
  busy = false; optKey = '';
  wantOptions(true);
  if (r.state.done && !finished) setTimeout(finish, 700);
}
function play(label, T, step) {
  return new Promise(res => { anim = { label, t: 0, T, step, done: res }; renderPrompt(); });
}
const turnTo = (p, k) => { const want = Math.atan2(p.y - person.y, p.x - person.x); person.heading += angDiff(person.heading, want) * Math.min(1, k); };
const crouchFor = z => clamp((1.02 - z) / 0.55, 0, 1);
function animate(q, ns) {
  const id = q.id, verb = id.split('(')[0];
  const before = S;
  person.look = null;
  if (verb === 'pick_up') {
    const name = id.slice(8, -1), fromCase = before.items[name].loc === 'case';
    const start = { p: items[name].getWorldPosition(V()), q: items[name].getWorldQuaternion(new THREE.Quaternion()) };
    scene.attach(items[name]);
    const cz = crouchFor(start.p.z);
    return play(`Picking up the ${W.items[name].label}`, 1.5, k => {
      turnTo(start.p, k * 6);
      person.look = start.p;
      const reach = ease(seg(k, 0, 0.35)), lift = ease(seg(k, 0.45, 0.95));
      person.crouch = lerp(0, cz, reach) * (1 - lift);
      const pose = { p: start.p.clone().lerp(carryPose(name).p, lift).add(V(0, 0, 0.08 * Math.sin(Math.PI * lift))), q: start.q.clone().slerp(carryPose(name).q, lift) };
      setItemPose(name, pose);
      const gp = graspPoints(name, pose);
      person.hand.r = gp.r; person.hand.l = gp.l;
      person.handW.r = gp.r ? Math.max(reach, lift) : 0; person.handW.l = gp.l ? Math.max(reach, lift) : 0;
      if (k >= 1) { person.crouch = 0; person.look = null; }
    });
  }
  if (verb === 'place' || verb === 'put_back') {
    const name = S.holding;
    const d = ns.items[name];
    let target;
    if (d.loc === 'case') { const tmp = d.place; target = itemInCase(name, tmp); } else target = itemHome(name);
    const start = carryPose(name);
    const cz = crouchFor(target.p.z);
    const label = verb === 'place' ? `Packing the ${W.items[name].label}` : `Putting the ${W.items[name].label} back`;
    return play(label, 1.6, k => {
      turnTo(target.p, k * 6);
      person.look = target.p;
      const move = ease(seg(k, 0.15, 0.75)), back = ease(seg(k, 0.8, 1));
      person.crouch = cz * Math.sin(Math.PI * Math.min(1, move * 0.5 + back * 0.5)) * (move > 0 ? 1 : 0);
      const cp = carryPose(name);
      const p = cp.p.clone().lerp(target.p, move).add(V(0, 0, 0.12 * Math.sin(Math.PI * move)));
      const pose = { p, q: cp.q.clone().slerp(target.q, move) };
      if (k < 0.8) setItemPose(name, pose); else setItemPose(name, target);
      const gp = graspPoints(name, k < 0.8 ? pose : target);
      person.hand.r = gp.r; person.hand.l = gp.l;
      person.handW.r = gp.r ? 1 - back : 0; person.handW.l = gp.l ? 1 - back : 0;
      if (k >= 1) { person.crouch = 0; person.look = null; }
    });
  }
  if (id === 'close_lid' || id === 'open_lid') {
    const a0 = CASE.lidAngle, a1 = id === 'close_lid' ? Math.PI : 0;
    return play(id === 'close_lid' ? 'Closing the lid' : 'Opening the lid', 1.9, k => {
      const m = ease(seg(k, 0.25, 0.85));
      CASE.lidAngle = lerp(a0, a1, m); updateCaseParts();
      const edge = dy => CASE.pivot.localToWorld(V(-CASE.w * 0.92, dy, -CASE.h + CASE.DL + 0.01));
      const reach = ease(seg(k, 0, 0.25)) * (1 - ease(seg(k, 0.88, 1)));
      const er = edge(-0.16), el = edge(0.16);
      turnTo(CASE.root.position, k * 5);
      person.look = er;
      person.hand.r = er; person.hand.l = el; person.handW.r = person.handW.l = reach;
      person.crouch = 0.35 * reach;
      if (k >= 1) { person.crouch = 0; person.look = null; }
    });
  }
  if (verb === 'zip_to') {
    const z0 = CASE.zip, z1 = ns.zip_open;
    return play(z1 < z0 ? 'Zipping the case shut' : 'Unzipping', 2.2, k => {
      const m = ease(seg(k, 0.2, 0.85));
      CASE.zip = lerp(z0, z1, m); updateCaseParts();
      const reach = ease(seg(k, 0, 0.2)) * (1 - ease(seg(k, 0.88, 1)));
      const a = CASE.sliders[0].getWorldPosition(V()), b = CASE.sliders[1].getWorldPosition(V());
      turnTo(a.clone().lerp(b, 0.5), k * 5); person.look = a.clone().lerp(b, 0.5);
      const { l } = person.frame();
      const aLeft = a.clone().sub(V(person.x, person.y, 0)).dot(l) > b.clone().sub(V(person.x, person.y, 0)).dot(l);
      person.hand.l = aLeft ? a : b; person.hand.r = aLeft ? b : a; person.handW.r = person.handW.l = reach;
      person.crouch = 0.45 * reach;
      if (k >= 1) { person.crouch = 0; person.look = null; }
    });
  }
  if (id === 'dock_zip_pulls' || verb === 'turn_dial' || id === 'press_lock_button') {
    const k0 = verb === 'turn_dial' ? Number(id.match(/dial_(\d)/)[1]) - 1 : -1;
    const d0 = CASE.dialsV.slice(), d1 = ns.dials;
    const label = id === 'dock_zip_pulls' ? 'Pushing the zip pulls into the lock' : verb === 'turn_dial' ? `Turning dial ${k0 + 1}` : 'Pressing the lock button';
    return play(label, 1.0, k => {
      const reach = ease(seg(k, 0, 0.35)) * (1 - ease(seg(k, 0.75, 1)));
      const p = k0 >= 0 ? CASE.dials[k0].getWorldPosition(V()) : CASE.button.getWorldPosition(V());
      if (id === 'dock_zip_pulls' && k > 0.45) { CASE.docked = true; updateCaseParts(); }
      if (k0 >= 0 && k > 0.5) { CASE.dialsV = d1.slice(); updateCaseParts(); }
      turnTo(p, k * 5); person.look = p;
      person.hand.r = p; person.handW.r = reach; person.handW.l = 0;
      person.crouch = (CASE.upright ? 0.55 : 0.4) * reach;
      if (k >= 1) { person.crouch = 0; person.look = null; }
    });
  }
  if (verb === 'set_handle') {
    const e0 = CASE.trolley, e1 = [0, 0.2, 0.38][ns.trolley] || 0;
    return play(e1 > e0 ? 'Pulling out the handle' : 'Pushing the handle in', 1.2, k => {
      const reach = ease(seg(k, 0, 0.3)) * (1 - ease(seg(k, 0.85, 1)));
      CASE.trolley = lerp(e0, e1, ease(seg(k, 0.3, 0.8))); updateCaseParts();
      const p = CASE.bar.getWorldPosition(V());
      turnTo(p, k * 5); person.look = p;
      person.hand.r = p; person.handW.r = reach; person.handW.l = 0;
      person.crouch = 0.3 * reach;
      if (k >= 1) { person.crouch = 0; person.look = null; }
    });
  }
  if (id === 'stand_up_case' || verb === 'roll_case') {
    const p0 = CASE.root.position.clone(), q0 = CASE.root.quaternion.clone();
    const tgt = uprightPose(ns.case_at);
    return play(id === 'stand_up_case' ? 'Standing the case up' : 'Rolling the case', 2.2, k => {
      const reach = ease(seg(k, 0, 0.2)) * (1 - ease(seg(k, 0.85, 1)));
      const m = ease(seg(k, 0.2, 0.85));
      CASE.root.position.copy(p0.clone().lerp(tgt.p, m).add(V(0, 0, (id === 'stand_up_case' ? 0.25 : 0) * Math.sin(Math.PI * m))));
      CASE.root.quaternion.copy(q0.clone().slerp(tgt.q, m));
      const p = CASE.bar.getWorldPosition(V());
      turnTo(p, k * 5); person.look = p;
      person.hand.r = p; person.hand.l = p.clone().add(V(0, 0, -0.05)); person.handW.r = person.handW.l = reach;
      person.crouch = 0.3 * reach;
      if (k >= 1) { person.crouch = 0; person.look = null; CASE.upright = true; }
    });
  }
  return play(q.text, 0.6, () => {});
}

// ------------------------------------------------------------------ talking
async function say(text) {
  text = text.trim();
  if (!text) return;
  if (!py.ready || !started) { toast('The simulator is still loading.'); return; }
  try {
    const r = await call('game_say', { text, t: gameTime() });
    renderConv(r.conv);
    if (!r.said.length) toast('The assistant slot is empty: no reply.');
  } catch (e) { toast(String(e.message || e), true); }
}

// ------------------------------------------------------------------ flow
let toastTimer = 0;
function toast(msg, bad = false) {
  const t = $('toast'); t.textContent = msg; t.className = 'panel' + (bad ? ' bad' : ''); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 3200);
}
function pyStatus() {
  const st = $('loadStatus');
  if (py.error) { st.textContent = 'The simulator could not load: ' + py.error; st.className = 'status err'; }
  else if (!loaded3d) st.textContent = 'Loading the apartment…';
  else if (!py.ready) st.textContent = 'Apartment ready. Simulator: ' + pyProgress;
  else st.textContent = 'Ready.';
  renderPrompt();
}
async function newGame() {
  finished = false; started = false; busy = false; anim = null;
  person.x = 1.55; person.y = 0.64; person.heading = Math.PI / 2; person.crouch = 0; person.handW.r = person.handW.l = 0;
  cam.yaw = 0.9; cam.snap = true; path.length = 0;
  $('finish').hidden = true;
  if (!py.ready) { renderChecklist(null); return; }
  const r = await call('game_new', { task });
  applyState(r.state);
  optKey = ''; opts = { options: [], blocked: [] };
  if ($('intro').hidden) { started = true; startT = now(); }
  wantOptions(true);
}
function onPythonReady() { newGame(); }
function finish() {
  finished = true; finishT = now() - startT;
  const cl = S.checklist;
  $('finTitle').textContent = 'Packed and ready.';
  $('finText').textContent = `Everything on the ${W.tasks[task].title.toLowerCase()} checklist is done: every rule held, the case is shut${W.tasks[task].lock ? ', locked' : ''} and standing on its wheels.`;
  $('finTime').textContent = fmtTime(finishT);
  call('game_record', {}).then(rec => { $('finActs').textContent = rec.actions.filter(a => !a.walked).length; });
  $('finGoals').textContent = `${cl.filter(c => c.done).length}/${cl.length}`;
  $('finish').hidden = false;
  $('again').focus();
}
async function download() {
  try {
    const rec = await call('game_record', { path: path.slice() });
    const blob = new Blob([JSON.stringify(rec, null, 1)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = `packing-${task}-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`;
    document.body.appendChild(a); a.click(); a.remove();
  } catch (e) { toast(String(e.message || e), true); }
}

// ------------------------------------------------------------------ input
const keys = new Set();
const typing = () => document.activeElement && (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT');
addEventListener('keydown', e => {
  if (typing()) { if (e.key === 'Escape') { document.activeElement.blur(); canvas.focus(); } return; }
  if (!$('intro').hidden) { if (e.key === 'Enter' && !$('go').disabled) { e.preventDefault(); $('go').click(); } return; }
  const k = e.key.toLowerCase();
  keys.add(k);
  if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' '].includes(k)) e.preventDefault();
  if (e.repeat) return;
  if (k === 'e' || k === ' ' || k === 'enter') doOption(0);
  else if (/^[1-9]$/.test(k)) doOption(Number(k) === 1 ? 0 : Number(k) - 1);
  else if (k === 'c') cycleCamera();
  else if (k === 't') { e.preventDefault(); $('say').focus(); }
  else if (k === 'h' || k === '?') showHelp();
});
addEventListener('keyup', e => keys.delete(e.key.toLowerCase()));
addEventListener('blur', () => keys.clear());
let drag = null;
canvas.addEventListener('pointerdown', e => { if (e.pointerType === 'mouse') { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); } canvas.focus(); });
canvas.addEventListener('pointermove', e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag = { x: e.clientX, y: e.clientY };
  if (cam.mode === 'first person') { person.heading -= dx * 0.005; cam.egoPitch = clamp(cam.egoPitch - dy * 0.004, -1.2, 1.0); }
  else { cam.yaw -= dx * 0.006; cam.pitch = clamp(cam.pitch + dy * 0.004, 0.12, 1.25); cam.touched = now(); }
});
canvas.addEventListener('pointerup', () => { drag = null; });
canvas.addEventListener('wheel', e => { e.preventDefault(); cam.dist = clamp(cam.dist * Math.exp(e.deltaY * 0.001), 1.0, 5.5); }, { passive: false });
function cycleCamera() {
  cam.mode = MODES[(MODES.indexOf(cam.mode) + 1) % MODES.length]; cam.snap = true;
  $('camBtn').textContent = 'Camera: ' + cam.mode;
}
$('camBtn').onclick = () => { cycleCamera(); canvas.focus(); };
$('opts').addEventListener('click', e => { const b = e.target.closest('[data-i]'); if (b) { doOption(Number(b.dataset.i)); canvas.focus(); } });
$('sayForm').addEventListener('submit', e => { e.preventDefault(); const v = $('say').value; $('say').value = ''; $('say').blur(); canvas.focus(); say(v); });
$('restart').onclick = () => { newGame(); canvas.focus(); };
$('task').onchange = () => { task = $('task').value; newGame(); canvas.focus(); };
$('again').onclick = () => { newGame(); canvas.focus(); };
$('keepLooking').onclick = () => { $('finish').hidden = true; canvas.focus(); };
$('download').onclick = download;
$('helpBtn').onclick = showHelp;
// small screens: the checklist and the assistant fold to their headers
for (const b of document.querySelectorAll('.foldBtn')) b.onclick = () => {
  const p = b.closest('.panel'), open = !p.classList.contains('open');
  p.classList.toggle('open', open); b.setAttribute('aria-expanded', open);
};
function showHelp() { $('intro').hidden = false; $('go').textContent = 'Back to packing'; $('go').disabled = false; $('go').focus(); }
$('go').onclick = () => {
  $('intro').hidden = true; canvas.focus();
  if (py.ready && S && !started && !finished) { started = true; startT = now(); }
};
// touch: a drag pad for walking
const pad = { x: 0, y: 0, on: false };
if (matchMedia('(pointer: coarse)').matches) {
  const el = $('pad'); el.hidden = false;
  const knob = el.querySelector('i');
  const move = e => {
    const r = el.getBoundingClientRect(), dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
    const d = Math.min(1, Math.hypot(dx, dy) / (r.width / 2)), a = Math.atan2(dy, dx);
    pad.x = Math.cos(a) * d; pad.y = Math.sin(a) * d; pad.on = true;
    knob.style.transform = `translate(${pad.x * 36}px,${pad.y * 36}px)`;
  };
  el.addEventListener('pointerdown', e => { el.setPointerCapture(e.pointerId); move(e); });
  el.addEventListener('pointermove', e => { if (pad.on) move(e); });
  el.addEventListener('pointerup', () => { pad.on = false; pad.x = pad.y = 0; knob.style.transform = ''; });
}
// image quality: high (ambient occlusion, sharp shadows), medium, low
const QUALITY = ['high', 'medium', 'low'];
let quality = 'high';
function setQuality(q) {
  quality = q; $('qualBtn').textContent = 'Quality: ' + q;
  gtao.enabled = q === 'high';
  renderer.setPixelRatio(q === 'low' ? 1 : Math.min(devicePixelRatio, q === 'high' ? 2 : 1.25));
  renderer.shadowMap.enabled = q !== 'low';
  sun.shadow.mapSize.set(q === 'high' ? 4096 : 2048, q === 'high' ? 4096 : 2048);
  if (sun.shadow.map) { sun.shadow.map.dispose(); sun.shadow.map = null; }
  scene.traverse(o => { if (o.material) [].concat(o.material).forEach(m => { m.needsUpdate = true; }); });
  resize();
}
$('qualBtn').onclick = () => { setQuality(QUALITY[(QUALITY.indexOf(quality) + 1) % 3]); canvas.focus(); };
function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  composer.setSize(innerWidth, innerHeight);
  composer.setPixelRatio(renderer.getPixelRatio());
}
addEventListener('resize', resize);

// ------------------------------------------------------------------ the loop
let person = null, last = now(), slow = 0, pathAt = 0;
function tick() {
  requestAnimationFrame(tick);
  const t = now(), real = t - last, dt = Math.min(0.05, real); last = t;
  if (!person) return;
  // walking
  let mx = 0, my = 0;
  if (!busy && $('finish').hidden && $('intro').hidden) {
    const fw = camForward(), rt = V(fw.y, -fw.x, 0);
    const k = s => keys.has(s);
    let a = 0, b = 0;
    if (k('w') || k('arrowup')) a += 1; if (k('s') || k('arrowdown')) a -= 1;
    if (cam.mode === 'first person') {
      if (k('a')) b -= 1; if (k('d')) b += 1;
      if (k('arrowleft') || k('q')) person.heading += dt * 2.2; if (k('arrowright') || k('r')) person.heading -= dt * 2.2;
    } else {
      if (k('a') || k('arrowleft')) b -= 1; if (k('d') || k('arrowright')) b += 1;
      if (k('q')) { cam.yaw += dt * 1.8; cam.touched = now(); } if (k('r')) { cam.yaw -= dt * 1.8; cam.touched = now(); }
    }
    if (pad.on) { a -= pad.y; b += pad.x; }
    const v = fw.clone().multiplyScalar(a).add(rt.multiplyScalar(b));
    if (v.lengthSq() > 1e-4) {
      if (v.length() > 1) v.normalize();
      const sp = keys.has('shift') ? 2.1 : 1.3;
      mx = v.x * sp; my = v.y * sp;
      if (cam.mode !== 'first person') person.heading += angDiff(person.heading, Math.atan2(v.y, v.x)) * Math.min(1, dt * 10);
      if (S && !started && py.ready && $('intro').hidden) { started = true; startT = now(); }
    }
  }
  const sp = Math.hypot(mx, my);
  if (sp > 0) { const [nx, ny] = collide(person.x + mx * dt, person.y + my * dt); person.speed = Math.hypot(nx - person.x, ny - person.y) / Math.max(dt, 1e-4); person.x = nx; person.y = ny; }
  else person.speed *= Math.exp(-dt * 12);
  if (!busy) { person.handW.r *= Math.exp(-dt * 10); person.handW.l *= Math.exp(-dt * 10); }
  // the action being animated
  if (anim) {
    anim.t += Math.min(real, 0.25); const k = Math.min(1, anim.t / anim.T); anim.step(k);   // real time, even on slow frames
    if (k >= 1) { const d = anim.done; anim = null; d(); }
  }
  person.update(dt);
  if (S && S.holding && !anim) setItemPose(S.holding, carryPose(S.holding));   // during an action, the action poses it
  if (started && !finished && t - pathAt > 0.25) { pathAt = t; path.push([+gameTime().toFixed(2), +person.x.toFixed(3), +person.y.toFixed(3), +person.heading.toFixed(3)]); }
  updateCamera(dt);
  wantOptions(false);
  // the floating tag over the primary option's target
  const tag = $('tag'), q0 = !busy && opts.options[0];
  const tp = q0 && targetPoint(q0);
  if (tp) {
    const s = tp.clone().project(camera);
    if (s.z < 1 && Math.abs(s.x) < 1.1 && Math.abs(s.y) < 1.1) {
      tag.hidden = false; tag.textContent = 'E · ' + q0.text;
      tag.style.left = `${(s.x + 1) / 2 * innerWidth}px`; tag.style.top = `${(1 - s.y) / 2 * innerHeight - 6}px`;
    } else tag.hidden = true;
  } else tag.hidden = true;
  $('clock').textContent = fmtTime(gameTime());
  $('where').textContent = W ? W.rooms[roomAt(person.x, person.y)].label.replace(/^the /, '') : '';
  // drop to a lighter setting if frames are slow
  if (quality === 'high' && dt > 0.045) { slow += dt; if (slow > 4) { setQuality('medium'); toast('Switched to medium quality to keep it smooth.'); } } else slow = Math.max(0, slow - dt * 0.5);
  if (quality === 'high' || quality === 'medium') composer.render(dt); else renderer.render(scene, camera);
}

// ------------------------------------------------------------------ start
async function main() {
  startPython();
  W = await (await fetch('pack/world3d.json', { cache: 'no-cache' })).json();
  for (const [k, t] of Object.entries(W.tasks)) $('task').insertAdjacentHTML('beforeend', `<option value="${k}">${esc(t.title)}</option>`);
  const want = new URLSearchParams(location.search).get('task');
  if (want && W.tasks[want]) task = want;
  $('task').value = task;
  makeMaterials();
  buildLights();
  buildApartment();
  buildFurniture();
  buildItems();
  buildCase();
  person = new Person();
  setCaseVisual({ lid: 'open', zip_open: 1, trolley: 0, dials: [0, 0, 0], pulls: 'free', case_pose: 'on_bed', case_at: 'bed' });
  for (const n of Object.keys(items)) setItemPose(n, itemHome(n));
  renderChecklist(null); renderConv([]);
  manager.onLoad = () => {
    loaded3d = true; $('loadBar').style.width = '100%'; pyStatus();
    $('go').disabled = false; $('go').textContent = py.ready ? 'Start packing' : 'Start walking';
  };
  resize();
  cam.snap = true;
  tick();
}
window.PACK = {                                          // for tests and the console
  get state() { return S; }, get options() { return opts; }, get person() { return person; }, get busy() { return busy; },
  get ready() { return py.ready && !!S; }, get loaded() { return loaded3d; }, get finished() { return finished; },
  teleport(x, y, heading) { person.x = x; person.y = y; person.heading = heading; optKey = ''; wantOptions(true); },
  act: doOption, call, setQuality, get cam() { return cam; },
  pick(px, py) {                                         // what is under a pixel (debugging)
    const r = new THREE.Raycaster(); r.setFromCamera(new THREE.Vector2(px / innerWidth * 2 - 1, 1 - py / innerHeight * 2), camera);
    return r.intersectObjects(scene.children, true).slice(0, 4).map(h => `${h.object.name || h.object.parent?.name || h.object.type} ${h.distance.toFixed(2)} ${h.point.toArray().map(v => v.toFixed(2))}`);
  },
  get camPos() { return camera.position.toArray(); },
};
main().catch(e => { console.error(e); $('loadStatus').textContent = 'Could not start: ' + e.message; $('loadStatus').className = 'status err'; });
