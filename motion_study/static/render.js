// render.js - keyframe selection and the four output renderers.
//
// Everything is drawn from vector data (outlines + keypoints) in source-pixel
// coordinates, through a "pen" that is either a <canvas> (PNG/JPEG/WebP) or an
// SVG string builder. The same drawing code produces the preview, the raster
// exports at any scale, and the vector SVG.

import { largestFilled, traceOuter, resampleClosed, smoothClosed } from './contour.js';

export const FONT = 'Helvetica, Arial, sans-serif';
export const THEMES = {
  white: { bg: '#ffffff', ink: '#000000' },
  black: { bg: '#000000', ink: '#ffffff' },
};

// COCO-17 keypoints
const NOSE = 0, LSH = 5, RSH = 6, LEL = 7, REL = 8, LWR = 9, RWR = 10,
  LHIP = 11, RHIP = 12, LKN = 13, RKN = 14, LAN = 15, RAN = 16;
const LIMBS = [[LSH, LEL], [LEL, LWR], [RSH, REL], [REL, RWR], [LHIP, LKN], [LKN, LAN],
  [RHIP, RKN], [RKN, RAN], [LSH, RSH], [LHIP, RHIP]];
const KP_CONF = 0.35;

// Trajectory joints: key -> [label, legend name, point getter, dash pattern (in line widths)]
export const TRAJ = {
  head: ['t', 'head centre', (k) => headOf(k)?.c, []],
  shoulder: ['s', 'shoulder centre', (k) => mid(k, LSH, RSH), [6, 3]],
  lwrist: ['mL', 'left wrist', (k) => pt(k, LWR), [2, 3]],
  rwrist: ['mR', 'right wrist', (k) => pt(k, RWR), [2, 3]],
  hip: ['a', 'hip centre', (k) => mid(k, LHIP, RHIP), [10, 3, 2, 3]],
  lknee: ['gL', 'left knee', (k) => pt(k, LKN), [5, 4]],
  rknee: ['gR', 'right knee', (k) => pt(k, RKN), [5, 4]],
  lankle: ['pL', 'left ankle', (k) => pt(k, LAN), []],
  rankle: ['pR', 'right ankle', (k) => pt(k, RAN), []],
};

function pt(kp, i) { return kp && kp[i] && kp[i][2] >= KP_CONF ? [kp[i][0], kp[i][1]] : null; }
function mid(kp, a, b) {
  const p = pt(kp, a), q = pt(kp, b);
  return p && q ? [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2] : (p || q);
}
function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
function headOf(kp) {
  if (!kp) return null;
  const face = [0, 1, 2, 3, 4].map((i) => pt(kp, i)).filter(Boolean);
  if (!face.length) return null;
  const c = [face.reduce((s, p) => s + p[0], 0) / face.length, face.reduce((s, p) => s + p[1], 0) / face.length];
  const le = pt(kp, 3), re = pt(kp, 4), neck = mid(kp, LSH, RSH);
  const ls = pt(kp, LSH), rs = pt(kp, RSH), sw = ls && rs ? dist(ls, rs) : 0;
  let r = le && re ? dist(le, re) * 0.62 : 0;
  if (!r && neck) r = dist(c, neck) * 0.4;
  if (!r) r = sw * 0.3 || 12;
  if (sw) r = Math.min(Math.max(r, sw * 0.18), sw * 0.38); // keep the head in proportion to the shoulders
  return { c, r };
}

// --------------------------------------------------------------------------- //
// Seeded randomness and the hand-drawn stroke generator (port of motion_sketch.sketchify)
// --------------------------------------------------------------------------- //
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rng) {
  let u = 0; while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}
function smoothNoise(rng, n, sigma) {
  sigma = Math.max(1, sigma);
  const r = Math.floor(3 * sigma), m = n + 2 * r;
  const k = []; let ks = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-0.5 * (i / sigma) ** 2); k.push(v); ks += v; }
  const out = [];
  for (let ax = 0; ax < 2; ax++) {
    const raw = new Float64Array(m);
    for (let i = 0; i < m; i++) raw[i] = gauss(rng);
    const sm = new Float64Array(n);
    let mean = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < k.length; j++) s += raw[i + j] * k[j];
      sm[i] = s / ks; mean += sm[i];
    }
    mean /= n;
    let v = 0; for (let i = 0; i < n; i++) v += (sm[i] - mean) ** 2;
    const sd = Math.sqrt(v / n) || 1;
    for (let i = 0; i < n; i++) sm[i] = (sm[i] - mean) / sd;
    out.push(sm);
  }
  return out;
}
function unit(dx, dy) { const n = Math.hypot(dx, dy); return n > 1e-6 ? [dx / n, dy / n] : [0, 0]; }

export function sketchify(points, seed, passes, wobble, overshoot) {
  const rng = mulberry32(seed);
  const n = points.length;
  let per = 0;
  for (let i = 0; i < n; i++) per += dist(points[i], points[(i + 1) % n]);
  const spacing = per / n || 1;
  const sigmaPts = Math.max(10, per / 80) / spacing;
  const strokes = [];
  for (let p = 0; p < passes; p++) {
    const start = Math.floor(rng() * n);
    const extra = overshoot > 0 ? Math.round(overshoot * (1 + 1.5 * rng()) / spacing) : 0;
    let path = [];
    for (let i = 0; i < n + extra; i++) path.push(points[(start + i) % n]);
    if (overshoot > 0 && path.length > 3) {
      const hd = unit(path[0][0] - path[2][0], path[0][1] - path[2][1]);
      const L = path.length, td = unit(path[L - 1][0] - path[L - 3][0], path[L - 1][1] - path[L - 3][1]);
      const a = overshoot * (0.2 + 0.6 * rng()), b = overshoot * (0.5 + 0.7 * rng());
      path = [[path[0][0] + hd[0] * a, path[0][1] + hd[1] * a], ...path,
        [path[L - 1][0] + td[0] * b, path[L - 1][1] + td[1] * b]];
    }
    const amp = wobble * (p === 0 ? 0.6 : 0.9);
    const [fx, fy] = smoothNoise(rng, path.length, sigmaPts);
    const [sx, sy] = smoothNoise(rng, path.length, sigmaPts * 4);
    const dx = gauss(rng) * wobble * 0.5, dy = gauss(rng) * wobble * 0.5;
    strokes.push(path.map((q, i) => [q[0] + (fx[i] + 0.7 * sx[i]) * amp + dx, q[1] + (fy[i] + 0.7 * sy[i]) * amp + dy]));
  }
  return strokes;
}

// --------------------------------------------------------------------------- //
// Silhouette rasters for the shape-change metric
// --------------------------------------------------------------------------- //
export class MaskCache {
  constructor() { this.map = new Map(); this.key = ''; this.cv = document.createElement('canvas'); }
  prepare(valid) {
    const b = unionBox(valid);
    const key = b.map((v) => Math.round(v)).join(',');
    if (key !== this.key) {
      this.key = key; this.map.clear();
      const w = b[2] - b[0], h = b[3] - b[1];
      this.k = 160 / Math.max(w, h, 1);
      this.ox = b[0]; this.oy = b[1];
      this.W = Math.ceil(w * this.k) + 2; this.H = Math.ceil(h * this.k) + 2;
      this.cv.width = this.W; this.cv.height = this.H;
      this.ctx = this.cv.getContext('2d', { willReadFrequently: true });
    }
  }
  get(f) {
    let m = this.map.get(f.i);
    if (m) return m;
    const { ctx, W, H, k, ox, oy } = this;
    ctx.clearRect(0, 0, W, H);
    ctx.beginPath();
    f.c.forEach((p, j) => { const x = (p[0] - ox) * k + 1, y = (p[1] - oy) * k + 1; j ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath(); ctx.fillStyle = '#000'; ctx.fill();
    const d = ctx.getImageData(0, 0, W, H).data;
    const bits = new Uint8Array(W * H); let cnt = 0;
    for (let i = 0; i < bits.length; i++) if (d[i * 4 + 3] > 127) { bits[i] = 1; cnt++; }
    m = { bits, cnt };
    this.map.set(f.i, m);
    return m;
  }
}
// max(pixels gained, pixels lost): equals the area change when the silhouette only grows or
// shrinks, and the fraction of the body that moved when it only shifts or changes pose.
function changedCount(cur, prev) {
  let gain = 0, loss = 0; const x = cur.bits, y = prev.bits;
  for (let i = 0; i < x.length; i++) { if (x[i] > y[i]) gain++; else if (x[i] < y[i]) loss++; }
  return Math.max(gain, loss);
}

// --------------------------------------------------------------------------- //
// Step 4: keyframe selection
// --------------------------------------------------------------------------- //
// metric 'shape': change = max(|M_f \ M_k|, |M_k \ M_f|) / area(M_k)  (sees both area change
//                 and motion, relative to the preceding keyframe)
// metric 'area' : change = |A_f - A_k| / A_k          (moving-average-smoothed area only)
export function selectKeyframes(frames, s, masks) {
  const valid = frames.filter((f) => f.c && f.c.length > 2);
  const series = [];
  if (!valid.length) return { valid, keys: [], series };
  const h = Math.max(0, Math.floor((s.smooth - 1) / 2));
  const areaS = valid.map((_, k) => {
    let sum = 0, n = 0;
    for (let j = Math.max(0, k - h); j <= Math.min(valid.length - 1, k + h); j++) { sum += valid[j].area; n++; }
    return sum / n;
  });
  if (s.metric === 'shape') masks.prepare(valid);
  const thr = s.threshold / 100;
  const keys = [0];
  let last = 0;
  series.push({ f: valid[0], d: 0, key: true, area: areaS[0] });
  for (let k = 1; k < valid.length; k++) {
    let d;
    if (s.metric === 'shape') {
      const a = masks.get(valid[k]), b = masks.get(valid[last]);
      d = changedCount(a, b) / Math.max(1, b.cnt);
    } else {
      d = Math.abs(areaS[k] - areaS[last]) / Math.max(1, areaS[last]);
    }
    const isKey = d >= thr && valid[k].t - valid[last].t >= s.minGap - 1e-9;
    series.push({ f: valid[k], d, key: isKey, area: areaS[k] });
    if (isKey) { keys.push(k); last = k; }
  }
  if (s.includeLast && valid.length > 1 && last !== valid.length - 1) {
    keys.push(valid.length - 1); series[series.length - 1].key = true; series[series.length - 1].forced = true;
  }
  const kf = keys.map((k, j) => ({ ...valid[k], n: j + 1, d: series[k].d, dt: j ? valid[k].t - valid[keys[j - 1]].t : 0 }));
  return { valid, keys: kf, series };
}

// --------------------------------------------------------------------------- //
// Pens
// --------------------------------------------------------------------------- //
export class CanvasPen {
  constructor(canvas, W, H, scale) {
    canvas.width = Math.max(1, Math.round(W * scale));
    canvas.height = Math.max(1, Math.round(H * scale));
    const ctx = (this.ctx = canvas.getContext('2d'));
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  }
  rect(x, y, w, h, o) { const c = this.ctx; c.globalAlpha = o.alpha ?? 1; c.fillStyle = o.fill; c.fillRect(x, y, w, h); }
  save() { this.ctx.save(); }
  restore() { this.ctx.restore(); }
  translate(x, y) { this.ctx.translate(x, y); }
  clip(x, y, w, h) { const c = this.ctx; c.beginPath(); c.rect(x, y, w, h); c.clip(); }
  _paint(o) {
    const c = this.ctx;
    if (o.fill) { c.globalAlpha = o.fillAlpha ?? 1; c.fillStyle = o.fill; c.fill(o.evenodd ? 'evenodd' : 'nonzero'); }
    if (o.stroke) {
      c.globalAlpha = o.alpha ?? 1; c.strokeStyle = o.stroke; c.lineWidth = o.width ?? 1;
      c.setLineDash(o.dash ? o.dash.map((v) => v * (o.width ?? 1)) : []);
      c.stroke(); c.setLineDash([]);
    }
  }
  path(pts, o) {
    if (!pts || pts.length < 2) return;
    const c = this.ctx; c.beginPath(); c.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) c.lineTo(pts[i][0], pts[i][1]);
    if (o.closed) c.closePath();
    this._paint(o);
  }
  circle(x, y, r, o) { const c = this.ctx; c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); this._paint(o); }
  text(str, x, y, o) {
    const c = this.ctx; c.globalAlpha = o.alpha ?? 1; c.fillStyle = o.color;
    c.font = `${o.weight || 400} ${o.size}px ${FONT}`; c.textAlign = o.align || 'left';
    c.textBaseline = o.baseline || 'alphabetic'; c.fillText(str, x, y);
  }
  measure(str, size, weight = 400) { this.ctx.font = `${weight} ${size}px ${FONT}`; return this.ctx.measureText(str).width; }
}

const f1 = (v) => Math.round(v * 10) / 10;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
let _measureCtx = null;

export class SvgPen {
  constructor(W, H, scale = 1) {
    this.W = W; this.H = H; this.scale = scale; this.p = []; this.open = 0; this.stack = []; this.cid = 0;
    this.prefix = 'k' + Math.random().toString(36).slice(2, 7); // unique clip ids when several SVGs share a page
  }
  rect(x, y, w, h, o) { this.p.push(`<rect x="${f1(x)}" y="${f1(y)}" width="${f1(w)}" height="${f1(h)}" fill="${o.fill}"${o.alpha != null && o.alpha < 1 ? ` fill-opacity="${o.alpha}"` : ''}/>`); }
  save() { this.stack.push(this.open); }
  restore() { const d = this.stack.pop(); while (this.open > d) { this.p.push('</g>'); this.open--; } }
  translate(x, y) { this.p.push(`<g transform="translate(${f1(x)} ${f1(y)})">`); this.open++; }
  clip(x, y, w, h) {
    const id = `${this.prefix}c${this.cid++}`;
    this.p.push(`<clipPath id="${id}"><rect x="${f1(x)}" y="${f1(y)}" width="${f1(w)}" height="${f1(h)}"/></clipPath><g clip-path="url(#${id})">`);
    this.open++;
  }
  _attrs(o) {
    let a = '';
    a += o.fill ? ` fill="${o.fill}"${o.fillAlpha != null && o.fillAlpha < 1 ? ` fill-opacity="${+o.fillAlpha.toFixed(4)}"` : ''}${o.evenodd ? ' fill-rule="evenodd"' : ''}` : ' fill="none"';
    if (o.stroke) {
      a += ` stroke="${o.stroke}" stroke-width="${+(o.width ?? 1).toFixed(3)}" stroke-linecap="round" stroke-linejoin="round"`;
      if (o.alpha != null && o.alpha < 1) a += ` stroke-opacity="${+o.alpha.toFixed(4)}"`;
      if (o.dash && o.dash.length) a += ` stroke-dasharray="${o.dash.map((v) => f1(v * (o.width ?? 1))).join(' ')}"`;
    }
    return a;
  }
  path(pts, o) {
    if (!pts || pts.length < 2) return;
    let d = `M${f1(pts[0][0])} ${f1(pts[0][1])}`;
    for (let i = 1; i < pts.length; i++) d += `L${f1(pts[i][0])} ${f1(pts[i][1])}`;
    if (o.closed) d += 'Z';
    this.p.push(`<path d="${d}"${this._attrs(o)}/>`);
  }
  circle(x, y, r, o) { this.p.push(`<circle cx="${f1(x)}" cy="${f1(y)}" r="${f1(r)}"${this._attrs(o)}/>`); }
  text(str, x, y, o) {
    const anchor = { left: 'start', center: 'middle', right: 'end' }[o.align || 'left'];
    const base = { top: 'hanging', middle: 'middle', alphabetic: 'alphabetic', bottom: 'text-after-edge' }[o.baseline || 'alphabetic'];
    this.p.push(`<text x="${f1(x)}" y="${f1(y)}" font-family="${FONT}" font-size="${f1(o.size)}" font-weight="${o.weight || 400}" fill="${o.color}"${o.alpha != null && o.alpha < 1 ? ` fill-opacity="${o.alpha}"` : ''} text-anchor="${anchor}" dominant-baseline="${base}">${esc(str)}</text>`);
  }
  measure(str, size, weight = 400) {
    _measureCtx = _measureCtx || document.createElement('canvas').getContext('2d');
    _measureCtx.font = `${weight} ${size}px ${FONT}`; return _measureCtx.measureText(str).width;
  }
  toString() {
    let close = ''; for (let i = 0; i < this.open; i++) close += '</g>';
    return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${Math.round(this.W * this.scale)}" height="${Math.round(this.H * this.scale)}" viewBox="0 0 ${f1(this.W)} ${f1(this.H)}">\n${this.p.join('\n')}${close}\n</svg>\n`;
  }
}

// --------------------------------------------------------------------------- //
// Geometry helpers
// --------------------------------------------------------------------------- //
function grow(b, x, y) { if (x < b[0]) b[0] = x; if (y < b[1]) b[1] = y; if (x > b[2]) b[2] = x; if (y > b[3]) b[3] = y; }
function frameBox(f) {
  if (f._b) return f._b;
  const b = [Infinity, Infinity, -Infinity, -Infinity];
  for (const p of f.c) grow(b, p[0], p[1]);
  if (f.box) { grow(b, f.box[0], f.box[1]); grow(b, f.box[2], f.box[3]); }
  if (f.kp) for (const k of f.kp) if (k[2] >= KP_CONF) grow(b, k[0], k[1]);
  for (const h of f.h || []) for (const p of h.p) grow(b, p[0], p[1]);
  Object.defineProperty(f, '_b', { value: b, enumerable: false, writable: true });
  return b;
}
function unionOf(boxes) {
  const b = [Infinity, Infinity, -Infinity, -Infinity];
  for (const q of boxes) if (q) { grow(b, q[0], q[1]); grow(b, q[2], q[3]); }
  return b;
}
export function unionBox(list) { return unionOf(list.map(frameBox)); }
function padBox(b, m) { const p = m * Math.max(b[2] - b[0], b[3] - b[1]); return [b[0] - p, b[1] - p, b[2] + p, b[3] + p]; }
function median(a) { if (!a.length) return 0; const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; }
export function fmtT(t) { return t.toFixed(t < 100 ? 2 : 1); }

// --------------------------------------------------------------------------- //
// Hands (21 MediaPipe hand landmarks per hand, in source pixels)
// --------------------------------------------------------------------------- //
const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const HAND_BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
export const SIDES = { L: 'left', R: 'right' };
export function handOf(f, side) { return (f.h || []).find((h) => h.side === side) || null; }
function palmWidth(p) { return Math.max(dist(p[5], p[17]), dist(p[0], p[9]) * 0.7, 1); }
function boxOfPts(pts) { const b = [Infinity, Infinity, -Infinity, -Infinity]; for (const p of pts) grow(b, p[0], p[1]); return b; }

// Simple hand shape (as in a figure drawing): palm + tapered finger capsules + a wrist stub.
// X maps source px -> raster px, k is raster px per source px.
function drawHandShape(ctx, p, X, k) {
  const pw = palmWidth(p);
  ctx.fillStyle = ctx.strokeStyle = '#000'; ctx.lineCap = ctx.lineJoin = 'round';
  ctx.beginPath(); [0, 1, 5, 9, 13, 17].forEach((i, j) => { const [x, y] = X(p[i]); j ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.closePath(); ctx.fill(); ctx.lineWidth = pw * 0.3 * k; ctx.stroke();
  const back = [p[0][0] + (p[0][0] - p[9][0]) * 0.3, p[0][1] + (p[0][1] - p[9][1]) * 0.3];
  ctx.lineWidth = pw * 0.75 * k; ctx.beginPath(); ctx.moveTo(...X(p[0])); ctx.lineTo(...X(back)); ctx.stroke();
  FINGERS.forEach((chain, fi) => {
    const base = (fi === 0 ? 0.3 : fi === 4 ? 0.2 : 0.24) * pw * k;
    const pts = fi === 0 ? [p[0], ...chain.map((i) => p[i])] : chain.map((i) => p[i]);
    for (let s = 1; s < pts.length; s++) {
      ctx.lineWidth = base * (1 - 0.12 * (s - 1));
      ctx.beginPath(); ctx.moveTo(...X(pts[s - 1])); ctx.lineTo(...X(pts[s])); ctx.stroke();
    }
  });
}

// Raster helpers: one reusable canvas; alpha > 127 = inside.
let _rc = null;
function raster(x0, y0, W, H, k) {
  const cw = Math.ceil(W * k) + 2, ch = Math.ceil(H * k) + 2;
  _rc = _rc || document.createElement('canvas');
  _rc.width = cw; _rc.height = ch;
  const ctx = _rc.getContext('2d', { willReadFrequently: true });
  ctx.clearRect(0, 0, cw, ch);
  const X = (q) => [(q[0] - x0) * k + 1, (q[1] - y0) * k + 1];
  const read = () => {
    const d = ctx.getImageData(0, 0, cw, ch).data, bin = new Uint8Array(cw * ch);
    for (let i = 0; i < bin.length; i++) bin[i] = d[i * 4 + 3] > 127 ? 1 : 0;
    return bin;
  };
  const trace = (bin, spacing, smooth) => {
    const lf = largestFilled(bin, cw, ch);
    if (!lf) return null;
    const raw = traceOuter(lf.bin, cw, ch).map(([x, y]) => [(x - 0.5) / k + x0, (y - 0.5) / k + y0]);
    return raw.length > 2 ? smoothClosed(resampleClosed(raw, spacing), smooth, spacing) : null;
  };
  const at = (bin, q) => { const x = Math.round((q[0] - x0) * k + 1), y = Math.round((q[1] - y0) * k + 1); return x >= 0 && y >= 0 && x < cw && y < ch && bin[y * cw + x] === 1; };
  const poly = (pts) => { ctx.beginPath(); pts.forEach((q, j) => { const [x, y] = X(q); j ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.closePath(); };
  return { ctx, X, read, trace, at, poly, cw, ch };
}

function handOutline(p) {
  const pw = palmWidth(p), b = boxOfPts(p), pad = pw * 0.6;
  const R = raster(b[0] - pad, b[1] - pad, b[2] - b[0] + 2 * pad, b[3] - b[1] + 2 * pad, 200 / (Math.max(b[2] - b[0], b[3] - b[1]) + 2 * pad));
  drawHandShape(R.ctx, p, R.X, 200 / (Math.max(b[2] - b[0], b[3] - b[1]) + 2 * pad));
  return R.trace(R.read(), pw / 14, pw / 25);
}

// Limb chains with widths as fractions of torso length (shoulder-centre to hip-centre).
const LIMB_CHAINS = [
  [[LSH, LEL, LWR], [0.25, 0.19, 0.13]], [[RSH, REL, RWR], [0.25, 0.19, 0.13]],
  [[LHIP, LKN, LAN], [0.36, 0.25, 0.14]], [[RHIP, RKN, RAN], [0.36, 0.25, 0.14]],
];

// The pose outline of one frame:
//   outer - the tracked silhouette, with the landmark hand shapes merged in (fingers become part of it)
//   inner - limb and hand contours that lie inside the silhouette (an arm across the body, folded
//           legs, a hand in front of the torso), kept away from the outer edge so lines don't double.
function poseParts(f, s) {
  const hands = s.handDetail ? (f.h || []) : [];
  const kp = f.kp, neck = kp && mid(kp, LSH, RSH), hip = kp && mid(kp, LHIP, RHIP);
  const T = neck && hip ? dist(neck, hip) : 0;
  const doInner = s.limbs && kp && T > 0;
  if (!hands.length && !doInner) return { outer: f.c, inner: [] };
  const b = frameBox(f), pad = 0.05 * Math.max(b[2] - b[0], b[3] - b[1]) + 4;
  const W = b[2] - b[0] + 2 * pad, H = b[3] - b[1] + 2 * pad, k = Math.min(1.5, 900 / Math.max(W, H));
  const R = raster(b[0] - pad, b[1] - pad, W, H, k), ctx = R.ctx;
  const spacing = 4 * Math.max(1, 1 / k / 1.5);
  ctx.fillStyle = '#000'; R.poly(f.c); ctx.fill();
  for (const h of hands) drawHandShape(ctx, h.p, R.X, k);
  const outer = hands.length ? (R.trace(R.read(), spacing, spacing) || f.c) : f.c;
  if (!doInner) return { outer, inner: [] };

  // silhouette eroded by dmin: only contour points well inside it are drawn
  const dmin = Math.max(3, 0.04 * T * s.limbWidth);
  ctx.clearRect(0, 0, R.cw, R.ch); ctx.fillStyle = '#000'; R.poly(outer); ctx.fill();
  ctx.globalCompositeOperation = 'destination-out'; ctx.lineWidth = 2 * dmin * k; ctx.lineJoin = 'round'; ctx.stroke();
  ctx.globalCompositeOperation = 'source-over';
  const inside = R.read();
  ctx.clearRect(0, 0, R.cw, R.ch);
  for (const h of hands) drawHandShape(ctx, h.p, R.X, k);
  const inHand = R.read();

  const inner = [];
  const keepRuns = (pts, ok) => {
    let run = [];
    const flush = () => { if (run.length >= 5) inner.push(run); run = []; };
    // start the scan at a point that is not kept, so a run never wraps around the start
    const s0 = Math.max(0, pts.findIndex((q) => !ok(q)));
    for (let j = 0; j < pts.length; j++) { const q = pts[(s0 + j) % pts.length]; if (ok(q)) run.push(q); else flush(); }
    flush();
  };
  for (const [chain, wf] of LIMB_CHAINS) {
    const P = chain.map((i) => pt(kp, i));
    if (!P[0] || !P[1]) continue;
    ctx.clearRect(0, 0, R.cw, R.ch); ctx.fillStyle = '#000';
    for (let j = 1; j < P.length && P[j]; j++) {
      const a = P[j - 1], c = P[j], wa = wf[j - 1] * T * s.limbWidth / 2, wc = wf[j] * T * s.limbWidth / 2;
      const n = unit(-(c[1] - a[1]), c[0] - a[0]);
      R.poly([[a[0] + n[0] * wa, a[1] + n[1] * wa], [c[0] + n[0] * wc, c[1] + n[1] * wc], [c[0] - n[0] * wc, c[1] - n[1] * wc], [a[0] - n[0] * wa, a[1] - n[1] * wa]]);
      ctx.fill();
      for (const [q, r] of [[a, wa], [c, wc]]) { const [x, y] = R.X(q); ctx.beginPath(); ctx.arc(x, y, r * k, 0, Math.PI * 2); ctx.fill(); }
    }
    const o = R.trace(R.read(), 3, 3);
    if (!o) continue;
    const root = P[0], rr = wf[0] * T * s.limbWidth * 0.55;
    keepRuns(o, (q) => dist(q, root) > rr && R.at(inside, q) && !R.at(inHand, q));
  }
  for (const h of hands) {
    const o = handOutline(h.p);
    if (o) keepRuns(o, (q) => R.at(inside, q));
  }
  return { outer, inner };
}
function handFocusBox(f, side) {
  const h = handOf(f, side);
  if (h) { const pw = palmWidth(h.p); return boxOfPts(h.p).map((v, i) => v + (i < 2 ? -0.45 : 0.45) * pw); }
  const kp = f.kp, W = kp && pt(kp, side === 'L' ? LWR : RWR), E = kp && pt(kp, side === 'L' ? LEL : REL);
  if (!W) return null;
  const r = E ? dist(W, E) * 0.55 : 60;
  return [W[0] - r, W[1] - r, W[0] + r, W[1] + r];
}

// --------------------------------------------------------------------------- //
// Joint angles (2D, in the image plane)
// --------------------------------------------------------------------------- //
function angleAt(a, b, c) {
  if (!a || !b || !c) return null;
  const v1 = [a[0] - b[0], a[1] - b[1]], v2 = [c[0] - b[0], c[1] - b[1]];
  const n = Math.hypot(...v1) * Math.hypot(...v2);
  return n ? (Math.acos(Math.max(-1, Math.min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / n))) * 180) / Math.PI : null;
}
function angleBetween(v1, v2) {
  const n = Math.hypot(...v1) * Math.hypot(...v2);
  return n ? (Math.acos(Math.max(-1, Math.min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / n))) * 180) / Math.PI : null;
}
export const ANGLES = [
  { key: 'trunk', label: 'Trunk inclination', short: 'T', help: 'hip-centre → shoulder-centre vs. vertical (0° = upright)' },
  { key: 'neck', label: 'Neck flexion', short: 'N', help: 'head direction vs. trunk line (0° = in line)' },
  { key: 'elbowL', label: 'Left elbow', short: 'E', help: 'shoulder–elbow–wrist (180° = straight)' },
  { key: 'elbowR', label: 'Right elbow', short: 'E', help: 'shoulder–elbow–wrist (180° = straight)' },
  { key: 'wristL', label: 'Left wrist bend', short: 'W', help: 'hand (wrist → middle knuckle) vs. forearm (0° = straight)' },
  { key: 'wristR', label: 'Right wrist bend', short: 'W', help: 'hand (wrist → middle knuckle) vs. forearm (0° = straight)' },
  { key: 'gripL', label: 'Left grip aperture', short: 'G', unit: '×', help: 'thumb tip – index tip distance in palm widths' },
  { key: 'gripR', label: 'Right grip aperture', short: 'G', unit: '×', help: 'thumb tip – index tip distance in palm widths' },
  { key: 'hipL', label: 'Left hip', short: 'H', help: 'shoulder–hip–knee (≈90° when sitting upright)' },
  { key: 'hipR', label: 'Right hip', short: 'H', help: 'shoulder–hip–knee (≈90° when sitting upright)' },
  { key: 'kneeL', label: 'Left knee', short: 'K', help: 'hip–knee–ankle (180° = straight)' },
  { key: 'kneeR', label: 'Right knee', short: 'K', help: 'hip–knee–ankle (180° = straight)' },
];
export function anglesOf(f) {
  if (f._a !== undefined) return f._a;
  let a = null;
  const kp = f.kp;
  if (kp) {
    const P = (i) => pt(kp, i), neck = mid(kp, LSH, RSH), hip = mid(kp, LHIP, RHIP);
    const ears = mid(kp, 3, 4) || P(NOSE);
    a = {
      trunk: neck && hip ? angleBetween([neck[0] - hip[0], neck[1] - hip[1]], [0, -1]) : null,
      neck: neck && hip && ears ? angleBetween([neck[0] - hip[0], neck[1] - hip[1]], [ears[0] - neck[0], ears[1] - neck[1]]) : null,
      elbowL: angleAt(P(LSH), P(LEL), P(LWR)), elbowR: angleAt(P(RSH), P(REL), P(RWR)),
      hipL: angleAt(P(LSH), P(LHIP), P(LKN)), hipR: angleAt(P(RSH), P(RHIP), P(RKN)),
      kneeL: angleAt(P(LHIP), P(LKN), P(LAN)), kneeR: angleAt(P(RHIP), P(RKN), P(RAN)),
    };
    for (const side of ['L', 'R']) {
      const h = handOf(f, side), W = P(side === 'L' ? LWR : RWR), E = P(side === 'L' ? LEL : REL);
      a['wrist' + side] = h && W && E ? angleBetween([W[0] - E[0], W[1] - E[1]], [h.p[9][0] - h.p[0][0], h.p[9][1] - h.p[0][1]]) : null;
      a['grip' + side] = h ? dist(h.p[4], h.p[8]) / palmWidth(h.p) : null;
    }
  }
  Object.defineProperty(f, '_a', { value: a, enumerable: false, writable: true });
  return a;
}
export function fmtAngle(def, v) { return v == null ? '–' : def.unit === '×' ? `${v.toFixed(2)}×` : `${Math.round(v)}°`; }

// --------------------------------------------------------------------------- //
// Open hand-drawn strokes (posture lines, finger lines)
// --------------------------------------------------------------------------- //
function densify(points, spacing) {
  const out = [points[0]];
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1], b = points[i], n = Math.max(1, Math.ceil(dist(a, b) / spacing));
    for (let k = 1; k <= n; k++) out.push([a[0] + ((b[0] - a[0]) * k) / n, a[1] + ((b[1] - a[1]) * k) / n]);
  }
  return out;
}
export function sketchOpen(points, seed, passes, wobble, ext = 0) {
  const rng = mulberry32(seed);
  let len = 0; for (let i = 1; i < points.length; i++) len += dist(points[i - 1], points[i]);
  if (len < 1) return [];
  const spacing = Math.max(0.5, len / 60), path0 = densify(points, spacing);
  const strokes = [];
  for (let p = 0; p < passes; p++) {
    let path = path0;
    if (ext > 0 && path.length > 2) {
      const L = path.length, hd = unit(path[0][0] - path[1][0], path[0][1] - path[1][1]), td = unit(path[L - 1][0] - path[L - 2][0], path[L - 1][1] - path[L - 2][1]);
      const a = ext * rng(), b = ext * (0.3 + rng());
      path = [[path[0][0] + hd[0] * a, path[0][1] + hd[1] * a], ...path, [path[L - 1][0] + td[0] * b, path[L - 1][1] + td[1] * b]];
    }
    const [fx, fy] = smoothNoise(rng, path.length, Math.max(3, path.length / 5));
    const amp = wobble * (p ? 0.9 : 0.6), dx = gauss(rng) * wobble * 0.4, dy = gauss(rng) * wobble * 0.4;
    strokes.push(path.map((q, i) => [q[0] + fx[i] * amp + dx, q[1] + fy[i] * amp + dy]));
  }
  return strokes;
}

// Gesture lines through the body (posture): spine, shoulders, pelvis, arms into the hands, legs.
function postureChains(f) {
  const kp = f.kp; if (!kp) return [];
  const neck = mid(kp, LSH, RSH), hip = mid(kp, LHIP, RHIP), head = headOf(kp);
  const chains = [[head?.c, neck, hip], [pt(kp, LSH), pt(kp, RSH)], [pt(kp, LHIP), pt(kp, RHIP)]];
  for (const [side, sh, el, wr, hp, kn, an] of [['L', LSH, LEL, LWR, LHIP, LKN, LAN], ['R', RSH, REL, RWR, RHIP, RKN, RAN]]) {
    const h = handOf(f, side);
    chains.push([pt(kp, sh), pt(kp, el), pt(kp, wr), h ? h.p[9] : null]);
    chains.push([pt(kp, hp), pt(kp, kn), pt(kp, an)]);
  }
  // split each chain at missing points
  const out = [];
  for (const c of chains) {
    let run = [];
    for (const p of c) { if (p) run.push(p); else { if (run.length > 1) out.push(run); run = []; } }
    if (run.length > 1) out.push(run);
  }
  return out;
}

// --------------------------------------------------------------------------- //
// Model: everything the renderers need, recomputed when data or settings change
// --------------------------------------------------------------------------- //
export class Study {
  constructor() { this.masks = new MaskCache(); this.cache = new Map(); this.frames = []; this.meta = {}; }
  update(frames, meta, s) {
    this.frames = frames; this.meta = meta; this.s = s;
    const sel = selectKeyframes(frames, s, this.masks);
    Object.assign(this, sel);
    this.u = Math.max(0.25, median(this.valid.map((f) => frameBox(f)[3] - frameBox(f)[1])) / 400) || 1;
    this.theme = THEMES[s.bg] || THEMES.white;
    this.span = frames.length ? [frames[0].t, frames[frames.length - 1].t] : [0, 0];
    this.hasHands = this.valid.some((f) => f.h && f.h.length);
    return this;
  }
  _memo(key, make) {
    let v = this.cache.get(key);
    if (v === undefined) { v = make(); if (this.cache.size > 6000) this.cache.clear(); this.cache.set(key, v); }
    return v;
  }
  seed(f, salt) { return (Math.imul(f.i + 1 + salt * 7919, 2654435761) ^ this.s.seed) >>> 0; }
  parts(f) {
    const s = this.s;
    return this._memo(`o|${f.i}|${s.handDetail}|${s.limbs}|${s.limbWidth}`, () => poseParts(f, s));
  }
  outline(f) { return this.parts(f).outer; }
  strokes(f) {
    const s = this.s, u = this.u;
    return this._memo(`b|${f.i}|${s.passes}|${s.wobble}|${s.seed}|${u.toFixed(3)}|${s.handDetail}|${s.limbs}|${s.limbWidth}`,
      () => sketchify(this.outline(f), this.seed(f, 0), s.passes, s.wobble * 2 * u, 6 * u));
  }
  innerStrokes(f) {
    const s = this.s, u = this.u;
    return this._memo(`i|${f.i}|${s.passes}|${s.wobble}|${s.seed}|${u.toFixed(3)}|${s.handDetail}|${s.limbs}|${s.limbWidth}`,
      () => this.parts(f).inner.flatMap((c, j) => sketchOpen(c, this.seed(f, 40 + j), Math.max(1, s.passes - 1), s.wobble * 1.5 * u, 3 * u)));
  }
  posture(f) {
    const s = this.s, u = this.u;
    return this._memo(`p|${f.i}|${s.wobble}|${s.seed}|${u.toFixed(3)}`,
      () => postureChains(f).flatMap((c, j) => sketchOpen(c, this.seed(f, 20 + j), 2, s.wobble * 1.5 * u, 5 * u)));
  }
}

// --------------------------------------------------------------------------- //
// Drawing primitives for one pose
// --------------------------------------------------------------------------- //
function drawSketch(pen, st, f, alpha, opts = {}) {
  const ink = st.theme.ink, s = st.s, lw = s.lineWidth * st.u * (opts.weight ?? 1);
  if (opts.tone) pen.path(st.outline(f), { fill: ink, fillAlpha: opts.tone, closed: true });
  st.strokes(f).forEach((q, i) => pen.path(q, { stroke: ink, width: i ? lw * 0.75 : lw, alpha }));
  if (opts.inner !== false) st.innerStrokes(f).forEach((q, i) => pen.path(q, { stroke: ink, width: lw * 0.8, alpha: alpha * 0.85 }));
  if (s.postureLines && opts.posture !== false) st.posture(f).forEach((q) => pen.path(q, { stroke: ink, width: lw * 0.7, alpha: alpha * 0.55 }));
}

function skeletonLines(kp) {
  const lines = [];
  for (const [a, b] of LIMBS) { const p = pt(kp, a), q = pt(kp, b); if (p && q) lines.push([p, q]); }
  const neck = mid(kp, LSH, RSH), hip = mid(kp, LHIP, RHIP), head = headOf(kp);
  if (neck && hip) lines.push([neck, hip]);
  if (neck && head) {
    const d = dist(neck, head.c);
    if (d > head.r) { const t = 1 - head.r / d; lines.push([neck, [neck[0] + (head.c[0] - neck[0]) * t, neck[1] + (head.c[1] - neck[1]) * t]]); }
  }
  return { lines, head, neck, hip };
}

function drawHandSkeleton(pen, st, f, alpha, lw, sides = ['L', 'R']) {
  const ink = st.theme.ink, kp = f.kp;
  for (const side of sides) {
    const h = handOf(f, side);
    if (!h) continue;
    const w = Math.min(lw, palmWidth(h.p) * 0.06);
    for (const [a, b] of HAND_BONES) pen.path([h.p[a], h.p[b]], { stroke: ink, width: w, alpha });
    const W = kp && pt(kp, side === 'L' ? LWR : RWR);
    if (W) pen.path([W, h.p[0]], { stroke: ink, width: w, alpha });
    for (const i of [0, 4, 8, 12, 16, 20]) pen.circle(h.p[i][0], h.p[i][1], w * 1.3, { fill: ink, fillAlpha: alpha });
  }
}

function drawAngleLabels(pen, st, f, fs, alpha) {
  const a = anglesOf(f), kp = f.kp;
  if (!a || !kp) return;
  const ink = st.theme.ink, o = fs * 0.5;
  const put = (p, txt, dx = o, dy = -o) => p && pen.text(txt, p[0] + dx, p[1] + dy, { size: fs, color: ink, alpha, weight: 600 });
  const hip = mid(kp, LHIP, RHIP), neck = mid(kp, LSH, RSH);
  if (a.trunk != null) put(hip, `T ${Math.round(a.trunk)}°`);
  if (a.neck != null) put(neck, `N ${Math.round(a.neck)}°`);
  for (const [side, el, wr, kn] of [['L', LEL, LWR, LKN], ['R', REL, RWR, RKN]]) {
    if (a['elbow' + side] != null) put(pt(kp, el), `E ${Math.round(a['elbow' + side])}°`);
    if (a['wrist' + side] != null) put(pt(kp, wr), `W ${Math.round(a['wrist' + side])}°`, o, fs * 1.2);
    if (a['knee' + side] != null) put(pt(kp, kn), `K ${Math.round(a['knee' + side])}°`);
  }
}

function drawSkeleton(pen, st, f, alpha, opts = {}) {
  const ink = st.theme.ink, u = st.u, s = st.s;
  if (opts.shade && f.c) pen.path(f.c, { fill: ink, fillAlpha: opts.shade, closed: true });
  if (!f.kp) return;
  const lw = s.lineWidth * u * (opts.weight ?? 2);
  const { lines, head, neck, hip } = skeletonLines(f.kp);
  for (const [p, q] of lines) pen.path([p, q], { stroke: ink, width: lw, alpha });
  if (head) pen.circle(head.c[0], head.c[1], head.r, { stroke: ink, width: lw, alpha });
  if (opts.joints !== false) {
    const r = lw * 1.4;
    const pts = [LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN].map((i) => pt(f.kp, i));
    pts.push(neck, hip);
    for (const p of pts) if (p) pen.circle(p[0], p[1], r, { fill: ink, fillAlpha: Math.min(1, alpha * 1.2) });
  }
  if (s.handSkeleton) drawHandSkeleton(pen, st, f, alpha, lw * 0.5);
  if (opts.angles && s.angleLabels) drawAngleLabels(pen, st, f, opts.angles, Math.min(1, alpha + 0.1));
}

// --------------------------------------------------------------------------- //
// Shared layout pieces: header, timeline band
// --------------------------------------------------------------------------- //
function decorSizes(st, contentW, contentH) {
  const fs = Math.max(10, Math.min(contentW * 0.018, contentH * 0.03), 9 * st.u);
  const on = st.s.labels;
  return { fs, pad: on ? fs * 1.2 : fs * 0.6, head: on ? fs * 3.4 : 0, foot: on ? fs * 4.2 : 0 };
}

function drawHeader(pen, st, L, title) {
  if (!st.s.labels) return;
  const ink = st.theme.ink, { fs, pad } = L;
  pen.text(title, pad, pad + fs * 1.1, { size: fs * 1.15, weight: 700, color: ink });
  const metric = st.s.metric === 'shape' ? 'silhouette change' : 'area change';
  const sub = `${st.keys.length} keyframes  ·  Δ ≥ ${st.s.threshold}% ${metric}  ·  t ${fmtT(st.span[0])}–${fmtT(st.span[1])} s`;
  pen.text(sub, pad, pad + fs * 2.45, { size: fs * 0.75, color: ink, alpha: 0.7 });
}

function drawTimeline(pen, st, L) {
  if (!st.s.labels || !st.keys.length) return;
  const ink = st.theme.ink, { fs, pad, W, H } = L;
  const x0 = pad + fs * 2.5, x1 = W - pad - fs * 2.5, y = H - pad - fs * 2.2;
  const [t0, t1] = st.span, tw = Math.max(1e-6, t1 - t0);
  const X = (t) => x0 + ((t - t0) / tw) * (x1 - x0);
  pen.path([[x0, y], [x1, y]], { stroke: ink, width: fs * 0.08, alpha: 0.6 });
  let gapStart = null;
  for (const f of st.frames) {
    if (!f.c && gapStart === null) gapStart = f.t;
    if (f.c && gapStart !== null) { pen.path([[X(gapStart), y], [X(f.t), y]], { stroke: st.theme.bg, width: fs * 0.12 }); gapStart = null; }
  }
  let lastLabel = -Infinity;
  st.keys.forEach((k) => {
    const x = X(k.t);
    pen.path([[x, y - fs * 0.45], [x, y + fs * 0.45]], { stroke: ink, width: fs * 0.09 });
    const w = pen.measure(String(k.n), fs * 0.7);
    if (x - w / 2 > lastLabel + fs * 0.35) { pen.text(String(k.n), x, y - fs * 0.7, { size: fs * 0.7, color: ink, align: 'center' }); lastLabel = x + w / 2; }
  });
  pen.text(`${fmtT(t0)} s`, x0 - fs * 0.5, y, { size: fs * 0.7, color: ink, alpha: 0.7, align: 'right', baseline: 'middle' });
  pen.text(`${fmtT(t1)} s`, x1 + fs * 0.5, y, { size: fs * 0.7, color: ink, alpha: 0.7, baseline: 'middle' });
  pen.text('time  (keyframe numbers above ticks)', (x0 + x1) / 2, y + fs * 1.5, { size: fs * 0.65, color: ink, alpha: 0.6, align: 'center' });
}

// --------------------------------------------------------------------------- //
// Grids. An item is one cell: { k: keyframe, box: focus box, ghosts: [items], labels: [line1, line2] }
// --------------------------------------------------------------------------- //
function gridLayout(st, items, { pairs = false } = {}) {
  const s = st.s, m = 0.08;
  const boxes = items.map((it) => it.box).filter(Boolean);
  let cw, ch, region = null;
  if (!boxes.length) return null;
  if (s.framing === 'fixed') {
    region = padBox(unionOf(boxes), m * 0.6);
    cw = region[2] - region[0]; ch = region[3] - region[1];
  } else {
    const ws = boxes.map((b) => b[2] - b[0]), hs = boxes.map((b) => b[3] - b[1]);
    const side = Math.max(...ws, ...hs);
    cw = Math.max(...ws) + 2 * m * side; ch = Math.max(...hs) + 2 * m * side;
  }
  const cfs = Math.max(6, Math.min(cw * 0.065, ch * 0.05));
  const lab = s.labels ? cfs * 2.7 : 0;
  ch += lab;
  const n = items.length;
  let cols = s.cols > 0 ? Math.min(s.cols * (pairs ? 2 : 1), n) : Math.max(1, Math.min(n, Math.round(Math.sqrt((n * ch) / cw * 1.5))));
  if (pairs && cols % 2) cols = Math.min(n, cols + 1);
  const rows = Math.ceil(n / cols);
  const D = decorSizes(st, cols * cw, rows * ch);
  const W = cols * cw + 2 * D.pad, H = rows * ch + 2 * D.pad + D.head + D.foot;
  return { ...D, cfs, lab, cw, ch, cols, rows, region, W, H, gx: D.pad, gy: D.pad + D.head };
}

function buildGrid(st, title, items, drawItem, opts = {}) {
  if (!items.length) return null;
  const L = gridLayout(st, items, opts);
  if (!L) return null;
  const s = st.s, ink = st.theme.ink;
  return {
    W: L.W, H: L.H,
    draw(pen) {
      pen.rect(0, 0, L.W, L.H, { fill: st.theme.bg });
      drawHeader(pen, st, L, title);
      items.forEach((it, j) => {
        const cx = L.gx + (j % L.cols) * L.cw, cy = L.gy + Math.floor(j / L.cols) * L.ch;
        pen.save(); pen.translate(cx, cy); pen.clip(0, 0, L.cw, L.ch);
        if (it.box) {
          const h = L.ch - L.lab;
          const [ox, oy] = L.region ? [L.region[0], L.region[1]] : [(it.box[0] + it.box[2]) / 2 - L.cw / 2, (it.box[1] + it.box[3]) / 2 - h / 2];
          pen.save(); pen.translate(-ox, L.lab - oy);
          for (const g of s.ghosts ? it.ghosts.slice(-s.ghosts) : []) drawItem(pen, g, s.opacity, true, L);
          drawItem(pen, it, s.current, false, L);
          pen.restore();
        }
        if (s.labels) {
          const f = L.cfs, x = f * 0.5, [l1, l2, l3] = it.labels;
          pen.text(l1, x, f * 1.2, { size: f, weight: 700, color: ink });
          pen.text(l2, x + pen.measure(l1, f, 700) + f * 0.4, f * 1.2, { size: f * 0.75, color: ink, alpha: 0.8 });
          if (l3) pen.text(l3, x, f * 2.2, { size: f * 0.6, color: ink, alpha: 0.6 });
        }
        pen.restore();
      });
      const lw = Math.max(0.5, L.fs * 0.05);
      for (let c = 0; c <= L.cols; c++) pen.path([[L.gx + c * L.cw, L.gy], [L.gx + c * L.cw, L.gy + L.rows * L.ch]], { stroke: ink, width: lw, alpha: c % 2 === 0 || !opts.pairs ? 0.18 : 0.07 });
      for (let r = 0; r <= L.rows; r++) pen.path([[L.gx, L.gy + r * L.ch], [L.gx + L.cols * L.cw, L.gy + r * L.ch]], { stroke: ink, width: lw, alpha: 0.18 });
      drawTimeline(pen, st, L);
    },
  };
}

function poseItems(st) {
  const items = st.keys.map((k, j) => ({
    k, box: frameBox(k),
    labels: [`${k.n}`, `${fmtT(k.t)} s`, j ? `fr ${k.i} · Δt ${fmtT(k.dt)} s · Δ ${(k.d * 100).toFixed(0)}%` : `fr ${k.i}`],
  }));
  items.forEach((it, j) => (it.ghosts = items.slice(0, j)));
  return items;
}

function handItems(st) {
  const items = [];
  for (const k of st.keys) for (const side of ['L', 'R']) {
    const a = anglesOf(k) || {}, h = handOf(k, side);
    const bits = [];
    if (a['wrist' + side] != null) bits.push(`wrist ${Math.round(a['wrist' + side])}°`);
    if (a['grip' + side] != null) bits.push(`grip ${a['grip' + side].toFixed(2)}×`);
    items.push({ k, side, box: handFocusBox(k, side), labels: [`${k.n}${side}`, `${fmtT(k.t)} s`, h ? bits.join(' · ') || ' ' : 'hand not detected'] });
  }
  items.forEach((it, j) => (it.ghosts = items.slice(0, j).filter((g) => g.side === it.side)));
  return items;
}

// --------------------------------------------------------------------------- //
// Overlaps
// --------------------------------------------------------------------------- //
function buildOverlap(st, title, region, body) {
  if (!st.keys.length || !region || !isFinite(region[0])) return null;
  const cw = region[2] - region[0], ch = region[3] - region[1];
  const D = decorSizes(st, cw, ch);
  const L = { ...D, region, cw, ch, W: cw + 2 * D.pad, H: ch + 2 * D.pad + D.head + D.foot, gx: D.pad, gy: D.pad + D.head };
  return {
    W: L.W, H: L.H,
    draw(pen) {
      pen.rect(0, 0, L.W, L.H, { fill: st.theme.bg });
      drawHeader(pen, st, L, title);
      pen.save(); pen.translate(L.gx - region[0], L.gy - region[1]); pen.clip(region[0], region[1], cw, ch);
      body(pen, L);
      pen.restore();
      drawTimeline(pen, st, L);
    },
  };
}

function numberPoses(pen, st, L) {
  if (!st.s.numbers) return;
  for (const k of st.keys) {
    const b = frameBox(k);
    pen.text(String(k.n), (b[0] + b[2]) / 2, b[1] - L.fs * 0.3, { size: L.fs * 0.7, color: st.theme.ink, alpha: 0.85, align: 'center' });
  }
}

// Centred moving average over a contiguous run of joint positions (reduces pose jitter).
function smoothRun(run, win) {
  const h = Math.max(0, Math.floor(((win | 0) - 1) / 2));
  if (!h) return run;
  return run.map((_, i) => {
    let x = 0, y = 0, n = 0;
    for (let j = Math.max(0, i - h); j <= Math.min(run.length - 1, i + h); j++) { x += run[j][0]; y += run[j][1]; n++; }
    return [x / n, y / n];
  });
}

// Trajectory of one point getter through keyframes (or every frame), with keyframe dots and an end label.
function drawTrajectory(pen, st, L, get, lab, dash, width) {
  const s = st.s, ink = st.theme.ink;
  let run = [], last = null;
  const flush = () => { if (run.length > 1) pen.path(s.trajAll ? smoothRun(run, s.trajSmooth) : run, { stroke: ink, width, alpha: s.trajOpacity, dash }); run = []; };
  for (const f of s.trajAll ? st.frames : st.keys) {
    const p = f.c ? get(f) : null;
    if (!p) { flush(); continue; }
    run.push(p); last = p;
  }
  flush();
  for (const k of st.keys) { const p = get(k); if (p) pen.circle(p[0], p[1], width * 1.1, { fill: ink, fillAlpha: s.trajOpacity }); }
  if (last) pen.text(lab, last[0] + L.fs * 0.4, last[1], { size: L.fs * 0.7, color: ink, weight: 700, baseline: 'middle' });
  return !!last;
}

function drawLegend(pen, st, L, legend, width) {
  if (!legend.length || !st.s.labels) return;
  const ink = st.theme.ink, fs = L.fs * 0.62, x = L.region[0] + fs * 0.5;
  let y = L.region[1] + fs * 1.2;
  for (const [lab, name, dash] of legend) {
    pen.path([[x, y], [x + fs * 3, y]], { stroke: ink, width, alpha: st.s.trajOpacity, dash });
    pen.text(`${lab}  ${name}`, x + fs * 3.6, y, { size: fs, color: ink, alpha: 0.85, baseline: 'middle' });
    y += fs * 1.45;
  }
}

function drawSkeletonOverlap(pen, st, L) {
  const s = st.s, ink = st.theme.ink, u = st.u, n = Math.max(1, st.keys.length);
  // Fills accumulate with every pose, so their per-pose opacity shrinks as poses are added:
  // the densest overlap of all n poses stays near `target` instead of going solid.
  const perPose = (target) => 1 - Math.pow(1 - target, 1 / n);
  if (s.shade) for (const k of st.keys) pen.path(k.c, { fill: ink, fillAlpha: Math.min(s.opacity * 0.5, perPose(0.35)), closed: true });
  if (s.sweep) { // each bone sweeps a quad between consecutive keyframes (the "painted" volume)
    for (let j = 1; j < st.keys.length; j++) {
      const A = st.keys[j - 1].kp, B = st.keys[j].kp;
      if (!A || !B) continue;
      for (const bone of [...LIMBS, 'spine']) {
        const pa = bone === 'spine' ? [mid(A, LSH, RSH), mid(A, LHIP, RHIP)] : [pt(A, bone[0]), pt(A, bone[1])];
        const pb = bone === 'spine' ? [mid(B, LSH, RSH), mid(B, LHIP, RHIP)] : [pt(B, bone[0]), pt(B, bone[1])];
        if (pa[0] && pa[1] && pb[0] && pb[1]) pen.path([pa[0], pa[1], pb[1], pb[0]], { fill: ink, fillAlpha: Math.min(s.opacity * 0.8, perPose(0.6)), closed: true });
      }
    }
  }
  for (const k of st.keys) drawSkeleton(pen, st, k, s.skelOverlap, { joints: true, weight: 1.3 });
  if (s.traj) {
    const legend = [], w = s.lineWidth * u * 0.8;
    for (const [key, [lab, name, get, dash]] of Object.entries(TRAJ)) {
      if (!s.trajJoints[key]) continue;
      if (drawTrajectory(pen, st, L, (f) => f.kp && get(f.kp), lab, dash, w)) legend.push([lab, name, dash]);
    }
    drawLegend(pen, st, L, legend, w);
  }
  numberPoses(pen, st, L);
}

const HAND_TRAJ = [[0, 'wrist', []], [4, 'thumb tip', [2, 3]], [8, 'index tip', [6, 3]]];

function handRegion(st) {
  const boxes = [];
  for (const k of st.keys) for (const side of ['L', 'R']) if (handOf(k, side)) boxes.push(handFocusBox(k, side));
  return boxes.length ? padBox(unionOf(boxes), 0.08) : null;
}

function drawHandOverlap(pen, st, L) {
  const s = st.s, ink = st.theme.ink;
  for (const k of st.keys) drawSketch(pen, st, k, s.opacity, { posture: false });
  if (s.traj) {
    const legend = [];
    let w = 1;
    for (const k of st.keys) for (const sd of ['L', 'R']) { const h = handOf(k, sd); if (h) { w = palmWidth(h.p) * 0.03; break; } }
    for (const side of ['L', 'R']) for (const [i, name, dash] of HAND_TRAJ) {
      const lab = `${side}${name === 'wrist' ? 'w' : name[0]}`;
      if (drawTrajectory(pen, st, L, (f) => handOf(f, side)?.p[i], lab, dash, w)) legend.push([lab, `${SIDES[side]} ${name}`, dash]);
    }
    drawLegend(pen, st, L, legend, w);
  }
}

// --------------------------------------------------------------------------- //
// The outputs. Each returns { W, H, draw(pen) } in source-pixel units, or null.
// --------------------------------------------------------------------------- //
export const OUTPUTS = [
  {
    id: 'sketch-grid', title: 'Sketch grid', file: 'sketch_grid',
    build: (st) => buildGrid(st, `Sketch grid — ${st.meta.name || ''}`, poseItems(st),
      (pen, it, a, ghost) => drawSketch(pen, st, it.k, a, ghost ? { posture: false, inner: false } : {})),
  },
  {
    id: 'sketch-overlap', title: 'Sketch overlap', file: 'sketch_overlap',
    build: (st) => buildOverlap(st, `Sketch overlap — ${st.meta.name || ''}`, padBox(unionBox(st.keys), 0.05), (pen, L) => {
      for (const k of st.keys) drawSketch(pen, st, k, st.s.opacity, { tone: st.s.tone ? st.s.opacity * 0.5 : 0 });
      numberPoses(pen, st, L);
    }),
  },
  {
    id: 'skeleton-grid', title: 'Skeleton grid', file: 'skeleton_grid',
    build: (st) => buildGrid(st, `Skeleton grid — ${st.meta.name || ''}`, poseItems(st),
      (pen, it, a, ghost, L) => drawSkeleton(pen, st, it.k, ghost ? Math.min(1, a * 2) : st.s.skelOpacity,
        { shade: st.s.shade && !ghost ? 0.12 : 0, joints: !ghost, weight: ghost ? 1.4 : 2, angles: ghost ? 0 : L.cfs * 0.62 })),
  },
  {
    id: 'skeleton-overlap', title: 'Skeleton overlap', file: 'skeleton_overlap',
    build: (st) => buildOverlap(st, `Skeleton overlap — ${st.meta.name || ''}`, padBox(unionBox(st.keys), 0.05), (pen, L) => drawSkeletonOverlap(pen, st, L)),
  },
  {
    id: 'hand-grid', title: 'Hands grid', file: 'hands_grid', hands: true,
    build: (st) => st.hasHands ? buildGrid(st, `Hands, wrists and fingers — ${st.meta.name || ''}`, handItems(st),
      (pen, it, a, ghost) => {
        drawSketch(pen, st, it.k, a, { posture: false, inner: !ghost });
        if (!ghost && st.s.handSkeleton) drawHandSkeleton(pen, st, it.k, a * 0.4, st.s.lineWidth * st.u, [it.side]);
      }, { pairs: true }) : null,
  },
  {
    id: 'hand-overlap', title: 'Hands overlap', file: 'hands_overlap', hands: true,
    build: (st) => st.hasHands ? buildOverlap(st, `Hands overlap — ${st.meta.name || ''}`, handRegion(st), (pen, L) => drawHandOverlap(pen, st, L)) : null,
  },
];

// Pick a raster scale that keeps the canvas within browser limits.
export function safeScale(W, H, want) {
  const maxSide = 16000, maxArea = 160e6;
  return Math.max(0.05, Math.min(want, maxSide / W, maxSide / H, Math.sqrt(maxArea / (W * H))));
}
