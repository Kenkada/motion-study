// engine.js - in-browser analysis with MediaPipe Tasks Vision (free, runs locally in the page).
// Produces the same per-frame records as the Python server:
//   { i, t, area, box, c, kp (COCO-17 [x,y,conf]), h: [{side, p: 21x[x,y], s}] }

import { binaryFromFloat, outlineFromBinary } from './contour.js';

const TV = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35';
const MODELS = {
  lite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
  full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
  heavy: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
  hand: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
};
// MediaPipe 33-point pose -> COCO 17 (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
const MP2COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

let mp = null, fileset = null;
const cache = {};

async function vision() {
  if (!mp) {
    mp = await import(`${TV}/vision_bundle.mjs`);
    fileset = await mp.FilesetResolver.forVisionTasks(`${TV}/wasm`);
  }
  return mp;
}

async function create(Klass, opts, cpuOnly = false) {
  if (cpuOnly) return Klass.createFromOptions(fileset, { ...opts, baseOptions: { ...opts.baseOptions, delegate: 'CPU' } });
  try {
    return await Klass.createFromOptions(fileset, { ...opts, baseOptions: { ...opts.baseOptions, delegate: 'GPU' } });
  } catch (e) {
    console.warn('GPU delegate unavailable, using CPU', e);
    return await Klass.createFromOptions(fileset, { ...opts, baseOptions: { ...opts.baseOptions, delegate: 'CPU' } });
  }
}

export async function getPose(quality = 'full') {
  await vision();
  if (!cache['pose_' + quality]) {
    cache['pose_' + quality] = await create(mp.PoseLandmarker, {
      baseOptions: { modelAssetPath: MODELS[quality] || MODELS.full },
      runningMode: 'IMAGE', numPoses: 6, outputSegmentationMasks: true,
      minPoseDetectionConfidence: 0.4, minPosePresenceConfidence: 0.4, minTrackingConfidence: 0.4,
    }, true); // CPU: on many GPUs the segmentation mask reads back as all zeros, and the outline matters most
  }
  return cache['pose_' + quality];
}

export async function getHand() {
  await vision();
  if (!cache.hand) {
    cache.hand = await create(mp.HandLandmarker, {
      baseOptions: { modelAssetPath: MODELS.hand },
      runningMode: 'IMAGE', numHands: 2,
      minHandDetectionConfidence: 0.3, minHandPresenceConfidence: 0.3, minTrackingConfidence: 0.3,
    });
  }
  return cache.hand;
}

// --------------------------------------------------------------------------- //
// Sources held in the browser (nothing is uploaded)
// --------------------------------------------------------------------------- //
const once = (el, ev) => new Promise((res, rej) => {
  const ok = () => { el.removeEventListener('error', bad); res(); };
  const bad = () => { el.removeEventListener(ev, ok); rej(new Error('the browser cannot decode this file')); };
  el.addEventListener(ev, ok, { once: true }); el.addEventListener('error', bad, { once: true });
});

const COMMON_FPS = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 90, 100, 120];

export class BrowserVideo {
  constructor(file) {
    this.kind = 'video'; this.name = file.name;
    this.v = document.createElement('video');
    Object.assign(this.v, { muted: true, playsInline: true, preload: 'auto' });
    this.v.src = URL.createObjectURL(file);
  }
  async open() {
    await once(this.v, 'loadedmetadata');
    this.width = this.v.videoWidth; this.height = this.v.videoHeight;
    this.duration = this.v.duration;
    if (!this.width) throw new Error('the browser cannot decode this video (try MP4/H.264 or WebM)');
    this.fps = await this.estimateFps();
    this.count = Math.max(1, Math.floor(this.duration * this.fps + 1e-6));
    return this.meta();
  }
  setFps(fps) { this.fps = fps; this.count = Math.max(1, Math.floor(this.duration * fps + 1e-6)); }
  meta() { return { kind: 'video', width: this.width, height: this.height, fps: this.fps, count: this.count, duration: this.duration, name: this.name }; }
  // Measures the frame rate from presented-frame timestamps while playing a few frames muted.
  async estimateFps() {
    const v = this.v;
    if (!('requestVideoFrameCallback' in v)) return 30;
    const times = [];
    const done = new Promise((res) => {
      const step = (_, m) => { times.push(m.mediaTime); if (times.length < 14) v.requestVideoFrameCallback(step); else res(); };
      v.requestVideoFrameCallback(step);
      setTimeout(res, 2500);
    });
    try { await v.play(); await done; } catch { /* autoplay blocked */ }
    v.pause();
    const d = [];
    for (let k = 1; k < times.length; k++) if (times[k] > times[k - 1]) d.push(times[k] - times[k - 1]);
    if (d.length < 4) return 30;
    d.sort((a, b) => a - b);
    const fps = 1 / d[Math.floor(d.length / 4)]; // lower quartile of the gap = the true frame interval
    const near = COMMON_FPS.find((f) => Math.abs(f - fps) / f < 0.03);
    return near || Math.round(fps * 100) / 100;
  }
  async frame(i) {
    const t = Math.min(this.duration - 0.001, (i + 0.5) / this.fps);
    if (Math.abs(this.v.currentTime - t) > 1e-4) { const p = once(this.v, 'seeked'); this.v.currentTime = t; await p; }
    return this.v;
  }
}

export class BrowserImages {
  constructor(files, interval = 1) {
    this.kind = 'images'; this.files = [...files].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    this.name = `${files.length} image${files.length > 1 ? 's' : ''}`; this.fps = 1 / interval; this._i = -1;
  }
  async open() {
    const b = await createImageBitmap(this.files[0]);
    this.width = b.width; this.height = b.height; this.count = this.files.length;
    this.duration = this.count / this.fps; this._i = 0; this._b = b;
    return this.meta();
  }
  setFps(fps) { this.fps = fps; this.duration = this.count / fps; }
  meta() { return { kind: 'images', width: this.width, height: this.height, fps: this.fps, count: this.count, duration: this.duration, name: this.name }; }
  async frame(i) {
    if (i !== this._i) {
      this._b?.close?.();
      const b = await createImageBitmap(this.files[i]);
      if (b.width !== this.width || b.height !== this.height) {
        const c = new OffscreenCanvas(this.width, this.height); c.getContext('2d').drawImage(b, 0, 0, this.width, this.height);
        this._b = c.transferToImageBitmap();
      } else this._b = b;
      this._i = i;
    }
    return this._b;
  }
}

// --------------------------------------------------------------------------- //
// Helpers shared by detection and analysis
// --------------------------------------------------------------------------- //
function inferCanvas(src, maxW) {
  const scale = src.width > maxW ? src.width / maxW : 1;
  const c = document.createElement('canvas');
  c.width = Math.round(src.width / scale); c.height = Math.round(src.height / scale);
  return { c, ctx: c.getContext('2d', { willReadFrequently: true }), scale };
}

function personsFrom(res, src, spacing) {
  const out = [];
  (res.landmarks || []).forEach((lm, k) => {
    // Joints MediaPipe extrapolates outside the picture (e.g. legs cut off by the camera) count as not visible.
    const inside = (q) => q.x >= -0.01 && q.x <= 1.01 && q.y >= -0.01 && q.y <= 1.01;
    const kp = MP2COCO.map((j) => [lm[j].x * src.width, lm[j].y * src.height, inside(lm[j]) ? (lm[j].visibility ?? 1) : 0]);
    let o = null;
    const m = res.segmentationMasks?.[k];
    if (m) {
      const data = m.getAsFloat32Array();
      o = outlineFromBinary(binaryFromFloat(data, m.width, m.height), m.width, m.height, src.width / m.width, { spacing, smooth: 4, minArea: 400 });
    }
    let box = o?.box;
    if (!box) {
      const vis = kp.filter((p) => p[2] > 0.3);
      if (!vis.length) return;
      box = [Math.min(...vis.map((p) => p[0])), Math.min(...vis.map((p) => p[1])), Math.max(...vis.map((p) => p[0])), Math.max(...vis.map((p) => p[1]))];
    }
    const conf = lm.reduce((s, p) => s + (p.visibility ?? 1), 0) / lm.length;
    out.push({ kp, outline: o?.c || null, area: o?.area || 0, box, conf });
  });
  return out;
}

export function iou(a, b) {
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]), x1 = Math.min(a[2], b[2]), y1 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}
const ctr = (b) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
const r1 = (v) => Math.round(v * 10) / 10;

// Square crop around one hand: beyond the wrist along the forearm, sized from the forearm.
// The same rule is used by the Python server (analysis.hand_crop).
export function handCrop(kp, side, off = 0.45, scale = 1.3) {
  const [el, wr] = side === 'L' ? [7, 9] : [8, 10];
  const W = kp[wr], E = kp[el];
  if (!W || W[2] < 0.2) return null;
  const sh = kp[5][2] > 0.2 && kp[6][2] > 0.2 ? Math.hypot(kp[5][0] - kp[6][0], kp[5][1] - kp[6][1]) : 0;
  const fore = E && E[2] > 0.2 ? Math.hypot(W[0] - E[0], W[1] - E[1]) : 0;
  const L = fore || sh * 0.8 || 100;
  const size = Math.max(L * scale, 48);
  let cx = W[0], cy = W[1];
  if (fore) { cx += ((W[0] - E[0]) / fore) * L * off; cy += ((W[1] - E[1]) / fore) * L * off; }
  return { x: cx - size / 2, y: cy - size / 2, size, wx: W[0], wy: W[1] };
}

const HAND_PX = 256;
const HAND_TRIES = [[0.45, 1.3, 'none'], [0.6, 1.8, 'none'], [0.6, 1.8, 'contrast(1.6) saturate(0.6)']];
let handCanvas = null;
export async function handsFor(img, kp) {
  const hl = await getHand();
  handCanvas = handCanvas || Object.assign(document.createElement('canvas'), { width: HAND_PX, height: HAND_PX });
  const ctx = handCanvas.getContext('2d');
  const picks = [];
  for (const side of ['L', 'R']) {
    let best = null;
    for (const [off, sc, filter] of HAND_TRIES) { // retry larger, then contrast-boosted, until a hand is found
      const cr = handCrop(kp, side, off, sc);
      if (!cr) break;
      ctx.filter = 'none'; ctx.fillStyle = '#000'; ctx.fillRect(0, 0, HAND_PX, HAND_PX);
      ctx.filter = filter;
      ctx.drawImage(img, cr.x, cr.y, cr.size, cr.size, 0, 0, HAND_PX, HAND_PX);
      ctx.filter = 'none';
      const res = hl.detect(handCanvas);
      let bd = cr.size * 0.45;
      (res.landmarks || []).forEach((lm, k) => {
        const p = lm.map((q) => [cr.x + q.x * cr.size, cr.y + q.y * cr.size]);
        const d = Math.hypot(p[0][0] - cr.wx, p[0][1] - cr.wy);
        if (d < bd) { bd = d; best = { side, p: p.map(([x, y]) => [r1(x), r1(y)]), s: +(res.handedness?.[k]?.[0]?.score ?? 0).toFixed(3), d: d / cr.size }; }
      });
      if (best) break;
    }
    if (best) picks.push(best);
  }
  // Both crops can latch onto the same hand when the hands are close: keep the closer match.
  if (picks.length === 2) {
    const [a, b] = picks;
    const palm = Math.hypot(a.p[9][0] - a.p[0][0], a.p[9][1] - a.p[0][1]);
    if (Math.hypot(a.p[0][0] - b.p[0][0], a.p[0][1] - b.p[0][1]) < 0.5 * palm) {
      picks.splice(a.d <= b.d ? 1 : 0, 1);
    }
  }
  picks.forEach((h) => delete h.d);
  return picks.length ? picks : null;
}

// --------------------------------------------------------------------------- //
// Step 3a: people on one frame
// --------------------------------------------------------------------------- //
export async function detectPeople(src, i, { quality = 'full', maxW = 960 } = {}) {
  const pose = await getPose(quality);
  await pose.setOptions({ runningMode: 'IMAGE' });
  const img = await src.frame(i);
  const { c, ctx } = inferCanvas(src, maxW);
  ctx.drawImage(img, 0, 0, c.width, c.height);
  let people = [];
  pose.detect(c, (res) => { people = personsFrom(res, src, 6); });
  people.sort((a, b) => ctr(a.box)[0] - ctr(b.box)[0]);
  return people.map((p, k) => ({ index: k + 1, box: p.box.map(r1), conf: +p.conf.toFixed(3), outline: p.outline }));
}

// --------------------------------------------------------------------------- //
// Step 3b/4: follow one person through the range
// --------------------------------------------------------------------------- //
let ts = 0;
export async function analyse(src, { start, end, step, box, quality = 'full', maxW = 960, hands = true, onFrame, cancelled, onStatus }) {
  onStatus?.('loading models');
  const pose = await getPose(quality);
  if (hands) await getHand();
  await pose.setOptions({ runningMode: 'VIDEO' });
  const { c, ctx } = inferCanvas(src, maxW);
  let last = box.slice(), lost = 0;
  const t0 = performance.now();
  let n = 0;
  onStatus?.('tracking');
  for (let i = start; i <= end; i += step) {
    if (cancelled()) break;
    const img = await src.frame(i);
    ctx.drawImage(img, 0, 0, c.width, c.height);
    ts = Math.max(ts + 1, Math.round(performance.now()));
    let people = [];
    pose.detectForVideo(c, ts, (res) => { people = personsFrom(res, src, 4); });
    const rec = { i, t: +(i / src.fps).toFixed(4), area: 0, box: null, c: null, kp: null, h: null };
    let pick = null;
    if (people.length) {
      const sc = people.map((p) => iou(p.box, last));
      let k = sc.indexOf(Math.max(...sc));
      if (sc[k] > 0.25) pick = people[k];
      else {
        const diag = Math.hypot(last[2] - last[0], last[3] - last[1]);
        const ds = people.map((p) => Math.hypot(ctr(p.box)[0] - ctr(last)[0], ctr(p.box)[1] - ctr(last)[1]));
        k = ds.indexOf(Math.min(...ds));
        if (ds[k] < 0.6 * diag * (1 + 0.1 * lost)) pick = people[k];
      }
    }
    if (pick && pick.outline) {
      rec.area = Math.round(pick.area); rec.box = pick.box.map(r1);
      rec.c = pick.outline.map(([x, y]) => [r1(x), r1(y)]);
      rec.kp = pick.kp.map(([x, y, v]) => [r1(x), r1(y), +v.toFixed(3)]);
      if (hands) rec.h = await handsFor(img, rec.kp);
      last = pick.box; lost = 0;
    } else lost++;
    n++;
    onFrame(rec, n, n / ((performance.now() - t0) / 1000));
    if (n % 2 === 0) await new Promise((r) => setTimeout(r, 0)); // let the page repaint
  }
}
