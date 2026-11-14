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
  person: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/latest/efficientdet_lite2.tflite',
  segment: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
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

// Pose + silhouette. Two instances: 'scene' looks for several people on a whole frame,
// 'crop' runs on a crop around one person (the main path: sharper mask, far fewer misses).
// CPU only: on many GPUs the segmentation mask reads back as all zeros, and the outline matters most.
export async function getPose(quality = 'full', kind = 'crop') {
  await vision();
  const key = `pose_${quality}_${kind}`;
  if (!cache[key]) {
    cache[key] = await create(mp.PoseLandmarker, {
      baseOptions: { modelAssetPath: MODELS[quality] || MODELS.full },
      runningMode: 'IMAGE', numPoses: kind === 'scene' ? 6 : 1, outputSegmentationMasks: true,
      minPoseDetectionConfidence: 0.2, minPosePresenceConfidence: 0.2, minTrackingConfidence: 0.2,
    }, true);
  }
  return cache[key];
}

// Person segmentation (hair, skin, clothes...; everything but background = the person). Used when
// the pose model finds no one in the crop, so the outline does not drop out.
export async function getSegmenter() {
  await vision();
  if (!cache.seg) {
    cache.seg = await create(mp.ImageSegmenter, {
      baseOptions: { modelAssetPath: MODELS.segment },
      runningMode: 'IMAGE', outputCategoryMask: true, outputConfidenceMasks: false,
    }, true);
  }
  return cache.seg;
}

// Person boxes (EfficientDet-Lite2, COCO): finds people the pose detector misses
// (crouched, side-on, small, cluttered scenes).
export async function getDetector() {
  await vision();
  if (!cache.det) {
    cache.det = await create(mp.ObjectDetector, {
      baseOptions: { modelAssetPath: MODELS.person },
      runningMode: 'IMAGE', categoryAllowlist: ['person'], scoreThreshold: 0.25, maxResults: 10,
    }, true);
  }
  return cache.det;
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

// Joints MediaPipe extrapolates outside the picture (e.g. legs cut off by the camera) count as not visible.
function toKp(lm, mapX, mapY, W, H) {
  return MP2COCO.map((j) => {
    const x = mapX(lm[j].x), y = mapY(lm[j].y);
    const inPic = x >= -0.01 * W && x <= 1.01 * W && y >= -0.01 * H && y <= 1.01 * H;
    return [x, y, inPic ? (lm[j].visibility ?? 1) : 0];
  });
}
function boxOfKp(kp) {
  const vis = kp.filter((p) => p[2] > 0.3);
  if (!vis.length) return null;
  return [Math.min(...vis.map((p) => p[0])), Math.min(...vis.map((p) => p[1])), Math.max(...vis.map((p) => p[0])), Math.max(...vis.map((p) => p[1]))];
}

// People found by the whole-frame pose detector (inference canvas -> source coords).
function scenePersons(res, src, spacing) {
  const out = [];
  (res.landmarks || []).forEach((lm, k) => {
    const kp = toKp(lm, (x) => x * src.width, (y) => y * src.height, src.width, src.height);
    let o = null;
    const m = res.segmentationMasks?.[k];
    if (m) o = outlineFromBinary(binaryFromFloat(m.getAsFloat32Array(), m.width, m.height), m.width, m.height, src.width / m.width, { spacing, smooth: 4, minArea: 400 });
    const box = o?.box || boxOfKp(kp);
    if (!box) return;
    const conf = lm.reduce((s, p) => s + (p.visibility ?? 1), 0) / lm.length;
    out.push({ kp, outline: o?.c || null, area: o?.area || 0, box, conf });
  });
  return out;
}

// Pose + silhouette of the one person inside a square crop around `box` (source px).
const CROP_PX = 512;
let cropCanvas = null;
function drawCrop(img, box) {
  const cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2;
  const side = Math.max(box[2] - box[0], box[3] - box[1]) * 1.3 + 16;
  cropCanvas = cropCanvas || Object.assign(document.createElement('canvas'), { width: CROP_PX, height: CROP_PX });
  const ctx = cropCanvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, CROP_PX, CROP_PX);
  ctx.drawImage(img, cx - side / 2, cy - side / 2, side, side, 0, 0, CROP_PX, CROP_PX);
  return { x0: cx - side / 2, y0: cy - side / 2, side };
}

// Crop mask (1 = person) -> outline in source px; mask pixels outside the picture are dropped.
function cropOutline(bin, mw, mh, cr, src, spacing) {
  const { x0, y0, side } = cr, mk = side / mw;
  for (let yy = 0; yy < mh; yy++) for (let xx = 0; xx < mw; xx++) {
    const sx = x0 + (xx + 0.5) * mk, sy = y0 + (yy + 0.5) * mk;
    if (sx < 0 || sy < 0 || sx >= src.width || sy >= src.height) bin[yy * mw + xx] = 0;
  }
  const o = outlineFromBinary(bin, mw, mh, mk, { spacing, smooth: 4, minArea: 400 });
  if (!o) return null;
  return { outline: o.c.map(([x, y]) => [x + x0, y + y0]), area: o.area, box: [o.box[0] + x0, o.box[1] + y0, o.box[2] + x0, o.box[3] + y0] };
}

// Pose + silhouette of the one person inside a square crop around `box` (source px).
function cropPose(pose, img, src, box, spacing = 4) {
  const cr = drawCrop(img, box);
  let out = null;
  pose.detect(cropCanvas, (res) => {
    const lm = res.landmarks?.[0], m = res.segmentationMasks?.[0];
    if (!lm || !m) return;
    const o = cropOutline(binaryFromFloat(m.getAsFloat32Array(), m.width, m.height), m.width, m.height, cr, src, spacing);
    if (!o) return;
    const kp = toKp(lm, (x) => cr.x0 + x * cr.side, (y) => cr.y0 + y * cr.side, src.width, src.height);
    out = { ...o, kp, conf: lm.reduce((s, p) => s + (p.visibility ?? 1), 0) / lm.length };
  });
  return out;
}

// Silhouette only (no joints), from the person segmenter, inside a crop around `box`.
function cropSegment(seg, img, src, box, spacing = 4) {
  const cr = drawCrop(img, box);
  let out = null;
  seg.segment(cropCanvas, (res) => {
    const m = res.categoryMask;
    if (!m) return;
    const d = m.getAsUint8Array(), bin = new Uint8Array(d.length);
    for (let k = 0; k < d.length; k++) bin[k] = d[k] !== 0 ? 1 : 0; // 0 = background
    const o = cropOutline(bin, m.width, m.height, cr, src, spacing);
    if (o) out = { ...o, kp: null, conf: 0.5 };
  });
  return out;
}

async function personBoxes(img, src, maxW) {
  const det = await getDetector();
  const { c, ctx, scale } = inferCanvas(src, maxW);
  ctx.drawImage(img, 0, 0, c.width, c.height);
  return { canvas: c, boxes: det.detect(c).detections.map((d) => {
    const b = d.boundingBox;
    return { box: [b.originX * scale, b.originY * scale, (b.originX + b.width) * scale, (b.originY + b.height) * scale], score: d.categories[0].score };
  }) };
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
// Candidates from the person detector and from the whole-frame pose detector are merged,
// then each is refined with pose on a crop around it.
export async function detectPeople(src, i, { quality = 'full', maxW = 960 } = {}) {
  const img = await src.frame(i);
  const { canvas, boxes } = await personBoxes(img, src, maxW);
  const scene = await getPose(quality, 'scene'), crop = await getPose(quality, 'crop');
  let found = [];
  scene.detect(canvas, (res) => { found = scenePersons(res, src, 6); });
  const cands = boxes.map((b) => ({ box: b.box, conf: b.score }));
  for (const p of found) if (!cands.some((c) => iou(c.box, p.box) > 0.4)) cands.push({ box: p.box, conf: p.conf });
  const seg = await getSegmenter();
  const people = [];
  for (const c of cands) {
    const r = cropPose(crop, img, src, c.box, 6) || cropSegment(seg, img, src, c.box, 6);
    const box = r ? r.box : c.box;
    if (people.some((p) => iou(p.box, box) > 0.6)) continue; // same person found twice
    people.push({ box, conf: c.conf, outline: r?.outline || null });
  }
  people.sort((a, b) => ctr(a.box)[0] - ctr(b.box)[0]);
  return people.map((p, k) => ({ index: k + 1, box: p.box.map(r1), conf: +p.conf.toFixed(3), outline: p.outline }));
}

// --------------------------------------------------------------------------- //
// Step 3b/4: follow one person through the range
// --------------------------------------------------------------------------- //
// Each frame: pose on a crop around the person's last box. Every REANCHOR frames, and whenever
// the person is lost, the person detector re-centres the crop on the best-matching person.
const REANCHOR = 10;
export async function analyse(src, { start, end, step, box, quality = 'full', maxW = 960, hands = true, onFrame, cancelled, onStatus }) {
  onStatus?.('loading models');
  const crop = await getPose(quality, 'crop');
  const seg = await getSegmenter();
  await getDetector();
  if (hands) await getHand();
  let last = box.slice(), lost = 0, n = 0, lastArea = 0;
  const t0 = performance.now();
  onStatus?.('tracking');
  const match = (cands) => {
    if (!cands.length) return null;
    const sc = cands.map((c) => iou(c.box, last));
    let k = sc.indexOf(Math.max(...sc));
    if (sc[k] > 0.2) return cands[k];
    const diag = Math.hypot(last[2] - last[0], last[3] - last[1]);
    const ds = cands.map((c) => Math.hypot(ctr(c.box)[0] - ctr(last)[0], ctr(c.box)[1] - ctr(last)[1]));
    k = ds.indexOf(Math.min(...ds));
    return ds[k] < 0.6 * diag * (1 + 0.15 * lost) ? cands[k] : null;
  };
  for (let i = start; i <= end; i += step) {
    if (cancelled()) break;
    const img = await src.frame(i);
    const rec = { i, t: +(i / src.fps).toFixed(4), area: 0, box: null, c: null, kp: null, h: null };
    let anchor = last;
    if (lost > 0 || n % REANCHOR === 0) {
      const m = match((await personBoxes(img, src, maxW)).boxes);
      if (m) anchor = m.box;
    }
    let r = cropPose(crop, img, src, anchor);
    // a crop around a box that has drifted can miss: retry once around the last good box
    if ((!r || iou(r.box, last) < 0.1) && anchor !== last) r = cropPose(crop, img, src, last);
    // no pose found: take the silhouette from the segmenter so the outline does not drop out
    if (!r || iou(r.box, last) < 0.1) { const sgm = cropSegment(seg, img, src, anchor); if (sgm) r = sgm; }
    // a partial pose mask (silhouette suddenly much smaller): prefer the segmenter's if it fits better,
    // keeping the joints from the pose model
    else if (lastArea && r.area < 0.55 * lastArea) {
      const sgm = cropSegment(seg, img, src, anchor);
      if (sgm && Math.abs(sgm.area - lastArea) < Math.abs(r.area - lastArea)) r = { ...sgm, kp: r.kp };
    }
    if (r && (iou(r.box, last) > 0.1 || lost > 0)) {
      rec.area = Math.round(r.area); rec.box = r.box.map(r1);
      rec.c = r.outline.map(([x, y]) => [r1(x), r1(y)]);
      rec.kp = r.kp ? r.kp.map(([x, y, v]) => [r1(x), r1(y), +v.toFixed(3)]) : null;
      if (hands && rec.kp) rec.h = await handsFor(img, rec.kp);
      last = r.box; lost = 0;
      lastArea = lastArea ? 0.7 * lastArea + 0.3 * r.area : r.area;
    } else lost++;
    n++;
    onFrame(rec, n, n / ((performance.now() - t0) / 1000));
    if (n % 2 === 0) await new Promise((res) => setTimeout(res, 0)); // let the page repaint
  }
}
