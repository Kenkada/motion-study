// app.js - UI flow: open a video -> choose person -> live study.
// Two engines produce the same per-frame data:
//   browser: MediaPipe in this page (works on GitHub Pages, nothing is uploaded)
//   server : the local Python server (YOLOv8 segmentation + ByteTrack + MediaPipe hands)
import { Study, OUTPUTS, CanvasPen, SvgPen, TRAJ, ANGLES, anglesOf, fmtAngle, safeScale, fmtT } from './render.js';
import { makeZip } from './zip.js';
import * as engine from './engine.js';
import { COLLECT_URL, COLLECT_KEY } from './config.js';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const DEFAULTS = {
  metric: 'shape', threshold: 10, smooth: 3, minGap: 0, includeLast: true,
  bg: 'white', opacity: 0.12, current: 1, lineWidth: 1.2, passes: 3, wobble: 1, seed: 0,
  cols: 0, ghosts: 3, framing: 'follow', labels: true, numbers: false, tone: false,
  handDetail: true, limbs: true, limbWidth: 1, postureLines: false, handSkeleton: true, angleLabels: true,
  skelOpacity: 0.85, skelOverlap: 0.3, shade: true, sweep: true, traj: true, trajOpacity: 0.7, trajSmooth: 5, trajAll: false,
  trajJoints: { head: true, shoulder: true, lwrist: true, rwrist: true, hip: true, lknee: true, rknee: true, lankle: true, rankle: true },
};

const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};

const state = {
  mode: 'browser',  // 'browser' | 'server' | 'file' (a reopened frames.json)
  serverUp: false,
  job: null,        // source meta (+ id in server mode)
  src: null,        // BrowserVideo / BrowserImages in browser mode
  people: [],
  detectFrame: -1,
  frames: [],
  params: {},
  running: false,
  cancel: false,
  s: { ...structuredClone(DEFAULTS), ...(store.get('ms-settings') || {}) },
};
state.s.trajJoints = { ...DEFAULTS.trajJoints, ...(state.s.trajJoints || {}) };
$$('.collect-note').forEach((el) => (el.hidden = !COLLECT_URL));
const study = new Study();
let built = {};
// For use from the browser console (e.g. scripted exports): window.motionStudy.state / .study / .built()
window.motionStudy = { state, study, built: () => built };

// --------------------------------------------------------------------------- //
// Helpers
// --------------------------------------------------------------------------- //
async function api(path, opts = {}) {
  const r = await fetch(path, {
    ...opts,
    headers: opts.body && typeof opts.body === 'string' ? { 'Content-Type': 'application/json' } : undefined,
  });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  return r.json();
}
function showStep(name) {
  for (const s of ['upload', 'select', 'results']) $(`#step-${s}`).hidden = s !== name;
  $$('.steps span').forEach((el) => el.classList.toggle('on', el.dataset.step === name));
  window.scrollTo({ top: 0 });
}
function setProgress(el, frac, text) {
  el.hidden = false;
  el.firstElementChild.style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
  el.lastElementChild.textContent = text;
}
function baseName() {
  const n = (state.job?.name || 'study').replace(/\.[^.]+$/, '');
  return n.replace(/[^A-Za-z0-9._-]+/g, '_').slice(0, 60) || 'study';
}
function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
const nf = new Intl.NumberFormat();
// Browser sources can only seek one frame at a time: serialise preview, detection and analysis.
let srcQueue = Promise.resolve();
function withSrc(fn) { const p = srcQueue.then(fn, fn); srcQueue = p.catch(() => {}); return p; }

// --------------------------------------------------------------------------- //
// Step 1: open a file
// --------------------------------------------------------------------------- //
const drop = $('#drop'), fileInput = $('#file');
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); handleFiles([...e.dataTransfer.files]); });
drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
fileInput.addEventListener('change', () => { handleFiles([...fileInput.files]); fileInput.value = ''; });
$$('input[name=engine]').forEach((r) => r.addEventListener('change', () => { store.set('ms-engine', r.value); }));
$('#open-json').addEventListener('change', async (e) => {
  const f = e.target.files[0]; e.target.value = '';
  if (f) openSavedStudy(f);
});

function chosenEngine() {
  if (!state.serverUp) return 'browser';
  return $('input[name=engine]:checked')?.value || 'browser';
}

function uploadOne(file, jobId, done, total) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const q = new URLSearchParams({ name: file.name });
    if (jobId) q.set('job', jobId);
    xhr.open('POST', `api/upload?${q}`);
    xhr.upload.onprogress = (e) => {
      const frac = (done + (e.lengthComputable ? e.loaded / e.total : 0)) / total;
      setProgress($('#upload-progress'), frac, `Uploading ${file.name} · ${Math.round(frac * 100)}%`);
    };
    xhr.onload = () => (xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(xhr.responseText || xhr.statusText)));
    xhr.onerror = () => reject(new Error('upload failed (is the server running?)'));
    xhr.send(file);
  });
}

const IMG_RE = /\.(jpe?g|png|webp|bmp|tiff?|avif|gif)$/i;
async function handleFiles(files) {
  const json = files.find((f) => /\.json$/i.test(f.name));
  if (json) return openSavedStudy(json);
  files = files.filter((f) => /^(video|image)\//.test(f.type) || /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(f.name) || IMG_RE.test(f.name));
  if (!files.length) return alert('Choose a video file or image files.');
  const videos = files.filter((f) => !/^image\//.test(f.type) && !IMG_RE.test(f.name));
  if (videos.length && files.length > 1) return alert('Open one video, or several images - not both.');
  try {
    if (chosenEngine() === 'server') {
      let id = null;
      for (let i = 0; i < files.length; i++) id = (await uploadOne(files[i], id, i, files.length)).id;
      setProgress($('#upload-progress'), 1, 'Opening…');
      const meta = await api(`api/jobs/${id}/open`, { method: 'POST', body: JSON.stringify({ interval: 1 }) });
      $('#upload-progress').hidden = true;
      state.mode = 'server';
      enterSelect(meta, 0);
    } else {
      setProgress($('#upload-progress'), 0.5, 'Opening in the browser… (measuring the frame rate)');
      const src = videos.length ? new engine.BrowserVideo(files[0]) : new engine.BrowserImages(files, 1);
      const meta = await src.open();
      $('#upload-progress').hidden = true;
      state.mode = 'browser'; state.src = src;
      enterSelect(meta, 0);
    }
  } catch (e) {
    setProgress($('#upload-progress'), 0, `Error: ${e.message}`);
  }
}

async function openSavedStudy(file) {
  try {
    const d = JSON.parse(await file.text());
    if (!Array.isArray(d.frames)) throw new Error('not a Motion Study frames.json');
    state.mode = 'file'; state.src = null;
    state.job = { name: d.source?.name || file.name, ...(d.source || {}) };
    state.params = d.source || {};
    state.frames = d.frames;
    if (d.settings) { state.s = { ...structuredClone(DEFAULTS), ...d.settings }; state.s.trajJoints = { ...DEFAULTS.trajJoints, ...(d.settings.trajJoints || {}) }; syncControls(); }
    location.hash = '';
    enterResults(false);
  } catch (e) {
    alert(`Could not open ${file.name}: ${e.message}`);
  }
}

async function loadRecent() {
  if (!state.serverUp) { $('#recent').hidden = true; return; }
  try {
    const jobs = await api('api/jobs');
    const ul = $('#recent-list'); ul.innerHTML = '';
    for (const j of jobs) {
      const li = document.createElement('li');
      li.innerHTML = `<a href="#job=${j.id}"></a><span class="muted small"></span><button class="link">delete</button>`;
      li.querySelector('a').textContent = j.name || j.id;
      li.querySelector('span').textContent = `${j.modified}${j.analysed ? ' · analysed' : ''}`;
      li.querySelector('button').onclick = async () => {
        if (!confirm(`Delete "${j.name}" and its uploaded file from the jobs folder?`)) return;
        await api(`api/jobs/${j.id}`, { method: 'DELETE' }); loadRecent();
      };
      ul.appendChild(li);
    }
    $('#recent').hidden = !jobs.length;
  } catch { $('#recent').hidden = true; }
}

// --------------------------------------------------------------------------- //
// Step 2: choose the person
// --------------------------------------------------------------------------- //
const slider = $('#frame-slider'), preview = $('#preview');
let previewImg = null, detectSeq = 0, modelsLoaded = false;

function setEngineOptions() {
  const server = state.mode === 'server';
  $('#quality').innerHTML = server
    ? '<option value="fast">Fast (YOLOv8n-seg)</option><option value="accurate">Accurate (YOLOv8s-seg, ~23 MB download on first use)</option>'
    : '<option value="full">Standard (MediaPipe pose, 9 MB)</option><option value="heavy">Accurate (MediaPipe pose heavy, 29 MB)</option><option value="lite">Fast (MediaPipe pose lite, 5 MB)</option>';
  $('#conf-row').hidden = !server;
  $('#fps-row').hidden = server || state.job.kind !== 'video';
  $('#fps').value = state.job.fps;
  $('#engine-note').textContent = server
    ? 'Runs on the local server: YOLOv8 segmentation and tracking, YOLOv8 pose, MediaPipe hands.'
    : 'Runs in this browser with MediaPipe (pose, silhouette and 21-point hands). The video stays on this computer.';
}

function enterSelect(meta, frame) {
  state.job = meta; state.people = []; state.detectFrame = -1;
  location.hash = state.mode === 'server' ? `job=${meta.id}` : '';
  showStep('select');
  setEngineOptions();
  slider.max = Math.max(0, meta.count - 1);
  slider.value = frame;
  $('#img-interval-row').hidden = meta.kind !== 'images';
  $('#end-time').value = '';
  $('#end-time').placeholder = `${fmtT(meta.duration)} (end)`;
  $('#person-index').value = '';
  $('#select-error').textContent = '';
  onSlider();
}

function frameLabel(i) {
  const m = state.job;
  return m.kind === 'images' ? `image ${i + 1} of ${m.count}` : `frame ${i} of ${m.count - 1} · ${fmtT(i / m.fps)} s`;
}

function onSlider() {
  const i = +slider.value;
  $('#frame-readout').textContent = frameLabel(i);
  updateEstimate();
  loadPreview(i);
  detectSoon(i);
}
slider.addEventListener('input', onSlider);
const detectSoon = debounce((i) => detect(i), 350);

function loadPreview(i) {
  if (state.mode === 'server') {
    const img = new Image();
    img.onload = () => { if (+slider.value === i) { previewImg = img; drawPreview(); } };
    img.src = `api/jobs/${state.job.id}/frame?i=${i}&w=1280`;
    return;
  }
  withSrc(async () => {
    if (+slider.value !== i) return;
    const frame = await state.src.frame(i);
    const k = Math.min(1, 1280 / state.job.width), c = document.createElement('canvas');
    c.width = Math.round(state.job.width * k); c.height = Math.round(state.job.height * k);
    c.getContext('2d').drawImage(frame, 0, 0, c.width, c.height);
    if (+slider.value === i) { previewImg = c; drawPreview(); }
  });
}

function drawPreview() {
  if (!previewImg) return;
  const c = preview, ctx = c.getContext('2d');
  c.width = previewImg.naturalWidth || previewImg.width; c.height = previewImg.naturalHeight || previewImg.height;
  ctx.drawImage(previewImg, 0, 0);
  if (state.detectFrame !== +slider.value) return;
  const k = c.width / state.job.width;
  const sel = +$('#person-index').value;
  const fs = Math.max(14, c.width / 45);
  for (const p of state.people) {
    const [x0, y0, x1, y1] = p.box.map((v) => v * k);
    const on = p.index === sel;
    ctx.lineWidth = Math.max(2, c.width / 400) * (on ? 2 : 1);
    ctx.strokeStyle = on ? '#ffd400' : 'rgba(255,255,255,.95)';
    ctx.setLineDash(on ? [] : [8, 5]);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.setLineDash([]);
    if (p.outline && on) {
      ctx.beginPath(); p.outline.forEach(([x, y], j) => (j ? ctx.lineTo(x * k, y * k) : ctx.moveTo(x * k, y * k)));
      ctx.closePath(); ctx.fillStyle = 'rgba(255,212,0,.25)'; ctx.fill();
    }
    const r = fs * 0.9, cx = Math.max(r, x0 + r + 4), cy = Math.max(r, y0 + r + 4);
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = on ? '#ffd400' : 'rgba(0,0,0,.8)'; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = '#fff'; ctx.stroke();
    ctx.fillStyle = on ? '#000' : '#fff'; ctx.font = `700 ${fs}px system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(p.index), cx, cy + 1);
  }
}

async function detect(i) {
  const seq = ++detectSeq;
  $('#detect-status').textContent = state.mode === 'browser' && !modelsLoaded ? 'Loading the detection models (first time only)…' : 'Detecting people…';
  $('#people-list').innerHTML = '';
  try {
    let people;
    if (state.mode === 'server') {
      const r = await api(`api/jobs/${state.job.id}/detect`, {
        method: 'POST',
        body: JSON.stringify({ frame: i, quality: $('#quality').value, conf: +$('#conf').value, max_width: +$('#max-width').value }),
      });
      people = r.people;
    } else {
      people = await withSrc(() => engine.detectPeople(state.src, i, { quality: $('#quality').value, maxW: +$('#max-width').value }));
      modelsLoaded = true;
    }
    if (seq !== detectSeq) return;
    state.people = people; state.detectFrame = i;
    const n = people.length;
    $('#detect-status').textContent = n ? `${n} ${n === 1 ? 'person' : 'people'} found` : 'No people found on this frame - try another frame';
    if (!n) submitInfo('no-people', { frame: i, detect: { quality: $('#quality').value, max_width: +$('#max-width').value } });
    const list = $('#people-list');
    list.innerHTML = n ? '<span class="muted small" style="width:100%">Indices on this frame:</span>' : '';
    for (const p of people) {
      const b = document.createElement('button');
      b.textContent = p.index; b.title = `confidence ${Math.round(p.conf * 100)}%`;
      b.onclick = () => { $('#person-index').value = p.index; drawPreview(); };
      list.appendChild(b);
    }
    if (n === 1 && !$('#person-index').value) $('#person-index').value = 1;
    drawPreview();
  } catch (e) {
    if (seq === detectSeq) $('#detect-status').textContent = `Detection failed: ${e.message}`;
    submitInfo('detect-error', { frame: i, error: String(e.message || e) });
  }
}
$('#person-index').addEventListener('input', drawPreview);
['#quality', '#conf', '#max-width'].forEach((s) => $(s).addEventListener('change', () => detect(+slider.value)));
['#end-time', '#frame-step'].forEach((s) => $(s).addEventListener('input', updateEstimate));
$('#img-interval').addEventListener('change', async () => {
  const interval = +$('#img-interval').value || 1;
  if (state.mode === 'server') state.job = await api(`api/jobs/${state.job.id}/open`, { method: 'POST', body: JSON.stringify({ interval }) });
  else { state.src.setFps(1 / interval); state.job = state.src.meta(); }
  onSlider();
});
$('#fps').addEventListener('change', () => {
  const fps = +$('#fps').value;
  if (state.mode !== 'browser' || !(fps > 0)) return;
  state.src.setFps(fps); state.job = state.src.meta();
  slider.max = Math.max(0, state.job.count - 1);
  onSlider();
});

function rangeFrames() {
  const m = state.job, start = +slider.value;
  const endT = $('#end-time').value === '' ? null : +$('#end-time').value;
  const end = endT == null ? m.count - 1 : Math.min(m.count - 1, Math.round(endT * m.fps));
  const step = Math.max(1, Math.round(+$('#frame-step').value || 1));
  return { start, end, step, n: end >= start ? Math.floor((end - start) / step) + 1 : 0 };
}
function updateEstimate() {
  if (!state.job) return;
  const { n, start, end } = rangeFrames();
  const m = state.job;
  $('#estimate').textContent = n
    ? `${nf.format(n)} frames to analyse (${fmtT(start / m.fps)}–${fmtT(end / m.fps)} s). Roughly ${Math.max(1, Math.round(n / 6))} s.`
    : 'The end is before the start frame.';
}

$('#btn-analyse').addEventListener('click', startAnalysis);
$('#person-index').addEventListener('keydown', (e) => { if (e.key === 'Enter') startAnalysis(); });
$('#btn-new').addEventListener('click', goHome);
function goHome() { clearTimeout(pollTimer); state.cancel = true; location.hash = ''; showStep('upload'); loadRecent(); }

async function startAnalysis() {
  const err = $('#select-error'); err.textContent = '';
  const idx = parseInt($('#person-index').value, 10);
  if (state.detectFrame !== +slider.value || !state.people.length) { err.textContent = 'Wait for detection to finish on this frame (or pick a frame with people).'; return; }
  const person = state.people.find((p) => p.index === idx);
  if (!person) { err.textContent = `Type a person index shown on the frame (${state.people.map((p) => p.index).join(', ')}).`; return; }
  const { start, end, step, n } = rangeFrames();
  if (!n) { err.textContent = 'End time is before the start frame.'; return; }
  const hands = $('#with-hands').checked;
  $('#btn-analyse').disabled = true;
  try {
    if (state.mode === 'server') {
      await api(`api/jobs/${state.job.id}/analyse`, {
        method: 'POST',
        body: JSON.stringify({ person: idx, start, end, step, quality: $('#quality').value, conf: +$('#conf').value, max_width: +$('#max-width').value, pose: true, hands }),
      });
      state.frames = []; state.params = {};
      state.run = { started: new Date().toISOString(), t0: performance.now(), status: 'running' };
      enterResults(true);
    } else {
      runBrowser({ person: idx, box: person.box, start, end, step, quality: $('#quality').value, maxW: +$('#max-width').value, hands });
    }
  } catch (e) {
    err.textContent = e.message;
  } finally {
    $('#btn-analyse').disabled = false;
  }
}

function runBrowser(p) {
  const m = state.job;
  state.frames = []; state.cancel = false;
  state.run = { started: new Date().toISOString(), t0: performance.now(), status: 'running' };
  state.params = { person: p.person, start: p.start, end: p.end, step: p.step, quality: p.quality, max_width: p.maxW, pose: true, hands: p.hands,
    box: p.box, fps: m.fps, width: m.width, height: m.height, kind: m.kind, name: m.name, engine: 'browser' };
  enterResults(true);
  const total = Math.floor((p.end - p.start) / p.step) + 1;
  let status = 'loading models';
  const show = (n, fps) => {
    setProgress($('#run-progress'), n / total, `${status === 'tracking' ? 'Tracking' : status} · ${nf.format(n)} / ${nf.format(total)} frames${fps ? ` · ${fps.toFixed(1)} fps` : ''}`);
    updateRunStatus();
  };
  show(0);
  withSrc(() => engine.analyse(state.src, {
    start: p.start, end: p.end, step: p.step, box: p.box, quality: p.quality, maxW: p.maxW, hands: p.hands,
    cancelled: () => state.cancel,
    onStatus: (s) => { status = s; show(state.frames.length); },
    onFrame: (rec, n, fps) => { state.frames.push(rec); show(n, fps); refreshSoon(900); },
  })).then(() => {
    state.running = false;
    state.run.status = state.cancel ? 'stopped' : 'done';
    setProgress($('#run-progress'), 1, state.cancel ? `Stopped at ${nf.format(state.frames.length)} frames` : `Done · ${nf.format(state.frames.length)} frames analysed`);
  }).catch((e) => {
    state.running = false;
    state.run.status = 'error'; state.run.error = String(e.message || e);
    setProgress($('#run-progress'), 0, `Error: ${e.message}`);
  }).finally(() => {
    $('#btn-cancel').hidden = true;
    state.run.seconds = +((performance.now() - state.run.t0) / 1000).toFixed(1);
    if (!$('#step-results').hidden) { refresh(); submitRun('analysis'); } else refreshSoon(0);
  });
}

// --------------------------------------------------------------------------- //
// Step 3: live results
// --------------------------------------------------------------------------- //
let pollTimer = null;
function enterResults(running) {
  showStep('results');
  $('#study-name').textContent = state.job.name;
  state.running = running;
  $('#btn-save').hidden = state.mode !== 'server';
  $('#btn-reselect').hidden = state.mode === 'file';
  $('#btn-cancel').hidden = !running;
  buildOutputCards();
  if (state.mode === 'server') poll();
  else {
    if (!running) setProgress($('#run-progress'), 1, `Opened saved study · ${nf.format(state.frames.length)} frames`);
    updateRunStatus(); refreshSoon(0);
  }
}

function updateRunStatus() {
  const found = state.frames.filter((f) => f.c).length, hands = state.frames.filter((f) => f.h && f.h.length).length;
  const p = state.params;
  $('#run-status').textContent = `person found in ${nf.format(found)} of ${nf.format(state.frames.length)} frames · hands in ${nf.format(hands)}`
    + (p.width ? ` · ${p.width}×${p.height} @ ${(+p.fps).toFixed(2)} fps` : '');
}

async function poll() {
  clearTimeout(pollTimer);
  try {
    const r = await api(`api/jobs/${state.job.id}/status?since=${state.frames.length}`);
    if (r.since === state.frames.length) state.frames.push(...r.frames);
    state.params = r.params;
    const wasRunning = state.running;
    state.running = r.state === 'running';
    const frac = r.total ? r.done / r.total : (r.state === 'done' ? 1 : 0);
    let text;
    if (state.running) text = `${r.message === 'tracking' ? 'Tracking' : r.message} · ${nf.format(r.done)} / ${nf.format(r.total)} frames · ${r.fps} fps`;
    else if (r.state === 'error') text = `Error: ${r.error}`;
    else if (r.state === 'cancelled') text = `Stopped at ${nf.format(r.done)} frames`;
    else text = `Done · ${nf.format(state.frames.length)} frames analysed`;
    setProgress($('#run-progress'), frac, text);
    updateRunStatus();
    $('#btn-cancel').hidden = !state.running;
    refreshSoon(state.running ? 900 : 0);
    if (state.running) pollTimer = setTimeout(poll, 700);
    else if (wasRunning && state.run) {
      Object.assign(state.run, { status: r.state === 'cancelled' ? 'stopped' : r.state, error: r.error || undefined,
        seconds: +((performance.now() - state.run.t0) / 1000).toFixed(1) });
      refresh(); submitRun('analysis');
    }
  } catch (e) {
    setProgress($('#run-progress'), 0, `Lost contact with the server: ${e.message}`);
    pollTimer = setTimeout(poll, 3000);
  }
}
$('#btn-cancel').addEventListener('click', () => {
  if (state.mode === 'server') api(`api/jobs/${state.job.id}/cancel`, { method: 'POST' });
  else state.cancel = true;
});
$('#btn-reselect').addEventListener('click', async () => {
  if (state.running) {
    if (state.mode === 'server') await api(`api/jobs/${state.job.id}/cancel`, { method: 'POST' });
    else state.cancel = true;
  }
  clearTimeout(pollTimer);
  enterSelect(state.job, state.params.start || 0);
});
$('#btn-upload2').addEventListener('click', goHome);

// Output cards
function buildOutputCards() {
  const wrap = $('#out-grid'); wrap.innerHTML = '';
  for (const o of OUTPUTS) {
    const el = document.createElement('div');
    el.className = 'out' + (o.hands ? ' hands' : ''); el.dataset.id = o.id;
    el.innerHTML = `<div class="out-head"><div><strong>${o.title}</strong> <span class="dims"></span></div>
      <div class="btns"><button data-f="png">PNG</button><button data-f="jpeg">JPEG</button><button data-f="webp">WebP</button><button data-f="svg">SVG</button></div></div>
      <div class="out-stage" title="Click to enlarge"><canvas></canvas><div class="out-empty" hidden></div></div>`;
    el.querySelectorAll('[data-f]').forEach((b) => (b.onclick = () => exportOne(o, b.dataset.f)));
    el.querySelector('.out-stage').onclick = () => openLightbox(o);
    wrap.appendChild(el);
  }
}

let refreshTimer = null, lastRefresh = 0;
function refreshSoon(throttle = 0) {
  if (refreshTimer) return;
  const wait = Math.max(0, lastRefresh + throttle - performance.now());
  refreshTimer = setTimeout(() => { refreshTimer = null; lastRefresh = performance.now(); refresh(); }, wait);
}

function meta() {
  return { name: state.job?.name, ...state.params };
}

function refresh() {
  if (!state.frames.length && !state.running) return;
  study.update(state.frames, meta(), state.s);
  const scaleExp = +$('#export-scale').value;
  const dpr = window.devicePixelRatio || 1;
  built = {};
  for (const o of OUTPUTS) {
    const el = $(`.out[data-id="${o.id}"]`);
    const canvas = el.querySelector('canvas'), empty = el.querySelector('.out-empty');
    const b = (built[o.id] = study.keys.length ? o.build(study) : null);
    const noPose = o.id.startsWith('skeleton') && !study.valid.some((f) => f.kp);
    if (!b || noPose) {
      built[o.id] = null;
      canvas.hidden = true; empty.hidden = false;
      empty.textContent = !study.keys.length
        ? (state.running ? 'Waiting for the first tracked frames…' : 'The person was not found in the analysed frames.')
        : o.hands ? (state.running ? 'Waiting for hand landmarks…' : 'No hands found (hand tracking was off, or the hands were not visible).')
          : 'No skeleton data.';
      el.querySelector('.dims').textContent = '';
      continue;
    }
    canvas.hidden = false; empty.hidden = true;
    const stage = el.querySelector('.out-stage');
    const cssW = Math.max(200, stage.clientWidth - 20), cssH = window.innerHeight * 0.7;
    const scale = Math.min(cssW / b.W, cssH / b.H) * dpr;
    b.draw(new CanvasPen(canvas, b.W, b.H, scale));
    canvas.style.width = `${(b.W * scale) / dpr}px`;
    const s = safeScale(b.W, b.H, scaleExp);
    el.querySelector('.dims').textContent = `${nf.format(Math.round(b.W * s))} × ${nf.format(Math.round(b.H * s))} px`;
  }
  const n = study.keys.length;
  $('#kf-count').textContent = `${n} keyframe${n === 1 ? '' : 's'} from ${nf.format(study.valid.length)} tracked frames`;
  drawCharts();
  drawAngleChart();
  fillTable();
  if (!$('#lightbox').hidden) renderLightbox();
}
window.addEventListener('resize', debounce(() => { if (!$('#step-results').hidden) refresh(); }, 200));

// --------------------------------------------------------------------------- //
// Controls
// --------------------------------------------------------------------------- //
const pct = (v) => `${Math.round(v * 100)}%`;
const FMT = {
  threshold: (v) => `${v}%`, smooth: (v) => `${v}`, minGap: (v) => `${(+v).toFixed(2)} s`,
  opacity: pct, current: pct, skelOpacity: pct, skelOverlap: pct, trajOpacity: pct,
  lineWidth: (v) => (+v).toFixed(1), passes: (v) => `${v}`, wobble: (v) => (+v).toFixed(1),
  limbWidth: (v) => `${Math.round(v * 100)}%`,
  cols: (v) => (+v ? `${v}` : 'auto'), ghosts: (v) => `${v}`, trajSmooth: (v) => (+v > 1 ? `${v} frames` : 'off'),
};
function syncControls() {
  const s = state.s;
  for (const el of $$('[data-s]')) {
    const k = el.dataset.s;
    if (el.type === 'checkbox') el.checked = !!s[k]; else el.value = s[k];
    const out = $(`[data-o="${k}"]`); if (out) out.textContent = FMT[k] ? FMT[k](s[k]) : s[k];
  }
  $$('[data-bg]').forEach((b) => b.classList.toggle('sel', b.dataset.bg === s.bg));
  $$('[data-show]').forEach((el) => (el.hidden = s.metric !== el.dataset.show));
  $$('[data-show-if]').forEach((el) => (el.hidden = !s[el.dataset.showIf]));
  $$('#traj-joints input').forEach((el) => (el.checked = !!s.trajJoints[el.value]));
}
function changed() { store.set('ms-settings', state.s); syncControls(); refreshSoon(state.running ? 300 : 0); }
for (const el of $$('[data-s]')) {
  el.addEventListener('input', () => {
    const k = el.dataset.s;
    state.s[k] = el.type === 'checkbox' ? el.checked : el.tagName === 'SELECT' ? el.value : +el.value;
    changed();
  });
}
$$('[data-bg]').forEach((b) => b.addEventListener('click', () => { state.s.bg = b.dataset.bg; changed(); }));
const jw = $('#traj-joints');
for (const [k, [lab, name]] of Object.entries(TRAJ)) {
  const l = document.createElement('label'); l.className = 'check';
  l.innerHTML = `<input type="checkbox" value="${k}"> ${lab} · ${name}`;
  l.querySelector('input').addEventListener('input', (e) => { state.s.trajJoints[k] = e.target.checked; changed(); });
  jw.appendChild(l);
}
$('#btn-reset').addEventListener('click', () => { state.s = structuredClone(DEFAULTS); changed(); });
$('#export-scale').addEventListener('change', () => refreshSoon());
syncControls();

// --------------------------------------------------------------------------- //
// Charts (timeline, angles)
// --------------------------------------------------------------------------- //
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
let chartGeom = null, angleGeom = null;

function prepCanvas(c) {
  const dpr = window.devicePixelRatio || 1, r = c.getBoundingClientRect();
  c.width = Math.max(1, Math.round(r.width * dpr)); c.height = Math.max(1, Math.round(r.height * dpr));
  const ctx = c.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

function drawCharts(hoverIdx = null) {
  const ser = study.series || [];
  const ink = cssVar('--ink'), muted = cssVar('--muted'), line = cssVar('--line');
  const [t0, t1] = study.span; const tw = Math.max(1e-6, t1 - t0);
  const L = 44, R = 14;
  {
    const { ctx, w, h } = prepCanvas($('#chart-change'));
    const T = 18, B = 22, X = (t) => L + ((t - t0) / tw) * (w - L - R);
    const maxD = Math.max(state.s.threshold * 1.6, ...ser.map((p) => p.d * 100), 1);
    const Y = (d) => T + (1 - d / maxD) * (h - T - B);
    chartGeom = { X, ser };
    ctx.font = '11px system-ui, sans-serif'; ctx.strokeStyle = line; ctx.lineWidth = 1;
    ctx.fillStyle = cssVar('--soft');
    let g0 = null;
    for (const f of state.frames) {
      if (!f.c && g0 === null) g0 = f.t;
      if (f.c && g0 !== null) { ctx.fillRect(X(g0), T, Math.max(1, X(f.t) - X(g0)), h - T - B); g0 = null; }
    }
    if (g0 !== null) ctx.fillRect(X(g0), T, Math.max(1, X(t1) - X(g0)), h - T - B);
    ctx.fillStyle = muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    const stepY = maxD > 60 ? 20 : maxD > 30 ? 10 : 5;
    for (let v = 0; v <= maxD; v += stepY) {
      ctx.beginPath(); ctx.moveTo(L, Y(v) + 0.5); ctx.lineTo(w - R, Y(v) + 0.5); ctx.stroke();
      ctx.fillText(`${v}%`, L - 6, Y(v));
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const nt = Math.max(2, Math.floor((w - L - R) / 80));
    for (let k = 0; k <= nt; k++) { const t = t0 + (tw * k) / nt; ctx.fillText(`${fmtT(t)} s`, X(t), h - B + 6); }
    ctx.setLineDash([5, 4]); ctx.strokeStyle = ink; ctx.globalAlpha = 0.6;
    ctx.beginPath(); ctx.moveTo(L, Y(state.s.threshold)); ctx.lineTo(w - R, Y(state.s.threshold)); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    ctx.fillStyle = ink; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.fillText(`threshold ${state.s.threshold}%`, w - R, Y(state.s.threshold) - 3);
    ctx.strokeStyle = ink; ctx.lineWidth = 1.5; ctx.beginPath();
    let prevT = null;
    ser.forEach((p, k) => {
      const x = X(p.f.t), y = Y(p.d * 100);
      if (k === 0 || p.f.t - prevT > 3 / (state.params.fps || 30) * (state.params.step || 1)) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      prevT = p.f.t;
    });
    ctx.stroke();
    const keyPts = ser.filter((p) => p.key);
    const every = Math.max(1, Math.ceil((keyPts.length * 22) / (w - L - R)));
    keyPts.forEach((p, j) => {
      const x = X(p.f.t), y = Y(p.d * 100);
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fillStyle = cssVar('--card'); ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = ink; ctx.stroke();
      if (j % every === 0) { ctx.fillStyle = ink; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(String(j + 1), x, Math.max(T - 2, y - 6)); }
    });
    if (hoverIdx != null && ser[hoverIdx]) {
      const p = ser[hoverIdx], x = X(p.f.t);
      ctx.strokeStyle = muted; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, h - B); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, Y(p.d * 100), 4, 0, Math.PI * 2); ctx.fillStyle = ink; ctx.fill();
    }
  }
  {
    const { ctx, w, h } = prepCanvas($('#chart-area'));
    const T = 14, B = 6, X = (t) => L + ((t - t0) / tw) * (w - L - R);
    const areas = ser.map((p) => p.area);
    const lo = Math.min(...areas, Infinity), hi = Math.max(...areas, 0);
    const Y = (a) => T + (1 - (a - lo) / Math.max(1, hi - lo)) * (h - T - B);
    ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('silhouette area (px²)', L, 0);
    if (ser.length) {
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(nf.format(Math.round(hi)), L - 6, Y(hi)); ctx.fillText(nf.format(Math.round(lo)), L - 6, Y(lo));
      ctx.strokeStyle = ink; ctx.globalAlpha = 0.75; ctx.lineWidth = 1.5; ctx.beginPath();
      ser.forEach((p, k) => (k ? ctx.lineTo(X(p.f.t), Y(p.area)) : ctx.moveTo(X(p.f.t), Y(p.area))));
      ctx.stroke(); ctx.globalAlpha = 1;
      for (const p of ser) if (p.key) { ctx.beginPath(); ctx.arc(X(p.f.t), Y(p.area), 2.5, 0, Math.PI * 2); ctx.fillStyle = ink; ctx.fill(); }
      if (hoverIdx != null && ser[hoverIdx]) {
        const x = X(ser[hoverIdx].f.t);
        ctx.strokeStyle = muted; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, h - B); ctx.stroke();
      }
    }
  }
}

function chartHover(e) {
  if (!chartGeom || !chartGeom.ser.length) return;
  const r = e.currentTarget.getBoundingClientRect(), x = e.clientX - r.left;
  const { X, ser } = chartGeom;
  let best = 0, bd = Infinity;
  ser.forEach((p, k) => { const d = Math.abs(X(p.f.t) - x); if (d < bd) { bd = d; best = k; } });
  drawCharts(best);
  const p = ser[best], tip = $('#chart-tip');
  const kn = study.keys.findIndex((k) => k.i === p.f.i);
  tip.innerHTML = `<b>${fmtT(p.f.t)} s</b> · frame ${p.f.i}<br>change ${(p.d * 100).toFixed(1)}% · area ${nf.format(Math.round(p.area))} px²${kn >= 0 ? `<br><b>keyframe ${kn + 1}</b>` : ''}`;
  tip.hidden = false;
  tip.style.left = `${Math.min(r.width - tip.offsetWidth - 4, Math.max(4, X(p.f.t) + 12))}px`;
  tip.style.top = '8px';
}
['#chart-change', '#chart-area'].forEach((s) => {
  $(s).addEventListener('mousemove', chartHover);
  $(s).addEventListener('mouseleave', () => { $('#chart-tip').hidden = true; drawCharts(); });
});

// One angle at a time (one series, one axis), chosen from the menu.
const angleSel = $('#angle-pick');
angleSel.innerHTML = ANGLES.map((a) => `<option value="${a.key}">${a.label}</option>`).join('');
angleSel.value = store.get('ms-angle') || 'wristR';
angleSel.addEventListener('change', () => { store.set('ms-angle', angleSel.value); drawAngleChart(); });

function drawAngleChart(hover = null) {
  const def = ANGLES.find((a) => a.key === angleSel.value) || ANGLES[0];
  $('#angle-help').textContent = def.help;
  const pts = study.valid.map((f) => ({ f, v: anglesOf(f)?.[def.key] })).filter((p) => p.v != null);
  const ink = cssVar('--ink'), muted = cssVar('--muted'), line = cssVar('--line');
  const { ctx, w, h } = prepCanvas($('#chart-angle'));
  const [t0, t1] = study.span, tw = Math.max(1e-6, t1 - t0), L = 44, R = 14, T = 14, B = 22;
  const X = (t) => L + ((t - t0) / tw) * (w - L - R);
  const isGrip = def.unit === '×';
  const hi = isGrip ? Math.max(1.5, ...pts.map((p) => p.v)) : 180;
  const Y = (v) => T + (1 - v / hi) * (h - T - B);
  angleGeom = { X, pts, def };
  ctx.font = '11px system-ui, sans-serif'; ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.fillStyle = muted;
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  const ticks = isGrip ? [0, 0.5, 1, 1.5, 2, 3].filter((v) => v <= hi) : [0, 45, 90, 135, 180];
  for (const v of ticks) { ctx.beginPath(); ctx.moveTo(L, Y(v) + 0.5); ctx.lineTo(w - R, Y(v) + 0.5); ctx.stroke(); ctx.fillText(isGrip ? `${v}×` : `${v}°`, L - 6, Y(v)); }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  const nt = Math.max(2, Math.floor((w - L - R) / 80));
  for (let k = 0; k <= nt; k++) { const t = t0 + (tw * k) / nt; ctx.fillText(`${fmtT(t)} s`, X(t), h - B + 6); }
  if (!pts.length) { ctx.fillText('no data for this measure', (L + w - R) / 2, h / 2 - 8); return; }
  ctx.strokeStyle = ink; ctx.lineWidth = 1.5; ctx.beginPath();
  let prevT = null;
  pts.forEach((p, k) => {
    const x = X(p.f.t), y = Y(p.v);
    if (!k || p.f.t - prevT > 3 / (state.params.fps || 30) * (state.params.step || 1)) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    prevT = p.f.t;
  });
  ctx.stroke();
  for (const k of study.keys) { const v = anglesOf(k)?.[def.key]; if (v != null) { ctx.beginPath(); ctx.arc(X(k.t), Y(v), 3.5, 0, Math.PI * 2); ctx.fillStyle = cssVar('--card'); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = ink; ctx.stroke(); } }
  if (hover != null && pts[hover]) {
    const x = X(pts[hover].f.t);
    ctx.strokeStyle = muted; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, h - B); ctx.stroke();
  }
}
$('#chart-angle').addEventListener('mousemove', (e) => {
  if (!angleGeom || !angleGeom.pts.length) return;
  const r = e.currentTarget.getBoundingClientRect(), x = e.clientX - r.left, { X, pts, def } = angleGeom;
  let best = 0, bd = Infinity;
  pts.forEach((p, k) => { const d = Math.abs(X(p.f.t) - x); if (d < bd) { bd = d; best = k; } });
  drawAngleChart(best);
  const p = pts[best], tip = $('#angle-tip'), kn = study.keys.findIndex((k) => k.i === p.f.i);
  tip.innerHTML = `<b>${fmtT(p.f.t)} s</b> · frame ${p.f.i}<br>${def.label}: ${fmtAngle(def, p.v)}${kn >= 0 ? `<br><b>keyframe ${kn + 1}</b>` : ''}`;
  tip.hidden = false;
  tip.style.left = `${Math.min(r.width - tip.offsetWidth - 4, Math.max(4, X(p.f.t) + 12))}px`;
  tip.style.top = '8px';
});
$('#chart-angle').addEventListener('mouseleave', () => { $('#angle-tip').hidden = true; drawAngleChart(); });

// --------------------------------------------------------------------------- //
// Keyframe table + CSV
// --------------------------------------------------------------------------- //
function centroid(c) {
  let a = 0, cx = 0, cy = 0;
  for (let i = 0; i < c.length; i++) {
    const [x0, y0] = c[i], [x1, y1] = c[(i + 1) % c.length], cr = x0 * y1 - x1 * y0;
    a += cr; cx += (x0 + x1) * cr; cy += (y0 + y1) * cr;
  }
  return a ? [cx / (3 * a), cy / (3 * a)] : c[0];
}
function keyRows() {
  return study.keys.map((k, j) => {
    const prev = j ? study.keys[j - 1] : null, c = centroid(k.c), pc = prev ? centroid(prev.c) : null;
    return {
      n: k.n, frame: k.i, t: k.t, dt: k.dt, area: k.area,
      areaPct: prev ? ((k.area - prev.area) / prev.area) * 100 : null,
      change: j ? k.d * 100 : null, cx: c[0], cy: c[1],
      shift: pc ? Math.hypot(c[0] - pc[0], c[1] - pc[1]) : null, kp: k.kp, h: k.h, ang: anglesOf(k) || {},
    };
  });
}
const BASE_COLS = ['#', 'Frame', 'Time (s)', 'Δt (s)', 'Area (px²)', 'Area vs prev.', 'Change', 'Centre shift (px)'];
function rowCells(r) {
  const fx = (v, d = 1) => (v == null ? '–' : v.toFixed(d));
  return [r.n, r.frame, fmtT(r.t), r.n > 1 ? fmtT(r.dt) : '–', nf.format(r.area),
    r.areaPct == null ? '–' : `${r.areaPct > 0 ? '+' : ''}${fx(r.areaPct)}%`, r.change == null ? '–' : `${fx(r.change)}%`, fx(r.shift),
    ...ANGLES.map((a) => fmtAngle(a, r.ang[a.key]))];
}
function fillTable() {
  $('#kf-table thead').innerHTML = `<tr>${[...BASE_COLS, ...ANGLES.map((a) => a.label)].map((c) => `<th>${c}</th>`).join('')}</tr>`;
  $('#kf-table tbody').innerHTML = keyRows().map((r) => `<tr>${rowCells(r).map((c) => `<td>${c}</td>`).join('')}</tr>`).join('');
  $('#table-note').textContent = (study.s.metric === 'shape'
    ? 'Change = max(pixels gained, pixels lost) vs. the previous keyframe, as % of its area.'
    : 'Change = |area − previous keyframe area| / previous area (smoothed).') + ' Angles are measured in the image plane (2D).';
}
const KP_NAMES = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle'];
const HAND_NAMES = ['wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip', 'index_mcp', 'index_pip', 'index_dip', 'index_tip', 'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
  'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip', 'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'];
function keyframesCsv() {
  const head = ['keyframe', 'frame', 'time_s', 'dt_s', 'area_px2', 'area_change_pct', `change_pct_${study.s.metric}`, 'centroid_x', 'centroid_y', 'centroid_shift_px',
    ...ANGLES.map((a) => (a.unit === '×' ? `${a.key}_palmwidths` : `${a.key}_deg`)),
    ...KP_NAMES.flatMap((n) => [`${n}_x`, `${n}_y`, `${n}_conf`]),
    ...['L', 'R'].flatMap((s) => HAND_NAMES.flatMap((n) => [`${s}hand_${n}_x`, `${s}hand_${n}_y`]))];
  const v = (x, d = 3) => (x == null ? '' : typeof x === 'number' ? +x.toFixed(d) : x);
  const lines = [head.join(',')];
  for (const r of keyRows()) {
    const kp = r.kp ? r.kp.flatMap((k) => [k[0], k[1], k[2]]) : KP_NAMES.flatMap(() => ['', '', '']);
    const hands = ['L', 'R'].flatMap((s) => { const h = (r.h || []).find((q) => q.side === s); return h ? h.p.flat() : HAND_NAMES.flatMap(() => ['', '']); });
    lines.push([r.n, r.frame, v(r.t, 4), v(r.dt, 4), r.area, v(r.areaPct), v(r.change), v(r.cx, 1), v(r.cy, 1), v(r.shift, 1),
      ...ANGLES.map((a) => v(r.ang[a.key], 2)), ...kp, ...hands].join(','));
  }
  return lines.join('\n') + '\n';
}
function rawJson() {
  return JSON.stringify({ source: meta(), settings: state.s, keyframes: study.keys.map((k) => k.i), keypoint_names: KP_NAMES, hand_landmark_names: HAND_NAMES, frames: state.frames });
}
$('#btn-csv').addEventListener('click', () => download(new Blob([keyframesCsv()], { type: 'text/csv' }), `${baseName()}_keyframes.csv`));
$('#btn-json').addEventListener('click', () => download(new Blob([rawJson()], { type: 'application/json' }), `${baseName()}_frames.json`));

// --------------------------------------------------------------------------- //
// Exports
// --------------------------------------------------------------------------- //
function renderRaster(o, fmt, want = null) {
  const b = built[o.id]; if (!b) return Promise.resolve(null);
  const scale = safeScale(b.W, b.H, want || +$('#export-scale').value);
  const c = document.createElement('canvas');
  b.draw(new CanvasPen(c, b.W, b.H, scale));
  const type = { png: 'image/png', jpeg: 'image/jpeg', webp: 'image/webp' }[fmt];
  return new Promise((res) => c.toBlob(res, type, fmt === 'png' ? undefined : fmt === 'webp' ? 1 : 0.95));
}
function renderSvg(o) {
  const b = built[o.id]; if (!b) return null;
  const pen = new SvgPen(b.W, b.H, 1); b.draw(pen); return pen.toString();
}
async function exportOne(o, fmt) {
  if (!built[o.id]) return;
  if (fmt === 'svg') return download(new Blob([renderSvg(o)], { type: 'image/svg+xml' }), `${baseName()}_${o.file}.svg`);
  const blob = await renderRaster(o, fmt);
  if (!blob) return alert('The browser could not create this image (too large). Try a smaller raster size.');
  download(blob, `${baseName()}_${o.file}.${fmt === 'jpeg' ? 'jpg' : fmt}`);
}

function reportHtml(svgs) {
  const m = meta(), s = state.s, rows = keyRows();
  const esc = (t) => String(t).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const fig = OUTPUTS.filter((o) => svgs[o.id]).map((o) => `<figure><figcaption>${o.title}</figcaption>${svgs[o.id].replace(/^<\?xml[^>]*>\s*/, '').replace('<svg ', '<svg style="width:100%;height:auto" ')}</figure>`).join('\n');
  const models = m.engine === 'browser'
    ? `MediaPipe pose landmarker (${esc(m.quality || 'full')}) with segmentation${m.hands ? ', MediaPipe hand landmarker (21 points)' : ''} - in the browser`
    : `YOLOv8 ${m.quality === 'accurate' ? 's' : 'n'}-seg + ByteTrack, YOLOv8n-pose${m.hands ? ', MediaPipe hand landmarker (21 points)' : ''}`;
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Motion study - ${esc(m.name || '')}</title>
<style>body{font:15px/1.5 Georgia,serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#111;background:#fff}
h1{font-size:26px}table{border-collapse:collapse;font:13px system-ui,sans-serif}td,th{border-bottom:1px solid #ddd;padding:4px 10px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}figure{margin:28px 0}figcaption{font:600 14px system-ui,sans-serif;margin-bottom:6px}.wrap{overflow-x:auto}</style></head><body>
<h1>Motion study · ${esc(m.name || '')}</h1>
<table><tbody>
<tr><td>Source</td><td>${m.width}×${m.height} px, ${(+m.fps).toFixed(3)} fps (${esc(m.kind)})</td></tr>
<tr><td>Analysed</td><td>frames ${m.start}–${m.end}, every ${m.step} frame(s) · ${fmtT(study.span[0])}–${fmtT(study.span[1])} s</td></tr>
<tr><td>Tracked</td><td>person ${m.person} (index on frame ${m.start}) · found in ${study.valid.length} of ${state.frames.length} frames</td></tr>
<tr><td>Keyframe rule</td><td>${s.metric === 'shape' ? 'silhouette change, max(pixels gained, pixels lost)' : 'area change'} ≥ ${s.threshold}% of the previous keyframe${s.minGap ? `, min gap ${s.minGap} s` : ''}${s.includeLast ? ', last frame included' : ''}</td></tr>
<tr><td>Keyframes</td><td>${rows.length}</td></tr>
<tr><td>Models</td><td>${models}</td></tr>
<tr><td>Angles</td><td>2D, measured in the image plane. ${ANGLES.map((a) => `${a.label}: ${a.help}`).join('; ')}.</td></tr>
</tbody></table>
${fig}
<h2>Keyframes</h2>
<div class="wrap"><table><thead><tr>${[...BASE_COLS, ...ANGLES.map((a) => a.label)].map((c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>
${rows.map((r) => `<tr>${rowCells(r).map((c) => `<td>${c}</td>`).join('')}</tr>`).join('\n')}
</tbody></table></div></body></html>`;
}

async function allFiles(btn, formats = ['png', 'svg'], scale = null) {
  const files = [], svgs = {};
  for (const o of OUTPUTS) {
    if (!built[o.id]) continue;
    if (btn) btn.textContent = `Rendering ${o.title}…`;
    await new Promise((r) => setTimeout(r, 0));
    for (const fmt of formats.filter((f) => f !== 'svg')) {
      const img = await renderRaster(o, fmt, scale);
      if (img) files.push({ name: `${o.file}.${fmt === 'jpeg' ? 'jpg' : fmt}`, data: img });
    }
    svgs[o.id] = renderSvg(o);
    files.push({ name: `${o.file}.svg`, data: svgs[o.id] });
  }
  files.push({ name: 'report.html', data: reportHtml(svgs) });
  files.push({ name: 'keyframes.csv', data: keyframesCsv() });
  files.push({ name: 'frames.json', data: rawJson() });
  return files;
}

$('#btn-save').addEventListener('click', async () => {
  const btn = $('#btn-save'); btn.disabled = true; const label = btn.textContent;
  try {
    let path = '';
    for (const f of await allFiles(btn)) {
      btn.textContent = `Saving ${f.name}…`;
      const r = await fetch(`api/jobs/${state.job.id}/outputs?name=${encodeURIComponent(f.name)}`, { method: 'POST', body: f.data });
      if (!r.ok) throw new Error(`could not save ${f.name}`);
      path = (await r.json()).path;
    }
    $('#save-note').textContent = `Saved to ${path.replace(/[\\/][^\\/]+$/, '')}`;
  } catch (e) {
    $('#save-note').textContent = e.message;
  } finally { btn.disabled = false; btn.textContent = label; }
});

$('#btn-zip').addEventListener('click', async () => {
  const btn = $('#btn-zip'); btn.disabled = true; const label = btn.textContent;
  try {
    download(await makeZip(await allFiles(btn)), `${baseName()}_motion_study.zip`);
  } finally { btn.disabled = false; btn.textContent = label; }
  submitRun('download'); // the settings the tester ended up with
});

// --------------------------------------------------------------------------- //
// Test-run collection (config.js -> collector/drive_collector.gs on the owner's Google Drive).
// Sends outputs and data only, never the video.
// --------------------------------------------------------------------------- //
function modelsUsed() {
  const p = state.params;
  if (state.mode === 'server' || p.engine === 'server') {
    return { engine: 'server (Python)', segmentation: `YOLOv8${p.quality === 'accurate' ? 's' : 'n'}-seg + ByteTrack`, pose: 'YOLOv8n-pose',
      hands: p.hands ? 'MediaPipe hand_landmarker (Python)' : null };
  }
  return { engine: 'browser (MediaPipe Tasks Vision, CPU / WebAssembly)', runtime: engine.TV,
    pose: engine.MODELS[p.quality || $('#quality').value] || engine.MODELS.full, person_detector: engine.MODELS.person,
    segmenter_fallback: engine.MODELS.segment, hands: p.hands === false ? null : engine.MODELS.hand };
}
function runInfo(kind, extra = {}) {
  const f = state.frames;
  return {
    kind, sent_at: new Date().toISOString(), page: location.href, user_agent: navigator.userAgent,
    screen: `${screen.width}x${screen.height} @${window.devicePixelRatio || 1}x`,
    source: state.job ? { ...state.job } : null, mode: state.mode, models: modelsUsed(),
    analysis_parameters: state.params, run: state.run ? { ...state.run, t0: undefined } : null,
    results: { frames_analysed: f.length, frames_with_person: f.filter((x) => x.c).length,
      frames_with_skeleton: f.filter((x) => x.kp).length, frames_with_hands: f.filter((x) => x.h && x.h.length).length,
      keyframes: study.keys.length, keyframe_frames: study.keys.map((k) => k.i), keyframe_times_s: study.keys.map((k) => k.t) },
    render_settings: state.s, export_scale: +$('#export-scale').value, ...extra,
  };
}
const toB64 = (blob) => new Promise((res, rej) => {
  const r = new FileReader(); r.onload = () => res(String(r.result).split(',')[1]); r.onerror = rej; r.readAsDataURL(blob);
});
let collectQueue = Promise.resolve(), lastSent = '';
async function send(kind, zip) {
  const body = JSON.stringify({ app: 'motion-study', key: COLLECT_KEY, kind, name: baseName(), zip });
  try {
    // text/plain keeps it a "simple" request (no CORS preflight, which Apps Script does not answer)
    const r = await fetch(COLLECT_URL, { method: 'POST', body, headers: { 'Content-Type': 'text/plain;charset=utf-8' } });
    return await r.json();
  } catch {
    await fetch(COLLECT_URL, { method: 'POST', mode: 'no-cors', body }); // reply unreadable: send blind
    return { ok: true, blind: true };
  }
}
function note(t) { const el = $('#collect-status'); if (el) el.textContent = t; }
// Everything: the six outputs in PNG, JPEG, WebP and SVG, report.html, CSV, frames.json, run_info.json.
function submitRun(kind) {
  if (!COLLECT_URL || !state.frames.length || state.mode === 'file') return;
  const sig = JSON.stringify([state.run?.started, state.s, $('#export-scale').value, state.frames.length]);
  if (sig === lastSent) return; // nothing new since the last copy
  lastSent = sig;
  collectQueue = collectQueue.then(async () => {
    try {
      note('Sending a copy of the results to the study organiser…');
      const LIMIT = 30 * 1024 * 1024;
      let files = await allFiles(null, ['png', 'jpeg', 'webp', 'svg']);
      let zip = await makeZip([...files, { name: 'run_info.json', data: JSON.stringify(runInfo(kind), null, 2) }]);
      if (zip.size > LIMIT) { // too big for one request: 1x rasters, PNG + SVG only
        const reduced = 'rasters at 1x, PNG and SVG only (the full set was too large to send)';
        files = await allFiles(null, ['png', 'svg'], 1);
        zip = await makeZip([...files, { name: 'run_info.json', data: JSON.stringify(runInfo(kind, { reduced }), null, 2) }]);
      }
      const r = await send(kind, await toB64(zip));
      note(r.ok ? 'A copy of the results was sent to the study organiser.' : `Could not send the results copy (${r.error}).`);
    } catch (e) {
      note(`Could not send the results copy (${e.message}).`);
    }
  });
}
// Small events without outputs (e.g. nobody detected), so failures show up too.
function submitInfo(kind, extra) {
  if (!COLLECT_URL) return;
  collectQueue = collectQueue.then(async () => {
    try { await send(kind, await toB64(await makeZip([{ name: 'run_info.json', data: JSON.stringify(runInfo(kind, extra), null, 2) }]))); } catch { /* ignore */ }
  });
}

// --------------------------------------------------------------------------- //
// Lightbox
// --------------------------------------------------------------------------- //
let lbOutput = null, lbActual = false;
function openLightbox(o) { if (!built[o.id]) return; lbOutput = o; lbActual = false; $('#lightbox').hidden = false; renderLightbox(); }
function renderLightbox() {
  const b = built[lbOutput?.id]; if (!b) return;
  const body = $('#lb-body'), c = $('#lb-canvas'), dpr = window.devicePixelRatio || 1;
  const want = lbActual ? safeScale(b.W, b.H, +$('#export-scale').value) : Math.min((body.clientWidth - 20) / b.W, (body.clientHeight - 20) / b.H) * dpr;
  const scale = safeScale(b.W, b.H, want);
  b.draw(new CanvasPen(c, b.W, b.H, scale));
  c.style.width = `${(b.W * scale) / (lbActual ? 1 : dpr)}px`;
  body.classList.toggle('actual', lbActual);
  $('#lb-title').textContent = lbOutput.title;
  $('#lb-size').textContent = lbActual ? `${c.width} × ${c.height} px (export size)` : '';
  $('#lb-zoom').textContent = lbActual ? 'Fit to screen' : 'Actual export size';
}
$('#lb-zoom').addEventListener('click', () => { lbActual = !lbActual; renderLightbox(); });
$('#lb-close').addEventListener('click', () => { $('#lightbox').hidden = true; });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('#lightbox').hidden = true; });

// --------------------------------------------------------------------------- //
// Start-up: is the local server there? (#job=<id> reopens a server study)
// --------------------------------------------------------------------------- //
async function route() {
  const m = location.hash.match(/job=([a-f0-9]{12})/);
  if (!m || !state.serverUp) { showStep('upload'); loadRecent(); return; }
  try {
    const info = await api(`api/jobs/${m[1]}`);
    state.mode = 'server'; state.job = info;
    if (info.frames_ready > 0 || info.state === 'running') {
      state.frames = []; enterResults(info.state === 'running');
    } else enterSelect(info, 0);
  } catch (e) {
    location.hash = ''; showStep('upload'); loadRecent();
  }
}
window.addEventListener('hashchange', () => {
  const m = location.hash.match(/job=([a-f0-9]{12})/);
  if (m && m[1] !== state.job?.id) { clearTimeout(pollTimer); route(); }
});

(async () => {
  try {
    const r = await fetch('api/health', { cache: 'no-store' });
    state.serverUp = r.ok && (await r.json()).ok === true;
  } catch { state.serverUp = false; }
  $('#engine-pick').hidden = !state.serverUp;
  if (state.serverUp) {
    const pref = store.get('ms-engine') || 'browser';
    const r = $(`input[name=engine][value=${pref}]`); if (r) r.checked = true;
  }
  route();
})();
