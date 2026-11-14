// contour.js - binary mask -> clean outer outline (JS port of motion_sketch's mask pipeline).
// Used for in-browser silhouettes (MediaPipe masks) and for the rough hand outlines.

const DIRS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]; // E SE S SW W NW N NE

// Keep the largest 8-connected component and fill its holes. Returns {bin, area} or null.
export function largestFilled(bin, w, h) {
  const label = new Int32Array(w * h);
  const stack = new Int32Array(w * h);
  let best = 0, bestSize = 0, cur = 0;
  for (let i = 0; i < w * h; i++) {
    if (!bin[i] || label[i]) continue;
    cur++; let sp = 0, size = 0; stack[sp++] = i; label[i] = cur;
    while (sp) {
      const p = stack[--sp]; size++;
      const x = p % w, y = (p / w) | 0;
      for (let d = 0; d < 8; d++) {
        const nx = x + DIRS[d][0], ny = y + DIRS[d][1];
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const q = ny * w + nx;
        if (bin[q] && !label[q]) { label[q] = cur; stack[sp++] = q; }
      }
    }
    if (size > bestSize) { bestSize = size; best = cur; }
  }
  if (!best) return null;
  // Fill holes: background reachable from the border stays background (4-connected).
  const out = new Uint8Array(w * h), seen = new Uint8Array(w * h);
  let sp = 0;
  const push = (q) => { if (!seen[q] && label[q] !== best) { seen[q] = 1; stack[sp++] = q; } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (sp) {
    const p = stack[--sp], x = p % w, y = (p / w) | 0;
    if (x > 0) push(p - 1); if (x < w - 1) push(p + 1); if (y > 0) push(p - w); if (y < h - 1) push(p + w);
  }
  let area = 0;
  for (let i = 0; i < w * h; i++) if (!seen[i]) { out[i] = 1; area++; }
  return { bin: out, area };
}

// Moore-neighbour tracing of the outer boundary of a filled single component.
export function traceOuter(bin, w, h) {
  let s = -1;
  for (let i = 0; i < w * h; i++) if (bin[i]) { s = i; break; }
  if (s < 0) return [];
  const fg = (x, y) => x >= 0 && y >= 0 && x < w && y < h && bin[y * w + x] === 1;
  const sx = s % w, sy = (s / w) | 0;
  const pts = [[sx, sy]];
  let x = sx, y = sy, back = 4; // came from the west
  const maxIter = 4 * w * h + 16;
  for (let it = 0; it < maxIter; it++) {
    let found = false;
    for (let k = 1; k <= 8; k++) {
      const d = (back + k) % 8, nx = x + DIRS[d][0], ny = y + DIRS[d][1];
      if (fg(nx, ny)) {
        const pd = (back + k - 1) % 8, px = x + DIRS[pd][0], py = y + DIRS[pd][1];
        const dx = px - nx, dy = py - ny;
        back = DIRS.findIndex((v) => v[0] === dx && v[1] === dy);
        x = nx; y = ny; found = true; break;
      }
    }
    if (!found) break; // isolated pixel
    if (x === sx && y === sy) break;
    pts.push([x, y]);
  }
  return pts;
}

export function resampleClosed(points, spacing) {
  const n0 = points.length;
  if (n0 < 3) return points;
  const cum = [0];
  for (let i = 0; i < n0; i++) {
    const a = points[i], b = points[(i + 1) % n0];
    cum.push(cum[i] + Math.hypot(b[0] - a[0], b[1] - a[1]));
  }
  const per = cum[n0];
  const n = Math.max(24, Math.round(per / spacing));
  const out = [];
  let j = 0;
  for (let k = 0; k < n; k++) {
    const t = (per * k) / n;
    while (cum[j + 1] < t) j++;
    const a = points[j], b = points[(j + 1) % n0], seg = cum[j + 1] - cum[j] || 1, u = (t - cum[j]) / seg;
    out.push([a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u]);
  }
  return out;
}

export function smoothClosed(points, radiusPx, spacing) {
  const sigma = radiusPx / spacing, n = points.length;
  if (sigma < 0.5 || n < 8) return points;
  const r = Math.min(Math.floor(3 * sigma), Math.floor(n / 2) - 1);
  const k = []; let ks = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-0.5 * (i / sigma) ** 2); k.push(v); ks += v; }
  return points.map((_, i) => {
    let x = 0, y = 0;
    for (let j = -r; j <= r; j++) { const p = points[(i + j + n) % n]; x += p[0] * k[j + r]; y += p[1] * k[j + r]; }
    return [x / ks, y / ks];
  });
}

// Binary mask (1 = person) at mask resolution -> outline in source pixels.
// scale: source px per mask px.  Returns {c, area, box} or null.
export function outlineFromBinary(bin, w, h, scale, { spacing = 4, smooth = 4, minArea = 0 } = {}) {
  const lf = largestFilled(bin, w, h);
  if (!lf) return null;
  const area = lf.area * scale * scale;
  if (area < minArea) return null;
  const raw = traceOuter(lf.bin, w, h);
  if (raw.length < 3) return null;
  const pts = raw.map(([x, y]) => [(x + 0.5) * scale - 0.5, (y + 0.5) * scale - 0.5]);
  const c = smoothClosed(resampleClosed(pts, spacing), smooth, spacing);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of c) { if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y; }
  return { c, area, box: [x0, y0, x1, y1] };
}

export function binaryFromFloat(data, w, h, thr = 0.5) {
  const bin = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) bin[i] = data[i] > thr ? 1 : 0;
  return bin;
}
