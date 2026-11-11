// Client-side plots from the returned JSON: SVG histograms, scatter plots and single tracks; a canvas track map.
import { esc } from './core.js';
import { f } from './fmt.js';

const W = 360, H = 230, M = { l: 44, r: 12, t: 12, b: 38 };
const finite = a => a.filter(Number.isFinite);

function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const i = (sorted.length - 1) * q, lo = Math.floor(i);
  return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (i - lo);
}

function ticks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
  return out;
}

function axes(xs, ys, xl, yl, xfmt = f, yfmt = f) {
  const [x0, x1] = xs.domain, [y0, y1] = ys.domain;
  let s = `<line class="axis" x1="${M.l}" y1="${H - M.b}" x2="${W - M.r}" y2="${H - M.b}"/><line class="axis" x1="${M.l}" y1="${M.t}" x2="${M.l}" y2="${H - M.b}"/>`;
  for (const v of xs.ticks || ticks(x0, x1)) s += `<text x="${xs(v)}" y="${H - M.b + 14}" text-anchor="middle">${xfmt(v)}</text>`;
  for (const v of ys.ticks || ticks(y0, y1, 4)) s += `<text x="${M.l - 5}" y="${ys(v) + 4}" text-anchor="end">${yfmt(v)}</text>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">${esc(xl)}</text>`;
  s += `<text transform="translate(11 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${esc(yl)}</text>`;
  return s;
}

function scale(d0, d1, r0, r1, log = false) {
  const t = log ? Math.log10 : v => v;
  const a = t(d0), b = t(d1);
  const s = v => r0 + (t(v) - a) / ((b - a) || 1) * (r1 - r0);
  s.domain = [d0, d1];
  if (log) s.ticks = ticks(Math.ceil(a), Math.floor(b), 6).filter(Number.isInteger).map(e => 10 ** e);
  return s;
}

// Histogram of values; the x range is clipped to the 0.5-99.5 percentiles unless `range` is given.
// marks: [{x, label}] drawn as dashed threshold lines.
export function histogram(values, { xlabel, bins = 30, range, marks = [], title = '' } = {}) {
  const v = finite(values).sort((a, b) => a - b);
  if (!v.length) return `<p class="mute small">No finite values${title ? ' for ' + esc(title) : ''}.</p>`;
  let [lo, hi] = range || [quantile(v, 0.005), quantile(v, 0.995)];
  if (hi <= lo) { lo -= 0.5; hi += 0.5; }
  const counts = new Array(bins).fill(0), w = (hi - lo) / bins;
  let out = 0;
  for (const x of v) { const i = Math.floor((x - lo) / w); if (i >= 0 && i < bins) counts[i]++; else if (x === hi) counts[bins - 1]++; else out++; }
  const xs = scale(lo, hi, M.l, W - M.r), ys = scale(0, Math.max(...counts), H - M.b, M.t);
  let s = axes(xs, ys, xlabel, 'tracks', f, v2 => String(v2));
  counts.forEach((c, i) => { if (c) s += `<rect class="bar" x="${xs(lo + i * w) + 0.5}" y="${ys(c)}" width="${Math.max(0.5, xs(lo + w) - xs(lo) - 1)}" height="${ys(0) - ys(c)}"><title>${f(lo + i * w)}–${f(lo + (i + 1) * w)}: ${c}</title></rect>`; });
  for (const m of marks) {
    if (m.x < lo || m.x > hi) continue;
    s += `<line class="thr" x1="${xs(m.x)}" x2="${xs(m.x)}" y1="${M.t}" y2="${H - M.b}"/><text class="thrl" x="${xs(m.x) + 4}" y="${M.t + 10}">${esc(m.label)}</text>`;
  }
  const note = out ? `<p class="small mute">${out} of ${v.length} values outside the plotted range.</p>` : '';
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Histogram of ${esc(xlabel)}">${s}</svg>${note}`;
}

// Scatter of points [{x, y, c}] with optional threshold lines {x: [..], y: [..]} and log axes.
export function scatter(points, { xlabel, ylabel, xlog = false, ylog = false, xrange, yrange, lines = {} } = {}) {
  const ok = p => Number.isFinite(p.x) && Number.isFinite(p.y) && (!xlog || p.x > 0) && (!ylog || p.y > 0);
  const pts = points.filter(ok);
  if (!pts.length) return '<p class="mute small">No points to plot.</p>';
  const sx = pts.map(p => p.x).sort((a, b) => a - b), sy = pts.map(p => p.y).sort((a, b) => a - b);
  const xr = xrange || [xlog ? sx[0] : Math.min(0, sx[0]), quantile(sx, 0.995)];
  const yr = yrange || [ylog ? sy[0] : Math.min(0, sy[0]), quantile(sy, 0.995)];
  const xs = scale(xr[0], xr[1], M.l, W - M.r, xlog), ys = scale(yr[0], yr[1], H - M.b, M.t, ylog);
  let s = axes(xs, ys, xlabel, ylabel), hidden = 0;
  for (const p of pts) {
    if (p.x < xr[0] || p.x > xr[1] || p.y < yr[0] || p.y > yr[1]) { hidden++; continue; }
    s += `<circle cx="${xs(p.x).toFixed(1)}" cy="${ys(p.y).toFixed(1)}" r="1.8" fill="${p.c}" fill-opacity=".75"/>`;
  }
  for (const x of lines.x || []) if (x >= xr[0] && x <= xr[1]) s += `<line class="thr" x1="${xs(x)}" x2="${xs(x)}" y1="${M.t}" y2="${H - M.b}"/>`;
  for (const y of lines.y || []) if (y >= yr[0] && y <= yr[1]) s += `<line class="thr" y1="${ys(y)}" y2="${ys(y)}" x1="${M.l}" x2="${W - M.r}"/>`;
  const note = hidden ? `<p class="small mute">${hidden} of ${pts.length} tracks outside the plotted range.</p>` : '';
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(ylabel)} against ${esc(xlabel)}">${s}</svg>${note}`;
}

// One trajectory: path, localisations, start (amber) and end (magenta) and the start-to-end line used by Dr and Pn.
export function trackSvg([frames, xs, ys], { unit = '', size = 420 } = {}) {
  const pts = frames.map((fr, i) => [xs[i], ys[i]]).filter(([x, y]) => x !== null && y !== null);
  if (!pts.length) return '<p class="mute">No coordinates.</p>';
  const x0 = Math.min(...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0]));
  const y0 = Math.min(...pts.map(p => p[1])), y1 = Math.max(...pts.map(p => p[1]));
  const span = Math.max(x1 - x0, y1 - y0) || 1, pad = 30, k = (size - 2 * pad) / span;
  const X = x => pad + (x - x0) * k + ((span - (x1 - x0)) * k) / 2, Y = y => pad + (y - y0) * k + ((span - (y1 - y0)) * k) / 2;
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)} ${Y(p[1]).toFixed(1)}`).join('');
  const [s, e] = [pts[0], pts[pts.length - 1]];
  const bar = 10 ** Math.floor(Math.log10(span / 3)), bl = bar * k;
  return `<svg class="chart" viewBox="0 0 ${size} ${size}" role="img" aria-label="Trajectory plot (image y axis points down)">
    <line x1="${X(s[0])}" y1="${Y(s[1])}" x2="${X(e[0])}" y2="${Y(e[1])}" stroke="#ffd23f" stroke-dasharray="5 4" stroke-width="1.2"/>
    <path d="${d}" fill="none" stroke="#2ee6ff" stroke-width="1.5" stroke-linejoin="round"/>
    ${pts.map(p => `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="2" fill="#7aa8ff"/>`).join('')}
    <circle cx="${X(s[0])}" cy="${Y(s[1])}" r="5" fill="#ffb02e"/><circle cx="${X(e[0])}" cy="${Y(e[1])}" r="5" fill="#ff4fd8"/>
    <line x1="${pad}" y1="${size - 12}" x2="${pad + bl}" y2="${size - 12}" stroke="#e8f1ff" stroke-width="2"/>
    <text x="${pad + bl + 6}" y="${size - 8}">${f(bar)} ${esc(unit)}</text></svg>`;
}

// All tracks on a canvas, image orientation (y down), coloured by colorOf(trackNumber) (null = skip).
export function drawMap(canvas, coords, colorOf) {
  const ids = Object.keys(coords);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const id of ids) {
    const [, xs, ys] = coords[id];
    for (let i = 0; i < xs.length; i++) {
      if (xs[i] === null || ys[i] === null) continue;
      x0 = Math.min(x0, xs[i]); x1 = Math.max(x1, xs[i]); y0 = Math.min(y0, ys[i]); y1 = Math.max(y1, ys[i]);
    }
  }
  const w = canvas.width, pad = 10, k = Math.min((w - 2 * pad) / ((x1 - x0) || 1), 2000);
  canvas.height = Math.max(120, Math.round((y1 - y0) * k + 2 * pad));
  const g = canvas.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0, 0, w, canvas.height);
  g.lineWidth = 1; g.lineJoin = 'round';
  for (const id of ids) {
    const c = colorOf(id);
    if (!c) continue;
    const [, xs, ys] = coords[id];
    g.strokeStyle = c; g.globalAlpha = 0.85; g.beginPath();
    let pen = false;
    for (let i = 0; i < xs.length; i++) {
      if (xs[i] === null || ys[i] === null) continue;
      const X = pad + (xs[i] - x0) * k, Y = pad + (ys[i] - y0) * k;
      if (pen) g.lineTo(X, Y); else { g.moveTo(X, Y); pen = true; }
    }
    g.stroke();
  }
  g.globalAlpha = 1;
  return { x0, x1, y0, y1, k, pad };
}
