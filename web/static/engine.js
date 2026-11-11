// Pure JS port of the repo's simple (geometric) method and of both classification rules. Used on the static site
// (uploads with the simple method; re-classifying precomputed Python metrics when thresholds change) and by the
// Validation page. Mirrors metrics.py (get_radius_of_gyration_simple, get_mean_step_length, get_scaled_rg),
// classification.py (classify_linear_motion_simple, classify_linear_motion) and io_utils.process_csv_file.
// engine.test.mjs checks it against values computed by the Python code (tests/fixtures/engine_cases.json).

// JSON numbers from the API: null = NaN, 'Infinity' strings = +/-Infinity.
export const num = v => (v === null || v === undefined ? NaN : typeof v === 'string' ? Number(v) : v);
export const jsonNum = v => (Number.isNaN(v) ? null : v === Infinity ? 'Infinity' : v === -Infinity ? '-Infinity' : v);

export const LINEAR = ['linear_unidirectional', 'linear_bidirectional', 'non_linear', 'linear', 'unclassified'];

// --- metrics.py --------------------------------------------------------------------------------------------------
export function rgSimple(pts) {
  const n = pts.length;
  if (n < 2) return NaN;
  let mx = 0, my = 0, mx2 = 0, my2 = 0;
  for (const [x, y] of pts) { mx += x; my += y; mx2 += x * x; my2 += y * y; }
  mx /= n; my /= n; mx2 /= n; my2 /= n;
  return Math.sqrt((mx2 - mx * mx) + (my2 - my * my));
}

export function meanStep(pts) {
  if (pts.length < 2) return NaN;
  let s = 0;
  for (let i = 1; i < pts.length; i++) s += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  return s / (pts.length - 1);
}

export function scaledRg(rg, step) {
  if (Number.isNaN(rg) || Number.isNaN(step) || step === 0) return NaN;
  return Math.sqrt(Math.PI / 2) * rg / step;
}

// --- classification.py -------------------------------------------------------------------------------------------
export function linearSimpleMetrics(pts) {
  const none = { directionality_ratio: NaN, mean_perpendicular_distance: NaN, normalized_perpendicular_distance: NaN };
  if (pts.length < 3) return none;
  const [sx, sy] = pts[0], [ex, ey] = pts[pts.length - 1];
  const net = Math.hypot(ex - sx, ey - sy);
  let total = 0;
  for (let i = 1; i < pts.length; i++) total += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  if (total === 0) return none;
  const out = { directionality_ratio: net / total, mean_perpendicular_distance: NaN, normalized_perpendicular_distance: NaN };
  if (net > 0) {
    const dx = (ex - sx) / net, dy = (ey - sy) / net;
    let sum = 0;
    for (const [x, y] of pts) {
      const vx = x - sx, vy = y - sy, p = vx * dx + vy * dy;
      sum += Math.hypot(vx - p * dx, vy - p * dy);
    }
    out.mean_perpendicular_distance = sum / pts.length;
    out.normalized_perpendicular_distance = out.mean_perpendicular_distance / net;
  }
  return out;
}

export function classifySimple(dr, pn, dirThr, perpThr) {
  if (Number.isNaN(dr)) return 'unclassified';
  const directional = dr >= dirThr, straight = Number.isNaN(pn) ? false : pn <= perpThr;
  if (directional && straight) return 'linear_unidirectional';
  if (!directional && straight) return 'linear_bidirectional';
  return 'non_linear';
}

export function classifyTensor(eig, align, dr, eigCut, alignCut) {
  if (Number.isNaN(eig) || Number.isNaN(align)) return 'unclassified';
  if (!(eig >= eigCut && align >= alignCut)) return 'non_linear';
  if (Number.isNaN(dr)) return 'linear';
  return dr >= 0.7 ? 'linear_unidirectional' : 'linear_bidirectional';
}

export const mobility = (srg, cut) => (srg < cut ? 'immobile' : 'mobile');

// --- io_utils.process_csv_file (simple method) ----------------------------------------------------------------------
// Points of one track in frame order, as process_csv_file builds its frame matrix: a later row for the same frame
// replaces an earlier one, and frames with a missing coordinate are dropped.
export function trackPoints(frames, xs, ys) {
  const idx = frames.map((f, i) => i).sort((a, b) => frames[a] - frames[b]);
  const byFrame = new Map();
  for (const i of idx) byFrame.set(frames[i], [xs[i], ys[i]]);
  return [...byFrame.keys()].sort((a, b) => a - b).map(f => byFrame.get(f)).filter(([x, y]) => !Number.isNaN(x) && !Number.isNaN(y));
}

export function simpleTrack(frames, xs, ys, p) {
  const pts = trackPoints(frames, xs, ys);
  const rg = rgSimple(pts), step = meanStep(pts), lin = linearSimpleMetrics(pts);
  const srg = scaledRg(rg, step);
  return {
    rg, mean_step_length: step, sRg: srg, mobility_classification: mobility(srg, p.srg_cutoff),
    linear_classification: classifySimple(lin.directionality_ratio, lin.normalized_perpendicular_distance,
      p.linear_directionality, p.linear_perpendicular),
    ...lin,
  };
}

const SIMPLE_COLUMNS = ['rg', 'mean_step_length', 'sRg', 'mobility_classification', 'linear_classification',
  'directionality_ratio', 'mean_perpendicular_distance', 'normalized_perpendicular_distance'];

// Parsed CSV -> coords {track: [frames, xs, ys]} (+ labels from a motion_type column), checking required columns.
export function tracksFromTable({ columns, rows }) {
  const need = ['track_number', 'frame', 'x', 'y'], missing = need.filter(c => !columns.includes(c));
  if (missing.length) throw new Error(`Missing required columns: ${missing.join(', ')}. The analyzer needs track_number, frame, x, y.`);
  if (!rows.length) throw new Error('The file has no rows.');
  const ci = Object.fromEntries(columns.map((c, i) => [c, i])), li = ci.motion_type;
  const coords = {}, labels = {};
  for (const r of rows) {
    const id = r[ci.track_number], f = Number(r[ci.frame]);
    if (!Number.isInteger(f)) throw new Error('The frame column must contain whole numbers (frame indices).');
    const t = coords[id] || (coords[id] = [[], [], []]);
    t[0].push(f); t[1].push(parseNum(r[ci.x])); t[2].push(parseNum(r[ci.y]));
    if (li !== undefined && !(id in labels)) labels[id] = r[li];
  }
  return { coords, labels: li === undefined ? null : labels };
}
const parseNum = s => (s === '' || s === undefined ? NaN : Number(s));

export function sortedIds(obj) {
  const ids = Object.keys(obj);
  const numeric = ids.every(k => k !== '' && !Number.isNaN(Number(k)));
  return numeric ? ids.sort((a, b) => a - b) : ids.sort();
}

// Simple-method analysis of a whole file, in the same JSON shape as the Python API (web/analysis.py).
export function analyseSimple(table, params, name = 'upload.csv') {
  const { coords, labels } = tracksFromTable(table);
  const tracks = [];
  for (const id of sortedIds(coords)) {
    const [f, x, y] = coords[id];
    if (f.length < params.cutoff_len) continue;
    const m = simpleTrack(f, x, y, params);
    const row = { track_number: Number.isNaN(Number(id)) ? id : Number(id), n_points: f.length,
      first_frame: Math.min(...f), last_frame: Math.max(...f) };
    for (const c of SIMPLE_COLUMNS) row[c] = typeof m[c] === 'number' ? jsonNum(m[c]) : m[c];
    if (labels) row.label = labels[id];
    tracks.push(row);
  }
  return { params: { ...params, method: 'simple' }, computed_by: 'browser', name, n_rows: table.rows.length,
    n_tracks_total: Object.keys(coords).length, columns: SIMPLE_COLUMNS, has_labels: !!labels, tracks, stats: null,
    coords, log: [`Found ${tracks.length} tracks with length >= ${params.cutoff_len}`] };
}

// Re-apply thresholds (and a stricter length cutoff) to tracks whose metrics Python already computed.
export function applyParams(result, params) {
  const base = result.params;
  if (params.method !== base.method) throw new Error('Changing the method needs the metrics of the other method.');
  if (params.cutoff_len < base.cutoff_len) {
    throw new Error(`These metrics were computed for tracks of at least ${base.cutoff_len} localisations; a lower cutoff needs a new run.`);
  }
  const tracks = result.tracks.filter(t => t.n_points >= params.cutoff_len).map(t => {
    const lin = params.method === 'simple'
      ? classifySimple(num(t.directionality_ratio), num(t.normalized_perpendicular_distance), params.linear_directionality, params.linear_perpendicular)
      : classifyTensor(num(t.eigenvalue_ratio), num(t.step_alignment), num(t.directionality_ratio), params.linear_eigenvalue_ratio, params.linear_step_alignment);
    return { ...t, mobility_classification: mobility(num(t.sRg), params.srg_cutoff), linear_classification: lin };
  });
  const same = Object.keys(base).every(k => base[k] === params[k]);
  return { ...result, params: { ...params }, tracks, stats: same ? result.stats : null,
    computed_by: same ? result.computed_by : result.computed_by === 'browser' ? 'browser' : 'python+browser' };
}

// Track counts by class (the analysed tracks only).
export function summarize(tracks) {
  const s = { n: tracks.length, mobile: 0, immobile: 0, linear_unidirectional: 0, linear_bidirectional: 0, non_linear: 0, linear: 0, unclassified: 0,
    mobile_linear: 0, mobile_nonlinear: 0 };
  for (const t of tracks) {
    s[t.mobility_classification]++;
    s[t.linear_classification]++;
    if (t.mobility_classification === 'mobile') {
      if (t.linear_classification.startsWith('linear')) s.mobile_linear++;
      else if (t.linear_classification === 'non_linear') s.mobile_nonlinear++;
    }
  }
  return s;
}

// Minimal CSV parser (RFC 4180 quotes; header row first).
export function parseCsv(text) {
  const rows = [];
  let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(cell); cell = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(cell); cell = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
    } else cell += c;
  }
  if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
  if (!rows.length) throw new Error('The file is empty.');
  return { columns: rows[0].map(s => s.trim()), rows: rows.slice(1) };
}
