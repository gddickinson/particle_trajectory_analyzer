// Formatting, class colours, unit conversion and the per-track column set (shared by tables, plots and CSV export).
import { esc } from './core.js';
import { num } from './engine.js';

export const COLORS = {
  linear_unidirectional: '#2ee6ff', linear_bidirectional: '#ffb02e', non_linear: '#ff4fd8', linear: '#7aa8ff',
  unclassified: '#6f7c99', mobile: '#3dff8f', immobile: '#6f7c99',
};
export const LABEL_COLORS = ['#2ee6ff', '#ffb02e', '#3dff8f', '#ff4fd8', '#b48cff', '#ff8a5c', '#7aa8ff'];
export const NAMES = {
  linear_unidirectional: 'linear unidirectional', linear_bidirectional: 'linear bidirectional', non_linear: 'non-linear',
  linear: 'linear', unclassified: 'unclassified', mobile: 'mobile', immobile: 'immobile',
};
export const nice = s => NAMES[s] || String(s ?? '').replace(/_/g, ' ');
export const cls = (s, color) => `<span class="cls" style="--c:${color || COLORS[s] || '#94a3c6'}">${esc(nice(s))}</span>`;

// Number formatting: NaN/null -> '–', Infinity -> '∞', otherwise 4 significant digits (integers as is).
export function f(v, digits = 4) {
  const x = num(v);
  if (Number.isNaN(x)) return '–';
  if (!Number.isFinite(x)) return x > 0 ? '∞' : '-∞';
  if (Number.isInteger(x) && Math.abs(x) < 1e6) return String(x);
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return x.toExponential(digits - 1);
  return String(Number(x.toPrecision(digits)));
}
export const pct = (a, b) => (b ? `${(100 * a / b).toFixed(1)}%` : '–');

// Units: x/y are in `coord_unit` ('px', 'um', 'nm' or 'sim'); µm per coordinate unit, or null if unknown.
export const COORD_UNITS = { px: 'pixels', um: 'µm', nm: 'nm', sim: 'simulation units' };
export function umPerUnit(u) {
  if (!u) return null;
  if (u.coord_unit === 'um') return 1;
  if (u.coord_unit === 'nm') return 1e-3;
  if (u.coord_unit === 'px' && u.pixel_size_um > 0) return u.pixel_size_um;
  return null;
}
export const unitLabel = u => ({ px: 'px', um: 'µm', nm: 'nm', sim: 'sim. units' }[u?.coord_unit] || 'coord. units');

// Per-track columns for the current result (method + units decide which appear).
export function columns(result) {
  const u = result.units || {}, k = umPerUnit(u), dt = u.frame_interval_s > 0 ? u.frame_interval_s : null;
  const ul = unitLabel(u), simple = result.params.method === 'simple';
  const c = [
    { key: 'track_number', label: 'track', get: t => t.track_number },
    { key: 'n_points', label: 'points', get: t => t.n_points },
  ];
  if (result.has_labels) c.push({ key: 'label', label: 'true motion (label)', get: t => t.label, cls: true, label_col: true });
  c.push({ key: 'mobility_classification', label: 'mobility', get: t => t.mobility_classification, cls: true },
    { key: 'linear_classification', label: 'linear class', get: t => t.linear_classification, cls: true },
    { key: 'sRg', label: 'sRg', get: t => num(t.sRg) },
    { key: 'rg', label: `Rg (${ul})`, get: t => num(t.rg) });
  if (k && u.coord_unit !== 'um') c.push({ key: 'rg_um', label: 'Rg (µm)', get: t => num(t.rg) * k, derived: true });
  c.push({ key: 'mean_step_length', label: `mean step (${ul})`, get: t => num(t.mean_step_length) });
  if (k && u.coord_unit !== 'um') c.push({ key: 'mean_step_um', label: 'mean step (µm)', get: t => num(t.mean_step_length) * k, derived: true });
  c.push({ key: 'directionality_ratio', label: 'Dr', get: t => num(t.directionality_ratio) });
  if (simple) {
    c.push({ key: 'normalized_perpendicular_distance', label: 'Pn', get: t => num(t.normalized_perpendicular_distance) },
      { key: 'mean_perpendicular_distance', label: `mean ⊥ dist. (${ul})`, get: t => num(t.mean_perpendicular_distance) });
  } else {
    c.push({ key: 'eigenvalue_ratio', label: 'λ1/λ2', get: t => num(t.eigenvalue_ratio) },
      { key: 'step_alignment', label: 'step alignment', get: t => num(t.step_alignment) },
      { key: 'asymmetry', label: 'asymmetry', get: t => num(t.asymmetry) },
      { key: 'skewness', label: 'skewness', get: t => num(t.skewness) },
      { key: 'kurtosis', label: 'kurtosis', get: t => num(t.kurtosis) });
  }
  c.push({ key: 'frames', label: 'frames spanned', get: t => t.last_frame - t.first_frame + 1 });
  if (dt) c.push({ key: 'duration_s', label: 'duration (s)', get: t => (t.last_frame - t.first_frame) * dt, derived: true });
  return c;
}

export function cell(col, t) {
  const v = col.get(t);
  if (col.cls) return col.label_col ? esc(nice(v)) : cls(v);
  return typeof v === 'number' ? f(v) : esc(v);
}

// One row per track, full precision, with units in the headers.
export function trackCsv(result) {
  const cols = columns(result);
  const q = s => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  const val = v => (typeof v === 'number' ? (Number.isNaN(v) ? '' : String(v)) : String(v ?? ''));
  const head = cols.map(c => q(c.key + (c.label.includes('(') ? ' ' + c.label.slice(c.label.indexOf('(')) : '')));
  return [head.join(','), ...result.tracks.map(t => cols.map(c => q(val(c.get(t)))).join(','))].join('\n') + '\n';
}

export function provenance(result) {
  const b = result.computed_by;
  if (b === 'python') return '<span class="badge py">computed by the Python pipeline (io_utils.process_csv_file)</span>';
  if (b === 'python+browser') return '<span class="badge js">metrics from Python; classes re-applied in the browser with your thresholds (engine.js, tested against Python)</span>';
  return '<span class="badge js">computed in your browser by engine.js, the tested JS port of the simple method</span>';
}
