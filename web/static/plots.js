// Plots of the current result, drawn in the browser from the returned JSON: track map (canvas), metric histograms
// with the thresholds (SVG) and the classification plane (SVG scatter).
import { $, app, esc, list, route } from './core.js';
import { drawMap, histogram, scatter } from './charts.js';
import { num } from './engine.js';
import { COLORS, LABEL_COLORS, cls, nice, provenance, umPerUnit, unitLabel } from './fmt.js';
import { coordsOf, getResult, sourceLine } from './state.js';

let colorBy = 'linear', hideImmobile = false;

route('plots', async () => {
  const r = await getResult();
  if (!r.has_labels && colorBy === 'label') colorBy = 'linear';
  app().innerHTML = `<p class="kicker">Plots</p><h1>Plots</h1>${provenance(r)}
    <p class="small mute">${esc(sourceLine(r))} · <a href="#/analyze">change data or parameters</a></p>
    <h2>Track map</h2>
    <div class="row"><div><label for="cb">Colour tracks by</label><select id="cb">
      <option value="linear">linear class</option><option value="mobility">mobility</option>${r.has_labels ? '<option value="label">true motion type (label)</option>' : ''}</select></div>
      <div><label><input type="checkbox" id="hi"> hide immobile tracks</label></div></div>
    <div id="lg" class="legend"></div>
    <div class="mapwrap"><canvas id="map" width="1100" height="600" role="img" aria-label="Map of all analysed tracks, coloured by class"></canvas></div>
    <p class="small mute">Analysed tracks only, in the file's coordinates (image orientation, y down).</p>
    <h2>Metric distributions</h2><div class="charts" id="hists"></div>
    <h2>Classification plane</h2><div class="charts" id="planes"></div>`;
  $('#cb').value = colorBy;
  $('#hi').checked = hideImmobile;
  $('#cb').onchange = e => { colorBy = e.target.value; map(r); };
  $('#hi').onchange = e => { hideImmobile = e.target.checked; map(r); };
  hists(r); planes(r);
  await map(r);
});

function colorFn(r) {
  const labels = [...new Set(r.tracks.map(t => t.label))].filter(x => x !== undefined);
  const lc = Object.fromEntries(labels.map((l, i) => [l, LABEL_COLORS[i % LABEL_COLORS.length]]));
  const key = t => (colorBy === 'mobility' ? t.mobility_classification : colorBy === 'label' ? t.label : t.linear_classification);
  const col = t => (colorBy === 'label' ? lc[t.label] : COLORS[key(t)]);
  const byId = new Map(r.tracks.map(t => [String(t.track_number), t]));
  const legend = [...new Set(r.tracks.map(key))].map(k => cls(k, colorBy === 'label' ? lc[k] : COLORS[k]));
  return { legend, of: id => { const t = byId.get(id); return t && !(hideImmobile && t.mobility_classification === 'immobile') ? col(t) : null; } };
}

async function map(r) {
  const c = colorFn(r);
  $('#lg').innerHTML = c.legend.join('');
  const canvas = $('#map');
  if (!canvas) return;
  const coords = await coordsOf(r);
  drawMap(canvas, coords, c.of);
}

const vals = (r, k, scale = 1) => r.tracks.map(t => num(t[k]) * scale);
const panel = (title, body, note = '') => `<div class="panel"><h3>${esc(title)}</h3>${body}${note ? `<p class="small mute">${note}</p>` : ''}</div>`;

function hists(r) {
  const p = r.params, k = umPerUnit(r.units), ul = k ? 'µm' : unitLabel(r.units), s = k || 1;
  const out = [
    panel('Scaled radius of gyration', histogram(vals(r, 'sRg'), { xlabel: 'sRg', marks: [{ x: p.srg_cutoff, label: 'cutoff' }] }),
      'Immobile below the cutoff, mobile at or above it.'),
    panel('Radius of gyration', histogram(vals(r, 'rg', s), { xlabel: `Rg (${ul})` })),
    panel('Mean step length', histogram(vals(r, 'mean_step_length', s), { xlabel: `mean step (${ul})` })),
    panel('Directionality ratio', histogram(vals(r, 'directionality_ratio'), { xlabel: 'Dr', range: [0, 1],
      marks: p.method === 'simple' ? [{ x: p.linear_directionality, label: 'θD' }] : [{ x: 0.7, label: '0.7' }] })),
  ];
  if (p.method === 'simple') {
    out.push(panel('Normalized perpendicular distance', histogram(vals(r, 'normalized_perpendicular_distance'), { xlabel: 'Pn', range: [0, Math.max(1, 4 * p.linear_perpendicular)], marks: [{ x: p.linear_perpendicular, label: 'θP' }] })));
  } else {
    out.push(panel('Eigenvalue ratio (log10)', histogram(vals(r, 'eigenvalue_ratio').map(Math.log10), { xlabel: 'log10 λ1/λ2', marks: [{ x: Math.log10(p.linear_eigenvalue_ratio), label: 'cutoff' }] })),
      panel('Step alignment', histogram(vals(r, 'step_alignment'), { xlabel: 'step alignment', range: [0, 1], marks: [{ x: p.linear_step_alignment, label: 'cutoff' }] })));
  }
  out.push(panel('Track length', histogram(r.tracks.map(t => t.n_points), { xlabel: 'localisations per track' })));
  $('#hists').innerHTML = out.join('');
}

function planes(r) {
  const p = r.params, mob = r.tracks.filter(t => t.mobility_classification === 'mobile');
  const pts = (xk, yk) => mob.map(t => ({ x: num(t[xk]), y: num(t[yk]), c: COLORS[t.linear_classification] }));
  const legend = `<div class="legend">${list(['linear_unidirectional', 'linear_bidirectional', 'non_linear', 'linear', 'unclassified'].filter(k => mob.some(t => t.linear_classification === k)), k => cls(k))}</div>`;
  const out = [];
  if (p.method === 'simple') {
    out.push(panel('Dr against Pn (mobile tracks)', legend + scatter(pts('normalized_perpendicular_distance', 'directionality_ratio'),
      { xlabel: 'Pn (normalized perpendicular distance)', ylabel: 'Dr', xrange: [0, Math.max(1, 4 * p.linear_perpendicular)], yrange: [0, 1], lines: { x: [p.linear_perpendicular], y: [p.linear_directionality] } }),
      `Linear when Pn ≤ θP (left of the dashed line); unidirectional when also Dr ≥ θD. ${nice('non_linear')} otherwise.`));
  } else {
    out.push(panel('Step alignment against eigenvalue ratio (mobile tracks)', legend + scatter(pts('eigenvalue_ratio', 'step_alignment'),
      { xlabel: 'λ1/λ2 (log scale)', ylabel: 'step alignment', xlog: true, yrange: [0, 1], lines: { x: [p.linear_eigenvalue_ratio], y: [p.linear_step_alignment] } }),
      'Linear when both are at or above their cutoffs (top right); then split at Dr = 0.7.'));
  }
  out.push(panel('sRg against track length (all tracks)', scatter(r.tracks.map(t => ({ x: t.n_points, y: num(t.sRg), c: COLORS[t.mobility_classification] })),
    { xlabel: 'localisations per track', ylabel: 'sRg', lines: { y: [p.srg_cutoff] } }), 'Green: mobile; grey: immobile.'));
  $('#planes').innerHTML = out.join('');
}
