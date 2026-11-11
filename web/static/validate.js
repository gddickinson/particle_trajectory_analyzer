// Validation: the synthetic demo (known motion types, from the repo's generator) analysed by the Python pipeline;
// thresholds move live (engine.applyParams re-classifies the Python metrics) and the confusion matrix updates.
import { $, app, esc, list, route, slider } from './core.js';
import { applyParams } from './engine.js';
import { COLORS, cls, f, nice, provenance } from './fmt.js';
import { demoResult, getInfo, setResult } from './state.js';

const OUT = ['immobile', 'linear_unidirectional', 'linear_bidirectional', 'non_linear', 'linear', 'unclassified'];
let method = null;
const chosen = {};

route('validate', async () => {
  const info = await getInfo();
  const syn = info.demos.find(d => d.id === 'synthetic');
  method = method || info.defaults.method;
  app().innerHTML = `<p class="kicker">Validation on simulated tracks</p><h1>Validation</h1>
    <p class="lead">${esc(syn.description)}</p>
    <p>Each track is analysed by the Python pipeline; moving a threshold re-applies the classification rules in the browser to
      those metrics (the JS rules are tested against Python). Rows are the true motion types, columns the result: immobile, or the
      linear class of a mobile track.</p>
    <div class="row"><div><label for="vm">Method</label><select id="vm"><option value="simple">simple (geometric)</option><option value="tensor">tensor</option></select></div>
      <div><button type="button" class="alt" id="vr">Defaults</button><button type="button" class="alt" id="vo">Open in Tracks and Plots</button></div></div>
    <div class="two"><div class="panel" id="vs"></div><div id="vc"></div></div>`;
  $('#vm').value = method;
  $('#vm').onchange = e => { method = e.target.value; draw(info); };
  $('#vr').onclick = () => { for (const k of Object.keys(chosen)) delete chosen[k]; draw(info); };
  await draw(info);
});

async function draw(info) {
  const base = await demoResult('synthetic', method);
  const d = info.defaults;
  const p = { ...base.params };
  for (const [k, v] of Object.entries(chosen)) p[k] = v;
  const host = $('#vs');
  host.innerHTML = '<h3>Thresholds</h3>';
  const ctl = (id, label, min, max, step) => slider(host, { id, label, min, max, step, value: p[id], fmt: v => f(v, 3) }, v => {
    chosen[id] = v; p[id] = v; matrix(base, p);
  });
  ctl('srg_cutoff', 'sRg cutoff', 0.5, 6, 0.01);
  if (method === 'simple') { ctl('linear_directionality', 'Directionality θD', 0, 1, 0.01); ctl('linear_perpendicular', 'Perpendicular θP', 0, 0.5, 0.005); }
  else { ctl('linear_eigenvalue_ratio', 'Eigenvalue ratio cutoff', 1, 60, 0.5); ctl('linear_step_alignment', 'Step alignment cutoff', 0, 1, 0.01); }
  host.insertAdjacentHTML('beforeend', `<p class="small mute">Defaults (config.py): sRg ${f(d.srg_cutoff)}, ${method === 'simple'
    ? `θD ${d.linear_directionality}, θP ${d.linear_perpendicular}` : `eigenvalue ratio ${d.linear_eigenvalue_ratio}, step alignment ${d.linear_step_alignment}`}.</p>`);
  $('#vo').onclick = () => { setResult({ ...applyParams(base, p), units: base.units }); location.hash = '#/plots'; };
  matrix(base, p);
}

function matrix(base, p) {
  const r = applyParams(base, p);
  const types = [...new Set(r.tracks.map(t => t.label))];
  const out = t => (t.mobility_classification === 'mobile' ? t.linear_classification : t.mobility_classification);
  const cols = OUT.filter(c => r.tracks.some(t => out(t) === c) || ['immobile', 'linear_unidirectional', 'linear_bidirectional', 'non_linear'].includes(c));
  const n = (ty, c) => r.tracks.filter(t => t.label === ty && out(t) === c).length;
  const shade = (k, frac) => `background:color-mix(in srgb, ${COLORS[k]} ${Math.round(45 * frac)}%, transparent)`;
  $('#vc').innerHTML = `${provenance(r)}
    <div class="scroll"><table class="cm"><thead><tr><th>true motion type</th>${list(cols, c => `<th>${cls(c)}</th>`)}</tr></thead>
    <tbody>${list(types, ty => {
      const total = r.tracks.filter(t => t.label === ty).length;
      return `<tr><td>${esc(nice(ty))}</td>${list(cols, c => { const v = n(ty, c); return `<td class="n"><span style="${shade(c, v / (total || 1))}">${v}</span></td>`; })}</tr>`;
    })}</tbody></table></div>
    <p class="small mute">${r.tracks.length} tracks, ${r.tracks[0]?.n_points} points each. The repo does not define an expected class for every
      motion type (for example directed random motion), so this shows agreement rather than an accuracy score.</p>`;
}
