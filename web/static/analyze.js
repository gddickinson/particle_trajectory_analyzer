// Analyze: data source, method, thresholds and units -> run the analysis (Python server, or the static fallbacks in
// static_api.js) -> summary, statistics and exports.
import { $, api, app, download, esc, isStatic, list, route } from './core.js';
import { summarize } from './engine.js';
import { COORD_UNITS, cls, f, pct, provenance, trackCsv } from './fmt.js';
import { getInfo, getResult, setResult, sourceLine, unitsFor } from './state.js';

const num = (id, label, unit, attrs, hint = '') => `<label for="${id}">${label} <span class="unit">${unit}</span></label>
  <input id="${id}" name="${id}" type="number" ${attrs}>${hint ? `<p class="hint">${hint}</p>` : ''}`;

route('analyze', async () => {
  const [info, result] = await Promise.all([getInfo(), getResult()]);
  const d = info.defaults;
  app().innerHTML = `<p class="kicker">Run the analysis</p><h1>Analyze tracks</h1>
  <p class="lead">The analysis is the repo's <code>io_utils.process_csv_file</code>, the same call as
    <code>python cli.py --file data.csv</code>. Defaults come from <code>config.py</code>.</p>
  <div class="side">
  <form class="params panel" id="pf" novalidate>
    <fieldset class="src"><legend>Data</legend>
      ${list(info.demos, x => `<label><input type="radio" name="source" value="${x.id}"> <span>${esc(x.title)}<br><small class="mute">${x.n_tracks.toLocaleString()} tracks</small></span></label>`)}
      <label><input type="radio" name="source" value="upload"> <span>Upload a CSV file</span></label>
      <div id="upwrap" class="hide"><label for="file">Track file (.csv with track_number, frame, x, y)</label>
        <input id="file" name="file" type="file" accept=".csv,text/csv"></div>
    </fieldset>
    <fieldset><legend>Method</legend>
      <label for="method">Rg and linearity method</label>
      <select id="method" name="method"><option value="simple">simple (geometric)</option><option value="tensor">tensor (gyration tensor)</option></select>
      ${num('cutoff_len', 'Minimum track length', 'localisations', 'min="1" step="1"', 'Tracks with fewer rows are skipped.')}
      ${num('srg_cutoff', 'sRg cutoff', 'dimensionless', 'min="0" step="any"', 'sRg below this is immobile.')}
    </fieldset>
    <fieldset id="fs-simple"><legend>Simple method thresholds</legend>
      ${num('linear_directionality', 'Directionality θD', 'Dr, 0–1', 'min="0" max="1" step="any"')}
      ${num('linear_perpendicular', 'Perpendicular θP', 'Pn, dimensionless', 'min="0" step="any"')}
    </fieldset>
    <fieldset id="fs-tensor"><legend>Tensor method thresholds</legend>
      ${num('linear_eigenvalue_ratio', 'Eigenvalue ratio', 'λ1/λ2', 'min="0" step="any"')}
      ${num('linear_step_alignment', 'Step alignment', '0–1', 'min="0" max="1" step="any"')}
    </fieldset>
    <fieldset><legend>Units (display only)</legend>
      <label for="coord_unit">x / y columns are in</label>
      <select id="coord_unit">${list(Object.entries(COORD_UNITS), ([k, v]) => `<option value="${k}">${v}</option>`)}</select>
      ${num('pixel_size_um', 'Pixel size', 'µm / pixel', 'min="0" step="any" placeholder="e.g. 0.108"')}
      ${num('frame_interval_s', 'Frame interval', 's / frame', 'min="0" step="any" placeholder="not set"',
        'sRg, Dr, Pn and the classes are dimensionless: units only convert Rg, step length and duration.')}
    </fieldset>
    <button type="submit" id="run">Run analysis</button><button type="button" class="alt" id="reset">Defaults</button>
    <p id="msg" class="small" role="status"></p>
  </form>
  <div id="out"></div></div>`;

  const form = $('#pf');
  const set = (id, v) => { const el = $('#' + id); el.value = v ?? ''; };
  const fill = (p, u, src) => {
    for (const k of Object.keys(d)) set(k, p[k]);
    set('coord_unit', u.coord_unit); set('pixel_size_um', u.pixel_size_um); set('frame_interval_s', u.frame_interval_s);
    const r = form.querySelector(`input[name=source][value="${src}"]`) || form.querySelector('input[name=source]');
    r.checked = true;
    toggle();
  };
  const toggle = () => {
    const m = $('#method').value, src = form.source.value;
    $('#fs-simple').classList.toggle('hide', m !== 'simple');
    $('#fs-tensor').classList.toggle('hide', m !== 'tensor');
    $('#upwrap').classList.toggle('hide', src !== 'upload');
    $('#pixel_size_um').disabled = $('#coord_unit').value !== 'px';
  };
  const units = () => ({ coord_unit: $('#coord_unit').value, pixel_size_um: +$('#pixel_size_um').value || null,
    frame_interval_s: +$('#frame_interval_s').value || null });

  fill(result.params, result.units, result.source?.id || 'upload');
  form.addEventListener('change', async e => {
    if (e.target.name === 'source' && e.target.value !== 'upload') {
      const u = unitsFor(info.demos.find(x => x.id === e.target.value));
      set('coord_unit', u.coord_unit); set('pixel_size_um', u.pixel_size_um); set('frame_interval_s', u.frame_interval_s);
    }
    toggle();
    if (['coord_unit', 'pixel_size_um', 'frame_interval_s'].includes(e.target.id)) { const cur = await getResult(); cur.units = units(); show(cur); }
  });
  $('#reset').onclick = () => fill(d, units(), form.source.value);

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const fd = new FormData();
    const src = form.source.value;
    if (src === 'upload') {
      const file = $('#file').files[0];
      if (!file) { $('#msg').innerHTML = '<span class="err">Choose a CSV file first.</span>'; return; }
      fd.append('file', file);
    } else fd.append('demo', src);
    for (const k of Object.keys(d)) {
      const v = $('#' + k).value;
      if (v === '') { $('#msg').innerHTML = `<span class="err">${esc(k)} is empty.</span>`; return; }
      fd.append(k, v);
    }
    $('#run').disabled = true;
    $('#msg').textContent = 'Running...';
    try {
      const r = await api('/analyze', { body: fd });
      r.units = units();
      setResult(r);
      $('#msg').innerHTML = `<span class="ok">Done: ${r.tracks.length} tracks analysed.</span>`;
      show(r);
    } catch (err) {
      $('#msg').innerHTML = `<span class="err">${esc(err.message)}</span>`;
    } finally { $('#run').disabled = false; }
  });
  show(result);
});

function show(r) {
  const s = summarize(r.tracks), lin = ['linear_unidirectional', 'linear_bidirectional', 'non_linear', 'linear', 'unclassified'].filter(k => s[k]);
  const cross = k => ['mobile', 'immobile'].map(m => r.tracks.filter(t => t.mobility_classification === m && t.linear_classification === k).length);
  const csvOk = r.run_id && !isStatic();
  $('#out').innerHTML = `${provenance(r)}
  <p class="small mute">${esc(sourceLine(r))}</p>
  <div class="tiles">
    <div class="tile"><b>${s.n.toLocaleString()}</b><span>tracks analysed (≥ ${r.params.cutoff_len} points)</span></div>
    <div class="tile"><b>${pct(s.mobile, s.n)}</b><span>mobile (sRg ≥ ${f(r.params.srg_cutoff)})</span></div>
    <div class="tile"><b>${pct(s.mobile_linear, s.mobile)}</b><span>of mobile tracks linear</span></div>
    <div class="tile"><b>${cross('linear_unidirectional')[0]} / ${cross('linear_bidirectional')[0]}</b><span>mobile unidirectional / bidirectional</span></div>
  </div>
  <div class="scroll"><table><thead><tr><th>linear class</th><th>mobile</th><th>immobile</th><th>all</th></tr></thead><tbody>
    ${list(lin, k => { const [m, i] = cross(k); return `<tr><td>${cls(k)}</td><td>${m}</td><td>${i}</td><td>${s[k]}</td></tr>`; })}
  </tbody></table></div>
  ${r.params.method === 'tensor' ? '<p class="small mute">Tensor method: linear tracks are split into unidirectional / bidirectional at Dr ≥ 0.7 (fixed in classification.py).</p>' : ''}
  <h3>Export</h3>
  <button type="button" id="x-tracks">Per-track CSV</button>
  <button type="button" class="alt" id="x-cli" ${csvOk ? '' : 'disabled'}>CLI-format _metrics.csv</button>
  <button type="button" class="alt" id="x-json">Result JSON</button>
  ${csvOk ? '' : `<p class="small mute">${isStatic() ? 'The per-localisation _metrics.csv (as the CLI writes it) needs the local app.' : 'Run the analysis to enable the per-localisation _metrics.csv (as the CLI writes it).'}</p>`}
  ${r.stats ? `<details><summary>Statistics as written by <code>--generate-stats</code> (io_utils.collect_file_statistics)</summary>
    <p class="small mute">As in the CLI's experiment_statistics.csv: total_tracks counts every track in the file, analysed_tracks those that met the length cutoff (short_tracks the rest); mobility percentages are of analysed tracks, linear percentages of mobile tracks.</p>
    <div class="scroll"><table><tbody>${list(Object.entries(r.stats), ([k, v]) => `<tr><td>${esc(k)}</td><td class="mono">${f(v, 5)}</td></tr>`)}</tbody></table></div></details>`
    : '<p class="small mute">Python statistics (collect_file_statistics) are shown for Python runs with unchanged parameters.</p>'}
  ${r.log?.length ? `<details><summary>Pipeline log</summary><pre>${esc(r.log.join('\n'))}</pre></details>` : ''}
  <p><a href="#/tracks">Per-track table →</a> · <a href="#/plots">Plots →</a></p>`;
  const stem = (r.name || 'tracks').replace(/\.csv$/i, '');
  $('#x-tracks').onclick = () => download(`${stem}_tracks_${r.params.method}.csv`, trackCsv(r), 'text/csv');
  $('#x-json').onclick = () => download(`${stem}_${r.params.method}.json`, JSON.stringify({ ...r, coords: undefined }), 'application/json');
  if (csvOk) $('#x-cli').onclick = () => { const a = document.createElement('a'); a.href = `api/runs/${r.run_id}/metrics.csv`; a.click(); };
}
