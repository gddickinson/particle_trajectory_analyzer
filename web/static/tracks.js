// Tracks: sortable, filterable per-track table of the current result; #/tracks/<id> shows one track.
import { $, app, download, esc, go, list, route } from './core.js';
import { trackSvg } from './charts.js';
import { num } from './engine.js';
import { cell, cls, columns, f, nice, provenance, trackCsv, unitLabel } from './fmt.js';
import { coordsOf, getResult, sourceLine } from './state.js';

const PAGE = 50;
const view = { filter: 'all', sort: 'track_number', dir: 1, page: 0, q: '' };

function filtered(r) {
  const cols = columns(r), col = cols.find(c => c.key === view.sort) || cols[0];
  const want = view.filter;
  let rows = r.tracks.filter(t => want === 'all' || t.mobility_classification === want || t.linear_classification === want ||
    (want === 'mobile_linear' && t.mobility_classification === 'mobile' && t.linear_classification.startsWith('linear')));
  if (view.q) rows = rows.filter(t => String(t.track_number).includes(view.q));
  const key = t => { const v = col.get(t); return typeof v === 'number' ? (Number.isNaN(v) ? -Infinity : v) : String(v); };
  return rows.slice().sort((a, b) => { const x = key(a), y = key(b); return (x < y ? -1 : x > y ? 1 : 0) * view.dir; });
}

route('tracks', async ([id]) => {
  const r = await getResult();
  if (id !== undefined) return detail(r, id);
  const opts = ['all', 'mobile', 'immobile', 'mobile_linear', 'linear_unidirectional', 'linear_bidirectional', 'non_linear', 'unclassified'];
  app().innerHTML = `<p class="kicker">Per-track results</p><h1>Tracks</h1>
    ${provenance(r)}<p class="small mute">${esc(sourceLine(r))} · <a href="#/analyze">change data or parameters</a></p>
    <div class="row">
      <div><label for="tf">Show</label><select id="tf">${list(opts, o => `<option value="${o}">${o === 'mobile_linear' ? 'mobile and linear' : nice(o)}</option>`)}</select></div>
      <div><label for="tq">Track number contains</label><input id="tq" type="search" inputmode="numeric"></div>
      <div><button type="button" id="tx">Download this view (CSV)</button></div>
    </div>
    <div id="tt"></div>`;
  $('#tf').value = view.filter; $('#tq').value = view.q;
  $('#tf').onchange = e => { view.filter = e.target.value; view.page = 0; draw(r); };
  $('#tq').oninput = e => { view.q = e.target.value.trim(); view.page = 0; draw(r); };
  $('#tx').onclick = () => download(`tracks_${view.filter}.csv`, trackCsv({ ...r, tracks: filtered(r) }), 'text/csv');
  draw(r);
});

function draw(r) {
  const cols = columns(r), rows = filtered(r), pages = Math.max(1, Math.ceil(rows.length / PAGE));
  view.page = Math.min(view.page, pages - 1);
  const shown = rows.slice(view.page * PAGE, (view.page + 1) * PAGE);
  const arrow = k => (view.sort === k ? (view.dir > 0 ? ' ▲' : ' ▼') : '');
  $('#tt').innerHTML = `<div class="pager"><span class="small mute">${rows.length.toLocaleString()} tracks · page ${view.page + 1} of ${pages}</span>
      <button type="button" class="alt" data-p="-1" ${view.page ? '' : 'disabled'}>Previous</button>
      <button type="button" class="alt" data-p="1" ${view.page < pages - 1 ? '' : 'disabled'}>Next</button></div>
    <div class="scroll"><table class="tbl"><thead><tr>${list(cols, c => `<th scope="col" aria-sort="${view.sort === c.key ? (view.dir > 0 ? 'ascending' : 'descending') : 'none'}"><button type="button" data-k="${c.key}">${esc(c.label)}${arrow(c.key)}</button></th>`)}</tr></thead>
    <tbody>${list(shown, t => `<tr class="click" tabindex="0" data-id="${esc(t.track_number)}">${list(cols, c => `<td>${cell(c, t)}</td>`)}</tr>`)}</tbody></table></div>
    <p class="small mute">Click a row (or press Enter on it) to see the track. Derived unit columns use the units set on the Analyze page.</p>`;
  $('#tt').querySelectorAll('th button').forEach(b => b.onclick = () => {
    view.dir = view.sort === b.dataset.k ? -view.dir : 1; view.sort = b.dataset.k; draw(r);
  });
  $('#tt').querySelectorAll('[data-p]').forEach(b => b.onclick = () => { view.page += +b.dataset.p; draw(r); });
  $('#tt').querySelectorAll('tr[data-id]').forEach(tr => {
    const open = () => go('#/tracks/' + encodeURIComponent(tr.dataset.id));
    tr.onclick = open;
    tr.onkeydown = e => { if (e.key === 'Enter') open(); };
  });
}

async function detail(r, id) {
  const rows = filtered(r), i = rows.findIndex(t => String(t.track_number) === id);
  const t = i >= 0 ? rows[i] : r.tracks.find(x => String(x.track_number) === id);
  if (!t) { app().innerHTML = `<h1>Track ${esc(id)} not found</h1><p><a href="#/tracks">All tracks</a></p>`; return; }
  const coords = (await coordsOf(r))[String(t.track_number)];
  const cols = columns(r), p = r.params;
  const nav = (j, label) => (j >= 0 && j < rows.length ? `<a class="btn alt" href="#/tracks/${encodeURIComponent(rows[j].track_number)}">${label}</a>` : '');
  const why = p.method === 'simple'
    ? `Dr = ${f(t.directionality_ratio)} (${f(t.directionality_ratio) === '–' ? 'not defined' : `${+t.directionality_ratio >= p.linear_directionality ? '≥' : '<'} θD ${f(p.linear_directionality)}`}),
       Pn = ${f(t.normalized_perpendicular_distance)} (${f(t.normalized_perpendicular_distance) === '–' ? 'not defined' : `${+t.normalized_perpendicular_distance <= p.linear_perpendicular ? '≤' : '>'} θP ${f(p.linear_perpendicular)}`})`
    : `λ1/λ2 = ${f(t.eigenvalue_ratio)} (cutoff ${f(p.linear_eigenvalue_ratio)}), step alignment = ${f(t.step_alignment)} (cutoff ${f(p.linear_step_alignment)}), Dr = ${f(t.directionality_ratio)} (0.7)`;
  app().innerHTML = `<p class="kicker"><a href="#/tracks">Tracks</a> / ${esc(r.source?.title || r.name)}</p>
    <h1>Track ${esc(t.track_number)}</h1>
    <p>${cls(t.mobility_classification)} ${cls(t.linear_classification)} ${t.label ? `<span class="tag c">true: ${esc(nice(t.label))}</span>` : ''}</p>
    <div class="two"><div class="panel">${coords ? trackSvg(coords, { unit: unitLabel(r.units) }) : '<p class="mute">No coordinates for this track.</p>'}
      <p class="small mute"><span style="color:#ffb02e">●</span> start · <span style="color:#ff4fd8">●</span> end · dashed: start-to-end line used by Dr and Pn. Image orientation (y down).</p></div>
    <div class="panel"><dl class="kv">${list(cols, c => `<dt>${esc(c.label)}</dt><dd>${cell(c, t)}</dd>`)}</dl>
      <p class="small">sRg = ${f(t.sRg)} is ${Number.isNaN(num(t.sRg)) ? 'not defined' : num(t.sRg) < p.srg_cutoff ? 'below' : 'at or above'} the cutoff ${f(p.srg_cutoff)} → ${esc(t.mobility_classification)}.<br>${why}</p></div></div>
    <p>${nav(i - 1, '← Previous')} ${nav(i + 1, 'Next →')} <a class="btn alt" href="#/tracks">All tracks</a></p>`;
}
