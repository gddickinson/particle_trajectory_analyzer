// Node test: engine.js reproduces the Python pipeline (tests/fixtures/engine_cases.json, written by
// scripts/make_engine_fixture.py from io_utils.process_csv_file). Run: node web/static/engine.test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { analyseSimple, applyParams, num, parseCsv, summarize } from './engine.js';

const fx = JSON.parse(readFileSync(new URL('../../tests/fixtures/engine_cases.json', import.meta.url), 'utf8'));
const close = (a, b, what) => {
  const x = num(a), y = num(b);
  if (Number.isNaN(y)) return assert.ok(Number.isNaN(x), `${what}: expected NaN, got ${x}`);
  assert.ok(Math.abs(x - y) <= 1e-9 * Math.max(1, Math.abs(y)), `${what}: ${x} vs ${y}`);
};
let checks = 0;

// 1. Simple method from the raw CSV text, as on the static site's upload path.
const table = parseCsv(fx.csv);
const js = analyseSimple(table, fx.params);
const py = fx.runs.simple.result.tracks;
assert.equal(js.tracks.length, py.length, 'number of analysed tracks');
py.forEach((p, i) => {
  const j = js.tracks[i];
  assert.equal(String(j.track_number), String(p.track_number));
  assert.equal(j.n_points, p.n_points);
  for (const k of ['rg', 'mean_step_length', 'sRg', 'directionality_ratio', 'mean_perpendicular_distance', 'normalized_perpendicular_distance']) {
    close(j[k], p[k], `track ${p.track_number} ${k}`); checks++;
  }
  assert.equal(j.mobility_classification, p.mobility_classification, `track ${p.track_number} mobility`);
  assert.equal(j.linear_classification, p.linear_classification, `track ${p.track_number} linear`);
  assert.equal(j.label, p.label);
  checks += 2;
});

// 2. Re-classifying Python metrics with other thresholds matches a Python run with those thresholds (both methods).
for (const method of ['simple', 'tensor']) {
  const run = fx.runs[method];
  for (const alt of run.alt) {
    const r = applyParams(run.result, alt.params);
    assert.equal(r.tracks.length, alt.classes.length);
    r.tracks.forEach((t, i) => {
      assert.deepEqual([t.mobility_classification, t.linear_classification], alt.classes[i], `${method} track ${t.track_number}`);
      checks++;
    });
    assert.equal(r.computed_by, 'python+browser');
  }
  // same params: unchanged, keeps provenance
  const same = applyParams(run.result, run.result.params);
  assert.equal(same.computed_by, 'python');
  // a stricter length cutoff only drops tracks
  const strict = applyParams(run.result, { ...run.result.params, cutoff_len: 30 });
  assert.ok(strict.tracks.every(t => t.n_points >= 30) && strict.tracks.length < run.result.tracks.length);
  assert.throws(() => applyParams(run.result, { ...run.result.params, cutoff_len: 1 }));
}

// 3. Counts and CSV parsing details.
const s = summarize(py);
assert.equal(s.mobile + s.immobile, py.length);
assert.deepEqual(parseCsv('a,b\n"x, ""y""",2\r\n3,\n').rows, [['x, "y"', '2'], ['3', '']]);
assert.throws(() => analyseSimple(parseCsv('track_number,frame,x\n1,0,1\n'), fx.params), /Missing required columns: y/);

console.log(`engine.test.mjs: ok (${checks} checks, ${py.length} tracks)`);
