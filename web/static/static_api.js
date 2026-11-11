// Static-site data layer: answers api() calls from the JSON exported by scripts/build_static.py, and runs what the
// browser can: re-classifying the precomputed demo metrics with new thresholds, and simple-method analysis of an
// uploaded CSV (engine.js, tested against the Python code). Everything else says it needs the local app.
import { analyseSimple, applyParams, parseCsv } from './engine.js';

const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export const NEEDS_SERVER = 'needs the full app running locally (Python server): see the README, "Run the web app"';

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (clean === 'analyze') return analyze(body);
  if (clean.startsWith('runs/')) throw new Error('The CLI-format CSV export ' + NEEDS_SERVER + '.');
  return file(clean);
}

function params(fd) {
  const p = { method: fd.get('method') };
  for (const k of ['cutoff_len', 'srg_cutoff', 'linear_directionality', 'linear_perpendicular', 'linear_eigenvalue_ratio', 'linear_step_alignment']) {
    const v = Number(fd.get(k));
    if (!Number.isFinite(v)) throw new Error(`${k} must be a number`);
    p[k] = v;
  }
  if (!Number.isInteger(p.cutoff_len) || p.cutoff_len < 1) throw new Error('cutoff_len must be a whole number of at least 1');
  return p;
}

async function analyze(fd) {
  const p = params(fd), upload = fd.get('file');
  if (upload && upload.name) {
    if (p.method !== 'simple') throw new Error('The tensor method ' + NEEDS_SERVER + '. The simple method runs in your browser.');
    const r = analyseSimple(parseCsv(await upload.text()), p, upload.name);
    r.source = { id: 'upload', title: upload.name, file: upload.name, description: 'Uploaded file (analysed in your browser)' };
    return r;
  }
  const demo = fd.get('demo');
  const base = await file(`demo/${demo}/${p.method}`);
  return { ...applyParams(base, p), coords_ref: `/demo/${demo}/coords` };
}
