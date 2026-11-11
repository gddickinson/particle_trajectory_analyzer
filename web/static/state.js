// The current analysis result, shared by the Analyze, Tracks and Plots pages. Until the user runs an analysis, the
// real demo dataset analysed with the config.py defaults is shown.
import { api } from './core.js';

let info = null, current = null;
export async function getInfo() { return info || (info = await api('/info')); }

// Units travel with the result (they only change displayed lengths and times, never the analysis).
export function unitsFor(source) {
  return { coord_unit: source?.coord_unit || 'px', pixel_size_um: source?.pixel_size_um ?? null,
    frame_interval_s: source?.frame_interval_s ?? null };
}

export async function demoResult(id, method) {
  const r = await api(`/demo/${id}/${method}`);
  return { ...r, coords_ref: `/demo/${id}/coords`, units: unitsFor(r.source) };
}

export function setResult(r) {
  if (!r.units) r.units = unitsFor(r.source);
  current = r;
  return r;
}

export async function getResult() {
  if (current) return current;
  const i = await getInfo();
  return setResult(await demoResult(i.demos[0].id, i.defaults.method));
}

export async function coordsOf(r) {
  if (r.coords) return r.coords;
  if (r.coords_ref) r.coords = await api(r.coords_ref);
  return r.coords || {};
}

// Header line used on result pages.
export function sourceLine(r) {
  const s = r.source || {};
  return `${s.title || r.name} · ${r.tracks.length} tracks analysed (of ${r.n_tracks_total}) · ${r.params.method} method`;
}
