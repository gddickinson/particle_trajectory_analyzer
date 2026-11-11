// Home: hero, real counts from the demo datasets, and cards to the pages.
import { app, esc, isStatic, list, route } from './core.js';
import { getInfo } from './state.js';

route('home', async () => {
  const info = await getInfo();
  const [real, syn] = info.demos;
  const d = info.defaults;
  app().innerHTML = `
  <section class="hero">
    <p class="kicker">Single-particle tracking · geometric classification</p>
    <h1 class="glow">Particle Trajectory Analyzer</h1>
    <p class="lead">Upload a track table (track_number, frame, x, y) and classify every trajectory as <b>mobile or immobile</b>
      from its scaled radius of gyration, and as <b>linear unidirectional, linear bidirectional or non-linear</b> with the
      simple geometric method (directionality ratio and normalized perpendicular distance) or the gyration-tensor method.</p>
    <a class="btn" href="#/analyze">Analyze tracks</a><a class="btn alt" href="#/plots">See the demo plots</a><a class="btn alt" href="#/method">How it works</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>${real.n_tracks.toLocaleString()}</b><span>tracks in the real demo file</span></div>
    <div class="tile"><b>${real.n_rows.toLocaleString()}</b><span>localisations in it</span></div>
    <div class="tile"><b>${syn.n_tracks}</b><span>simulated tracks with known motion type</span></div>
    <div class="tile"><b>${d.srg_cutoff.toFixed(3)}</b><span>default sRg cutoff (mobile ≥)</span></div>
    <div class="tile"><b>${d.linear_directionality} / ${d.linear_perpendicular}</b><span>default Dr / Pn thresholds</span></div>
  </div>
  ${isStatic() ? `<div class="callout">This is the static showcase. The demo datasets were analysed by the repo's Python code when the site was built;
    threshold changes are re-applied in your browser, and an uploaded CSV can be analysed here with the simple method (a JS port
    tested against the Python output). The tensor method on your own files needs the local app (see About).</div>` : ''}
  <h2>Pages</h2>
  <div class="grid">
    <a class="card" href="#/analyze"><b>Analyze</b><span>Choose a demo dataset or upload a CSV, set the method, thresholds and units, run the analysis and export the results.</span></a>
    <a class="card" href="#/tracks"><b>Tracks</b><span>Per-track table of Rg, sRg, step length, directionality and classes, sortable and filterable; open any track to see its path.</span></a>
    <a class="card" href="#/plots"><b>Plots</b><span>Track map coloured by class, histograms of each metric with the thresholds, and the classification plane.</span></a>
    <a class="card" href="#/validate"><b>Validation</b><span>${syn.n_tracks} simulated tracks of five known motion types: move the thresholds and watch the confusion matrix.</span></a>
    <a class="card" href="#/method"><b>The geometric method</b><span>The mathematics from the README: Rg, sRg, Dr, Pn and the classification rules.</span></a>
    <a class="card" href="#/manual"><b>User manual</b><span>Input format, both methods, outputs, command line and troubleshooting.</span></a>
  </div>
  <h2>Demo datasets</h2>
  <div class="grid wide">${list(info.demos, x => `<div class="card"><b>${esc(x.title)}</b><small>${esc(x.file)}</small>
    <span>${esc(x.description)}</span><small>${x.n_tracks.toLocaleString()} tracks · ${x.n_rows.toLocaleString()} localisations${x.pixel_size_um ? ` · pixel size ${x.pixel_size_um} µm` : ''}</small></div>`)}</div>`;
});
