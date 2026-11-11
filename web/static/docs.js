// Docs pages rendered from the repo's markdown (README.md = the method, trajectory_analyzer_manual.md = the manual)
// and the About page.
import { api, app, esc, isStatic, route } from './core.js';
import { markdown, wireAnchors } from './md.js';
import { getInfo } from './state.js';

async function doc(name, kicker, cut = null) {
  const d = await api('/docs/' + name);
  let src = d.markdown.replace(/!\[[^\]]*\]\([^)]*\)/g, '');
  if (cut && src.includes(cut)) src = src.slice(0, src.indexOf(cut));
  const title = (src.match(/^#\s+(.*)$/m) || [, d.title])[1];
  app().innerHTML = `<p class="kicker">${esc(kicker)} · <span class="mono">${esc(d.file)}</span></p><h1>${esc(title)}</h1>
    <div class="md">${markdown(src, { skipTitle: true })}</div>`;
  wireAnchors(app());
}

route('method', () => doc('readme', 'The geometric method', '## Web interface'));
route('manual', () => doc('manual', 'User manual'));

route('about', async () => {
  const info = await getInfo();
  app().innerHTML = `<p class="kicker">About</p><h1>About</h1>
  <p class="lead">A Python tool for single-particle tracking data: it computes the radius of gyration (Rg), the scaled radius of
    gyration (sRg), the mean step length and linearity metrics for every trajectory, and classifies trajectories as mobile or
    immobile and as linear unidirectional, linear bidirectional or non-linear.</p>
  <h2>This web interface</h2>
  <p>A front end to the same Python code: <code>web/analysis.py</code> calls <code>io_utils.process_csv_file</code> exactly as the
    command line does, and the browser draws the tables and plots from the JSON it returns.
    ${isStatic() ? 'You are on the static showcase: the demo datasets were analysed by the Python code when the site was built.' : 'You are running the local app.'}</p>
  <div class="scroll"><table><thead><tr><th></th><th>Local app</th><th>Static site</th></tr></thead><tbody>
    <tr><td>Demo datasets, default parameters</td><td>Python, live</td><td>Python, precomputed at build time</td></tr>
    <tr><td>Demo datasets, other thresholds or a higher length cutoff</td><td>Python, live</td><td>Python metrics, classes re-applied in the browser</td></tr>
    <tr><td>Your CSV, simple method</td><td>Python, live</td><td>In the browser (engine.js, a JS port tested against Python)</td></tr>
    <tr><td>Your CSV, tensor method</td><td>Python, live</td><td>Needs the local app</td></tr>
    <tr><td>Per-localisation <code>_metrics.csv</code> (CLI format)</td><td>Yes</td><td>Needs the local app</td></tr>
  </tbody></table></div>
  <h2>Run it</h2>
  <h3>Command line (the analysis itself)</h3>
  <pre>pip install -r requirements.txt
python cli.py --file data.csv --method simple
python trajectory_analyzer.py --directory ./data --method tensor --generate-stats</pre>
  <h3>Web app</h3>
  <pre>pip install -r requirements-web.txt
python run_web.py            # http://127.0.0.1:8106</pre>
  <h3>Static site</h3>
  <pre>python scripts/build_static.py --out site   # published by .github/workflows/pages.yml</pre>
  <h2>Defaults</h2>
  <p>From <code>config.py</code>: method ${esc(info.defaults.method)}, minimum length ${info.defaults.cutoff_len},
    sRg cutoff ${info.defaults.srg_cutoff}, θD ${info.defaults.linear_directionality}, θP ${info.defaults.linear_perpendicular},
    eigenvalue ratio ${info.defaults.linear_eigenvalue_ratio}, step alignment ${info.defaults.linear_step_alignment}.
    The command line and <code>io_utils.process_csv_file</code> use the same defaults.</p>
  <h2>Credits</h2>
  <p>The simple Rg and sRg method was adapted from R code by J. Alfredo Freites (alfredo_sRg/); the scaled radius of gyration
    follows Golan and Sherman, Nature Communications 2017. Code by George Dickinson, MIT licence.
    Source: <a href="https://github.com/gddickinson/particle_trajectory_analyzer" target="_blank" rel="noopener">github.com/gddickinson/particle_trajectory_analyzer</a>.</p>`;
});
