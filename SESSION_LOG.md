# Session log

## 2026-11-11: web interface and static GitHub Pages site (branch web-interface)
- Added `web/` (FastAPI app, `analysis.py` wrapper around `io_utils.process_csv_file`, `demos.py`) and the ES-module
  front end in `web/static/` (pages: home, analyze, tracks, plots, validate, method, manual, about), following the
  shared web kit (KIT.md): `run_web.py` (port 8106), `scripts/build_static.py`, `scripts/screenshots.py`,
  `.github/workflows/pages.yml`, `requirements-web.txt`, `tests/test_web.py`.
- Demo datasets: real tracks in `alfredo_sRg/data/` and synthetic tracks from the repo's generator.
- Factored the generator functions out of `testing/trajectory_generator.py` into `testing/trajectory_sim.py`
  (numpy/pandas only) so the web app does not need matplotlib/seaborn/scikit-learn; the generated trajectories are
  byte-identical (checked by hashing seed-42 output before and after).
- `web/static/engine.js`: JS port of the simple method and both classification rules; `engine.test.mjs` checks it
  against `tests/fixtures/engine_cases.json` (written by `scripts/make_engine_fixture.py` from the Python pipeline).
- No analysis code (metrics.py, classification.py, io_utils.py, cli.py, config.py) was changed.
- Open points: CLI defaults differ from config.py (method tensor, eigenvalue ratio 5.0 vs 20.0); README §3.3 says gaps
  of ≤3 frames are interpolated, but process_csv_file skips missing frames without interpolating;
  `testing/trajectory_generator.py` is still 715 lines (over the 500-line budget; not split here) and `testing/trackPlotter.py` 418.
