# INTERFACE.md: navigation map

## Analysis (Python, flat modules at the repo root)
| File | Contents |
|------|----------|
| `config.py` | Default parameters (CUTOFF_LEN, SRG_CUTOFF, linear thresholds, RG_METHOD, output options). Single source of defaults for direct execution and the web app. |
| `metrics.py` | `get_radius_of_gyration_simple`, `get_trajectory_metrics_tensor`, `get_mean_step_length`, `get_scaled_rg`, `calculate_step_angle_variation`. |
| `classification.py` | `classify_linear_motion_simple` (Dr, Pn), `classify_linear_motion` (eigenvalue ratio, step alignment). |
| `io_utils.py` | `validate_csv_columns`, `process_csv_file` (the pipeline: per-track metrics merged onto the rows), `save_processed_data`, `collect_file_statistics`, `generate_stats_file`, `process_directory`. Imports metrics + classification. |
| `cli.py` | `run_script`, `main` (argparse / direct execution). Imports io_utils, config. |
| `trajectory_analyzer.py` | Backward-compatible wrapper re-exporting the above. |
| `testing/trajectory_sim.py` | Synthetic trajectory generators (5 motion types, seed 42), `generate_trajectories`, `trajectories_to_dataframe`. numpy/pandas only. |
| `testing/trajectory_generator.py` | Threshold study: uses trajectory_sim, plots with matplotlib/seaborn, ROC with scikit-learn. |
| `testing/trackPlotter.py` | Tkinter viewer for `_metrics.csv` files. |
| `alfredo_sRg/` | Reference R/notebook sRg code and the experimental track data used as the real demo dataset. |
| `tests/` | pytest: `test_metrics.py`, `test_classification.py`, `test_io_utils.py`, `test_web.py`; `fixtures/engine_cases.json`. |

## Web interface (`web/` package, `__WEBMODULE__ = web.app`)
| File | Contents |
|------|----------|
| `run_web.py` | Starts uvicorn on 127.0.0.1:8106. |
| `web/app.py` | FastAPI: `/api/info`, `/api/demo/{id}/{simple,tensor,coords}`, `/api/docs/{manual,readme}`, `POST /api/analyze` (upload or demo + params), `/api/runs/{id}/metrics.csv`; `STATIC_ENDPOINTS`. |
| `web/analysis.py` | `DEFAULTS` (from config.py), `parse_params`, `read_tracks_csv`, `run_pipeline` (calls io_utils.process_csv_file), `track_table`, `track_coords`, `file_stats`, `analyse`, `clean` (JSON-safe numbers). |
| `web/demos.py` | `DEMOS` (real, synthetic), `demo_frame`, `demo_meta`, `demo_result`, `demo_coords`. |
| `scripts/build_static.py` | Static site export (every STATIC_ENDPOINTS GET to JSON). |
| `scripts/screenshots.py` | Playwright smoke test + `docs/img/*.png`. |
| `scripts/make_engine_fixture.py` | Writes `tests/fixtures/engine_cases.json` from the Python pipeline. |

### Front end (`web/static/`, ES modules, no build step)
| File | Contents |
|------|----------|
| `index.html`, `style.css` (shared kit theme), `style_app.css`, `icon.svg` | Shell and styles. |
| `core.js` | Kit core: `api`, router (`route`, `lazy`, `start`), `slider`, `needsServer`, `download`. |
| `main.js` | Registers lazy pages, mode label, random-walk background. |
| `static_api.js` | Static-site answers: exported JSON; `analyze` via engine.js (re-classify demos, simple-method uploads). |
| `engine.js` | Pure JS port: simple metrics, both classification rules, `analyseSimple`, `applyParams`, `summarize`, `parseCsv`. Tested by `engine.test.mjs`. |
| `state.js` | Current result shared across pages; demo loading; units; `coordsOf`. |
| `fmt.js` | Number formatting, class colours, unit conversion, per-track `columns`, `trackCsv`, provenance badge. |
| `charts.js` | SVG `histogram`, `scatter`, `trackSvg`; canvas `drawMap`. |
| `md.js` | Markdown renderer for README/manual. |
| `home.js`, `analyze.js`, `tracks.js`, `plots.js`, `validate.js`, `docs.js` | Pages (`docs.js`: method, manual, about). |
