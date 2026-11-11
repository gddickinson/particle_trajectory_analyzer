# Particle Trajectory Analyzer

@INTERFACE.md

- Tests: `python -m pytest -q tests` (analysis + web); web only: `python -m pytest -q tests/test_web.py`.
- Web app: `python run_web.py` (port 8106). Static site: `python scripts/build_static.py --out site`.
- After changing metrics.py / classification.py / io_utils.py, run `python scripts/make_engine_fixture.py` and
  `node web/static/engine.test.mjs`: the JS port in web/static/engine.js must keep matching Python.
- Keep every file under 500 lines; update INTERFACE.md and SESSION_LOG.md.
