"""Web interface for the Particle Trajectory Analyzer (FastAPI + plain ES-module front end).

Contents:
- app.py       FastAPI app: static front end, JSON API under /api, STATIC_ENDPOINTS for the GitHub Pages export
- analysis.py  runs io_utils.process_csv_file on uploads/demos and reshapes the result to JSON (per-track table,
               coordinates, collect_file_statistics); parameter defaults from config.py
- demos.py     demo datasets: real tracks from alfredo_sRg/data and synthetic tracks from testing/trajectory_sim.py
- static/      the browser front end (see INTERFACE.md)
"""
