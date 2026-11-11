"""FastAPI web app for the Particle Trajectory Analyzer: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8106, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.

The analysis is the repo's own io_utils.process_csv_file (see web/analysis.py); demo datasets are in web/demos.py.
"""
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .analysis import DEFAULTS, ROOT, AnalysisError, analyse, parse_params, read_tracks_csv
from .demos import DEMOS, demo_coords, demo_frame, demo_meta, demo_result

STATIC = Path(__file__).resolve().parent / "static"
TITLE = "Particle Trajectory Analyzer"
MAX_UPLOAD = 200 * 1024 * 1024
DOCS = {"manual": ("User manual", "trajectory_analyzer_manual.md"),
        "readme": ("The geometric method", "README.md")}

app = FastAPI(title=TITLE)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# Recent runs (merged per-localisation DataFrames) for the CLI-format CSV export; small LRU, local app only.
RUNS: "OrderedDict[str, tuple]" = OrderedDict()
MAX_RUNS = 6


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary for the home page: defaults (config.py), demo datasets and their counts."""
    return {
        "title": TITLE,
        "version": "1",
        "defaults": DEFAULTS,
        "demos": [demo_meta(d) for d in DEMOS],
        "docs": [{"id": k, "title": v[0], "file": v[1]} for k, v in DOCS.items()],
    }


@app.get("/api/demo/{demo_id}/coords")
def demo_tracks(demo_id: str):
    """Track coordinates of a demo dataset: {track_number: [frames, xs, ys]}."""
    if demo_id not in DEMOS:
        raise HTTPException(404, "unknown demo")
    return demo_coords(demo_id)


@app.get("/api/demo/{demo_id}/{method}")
def demo(demo_id: str, method: str):
    """A demo dataset analysed by the real pipeline with the config.py defaults and the given method."""
    if demo_id not in DEMOS or method not in ("simple", "tensor"):
        raise HTTPException(404, "unknown demo or method")
    return demo_result(demo_id, method)


@app.get("/api/docs/{name}")
def docs(name: str):
    """A markdown document from the repo, rendered in the browser."""
    if name not in DOCS:
        raise HTTPException(404, "unknown document")
    title, file = DOCS[name]
    return {"id": name, "title": title, "file": file, "markdown": (ROOT / file).read_text(encoding="utf-8")}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile | None = File(None), demo: str = Form(""),
    method: str = Form(""), cutoff_len: str = Form(""), srg_cutoff: str = Form(""),
    linear_directionality: str = Form(""), linear_perpendicular: str = Form(""),
    linear_eigenvalue_ratio: str = Form(""), linear_step_alignment: str = Form(""),
):
    """Run io_utils.process_csv_file on an uploaded track CSV (or a demo dataset) with the given parameters."""
    form = dict(method=method, cutoff_len=cutoff_len, srg_cutoff=srg_cutoff,
                linear_directionality=linear_directionality, linear_perpendicular=linear_perpendicular,
                linear_eigenvalue_ratio=linear_eigenvalue_ratio, linear_step_alignment=linear_step_alignment)
    try:
        params = parse_params(form)
        if file is not None and file.filename:
            data = await file.read(MAX_UPLOAD + 1)
            if len(data) > MAX_UPLOAD:
                raise AnalysisError("File too large (limit 200 MB); use the command line for bigger files.")
            name = Path(file.filename).name
            df = read_tracks_csv(data)
            result, merged = analyse(df, params, name)
            result["source"] = {"id": "upload", "title": name, "file": name, "description": "Uploaded file",
                                "coord_unit": None, "pixel_size_um": None, "frame_interval_s": None,
                                "n_rows": result["n_rows"], "n_tracks": result["n_tracks_total"]}
        elif demo in DEMOS:
            result, merged = analyse(demo_frame(demo), params, DEMOS[demo]["csv_name"], with_coords=False)
            result["source"] = demo_meta(demo)
            result["coords_ref"] = f"/demo/{demo}/coords"
        else:
            raise AnalysisError("Choose a demo dataset or upload a CSV file.")
    except AnalysisError as e:
        raise HTTPException(400, str(e))
    run_id = uuid.uuid4().hex[:12]
    RUNS[run_id] = (Path(result["name"]).stem, merged)
    while len(RUNS) > MAX_RUNS:
        RUNS.popitem(last=False)
    result["run_id"] = run_id
    return result


@app.get("/api/runs/{run_id}/metrics.csv")
def run_csv(run_id: str):
    """The per-localisation output exactly as the CLI writes <name>_metrics.csv (io_utils.save_processed_data)."""
    if run_id not in RUNS:
        raise HTTPException(404, "run not found (the server keeps the last few runs only)")
    stem, merged = RUNS[run_id]
    return Response(merged.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{stem}_metrics.csv"'})


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/docs/manual", "/docs/readme"] + [
    f"/demo/{d}/{m}" for d in DEMOS for m in ("simple", "tensor", "coords")]
