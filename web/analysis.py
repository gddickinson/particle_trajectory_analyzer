"""Run the repo's real analysis pipeline for the web app and turn the result into JSON.

The analysis itself is io_utils.process_csv_file (the same call the CLI makes): this module only writes the input to a
CSV file, calls it, and reshapes the merged per-localisation DataFrame into one row per analysed track plus the
coordinates the browser needs to draw the tracks. Statistics come from io_utils.collect_file_statistics (what the CLI's
--generate-stats writes). Nothing here changes a number the pipeline produces.
"""
import contextlib
import io
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from io_utils import collect_file_statistics, process_csv_file, validate_csv_columns  # noqa: E402

# Parameter defaults: the repo's config.py (the single source of defaults for direct execution).
DEFAULTS = {
    "method": config.RG_METHOD,
    "cutoff_len": config.CUTOFF_LEN,
    "srg_cutoff": config.SRG_CUTOFF,
    "linear_directionality": config.LINEAR_DIRECTIONALITY_CUTOFF,
    "linear_perpendicular": config.LINEAR_PERPENDICULAR_CUTOFF,
    "linear_eigenvalue_ratio": config.LINEAR_EIGENVALUE_RATIO_CUTOFF,
    "linear_step_alignment": config.LINEAR_STEP_ALIGNMENT_CUTOFF,
}

# Per-track columns written by process_csv_file (simple and tensor methods, all options on).
METRIC_COLUMNS = [
    "rg", "mean_step_length", "sRg", "mobility_classification", "linear_classification",
    "directionality_ratio", "mean_perpendicular_distance", "normalized_perpendicular_distance",
    "eigenvalue_ratio", "step_alignment", "asymmetry", "skewness", "kurtosis",
]
LABEL_COLUMN = "motion_type"  # ground-truth column written by testing/trajectory_generator.py


class AnalysisError(ValueError):
    """Bad input (message is shown to the user)."""


def clean(v):
    """JSON-safe scalar: NaN -> None, +/-inf -> 'Infinity'/'-Infinity', numpy -> python."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        return f
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def parse_params(form: dict) -> dict:
    """Validate analysis parameters (strings from a form or numbers) against DEFAULTS."""
    p = dict(DEFAULTS)
    for k, v in (form or {}).items():
        if k not in DEFAULTS or v in (None, ""):
            continue
        if k == "method":
            if v not in ("simple", "tensor"):
                raise AnalysisError("method must be 'simple' or 'tensor'")
            p[k] = v
        elif k == "cutoff_len":
            try:
                p[k] = int(v)
            except (TypeError, ValueError):
                raise AnalysisError("cutoff_len must be a whole number of localisations")
            if p[k] < 1:
                raise AnalysisError("cutoff_len must be at least 1")
        else:
            try:
                p[k] = float(v)
            except (TypeError, ValueError):
                raise AnalysisError(f"{k} must be a number")
            if not math.isfinite(p[k]):
                raise AnalysisError(f"{k} must be finite")
    return p


def read_tracks_csv(data) -> pd.DataFrame:
    """Read a CSV (path or bytes) and check the columns the pipeline needs."""
    try:
        df = pd.read_csv(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    except Exception as e:  # pandas raises several types for unreadable files
        raise AnalysisError(f"Could not read the file as CSV: {e}")
    try:
        validate_csv_columns(df)
    except ValueError as e:
        raise AnalysisError(f"{e}. The analyzer needs columns track_number, frame, x, y (see the manual).")
    if df.empty:
        raise AnalysisError("The file has no rows.")
    if not pd.api.types.is_integer_dtype(df["frame"]):
        raise AnalysisError("The frame column must contain whole numbers (frame indices).")
    return df


def run_pipeline(df: pd.DataFrame, params: dict, name: str = "upload"):
    """Write df to a temporary CSV and run io_utils.process_csv_file on it. Returns (merged_df, log)."""
    log = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = tmp
        path = Path(tmp) / (Path(name).stem + ".csv")
        df.to_csv(path, index=False)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            merged = process_csv_file(
                str(path), params["cutoff_len"], params["srg_cutoff"],
                params["linear_eigenvalue_ratio"], params["linear_step_alignment"],
                params["method"], True, True,
                params["linear_directionality"], params["linear_perpendicular"],
            )
    text = log.getvalue().replace(str(Path(tmpdir)) + "/", "")
    if merged is None:
        raise AnalysisError("The analysis failed:\n" + text[-1500:])
    return merged, text


def track_table(merged: pd.DataFrame) -> list:
    """One row per analysed track (tracks shorter than the cutoff have no metrics and are left out)."""
    g = merged.groupby("track_number", sort=True)
    first = g.first()
    size = g.size()
    fmin, fmax = g["frame"].min(), g["frame"].max()
    cols = [c for c in METRIC_COLUMNS if c in first.columns]
    rows = []
    for tn in first.index:
        row = first.loc[tn]
        if pd.isna(row.get("mobility_classification")):
            continue
        r = {"track_number": clean(tn), "n_points": int(size[tn]), "first_frame": clean(fmin[tn]),
             "last_frame": clean(fmax[tn])}
        for c in cols:
            r[c] = clean(row[c])
        if LABEL_COLUMN in first.columns:
            r["label"] = clean(row[LABEL_COLUMN])
        rows.append(r)
    return rows


def track_coords(df: pd.DataFrame, digits: int = 4) -> dict:
    """{track_number: [frames, xs, ys]} sorted by frame, for drawing (rounded to `digits` decimals)."""
    out = {}
    d = df[["track_number", "frame", "x", "y"]].sort_values(["track_number", "frame"], kind="stable")
    for tn, t in d.groupby("track_number", sort=True):
        out[str(clean(tn))] = [t["frame"].astype(int).tolist(),
                               [clean(v) for v in t["x"].round(digits)], [clean(v) for v in t["y"].round(digits)]]
    return out


def file_stats(merged: pd.DataFrame, name: str) -> dict:
    """The per-file statistics the CLI writes with --generate-stats (io_utils.collect_file_statistics)."""
    stats = {}
    collect_file_statistics(merged, name, stats)
    return {k: clean(v) for k, v in stats[name].items()}


def analyse(df: pd.DataFrame, params: dict, name: str, with_coords: bool = True):
    """Full run: returns (result dict, merged DataFrame)."""
    merged, log = run_pipeline(df, params, name)
    result = {
        "params": params,
        "computed_by": "python",
        "name": name,
        "n_rows": int(len(df)),
        "n_tracks_total": int(df["track_number"].nunique()),
        "columns": [c for c in METRIC_COLUMNS if c in merged.columns],
        "has_labels": LABEL_COLUMN in df.columns,
        "tracks": track_table(merged),
        "stats": file_stats(merged, Path(name).stem),
        "log": log.strip().splitlines()[-8:],
    }
    if with_coords:
        result["coords"] = track_coords(df)
    return result, merged
