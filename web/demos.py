"""Demo datasets for the web app and the static site, analysed by the real pipeline (web.analysis.analyse).

- real: alfredo_sRg/data/GB_165_..._tracks.csv, the experimental single-particle tracks already in the repo (the file
  name is all the repo records about the experiment). Its x [nm] / x columns give the pixel size (108 nm); the frame interval is not recorded in the file.
- synthetic: testing/trajectory_sim.generate_trajectories with the generator's own settings (50 tracks x 5 motion types,
  100 points, seed 42). Track numbers restart per motion type in the generator, so they are renumbered here
  (type index * 1000 + track) and the motion_type column is kept as ground truth.
"""
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from .analysis import DEFAULTS, ROOT, analyse

if str(ROOT / "testing") not in sys.path:
    sys.path.insert(0, str(ROOT / "testing"))

REAL_FILE = ROOT / "alfredo_sRg" / "data" / (
    "GB_165_2022_03_01_HTEndothelial_NonBapta_plate1_2_MMStack_Default_bin10_locsID_tracks.csv")


def _pixel_size_um(df: pd.DataFrame):
    """Pixel size in um from the file's own nm columns (x [nm] / x), or None if absent."""
    if "x [nm]" not in df.columns:
        return None
    ok = df["x"].abs() > 1e-9
    return round(float(np.median(df.loc[ok, "x [nm]"] / df.loc[ok, "x"])) / 1000.0, 6)


def load_real() -> pd.DataFrame:
    return pd.read_csv(REAL_FILE)


def load_synthetic() -> pd.DataFrame:
    import trajectory_sim as sim
    trajs = sim.generate_trajectories(sim.N_TRAJECTORIES, sim.N_POINTS)
    frames = sim.trajectories_to_dataframe(trajs)
    parts = []
    for i, (motion, df) in enumerate(frames.items(), start=1):
        df = df.copy()
        df["track_number"] = i * 1000 + df["track_number"]
        parts.append(df[["track_number", "frame", "x", "y", "motion_type"]])
    return pd.concat(parts, ignore_index=True)


def _describe_synthetic():
    import trajectory_sim as sim
    return (f"Simulated by the repo's own generator (testing/trajectory_sim.py, seed {sim.RANDOM_SEED}): "
            f"{sim.N_TRAJECTORIES} tracks of {sim.N_POINTS} points for each of {len(sim.TRAJECTORY_PARAMS)} motion "
            "types (" + ", ".join(m.replace('_', ' ') for m in sim.TRAJECTORY_PARAMS) + "). Coordinates are in "
            "simulation units (step size 1); the true motion type is kept for validation.")


DEMOS = {
    "real": {
        "title": "Real tracks (alfredo_sRg reference data)",
        "file": str(REAL_FILE.relative_to(ROOT)),
        "csv_name": REAL_FILE.name,
        "description": "Experimental single-particle tracks from the repo's reference data (alfredo_sRg/data, recording "
                       "GB_165_2022_03_01_HTEndothelial_NonBapta_plate1_2; the file name is all the repo records about "
                       "the experiment). Coordinates are in pixels; the pixel size comes from the file's x [nm] column.",
        "coord_unit": "px",
        "frame_interval_s": None,
        "load": load_real,
    },
    "synthetic": {
        "title": "Synthetic trajectories (simulated, with ground truth)",
        "file": "testing/trajectory_sim.py",
        "csv_name": "synthetic_trajectories.csv",
        "description": None,
        "coord_unit": "sim",
        "frame_interval_s": None,
        "load": load_synthetic,
    },
}


@lru_cache(maxsize=4)
def demo_frame(demo_id: str) -> pd.DataFrame:
    if demo_id not in DEMOS:
        raise KeyError(demo_id)
    return DEMOS[demo_id]["load"]()


def demo_meta(demo_id: str) -> dict:
    d = DEMOS[demo_id]
    df = demo_frame(demo_id)
    return {
        "id": demo_id, "title": d["title"], "file": d["file"],
        "description": d["description"] or _describe_synthetic(),
        "coord_unit": d["coord_unit"], "pixel_size_um": _pixel_size_um(df),
        "frame_interval_s": d["frame_interval_s"],
        "n_rows": int(len(df)), "n_tracks": int(df["track_number"].nunique()),
    }


@lru_cache(maxsize=8)
def demo_result(demo_id: str, method: str) -> dict:
    """The demo analysed with config.py defaults and the given method (no coordinates: see demo_coords)."""
    params = dict(DEFAULTS, method=method)
    result, _ = analyse(demo_frame(demo_id), params, DEMOS[demo_id]["csv_name"], with_coords=False)
    result["source"] = demo_meta(demo_id)
    return result


@lru_cache(maxsize=4)
def demo_coords(demo_id: str) -> dict:
    from .analysis import track_coords
    return track_coords(demo_frame(demo_id))
