"""Write tests/fixtures/engine_cases.json: inputs and the Python pipeline's outputs for the JS engine test.

web/static/engine.test.mjs runs web/static/engine.js on the same CSV text and checks it reproduces these numbers;
tests/test_web.py checks the fixture still matches what the Python code computes now.

Run: python scripts/make_engine_fixture.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.analysis import DEFAULTS, analyse, clean  # noqa: E402
from web.demos import load_synthetic  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "engine_cases.json"
ALT = [  # alternative thresholds the JS re-classification must reproduce
    dict(srg_cutoff=5.0, linear_directionality=0.6, linear_perpendicular=0.3, linear_eigenvalue_ratio=5.0,
         linear_step_alignment=0.6),
    dict(srg_cutoff=20.0, linear_directionality=0.95, linear_perpendicular=0.05, linear_eigenvalue_ratio=50.0,
         linear_step_alignment=0.85),
]


def input_frame() -> pd.DataFrame:
    """Synthetic tracks (8 per motion type, first 40 points) plus edge cases: gaps, a missing coordinate, a
    stationary track, a 2-point track, an exact back-and-forth track and an exactly straight one."""
    syn = load_synthetic()
    syn = syn[(syn.track_number % 1000 <= 8) & (syn.frame < 40)]
    edge = []
    rng = np.random.default_rng(7)
    for f in range(0, 30, 3):  # gaps of 2 frames
        edge.append((1, f, 5 + rng.normal(), 5 + rng.normal(), "edge"))
    for f in range(12):  # one missing coordinate
        edge.append((2, f, f * 0.5 + rng.normal(0, .1), np.nan if f == 4 else rng.normal(0, .1), "edge"))
    edge += [(3, f, 2.0, 3.0, "edge") for f in range(6)]  # stationary
    edge += [(4, 0, 0.0, 0.0, "edge"), (4, 1, 1.0, 1.0, "edge")]  # too short for the linear metrics
    edge += [(5, f, float(v), 0.0, "edge") for f, v in enumerate([0, 1, 2, 3, 2, 1, 0])]  # returns to start
    edge += [(6, f, 0.5 * f, 0.25 * f, "edge") for f in range(10)]  # exactly straight
    e = pd.DataFrame(edge, columns=["track_number", "frame", "x", "y", "motion_type"])
    return pd.concat([syn, e], ignore_index=True)


def build() -> dict:
    df = input_frame()
    base = dict(DEFAULTS, cutoff_len=2)
    out = {"csv": df.to_csv(index=False), "params": base, "runs": {}}
    for method in ("simple", "tensor"):
        res, _ = analyse(df, dict(base, method=method), "fixture.csv", with_coords=False)
        alt = []
        for a in ALT:
            r2, _ = analyse(df, dict(base, method=method, **a), "fixture.csv", with_coords=False)
            alt.append({"params": dict(base, method=method, **a),
                        "classes": [[t["mobility_classification"], t["linear_classification"]] for t in r2["tracks"]]})
        out["runs"][method] = {"result": {"params": res["params"], "computed_by": "python", "tracks": res["tracks"]},
                               "alt": alt}
    return json.loads(json.dumps(out, default=clean))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    OUT.write_text(json.dumps(data, indent=None, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB, {len(data['runs']['simple']['result']['tracks'])} tracks)")


if __name__ == "__main__":
    main()
