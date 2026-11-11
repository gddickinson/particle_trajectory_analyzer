"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific API tests ------------------------------------------------------------------------------------
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from io_utils import process_csv_file  # noqa: E402
from web.analysis import DEFAULTS  # noqa: E402
from web.demos import load_synthetic  # noqa: E402


def _small_csv():
    df = load_synthetic()
    return df[(df.track_number % 1000 <= 4) & (df.frame < 30)]


@pytest.mark.parametrize("method", ["simple", "tensor"])
def test_upload_matches_pipeline(tmp_path, method):
    """POST /api/analyze returns exactly what io_utils.process_csv_file computes for the same file."""
    df = _small_csv()
    path = tmp_path / "tracks.csv"
    df.to_csv(path, index=False)
    r = client.post("/api/analyze", files={"file": ("tracks.csv", path.read_bytes(), "text/csv")},
                    data={"method": method, "cutoff_len": "3", "linear_directionality": "0.7"})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["params"]["method"] == method and res["params"]["linear_directionality"] == 0.7
    with contextlib.redirect_stdout(io.StringIO()):
        ref = process_csv_file(str(path), 3, DEFAULTS["srg_cutoff"], DEFAULTS["linear_eigenvalue_ratio"],
                               DEFAULTS["linear_step_alignment"], method, True, True, 0.7, DEFAULTS["linear_perpendicular"])
    first = ref.groupby("track_number").first()
    assert len(res["tracks"]) == len(first)
    for t in res["tracks"]:
        row = first.loc[t["track_number"]]
        assert t["linear_classification"] == row["linear_classification"]
        assert t["mobility_classification"] == row["mobility_classification"]
        for k in ("rg", "sRg", "directionality_ratio"):
            assert math.isclose(t[k], row[k], rel_tol=0, abs_tol=0)
    assert set(res["coords"]) == {str(k) for k in first.index}
    csv = client.get(f"/api/runs/{res['run_id']}/metrics.csv")
    assert csv.status_code == 200
    back = pd.read_csv(io.StringIO(csv.text))
    pd.testing.assert_frame_equal(back.reset_index(drop=True), ref.reset_index(drop=True), check_dtype=False)


def test_upload_errors():
    r = client.post("/api/analyze", files={"file": ("bad.csv", b"track_number,frame,x\n1,0,1\n", "text/csv")})
    assert r.status_code == 400 and "Missing required columns" in r.text
    r = client.post("/api/analyze", data={"demo": "nope"})
    assert r.status_code == 400
    r = client.post("/api/analyze", data={"demo": "synthetic", "method": "magic"})
    assert r.status_code == 400


def test_demo_analyze_and_static_agree():
    """POST with a demo and default params gives the same classes as the exported GET (static site)."""
    get = client.get("/api/demo/synthetic/simple").json()
    post = client.post("/api/analyze", data={"demo": "synthetic"}).json()
    assert [t["linear_classification"] for t in get["tracks"]] == [t["linear_classification"] for t in post["tracks"]]
    assert get["source"]["n_tracks"] == 250 and post["coords_ref"] == "/demo/synthetic/coords"


def test_real_demo_counts_and_pixel_size():
    info = client.get("/api/info").json()
    real = next(d for d in info["demos"] if d["id"] == "real")
    assert real["n_tracks"] == 6796 and real["n_rows"] == 80645 and real["pixel_size_um"] == 0.108
    res = client.get("/api/demo/real/simple").json()
    assert len(res["tracks"]) == 4819  # tracks with >= 3 localisations
    assert res["stats"]["total_tracks"] == 6796
    assert res["stats"]["analysed_tracks"] == 4819 and res["stats"]["short_tracks"] == 1977


def test_synthetic_is_reproducible():
    a, b = load_synthetic(), load_synthetic()
    assert np.array_equal(a[["x", "y"]].to_numpy(), b[["x", "y"]].to_numpy())


def test_engine_fixture_matches_python():
    """tests/fixtures/engine_cases.json (used by engine.test.mjs) still matches the Python pipeline."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_engine_fixture
    fresh = make_engine_fixture.build()
    stored = json.loads((ROOT / "tests" / "fixtures" / "engine_cases.json").read_text())
    # Numbers, not text: libm differs in the last digit between macOS and Linux (CI)
    a = pd.read_csv(io.StringIO(stored["csv"]))
    b = pd.read_csv(io.StringIO(fresh["csv"]))
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)
    for m in ("simple", "tensor"):
        assert stored["runs"][m]["alt"] == fresh["runs"][m]["alt"]
        for s, f in zip(stored["runs"][m]["result"]["tracks"], fresh["runs"][m]["result"]["tracks"]):
            assert s.keys() == f.keys()
            for k in s:
                if isinstance(s[k], float):
                    assert math.isclose(s[k], f[k], rel_tol=1e-9, abs_tol=1e-12)
                else:
                    assert s[k] == f[k]


def test_docs_endpoints():
    assert "# Trajectory Analyzer User Manual" in client.get("/api/docs/manual").json()["markdown"]
    assert client.get("/api/docs/nope").status_code == 404
