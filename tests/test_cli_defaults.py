"""
Tests for the defaults (config.py is the single source) and the CLI entry point.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import cli  # noqa: E402
import config  # noqa: E402
from io_utils import process_csv_file  # noqa: E402


def _tracks_csv(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for t in range(4):
        x = y = 0.0
        for f in range(20):
            rows.append({'track_number': t, 'frame': f, 'x': x, 'y': y})
            x += rng.normal(); y += rng.normal()
    path = tmp_path / "tracks.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_parser_defaults_come_from_config():
    a = cli.build_parser().parse_args(['--file', 'x.csv'])
    assert a.method == config.RG_METHOD
    assert a.cutoff_len == config.CUTOFF_LEN
    assert a.srg_cutoff == config.SRG_CUTOFF
    assert a.linear_eigenvalue_ratio == config.LINEAR_EIGENVALUE_RATIO_CUTOFF
    assert a.linear_step_alignment == config.LINEAR_STEP_ALIGNMENT_CUTOFF
    assert a.linear_directionality == config.LINEAR_DIRECTIONALITY_CUTOFF
    assert a.linear_perpendicular == config.LINEAR_PERPENDICULAR_CUTOFF


def test_process_csv_file_defaults_follow_config(tmp_path, monkeypatch):
    path = _tracks_csv(tmp_path)
    df = process_csv_file(str(path))
    assert ('normalized_perpendicular_distance' in df.columns) == (config.RG_METHOD == 'simple')
    monkeypatch.setattr(config, 'RG_METHOD', 'tensor')
    assert 'eigenvalue_ratio' in process_csv_file(str(path)).columns
    monkeypatch.setattr(config, 'CUTOFF_LEN', 50)
    assert process_csv_file(str(path))['rg'].isna().all()


def test_command_line_arguments_are_used(tmp_path, monkeypatch):
    """With arguments, the CLI runs them even when USE_DIRECT_EXECUTION is set."""
    monkeypatch.setattr(config, 'USE_DIRECT_EXECUTION', True)
    path = _tracks_csv(tmp_path)
    cli.main(['--file', str(path), '--method', 'tensor', '--generate-stats'])
    out = pd.read_csv(tmp_path / 'results' / 'tracks_metrics.csv')
    assert 'eigenvalue_ratio' in out.columns
    assert (tmp_path / 'results' / 'experiment_statistics.csv').exists()


def test_direct_execution_without_path_does_not_crash(monkeypatch, capsys):
    monkeypatch.setattr(config, 'USE_DIRECT_EXECUTION', True)
    monkeypatch.setattr(config, 'DIRECTORY_PATH', None)
    monkeypatch.setattr(config, 'EXECUTION_MODE', 'directory')
    cli.main([])
    assert 'No input path given' in capsys.readouterr().out


def test_output_options_follow_readme_and_config():
    """README 5.1: INCLUDE_LINEAR_METRICS = True; the CLI's output defaults come from config.py."""
    assert config.INCLUDE_LINEAR_METRICS is True
    a = cli.build_parser().parse_args(['--file', 'x.csv'])
    assert a.include_shape_metrics == config.INCLUDE_SHAPE_METRICS
    assert a.include_linear_metrics == config.INCLUDE_LINEAR_METRICS
    assert a.mobile_only == config.MOBILE_ONLY
    b = cli.build_parser().parse_args(['--file', 'x.csv', '--no-linear-metrics', '--no-shape-metrics', '--all-tracks'])
    assert not (b.include_linear_metrics or b.include_shape_metrics or b.mobile_only)


def test_web_reads_output_options_from_config(tmp_path, monkeypatch):
    from web.analysis import DEFAULTS, analyse
    df = pd.read_csv(_tracks_csv(tmp_path))
    monkeypatch.setattr(config, 'INCLUDE_SHAPE_METRICS', False)
    res, _ = analyse(df, dict(DEFAULTS, method='tensor'), 't.csv', with_coords=False)
    assert 'asymmetry' not in res['columns']
    monkeypatch.setattr(config, 'INCLUDE_SHAPE_METRICS', True)
    res, _ = analyse(df, dict(DEFAULTS, method='tensor'), 't.csv', with_coords=False)
    assert 'asymmetry' in res['columns']


@pytest.mark.parametrize("mode", ["file", "directory"])
def test_output_dir_holds_stats_file(tmp_path, mode):
    path = _tracks_csv(tmp_path)
    out = tmp_path / 'out'
    target = str(path) if mode == 'file' else str(tmp_path)
    cli.main([f'--{mode}', target, '--output-dir', str(out), '--generate-stats'])
    assert (out / 'tracks_metrics.csv').exists()
    assert (out / 'experiment_statistics.csv').exists()
    assert not (tmp_path / 'results').exists()
