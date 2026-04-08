"""
Tests for IO utilities - CSV validation and processing.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from io_utils import validate_csv_columns, process_csv_file


class TestValidateCSVColumns:
    """Test CSV column validation."""

    def test_valid_columns(self):
        df = pd.DataFrame({
            'track_number': [1],
            'frame': [0],
            'x': [1.0],
            'y': [2.0],
        })
        assert validate_csv_columns(df) is True

    def test_missing_columns(self):
        df = pd.DataFrame({'track_number': [1], 'frame': [0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_csv_columns(df)

    def test_custom_required_columns(self):
        df = pd.DataFrame({'a': [1], 'b': [2]})
        assert validate_csv_columns(df, required_cols=['a', 'b']) is True

    def test_empty_dataframe_with_columns(self):
        df = pd.DataFrame(columns=['track_number', 'frame', 'x', 'y'])
        assert validate_csv_columns(df) is True


class TestProcessCSVFile:
    """Test CSV file processing."""

    def test_process_synthetic_data(self, tmp_path):
        """Process a synthetic CSV file and verify output."""
        rng = np.random.default_rng(42)
        records = []
        for tid in range(3):
            x, y = 0.0, 0.0
            for frame in range(20):
                records.append({
                    'track_number': tid,
                    'frame': frame,
                    'x': x,
                    'y': y,
                })
                x += rng.normal(0, 1)
                y += rng.normal(0, 1)

        df = pd.DataFrame(records)
        csv_path = tmp_path / "test_tracks.csv"
        df.to_csv(csv_path, index=False)

        result = process_csv_file(
            str(csv_path), cutoff_len=3, method='simple',
            include_shape_metrics=False, include_linear_metrics=False
        )

        assert result is not None
        assert 'rg' in result.columns
        assert 'sRg' in result.columns
        assert 'mobility_classification' in result.columns
        assert 'linear_classification' in result.columns

    def test_nonexistent_file(self):
        result = process_csv_file("/nonexistent/path.csv")
        assert result is None

    def test_short_tracks_excluded(self, tmp_path):
        """Tracks shorter than cutoff_len should not get metrics."""
        df = pd.DataFrame({
            'track_number': [0, 0, 1],
            'frame': [0, 1, 0],
            'x': [0.0, 1.0, 5.0],
            'y': [0.0, 0.0, 5.0],
        })
        csv_path = tmp_path / "short_tracks.csv"
        df.to_csv(csv_path, index=False)

        result = process_csv_file(str(csv_path), cutoff_len=3, method='simple')
        assert result is not None
        # Track 0 has 2 points (< cutoff 3), track 1 has 1 point
        # Both should have NaN metrics since neither meets cutoff
        assert 'rg' in result.columns
        assert result['rg'].isna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
